"""Populate Nora Cohen with realistic inpatient data via OpenEMR's standard REST API.

Why this script exists:
  - OpenEMR's FHIR API only allows writes for Patient/Practitioner/Organization;
    everything else must go through the standard REST API at /apis/default/api/.
  - The standard API requires a user-context OAuth token (`password` grant)
    with `user/<resource>.cruds` scopes — distinct from the FHIR `system/*`
    flow used by the running agent.
  - We registered a separate `agent_forge_seed` OAuth client just for this.

What it creates for Cohen:
  - 2 allergies   (Penicillin / Sulfa)
  - 4 problems    (HTN, T2DM, CKD3, AFib)
  - 4 medications (Lisinopril, Metformin, Apixaban, Atorvastatin)
  - 1 encounter   (today, hypertensive urgency)
  - 1 vital set   (BP 158/94 etc.)

Idempotency: the script is run-once. To re-seed, delete via the OpenEMR UI or
truncate Cohen's rows from the `lists`/`form_encounter`/`form_vitals` tables.

Run: PYTHONPATH=. uv run python scripts/seed_cohen.py
"""

import asyncio
import os
import sys
from datetime import date

import httpx
import urllib3
from dotenv import load_dotenv

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

load_dotenv()

BASE = "https://localhost:9300"
TOKEN_URL = f"{BASE}/oauth2/default/token"
COHEN_PUUID = "a1a6044b-c6af-40a4-80aa-4c5ce61014da"
COHEN_PID = 8  # from the OpenEMR UI: "Nora Cohen (8)"
TODAY = date.today().isoformat()

SCOPE = " ".join([
    "openid", "api:oemr",
    "user/patient.cruds", "user/allergy.cruds", "user/medical_problem.cruds",
    "user/medication.cruds", "user/encounter.cruds", "user/vital.cruds",
    "user/practitioner.cruds",
])


def _dt(d: str) -> str:
    """OpenEMR's lists/* date validators accept ISO Y-m-d (no time component)."""
    return d


ALLERGIES = [
    {"title": "Penicillin", "comments": "Hives, moderate"},
    {"title": "Sulfa drugs", "comments": "Rash, mild"},
]

PROBLEMS = [
    {"title": "Hypertension", "begdate": _dt("2024-10-01")},
    {"title": "Type 2 diabetes mellitus", "begdate": _dt("2023-05-12")},
    {"title": "Chronic kidney disease, stage 3", "begdate": _dt("2025-02-08")},
    {"title": "Atrial fibrillation", "begdate": _dt("2024-11-20")},
]

# Medications go through ListService whose validator demands Y-m-d H:i:s (the
# allergy + medical_problem controllers accept Y-m-d). Hard-code the seconds
# so the seed survives both validators without a second helper.
MEDICATIONS = [
    {"title": "Lisinopril 20 mg PO daily", "begdate": "2024-10-01 00:00:00"},
    {"title": "Metformin 1000 mg PO BID", "begdate": "2023-05-12 00:00:00"},
    {"title": "Apixaban 5 mg PO BID", "begdate": "2024-11-20 00:00:00"},
    {"title": "Atorvastatin 40 mg PO QHS", "begdate": "2023-05-12 00:00:00"},
]

ENCOUNTER = {
    "pc_catid": 5,
    "facility_id": 3,
    "billing_facility": 3,
    "sensitivity": "normal",
    "reason": "Hypertensive urgency, headache, mild SOB",
    "date": TODAY,
    "onset_date": TODAY,
    "provider_id": 1,
    "class_code": "AMB",
}

VITALS = {
    "bps": "158",
    "bpd": "94",
    "pulse": "78",
    "respiration": "16",
    "temperature": "98.6",
    "temp_method": "Oral",
    "oxygen_saturation": "97",
    "weight": "175",
    "height": "65",
}


async def get_token(client: httpx.AsyncClient) -> str:
    cid = os.environ["OPENEMR_SEED_CLIENT_ID"]
    csec = os.environ["OPENEMR_SEED_CLIENT_SECRET"]
    body = {
        "grant_type": "password",
        "client_id": cid,
        "client_secret": csec,
        "username": "admin",
        "password": "pass",
        "user_role": "users",
        "scope": SCOPE,
    }
    r = await client.post(TOKEN_URL, data=body)
    r.raise_for_status()
    return r.json()["access_token"]


async def post(
    client: httpx.AsyncClient,
    token: str,
    path: str,
    body: dict,
    label: str,
) -> dict:
    r = await client.post(
        f"{BASE}{path}",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json=body,
    )
    if r.status_code >= 400:
        print(f"  ✗ {label} HTTP {r.status_code}: {r.text[:300]}", file=sys.stderr)
        r.raise_for_status()

    payload = r.json()
    # Three response shapes seen in the standard API:
    #   - {"validationErrors":[], "internalErrors":[], "data":{"id":..,"uuid":..}, "links":[]}
    #   - {"validationErrors":[], "internalErrors":[], "data":{"eid":..,"euuid":..}, "links":[]}
    #   - {"id": ..}                                    (flat, e.g. medication)
    # Validation failure (HTTP 200 + non-empty validationErrors + data:[]):
    #   - {"validationErrors":{"begdate":{...}}, "data":[], ...}
    val_errs = payload.get("validationErrors") if isinstance(payload, dict) else None
    if isinstance(val_errs, dict) and val_errs:
        print(f"  ✗ {label} validation: {val_errs}", file=sys.stderr)
        sys.exit(1)

    data = payload.get("data") if isinstance(payload, dict) else None
    if isinstance(data, dict) and data:
        flat = data
    elif isinstance(payload, dict) and ({"id", "uuid", "eid", "euuid", "vid", "fid"} & payload.keys()):
        flat = payload
    else:
        print(f"  ✗ {label} unrecognized response: {payload}", file=sys.stderr)
        sys.exit(1)

    label_id = flat.get("id") or flat.get("eid") or "(no id)"
    label_uuid = flat.get("uuid") or flat.get("euuid") or ""
    print(f"  ✓ {label} (id={label_id}{', uuid=' + label_uuid if label_uuid else ''})")
    return flat


async def main() -> None:
    async with httpx.AsyncClient(verify=False, timeout=30) as client:
        print("Acquiring user-context token...")
        token = await get_token(client)
        print("  ✓ got token\n")

        print("Allergies:")
        for entry in ALLERGIES:
            await post(client, token, f"/apis/default/api/patient/{COHEN_PUUID}/allergy", entry, entry["title"])

        print("\nMedical problems:")
        for entry in PROBLEMS:
            await post(client, token, f"/apis/default/api/patient/{COHEN_PUUID}/medical_problem", entry, entry["title"])

        print("\nMedications:")
        for entry in MEDICATIONS:
            await post(client, token, f"/apis/default/api/patient/{COHEN_PID}/medication", entry, entry["title"])

        print("\nEncounter:")
        enc = await post(client, token, f"/apis/default/api/patient/{COHEN_PUUID}/encounter", ENCOUNTER, "today")
        eid_numeric = enc.get("eid")
        if not eid_numeric:
            print("  ✗ encounter response missing 'eid' for vitals POST", file=sys.stderr)
            sys.exit(1)

        print("\nVitals:")
        await post(
            client, token,
            f"/apis/default/api/patient/{COHEN_PID}/encounter/{eid_numeric}/vital",
            VITALS, "BP/HR/T/SpO2",
        )

        print("\n✓ Cohen seeded.")


if __name__ == "__main__":
    asyncio.run(main())
