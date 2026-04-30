"""Seed three additional demo patients into OpenEMR via the standard REST API.

Like seed_cohen.py, but parameterized: each entry in PROFILES is a dict that
walks the same create-then-attach pipeline, plus a SOAP note (Cohen's seeder
predates the SOAP-note tool):

    POST /apis/default/api/patient                         → pid + uuid
    POST /apis/default/api/patient/{uuid}/allergy           ×N
    POST /apis/default/api/patient/{uuid}/medical_problem   ×N
    POST /apis/default/api/patient/{pid}/medication         ×N    (uses pid)
    POST /apis/default/api/patient/{uuid}/encounter         → eid
    POST /apis/default/api/patient/{pid}/encounter/{eid}/vital
    POST /apis/default/api/patient/{pid}/encounter/{eid}/soap_note

Idempotency: searches by fname+lname+DOB before each create. If the patient
already exists, the whole profile is skipped (re-seeding partial state would
duplicate allergies/problems/meds — drop the patient via the OpenEMR UI to
re-seed cleanly).

Why three patients (in addition to Cohen):
  - Roberts: heart-failure exacerbation. Exercises drug-drug interaction
    checks (Apixaban + new IV antibiotics).
  - Patel: elderly urosepsis with mild dementia. Exercises renal dose
    adjustment + delirium/confounder reasoning.
  - Hale: 16 yo with a distal radius fracture. Exercises structured care-plan
    surfacing (imaging + reduction + cast + meds + return precautions) on a
    young patient with minimal chronic-disease background.

Run: PYTHONPATH=. uv run python scripts/seed_demo_patients.py

Honors OPENEMR_BASE_URL (default https://localhost:9300) so it can run
locally or on the Hetzner VPS without code changes.
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

BASE = os.environ.get("OPENEMR_BASE_URL", "https://localhost:9300")
TOKEN_URL = f"{BASE}/oauth2/default/token"
TODAY = date.today().isoformat()

# Each profile gets a distinct admission time spread across the day so the
# calendar dashboard shows them as four separate inpatients with realistic
# admit timing rather than four overlapping 6 PM rows. UTC times here; the
# UI renders in the viewer's local timezone.
ADMIT_TIME = {
    "roberts": "06:00:00",   # early-morning HF decompensation
    "cohen":   "10:30:00",   # mid-morning hypertensive urgency
    "patel":   "14:15:00",   # early-afternoon urosepsis admit
    "hale":    "18:45:00",   # after-school injury / ED arrival
}

# Same scope set as seed_cohen.py + soap_note.crus (this script writes SOAP
# notes; Cohen's seeder did not, so the existing seed client may need its
# scope row extended for soap_note — see register_seed_client.py).
SCOPE = " ".join([
    "openid", "api:oemr",
    "user/patient.cruds", "user/allergy.cruds", "user/medical_problem.cruds",
    "user/medication.cruds", "user/encounter.cruds", "user/vital.cruds",
    "user/soap_note.crus", "user/practitioner.cruds",
])


PROFILES: list[dict] = [
    {
        "label": "Roberts (HF exacerbation)",
        "demo": {
            "title": "Mr",
            "fname": "Marcus", "lname": "Roberts", "DOB": "1948-03-22",
            "sex": "Male",
            "street": "412 Birchwood Dr", "city": "Austin", "state": "TX",
            "postal_code": "78704", "country_code": "US",
            "phone_contact": "512-555-0142",
        },
        "allergies": [
            {"title": "ACE inhibitors", "comments": "Persistent dry cough"},
            {"title": "Iodine contrast", "comments": "Hives, prior CT scan"},
        ],
        "problems": [
            {"title": "Heart failure with reduced ejection fraction (HFrEF, LVEF 25%)", "begdate": "2022-01-15"},
            {"title": "Atrial fibrillation", "begdate": "2021-08-10"},
            {"title": "Chronic obstructive pulmonary disease", "begdate": "2019-03-01"},
            {"title": "Chronic kidney disease, stage 2", "begdate": "2023-06-12"},
        ],
        "medications": [
            {"title": "Furosemide 40 mg PO BID", "begdate": "2025-12-01 00:00:00"},
            {"title": "Carvedilol 25 mg PO BID", "begdate": "2022-02-01 00:00:00"},
            {"title": "Spironolactone 25 mg PO daily", "begdate": "2024-04-15 00:00:00"},
            {"title": "Apixaban 5 mg PO BID", "begdate": "2021-08-15 00:00:00"},
        ],
        "encounter": {
            "pc_catid": 5, "facility_id": 3, "billing_facility": 3,
            "sensitivity": "normal",
            "reason": "Acute decompensated heart failure, dyspnea, 8 lb weight gain in 5 days",
            "date": f"{TODAY} {ADMIT_TIME['roberts']}", "onset_date": f"{TODAY} {ADMIT_TIME['roberts']}",
            "provider_id": 1, "class_code": "AMB",
        },
        "vitals": {
            "bps": "142", "bpd": "88", "pulse": "110", "respiration": "22",
            "temperature": "98.2", "temp_method": "Oral",
            "oxygen_saturation": "91", "weight": "198", "height": "70",
        },
        "soap": {
            "subjective": (
                "78 yo M w/ HFrEF (LVEF 25%), AFib on apixaban, COPD, CKD2 presents "
                "w/ progressive DOE x5d, 8 lb weight gain, orthopnea (3 pillows now, "
                "baseline 1), bilateral LE swelling. No CP. No fever. Adherent to "
                "all home meds. Endorses dietary indiscretion (Easter ham 3 days ago)."
            ),
            "objective": (
                "VS: BP 142/88, HR 110 irregularly irregular, RR 22, SpO2 91% RA, "
                "T 98.2. JVD to angle of jaw. Bilateral coarse crackles to mid-lung "
                "fields. S3 gallop, irregular rhythm. Bilateral LE pitting edema 2+. "
                "Cool extremities."
            ),
            "assessment": (
                "1. Acute on chronic decompensated HFrEF, NYHA III at baseline now IV. "
                "Likely volume overload from dietary indiscretion; AFib w/ RVR likely "
                "contributing. "
                "2. AFib with rapid ventricular response. "
                "3. CKD stage 2, baseline."
            ),
            "plan": (
                "1. IV furosemide 80 mg now, then 40 mg IV BID. Daily weights, strict I/O, target 2-3L net negative/day.\n"
                "2. Continue carvedilol, spironolactone, apixaban.\n"
                "3. Telemetry monitoring; cardiology consult.\n"
                "4. Labs: BMP, BNP, troponin x3, CXR, repeat BMP in AM.\n"
                "5. 2 g Na, 1.5 L fluid restriction.\n"
                "6. Repeat TTE if no clinical improvement at 48 h.\n"
                "7. Goals-of-care discussion at next visit if stable."
            ),
        },
    },
    {
        "label": "Patel (urosepsis, elderly)",
        "demo": {
            "title": "Mrs",
            "fname": "Anjali", "lname": "Patel", "DOB": "1942-09-08",
            "sex": "Female",
            "street": "Sunset Hills ALF, 89 Maple St",
            "city": "Austin", "state": "TX",
            "postal_code": "78745", "country_code": "US",
            "phone_contact": "512-555-0167",
        },
        "allergies": [
            {"title": "Vancomycin", "comments": "Red man syndrome"},
            {"title": "Codeine", "comments": "Severe nausea + pruritus"},
        ],
        "problems": [
            {"title": "Acute pyelonephritis with sepsis", "begdate": "2026-04-28"},
            {"title": "Recurrent urinary tract infections", "begdate": "2023-09-01"},
            {"title": "Dementia, mild", "begdate": "2024-01-10"},
            {"title": "Hypothyroidism", "begdate": "2018-05-22"},
            {"title": "Chronic kidney disease, stage 3", "begdate": "2022-11-04"},
        ],
        "medications": [
            {"title": "Ceftriaxone 1 g IV daily", "begdate": "2026-04-28 00:00:00"},
            {"title": "Donepezil 10 mg PO QHS", "begdate": "2024-01-15 00:00:00"},
            {"title": "Levothyroxine 75 mcg PO daily", "begdate": "2018-06-01 00:00:00"},
            {"title": "Tamsulosin 0.4 mg PO QHS", "begdate": "2023-09-15 00:00:00"},
        ],
        "encounter": {
            "pc_catid": 5, "facility_id": 3, "billing_facility": 3,
            "sensitivity": "normal",
            "reason": "Sepsis 2/2 acute pyelonephritis, AMS, fever 102.4 F",
            "date": f"{TODAY} {ADMIT_TIME['patel']}", "onset_date": f"{TODAY} {ADMIT_TIME['patel']}",
            "provider_id": 1, "class_code": "AMB",
        },
        "vitals": {
            "bps": "88", "bpd": "52", "pulse": "118", "respiration": "24",
            "temperature": "102.4", "temp_method": "Oral",
            "oxygen_saturation": "95", "weight": "132", "height": "62",
        },
        "soap": {
            "subjective": (
                "83 yo F w/ recurrent UTIs, mild dementia, hypothyroidism, CKD3 "
                "presents from assisted living facility w/ 1 day of acute confusion, "
                "fever 102.4 F, dysuria, suprapubic pain. Last UTI 6 weeks ago, "
                "treated with Bactrim. Per family, baseline alert and conversational; "
                "today disoriented to place and time."
            ),
            "objective": (
                "VS: BP 88/52, HR 118 sinus tachycardia, T 102.4 F, RR 24, SpO2 95% RA. "
                "Ill-appearing, dry mucous membranes. R CVA tenderness. Suprapubic "
                "tenderness. Mental status: oriented to person only, follows simple "
                "commands."
            ),
            "assessment": (
                "1. Septic shock 2/2 acute pyelonephritis. SIRS criteria met (HR, RR, T). "
                "2. Acute encephalopathy, likely septic, on background of mild dementia. "
                "3. Acute kidney injury on CKD3 (anticipated, will check)."
            ),
            "plan": (
                "1. IV NS 30 mL/kg bolus over 3 h, then reassess MAP and lactate.\n"
                "2. Ceftriaxone 1 g IV daily already given today; continue.\n"
                "3. Cultures: blood x2, urine, lactate. Labs: BMP, CBC, UA, LFTs.\n"
                "4. Tylenol 650 mg PO q6h PRN fever.\n"
                "5. Hold tamsulosin while hypotensive.\n"
                "6. Telemetry; ICU evaluation if MAP < 65 after fluid challenge.\n"
                "7. Renal-dose adjust all meds for current CrCl.\n"
                "8. Goals-of-care discussion with family in AM given dementia history."
            ),
        },
    },
    {
        "label": "Hale (broken arm, teen)",
        "demo": {
            "title": "Mr",
            "fname": "Liam", "lname": "Hale", "DOB": "2009-11-14",
            "sex": "Male",
            "street": "118 Cedarcrest Ln",
            "city": "Austin", "state": "TX",
            "postal_code": "78731", "country_code": "US",
            "phone_contact": "512-555-0193",
        },
        "allergies": [],
        "problems": [
            {"title": "Closed displaced distal radius fracture, right", "begdate": TODAY},
            {"title": "Asthma, mild intermittent", "begdate": "2018-04-01"},
        ],
        "medications": [
            {"title": "Acetaminophen 650 mg PO q6h PRN", "begdate": f"{TODAY} 00:00:00"},
            {"title": "Ibuprofen 400 mg PO q8h x5 days, then PRN", "begdate": f"{TODAY} 00:00:00"},
            {"title": "Albuterol HFA 90 mcg, 2 puffs q4-6h PRN", "begdate": "2018-04-01 00:00:00"},
        ],
        "encounter": {
            "pc_catid": 5, "facility_id": 3, "billing_facility": 3,
            "sensitivity": "normal",
            "reason": "ED visit: fall from skateboard onto outstretched right hand, R wrist injury",
            "date": f"{TODAY} {ADMIT_TIME['hale']}", "onset_date": f"{TODAY} {ADMIT_TIME['hale']}",
            "provider_id": 1, "class_code": "AMB",
        },
        "vitals": {
            "bps": "118", "bpd": "72", "pulse": "86", "respiration": "16",
            "temperature": "98.4", "temp_method": "Oral",
            "oxygen_saturation": "99", "weight": "142", "height": "70",
        },
        "soap": {
            "subjective": (
                "16 yo M, no significant PMH other than mild intermittent asthma "
                "(rare albuterol use), presents to ED s/p mechanical fall from "
                "skateboard onto outstretched R hand approximately 2 hours ago. "
                "Immediate pain in R wrist, swelling, unable to use hand. Denies LOC, "
                "head strike, neck pain, or other injuries. NPO since lunch."
            ),
            "objective": (
                "VS: BP 118/72, HR 86, RR 16, SpO2 99% RA, T 98.4. Alert, comfortable "
                "in NAD. R wrist with obvious dorsal angulation, dorsal swelling, "
                "ecchymosis. Limited ROM 2/2 pain. Neurovascular: 2+ radial pulse, "
                "capillary refill < 2 s, sensation intact in median/ulnar/radial "
                "distributions, normal motor function of all fingers."
            ),
            "assessment": (
                "1. Closed displaced distal radius fracture, R, with ~20 deg dorsal "
                "angulation. No intra-articular extension on imaging. Neurovascularly "
                "intact pre- and post-reduction. "
                "2. Asthma, mild intermittent — well controlled, not active."
            ),
            "plan": (
                "1. Wrist XR 3-view (PA, lateral, oblique) — done; confirms displaced distal radius fx, ~20 deg dorsal angulation, no intra-articular extension.\n"
                "2. Closed reduction performed under hematoma block (5 cc 1% lidocaine). Post-reduction films show acceptable alignment.\n"
                "3. Short-arm fiberglass cast applied in neutral position.\n"
                "4. Acetaminophen 650 mg PO q6h scheduled + Ibuprofen 400 mg PO q8h scheduled x5 days, then PRN.\n"
                "5. Sling x48 h for comfort.\n"
                "6. Ortho follow-up at 7-10 days for repeat XR in cast.\n"
                "7. Return precautions reviewed: pain unrelieved by oral meds, finger numbness/tingling, finger pallor or coolness, cast becoming loose/cracked or wet.\n"
                "8. Activity restrictions: no sports x6 weeks min; computer/writing OK after acute pain resolves."
            ),
        },
    },
]


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


async def post_json(
    client: httpx.AsyncClient,
    token: str,
    path: str,
    body: dict,
    label: str,
) -> dict:
    """POST JSON, return the flat data dict (id/uuid/eid/euuid/sid as available)."""
    r = await client.post(
        f"{BASE}{path}",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json=body,
    )
    if r.status_code >= 400:
        print(f"  ✗ {label} HTTP {r.status_code}: {r.text[:300]}", file=sys.stderr)
        r.raise_for_status()

    payload = r.json()
    val_errs = payload.get("validationErrors") if isinstance(payload, dict) else None
    if isinstance(val_errs, dict) and val_errs:
        print(f"  ✗ {label} validation: {val_errs}", file=sys.stderr)
        sys.exit(1)

    data = payload.get("data") if isinstance(payload, dict) else None
    if isinstance(data, dict) and data:
        flat = data
    elif isinstance(payload, dict) and (
        {"id", "uuid", "eid", "euuid", "vid", "fid", "sid", "pid"} & payload.keys()
    ):
        flat = payload
    else:
        print(f"  ✗ {label} unrecognized response: {payload}", file=sys.stderr)
        sys.exit(1)

    label_id = flat.get("id") or flat.get("eid") or flat.get("sid") or flat.get("pid") or "(no id)"
    label_uuid = flat.get("uuid") or flat.get("euuid") or ""
    print(f"  ✓ {label} (id={label_id}{', uuid=' + label_uuid if label_uuid else ''})")
    return flat


async def find_patient(
    client: httpx.AsyncClient, token: str, fname: str, lname: str, dob: str,
) -> dict | None:
    """Idempotency check: return {pid, uuid} if a matching patient already exists."""
    r = await client.get(
        f"{BASE}/apis/default/api/patient",
        headers={"Authorization": f"Bearer {token}"},
        params={"fname": fname, "lname": lname, "DOB": dob},
    )
    if r.status_code != 200:
        return None
    payload = r.json()
    rows = payload.get("data") if isinstance(payload, dict) else None
    if isinstance(rows, list) and rows:
        first = rows[0]
        return {
            "pid": first.get("pid") or first.get("id"),
            "uuid": first.get("uuid"),
        }
    return None


async def seed_profile(
    client: httpx.AsyncClient, token: str, profile: dict,
) -> dict | None:
    print(f"\n=== {profile['label']} ===")
    demo = profile["demo"]

    existing = await find_patient(client, token, demo["fname"], demo["lname"], demo["DOB"])
    if existing and existing.get("pid") and existing.get("uuid"):
        print(
            f"  ↺ already exists (pid={existing['pid']}, uuid={existing['uuid']}); "
            "skipping entire profile (delete in OpenEMR UI to re-seed)."
        )
        return {"label": profile["label"], "pid": existing["pid"], "uuid": existing["uuid"], "status": "exists"}

    flat = await post_json(
        client, token, "/apis/default/api/patient", demo,
        f"{demo['fname']} {demo['lname']}",
    )
    pid = flat.get("pid") or flat.get("id")
    puuid = flat.get("uuid")
    if not puuid:
        # Fall back to a search to recover the uuid the server just minted.
        recovered = await find_patient(client, token, demo["fname"], demo["lname"], demo["DOB"])
        if recovered and recovered.get("uuid"):
            puuid = recovered["uuid"]
            pid = pid or recovered.get("pid")
    if not pid or not puuid:
        print(f"  ✗ unable to resolve pid/uuid for new patient: {flat}", file=sys.stderr)
        sys.exit(1)
    pid, puuid = str(pid), str(puuid)
    print(f"  pid={pid}, uuid={puuid}")

    if profile["allergies"]:
        print("Allergies:")
        for entry in profile["allergies"]:
            await post_json(client, token, f"/apis/default/api/patient/{puuid}/allergy", entry, entry["title"])

    print("\nMedical problems:")
    for entry in profile["problems"]:
        await post_json(client, token, f"/apis/default/api/patient/{puuid}/medical_problem", entry, entry["title"])

    print("\nMedications:")
    for entry in profile["medications"]:
        await post_json(client, token, f"/apis/default/api/patient/{pid}/medication", entry, entry["title"])

    print("\nEncounter:")
    enc = await post_json(client, token, f"/apis/default/api/patient/{puuid}/encounter", profile["encounter"], "today")
    eid = enc.get("eid")
    if not eid:
        print("  ✗ encounter response missing eid", file=sys.stderr)
        sys.exit(1)

    print("\nVitals:")
    await post_json(
        client, token,
        f"/apis/default/api/patient/{pid}/encounter/{eid}/vital",
        profile["vitals"], "BP/HR/T/SpO2",
    )

    print("\nSOAP note:")
    await post_json(
        client, token,
        f"/apis/default/api/patient/{pid}/encounter/{eid}/soap_note",
        profile["soap"], "subjective/objective/assessment/plan",
    )

    print(f"✓ {profile['label']} seeded (pid={pid}, uuid={puuid}, eid={eid})")
    return {"label": profile["label"], "pid": pid, "uuid": puuid, "status": "created"}


async def main() -> None:
    async with httpx.AsyncClient(verify=False, timeout=30) as client:
        print("Acquiring user-context token...")
        token = await get_token(client)
        print("  ✓ got token")

        results = []
        for profile in PROFILES:
            results.append(await seed_profile(client, token, profile))

        print("\n=== Summary ===")
        for r in results:
            if r:
                print(f"  {r['status']:8} {r['label']:35} pid={r['pid']:>4}  uuid={r['uuid']}")


if __name__ == "__main__":
    asyncio.run(main())
