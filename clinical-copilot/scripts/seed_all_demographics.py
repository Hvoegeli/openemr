"""Seed all 20 demo patient demographics into OpenEMR for the Copilot demo.

Purpose: when `copilot--branch-2` lands on master and master is deployed
to a fresh Hetzner OpenEMR (which ships with an empty `patient_data`
table), this script repopulates the patient picker so the Copilot UI is
immediately usable end-to-end without any manual data entry.

Two cohorts:

1. **OpenEMR built-ins (14)** — Ted Shaw, Eduardo Perez, etc. Originally
   shipped pre-loaded by the `development-easy` Docker image. Captured
   here as Python literals because Hetzner master uses a different
   provisioning path that does not auto-seed them.

2. **Copilot demo patients (6)** — Roberts, Patel, Hale (acute clinical
   demos) plus Chen, Reyes, Kowalski (extraction/document-upload demos).

Idempotent by `lname + DOB`. Re-runs skip every patient that already
exists and create only the missing ones. Per-patient failures do NOT
abort the whole run — the next patient is attempted, and the final
summary lists what failed.

# Relationship to other seed scripts

  - `seed_demo_patients.py` — adds full clinical context (allergies,
    problems, medications, encounters, vitals, SOAP notes) for Roberts,
    Patel, and Hale. **Run it BEFORE this script if you want their rich
    state**, because it skips the entire profile if the patient already
    exists. If this script ran first and created bare demographics for
    them, `seed_demo_patients.py` will idempotent-skip and the rich
    state will be missing.

  - `seed_chen.py`, `seed_reyes.py`, `seed_kowalski.py` — sister scripts
    that print the new PUUID for `.env`. Run any of them if you want the
    convenience of the printed env line; otherwise this script covers
    them too (just without printing the PUUID).

# Recommended workflow on a fresh OpenEMR install

    PYTHONPATH=. uv run python scripts/seed_demo_patients.py    # rich state for 3
    PYTHONPATH=. uv run python scripts/seed_all_demographics.py # remaining 17

# Targeting environments

Reads `OPENEMR_FHIR_BASE_URL` from `.env`, so the same script targets
local docker stack OR Hetzner depending on which env is loaded. To seed
Hetzner master after the branch is merged + deployed:

    ssh hetzner-master \\
      'cd /root/openemr-copilot && PYTHONPATH=. uv run python scripts/seed_all_demographics.py'

Run locally:
    cd clinical-copilot && PYTHONPATH=. uv run python scripts/seed_all_demographics.py
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any

import httpx

from app.config import settings


# Each entry encodes ONLY the populated fields. OpenEMR's /api/patient
# accepts a partial body; missing fields land as NULL/empty. The 14
# built-ins are sparse (no race/ethnicity, sometimes no sex) because
# that's how dev-easy ships them. The 6 Copilot demos are richer
# because they were created via the API during the project.
PATIENTS: list[dict[str, Any]] = [
    # --- OpenEMR built-in demo dataset (14) ---
    {
        "fname": "Ted",
        "lname": "Shaw",
        "DOB": "1947-03-11",
        "sex": "Male",
        "language": "english",
        "street": "222 1st Avenue",
        "city": "San Diego",
        "state": "CA",
        "postal_code": "92101",
        "email": "info@pennfirm.com",
    },
    {
        "fname": "Eduardo",
        "lname": "Perez",
        "DOB": "1957-01-09",
        "sex": "Male",
        "language": "english",
        "street": "789 Third Avenue",
        "city": "San Diego",
        "state": "CA",
        "email": "info@pennfirm.com",
    },
    {
        "fname": "Farrah",
        "mname": "A.",
        "lname": "Rolle",
        "DOB": "1973-10-11",
        "sex": "Female",
        "language": "english",
        "street": "111 Main Street",
        "city": "San Luis",
        "state": "CA",
        "postal_code": "92101",
        "email": "frolle@pennfirm.com",
    },
    {
        "fname": "Nora",
        "lname": "Cohen",
        "DOB": "1967-06-04",
        "sex": "Female",
        "language": "spanish",
        "street": "155 First Avenue",
        "city": "San Luis",
        "state": "CA",
        "postal_code": "92101",
    },
    {
        "fname": "Jim",
        "lname": "Moses",
        "DOB": "1945-02-14",
        "sex": "Male",
        "city": "Los Angeles",
        "state": "CA",
    },
    {
        "fname": "Richard",
        "lname": "Jones",
        "DOB": "1940-12-16",
        "sex": "Male",
        "street": "400 West Broadway",
        "city": "San Diego",
        "state": "CA",
        "postal_code": "92101",
        "email": "richard@pennfirm.com",
    },
    {
        "fname": "Ilias",
        "lname": "Jenane",
        "DOB": "1933-03-22",
        "sex": "Female",
        "language": "english",
        "street": "145 N. East Street",
        "city": "La Mesa",
        "state": "CA",
        "postal_code": "92111",
    },
    {
        "fname": "John",
        "mname": "D",
        "lname": "Dockerty",
        "DOB": "1977-05-02",
        "sex": "Male",
        "language": "english",
        "street": "800 West Way",
        "city": "San Diego",
        "state": "CA",
        "postal_code": "92101",
    },
    {
        "fname": "James",
        "lname": "Janssen",
        "DOB": "1966-04-28",
        "sex": "Male",
        "language": "english",
        "street": "111 North Street",
        "city": "Irvine",
        "state": "CA",
        "postal_code": "90205",
    },
    {
        "fname": "Jason",
        "lname": "Binder",
        "DOB": "1961-12-11",
        "sex": "Male",
        "language": "english",
        "street": "100 West Sepulveda",
        "city": "Los Angeles",
        "state": "CA",
        "postal_code": "92020",
    },
    {
        "fname": "Robert",
        "lname": "Dickey",
        "DOB": "1955-04-12",
        "sex": "Male",
        "language": "english",
        "street": "111 North Kearny",
        "city": "Torrance",
        "state": "CA",
        "postal_code": "91040",
    },
    {
        "fname": "Jillian",
        "lname": "Mahoney",
        "DOB": "1968-08-11",
        "sex": "Female",
        "language": "english",
        "street": "444 North State Street",
        "city": "Santa Ana",
        "state": "CA",
        "postal_code": "90204",
    },
    {
        "fname": "Wallace",
        "lname": "Buckley",
        "DOB": "1952-04-03",
        "sex": "Male",
        "language": "english",
        "street": "123 West Street",
        "city": "Barstow",
        # Preserved exactly as it appears in the dev-easy demo dataset —
        # spelled-out form rather than the USPS abbreviation.
        "state": "California",
        "postal_code": "90400",
    },
    {
        "fname": "Brent",
        "lname": "Perez",
        "DOB": "1960-01-01",
        "sex": "Male",
        "language": "english",
        "street": "1234 1st Avenue",
        "city": "San Diego",
        "state": "CA",
        "postal_code": "92101",
        "email": "bperez@pennfirm.com",
    },
    # --- Copilot demo cohort (6) ---
    {
        "fname": "Marcus",
        "lname": "Roberts",
        "DOB": "1948-03-22",
        "sex": "Male",
        "street": "412 Birchwood Dr",
        "city": "Austin",
        "state": "TX",
        "postal_code": "78704",
        "country_code": "US",
    },
    {
        "fname": "Anjali",
        "lname": "Patel",
        "DOB": "1942-09-08",
        "sex": "Female",
        "street": "Sunset Hills ALF, 89 Maple St",
        "city": "Austin",
        "state": "TX",
        "postal_code": "78745",
        "country_code": "US",
    },
    {
        "fname": "Liam",
        "lname": "Hale",
        "DOB": "2009-11-14",
        "sex": "Male",
        "street": "118 Cedarcrest Ln",
        "city": "Austin",
        "state": "TX",
        "postal_code": "78731",
        "country_code": "US",
    },
    {
        "fname": "Margaret",
        "mname": "L.",
        "lname": "Chen",
        "DOB": "1967-08-14",
        "sex": "Female",
        "race": "asian",
        "ethnicity": "not_hisp_or_lat",
        "language": "English",
        "street": "4421 Magnolia Ave, Apt 3B",
        "city": "Berkeley",
        "state": "CA",
        "postal_code": "94705",
        "country_code": "US",
        "phone_cell": "(510) 555-0148",
        "email": "mchen.demo@example.test",
    },
    {
        "fname": "Sofia",
        "mname": "M.",
        "lname": "Reyes",
        "DOB": "1969-11-23",
        "sex": "Female",
        "race": "decline_to_specify",
        "ethnicity": "hisp_or_lat",
        "language": "English",
        "street": "1100 South Lamar Blvd",
        "city": "Austin",
        "state": "TX",
        "postal_code": "78704",
        "country_code": "US",
        "phone_cell": "(512) 555-0177",
        "email": "sreyes.demo@example.test",
    },
    {
        "fname": "Robert",
        "mname": "J.",
        "lname": "Kowalski",
        "DOB": "1958-06-18",
        "sex": "Male",
        "race": "white",
        "ethnicity": "not_hisp_or_lat",
        "language": "English",
        "street": "2200 N Clark Street",
        "city": "Chicago",
        "state": "IL",
        "postal_code": "60614",
        "country_code": "US",
        "phone_cell": "(312) 555-0192",
        "email": "rkowalski.demo@example.test",
    },
]


def _api_base() -> str:
    return settings.openemr_fhir_base_url.rstrip("/").rsplit("/", 1)[0] + "/api"


async def _get_token(client: httpx.AsyncClient) -> str:
    if not (settings.openemr_seed_client_id and settings.openemr_seed_client_secret):
        print(
            "OPENEMR_SEED_CLIENT_ID/SECRET not set in .env — run "
            "scripts/register_seed_client.py first.",
            file=sys.stderr,
        )
        sys.exit(1)
    r = await client.post(
        settings.openemr_oauth_token_url,
        data={
            "grant_type": "password",
            "client_id": settings.openemr_seed_client_id,
            "client_secret": settings.openemr_seed_client_secret,
            "username": "admin",
            "password": "pass",
            "user_role": "users",
            "scope": "openid api:oemr api:fhir user/patient.cruds",
        },
    )
    if r.status_code != 200:
        print(
            f"Seed-client password-grant failed ({r.status_code}): "
            f"{r.text[:300]}",
            file=sys.stderr,
        )
        sys.exit(1)
    return r.json()["access_token"]


async def _find_existing(
    client: httpx.AsyncClient, token: str, demographics: dict[str, Any],
) -> str | None:
    r = await client.get(
        f"{_api_base()}/patient",
        params={"lname": demographics["lname"], "DOB": demographics["DOB"]},
        headers={"Authorization": f"Bearer {token}"},
    )
    if r.status_code != 200:
        return None
    body = r.json()
    items = body.get("data") if isinstance(body, dict) else None
    if not isinstance(items, list) or not items:
        return None
    for item in items:
        if not isinstance(item, dict):
            continue
        if (
            item.get("fname") == demographics["fname"]
            and item.get("lname") == demographics["lname"]
            and item.get("DOB") == demographics["DOB"]
        ):
            return item.get("uuid") or item.get("puuid")
    return None


async def _create(
    client: httpx.AsyncClient, token: str, demographics: dict[str, Any],
) -> str:
    r = await client.post(
        f"{_api_base()}/patient",
        json=demographics,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    if r.status_code >= 400:
        raise RuntimeError(
            f"Patient create failed ({r.status_code}): {r.text[:400]}"
        )
    body = r.json()
    val = body.get("validationErrors") if isinstance(body, dict) else None
    if isinstance(val, dict) and val:
        raise RuntimeError(f"Patient validation failed: {val}")
    data = body.get("data") if isinstance(body, dict) else body
    flat = data if isinstance(data, dict) else {}
    puuid = flat.get("uuid") or flat.get("puuid") or flat.get("pubpid")
    if not puuid:
        raise RuntimeError(f"Patient create returned no uuid: {body}")
    return str(puuid)


async def _run() -> int:
    async with httpx.AsyncClient(verify=False, timeout=30) as client:
        print(f"Targeting OpenEMR at: {settings.openemr_fhir_base_url}")
        print("Acquiring seed-client token...")
        token = await _get_token(client)
        print("  ✓ token acquired\n")

        created: list[str] = []
        skipped: list[str] = []
        failed: list[tuple[str, str]] = []

        for demo in PATIENTS:
            label = f"{demo['fname']} {demo['lname']} (DOB {demo['DOB']})"
            try:
                existing = await _find_existing(client, token, demo)
                if existing:
                    print(f"  · {label}: already present (PUUID {existing})")
                    skipped.append(label)
                    continue
                puuid = await _create(client, token, demo)
                print(f"  ✓ {label}: created (PUUID {puuid})")
                created.append(label)
            except Exception as e:  # noqa: BLE001
                # Continue past per-patient failures so one bad row
                # doesn't strand the rest of the seed run.
                print(f"  ✗ {label}: {e}", file=sys.stderr)
                failed.append((label, str(e)))

        print(
            f"\nSummary: created={len(created)}  "
            f"skipped={len(skipped)}  failed={len(failed)}  "
            f"total={len(PATIENTS)}"
        )
        if failed:
            print("\nFailed patients:")
            for label, msg in failed:
                print(f"  - {label}: {msg}")
            return 1
        return 0


def main() -> None:
    sys.exit(asyncio.run(_run()))


if __name__ == "__main__":
    main()
