"""Seed Margaret L. Chen as a Patient resource in OpenEMR for Week 2 smoke testing.

Lightweight by design — only writes demographics so the FHIR DocumentReference
write in `scripts/smoke_document_writer.py` has a valid `subject.reference`
to point at. Allergies / problems / medications / family history come later
from the Phase 2 intake-form extractor; pre-seeding them here would defeat
the purpose of the extraction demo.

Identity comes from `data/demo_documents/real/p01-chen-intake-typed.pdf`
(verbatim — the same fields the VLM will extract from that PDF in Phase 2).

After running, paste the printed `DEMO_PATIENT_PUUID=...` line into
`clinical-copilot/.env` so the smoke test can find her.

Run: `cd clinical-copilot && PYTHONPATH=. uv run python scripts/seed_chen.py`
"""

from __future__ import annotations

import asyncio
import sys

import httpx

from app.config import settings


# Demographics straight from p01-chen-intake-typed.pdf
CHEN_DEMOGRAPHICS = {
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
}


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


async def _find_existing_chen(client: httpx.AsyncClient, token: str) -> str | None:
    """Return PUUID if a patient with same family_name + DOB already exists."""
    r = await client.get(
        f"{_api_base()}/patient",
        params={"lname": "Chen", "DOB": "1967-08-14"},
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
            item.get("fname") == "Margaret"
            and item.get("lname") == "Chen"
            and item.get("DOB") == "1967-08-14"
        ):
            return item.get("uuid") or item.get("puuid")
    return None


async def _create_chen(client: httpx.AsyncClient, token: str) -> str:
    r = await client.post(
        f"{_api_base()}/patient",
        json=CHEN_DEMOGRAPHICS,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    if r.status_code >= 400:
        print(
            f"Patient create failed ({r.status_code}): {r.text[:400]}",
            file=sys.stderr,
        )
        sys.exit(1)
    body = r.json()
    val = body.get("validationErrors") if isinstance(body, dict) else None
    if isinstance(val, dict) and val:
        print(f"Patient validation failed: {val}", file=sys.stderr)
        sys.exit(1)
    data = body.get("data") if isinstance(body, dict) else body
    flat = data if isinstance(data, dict) else {}
    puuid = flat.get("uuid") or flat.get("puuid") or flat.get("pubpid")
    if not puuid:
        print(
            f"Patient create returned no uuid: {body}",
            file=sys.stderr,
        )
        sys.exit(1)
    return str(puuid)


async def _run() -> int:
    async with httpx.AsyncClient(verify=False, timeout=30) as client:
        print("Acquiring seed-client token...")
        token = await _get_token(client)
        print("  ✓ token acquired")

        print("Checking for existing Chen (idempotent seed)...")
        existing = await _find_existing_chen(client, token)
        if existing:
            print(f"  ✓ Chen already seeded with PUUID = {existing}")
            puuid = existing
        else:
            print("  Creating Margaret Chen...")
            puuid = await _create_chen(client, token)
            print(f"  ✓ Created Patient with PUUID = {puuid}")

        print("\nAppend to clinical-copilot/.env:")
        print(f"DEMO_PATIENT_PUUID={puuid}")
        print(
            "\nThen run: PYTHONPATH=. uv run python scripts/smoke_document_writer.py"
        )
        return 0


def main() -> None:
    sys.exit(asyncio.run(_run()))


if __name__ == "__main__":
    main()
