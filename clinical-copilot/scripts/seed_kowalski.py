"""Seed Robert J. Kowalski as a Patient resource in OpenEMR for Week 2 demo.

Sister script to `seed_chen.py` and `seed_reyes.py` — same idempotent
pattern, same minimum-demographics philosophy. Robert's stress-test
value is the **mixed format** pair: a clipboard-photo handwritten intake
(`p04-kowalski-intake-handwritten.png` — most visually complex of the
three) plus a typed CMP PDF (`p04-kowalski-cmp.pdf` — has an HTML-entity
rendering bug `CO&sub2;` that tests how the extractor handles malformed
labels).

Identity comes from `data/demo_documents/real/README.md`. DOB is a
plausible adult guess; the actual DOB on the handwritten intake is
what the extractor will read.

After running, paste the printed `DEMO_PATIENT_PUUID_KOWALSKI=...` line
into `clinical-copilot/.env` so future smoke scripts can target him.

Run: `cd clinical-copilot && PYTHONPATH=. uv run python scripts/seed_kowalski.py`
"""

from __future__ import annotations

import asyncio
import sys

import httpx

from app.config import settings


KOWALSKI_DEMOGRAPHICS = {
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


async def _find_existing(client: httpx.AsyncClient, token: str) -> str | None:
    r = await client.get(
        f"{_api_base()}/patient",
        params={"lname": KOWALSKI_DEMOGRAPHICS["lname"], "DOB": KOWALSKI_DEMOGRAPHICS["DOB"]},
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
            item.get("fname") == KOWALSKI_DEMOGRAPHICS["fname"]
            and item.get("lname") == KOWALSKI_DEMOGRAPHICS["lname"]
            and item.get("DOB") == KOWALSKI_DEMOGRAPHICS["DOB"]
        ):
            return item.get("uuid") or item.get("puuid")
    return None


async def _create(client: httpx.AsyncClient, token: str) -> str:
    r = await client.post(
        f"{_api_base()}/patient",
        json=KOWALSKI_DEMOGRAPHICS,
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
        print(f"Patient create returned no uuid: {body}", file=sys.stderr)
        sys.exit(1)
    return str(puuid)


async def _run() -> int:
    async with httpx.AsyncClient(verify=False, timeout=30) as client:
        print("Acquiring seed-client token...")
        token = await _get_token(client)
        print("  ✓ token acquired")

        print("Checking for existing Kowalski (idempotent seed)...")
        existing = await _find_existing(client, token)
        if existing:
            print(f"  ✓ Kowalski already seeded with PUUID = {existing}")
            puuid = existing
        else:
            print("  Creating Robert J. Kowalski...")
            puuid = await _create(client, token)
            print(f"  ✓ Created Patient with PUUID = {puuid}")

        print("\nAppend to clinical-copilot/.env:")
        print(f"DEMO_PATIENT_PUUID_KOWALSKI={puuid}")
        return 0


def main() -> None:
    sys.exit(asyncio.run(_run()))


if __name__ == "__main__":
    main()
