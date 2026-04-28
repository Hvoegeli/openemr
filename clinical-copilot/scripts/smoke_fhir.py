"""End-to-end FHIR smoke test.

Checks: token auth → /metadata → list patients → fetch one patient.
Run AFTER enabling agent_forge in OpenEMR admin → System → API Clients.

Run: uv run python scripts/smoke_fhir.py
"""

import asyncio

from app.fhir.client import FhirClient


async def main() -> None:
    client = FhirClient()
    try:
        print("Requesting token...")
        token = await client._ensure_token()
        print(f"  ✓ token acquired ({len(token)} chars)")

        print("\nFetching FHIR /metadata...")
        meta = await client.get("metadata")
        rest = meta.get("rest", [{}])[0]
        n_resources = len(rest.get("resource", []))
        print(f"  ✓ FHIR server reports {n_resources} resource types")

        print("\nSearching Patient (first 5)...")
        patients = await client.search("Patient", {"_count": 5})
        print(f"  ✓ found {len(patients)} patients")
        for p in patients:
            name = (p.get("name") or [{}])[0]
            full = " ".join(name.get("given", [])) + " " + name.get("family", "")
            print(f"    - {p['id']}: {full.strip()}")

        if patients:
            pid = patients[0]["id"]
            print(f"\nFetching Patient/{pid}...")
            single = await client.get(f"Patient/{pid}")
            print(f"  ✓ resourceType={single['resourceType']} id={single['id']}")

        print("\n✓ FHIR smoke test passed.")
    finally:
        await client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
