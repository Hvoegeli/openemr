"""Register a second OAuth client specifically for seeding demo data.

The primary `agent_forge` client uses `private_key_jwt` + `system/*.read` scopes
and is what the running agent uses for FHIR queries. That client cannot write
clinical data because OpenEMR's FHIR API only writes a small subset of
resources.

This script registers `agent_forge_seed`, a separate client with
`client_secret_post` + `password` grant + `user/*.{read,write}` scopes, so we
can POST allergies, problems, medications, encounters, and vitals to the
standard REST API as user `admin`.

Outputs SEED_CLIENT_ID and SEED_CLIENT_SECRET which the seed script will read
from .env. Also writes is_enabled=1 directly to the DB so no manual UI step is
needed.

Run: PYTHONPATH=. uv run python scripts/register_seed_client.py
"""

import json
import subprocess
import sys

import httpx
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

REGISTRATION_URL = "https://localhost:9300/oauth2/default/registration"
DEV_EASY_DIR = "/Users/harrisonvoegeli/openemr/docker/development-easy"

SCOPES = " ".join(
    [
        "openid",
        "api:oemr",
        "api:fhir",
        "user/Patient.read",
        "user/AllergyIntolerance.read",
        "user/AllergyIntolerance.write",
        "user/Condition.read",
        "user/Condition.write",
        "user/MedicationRequest.read",
        "user/MedicationRequest.write",
        "user/Encounter.read",
        "user/Encounter.write",
        "user/Observation.read",
        "user/Observation.write",
    ]
)


def register() -> dict:
    payload = {
        "application_type": "private",
        "redirect_uris": ["https://localhost:9300/callback"],
        "client_name": "agent_forge_seed",
        "token_endpoint_auth_method": "client_secret_post",
        "contacts": ["seed@agent_forge.local"],
        "scope": SCOPES,
    }
    resp = httpx.post(REGISTRATION_URL, json=payload, verify=False, timeout=30)
    if resp.status_code not in (200, 201):
        print(f"Registration failed: HTTP {resp.status_code}", file=sys.stderr)
        print(resp.text, file=sys.stderr)
        sys.exit(1)
    return resp.json()


def enable_in_db_and_grant_password(client_id: str) -> None:
    """Mark the client enabled and add `password` to grant_types via direct SQL."""
    sql = f"""
        UPDATE oauth_clients
        SET is_enabled = 1,
            grant_types = 'authorization_code client_credentials password'
        WHERE client_id = '{client_id}';
    """
    cmd = [
        "docker", "compose", "exec", "-T", "mysql",
        "mariadb", "-u", "openemr", "-popenemr", "openemr", "-e", sql,
    ]
    result = subprocess.run(cmd, cwd=DEV_EASY_DIR, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"DB update failed: {result.stderr}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    print("Registering agent_forge_seed client...")
    result = register()
    client_id = result["client_id"]
    client_secret = result.get("client_secret")
    if not client_secret:
        print("Registration did not return client_secret — aborting.", file=sys.stderr)
        print(json.dumps(result, indent=2), file=sys.stderr)
        sys.exit(1)
    print(f"  client_id     = {client_id}")
    print(f"  client_secret = {client_secret[:8]}...{client_secret[-4:]}")

    print("Enabling client + adding password grant in DB...")
    enable_in_db_and_grant_password(client_id)
    print("  ✓ client enabled, password grant allowed")

    print("\nAppend to .env:")
    print(f"OPENEMR_SEED_CLIENT_ID={client_id}")
    print(f"OPENEMR_SEED_CLIENT_SECRET={client_secret}")


if __name__ == "__main__":
    main()
