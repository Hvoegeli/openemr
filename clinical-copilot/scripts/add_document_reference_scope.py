"""Add the Week 2 DocumentReference scopes to an EXISTING agent_forge_seed
client without re-registering (which would mint a new client_id/secret pair
and force an .env edit).

Use this when you have a Week 1 .env wired up and don't want to rotate the
seed client credentials. Otherwise just re-run `register_seed_client.py` —
the SCOPES list there now includes the new DocumentReference scopes.

Run: PYTHONPATH=. uv run python scripts/add_document_reference_scope.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
# Resolve the OpenEMR docker-compose dir relative to the repo root so the
# script runs unmodified on macOS (`/Users/.../openemr`), Hetzner
# (`/root/openemr`), and any other clone location. Override with the
# `OPENEMR_DOCKER_DIR` env var for unusual setups.
DEV_EASY_DIR = os.environ.get(
    "OPENEMR_DOCKER_DIR",
    str(REPO_ROOT.parent / "docker" / "development-easy"),
)

NEW_SCOPES = [
    # Standard non-FHIR API (multipart upload) — required for the actual
    # write path used by app/fhir/writer.py:write_document_reference.
    "user/document.cruds",
    # FHIR DocumentReference read — used after upload to resolve the new
    # doc's FHIR resource id, and by the chart-summarizer for the
    # Supporting Documents tab.
    "user/DocumentReference.read",
    "user/DocumentReference.write",
]


def _load_seed_client_id() -> str:
    load_dotenv(REPO_ROOT / ".env")
    client_id = os.getenv("OPENEMR_SEED_CLIENT_ID")
    if not client_id:
        print(
            "OPENEMR_SEED_CLIENT_ID not set in .env — run "
            "scripts/register_seed_client.py first to create the seed client.",
            file=sys.stderr,
        )
        sys.exit(1)
    return client_id


def _current_scopes(client_id: str) -> str:
    sql = (
        "SELECT scope FROM oauth_clients "
        f"WHERE client_id = '{client_id}'\\G"
    )
    cmd = [
        "docker", "compose", "exec", "-T", "mysql",
        "mariadb", "-u", "openemr", "-popenemr", "openemr", "-e", sql,
    ]
    result = subprocess.run(
        cmd, cwd=DEV_EASY_DIR, capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        print(f"DB read failed: {result.stderr}", file=sys.stderr)
        sys.exit(1)
    # Output looks like "*** 1. row ***\nscope: openid api:fhir ..."
    for line in result.stdout.splitlines():
        if line.strip().startswith("scope:"):
            return line.split(":", 1)[1].strip()
    return ""


def _set_scopes(client_id: str, new_scope_str: str) -> None:
    # Escape single quotes for the SQL string. Scope values from OpenEMR's
    # registration flow do not contain quotes, so a simple replace suffices.
    safe = new_scope_str.replace("'", "''")
    sql = (
        f"UPDATE oauth_clients SET scope = '{safe}' "
        f"WHERE client_id = '{client_id}';"
    )
    cmd = [
        "docker", "compose", "exec", "-T", "mysql",
        "mariadb", "-u", "openemr", "-popenemr", "openemr", "-e", sql,
    ]
    result = subprocess.run(
        cmd, cwd=DEV_EASY_DIR, capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        print(f"DB update failed: {result.stderr}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    client_id = _load_seed_client_id()
    print(f"Updating scopes for client_id = {client_id}")

    current = _current_scopes(client_id)
    if not current:
        print(
            "Could not read current scopes — verify the seed client exists "
            "in oauth_clients and the development-easy docker compose stack "
            "is running.",
            file=sys.stderr,
        )
        sys.exit(1)

    have = set(current.split())
    missing = [s for s in NEW_SCOPES if s not in have]
    if not missing:
        print("  ✓ DocumentReference scopes already present — nothing to do.")
        return

    new_scope_str = current + " " + " ".join(missing)
    _set_scopes(client_id, new_scope_str)
    print(f"  ✓ added: {', '.join(missing)}")
    print("\nNote: existing access tokens may not include the new scope —")
    print("the next password-grant token request will pick it up.")


if __name__ == "__main__":
    main()
