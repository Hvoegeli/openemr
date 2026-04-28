"""One-time setup: register agent_forge as an OAuth2 client in OpenEMR.

OpenEMR's SMART v2 registration endpoint requires a JWKS even when using
client_secret auth. This script:
  1. Generates an RSA keypair and saves the private key to secrets/agent_key.pem
  2. Constructs a JWKS from the public key
  3. POSTs dynamic registration to /oauth2/default/registration
  4. Prints the client_id (and client_secret if returned) to paste into .env
  5. Tells you what to click in the OpenEMR admin UI to activate the client

Run: uv run python scripts/register_oauth_client.py
"""

import base64
import json
import sys
import urllib3
from pathlib import Path

import httpx
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

REGISTRATION_URL = "https://localhost:9300/oauth2/default/registration"
KEY_PATH = Path("secrets/agent_key.pem")
JWKS_PATH = Path("secrets/agent_jwks.json")

SCOPES = " ".join(
    [
        "system/Patient.read",
        "system/Encounter.read",
        "system/Observation.read",
        "system/Condition.read",
        "system/AllergyIntolerance.read",
        "system/MedicationRequest.read",
        "system/DocumentReference.read",
        "system/Practitioner.read",
    ]
)


def _b64u(n: int) -> str:
    """base64url-encode an integer (no padding) — JWK format."""
    raw = n.to_bytes((n.bit_length() + 7) // 8, "big")
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def generate_keypair() -> tuple[rsa.RSAPrivateKey, dict]:
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_numbers = private_key.public_key().public_numbers()
    jwk = {
        "kty": "RSA",
        "alg": "RS384",
        "use": "sig",
        "kid": "agent_forge_key_1",
        "n": _b64u(public_numbers.n),
        "e": _b64u(public_numbers.e),
    }
    return private_key, {"keys": [jwk]}


def save_keys(private_key: rsa.RSAPrivateKey, jwks: dict) -> None:
    KEY_PATH.parent.mkdir(parents=True, exist_ok=True)
    KEY_PATH.write_bytes(
        private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    KEY_PATH.chmod(0o600)
    JWKS_PATH.write_text(json.dumps(jwks, indent=2))


def register(jwks: dict) -> dict:
    payload = {
        "application_type": "private",
        "redirect_uris": ["https://localhost:9300/callback"],
        "client_name": "agent_forge",
        "token_endpoint_auth_method": "private_key_jwt",
        "contacts": ["agent@local"],
        "scope": SCOPES,
        "jwks": jwks,
    }
    resp = httpx.post(REGISTRATION_URL, json=payload, verify=False, timeout=30)
    if resp.status_code not in (200, 201):
        print(f"Registration failed: HTTP {resp.status_code}", file=sys.stderr)
        print(resp.text, file=sys.stderr)
        sys.exit(1)
    return resp.json()


def main() -> None:
    print("Generating RSA keypair...")
    private_key, jwks = generate_keypair()
    save_keys(private_key, jwks)
    print(f"  private key  → {KEY_PATH}")
    print(f"  jwks         → {JWKS_PATH}")

    print("\nRegistering with OpenEMR...")
    result = register(jwks)
    client_id = result.get("client_id")
    client_secret = result.get("client_secret")  # may be None for private_key_jwt

    print("\n✓ Client registered. Paste these into .env:\n")
    print(f"OPENEMR_CLIENT_ID={client_id}")
    if client_secret:
        print(f"OPENEMR_CLIENT_SECRET={client_secret}")
    print("\nFull registration response:")
    print(json.dumps(result, indent=2))

    print(
        "\n→ NEXT: open https://localhost:9300/ → log in as admin/pass → "
        "Admin → System → API Clients → find 'agent_forge' → click 'Enable Client'."
    )


if __name__ == "__main__":
    main()
