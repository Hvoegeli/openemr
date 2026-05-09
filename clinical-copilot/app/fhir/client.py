"""OpenEMR FHIR client.

Uses SMART-on-FHIR Backend Services auth (`private_key_jwt`):
  1. We sign a short-lived JWT assertion with our private RSA key.
  2. POST that assertion to /oauth2/default/token with grant_type=client_credentials.
  3. Get a system-level access token.
  4. Cache it until ~30s before expiry, then refresh.

Self-signed cert verification is disabled because the local OpenEMR docker
stack uses a dev cert.
"""

import time
import uuid
from pathlib import Path

import httpx
import jwt

from app.config import settings


class FhirAuthError(RuntimeError):
    """Raised when OAuth credentials are missing or rejected."""


SCOPES = " ".join(
    [
        "system/Patient.read",
        "system/Encounter.read",
        "system/Observation.read",
        "system/Condition.read",
        "system/AllergyIntolerance.read",
        "system/MedicationRequest.read",
        "system/DocumentReference.read",
        "system/Binary.read",
        "system/Appointment.read",
        "system/Practitioner.read",
        "system/CareTeam.read",
    ]
)


class FhirClient:
    """Async client for OpenEMR's FHIR R4 API with private_key_jwt auth."""

    def __init__(self) -> None:
        self._token: str | None = None
        self._token_expires_at: float = 0.0
        self._http = httpx.AsyncClient(verify=False, timeout=30)
        self._private_key: bytes | None = None

    def _load_private_key(self) -> bytes:
        if self._private_key is None:
            self._private_key = Path(settings.openemr_private_key_path).read_bytes()
        return self._private_key

    def _build_client_assertion(self) -> str:
        now = int(time.time())
        claims = {
            "iss": settings.openemr_client_id,
            "sub": settings.openemr_client_id,
            "aud": settings.openemr_oauth_token_url,
            "jti": str(uuid.uuid4()),
            "iat": now,
            "exp": now + 300,
        }
        return jwt.encode(
            claims,
            self._load_private_key(),
            algorithm="RS384",
            headers={"kid": settings.openemr_kid},
        )

    async def _ensure_token(self) -> str:
        if self._token and time.time() < self._token_expires_at - 30:
            return self._token

        if not settings.openemr_client_id:
            raise FhirAuthError(
                "OPENEMR_CLIENT_ID not set. Run scripts/register_oauth_client.py first."
            )

        resp = await self._http.post(
            settings.openemr_oauth_token_url,
            data={
                "grant_type": "client_credentials",
                "scope": SCOPES,
                "client_assertion_type": "urn:ietf:params:oauth:client-assertion-type:jwt-bearer",
                "client_assertion": self._build_client_assertion(),
            },
        )
        if resp.status_code != 200:
            raise FhirAuthError(
                f"OAuth token request failed ({resp.status_code}): {resp.text}\n"
                "If the body says the client is not enabled, log into OpenEMR admin "
                "(https://localhost:9300/, admin/pass) → Admin → System → API Clients "
                "→ enable agent_forge."
            )

        data = resp.json()
        self._token = data["access_token"]
        self._token_expires_at = time.time() + data.get("expires_in", 3600)
        return self._token

    async def get(self, path: str, params: dict | None = None) -> dict:
        """GET a single FHIR resource. `path` is e.g. 'Patient/123'."""
        token = await self._ensure_token()
        url = f"{settings.openemr_fhir_base_url.rstrip('/')}/{path.lstrip('/')}"
        resp = await self._http.get(
            url,
            params=params,
            headers={"Authorization": f"Bearer {token}", "Accept": "application/fhir+json"},
        )
        resp.raise_for_status()
        return resp.json()

    async def search(self, resource: str, params: dict) -> list[dict]:
        """Search a FHIR resource. Returns the entries as a flat list of resources."""
        bundle = await self.get(resource, params=params)
        return [entry["resource"] for entry in bundle.get("entry", [])]

    async def get_raw(self, path: str) -> tuple[bytes, str]:
        """GET a FHIR resource as raw bytes + content-type.

        For endpoints that return non-JSON payloads — primarily `Binary/{id}`,
        which OpenEMR returns as the raw file bytes (not a FHIR-JSON envelope
        with base64 `data`). Returns `(content, content_type)`. Caller is
        responsible for streaming/serving the bytes.
        """
        token = await self._ensure_token()
        url = f"{settings.openemr_fhir_base_url.rstrip('/')}/{path.lstrip('/')}"
        resp = await self._http.get(
            url,
            headers={"Authorization": f"Bearer {token}"},
        )
        resp.raise_for_status()
        ctype = resp.headers.get("content-type", "application/octet-stream")
        return resp.content, ctype

    async def aclose(self) -> None:
        await self._http.aclose()
