"""Write-capable OpenEMR client for pushing finalized clinical-note vitals
into the EHR's standard vitals table.

The agent's read FhirClient uses ``private_key_jwt`` + ``client_credentials``
with ``system/*.read`` scopes — it cannot write. This module mirrors the
seed scripts: ``client_secret_post`` + ``password`` grant + ``user/*.cruds``
scopes hit OpenEMR's *standard* (non-FHIR) REST API which exposes the vitals
write endpoint that OpenEMR's own UI uses. The FHIR layer reads from the
same underlying ``form_vitals`` table, so a successful write here surfaces
on the next FHIR Observation search — that's what makes the trend chart
and the chart UI both reflect the doctor's input.

Best-effort by design: callers should treat any failure as "the local
JSON-store note is still authoritative" and leave a log line behind. The
clinical-note finalize flow must not fail just because the EHR write did.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

from app.config import settings

log = logging.getLogger("agent.fhir.writer")

# Seed-client scope set — must include user/encounter.cruds and
# user/vital.cruds for the encounter lookup + vitals POST below.
_SCOPE = " ".join(
    [
        "openid", "fhirUser", "offline_access",
        "api:oemr", "api:fhir", "api:port",
        "user/patient.cruds", "user/encounter.cruds", "user/vital.cruds",
    ]
)

# Map our canonical-key vitals onto OpenEMR's vitals-form column names.
# OpenEMR stores temperature in °F by default; the seed script confirms.
_FIELD_MAP: dict[str, str] = {
    "bp_systolic":      "bps",
    "bp_diastolic":     "bpd",
    "heart_rate":       "pulse",
    "respiratory_rate": "respiration",
    "temp_f":           "temperature",
    "spo2":             "oxygen_saturation",
}


class OpenEMRWriteError(RuntimeError):
    """Raised when the EHR rejects a write or the credentials are missing."""


class OpenEMRWriter:
    """Encapsulates the password-grant token + standard-API write surface."""

    def __init__(self) -> None:
        self._token: str | None = None
        self._token_expires_at: float = 0.0
        self._http = httpx.AsyncClient(verify=False, timeout=30)

    @property
    def _api_base(self) -> str:
        # Derive the standard-API base from the FHIR base, e.g.
        #   https://localhost:9300/apis/default/fhir  →
        #   https://localhost:9300/apis/default/api
        return settings.openemr_fhir_base_url.rstrip("/").rsplit("/", 1)[0] + "/api"

    async def aclose(self) -> None:
        await self._http.aclose()

    # ── auth ───────────────────────────────────────────────────────────
    async def _ensure_token(self) -> str:
        if self._token and time.time() < self._token_expires_at - 30:
            return self._token
        if not (settings.openemr_seed_client_id and settings.openemr_seed_client_secret):
            raise OpenEMRWriteError(
                "OPENEMR_SEED_CLIENT_ID / OPENEMR_SEED_CLIENT_SECRET not set"
            )
        body = {
            "grant_type": "password",
            "client_id": settings.openemr_seed_client_id,
            "client_secret": settings.openemr_seed_client_secret,
            "username": "admin",
            "password": "pass",
            "user_role": "users",
            "scope": _SCOPE,
        }
        r = await self._http.post(settings.openemr_oauth_token_url, data=body)
        if r.status_code != 200:
            raise OpenEMRWriteError(
                f"seed-client password-grant failed ({r.status_code}): {r.text[:300]}"
            )
        data = r.json()
        self._token = data["access_token"]
        self._token_expires_at = time.time() + data.get("expires_in", 3600)
        return self._token

    # ── lookups ────────────────────────────────────────────────────────
    async def _patient_numeric_pid(self, token: str, patient_uuid: str) -> int:
        """Resolve a FHIR Patient UUID to OpenEMR's internal numeric pid."""
        r = await self._http.get(
            f"{self._api_base}/patient/{patient_uuid}",
            headers={"Authorization": f"Bearer {token}"},
        )
        if r.status_code != 200:
            raise OpenEMRWriteError(
                f"patient lookup failed for {patient_uuid}: {r.status_code} {r.text[:200]}"
            )
        body = r.json()
        data = body.get("data") if isinstance(body, dict) else None
        if isinstance(data, dict):
            pid = data.get("pid")
        else:
            pid = body.get("pid") if isinstance(body, dict) else None
        if not pid:
            raise OpenEMRWriteError(f"patient {patient_uuid} response missing pid: {body}")
        return int(pid)

    async def _latest_encounter_eid(self, token: str, patient_uuid: str) -> int:
        """Find the patient's most recent encounter and return its numeric eid.

        Vitals attach to an encounter, so a write needs one. OpenEMR's standard
        API returns encounters newest-first when sorted by date.
        """
        r = await self._http.get(
            f"{self._api_base}/patient/{patient_uuid}/encounter",
            headers={"Authorization": f"Bearer {token}"},
        )
        if r.status_code != 200:
            raise OpenEMRWriteError(
                f"encounter lookup failed for {patient_uuid}: {r.status_code} {r.text[:200]}"
            )
        body = r.json()
        items = body.get("data") if isinstance(body, dict) else body
        if not isinstance(items, list) or not items:
            raise OpenEMRWriteError(f"no encounters on file for patient {patient_uuid}")
        # Pick the most recent by `date` field (string compare on ISO works)
        items.sort(key=lambda x: x.get("date") or "", reverse=True)
        eid = items[0].get("eid") or items[0].get("id")
        if not eid:
            raise OpenEMRWriteError(f"encounter response missing eid: {items[0]}")
        return int(eid)

    # ── write ──────────────────────────────────────────────────────────
    async def write_vitals(
        self, *, patient_uuid: str, vitals: dict[str, Any], when_iso: str,
    ) -> dict[str, Any]:
        """Push a finalized clinical note's vitals into OpenEMR.

        Returns a small result dict with the new vitals-row id and the
        encounter it landed on. Raises ``OpenEMRWriteError`` on any failure.
        """
        if not vitals:
            raise OpenEMRWriteError("no vitals to write")

        token = await self._ensure_token()
        pid = await self._patient_numeric_pid(token, patient_uuid)
        eid = await self._latest_encounter_eid(token, patient_uuid)

        body: dict[str, Any] = {"date": when_iso.replace("T", " ").split("+")[0][:19]}
        for canonical, openemr_field in _FIELD_MAP.items():
            v = vitals.get(canonical)
            if v in (None, ""):
                continue
            body[openemr_field] = str(v)
        # Temp method is required when temperature is sent.
        if "temperature" in body:
            body.setdefault("temp_method", "Oral")

        r = await self._http.post(
            f"{self._api_base}/patient/{pid}/encounter/{eid}/vital",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json=body,
        )
        if r.status_code >= 400:
            raise OpenEMRWriteError(
                f"vitals POST failed ({r.status_code}): {r.text[:300]}"
            )
        payload = r.json()
        # Standard API shape: {"validationErrors":[], "data":{"id":..., "uuid":...}}
        val = payload.get("validationErrors") if isinstance(payload, dict) else None
        if isinstance(val, dict) and val:
            raise OpenEMRWriteError(f"vitals validation failed: {val}")
        data = payload.get("data") if isinstance(payload, dict) else None
        flat = data if isinstance(data, dict) else payload
        vid = flat.get("id") if isinstance(flat, dict) else None
        log.info(
            "wrote vitals to OpenEMR patient=%s encounter_eid=%s vital_id=%s",
            patient_uuid, eid, vid,
        )
        return {"vital_id": vid, "encounter_eid": eid, "patient_pid": pid}
