"""Write-capable OpenEMR client.

Two write surfaces live behind this single class because they share the same
``client_secret_post`` + ``password``-grant token:

1. **Standard non-FHIR REST API** at ``/apis/default/api/`` — used for
   resources OpenEMR's FHIR layer doesn't write (vitals are the only Week 1
   user). The seed scripts use this same endpoint shape.

2. **FHIR API** at ``/apis/default/fhir/`` — used for the four resources
   OpenEMR's FHIR layer does write: Patient, Practitioner, Organization,
   DocumentReference. Week 2 adds ``DocumentReference`` writes here as the
   storage path for ingested lab PDFs and intake forms.

The agent's read ``FhirClient`` uses ``private_key_jwt`` +
``client_credentials`` with ``system/*.read`` scopes and cannot write — that
client is intentionally read-only. All writes go through ``OpenEMRWriter``
below.

Best-effort by design for vitals: callers should treat any failure as "the
local JSON-store note is still authoritative" and leave a log line behind.
For Week 2 ``write_document_reference``, failures bubble up — Phase 2
extraction depends on getting back a real DocumentReference ID to use as
the citation ``source_id``, so a silent failure here would corrupt the
extraction-citation contract.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import time
from datetime import datetime, timezone
from typing import Any, Literal

import httpx

from app.config import settings

log = logging.getLogger("agent.fhir.writer")

# Seed-client scope set — must include user/encounter.cruds and
# user/vital.cruds for the vitals POST, and user/DocumentReference.{read,write}
# for the Week 2 source-document persistence path.
_SCOPE = " ".join(
    [
        "openid", "fhirUser", "offline_access",
        "api:oemr", "api:fhir", "api:port",
        "user/patient.cruds", "user/encounter.cruds", "user/vital.cruds",
        "user/DocumentReference.read", "user/DocumentReference.write",
    ]
)

# Idempotency anchor: the SHA-256 of the source file goes into the new
# DocumentReference's `identifier` list under this system URI. Repeat
# uploads are detected by GET /DocumentReference?identifier=<sys>|<value>
# before the POST runs.
DOC_HASH_SYSTEM = "urn:agent_forge:sha256"

# LOINC-coded `type` for each supported document type. Keeps OpenEMR's
# DocumentReference categorization aligned with what real labs use.
DOC_TYPE_CODES: dict[str, dict[str, str]] = {
    "lab_pdf":     {"system": "http://loinc.org", "code": "11502-2", "display": "Laboratory report"},
    "intake_form": {"system": "http://loinc.org", "code": "52040-3", "display": "General intake history and physical note"},
}

DocumentType = Literal["lab_pdf", "intake_form"]


class DocumentAlreadyExists(Exception):
    """Raised by `write_document_reference` when an identical-hash document
    already exists. The exception carries the existing reference ID so the
    caller can use it without re-POSTing."""

    def __init__(self, reference_id: str) -> None:
        super().__init__(f"DocumentReference already exists with id={reference_id}")
        self.reference_id = reference_id

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

    @property
    def _fhir_base(self) -> str:
        """The FHIR R4 base URL — same as the read flow uses."""
        return settings.openemr_fhir_base_url.rstrip("/")

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
        # OpenEMR's vitals POST returns the new row as `vid` (form_vitals
        # primary key); patient/encounter writes use `id`. Try both.
        vid = None
        if isinstance(flat, dict):
            vid = flat.get("vid") or flat.get("id") or flat.get("uuid")
        log.info(
            "wrote vitals to OpenEMR patient=%s encounter_eid=%s vital_id=%s",
            patient_uuid, eid, vid,
        )
        return {"vital_id": vid, "encounter_eid": eid, "patient_pid": pid}

    # ── DocumentReference (FHIR API) ───────────────────────────────────

    @staticmethod
    def _sha256_hex(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    @staticmethod
    def build_document_reference_body(
        *,
        patient_uuid: str,
        doc_type: DocumentType,
        file_bytes: bytes,
        filename: str,
        mime_type: str = "application/pdf",
        creation_iso: str | None = None,
    ) -> dict[str, Any]:
        """Build the FHIR DocumentReference resource body for POST.

        Pulled out as a static method so the body shape is testable in
        isolation without an OpenEMR instance running. The hash that
        appears under `identifier` is the dedupe key — same value passed
        to `find_document_reference_by_hash` for the idempotency check.
        """
        if doc_type not in DOC_TYPE_CODES:
            raise OpenEMRWriteError(
                f"unsupported doc_type {doc_type!r}; expected one of {list(DOC_TYPE_CODES)}"
            )
        sha_hex = OpenEMRWriter._sha256_hex(file_bytes)
        creation = creation_iso or datetime.now(timezone.utc).isoformat()
        return {
            "resourceType": "DocumentReference",
            "status": "current",
            "type": {"coding": [DOC_TYPE_CODES[doc_type]]},
            "subject": {"reference": f"Patient/{patient_uuid}"},
            "date": creation,
            "identifier": [{
                "system": DOC_HASH_SYSTEM,
                "value": sha_hex,
            }],
            "content": [{
                "attachment": {
                    "contentType": mime_type,
                    "data": base64.b64encode(file_bytes).decode("ascii"),
                    "title": filename,
                    "creation": creation,
                },
            }],
        }

    async def find_document_reference_by_hash(self, sha_hex: str) -> str | None:
        """Search for an existing DocumentReference with our SHA-256 identifier.

        Returns the FHIR-style `DocumentReference/{id}` reference of the
        existing resource, or `None` if no match. Used by the idempotency
        check in `write_document_reference`.
        """
        token = await self._ensure_token()
        identifier = f"{DOC_HASH_SYSTEM}|{sha_hex}"
        r = await self._http.get(
            f"{self._fhir_base}/DocumentReference",
            params={"identifier": identifier},
            headers={"Authorization": f"Bearer {token}", "Accept": "application/fhir+json"},
        )
        if r.status_code != 200:
            raise OpenEMRWriteError(
                f"DocumentReference search failed ({r.status_code}): {r.text[:300]}"
            )
        bundle = r.json()
        # FHIR Bundle response — entries live under `entry[].resource`.
        entries = bundle.get("entry") if isinstance(bundle, dict) else None
        if not entries:
            return None
        first = entries[0].get("resource") if isinstance(entries[0], dict) else None
        if not isinstance(first, dict):
            return None
        rid = first.get("id")
        return f"DocumentReference/{rid}" if rid else None

    async def write_document_reference(
        self,
        *,
        patient_uuid: str,
        doc_type: DocumentType,
        file_bytes: bytes,
        filename: str,
        mime_type: str = "application/pdf",
        creation_iso: str | None = None,
    ) -> dict[str, Any]:
        """Persist a source document to OpenEMR as a FHIR DocumentReference.

        Idempotent: if a DocumentReference with the same SHA-256 identifier
        already exists, returns its ID without POSTing a duplicate.

        Returns:
            dict with keys:
              - `reference_id`: FHIR-style `DocumentReference/{id}`
              - `sha256`: hex SHA-256 of the file
              - `created`: True if newly created, False if found via dedupe
        """
        sha_hex = self._sha256_hex(file_bytes)
        existing = await self.find_document_reference_by_hash(sha_hex)
        if existing:
            log.info(
                "DocumentReference dedupe hit patient=%s sha256=%s -> %s",
                patient_uuid, sha_hex[:12], existing,
            )
            return {"reference_id": existing, "sha256": sha_hex, "created": False}

        body = self.build_document_reference_body(
            patient_uuid=patient_uuid,
            doc_type=doc_type,
            file_bytes=file_bytes,
            filename=filename,
            mime_type=mime_type,
            creation_iso=creation_iso,
        )
        token = await self._ensure_token()
        r = await self._http.post(
            f"{self._fhir_base}/DocumentReference",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/fhir+json",
                "Accept": "application/fhir+json",
            },
            json=body,
        )
        if r.status_code >= 400:
            raise OpenEMRWriteError(
                f"DocumentReference POST failed ({r.status_code}): {r.text[:300]}"
            )
        # OpenEMR's FHIR write returns 201 with Location header containing the
        # new resource path. Body shape can vary; use Location when present,
        # fall back to body.id.
        location = r.headers.get("Location") or r.headers.get("location")
        rid: str | None = None
        if location:
            # Location is "<base>/DocumentReference/<id>/_history/<v>" or
            # "<base>/DocumentReference/<id>". Pluck the id between the two.
            parts = location.rstrip("/").split("/DocumentReference/")
            if len(parts) == 2:
                tail = parts[1].split("/")[0]
                rid = tail
        if rid is None:
            try:
                payload = r.json()
                if isinstance(payload, dict):
                    rid = payload.get("id")
            except (ValueError, KeyError):
                rid = None
        if not rid:
            raise OpenEMRWriteError(
                f"DocumentReference POST returned no id (status={r.status_code}, "
                f"location={location!r}, body={r.text[:200]})"
            )
        reference_id = f"DocumentReference/{rid}"
        log.info(
            "wrote DocumentReference patient=%s doc_type=%s sha256=%s -> %s",
            patient_uuid, doc_type, sha_hex[:12], reference_id,
        )
        return {"reference_id": reference_id, "sha256": sha_hex, "created": True}
