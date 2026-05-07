"""Write-capable OpenEMR client.

Two write surfaces live behind this single class because they share the same
``client_secret_post`` + ``password``-grant token:

1. **Standard non-FHIR REST API** at ``/apis/default/api/`` — used for both
   the Week 1 vitals round-trip AND the Week 2 source-document upload.
   OpenEMR's FHIR layer does NOT route ``POST /fhir/DocumentReference``
   (despite advertising ``create`` in its CapabilityStatement), so document
   writes go through ``POST /apis/default/api/patient/{pid}/document`` with
   a multipart-form upload. Reads still come back through the FHIR layer
   (``GET /fhir/DocumentReference?patient=...``) — same Supporting
   Documents tab the chart UI already uses, no parallel surface.

2. **FHIR API** at ``/apis/default/fhir/`` — used by ``OpenEMRWriter`` only
   for *reads* (DocumentReference search after a multipart upload, to
   resolve the new resource's stable FHIR id). The agent's read
   ``FhirClient`` separately handles all the chart-summarizer FHIR queries
   with a different OAuth flow.

The agent's read ``FhirClient`` uses ``private_key_jwt`` +
``client_credentials`` with ``system/*.read`` scopes and cannot write — that
client is intentionally read-only. All writes go through ``OpenEMRWriter``
below.

Best-effort by design for vitals: callers should treat any failure as "the
local JSON-store note is still authoritative" and leave a log line behind.
For Week 2 ``write_document_reference``, failures bubble up — Phase 2
extraction depends on getting back a real DocumentReference id to use as
the citation ``source_id``, so a silent failure here would corrupt the
extraction-citation contract.

Idempotency for ``write_document_reference`` rides in the upload filename:
the SHA-256 of the file bytes is prepended (``sha256-<hex>__<original>``)
so a re-upload is detected by FHIR-GET search on attachment title without
needing a parallel dedupe table on the co-pilot side. Round-tripping
through OpenEMR keeps OpenEMR as the single source of truth for what's
been persisted.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import date
from typing import Any

import httpx

from app.config import settings
from app.extraction.schemas import DocumentType

log = logging.getLogger("agent.fhir.writer")

# Seed-client scope set — must include user/encounter.cruds and
# user/vital.cruds for the vitals POST, user/document.cruds for the
# Week 2 multipart document upload, user/DocumentReference.read for
# the FHIR search that resolves the uploaded doc's FHIR resource id,
# user/appointment.cruds for the calendar / scheduling endpoint, and
# user/{allergy,medical_problem,medication}.cruds for the Phase-3
# extracted-fact persistence path (write_allergy / write_medical_problem
# / write_medication). The scopes registered on the seed client must
# be a superset of these — the OAuth password grant only includes
# scopes that are BOTH requested here AND registered on the client.
_SCOPE = " ".join(
    [
        "openid", "fhirUser", "offline_access",
        "api:oemr", "api:fhir", "api:port",
        "user/patient.cruds", "user/encounter.cruds", "user/vital.cruds",
        "user/document.cruds", "user/DocumentReference.read",
        "user/appointment.cruds",
        "user/allergy.cruds", "user/medical_problem.cruds",
        "user/medication.cruds",
    ]
)

# Map our doc_type vocabulary to OpenEMR's category-path convention.
# OpenEMR's `getLastIdOfPath` looks up categories by `replace(LOWER(name),
# ' ', '')` so we must pass the path pre-normalized — the standard category
# names are "Lab Report" and "Patient Information" (top-level, parent=1).
# (Spaces and case in the path query string would otherwise silently fail
# to match, leaving the document unlinked from any category.)
DOC_CATEGORIES: dict[str, str] = {
    "lab_pdf":         "labreport",          # → OpenEMR "Lab Report" (id=2)
    "intake_form":     "patientinformation",  # → OpenEMR "Patient Information" (id=4)
    # The dev-easy seed has no dedicated "Referral" category; folding
    # referrals into "Patient Information" keeps them linked to the chart
    # without requiring a schema migration. The doc_type still rides in
    # our app-level taxonomy (DOC_TYPE_LABELS / DocumentType), so the
    # extractor and persistence branches treat referrals distinctly.
    # Title-based inference at the prefetch boundary picks "referral_letter"
    # back out via filename keyword.
    "referral_letter": "patientinformation",
    # The dev-easy seed has no dedicated HL7 category. Folding HL7
    # messages into "Patient Information" keeps them on the chart with
    # zero schema migration; the doc_type lives in our app-level
    # taxonomy and the title-keyword inference (`hl7` / `adt` / `oru`
    # in the filename) picks them back out at the prefetch boundary.
    "hl7_message":     "patientinformation",
    # No dedicated "Fax" category in the dev-easy seed either. Same
    # pattern: persist under "Patient Information" + recover the
    # doc_type via the `fax` filename keyword on the prefetch path.
    "fax_packet":      "patientinformation",
    # No dedicated "Workbook" category in the dev-easy seed. Same
    # pattern: persist under "Patient Information"; the `.xlsx` /
    # `workbook` filename keywords recover the doc_type at prefetch.
    "workbook":        "patientinformation",
}


def _idempotency_filename(sha_hex: str, original_filename: str) -> str:
    """Prepend the SHA-256 to the filename so re-uploads of the same bytes
    are detectable via FHIR DocumentReference search on attachment.title.

    Example: ``sha256-7c4a8d09…__p01-chen-lipid-panel.pdf``.
    """
    return f"sha256-{sha_hex}__{original_filename}"


class DocumentAlreadyExists(Exception):
    """Raised when a write would create a duplicate. Currently informational
    only — `write_document_reference` returns ``created=False`` on dedupe
    rather than raising. Exposed so future callers can opt into raise-style
    handling if useful."""

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
    async def write_patient(
        self,
        *,
        given_name: str,
        family_name: str,
        date_of_birth: str | None = None,
        sex: str | None = None,
        street: str | None = None,
        city: str | None = None,
        state: str | None = None,
        postal_code: str | None = None,
        phone: str | None = None,
    ) -> dict:
        """Create a new Patient via OpenEMR's standard REST API.

        Mirrors `seed_demo_patients.py`'s create-patient call: POST to
        `/patient` with fname/lname/DOB/sex/address fields and a normalized
        `Title` for compatibility with the patient-search idempotency check.

        Args:
            given_name: First name (maps to `fname`).
            family_name: Last name (maps to `lname`).
            date_of_birth: ISO YYYY-MM-DD or None when the doc didn't have one.
            sex: One of "Male", "Female", "Other", "Unknown" or None.
                The intake-form schema uses lowercase `male/female/other/unknown`;
                callers should capitalize before passing.
            street, city, state, postal_code: Address fields, all optional.
            phone: Best-known contact phone, optional.

        Returns:
            Flat dict from the standard-API envelope. Includes `pid` (legacy
            integer) and `uuid` (FHIR UUID).

        Raises:
            OpenEMRWriteError on non-2xx, validation errors, or unparsable body.
        """
        body: dict = {"fname": given_name, "lname": family_name}
        if date_of_birth:
            body["DOB"] = date_of_birth
        if sex:
            body["sex"] = sex
        if street:
            body["street"] = street
        if city:
            body["city"] = city
        if state:
            body["state"] = state
        if postal_code:
            body["postal_code"] = postal_code
        if phone:
            body["phone_contact"] = phone
        result = await self._post_standard_api(
            path="/patient",
            body=body,
            label=f"create_patient {family_name}, {given_name}",
        )
        log.info(
            "write_patient: created %s, %s pid=%s uuid=%s",
            family_name, given_name, result.get("pid"), result.get("uuid"),
        )
        return result

    async def resolve_user_id_by_username(self, username: str) -> int | None:
        """Resolve an OpenEMR username to its legacy ``users.id`` integer
        via ``GET /api/user?username=…``.

        OpenEMR's ``users`` table enforces a unique ``username`` so the
        lookup returns at most one row. The integer ``id`` returned is
        the value the appointment-write endpoint needs in ``pc_aid``.

        Returns ``None`` when no row matches, or when every match is
        marked inactive (``active=0`` / ``false``). Raises
        ``OpenEMRWriteError`` only on a non-200 transport response — a
        no-match returns 200 with an empty data list.
        """
        if not username:
            return None
        token = await self._ensure_token()
        r = await self._http.get(
            f"{self._api_base}/user",
            headers={"Authorization": f"Bearer {token}"},
            params={"username": username},
        )
        if r.status_code != 200:
            raise OpenEMRWriteError(
                f"user lookup failed for {username!r}: "
                f"{r.status_code} {r.text[:200]}"
            )
        try:
            body = r.json()
        except ValueError as exc:
            raise OpenEMRWriteError(
                f"user lookup returned non-JSON for {username!r}: "
                f"{r.text[:200]}"
            ) from exc
        rows = body.get("data") if isinstance(body, dict) else body
        if not isinstance(rows, list):
            return None
        for row in rows:
            if not isinstance(row, dict):
                continue
            # OpenEMR's users.active is a smallint(1) but the JSON
            # serializer surfaces it as int (0/1) on some routes and
            # bool elsewhere. Treat any falsy form as inactive.
            active = row.get("active")
            if active in (False, 0, "0"):
                continue
            rid = row.get("id")
            if rid is None:
                continue
            try:
                return int(rid)
            except (TypeError, ValueError):
                continue
        return None

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

    # ── DocumentReference (multipart upload + FHIR read) ──────────────

    @staticmethod
    def _sha256_hex(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    @staticmethod
    def resolve_doc_category(doc_type: DocumentType) -> str:
        """Return the OpenEMR category-path string for a doc type.

        The path string is what the standard upload endpoint expects in its
        ``?path=`` query parameter. Pre-normalized (lowercased, no spaces)
        because OpenEMR's `getLastIdOfPath` does a case-sensitive comparison
        against `replace(LOWER(name), ' ', '')`."""
        try:
            return DOC_CATEGORIES[doc_type]
        except KeyError as exc:
            raise OpenEMRWriteError(
                f"unsupported doc_type {doc_type!r}; expected one of {list(DOC_CATEGORIES)}"
            ) from exc

    async def find_document_reference_by_sha(
        self, patient_uuid: str, sha_hex: str,
    ) -> str | None:
        """Search FHIR DocumentReference for an attachment.title that starts
        with our SHA-256 prefix.

        Same SHA = same file bytes = same logical document, regardless of
        the human-readable suffix the user chose for the attachment.title.
        Matching by `startswith("sha256-<hex>__")` correctly dedupes a
        re-upload of identical bytes that the user renamed in the upload
        form. Matching by full filename equality (the prior behavior)
        missed those because a typed display name on each upload changed
        the suffix.

        Returns ``DocumentReference/{uuid}`` if found, ``None`` otherwise.
        The OpenEMR FHIR read flow is paginated by default (50 entries);
        for MVP we accept that very-many-document patients (>50) might
        miss a dedupe hit on a doc deeper in the bundle. Tracked as a
        post-MVP concern in W2_ARCHITECTURE.md §4.5.
        """
        prefix = f"sha256-{sha_hex}__"
        token = await self._ensure_token()
        r = await self._http.get(
            f"{self._fhir_base}/DocumentReference",
            params={"patient": patient_uuid},
            headers={"Authorization": f"Bearer {token}", "Accept": "application/fhir+json"},
        )
        if r.status_code != 200:
            raise OpenEMRWriteError(
                f"DocumentReference search failed ({r.status_code}): {r.text[:300]}"
            )
        bundle = r.json()
        entries = bundle.get("entry") if isinstance(bundle, dict) else None
        if not entries:
            return None
        for entry in entries:
            resource = entry.get("resource") if isinstance(entry, dict) else None
            if not isinstance(resource, dict):
                continue
            for content in resource.get("content") or []:
                if not isinstance(content, dict):
                    continue
                attach = content.get("attachment") or {}
                if not isinstance(attach, dict):
                    continue
                title = attach.get("title") or ""
                if isinstance(title, str) and title.startswith(prefix):
                    rid = resource.get("id")
                    if rid:
                        return f"DocumentReference/{rid}"
        return None

    async def write_document_reference(
        self,
        *,
        patient_uuid: str,
        doc_type: DocumentType,
        file_bytes: bytes,
        filename: str,
        mime_type: str = "application/pdf",
    ) -> dict[str, Any]:
        """Persist a source document to OpenEMR via the standard upload API
        and resolve the resulting FHIR DocumentReference id.

        Two phases:

        1. **Idempotency check** — compute SHA-256, prepend to the filename
           (``sha256-<hex>__<original>``), and search FHIR DocumentReference
           for an existing match by attachment.title. A hit returns the
           existing id without re-POSTing.

        2. **Upload + resolve** — multipart POST to
           ``/api/patient/{pid}/document?path={category}`` with the
           hash-prefixed filename. Then FHIR-GET the patient's
           DocumentReferences to find the just-uploaded resource by title
           and return its FHIR id.

        Returns:
            dict with keys:
              - ``reference_id``: ``DocumentReference/{fhir_uuid}``
              - ``sha256``: hex SHA-256 of the file
              - ``created``: True if newly uploaded, False if dedupe hit
        """
        category = self.resolve_doc_category(doc_type)
        sha_hex = self._sha256_hex(file_bytes)
        idem_filename = _idempotency_filename(sha_hex, filename)

        # Phase 1 — dedupe by SHA prefix. We match the SHA prefix only (not
        # the full hash-prefixed filename) because the user-supplied display
        # name in the upload form can vary across uploads of the same file
        # bytes. Same bytes = same SHA = same logical document.
        existing = await self.find_document_reference_by_sha(
            patient_uuid, sha_hex,
        )
        if existing:
            log.info(
                "DocumentReference dedupe hit patient=%s sha256=%s -> %s",
                patient_uuid, sha_hex[:12], existing,
            )
            return {"reference_id": existing, "sha256": sha_hex, "created": False}

        # Phase 2 — upload via standard non-FHIR API, then resolve via FHIR
        token = await self._ensure_token()
        pid = await self._patient_numeric_pid(token, patient_uuid)
        # NOTE: do NOT set Content-Type — httpx generates the multipart
        # boundary header automatically when `files=` is supplied. Setting
        # Content-Type manually breaks multipart parsing on the server.
        r = await self._http.post(
            f"{self._api_base}/patient/{pid}/document",
            params={"path": category},
            headers={"Authorization": f"Bearer {token}"},
            files={"document": (idem_filename, file_bytes, mime_type)},
        )
        if r.status_code >= 400:
            raise OpenEMRWriteError(
                f"document upload failed ({r.status_code}): {r.text[:300]}"
            )
        # OpenEMR's standard API returns boolean `true` on success and
        # `false`/error JSON on failure. Anything other than truthy means
        # the document service rejected the file.
        try:
            payload = r.json()
        except ValueError:
            payload = r.text
        # Validation errors come back with HTTP 200 + a populated
        # `validationErrors` field (same shape as the vitals POST). Surface
        # them with the actual error message instead of falling through to
        # the FHIR-GET-not-found path's much vaguer "couldn't locate" error.
        if isinstance(payload, dict):
            val = payload.get("validationErrors")
            if val:  # truthy dict OR non-empty list
                raise OpenEMRWriteError(
                    f"document upload validation failed: {val}"
                )
        if payload is False or payload is None or payload == "":
            raise OpenEMRWriteError(
                f"document upload returned non-truthy response: "
                f"status={r.status_code} body={r.text[:200]!r}"
            )

        # Resolve the new resource's FHIR id. Look up by SHA prefix —
        # we just verified above that no doc with this SHA existed
        # before the upload, so the only match is the one we just made.
        new_ref = await self.find_document_reference_by_sha(
            patient_uuid, sha_hex,
        )
        if not new_ref:
            raise OpenEMRWriteError(
                f"upload reported success but FHIR GET could not locate the "
                f"document (patient={patient_uuid}, sha256={sha_hex[:12]}...)"
            )
        log.info(
            "wrote DocumentReference patient=%s doc_type=%s sha256=%s -> %s",
            patient_uuid, doc_type, sha_hex[:12], new_ref,
        )
        return {"reference_id": new_ref, "sha256": sha_hex, "created": True}

    # ── Appointment (calendar / scheduling) ───────────────────────────

    async def write_appointment(
        self,
        *,
        patient_uuid: str,
        provider_user_id: int,
        event_date: str,
        start_time: str,
        duration_minutes: int = 30,
        title: str = "Office Visit",
        comments: str = "",
        category_id: int = 5,
        facility_id: int = 3,
        billing_facility_id: int = 3,
        status: str = "-",
    ) -> dict[str, Any]:
        """Create an appointment via the standard REST API.

        POSTs to ``/api/patient/{pid}/appointment`` with the OpenEMR
        ``pc_*`` field set documented in
        ``src/RestControllers/AppointmentRestController.php::post``.

        Args:
            patient_uuid: FHIR Patient UUID (resolved internally to pid).
            provider_user_id: Legacy ``users.id`` integer for the
                appointment provider (NOT the FHIR Practitioner UUID —
                the standard API expects the int row id).
            event_date: ISO date ``YYYY-MM-DD``.
            start_time: ``HH:MM`` 24-hour. Seconds optional but ignored.
            duration_minutes: Appointment length in minutes. Sent to
                OpenEMR in seconds via ``pc_duration``.
            title: Display title for the appointment chip.
            comments: Free-text note saved into ``pc_hometext``.
            category_id: ``pc_catid`` from
                ``openemr_postcalendar_categories``. Default 5 = Office
                Visit, which exists in the dev-easy demo dataset and is
                the most generic option.
            facility_id / billing_facility_id: ``pc_facility`` and
                ``pc_billing_location`` row ids from ``facility``.
                Default 3 matches the only facility in dev-easy demo.
            status: ``pc_apptstatus`` value — ``-`` is OpenEMR's code
                for "scheduled, not yet seen."

        Returns:
            ``{"appointment_id": <int eid>, "patient_pid": <int>}``
        """
        token = await self._ensure_token()
        pid = await self._patient_numeric_pid(token, patient_uuid)

        # OpenEMR's pc_duration is seconds, despite the API docs spelling
        # it as a string. Match the existing seed scripts' convention of
        # passing it as a string for safety against PHP-side str/int casts.
        duration_seconds = max(int(duration_minutes), 1) * 60

        # Normalize start_time to "HH:MM" — strip seconds if a caller
        # accidentally includes them.
        st_parts = start_time.split(":")
        if len(st_parts) >= 2:
            normalized_start = f"{int(st_parts[0]):02d}:{int(st_parts[1]):02d}"
        else:
            raise OpenEMRWriteError(
                f"start_time {start_time!r} must be HH:MM (24-hour)"
            )

        # OpenEMR validates pc_hometext as NotEmpty (despite the API docs
        # implying it's optional). Default to the appointment title when
        # the caller didn't supply comments so the write doesn't 502.
        body = {
            "pc_catid": str(category_id),
            "pc_title": title,
            "pc_duration": str(duration_seconds),
            "pc_hometext": comments.strip() or title,
            "pc_apptstatus": status,
            "pc_eventDate": event_date,
            "pc_startTime": normalized_start,
            "pc_facility": str(facility_id),
            "pc_billing_location": str(billing_facility_id),
            "pc_aid": str(provider_user_id),
        }

        r = await self._http.post(
            f"{self._api_base}/patient/{pid}/appointment",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            json=body,
        )
        if r.status_code >= 400:
            raise OpenEMRWriteError(
                f"appointment POST failed ({r.status_code}): {r.text[:300]}"
            )
        try:
            payload = r.json()
        except ValueError as exc:
            raise OpenEMRWriteError(
                f"appointment POST returned non-JSON body: {r.text[:300]}"
            ) from exc
        # OpenEMR returns validation errors via two shapes depending on
        # which layer rejected the write:
        #   1. {"validationErrors": {<field>: {<rule>: <msg>}}}
        #   2. {<field>: {<rule>: <msg>}}  (some fields validate at the
        #       top level — pc_hometext does this).
        # Normalize both before deciding the call failed.
        val = None
        if isinstance(payload, dict):
            val = payload.get("validationErrors")
            if not val and not (payload.get("data") or payload.get("id")):
                # Top-level dict with no `data`/`id` and at least one
                # field-shaped value (dict of rule -> message) — treat
                # the whole payload as a validation-error envelope.
                if all(isinstance(v, dict) for v in payload.values()) and payload:
                    val = payload
        if isinstance(val, dict) and val:
            raise OpenEMRWriteError(f"appointment validation failed: {val}")
        # Standard-API success shape: {"data": {"id": <int eid>}}.
        # `id` here is the postcalendar event id (pc_eid), not a FHIR uuid.
        data = payload.get("data") if isinstance(payload, dict) else None
        flat = data if isinstance(data, dict) else payload
        eid = None
        if isinstance(flat, dict):
            eid = flat.get("id") or flat.get("eid")
        if eid is None:
            raise OpenEMRWriteError(
                f"appointment POST succeeded but response missing id: {payload}"
            )
        log.info(
            "wrote appointment patient=%s provider=%s date=%s time=%s eid=%s",
            patient_uuid, provider_user_id, event_date, normalized_start, eid,
        )
        return {"appointment_id": int(eid), "patient_pid": pid}

    # ── Extracted-fact persistence (intake form + lab PDF -> chart) ───
    #
    # Bridge from `attach_and_extract`'s structured Pydantic JSON into
    # OpenEMR's native tables so the chat agent can read uploaded
    # information back through the existing FHIR / chart-card path.
    # Each method takes a `source_doc_id` (the FHIR
    # DocumentReference/{id} the fact was extracted from) and embeds a
    # back-reference into the row's `comments` field as a structured
    # `[copilot-source: DocumentReference/{id}]` tag, plus optional
    # bbox JSON. Two consumers later read this back:
    #   1. The agent's chart tools surface the fact in chat answers; the
    #      chat citation uses the new FHIR resource's id.
    #   2. The PDF-overlay UI (Sunday work) parses the bbox JSON to
    #      highlight the source region when the user clicks the citation.

    @staticmethod
    def _source_back_ref(source_doc_id: str | None, bbox: dict | None = None) -> str:
        """Render the `[copilot-source: ...]` tag embedded in `comments`.

        Empty string when there's no source — keeps the field clean for
        manually-entered records that didn't come from extraction.
        """
        if not source_doc_id:
            return ""
        ref = source_doc_id if "/" in source_doc_id else f"DocumentReference/{source_doc_id}"
        if bbox:
            try:
                bbox_str = json.dumps(bbox, separators=(",", ":"))
            except (TypeError, ValueError):
                bbox_str = ""
            return f"[copilot-source: {ref}; bbox={bbox_str}]" if bbox_str else f"[copilot-source: {ref}]"
        return f"[copilot-source: {ref}]"

    async def _post_standard_api(
        self, *, path: str, body: dict, label: str,
    ) -> dict:
        """POST a JSON body to the standard REST API and return the flat
        response dict (id/uuid/eid as available). Centralizes the
        validation-error handling used by every extracted-fact write so
        the per-method code stays small.
        """
        token = await self._ensure_token()
        r = await self._http.post(
            f"{self._api_base}{path}",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            json=body,
        )
        if r.status_code >= 400:
            raise OpenEMRWriteError(
                f"{label} POST failed ({r.status_code}): {r.text[:300]}"
            )
        try:
            payload = r.json()
        except ValueError as exc:
            raise OpenEMRWriteError(
                f"{label} POST returned non-JSON body: {r.text[:300]}"
            ) from exc
        # Same two validation-error envelope shapes as write_appointment.
        val = None
        if isinstance(payload, dict):
            val = payload.get("validationErrors")
            if not val and not (payload.get("data") or payload.get("id")):
                if all(isinstance(v, dict) for v in payload.values()) and payload:
                    val = payload
        if isinstance(val, dict) and val:
            raise OpenEMRWriteError(f"{label} validation failed: {val}")
        data = payload.get("data") if isinstance(payload, dict) else None
        flat = data if isinstance(data, dict) else payload
        if not isinstance(flat, dict):
            raise OpenEMRWriteError(f"{label} response not a dict: {payload}")
        return flat

    async def write_allergy(
        self,
        *,
        patient_uuid: str,
        substance: str,
        reaction: str | None = None,
        severity: str | None = None,
        source_doc_id: str | None = None,
        bbox: dict | None = None,
    ) -> dict:
        """POST /api/patient/{uuid}/allergy. Returns {id, uuid}."""
        comment_parts: list[str] = []
        if reaction:
            comment_parts.append(f"Reaction: {reaction}")
        if severity:
            comment_parts.append(f"Severity: {severity}")
        back_ref = self._source_back_ref(source_doc_id, bbox)
        if back_ref:
            comment_parts.append(back_ref)
        body = {"title": substance, "comments": " · ".join(comment_parts) or substance}
        flat = await self._post_standard_api(
            path=f"/patient/{patient_uuid}/allergy", body=body,
            label=f"allergy[{substance}]",
        )
        log.info(
            "wrote allergy patient=%s substance=%s id=%s src=%s",
            patient_uuid, substance, flat.get("id") or flat.get("uuid"), source_doc_id,
        )
        return flat

    async def write_medical_problem(
        self,
        *,
        patient_uuid: str,
        title: str,
        begdate: str | None = None,
        source_doc_id: str | None = None,
        bbox: dict | None = None,
    ) -> dict:
        """POST /api/patient/{uuid}/medical_problem. `begdate` is ISO
        YYYY-MM-DD; defaults to today when omitted."""
        body: dict[str, str] = {"title": title}
        body["begdate"] = begdate or date.today().isoformat()
        back_ref = self._source_back_ref(source_doc_id, bbox)
        if back_ref:
            body["comments"] = back_ref
        flat = await self._post_standard_api(
            path=f"/patient/{patient_uuid}/medical_problem", body=body,
            label=f"medical_problem[{title}]",
        )
        log.info(
            "wrote medical_problem patient=%s title=%s id=%s src=%s",
            patient_uuid, title, flat.get("id") or flat.get("uuid"), source_doc_id,
        )
        return flat

    async def write_medication(
        self,
        *,
        patient_uuid: str,
        title: str,
        begdate: str | None = None,
        source_doc_id: str | None = None,
        bbox: dict | None = None,
    ) -> dict:
        """POST /api/patient/{pid}/medication. `title` is the full prescription
        line (e.g. "Metformin 500 mg PO BID"); `begdate` is the start
        date in `YYYY-MM-DD HH:MM:SS` format expected by OpenEMR."""
        token = await self._ensure_token()
        pid = await self._patient_numeric_pid(token, patient_uuid)
        body: dict[str, str] = {"title": title}
        if begdate:
            # OpenEMR's medication endpoint accepts both 'YYYY-MM-DD' and
            # 'YYYY-MM-DD HH:MM:SS'. Normalize to the latter so the
            # rendered "started on" date is unambiguous in the chart.
            body["begdate"] = (
                begdate if " " in begdate else f"{begdate} 00:00:00"
            )
        else:
            body["begdate"] = f"{date.today().isoformat()} 00:00:00"
        back_ref = self._source_back_ref(source_doc_id, bbox)
        if back_ref:
            body["comments"] = back_ref
        flat = await self._post_standard_api(
            path=f"/patient/{pid}/medication", body=body,
            label=f"medication[{title}]",
        )
        log.info(
            "wrote medication patient=%s title=%s id=%s src=%s",
            patient_uuid, title, flat.get("id") or flat.get("uuid"), source_doc_id,
        )
        return flat

    async def write_lab_encounter_with_results(
        self,
        *,
        patient_uuid: str,
        results: list[dict],
        source_doc_id: str | None = None,
        encounter_date: str | None = None,
    ) -> dict:
        """Create an encounter for a lab PDF + write the results into its
        SOAP note's `objective` field.

        Lab results live in `procedure_result` in OpenEMR's data model,
        but the standard REST API doesn't expose a write endpoint for
        that table. The pragmatic surface available to us is
        encounter + SOAP note: the encounter shows up on the patient's
        timeline, and the SOAP note's text is returned by the
        `get_notes_24h` chat tool, which is how the agent will surface
        the values in answers.

        Each result is rendered as one objective-section line:
            HbA1c: 7.4 % (ref 4.0-5.6 %, abnormal flag H)

        `results` is a list of dicts with at minimum `test_name`,
        `value`, `unit`, `reference_range`, `abnormal_flag` —
        i.e. the LabResult schema flattened to a dict. Each result's
        `source_citation` contributes its `bbox` to a single per-encounter
        bbox manifest stored in the SOAP note's `subjective` section so
        the PDF overlay UI can highlight every cited region from the
        same encounter view.

        Returns: {encounter_id, soap_note_id}.
        """
        token = await self._ensure_token()
        pid = await self._patient_numeric_pid(token, patient_uuid)

        enc_date = encounter_date or date.today().isoformat()
        enc_body = {
            "pc_catid": 5,             # 'Office Visit' — only category present in dev-easy seed
            "facility_id": 3,          # 'Your Clinic Name Here' — only facility in dev-easy
            "billing_facility": 3,
            "sensitivity": "normal",
            "reason": f"Lab results from uploaded document {source_doc_id or '(unknown)'}",
            "date": f"{enc_date} 00:00:00",
            "onset_date": f"{enc_date} 00:00:00",
            "provider_id": 1,           # admin user — fallback when caller's id unknown
            "class_code": "AMB",
        }
        enc_flat = await self._post_standard_api(
            path=f"/patient/{patient_uuid}/encounter", body=enc_body,
            label="encounter[lab]",
        )
        eid = enc_flat.get("eid") or enc_flat.get("id")
        if not eid:
            raise OpenEMRWriteError(
                f"lab encounter response missing eid: {enc_flat}"
            )

        # Render each result as a fixed-shape line so a future regex /
        # parser can extract values back if needed.
        objective_lines = []
        bbox_manifest: list[dict] = []
        for r in results:
            test_name = str(r.get("test_name") or "Unknown test")
            value = r.get("value")
            unit = str(r.get("unit") or "").strip()
            ref = r.get("reference_range")
            flag = r.get("abnormal_flag")
            value_str = "" if value is None else str(value)
            line_parts = [f"{test_name}: {value_str}"]
            if unit:
                line_parts.append(unit)
            extras = []
            if ref:
                extras.append(f"ref {ref}")
            if flag and flag != "N":
                extras.append(f"flag {flag}")
            line = " ".join(line_parts)
            if extras:
                line += " (" + ", ".join(extras) + ")"
            objective_lines.append(line)

            citation = r.get("source_citation") or {}
            bbox = citation.get("bbox") if isinstance(citation, dict) else None
            if isinstance(bbox, dict):
                bbox_manifest.append({
                    "test_name": test_name,
                    "value": value_str,
                    "bbox": bbox,
                    "field_or_chunk_id": citation.get("field_or_chunk_id"),
                })

        # Subjective field carries the back-reference + bbox manifest so
        # the PDF overlay UI can grep for the manifest later. The
        # objective field is what the chat agent reads.
        subjective_parts: list[str] = []
        if source_doc_id:
            subjective_parts.append(self._source_back_ref(source_doc_id))
        if bbox_manifest:
            try:
                subjective_parts.append(
                    "[copilot-bbox-manifest: " + json.dumps(bbox_manifest, separators=(",", ":")) + "]"
                )
            except (TypeError, ValueError):
                pass
        subjective = " ".join(subjective_parts) or "Lab results extracted from uploaded document."

        soap_body = {
            "subjective": subjective,
            "objective": "\n".join(objective_lines) or "(no extracted results)",
            "assessment": f"Lab values automatically extracted from {source_doc_id or 'uploaded document'}.",
            "plan": "Review extracted values and confirm against the source document.",
        }
        soap_flat = await self._post_standard_api(
            path=f"/patient/{pid}/encounter/{eid}/soap_note", body=soap_body,
            label="soap_note[lab]",
        )
        log.info(
            "wrote lab encounter patient=%s eid=%s results=%d src=%s",
            patient_uuid, eid, len(results), source_doc_id,
        )
        return {
            "encounter_id": int(eid),
            "soap_note_id": soap_flat.get("id") or soap_flat.get("sid"),
            "result_count": len(results),
        }
