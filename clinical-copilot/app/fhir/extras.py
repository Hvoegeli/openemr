"""Secondary FHIR fetchers used by the redesigned UI.

These are NOT exposed as agent tools — they back the left-side tab panel
(Supporting Documents, Calendar). They still return the standard
`{data, sources}` shape so the same audit-log / citation-tracking pipeline
applies to them in the future.
"""

import asyncio
from typing import TypedDict

from app.fhir.adapter import (
    SourcedResult,
    _calc_age,
    _coded_display,
    _format_name,
    _narrative_text,
    _ref,
)
from app.fhir.client import FhirClient


class CalendarEntry(TypedDict, total=False):
    appointment_id: str | None
    patient_id: str
    name: str
    age: int | None
    sex: str | None
    time: str | None  # ISO datetime if from Appointment, else None
    reason: str | None
    seeded: bool  # True for fallback stubs


async def get_calendar_today(
    client: FhirClient,
    *,
    panel: frozenset[str] | None = None,
    active_patients_store=None,
) -> SourcedResult:
    """Return the inpatient roster for the dashboard.

    Demo behavior: every seeded patient is treated as currently admitted —
    they carry over day to day, regardless of encounter date. Production
    deployment would filter on encounter status (in-progress / arrived) or
    a separate "active inpatient" flag, but the demo has no discharge
    workflow so we surface every patient in the system.

    `panel` is the per-user patient-ID allow-list resolved by
    [app.access_control](../access_control.py). `None` means no filter
    (admin); `frozenset()` means an empty panel (logged-in user with no
    assigned patients). The filter is applied client-side — OpenEMR's FHIR
    layer accepts `?general-practitioner=Practitioner/{id}` on Patient
    search but we already have the broad list cached, so reusing it is
    cheaper than a second FHIR roundtrip per user.

    Each row carries the latest encounter's start time + reason for the
    chip's secondary line. Endpoint name kept as `today` for backwards
    compatibility with the front-end.
    """
    # `date.today()` returns server-local. Hetzner runs UTC; the doctor
    # using Copilot is in clinical-local TZ. At ~5pm clinical it's
    # already midnight on the server, so server-local "today" can be
    # one day ahead of the user's "today". `clinical_today()` reads
    # `settings.clinical_tz` so the deployment can flip TZ without
    # touching this site.
    from app.timeutil import clinical_today  # lazy: keeps module-load fast
    today_iso = clinical_today().isoformat()
    sources: list[str] = []

    # Appointment overlay temporarily disabled pending the time-zone
    # rework (see project memory "Time-zone permanent fix"). The
    # FHIR-Appointment-merge code path is preserved verbatim below the
    # `if APPOINTMENT_OVERLAY_ENABLED:` guard so flipping the flag back
    # to True re-enables the scheduling UX without re-deriving the
    # logic. The default-False branch is the original Week-1/Week-2
    # encounter-based row, byte-identical to the pre-Calendar-Option-A
    # behavior.
    APPOINTMENT_OVERLAY_ENABLED = False

    if APPOINTMENT_OVERLAY_ENABLED:
        patients, todays_appts = await asyncio.gather(
            _safe_search(client, "Patient", {"_count": 50}),
            _safe_search(client, "Appointment", {"date": today_iso, "_count": 200}),
        )
    else:
        patients = await _safe_search(client, "Patient", {"_count": 50})
        todays_appts = []

    # Demo-cut allowlist (static + runtime overrides). Admin sees the
    # cut too — see access_control.ACTIVE_PATIENT_NAMES + ActivePatientsStore.
    from app import access_control
    patients = access_control.filter_active(
        patients, dynamic_store=active_patients_store,
    )
    if panel is not None:
        patients = [p for p in patients if p.get("id") in panel]
    if not patients:
        return {"data": {"date": today_iso, "patients": []}, "sources": sources}

    for p in patients:
        sources.append(_ref(p))

    # Index today's appointments by Patient/{id}. Multiple appointments per
    # patient → keep the earliest one (the "next" slot they'll be seen in).
    appts_by_patient: dict[str, dict] = {}
    for a in todays_appts:
        ref = _participant_patient_ref(a)
        if not ref:
            continue
        existing = appts_by_patient.get(ref)
        if existing is None or (a.get("start") or "") < (existing.get("start") or ""):
            appts_by_patient[ref] = a
    if todays_appts:
        for a in todays_appts:
            sources.append(_ref(a))

    # Latest encounter per patient — gives "reason" + "time" fallback
    # when no scheduled appointment exists for today.
    #
    # Pre-fix: N+1 — one `Encounter?patient=X&_count=1&_sort=-date` per
    # patient. With OpenEMR's PHP session locking serializing FHIR reads
    # server-side, a 21-patient roster paid ~21× ~250ms ≈ 5s on the
    # admin view despite asyncio.gather giving us client-side parallelism.
    #
    # Post-fix: ONE search for the most-recent N encounters across the
    # whole panel, then bucket by patient id client-side. First-seen
    # within the patient bucket wins because the bundle is sorted
    # `-date` descending. `_count` is sized so each patient on a
    # reasonably busy roster (≤ 30 inpatients × 10 recent encounters
    # each) is covered without truncating any patient's latest.
    all_encs = await _safe_search(
        client, "Encounter",
        {"_sort": "-date", "_count": 300},
    )
    latest_by_patient: dict[str, dict] = {}
    for enc in all_encs:
        ref = (enc.get("subject") or {}).get("reference") or ""
        if not ref.startswith("Patient/"):
            continue
        pid = ref.removeprefix("Patient/")
        # First occurrence wins because `-date` sort puts the latest first.
        if pid not in latest_by_patient:
            latest_by_patient[pid] = enc

    entries: list[CalendarEntry] = []
    for p in patients:
        enc = latest_by_patient.get(p["id"])
        if enc:
            sources.append(_ref(enc))
        # Prefer today's scheduled Appointment over the latest encounter
        # for `time` + `reason`. The encounter still wins as a fallback so
        # the row is never blank for inpatients without today's slot.
        appt = appts_by_patient.get(f"Patient/{p['id']}")
        if appt:
            time_iso = appt.get("start")
            reason = (
                _coded_display((appt.get("serviceType") or [{}])[0])
                or _coded_display(appt.get("appointmentType") or {})
                or appt.get("description")
            )
            appt_id = appt.get("id")
        else:
            time_iso = (enc.get("period") or {}).get("start") if enc else None
            reason = (
                _coded_display((enc.get("type") or [{}])[0]) if enc else None
            )
            appt_id = None
        entries.append({
            "appointment_id": appt_id,
            "patient_id": p["id"],
            "name": _format_name(p),
            "age": _calc_age(p.get("birthDate")),
            "sex": p.get("gender"),
            "time": time_iso,
            "reason": reason,
            "seeded": False,
        })

    # Sort by time ascending when the appointment overlay is enabled so
    # the day self-organizes by slot. With the overlay off, preserve
    # the natural Patient-search order (matches pre-Option-A behavior).
    if APPOINTMENT_OVERLAY_ENABLED:
        entries.sort(key=lambda e: (e.get("time") is None, e.get("time") or ""))

    return {"data": {"date": today_iso, "patients": entries}, "sources": sources}


def _patient_refs_from_appts(
    appts: list[dict],
) -> tuple[list[str], dict[str, dict]]:
    """Pull the patient ref + display metadata out of each appointment."""
    seen: list[str] = []
    meta: dict[str, dict] = {}
    for a in appts:
        ref = _participant_patient_ref(a)
        if not ref or ref in meta:
            continue
        seen.append(ref)
        meta[ref] = {
            "appointment_id": a.get("id"),
            "time": a.get("start"),
            "reason": _coded_display((a.get("serviceType") or [{}])[0])
                or _coded_display(a.get("appointmentType") or {}),
        }
    return seen, meta


def _patient_refs_from_encs(
    encs: list[dict],
) -> tuple[list[str], dict[str, dict]]:
    """Same shape, sourced from Encounter.subject."""
    seen: list[str] = []
    meta: dict[str, dict] = {}
    for e in encs:
        ref = (e.get("subject") or {}).get("reference") or ""
        if not ref.startswith("Patient/") or ref in meta:
            continue
        seen.append(ref)
        meta[ref] = {
            "appointment_id": None,
            "time": (e.get("period") or {}).get("start"),
            "reason": _coded_display((e.get("type") or [{}])[0]),
        }
    return seen, meta


async def get_supporting_documents(
    client: FhirClient, *, patient_id: str
) -> SourcedResult:
    """List past visits + clinical documents for a patient.

    Combines `Encounter` history (most recent first) with
    `DocumentReference` so the doctor can pull up prior context. Returns
    a flat, time-sorted list — caller groups in the UI.

    Pairs with `format_supporting_documents_data` — that function is the
    formatter half (no I/O), used by `app.fhir.prewarm` to build the
    docs slice for every panel patient from a single roster-wide
    batched fetch.
    """
    encounters, docs = await asyncio.gather(
        _safe_search(client, "Encounter", {"patient": patient_id, "_count": 30}),
        _safe_search(client, "DocumentReference", {"patient": patient_id, "_count": 30}),
    )
    return format_supporting_documents_data(encounters=encounters, docs=docs)


def format_supporting_documents_data(
    *,
    encounters: list[dict],
    docs: list[dict],
) -> SourcedResult:
    """Pure formatter for the supporting-docs payload — no I/O.

    Factored out of `get_supporting_documents` so the prewarm path can
    pass pre-bucketed roster-wide encounter / document lists.
    """
    items: list[dict] = []
    sources: list[str] = []

    for e in encounters:
        sources.append(_ref(e))
        items.append({
            "kind": "encounter",
            "id": e["id"],
            "ref": _ref(e),
            "title": _coded_display((e.get("type") or [{}])[0]) or "Visit",
            "date": (e.get("period") or {}).get("start") or e.get("date"),
            "status": e.get("status"),
            "summary": _coded_display(
                (e.get("reasonCode") or [{}])[0]
            ) or _narrative_text(e),
        })

    for d in docs:
        sources.append(_ref(d))
        attachments = []
        for content in d.get("content") or []:
            att = content.get("attachment") or {}
            attachments.append({
                "title": att.get("title"),
                "content_type": att.get("contentType"),
                "url": _proxy_binary_url(att.get("url")),
            })
        # OpenEMR sometimes encodes the document type as a Coding whose
        # `display` is literally the string "unknown" (the seeded
        # "Lab Pdf"/"Intake Form" docs have proper displays; older
        # OpenEMR-native uploads tend to be the unknown ones). When the
        # title would otherwise collapse to a generic "Document", reach
        # into the first attachment's filename so the UI surfaces
        # something distinguishable (e.g. "Whitaker ADT-A08.hl7" instead
        # of three rows that all read "Document Patient Information").
        type_display = _coded_display(d.get("type") or {})
        if not type_display or type_display.strip().lower() == "unknown":
            type_display = _friendly_label_from_attachments(attachments) or "Document"
        items.append({
            "kind": "document",
            "id": d["id"],
            "ref": _ref(d),
            "title": type_display,
            "date": d.get("date"),
            "status": d.get("status"),
            "category": _coded_display((d.get("category") or [{}])[0]),
            "attachments": attachments,
        })

    items.sort(key=lambda x: x.get("date") or "", reverse=True)

    return {"data": {"items": items}, "sources": sources}


# ─── helpers ─────────────────────────────────────────────────────────────


def _friendly_label_from_attachments(attachments: list[dict]) -> str | None:
    """Best-effort document label when FHIR `type.text` is missing/unknown.

    Strategy: take the first attachment's filename, strip the SHA-256
    prefix the writer adds (`sha256-<hex>__`), and return what's left.
    The original filename usually carries enough signal for a clinician
    to recognize the document at a glance ("whitaker_adt_a08.hl7",
    "patel_referral.docx", "kowalski_fax_packet.tiff"). Returns None
    when no attachment carries a usable filename — caller falls back to
    the generic "Document" label.
    """
    for a in attachments or []:
        title = (a or {}).get("title") or ""
        title = title.strip()
        if not title:
            continue
        # `_idempotency_filename` prepends `sha256-<hex>__` to the
        # original name; strip it for display.
        if title.startswith("sha256-"):
            sep = title.find("__")
            if sep >= 0:
                title = title[sep + 2:]
        if title:
            return title
    return None


async def _safe_search(client: FhirClient, resource: str, params: dict) -> list[dict]:
    try:
        return await client.search(resource, params)
    except Exception:  # noqa: BLE001
        return []


async def _safe_get(client: FhirClient, path: str) -> dict | None:
    try:
        return await client.get(path)
    except Exception:  # noqa: BLE001
        return None


def _participant_patient_ref(appt: dict) -> str | None:
    for p in appt.get("participant") or []:
        ref = (p.get("actor") or {}).get("reference") or ""
        if ref.startswith("Patient/"):
            return ref
    return None


def _proxy_binary_url(url: str | None) -> str | None:
    """Rewrite an OpenEMR `Binary/{id}` URL to our `/api/binary/{id}` proxy.

    OpenEMR's attachment URL points at the container's localhost host,
    which the browser can't reach and which requires an OAuth bearer.
    Pass-through if the URL doesn't contain `/Binary/` so non-binary
    attachments (e.g. external links) keep their original href.
    """
    if not url or "/Binary/" not in url:
        return url
    binary_id = url.rsplit("/Binary/", 1)[-1].split("?", 1)[0].split("#", 1)[0]
    return f"/api/binary/{binary_id}"
