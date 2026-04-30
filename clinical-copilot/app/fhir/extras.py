"""Secondary FHIR fetchers used by the redesigned UI.

These are NOT exposed as agent tools — they back the left-side tab panel
(Supporting Documents, Calendar). They still return the standard
`{data, sources}` shape so the same audit-log / citation-tracking pipeline
applies to them in the future.
"""

import asyncio
from datetime import date
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


async def get_calendar_today(client: FhirClient) -> SourcedResult:
    """Return the inpatient roster for the dashboard.

    Demo behavior: every seeded patient is treated as currently admitted —
    they carry over day to day, regardless of encounter date. Production
    deployment would filter on encounter status (in-progress / arrived) or
    a separate "active inpatient" flag, but the demo has no discharge
    workflow so we surface every patient in the system.

    Each row carries the latest encounter's start time + reason for the
    chip's secondary line. Endpoint name kept as `today` for backwards
    compatibility with the front-end.
    """
    today_iso = date.today().isoformat()
    sources: list[str] = []

    patients = await _safe_search(client, "Patient", {"_count": 50})
    if not patients:
        return {"data": {"date": today_iso, "patients": []}, "sources": sources}

    for p in patients:
        sources.append(_ref(p))

    # Latest encounter per patient — gives "reason" + "time" without forcing
    # encounters to be re-dated daily. One round-trip in wall-clock time.
    latest_encs = await asyncio.gather(*[
        _safe_search(
            client, "Encounter",
            {"patient": p["id"], "_count": 1, "_sort": "-date"},
        )
        for p in patients
    ])

    entries: list[CalendarEntry] = []
    for p, enc_list in zip(patients, latest_encs):
        enc = enc_list[0] if enc_list else None
        if enc:
            sources.append(_ref(enc))
        entries.append({
            "appointment_id": None,
            "patient_id": p["id"],
            "name": _format_name(p),
            "age": _calc_age(p.get("birthDate")),
            "sex": p.get("gender"),
            "time": (enc.get("period") or {}).get("start") if enc else None,
            "reason": _coded_display((enc.get("type") or [{}])[0]) if enc else None,
            "seeded": False,
        })

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
    """
    encounters, docs = await asyncio.gather(
        _safe_search(client, "Encounter", {"patient": patient_id, "_count": 30}),
        _safe_search(client, "DocumentReference", {"patient": patient_id, "_count": 30}),
    )

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
                "url": att.get("url"),
            })
        items.append({
            "kind": "document",
            "id": d["id"],
            "ref": _ref(d),
            "title": _coded_display(d.get("type") or {}) or "Document",
            "date": d.get("date"),
            "status": d.get("status"),
            "category": _coded_display((d.get("category") or [{}])[0]),
            "attachments": attachments,
        })

    items.sort(key=lambda x: x.get("date") or "", reverse=True)

    return {"data": {"items": items}, "sources": sources}


# ─── helpers ─────────────────────────────────────────────────────────────


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
