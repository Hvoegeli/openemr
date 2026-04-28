"""High-level FHIR fetchers for the clinical co-pilot.

Each function returns the data the agent needs PLUS a `sources` list of FHIR
resource references (`Patient/123`, `Observation/8821`, ...). The verification
node downstream rejects any agent claim not backed by a source ID from a
tool call in the same conversation.
"""

import re
from typing import TypedDict

from app.fhir.client import FhirClient

_HTML_TAG_RE = re.compile(r"<[^>]+>")


class SourcedResult(TypedDict):
    data: dict | list
    sources: list[str]


def _ref(resource: dict) -> str:
    """Format a FHIR resource as a citation reference, e.g. 'Patient/123'."""
    return f"{resource['resourceType']}/{resource['id']}"


def _narrative_text(resource: dict) -> str | None:
    """Extract human-readable text from a FHIR resource's narrative `text.div`.

    OpenEMR puts free-text titles for allergies/problems/meds into the
    narrative when no SNOMED/RxNorm code is supplied. Strip the HTML wrapper
    so the agent sees a plain string.
    """
    div = (resource.get("text") or {}).get("div")
    if not isinstance(div, str):
        return None
    stripped = _HTML_TAG_RE.sub("", div).strip()
    return stripped or None


def _coded_display(coded: dict) -> str | None:
    """Best-effort label for a FHIR CodeableConcept: text, then first display.

    Skips FHIR `data-absent-reason` placeholder codings (system ends in
    `data-absent-reason`, code `unknown`/`asked-unknown`/`temp-unknown`/etc.) —
    OpenEMR emits those when no SNOMED/RxNorm code is supplied, and the real
    label is in the narrative `text.div` instead.
    """
    if not isinstance(coded, dict):
        return None
    text = coded.get("text")
    if isinstance(text, str) and text.strip():
        return text.strip()
    for coding in coded.get("coding") or []:
        if not isinstance(coding, dict):
            continue
        system = coding.get("system") or ""
        if "data-absent-reason" in system:
            continue
        display = coding.get("display")
        if isinstance(display, str) and display.strip():
            return display.strip()
        code_val = coding.get("code")
        if isinstance(code_val, str) and code_val.strip():
            return code_val.strip()
    return None


async def resolve_patient(
    client: FhirClient,
    *,
    query: str,
    doctor_panel_ids: list[str] | None = None,
) -> SourcedResult:
    """Resolve a free-text patient reference (bed number, last name, MRN) to a Patient.

    Returns the best match with an `alternatives` list when ambiguous so the
    agent can ask the doctor to disambiguate rather than guessing.
    """
    # Try MRN first (most precise), then last name, then bed number lookup
    # via the related Encounter.location. For now this is a simple name search;
    # bed and MRN paths are TODO once we have demo data shaped.
    matches = await client.search("Patient", {"family": query, "_count": 5})

    if doctor_panel_ids:
        matches = [p for p in matches if p["id"] in doctor_panel_ids]

    if not matches:
        return {"data": {"found": False, "query": query}, "sources": []}

    best = matches[0]
    return {
        "data": {
            "found": True,
            "patient_id": best["id"],
            "name": _format_name(best),
            "alternatives": [
                {"patient_id": p["id"], "name": _format_name(p)} for p in matches[1:]
            ],
        },
        "sources": [_ref(p) for p in matches],
    }


async def get_patient_card(client: FhirClient, *, patient_id: str) -> SourcedResult:
    """Fetch the structured data that powers the right-side patient card.

    All six FHIR queries fan out in parallel via `asyncio.gather` — sequential
    awaits added ~3s on a local stack and ~5s through cloudflared. They are
    independent (no result feeds the next), so parallelism is safe.
    """
    import asyncio

    # OpenEMR's FHIR layer is selective about which search params it honors.
    # Using broad searches and filtering client-side is safer than passing
    # `status` / `clinical-status` filters that silently return zero.
    patient, encounters_all, allergies, problems_all, meds, vitals = await asyncio.gather(
        client.get(f"Patient/{patient_id}"),
        client.search("Encounter", {"patient": patient_id, "_sort": "-date", "_count": 5}),
        client.search("AllergyIntolerance", {"patient": patient_id}),
        client.search("Condition", {"patient": patient_id}),
        client.search("MedicationRequest", {"patient": patient_id, "_count": 10}),
        client.search(
            "Observation",
            {"patient": patient_id, "category": "vital-signs", "_count": 10},
        ),
    )

    encounters = [
        e for e in encounters_all
        if e.get("status") in {"in-progress", "arrived", "triaged", "finished"}
    ][:1]
    problems = [
        c for c in problems_all
        if (
            (c.get("clinicalStatus") or {}).get("coding", [{}])[0].get("code") in {"active", "recurrence", "relapse", None}
        )
    ]

    sources = (
        [_ref(patient)]
        + [_ref(e) for e in encounters]
        + [_ref(a) for a in allergies]
        + [_ref(p) for p in problems]
        + [_ref(m) for m in meds]
        + [_ref(v) for v in vitals]
    )

    return {
        "data": {
            "name": _format_name(patient),
            "mrn": patient_id,
            "age": _calc_age(patient.get("birthDate")),
            "sex": patient.get("gender"),
            "current_encounter": encounters[0] if encounters else None,
            "allergies": [_format_allergy(a) for a in allergies],
            "active_problems": [_format_condition(c) for c in problems],
            "active_medications": [_format_med(m) for m in meds],
            "recent_vitals": [_format_vital(v) for v in vitals],
        },
        "sources": sources,
    }


# ─── formatters ──────────────────────────────────────────────────────────


def _format_name(patient: dict) -> str:
    name = (patient.get("name") or [{}])[0]
    given = " ".join(name.get("given", []))
    family = name.get("family", "")
    return f"{given} {family}".strip() or "(unknown)"


def _calc_age(birth_date: str | None) -> int | None:
    if not birth_date:
        return None
    from datetime import date

    born = date.fromisoformat(birth_date)
    today = date.today()
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))


def _format_allergy(a: dict) -> dict:
    return {
        "id": a["id"],
        "substance": _coded_display(a.get("code", {})) or _narrative_text(a),
        "reaction": [
            r.get("manifestation", [{}])[0].get("text") for r in a.get("reaction", [])
        ],
        "severity": a.get("criticality"),
    }


def _format_condition(c: dict) -> dict:
    return {
        "id": c["id"],
        "name": _coded_display(c.get("code", {})) or _narrative_text(c),
        "onset": c.get("onsetDateTime"),
    }


def _format_med(m: dict) -> dict:
    dosage = (m.get("dosageInstruction") or [{}])[0]
    return {
        "id": m["id"],
        "drug": _coded_display(m.get("medicationCodeableConcept", {})) or _narrative_text(m),
        "dose_text": dosage.get("text"),
        "started": m.get("authoredOn"),
    }


def _format_vital(v: dict) -> dict:
    return {
        "id": v["id"],
        "name": _coded_display(v.get("code", {})) or _narrative_text(v),
        "value": v.get("valueQuantity", {}).get("value"),
        "unit": v.get("valueQuantity", {}).get("unit"),
        "time": v.get("effectiveDateTime"),
    }
