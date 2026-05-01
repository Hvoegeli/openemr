"""Tool schemas for the clinical co-pilot.

Each `@tool` below exists only to expose a JSON schema and docstring to the
LLM via `model.bind_tools(...)`. The actual dispatch happens in
`execute_tools_node` so we can capture the `sources` list into agent state
for the citation validator.

Design rule: anything the agent needs to *know* (current time, drug
interactions, clinical rules) must come from a tool, not the LLM's training.
Tools always return `{data, sources}`. `sources` may be empty for non-FHIR
tools like `current_time`, but the LLM is still forbidden from synthesizing
facts that don't appear in some tool's `data`.
"""

from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from langchain_core.tools import tool

from app.config import settings
from app.fhir import adapter
from app.fhir.adapter import SourcedResult
from app.fhir.client import FhirClient


@tool
async def current_time() -> str:
    """Return the current date and time. Call this before any relative-date
    statement (e.g. "started 6 months ago", "labs from yesterday", "since
    last visit"). Without this tool the agent has no reliable anchor for
    "now" and must not produce relative-time language.

    Returns JSON with `data.iso_datetime`, `data.date`, `data.weekday`,
    `data.timezone`, plus an empty `sources` list (no FHIR provenance).
    """
    raise NotImplementedError("Dispatched in agent.graph.execute_tools_node")


@tool
async def resolve_patient(query: str) -> str:
    """Find a patient by last name (case-insensitive).

    Returns JSON with `data.patient_id`, `data.name`, and `data.alternatives`
    when the match is ambiguous, plus `sources` listing the candidate
    `Patient/...` references. Call this first whenever the doctor refers to
    a patient by name; never assume an ID.

    Args:
        query: The doctor's reference to the patient (last name for now).
    """
    raise NotImplementedError("Dispatched in agent.graph.execute_tools_node")


@tool
async def get_patient_card(patient_id: str) -> str:
    """Fetch the patient's structured chart summary.

    Returns JSON with `data` containing demographics, current encounter,
    allergies, active problems, active medications, recent vitals, and
    `sources` listing every FHIR resource cited in `data`. Use after
    `resolve_patient` to load the chart for the resolved patient.

    Args:
        patient_id: The FHIR Patient resource ID.
    """
    raise NotImplementedError("Dispatched in agent.graph.execute_tools_node")


@tool
async def get_vital_trends(patient_id: str) -> str:
    """Fetch pre-grouped vital-sign trends across FHIR + clinical notes.

    Returns JSON with `data` containing:
      - `current`: most recent reading per vital (heart_rate, bp_systolic,
        bp_diastolic, respiratory_rate, temp_f, spo2), each with date,
        value, unit, and source FHIR ref.
      - `trends`: ascending-time list per vital, each point with date,
        value, unit, source.
    Use this when the doctor asks about trends ("is HR climbing?", "what's
    the trajectory of BP?") or about a single most-recent vital. Cheaper
    and more accurate than reasoning over raw Observation lists from
    `get_patient_card`. Citations come back in `sources`.

    Args:
        patient_id: The FHIR Patient resource ID.
    """
    raise NotImplementedError("Dispatched in agent.graph.execute_tools_node")


@tool
async def get_observations_24h(patient_id: str, hours: int = 24) -> str:
    """Fetch Observations (labs + vitals) recorded in the last `hours`.

    Use for "what changed overnight?", "any new labs since yesterday?",
    "did her potassium come back?". Time-bound and narrower than
    `get_patient_card`'s `recent_vitals`, which spans all observations
    regardless of when they were taken.

    Returns JSON with `data.observations` (each row has id, name, value,
    unit, time, category — `vital-signs` / `laboratory` / etc.),
    `data.window_hours`, `data.cutoff_iso`, plus `sources` listing every
    Observation resource ID. Composite BP is split into separate Systolic
    and Diastolic rows. Default window is 24 hours; pass `hours=48` to
    widen, `hours=12` to narrow.

    Args:
        patient_id: The FHIR Patient resource ID.
        hours: Lookback window in hours. Defaults to 24.
    """
    raise NotImplementedError("Dispatched in agent.graph.execute_tools_node")


@tool
async def get_notes_24h(patient_id: str, hours: int = 24) -> str:
    """Fetch DocumentReferences (clinical notes, progress notes, discharge
    summaries) created in the last `hours`.

    Use for "what did the night team document?", "any new notes since
    rounds?". Returns metadata only — title, type, date, status, attachment
    titles. Content-body fetch is not supported by this tool yet; describe
    that a note exists and let the doctor open it.

    Returns JSON with `data.documents`, `data.window_hours`,
    `data.cutoff_iso`, plus `sources` listing every DocumentReference ID.

    Args:
        patient_id: The FHIR Patient resource ID.
        hours: Lookback window in hours. Defaults to 24.
    """
    raise NotImplementedError("Dispatched in agent.graph.execute_tools_node")


@tool
async def get_med_changes_24h(patient_id: str, hours: int = 24) -> str:
    """Fetch MedicationRequests authored in the last `hours` — new orders,
    dose changes, holds.

    Use for "what meds changed overnight?", "did the covering doctor start
    anything new?". Status transitions that didn't go through a new
    MedicationRequest (pure backend status flips) won't appear here — that's
    an OpenEMR limitation, not a tool gap.

    Returns JSON with `data.medications` (each with id, drug, dose_text,
    status, intent, authored_on), `data.window_hours`, `data.cutoff_iso`,
    plus `sources` listing every MedicationRequest ID.

    Args:
        patient_id: The FHIR Patient resource ID.
        hours: Lookback window in hours. Defaults to 24.
    """
    raise NotImplementedError("Dispatched in agent.graph.execute_tools_node")


TOOLS = [
    current_time,
    resolve_patient,
    get_patient_card,
    get_vital_trends,
    get_observations_24h,
    get_notes_24h,
    get_med_changes_24h,
]


async def _current_time_impl() -> SourcedResult:
    # Use the clinical TZ explicitly — astimezone() with no arg returns the
    # system's local time, but Hetzner runs UTC so the agent would otherwise
    # narrate "21:35" when the doctor's wall clock said 15:35 MDT.
    now = datetime.now(ZoneInfo(settings.clinical_tz))
    return {
        "data": {
            "iso_datetime": now.isoformat(timespec="seconds"),
            "date": now.date().isoformat(),
            "weekday": now.strftime("%A"),
            "timezone": str(now.tzinfo),
        },
        "sources": [],
    }


async def _get_vital_trends_impl(
    client: FhirClient, notes_store: Any, patient_id: str,
) -> SourcedResult:
    """Mirror of `app.main._vital_trends_compute` for the agent path.

    Pulls FHIR vital-signs observations and clinical-note vitals through the
    same `collect_vital_trends` helper the UI uses, then projects sources
    so the citation validator accepts every Observation/ClinicalNote ref
    the agent quotes.
    """
    # Imported lazily — `app.vitals` and `app.clinical_notes` would
    # otherwise pull in app-level state at module import time.
    from app.clinical_notes import now_utc
    from app.vitals import collect_vital_trends, latest_per_vital

    observations = await client.search(
        "Observation",
        {"patient": patient_id, "category": "vital-signs", "_count": 50},
    )
    notes = [
        n.to_doc_item()
        for n in notes_store.list_for_patient(patient_id, now=now_utc())
    ]
    trends = collect_vital_trends(observations, notes)

    sources: list[str] = []
    seen: set[str] = set()
    for points in trends.values():
        for p in points:
            ref = p.get("source")
            if ref and ref not in seen:
                seen.add(ref)
                sources.append(ref)

    return {
        "data": {
            "current": latest_per_vital(trends),
            "trends": trends,
        },
        "sources": sources,
    }


async def dispatch(
    name: str, args: dict, client: FhirClient, notes_store: Any,
) -> SourcedResult:
    """Run the actual adapter call for a tool name + args."""
    if name == "current_time":
        return await _current_time_impl()
    if name == "resolve_patient":
        return await adapter.resolve_patient(client, query=args["query"])
    if name == "get_patient_card":
        return await adapter.get_patient_card(client, patient_id=args["patient_id"])
    if name == "get_vital_trends":
        return await _get_vital_trends_impl(
            client, notes_store, patient_id=args["patient_id"],
        )
    if name == "get_observations_24h":
        return await adapter.get_observations_24h(
            client,
            patient_id=args["patient_id"],
            hours=int(args.get("hours", 24)),
        )
    if name == "get_notes_24h":
        return await adapter.get_notes_24h(
            client,
            patient_id=args["patient_id"],
            hours=int(args.get("hours", 24)),
        )
    if name == "get_med_changes_24h":
        return await adapter.get_med_changes_24h(
            client,
            patient_id=args["patient_id"],
            hours=int(args.get("hours", 24)),
        )
    raise ValueError(f"Unknown tool: {name}")
