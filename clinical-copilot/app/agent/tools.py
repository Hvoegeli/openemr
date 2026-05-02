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


@tool
async def clinical_flags(patient_id: str) -> str:
    """Surface chart-internal fact pairs the doctor would want to see together.

    Returns flags for well-known pairings drawn from data already in the
    chart — e.g. "metformin prescribed AND eGFR < 30", "warfarin AND INR > 4",
    "Bactrim AND documented sulfa allergy". Each flag is FACTUAL, not a
    recommendation: the agent surfaces the pair with citations and the
    doctor decides what to do. The agent itself does not advise on dose
    changes, drug substitutions, or actions.

    Call after `get_patient_card` whenever the doctor's question is about
    medication safety, contraindications, or "anything I should notice"
    on a patient. Empty result list does NOT mean "the chart is safe" —
    the rule set is intentionally narrow (5 rules covering common
    high-frequency pairings); it means none of those specific rules fired.

    Returns JSON with `data.flags` (each with `rule_id`, `summary` carrying
    inline `[ResourceType/ID]` citations, and `evidence` list) plus
    `sources` listing every cited resource ID. The `summary` is the
    user-facing text — quote it verbatim and the citation validator
    accepts the references.

    Args:
        patient_id: The FHIR Patient resource ID.
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
    clinical_flags,
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


async def _clinical_flags_impl(
    client: FhirClient, patient_id: str,
) -> SourcedResult:
    """Fetch the chart slice the rule engine needs, then evaluate it.

    Pulls active meds + allergies + active problems via `get_patient_card`,
    then a separate Observation search for the lab category over a 30-day
    lookback (so a stale-but-still-meaningful eGFR or K+ trip doesn't get
    silently missed because today's lab panel was incomplete). Lab rows go
    through `_format_vital` so the rule helpers see the same shape they
    use for vitals (id/name/value/unit/time)."""
    import asyncio
    from datetime import datetime, timedelta, timezone

    from app.agent.clinical_flags import evaluate_chart
    from app.fhir.adapter import _format_vital

    cutoff_iso = (datetime.now(timezone.utc) - timedelta(days=30)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    card_result, lab_obs = await asyncio.gather(
        adapter.get_patient_card(client, patient_id=patient_id),
        client.search(
            "Observation",
            {
                "patient": patient_id,
                "category": "laboratory",
                "date": f"ge{cutoff_iso}",
                "_count": 100,
            },
        ),
    )
    card_data = card_result["data"]
    labs: list[dict] = []
    for o in lab_obs:
        for row in _format_vital(o):
            labs.append(row)

    flags = evaluate_chart(
        active_meds=card_data.get("active_medications") or [],
        recent_labs=labs,
        active_problems=card_data.get("active_problems") or [],
        allergies=card_data.get("allergies") or [],
    )

    sources: list[str] = []
    seen: set[str] = set()
    for f in flags:
        for ref in f.evidence:
            if ref not in seen:
                seen.add(ref)
                sources.append(ref)

    return {
        "data": {
            "patient_id": patient_id,
            "rule_count": len(flags),
            "flags": [f.to_dict() for f in flags],
        },
        "sources": sources,
    }


# Tools that take a `patient_id` directly. For these, dispatch verifies the
# patient sits inside the caller's panel before running the FHIR fetch — a
# user who knows another patient's UUID still gets a "not found" answer rather
# than chart contents from outside their assignment.
_PATIENT_ID_TOOLS = frozenset({
    "get_patient_card",
    "get_vital_trends",
    "get_observations_24h",
    "get_notes_24h",
    "get_med_changes_24h",
    "clinical_flags",
})


async def dispatch(
    name: str,
    args: dict,
    client: FhirClient,
    notes_store: Any,
    panel: frozenset[str] | None = None,
) -> SourcedResult:
    """Run the actual adapter call for a tool name + args.

    `panel` is the per-user patient-ID allow-list resolved by
    [app.access_control](../access_control.py). `None` means no filter
    (admin); a non-`None` set restricts which patients the agent can fetch
    or resolve. Tools without a `patient_id` arg (`current_time`) ignore
    `panel`. `resolve_patient` filters its name-search results client-side
    via the existing `doctor_panel_ids` adapter parameter.

    Raises `PatientAccessDenied` for patient-ID-taking tools whose
    `patient_id` is outside `panel` — the caller (`execute_tools_node` in
    `graph.py`) translates that into a "not found"-shaped tool message so
    the LLM treats it as a typo rather than an error to surface. The
    `tool_err` recorded on the trace event preserves the audit signal.
    """
    from app.access_control import PatientAccessDenied

    if (
        name in _PATIENT_ID_TOOLS
        and panel is not None
        and (target := args.get("patient_id"))
        and target not in panel
    ):
        raise PatientAccessDenied(target)

    if name == "current_time":
        return await _current_time_impl()
    if name == "resolve_patient":
        return await adapter.resolve_patient(
            client,
            query=args["query"],
            doctor_panel_ids=list(panel) if panel is not None else None,
        )
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
    if name == "clinical_flags":
        return await _clinical_flags_impl(client, patient_id=args["patient_id"])
    raise ValueError(f"Unknown tool: {name}")
