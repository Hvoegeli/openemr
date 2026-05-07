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

from typing import Any

from langchain_core.tools import tool

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
async def get_notes_24h(patient_id: str, hours: int = 0) -> str:
    """Fetch DocumentReferences (clinical notes, progress notes, discharge
    summaries, lab reports, intake forms, imaging reports) for a patient.

    DEFAULT BEHAVIOUR (`hours=0`): returns ALL documents on file for the
    patient regardless of age. This is what you want for most queries —
    "show me her labs", "what intake forms do we have on him", "any
    imaging?". The tool's name retains "24h" for backwards compatibility
    with eval snapshots; behaviour is "all documents" unless hours is
    explicitly set to a positive integer.

    Set `hours` to a positive integer only when you specifically need a
    short window (e.g. "what did the night team document in the last
    shift?"), in which case the result is limited to docs whose date is
    within `hours` of now.

    Returns metadata only — title, type, date, status, attachment titles.
    Pair with `get_document_content(document_id)` to read the doc's actual
    pages.

    Returns JSON with `data.documents`, `data.window_hours`
    (0 means unbounded), `data.cutoff_iso` (null when unbounded), plus
    `sources` listing every DocumentReference ID.

    Args:
        patient_id: The FHIR Patient resource ID.
        hours: Lookback window in hours, or 0 for all documents. Defaults to 0.
    """
    raise NotImplementedError("Dispatched in agent.graph.execute_tools_node")


@tool
async def get_document_content(document_id: str) -> str:
    """Fetch the actual contents (rendered pages) of a `DocumentReference`
    so you can read what the document says — not just its metadata.

    Use this AFTER `get_notes_24h` (or any tool that surfaces a document
    id) when the doctor asks "what does the doc say?", "summarize the
    intake form", "any new info in that PDF?". `get_notes_24h` only
    returns metadata; this tool returns the rendered pages so you can
    actually read them.

    Returns JSON metadata (title, type, date, page_count, attachments) AND
    the rendered page images themselves as part of the tool result, so you
    can see the document directly. Cite findings as
    `[DocumentReference/{document_id}]`.

    Args:
        document_id: The FHIR DocumentReference id (without the
            `DocumentReference/` prefix).
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
async def retrieve_guidelines(query: str, k: int = 3) -> str:
    """Search the published clinical-guideline corpus (USPSTF + ADA Standards
    of Care) for snippets relevant to `query`. Use this when the doctor asks
    a "what does the guideline say" question, when proposing a screening
    decision (statin, aspirin, lipid panel, BP, T2DM, CRC), or when grounding
    a recommendation in published evidence.

    Returns JSON with `data.hits` (each with `chunk_id`, `source` like
    'USPSTF' or 'ADA', `title`, `year`, `url`, `text`, `score`, `rank`)
    plus `sources` listing every retrieved chunk as
    `Guideline/<chunk_id>` so the citation validator accepts inline
    `[Guideline/<chunk_id>]` references in the response.

    Citation format the agent must use when quoting a returned chunk:
    `[Guideline/<chunk_id>]` immediately after the quoted/paraphrased
    statement. Empty result is the truthful answer when nothing relevant
    is in the corpus — never make up a guideline.

    Args:
        query: Free-text query in the doctor's words. Concatenated with
            chunk title + topic_tags during BM25 scoring, so domain terms
            (e.g. "statin primary prevention", "HbA1c target") work well.
        k: Maximum hits to return. Defaults to 3; raise to 5 for broader
            survey questions.
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
    get_document_content,
    get_med_changes_24h,
    clinical_flags,
    retrieve_guidelines,
]

# Worker-scoped tool subsets per PRD W2 §4 (supervisor + 2 workers). Each
# worker LLM is bound only to its subset so the supervisor's routing
# decision is enforced at the model layer — not just by prompt asks. The
# answerer node has no tools at all.
#
# `INTAKE_TOOLS`: chart + uploaded-document reading. Owns time anchoring,
# patient resolution, structured chart pulls, recent-diff lookups, and the
# clinical_flags rule surface. Anything that touches OpenEMR's FHIR layer
# or the extracted-document pipeline lives here.
INTAKE_TOOLS = [
    current_time,
    resolve_patient,
    get_patient_card,
    get_vital_trends,
    get_observations_24h,
    get_notes_24h,
    get_document_content,
    get_med_changes_24h,
    clinical_flags,
]
# `EVIDENCE_TOOLS`: published-guideline retrieval. Stays narrow on purpose
# — adding chart tools here would re-merge the worker boundary the
# supervisor exists to make explicit.
EVIDENCE_TOOLS = [
    retrieve_guidelines,
]


async def _current_time_impl() -> SourcedResult:
    # `clinical_now()` is the centralized clinical-tz helper — same
    # function used by the schedule endpoint, calendar fetcher, and
    # eval invariants. astimezone() with no arg returns server-local
    # time (Hetzner is UTC), so the agent would otherwise narrate
    # "21:35" when the doctor's wall clock said 15:35 MDT.
    from app.timeutil import clinical_now  # noqa: PLC0415
    now = clinical_now()
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

    # `collect_vital_trends` preserves FHIR's raw UTC `effectiveDateTime`
    # in each point's `date` field. The web UI hides this with a
    # `toLocaleString` on render, but the agent sees the raw string and
    # narrates it verbatim — so a vital recorded at 15:35 clinical-local
    # gets summarized as "21:35", confusing the doctor. Re-stamp every
    # point's date through `_clinical_iso` (the same helper the chart
    # adapter uses on `effectiveDateTime` everywhere else) before the
    # tool result reaches the LLM. The UI path is untouched.
    from app.fhir.adapter import _clinical_iso  # noqa: PLC0415
    for points in trends.values():
        for p in points:
            d = p.get("date")
            if isinstance(d, str):
                p["date"] = _clinical_iso(d)

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


async def _retrieve_guidelines_impl(query: str, k: int = 3) -> SourcedResult:
    """Run BM25 retrieval over the guideline corpus + project sources.

    Sources are emitted as `Guideline/<chunk_id>` so the existing citation
    validator (`CITATION_RE = \\[([A-Z][a-zA-Z]+)/([a-zA-Z0-9._-]+)\\]`)
    accepts inline `[Guideline/<chunk_id>]` references in the response.
    No FHIR client is needed — the corpus is pre-loaded at module import."""
    from app.guidelines.retrieve import retrieve_guidelines

    hits = retrieve_guidelines(query, k=k)
    data_hits = [
        {
            "chunk_id":   h.chunk.chunk_id,
            "source":     h.chunk.source,
            "title":      h.chunk.title,
            "year":       h.chunk.year,
            "url":        h.chunk.url,
            "text":       h.chunk.text,
            "topic_tags": h.chunk.topic_tags,
            "score":      h.score,
            "rank":       h.rank,
        }
        for h in hits
    ]
    sources = [f"Guideline/{h.chunk.chunk_id}" for h in hits]
    return {
        "data": {"query": query, "hits": data_hits, "hit_count": len(hits)},
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
    if name == "get_document_content":
        return await adapter.get_document_content(
            client,
            document_id=args["document_id"],
            panel=panel,
        )
    if name == "get_med_changes_24h":
        return await adapter.get_med_changes_24h(
            client,
            patient_id=args["patient_id"],
            hours=int(args.get("hours", 24)),
        )
    if name == "clinical_flags":
        return await _clinical_flags_impl(client, patient_id=args["patient_id"])
    if name == "retrieve_guidelines":
        return await _retrieve_guidelines_impl(
            query=args["query"], k=int(args.get("k", 3)),
        )
    raise ValueError(f"Unknown tool: {name}")
