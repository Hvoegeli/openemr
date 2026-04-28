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

from datetime import datetime, timezone

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


TOOLS = [current_time, resolve_patient, get_patient_card]


async def _current_time_impl() -> SourcedResult:
    now = datetime.now(timezone.utc).astimezone()
    return {
        "data": {
            "iso_datetime": now.isoformat(timespec="seconds"),
            "date": now.date().isoformat(),
            "weekday": now.strftime("%A"),
            "timezone": str(now.tzinfo),
        },
        "sources": [],
    }


async def dispatch(name: str, args: dict, client: FhirClient) -> SourcedResult:
    """Run the actual adapter call for a tool name + args."""
    if name == "current_time":
        return await _current_time_impl()
    if name == "resolve_patient":
        return await adapter.resolve_patient(client, query=args["query"])
    if name == "get_patient_card":
        return await adapter.get_patient_card(client, patient_id=args["patient_id"])
    raise ValueError(f"Unknown tool: {name}")
