"""Tool schemas for the clinical co-pilot.

Each `@tool` below exists only to expose a JSON schema and docstring to the
LLM via `model.bind_tools(...)`. The actual dispatch happens in
`execute_tools_node` so we can capture the `sources` list into agent state
for the citation validator.
"""

from langchain_core.tools import tool

from app.fhir import adapter
from app.fhir.adapter import SourcedResult
from app.fhir.client import FhirClient


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


TOOLS = [resolve_patient, get_patient_card]


async def dispatch(name: str, args: dict, client: FhirClient) -> SourcedResult:
    """Run the actual adapter call for a tool name + args."""
    if name == "resolve_patient":
        return await adapter.resolve_patient(client, query=args["query"])
    if name == "get_patient_card":
        return await adapter.get_patient_card(client, patient_id=args["patient_id"])
    raise ValueError(f"Unknown tool: {name}")
