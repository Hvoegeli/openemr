"""Agent state for the LangGraph state machine.

`conversation_sources` is the trust anchor for the citation validator: every
FHIR source ID returned by a tool call gets appended here. The validator
later rejects any LLM response that cites an ID not in this list.
"""

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    conversation_sources: list[str]
    patient_id: str | None
    validation_attempts: int
    # `username` is the authenticated session owner (None if no auth context,
    # e.g. CLI smokes). Tool dispatch resolves it into a `panel` via
    # `app.access_control.get_panel_for_user` and gates patient-id-taking
    # tools against that panel.
    username: str | None
    # Per-turn toggle for medication-safety advisory mode. When True, the
    # graph swaps in a system prompt that relaxes R2 (allows drug-interaction
    # / dose-rule / contraindication reasoning grounded in the current
    # patient's chart) and mandates a "verify before acting" disclaimer at
    # the end of any response containing safety reasoning. Default False
    # preserves the chart-summarizer-only behavior the eval suite asserts.
    advisor_mode: bool
