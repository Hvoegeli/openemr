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
