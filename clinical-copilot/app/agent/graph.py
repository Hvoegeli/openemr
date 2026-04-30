"""LangGraph state machine for the clinical co-pilot.

Flow:
  START -> call_llm
    call_llm -[tool_calls present]-> execute_tools -> call_llm  (tool loop)
    call_llm -[no tool_calls]-> validate_citations
  validate_citations -[invalid + attempts<MAX]-> call_llm  (retry)
  validate_citations -[valid OR attempts>=MAX]-> END

The tool loop and validator together give the architecture's "tool output is
the source of truth" guarantee: the LLM can't reach FHIR directly, every
tool result's `sources` get appended to state.conversation_sources, and the
validator rejects the final response if it cites anything outside that set.
"""

import json
import logging
import time
from typing import Any, Literal

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph

from app.agent.state import AgentState
from app.agent.system_prompt import SYSTEM_PROMPT
from app.agent.tools import TOOLS, dispatch
from app.agent.validator import find_invalid_citations, find_uncited_clinical_claims
from app.fhir.client import FhirClient
from app.observability import record_tool_event

log = logging.getLogger("agent")

MAX_VALIDATION_ATTEMPTS = 2

VALIDATION_FAILURE_PREFIX = "VALIDATION FAILED:"


def message_text(message: BaseMessage) -> str:
    """Extract plain text from an LLM message regardless of content shape.

    `langchain_anthropic` returns `content` as a `str` for plain-text replies
    but as a list of content-block dicts (or pydantic-like objects with a
    `.text` attribute) when the response mixes text with tool-use or
    extended-thinking blocks. The validator and the API response need a flat
    string; without this helper we'd silently bypass citation checks for any
    list-shaped response.
    """
    content: Any = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if block.get("type") == "text" and isinstance(block.get("text"), str):
                    parts.append(block["text"])
            else:
                text_attr = getattr(block, "text", None)
                if isinstance(text_attr, str):
                    parts.append(text_attr)
        return "".join(parts)
    return ""


def build_graph(client: FhirClient, model_name: str = "claude-sonnet-4-6"):
    """Construct the compiled LangGraph for one or more conversation turns.

    The returned graph is reusable across turns — pass cumulative state in,
    get cumulative state back. Reset `validation_attempts` to 0 each new
    user turn (the CLI/driver layer does this).
    """
    model = ChatAnthropic(model_name=model_name, timeout=60, stop=None).bind_tools(TOOLS)

    async def call_llm(state: AgentState) -> dict:
        messages = [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
        response = await model.ainvoke(messages)
        return {"messages": [response]}

    async def execute_tools(state: AgentState) -> dict:
        last = state["messages"][-1]
        if not isinstance(last, AIMessage) or not last.tool_calls:
            return {}

        new_messages: list = []
        new_sources = list(state["conversation_sources"])
        new_patient_id = state.get("patient_id")

        for call in last.tool_calls:
            t0 = time.time()
            sources_added = 0
            tool_ok = True
            tool_err: str | None = None
            try:
                result = await dispatch(call["name"], call["args"], client)
                sources_added = len(result["sources"])
                new_sources.extend(result["sources"])
                if call["name"] == "resolve_patient" and isinstance(result["data"], dict) \
                        and result["data"].get("found"):
                    new_patient_id = result["data"].get("patient_id")
                content = json.dumps(result, default=str)
                log.info(
                    "tool=%s args=%s sources=%d",
                    call["name"], call["args"], sources_added,
                )
            except Exception as e:  # noqa: BLE001 — surface tool errors to the LLM
                content = json.dumps({"error": str(e), "tool": call["name"]})
                tool_ok = False
                tool_err = f"{type(e).__name__}: {e}"
                log.exception("tool=%s args=%s failed", call["name"], call["args"])
            new_messages.append(ToolMessage(content=content, tool_call_id=call["id"]))
            record_tool_event(
                name=call["name"],
                args=call["args"],
                started_at=t0,
                ok=tool_ok,
                sources_added=sources_added,
                error=tool_err,
            )

        return {
            "messages": new_messages,
            "conversation_sources": new_sources,
            "patient_id": new_patient_id,
        }

    async def validate_citations(state: AgentState) -> dict:
        last = state["messages"][-1]
        text = message_text(last)
        invalid = find_invalid_citations(text, state["conversation_sources"])
        uncited = find_uncited_clinical_claims(text)
        attempts = state.get("validation_attempts", 0)

        if not invalid and not uncited:
            log.info("validator: ok (attempts=%d)", attempts)
            return {}

        attempts += 1
        log.warning(
            "validator: invalid=%s uncited=%d attempts=%d",
            invalid, len(uncited), attempts,
        )
        if attempts >= MAX_VALIDATION_ATTEMPTS:
            # Attach a system-visible note rather than blocking the response
            # outright; the user will see the assistant's text with a warning
            # in the driver.
            return {"validation_attempts": attempts}

        # Fake-cite errors take precedence in the retry message — they're
        # the more dangerous failure mode (the LLM made up an ID).
        if invalid:
            retry_content = (
                f"{VALIDATION_FAILURE_PREFIX} these citations are not in any tool "
                f"result returned in this conversation: {', '.join(invalid)}. "
                f"Either restate your response using only citations that appear in "
                f"tool outputs, or say 'insufficient evidence in chart' for the "
                f"affected claims."
            )
        else:
            sample = "; ".join(s[:120] for s in uncited[:3])
            retry_content = (
                f"{VALIDATION_FAILURE_PREFIX} the following clinical claims are "
                f"missing inline `[ResourceType/ID]` citations: {sample}. "
                f"R1 requires every clinical fact to end with an inline citation "
                f"to a resource ID returned by a tool in this conversation. "
                f"Restate the response with proper citations on each clinical "
                f"claim, or say 'insufficient evidence in chart' for any claim "
                f"you cannot back with a tool result."
            )
        return {"messages": [HumanMessage(content=retry_content)], "validation_attempts": attempts}

    def route_after_llm(state: AgentState) -> Literal["tools", "validate"]:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return "validate"

    def route_after_validate(state: AgentState) -> Literal["llm", "__end__"]:
        last = state["messages"][-1]
        if (
            isinstance(last, HumanMessage)
            and isinstance(last.content, str)
            and last.content.startswith(VALIDATION_FAILURE_PREFIX)
        ):
            return "llm"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("llm", call_llm)
    graph.add_node("tools", execute_tools)
    graph.add_node("validate", validate_citations)

    graph.add_edge(START, "llm")
    graph.add_conditional_edges("llm", route_after_llm, {"tools": "tools", "validate": "validate"})
    graph.add_edge("tools", "llm")
    graph.add_conditional_edges("validate", route_after_validate, {"llm": "llm", END: END})

    return graph.compile()
