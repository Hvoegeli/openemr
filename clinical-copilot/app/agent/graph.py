"""LangGraph state machine for the clinical co-pilot.

PRD W2 §4 — supervisor + 2 workers topology.

Flow:
  START -> supervisor
    supervisor -[worker_route=intake_extractor]-> intake_extractor
    supervisor -[worker_route=evidence_retriever]-> evidence_retriever
    supervisor -[worker_route=answer]-> answerer
  intake_extractor -[tool_calls present]-> tools -> intake_extractor (loop)
  intake_extractor -[no tool_calls]-> supervisor (yield)
  evidence_retriever -[tool_calls present]-> tools -> evidence_retriever (loop)
  evidence_retriever -[no tool_calls]-> supervisor (yield)
  answerer -> validate
    validate -[invalid + attempts<MAX]-> answerer (retry)
    validate -[valid OR attempts>=MAX]-> END

Loop guard: the supervisor stops routing after `MAX_SUPERVISOR_ROUTES`
hops per turn and forces `answer`. Keeps a confused supervisor from
ping-ponging between workers indefinitely.

The validator is unchanged from the single-LLM topology — it still owns
"every clinical claim cites a source returned by a tool in this
conversation." Workers keep accumulating `conversation_sources`, so by
the time the answerer runs the source set already covers everything
either worker pulled.
"""

import json
import logging
import time
from typing import Any, Literal

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from app import access_control
from app.access_control import PatientAccessDenied
from app.agent.state import AgentState
from app.agent.system_prompt import (
    ADVISOR_MODE_ADDENDUM,
    ANSWERER_ADDENDUM,
    EVIDENCE_RETRIEVER_ADDENDUM,
    INTAKE_EXTRACTOR_ADDENDUM,
    SUPERVISOR_PROMPT,
    SYSTEM_PROMPT,
)
from app.agent.tools import EVIDENCE_TOOLS, INTAKE_TOOLS, dispatch
from app.agent.input_guard import detect_jailbreak, quarantine_marker
from app.agent.validator import find_invalid_citations, find_uncited_clinical_claims
from app.config import settings
from app.fhir.client import FhirClient
from app.observability import record_tool_event

log = logging.getLogger("agent")

MAX_VALIDATION_ATTEMPTS = 2
MAX_SUPERVISOR_ROUTES = 4

VALIDATION_FAILURE_PREFIX = "VALIDATION FAILED:"


class SupervisorDecision(BaseModel):
    """Structured output for the supervisor node.

    Bound as a forced tool on the supervisor LLM so every supervisor
    invocation produces a routing decision in a typed, inspectable shape
    (PRD W2 §4 — "logged handoffs").
    """

    next_worker: Literal["intake_extractor", "evidence_retriever", "answer"] = Field(
        description=(
            "Which graph node should run next. `intake_extractor` for "
            "patient-chart and uploaded-document data; "
            "`evidence_retriever` for published-guideline retrieval; "
            "`answer` to compose the final user-facing reply."
        ),
    )
    reason: str = Field(
        description=(
            "One-sentence rationale for the routing choice. Logged in "
            "the trace store and the supervisor.route observability event."
        ),
    )


def _format_tool_result(name: str, result: dict) -> str | list[dict]:
    """Serialize a tool dispatch result into the `ToolMessage.content` shape.

    Most tools return JSON-stringifiable structured data and the LLM reads
    it as text — that's the default path. `get_document_content` is the
    exception: its `data.pages_png_b64` carries rendered page images that
    Claude can only "see" if they're delivered as `image` content blocks
    in the tool result (text descriptions of "here's a base64 PNG" are
    useless to the model). For that tool we strip `pages_png_b64` from
    the JSON metadata, keep the metadata as a leading text block, and
    append one image block per page so Claude reads the actual document.
    """
    if name == "get_document_content" and isinstance(result.get("data"), dict):
        data = dict(result["data"])
        pages_b64 = data.pop("pages_png_b64", None)
        meta_payload = {"data": data, "sources": result.get("sources", [])}
        text_block: dict = {"type": "text", "text": json.dumps(meta_payload, default=str)}
        if not isinstance(pages_b64, list) or not pages_b64:
            return [text_block]
        blocks: list[dict] = [text_block]
        for b64 in pages_b64:
            if not isinstance(b64, str) or not b64:
                continue
            blocks.append({
                "type": "image",
                "source_type": "base64",
                "mime_type": "image/png",
                "data": b64,
            })
        return blocks
    return json.dumps(result, default=str)


def _strip_image_blocks_from_messages(messages: list) -> list:
    """Return a copy of `messages` with image content blocks dropped from
    every ToolMessage. Replaces each dropped image with one short text
    placeholder (only on messages that actually carried images), so the
    receiver knows pages were elided rather than missing entirely.

    Used by the supervisor's per-hop LLM call. The supervisor only
    decides which worker to route to next; it never reads pixels. The
    images live in graph state so the worker that produced them — and
    any other worker that re-enters with the same patient context —
    still has full access. We just stop re-serializing the same
    base64-encoded PNG pages on every supervisor hop.

    Cost impact: a 4-page fax packet (~1,700 vision tokens / page on
    Sonnet) re-sent across 4-5 supervisor hops is 27K-34K input tokens
    of pure waste per conversation. Trimming the supervisor view is
    ~10-25% of total agent input cost on document-grounded chats.

    The function never mutates the input list or any message in it; it
    returns fresh ToolMessage instances so graph state is unaffected.
    """
    out: list = []
    for msg in messages:
        if not isinstance(msg, ToolMessage) or not isinstance(msg.content, list):
            out.append(msg)
            continue
        had_image = False
        kept_blocks: list = []
        for block in msg.content:
            if isinstance(block, dict) and block.get("type") == "image":
                had_image = True
                continue
            kept_blocks.append(block)
        if not had_image:
            out.append(msg)
            continue
        # Inform the supervisor that pages were elided. The text portion
        # of the original ToolMessage (the JSON metadata block from
        # `_format_tool_result` for `get_document_content`) is preserved
        # above; this marker just makes the elision explicit so a future
        # debug log doesn't read like "where did the images go?".
        kept_blocks.append({
            "type": "text",
            "text": "[document page images omitted from supervisor view; available to extractor/retriever workers]",
        })
        out.append(ToolMessage(
            content=kept_blocks,
            tool_call_id=msg.tool_call_id,
            name=msg.name,
        ))
    return out


def _content_text_for_scan(content: str | list[dict]) -> str:
    """Extract the text portion of a tool result for the jailbreak scan.

    The scanner only operates on text — image blocks can't carry prompt
    injections through this path. For string content this is the identity;
    for list content we concatenate every `text`-typed block.
    """
    if isinstance(content, str):
        return content
    return "".join(
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
        and isinstance(block.get("text"), str)
    )


def _prepend_quarantine(
    content: str | list[dict], marker: str,
) -> str | list[dict]:
    """Prepend a quarantine marker without losing image blocks. For string
    content this is plain concatenation; for list content the marker
    becomes a new leading text block."""
    if isinstance(content, str):
        return marker + content
    return [{"type": "text", "text": marker}, *content]


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


def _build_llm_base(model_name: str):
    """Construct the LangChain chat model for the active provider, without
    binding tools yet. Tool binding is per-worker; the supervisor uses
    structured output instead of free tools.

    Default is `anthropic` (direct API to api.anthropic.com); flip
    `settings.llm_provider=bedrock` for a BAA-covered AWS Bedrock path.
    The `langchain_aws` import is lazy so installs that haven't pulled
    it in keep working on the default path.
    """
    provider = (settings.llm_provider or "anthropic").lower()
    if provider == "bedrock":
        try:
            from langchain_aws import ChatBedrockConverse  # noqa: PLC0415
        except ImportError as e:
            raise RuntimeError(
                "llm_provider=bedrock requires the langchain-aws package. "
                "Install with: uv add langchain-aws"
            ) from e
        log.info(
            "llm: routing through AWS Bedrock (region=%s, model=%s)",
            settings.aws_region, settings.bedrock_model_id,
        )
        return ChatBedrockConverse(
            model=settings.bedrock_model_id,
            region_name=settings.aws_region,
        )
    log.info("llm: routing through Anthropic direct (model=%s)", model_name)
    return ChatAnthropic(model_name=model_name, timeout=60, stop=None)


def _build_worker_llm(model_name: str, tools: list):
    """Return a chat model bound to a worker-scoped tool subset.

    The supervisor decides which worker runs; this binding makes that
    decision enforceable at the model layer — the worker physically
    cannot call a tool outside its subset.
    """
    base = _build_llm_base(model_name)
    if not tools:
        return base
    return base.bind_tools(tools)


def _build_supervisor_chain(model_name: str):
    """Return a chain that emits a `SupervisorDecision` per invocation.

    Uses LangChain's `with_structured_output` so the supervisor's only
    output shape is the typed routing decision — no free text, no
    accidental tool calls, no parsing.
    """
    base = _build_llm_base(model_name)
    return base.with_structured_output(SupervisorDecision)


def active_model_label(model_name: str) -> str:
    """Return the model identifier that's actually in use, for trace
    attribution. Bedrock and direct paths use different IDs; the audit
    log should reflect what was really called, not the build-time default."""
    if (settings.llm_provider or "").lower() == "bedrock":
        return settings.bedrock_model_id
    return model_name


def build_graph(
    client: FhirClient,
    notes_store: Any,
    assignments_store: Any | None = None,
    model_name: str = "claude-sonnet-4-6",
):
    """Construct the compiled LangGraph for one or more conversation turns.

    The returned graph is reusable across turns — pass cumulative state in,
    get cumulative state back. Reset `validation_attempts` AND `route_count`
    to 0 each new user turn (the CLI/driver layer does this).

    `notes_store` is the `ClinicalNoteStore` instance the dispatch needs to
    serve `get_vital_trends`. `assignments_store` is the `AssignmentStore`
    consulted by the per-tool ACL gate; passing `None` skips the gate (CLI
    smokes / eval replays do this so they keep their prior "see all" view).
    Both typed loosely as `Any` so this module doesn't depend on the
    app-level dataclass / circular-import chain.
    """
    intake_model = _build_worker_llm(model_name, INTAKE_TOOLS)
    evidence_model = _build_worker_llm(model_name, EVIDENCE_TOOLS)
    answerer_model = _build_worker_llm(model_name, [])
    supervisor_chain = _build_supervisor_chain(model_name)

    # Cached system messages per role. The supervisor prompt is small and
    # stable so it benefits less from caching than the worker prompts —
    # but the prompt-cache TTL is generous and the cost is uniform, so
    # everything goes through `cache_control: ephemeral`.
    sys_supervisor = SystemMessage(content=[{
        "type": "text",
        "text": SUPERVISOR_PROMPT,
        "cache_control": {"type": "ephemeral"},
    }])
    sys_intake = SystemMessage(content=[{
        "type": "text",
        "text": SYSTEM_PROMPT + INTAKE_EXTRACTOR_ADDENDUM,
        "cache_control": {"type": "ephemeral"},
    }])
    sys_evidence = SystemMessage(content=[{
        "type": "text",
        "text": SYSTEM_PROMPT + EVIDENCE_RETRIEVER_ADDENDUM,
        "cache_control": {"type": "ephemeral"},
    }])
    sys_answerer_default = SystemMessage(content=[{
        "type": "text",
        "text": SYSTEM_PROMPT + ANSWERER_ADDENDUM,
        "cache_control": {"type": "ephemeral"},
    }])
    sys_answerer_advisor = SystemMessage(content=[{
        "type": "text",
        "text": SYSTEM_PROMPT + ANSWERER_ADDENDUM + ADVISOR_MODE_ADDENDUM,
        "cache_control": {"type": "ephemeral"},
    }])

    # Anthropic's messages API rejects a conversation that ends in an
    # assistant turn ("This model does not support assistant message
    # prefill. The conversation must end with a user message."). On the
    # first hop the message tail IS the user's HumanMessage and we're
    # fine, but on subsequent hops a worker has just yielded with an
    # AIMessage (or a ToolMessage that LangChain renders as user-role,
    # which is fine — but a non-tool AIMessage trips the 400). Every
    # node that invokes the LLM with `state["messages"]` runs this pad
    # so the message tail is always a HumanMessage. The pad is local to
    # the LLM call and never persisted to graph state, so workers
    # downstream still see the un-padded history.
    def _pad_for_anthropic(messages: list, *, prompt: str) -> list:
        if not messages or not isinstance(messages[-1], HumanMessage):
            return [*messages, HumanMessage(content=prompt)]
        return list(messages)

    async def supervisor_node(state: AgentState) -> dict:
        rc = state.get("route_count", 0) + 1
        if rc > MAX_SUPERVISOR_ROUTES:
            log.warning(
                "supervisor: route_count=%d > %d, forcing answer",
                rc, MAX_SUPERVISOR_ROUTES,
            )
            record_tool_event(
                name="supervisor.route",
                args={"next_worker": "answer", "reason": "loop_guard"},
                started_at=time.time(),
                ok=True,
                sources_added=0,
            )
            return {"worker_route": "answer", "route_count": rc}

        t0 = time.time()
        # Trim image blocks before the supervisor sees the message stream.
        # The supervisor's only job is routing; it never reads pixels.
        # Workers (intake_extractor, evidence_retriever) still see the
        # un-trimmed history because they invoke the LLM via `state["messages"]`
        # directly. See `_strip_image_blocks_from_messages` for cost math.
        history = _pad_for_anthropic(
            _strip_image_blocks_from_messages(list(state["messages"])),
            prompt=(
                "Choose the next worker (intake_extractor, "
                "evidence_retriever, or answer) and call the routing "
                "tool with your decision."
            ),
        )
        decision = await supervisor_chain.ainvoke([sys_supervisor, *history])
        # `with_structured_output` returns a Pydantic instance.
        next_worker = decision.next_worker
        reason = decision.reason
        log.info(
            "supervisor: route=%s reason=%r (hop %d/%d)",
            next_worker, reason, rc, MAX_SUPERVISOR_ROUTES,
        )
        record_tool_event(
            name="supervisor.route",
            args={"next_worker": next_worker, "reason": reason},
            started_at=t0,
            ok=True,
            sources_added=0,
        )
        return {"worker_route": next_worker, "route_count": rc}

    async def intake_extractor(state: AgentState) -> dict:
        history = _pad_for_anthropic(
            list(state["messages"]),
            prompt="Continue the intake-extraction work. Call a tool or finish.",
        )
        messages = [sys_intake, *history]
        response = await intake_model.ainvoke(messages)
        return {"messages": [response]}

    async def evidence_retriever(state: AgentState) -> dict:
        history = _pad_for_anthropic(
            list(state["messages"]),
            prompt="Continue the evidence-retrieval work. Call a tool or finish.",
        )
        messages = [sys_evidence, *history]
        response = await evidence_model.ainvoke(messages)
        return {"messages": [response]}

    async def answerer(state: AgentState) -> dict:
        sys_msg = sys_answerer_advisor if state.get("advisor_mode") else sys_answerer_default
        history = _pad_for_anthropic(
            list(state["messages"]),
            prompt=(
                "Compose the final answer for the doctor using the chart "
                "facts gathered so far. Cite each clinical claim with an "
                "inline `[ResourceType/ID]` from the tool results."
            ),
        )
        messages = [sys_msg, *history]
        response = await answerer_model.ainvoke(messages)
        return {"messages": [response]}

    async def execute_tools(state: AgentState) -> dict:
        last = state["messages"][-1]
        if not isinstance(last, AIMessage) or not last.tool_calls:
            return {}

        new_messages: list = []
        new_sources = list(state["conversation_sources"])
        new_patient_id = state.get("patient_id")

        # Resolve the caller's patient panel once per node invocation. The
        # access_control module caches per-username for 5 min, so subsequent
        # tool calls in the same conversation hit the cache.
        username = state.get("username")
        panel = await access_control.get_panel_for_user(client, username, assignments_store)

        for call in last.tool_calls:
            t0 = time.time()
            sources_added = 0
            tool_ok = True
            tool_err: str | None = None
            try:
                result = await dispatch(call["name"], call["args"], client, notes_store, panel=panel)
                sources_added = len(result["sources"])
                new_sources.extend(result["sources"])
                if call["name"] == "resolve_patient" and isinstance(result["data"], dict) \
                        and result["data"].get("found"):
                    new_patient_id = result["data"].get("patient_id")
                content = _format_tool_result(call["name"], result)
                log.info(
                    "tool=%s args=%s sources=%d",
                    call["name"], call["args"], sources_added,
                )
            except PatientAccessDenied as e:
                # Privacy-conservative: surface the same shape `resolve_patient`
                # uses for a no-match. The LLM cannot tell denial from typo,
                # which is the desired property — confirming a patient exists
                # would leak panel boundaries. Audit signal lives on the
                # trace event via `tool_err`.
                content = json.dumps({
                    "data": {"found": False, "patient_id": e.patient_id},
                    "sources": [],
                })
                tool_ok = False
                tool_err = f"patient_access_denied:{e.patient_id}"
                log.warning(
                    "tool=%s denied user=%s patient=%s",
                    call["name"], username, e.patient_id,
                )
            except Exception as e:  # noqa: BLE001 — surface tool errors to the LLM
                content = json.dumps({"error": str(e), "tool": call["name"]})
                tool_ok = False
                tool_err = f"{type(e).__name__}: {e}"
                log.exception("tool=%s args=%s failed", call["name"], call["args"])

            # Layer 3 (defense-in-depth) — tool-output sanitizer. If the
            # serialized tool result contains text resembling a jailbreak
            # phrasing (e.g. an attacker planted "ignore previous
            # instructions" in a chart record), prepend a quarantine
            # header so the LLM sees the suspicious content as data, not
            # a directive. The data itself flows through unchanged so we
            # don't accidentally hide a legitimate clinical detail. For
            # multimodal results (image blocks from `get_document_content`)
            # we only scan the text portion — image bytes can't carry text
            # injections through this code path.
            jb_label = detect_jailbreak(_content_text_for_scan(content))
            if jb_label is not None:
                content = _prepend_quarantine(content, quarantine_marker(jb_label))
                log.warning(
                    "tool-output quarantine: tool=%s pattern=%s",
                    call["name"], jb_label,
                )
                if tool_err is None:
                    tool_err = f"quarantined:{jb_label}"

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

    def route_after_supervisor(
        state: AgentState,
    ) -> Literal["intake_extractor", "evidence_retriever", "answerer"]:
        route = state.get("worker_route") or "answer"
        if route == "intake_extractor":
            return "intake_extractor"
        if route == "evidence_retriever":
            return "evidence_retriever"
        return "answerer"

    def route_after_worker(state: AgentState) -> Literal["tools", "supervisor"]:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return "supervisor"

    def route_after_tools(
        state: AgentState,
    ) -> Literal["intake_extractor", "evidence_retriever", "answerer"]:
        # Re-enter whichever worker the supervisor picked. `worker_route`
        # is only set by the supervisor, so it survives an `execute_tools`
        # invocation and tells us where to loop back.
        route = state.get("worker_route")
        if route == "intake_extractor":
            return "intake_extractor"
        if route == "evidence_retriever":
            return "evidence_retriever"
        # Should not happen — answer never has tool calls. Safety fallback
        # to answerer so the graph terminates instead of looping.
        log.warning("route_after_tools: unexpected worker_route=%r; falling through to answerer", route)
        return "answerer"

    def route_after_validate(state: AgentState) -> Literal["answerer", "__end__"]:
        last = state["messages"][-1]
        if (
            isinstance(last, HumanMessage)
            and isinstance(last.content, str)
            and last.content.startswith(VALIDATION_FAILURE_PREFIX)
        ):
            return "answerer"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("intake_extractor", intake_extractor)
    graph.add_node("evidence_retriever", evidence_retriever)
    graph.add_node("answerer", answerer)
    graph.add_node("tools", execute_tools)
    graph.add_node("validate", validate_citations)

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {
            "intake_extractor": "intake_extractor",
            "evidence_retriever": "evidence_retriever",
            "answerer": "answerer",
        },
    )
    graph.add_conditional_edges(
        "intake_extractor",
        route_after_worker,
        {"tools": "tools", "supervisor": "supervisor"},
    )
    graph.add_conditional_edges(
        "evidence_retriever",
        route_after_worker,
        {"tools": "tools", "supervisor": "supervisor"},
    )
    graph.add_conditional_edges(
        "tools",
        route_after_tools,
        {
            "intake_extractor": "intake_extractor",
            "evidence_retriever": "evidence_retriever",
            "answerer": "answerer",
        },
    )
    graph.add_edge("answerer", "validate")
    graph.add_conditional_edges(
        "validate",
        route_after_validate,
        {"answerer": "answerer", END: END},
    )

    return graph.compile()
