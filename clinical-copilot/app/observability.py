"""Per-request trace store + cost accounting for the clinical co-pilot.

The brief asks for: per-request trace, latency, tool failures, tokens, cost,
wired in from the start. This module is the structural answer.

Design:
  - One `RequestTrace` per /chat or /chat/stream call. Created in main.py at
    request entry, finalized at request exit, pushed to a bounded ring buffer.
  - The trace is set on a `ContextVar` so graph nodes (call_llm, execute_tools)
    can record events without main.py having to thread state through.
  - `TokenUsageCallback` is a LangChain `BaseCallbackHandler` that captures
    input/output tokens from `on_llm_end`. Attached to each `model.ainvoke`
    via `RunnableConfig`.
  - Cost is computed per-trace from token counts using a model-keyed price
    table. Sonnet 4.6 today: $3 / 1M input, $15 / 1M output.

The store is in-memory only — fine for the demo and for the early submission.
A production deployment would persist to Postgres alongside the audit log
(see ARCHITECTURE.md §observability).
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from collections import deque
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from app.config import settings

log = logging.getLogger("agent.observability")


# ─── trace dataclasses ───────────────────────────────────────────────────


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0

    def add(self, other: TokenUsage) -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_read_tokens += other.cache_read_tokens
        self.cache_creation_tokens += other.cache_creation_tokens


# ─── pricing ─────────────────────────────────────────────────────────────
# USD per 1M tokens. Cache reads are billed at 0.1× input; cache writes at
# 1.25× input — we report both separately so the cost-tier analysis in
# ARCHITECTURE.md can be grounded in measured numbers, not estimates.
PRICING_USD_PER_MTOK: dict[str, dict[str, float]] = {
    "claude-sonnet-4-6": {
        "input": 3.0,
        "output": 15.0,
        "cache_read": 0.30,
        "cache_write_5m": 3.75,
    },
    "claude-opus-4-7": {
        "input": 15.0,
        "output": 75.0,
        "cache_read": 1.50,
        "cache_write_5m": 18.75,
    },
    "claude-haiku-4-5": {
        "input": 1.0,
        "output": 5.0,
        "cache_read": 0.10,
        "cache_write_5m": 1.25,
    },
}


def compute_cost_usd(model: str, usage: TokenUsage) -> float:
    prices = PRICING_USD_PER_MTOK.get(model)
    if prices is None:
        return 0.0
    return (
        usage.input_tokens * prices["input"]
        + usage.output_tokens * prices["output"]
        + usage.cache_read_tokens * prices["cache_read"]
        + usage.cache_creation_tokens * prices["cache_write_5m"]
    ) / 1_000_000


@dataclass
class ToolEvent:
    name: str
    args: dict[str, Any]
    started_at: float
    duration_ms: float
    ok: bool
    sources_added: int
    error: str | None = None


@dataclass
class LLMEvent:
    started_at: float
    duration_ms: float
    usage: TokenUsage
    finish_reason: str | None = None


@dataclass
class RequestTrace:
    request_id: str
    session_id: str
    username: str
    user_msg: str
    started_at: float
    finished_at: float | None = None
    duration_ms: float | None = None
    model: str = ""
    tool_events: list[ToolEvent] = field(default_factory=list)
    llm_events: list[LLMEvent] = field(default_factory=list)
    total_usage: TokenUsage = field(default_factory=TokenUsage)
    cost_usd: float = 0.0
    validator_attempts: int = 0
    validator_failed: bool = False
    error: str | None = None

    def finalize(self) -> None:
        now = time.time()
        self.finished_at = now
        self.duration_ms = (now - self.started_at) * 1000.0
        for ev in self.llm_events:
            self.total_usage.add(ev.usage)
        if self.model:
            self.cost_usd = compute_cost_usd(self.model, self.total_usage)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ─── trace store ─────────────────────────────────────────────────────────


class TraceStore:
    """Bounded ring buffer for finalized traces, newest-first on read."""

    def __init__(self, capacity: int = 200) -> None:
        self._buf: deque[RequestTrace] = deque(maxlen=capacity)
        self._by_id: dict[str, RequestTrace] = {}

    def add(self, trace: RequestTrace) -> None:
        if len(self._buf) == self._buf.maxlen and self._buf:
            evicted = self._buf[0]
            self._by_id.pop(evicted.request_id, None)
        self._buf.append(trace)
        self._by_id[trace.request_id] = trace

    def list_recent(self, limit: int = 50) -> list[RequestTrace]:
        return list(reversed(list(self._buf)))[:limit]

    def get(self, request_id: str) -> RequestTrace | None:
        return self._by_id.get(request_id)


# ─── current-request context ─────────────────────────────────────────────

_CURRENT: ContextVar[RequestTrace | None] = ContextVar("current_trace", default=None)


def current_trace() -> RequestTrace | None:
    return _CURRENT.get()


def set_current_trace(trace: RequestTrace | None):
    return _CURRENT.set(trace)


def reset_current_trace(token) -> None:
    _CURRENT.reset(token)


# ─── LangChain callback for token capture ────────────────────────────────


class TokenUsageCallback(BaseCallbackHandler):
    """Captures per-LLM-call token usage and pushes it onto the current trace.

    `langchain-anthropic` reports usage on `LLMResult.llm_output['usage']`
    (sometimes nested under `response_metadata` on the message instead, depending
    on version). We check both.
    """

    raise_error = False  # never break a request because of telemetry

    def __init__(self) -> None:
        self._t_start: float = 0.0

    def on_llm_start(self, *args: Any, **kwargs: Any) -> None:
        self._t_start = time.time()

    def on_chat_model_start(self, *args: Any, **kwargs: Any) -> None:
        self._t_start = time.time()

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        trace = current_trace()
        if trace is None:
            return
        usage = _extract_usage(response)
        finish_reason = _extract_finish_reason(response)
        trace.llm_events.append(LLMEvent(
            started_at=self._t_start,
            duration_ms=(time.time() - self._t_start) * 1000.0,
            usage=usage,
            finish_reason=finish_reason,
        ))


def _extract_usage(response: LLMResult) -> TokenUsage:
    """Pull token counts out of a LangChain LLMResult, tolerating shape drift."""
    usage = TokenUsage()
    # Path 1: top-level llm_output (Anthropic via langchain-anthropic)
    info = (response.llm_output or {}) if isinstance(response.llm_output, dict) else {}
    raw = info.get("usage") or info.get("token_usage") or {}
    if isinstance(raw, dict):
        usage.input_tokens = int(raw.get("input_tokens") or raw.get("prompt_tokens") or 0)
        usage.output_tokens = int(raw.get("output_tokens") or raw.get("completion_tokens") or 0)
        usage.cache_read_tokens = int(raw.get("cache_read_input_tokens") or 0)
        usage.cache_creation_tokens = int(raw.get("cache_creation_input_tokens") or 0)
    # Path 2: per-generation usage_metadata on the AIMessage
    if usage.input_tokens == 0 and usage.output_tokens == 0:
        for gen_list in response.generations:
            for gen in gen_list:
                msg = getattr(gen, "message", None)
                meta = getattr(msg, "usage_metadata", None) if msg is not None else None
                if isinstance(meta, dict):
                    usage.input_tokens += int(meta.get("input_tokens") or 0)
                    usage.output_tokens += int(meta.get("output_tokens") or 0)
                    details = meta.get("input_token_details") or {}
                    if isinstance(details, dict):
                        usage.cache_read_tokens += int(details.get("cache_read") or 0)
                        usage.cache_creation_tokens += int(details.get("cache_creation") or 0)
    return usage


def _extract_finish_reason(response: LLMResult) -> str | None:
    for gen_list in response.generations:
        for gen in gen_list:
            info = getattr(gen, "generation_info", None) or {}
            if isinstance(info, dict) and info.get("finish_reason"):
                return str(info["finish_reason"])
            msg = getattr(gen, "message", None)
            meta = getattr(msg, "response_metadata", None) if msg is not None else None
            if isinstance(meta, dict):
                stop = meta.get("stop_reason") or meta.get("finish_reason")
                if stop:
                    return str(stop)
    return None


# ─── tool event helper used by the graph ─────────────────────────────────


def record_tool_event(
    name: str,
    args: dict[str, Any],
    started_at: float,
    ok: bool,
    sources_added: int,
    error: str | None = None,
) -> None:
    trace = current_trace()
    if trace is None:
        return
    trace.tool_events.append(ToolEvent(
        name=name,
        args=args,
        started_at=started_at,
        duration_ms=(time.time() - started_at) * 1000.0,
        ok=ok,
        sources_added=sources_added,
        error=error,
    ))


# ─── LangSmith env wiring ────────────────────────────────────────────────


def init_langsmith() -> bool:
    """If `langsmith_tracing` is enabled and a key is present, set the env vars
    that LangChain's tracing layer reads at module-import time. Idempotent.

    Returns True if LangSmith tracing is now active, False otherwise.
    """
    if not settings.langsmith_tracing:
        return False
    if not settings.langsmith_api_key:
        log.warning("langsmith_tracing=true but no langsmith_api_key — disabling")
        return False
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_API_KEY", settings.langsmith_api_key)
    os.environ.setdefault("LANGCHAIN_PROJECT", settings.langsmith_project)
    # Newer LangSmith builds also read LANGSMITH_*; set both.
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGSMITH_API_KEY", settings.langsmith_api_key)
    os.environ.setdefault("LANGSMITH_PROJECT", settings.langsmith_project)
    log.info("langsmith tracing enabled · project=%s", settings.langsmith_project)
    return True


# ─── factory ─────────────────────────────────────────────────────────────


def new_request_trace(session_id: str, username: str, user_msg: str, model: str) -> RequestTrace:
    return RequestTrace(
        request_id=str(uuid.uuid4()),
        session_id=session_id,
        username=username,
        user_msg=user_msg,
        started_at=time.time(),
        model=model,
    )
