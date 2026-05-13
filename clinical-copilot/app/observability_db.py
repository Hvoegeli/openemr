"""SQLite-backed `TraceStore` so observability survives `systemctl restart`.

The in-memory `TraceStore` is fine for a single demo session but loses every
trace on restart — which doesn't satisfy HIPAA's audit-log retention bar
(MVP grader feedback flagged this on 2026-04-30). This module is a drop-in
replacement that writes the same `RequestTrace` objects to a SQLite file at
`data/traces.db` on every `add()`.

Same surface as the in-memory store (`add`, `list_recent`, `get`) so the
endpoints and the dashboard don't have to change. Production replacement
target is Postgres for multi-instance retention; SQLite is the right answer
for a single-process demo because it adds zero ops and survives restart.

We persist the `RequestTrace` as JSON-flattened columns + two JSONB-style
TEXT columns for the nested `tool_events` / `llm_events` lists. Reads
re-hydrate the dataclass tree; writers don't have to know the storage shape.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from dataclasses import asdict
from pathlib import Path

from app.observability import LLMEvent, RequestTrace, TokenUsage, ToolEvent

log = logging.getLogger("agent.observability.db")


class SqliteTraceStore:
    """Durable TraceStore backed by a single SQLite file.

    Thread-safe: SQLite's connection is wrapped in a lock for write-side
    serialization (we don't expect concurrent writers in this app, but a
    second uvicorn worker would otherwise race on commits).
    """

    SCHEMA = """
        CREATE TABLE IF NOT EXISTS request_traces (
            request_id TEXT PRIMARY KEY,
            session_id TEXT,
            username TEXT,
            user_msg TEXT,
            model TEXT,
            started_at REAL NOT NULL,
            finished_at REAL,
            duration_ms REAL,
            input_tokens INTEGER,
            output_tokens INTEGER,
            cache_read_tokens INTEGER,
            cache_creation_tokens INTEGER,
            cost_usd REAL,
            validator_attempts INTEGER,
            validator_failed INTEGER,
            route_count INTEGER NOT NULL DEFAULT 0,
            conversation_sources TEXT NOT NULL DEFAULT '[]',
            error TEXT,
            tool_events TEXT,
            llm_events TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_traces_started ON request_traces(started_at DESC);
        CREATE INDEX IF NOT EXISTS idx_traces_username ON request_traces(username, started_at DESC);
    """

    # Idempotent column-add migrations for DB files created before a column
    # existed. SQLite has no `ADD COLUMN IF NOT EXISTS`, so we read
    # `PRAGMA table_info` first and only `ALTER TABLE` for the gaps. Each
    # entry is `(column_name, column_definition)`; the definition must carry
    # a default so existing rows read back a sane value (old rows → 0 here).
    # This runs on every open — the equivalent of a startup migration step,
    # consistent with this project's "one file, migrate on open" stores.
    _COLUMN_MIGRATIONS: tuple[tuple[str, str], ...] = (
        ("route_count", "INTEGER NOT NULL DEFAULT 0"),
        ("conversation_sources", "TEXT NOT NULL DEFAULT '[]'"),
    )

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        # `check_same_thread=False` because uvicorn dispatches across the
        # event loop's worker pool; the lock above provides serialization.
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(self.SCHEMA)
        self._migrate_columns()
        self._conn.commit()
        log.info("trace store opened at %s", path)

    def _migrate_columns(self) -> None:
        """Add any columns missing from a pre-existing `request_traces` table."""
        existing = {
            row["name"]
            for row in self._conn.execute("PRAGMA table_info(request_traces)")
        }
        for name, definition in self._COLUMN_MIGRATIONS:
            if name in existing:
                continue
            self._conn.execute(
                f"ALTER TABLE request_traces ADD COLUMN {name} {definition}"
            )
            log.info("migrated request_traces: added column %s", name)

    def add(self, t: RequestTrace) -> None:
        usage = t.total_usage
        with self._lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO request_traces (
                    request_id, session_id, username, user_msg, model,
                    started_at, finished_at, duration_ms,
                    input_tokens, output_tokens, cache_read_tokens, cache_creation_tokens,
                    cost_usd, validator_attempts, validator_failed, route_count,
                    conversation_sources, error,
                    tool_events, llm_events
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    t.request_id, t.session_id, t.username, t.user_msg, t.model,
                    t.started_at, t.finished_at, t.duration_ms,
                    usage.input_tokens, usage.output_tokens,
                    usage.cache_read_tokens, usage.cache_creation_tokens,
                    t.cost_usd, t.validator_attempts, int(t.validator_failed),
                    t.route_count, json.dumps(list(t.conversation_sources)), t.error,
                    json.dumps([asdict(e) for e in t.tool_events]),
                    json.dumps([asdict(e) for e in t.llm_events]),
                ),
            )
            self._conn.commit()

    def list_recent(self, limit: int = 50) -> list[RequestTrace]:
        # Reads also take the lock — sqlite3's docs require serialized access
        # to a connection across threads. Without it, a read concurrent with
        # a write commit can hit "database is locked" or worse, see torn rows.
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM request_traces ORDER BY started_at DESC LIMIT ?",
                (limit,),
            )
            rows = cur.fetchall()
        return [self._row_to_trace(r) for r in rows]

    def get(self, request_id: str) -> RequestTrace | None:
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM request_traces WHERE request_id = ?",
                (request_id,),
            )
            row = cur.fetchone()
        return self._row_to_trace(row) if row else None

    @staticmethod
    def _row_to_trace(row: sqlite3.Row) -> RequestTrace:
        tool_payload = json.loads(row["tool_events"] or "[]")
        llm_payload = json.loads(row["llm_events"] or "[]")
        return RequestTrace(
            request_id=row["request_id"],
            session_id=row["session_id"] or "",
            username=row["username"] or "",
            user_msg=row["user_msg"] or "",
            started_at=row["started_at"],
            finished_at=row["finished_at"],
            duration_ms=row["duration_ms"],
            model=row["model"] or "",
            tool_events=[ToolEvent(**e) for e in tool_payload],
            llm_events=[
                LLMEvent(
                    started_at=e["started_at"],
                    duration_ms=e["duration_ms"],
                    usage=TokenUsage(**e["usage"]),
                    finish_reason=e.get("finish_reason"),
                    sources_at_call=list(e.get("sources_at_call") or []),
                )
                for e in llm_payload
            ],
            total_usage=TokenUsage(
                input_tokens=row["input_tokens"] or 0,
                output_tokens=row["output_tokens"] or 0,
                cache_read_tokens=row["cache_read_tokens"] or 0,
                cache_creation_tokens=row["cache_creation_tokens"] or 0,
            ),
            cost_usd=row["cost_usd"] or 0.0,
            validator_attempts=row["validator_attempts"] or 0,
            validator_failed=bool(row["validator_failed"]),
            route_count=row["route_count"] or 0,
            conversation_sources=json.loads(row["conversation_sources"] or "[]"),
            error=row["error"],
        )

    def close(self) -> None:
        self._conn.close()
