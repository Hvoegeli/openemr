"""SQLite-backed session + auth-event store.

Replaces the cookie-only "username in request.session" auth scheme with a
durable server-side session table so we get:

  - server-side revocation (admin can log a doctor out)
  - idle / absolute timeouts (cookie ttl alone can't enforce idle)
  - durable audit log of login_success / login_failure / logout /
    session_revoked / session_expired events

Storage is the same SQLite file as the trace store (`data/traces.db`). Two
new tables: `sessions` (active session rows, keyed by an opaque sid produced
by `secrets.token_urlsafe(32)`) and `auth_events` (immutable audit log).
The cookie now stores only the sid, signed by Starlette's SessionMiddleware
— the username + lifecycle live here, not in the cookie.
"""

from __future__ import annotations

import logging
import secrets
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("agent.auth.db")

# Server-side timeouts. Idle: 30 min without an authenticated request kicks
# the session back to /login (cookies can't enforce idle). Absolute: 12h
# caps any single session so a stolen cookie isn't valid forever. Both are
# checked at lookup time in `get_active_session`.
IDLE_TIMEOUT_S = 30 * 60
ABSOLUTE_TIMEOUT_S = 12 * 60 * 60


@dataclass
class Session:
    sid: str
    username: str
    created_at: float
    last_seen: float
    expires_at: float
    revoked_at: float | None
    user_agent: str | None
    ip: str | None


@dataclass
class AuthEvent:
    id: int
    event_type: str  # login_success | login_failure | logout | session_revoked | session_expired
    username: str | None
    sid: str | None
    at: float
    user_agent: str | None
    ip: str | None
    detail: str | None


class AuthStore:
    """Durable session + auth-event store backed by SQLite.

    Thread-safe via a single lock around the connection — same pattern as
    `SqliteTraceStore`. Sessions are an O(1) primary-key lookup so the
    extra DB hit on every authenticated request is well under a millisecond.
    """

    SCHEMA = """
        CREATE TABLE IF NOT EXISTS sessions (
            sid TEXT PRIMARY KEY,
            username TEXT NOT NULL,
            created_at REAL NOT NULL,
            last_seen REAL NOT NULL,
            expires_at REAL NOT NULL,
            revoked_at REAL,
            user_agent TEXT,
            ip TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(username, last_seen DESC);
        CREATE INDEX IF NOT EXISTS idx_sessions_active ON sessions(revoked_at);

        CREATE TABLE IF NOT EXISTS auth_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT NOT NULL,
            username TEXT,
            sid TEXT,
            at REAL NOT NULL,
            user_agent TEXT,
            ip TEXT,
            detail TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_auth_events_at ON auth_events(at DESC);
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        # `check_same_thread=False` because uvicorn dispatches across the
        # event loop's worker pool; the lock above provides serialization.
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(self.SCHEMA)
        self._conn.commit()
        log.info("auth store opened at %s", path)

    def create_session(
        self,
        username: str,
        user_agent: str | None = None,
        ip: str | None = None,
    ) -> str:
        """Create a fresh session row and return the sid for the cookie."""
        sid = secrets.token_urlsafe(32)
        now = time.time()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO sessions (sid, username, created_at, last_seen, expires_at, user_agent, ip)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (sid, username, now, now, now + ABSOLUTE_TIMEOUT_S, user_agent, ip),
            )
            self._conn.commit()
        return sid

    def get_active_session(self, sid: str) -> Session | None:
        """Return the active session for `sid` (refreshing last_seen) or None.

        None if the row doesn't exist, was revoked, or has timed out (idle
        or absolute). On timeout the row is marked revoked_at = now and a
        `session_expired` audit event is written, so the next lookup
        returns None silently and the audit panel reflects the lifecycle.
        """
        now = time.time()
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM sessions WHERE sid = ?", (sid,),
            )
            row = cur.fetchone()
            if row is None:
                return None
            if row["revoked_at"] is not None:
                return None
            absolute_expired = now > row["expires_at"]
            idle_expired = now - row["last_seen"] > IDLE_TIMEOUT_S
            if absolute_expired or idle_expired:
                self._conn.execute(
                    "UPDATE sessions SET revoked_at = ? WHERE sid = ?",
                    (now, sid),
                )
                self._conn.execute(
                    """
                    INSERT INTO auth_events (event_type, username, sid, at, user_agent, ip, detail)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "session_expired",
                        row["username"],
                        sid,
                        now,
                        row["user_agent"],
                        row["ip"],
                        "absolute" if absolute_expired else "idle",
                    ),
                )
                self._conn.commit()
                return None
            self._conn.execute(
                "UPDATE sessions SET last_seen = ? WHERE sid = ?", (now, sid),
            )
            self._conn.commit()
            return Session(
                sid=row["sid"],
                username=row["username"],
                created_at=row["created_at"],
                last_seen=now,
                expires_at=row["expires_at"],
                revoked_at=None,
                user_agent=row["user_agent"],
                ip=row["ip"],
            )

    def revoke_session(self, sid: str) -> str | None:
        """Mark an active session revoked. Returns username if a row was
        updated, None if the sid didn't exist or was already revoked.

        Pure state mutation — does NOT write an auth_event itself, since
        the right event type depends on context (logout vs admin-revoke).
        Callers are responsible for `log_event` after a successful revoke.
        """
        now = time.time()
        with self._lock:
            cur = self._conn.execute(
                "SELECT username FROM sessions WHERE sid = ? AND revoked_at IS NULL",
                (sid,),
            )
            row = cur.fetchone()
            if row is None:
                return None
            self._conn.execute(
                "UPDATE sessions SET revoked_at = ? WHERE sid = ?", (now, sid),
            )
            self._conn.commit()
            return row["username"]

    def list_active_sessions(self) -> list[Session]:
        """Active = not revoked, not idle-timed-out, not absolute-timed-out.

        Idle/absolute are also enforced at read time; without this filter,
        a doctor who closed their laptop at 3pm would still appear "active"
        in the admin panel until someone hit an endpoint with their cookie.
        """
        now = time.time()
        with self._lock:
            cur = self._conn.execute(
                """
                SELECT * FROM sessions
                WHERE revoked_at IS NULL
                ORDER BY last_seen DESC
                """,
            )
            rows = cur.fetchall()
        out: list[Session] = []
        for r in rows:
            if now > r["expires_at"]:
                continue
            if now - r["last_seen"] > IDLE_TIMEOUT_S:
                continue
            out.append(self._row_to_session(r))
        return out

    def log_event(
        self,
        event_type: str,
        *,
        username: str | None = None,
        sid: str | None = None,
        user_agent: str | None = None,
        ip: str | None = None,
        detail: str | None = None,
    ) -> None:
        """Append one row to `auth_events`. Always durable — auth audit
        events are never overwritten or deleted from this code path."""
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO auth_events (event_type, username, sid, at, user_agent, ip, detail)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (event_type, username, sid, time.time(), user_agent, ip, detail),
            )
            self._conn.commit()

    def list_auth_events(self, limit: int = 200) -> list[AuthEvent]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM auth_events ORDER BY at DESC LIMIT ?",
                (limit,),
            )
            rows = cur.fetchall()
        return [
            AuthEvent(
                id=r["id"],
                event_type=r["event_type"],
                username=r["username"],
                sid=r["sid"],
                at=r["at"],
                user_agent=r["user_agent"],
                ip=r["ip"],
                detail=r["detail"],
            )
            for r in rows
        ]

    @staticmethod
    def _row_to_session(row: sqlite3.Row) -> Session:
        return Session(
            sid=row["sid"],
            username=row["username"],
            created_at=row["created_at"],
            last_seen=row["last_seen"],
            expires_at=row["expires_at"],
            revoked_at=row["revoked_at"],
            user_agent=row["user_agent"],
            ip=row["ip"],
        )

    def close(self) -> None:
        self._conn.close()
