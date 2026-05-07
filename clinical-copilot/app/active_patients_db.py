"""SQLite-backed dynamic active-patients store.

Companion to `access_control.ACTIVE_PATIENT_NAMES` (the static demo
allow-list). When a doctor uploads a document for a brand-new patient
via the create-from-upload flow, the resulting FHIR Patient's name
is stored here so the patient immediately appears in the calendar /
admin / picker views — without a code change.

`filter_active(patients)` UNIONs static + dynamic before deciding
which Patient resources to surface. Dynamic entries are the
lowercased (family, given) tuple to match the static set's shape.

Storage shares the same SQLite file as the other sidecar stores
(`data/traces.db`); same `check_same_thread=False` + lock pattern.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
from pathlib import Path

log = logging.getLogger("agent.active_patients.db")


class ActivePatientsStore:
    """Lowercased (family, given) name tuples added at runtime."""

    SCHEMA = """
        CREATE TABLE IF NOT EXISTS active_patient_overrides (
            family TEXT NOT NULL,
            given TEXT NOT NULL,
            added_by TEXT NOT NULL,
            added_at REAL NOT NULL,
            PRIMARY KEY (family, given)
        );
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(self.SCHEMA)
        self._conn.commit()
        log.info("active-patients store opened at %s", path)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def add(self, *, family: str, given: str, added_by: str) -> None:
        """Add a (family, given) pair. Idempotent — second add is a no-op."""
        f = (family or "").strip().lower()
        g = (given or "").strip().lower()
        if not f:
            return
        now = time.time()
        with self._lock:
            self._conn.execute(
                """
                INSERT OR IGNORE INTO active_patient_overrides
                    (family, given, added_by, added_at)
                VALUES (?, ?, ?, ?)
                """,
                (f, g, added_by, now),
            )
            self._conn.commit()
        log.info("active-patients add: family=%s given=%s by=%s", f, g, added_by)

    def remove(self, *, family: str, given: str) -> bool:
        """Remove a (family, given) override. Returns True if a row was deleted."""
        f = (family or "").strip().lower()
        g = (given or "").strip().lower()
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM active_patient_overrides WHERE family=? AND given=?",
                (f, g),
            )
            self._conn.commit()
            return cur.rowcount > 0

    def all(self) -> frozenset[tuple[str, str]]:
        """Return the full override set as a frozenset of (family, given)."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT family, given FROM active_patient_overrides"
            )
            rows = cur.fetchall()
        return frozenset((r["family"], r["given"]) for r in rows)
