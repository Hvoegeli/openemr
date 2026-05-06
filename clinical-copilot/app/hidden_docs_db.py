"""SQLite-backed soft-hide store for DocumentReference rows.

`/api/document/{id}/hide` flips a row in this table; the supporting-docs
list filters by membership so hidden documents drop off the default
view. The underlying OpenEMR DocumentReference is preserved — this is a
view-side toggle, not a deletion. The user can flip `?include_hidden=true`
on the docs endpoint to see them again, and `/api/document/{id}/unhide`
restores the doc to the default view.

Storage shares the same SQLite file as the trace store + auth store +
assignments (`data/traces.db`); same `check_same_thread=False` + lock
pattern as `AuthStore`/`AssignmentStore` so concurrent uvicorn workers
serialize cleanly.

Why a sidecar store and not FHIR `DocumentReference.status = superseded`:
OpenEMR's FHIR module does not currently accept PUT on DocumentReference
in our build, and re-writing status via the standard API leaks across
patient charts (the same row could be hidden for one user and not
another in a multi-tenant deploy). Keeping the hide flag here means it
is reversible, auditable, and never touches the source-of-truth FHIR
record.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("agent.hidden_docs.db")


@dataclass
class HiddenDoc:
    document_id: str
    hidden_by: str
    hidden_at: float


class HiddenDocsStore:
    """`document_id` membership store for soft-hidden DocumentReferences.

    Methods are thread-safe via a per-instance lock — concurrent uvicorn
    workers go through the same serialization as the other stores.
    `document_id` is the bare uuid (no `DocumentReference/` prefix), so
    callers consistently strip the prefix before lookup.
    """

    SCHEMA = """
        CREATE TABLE IF NOT EXISTS hidden_documents (
            document_id TEXT PRIMARY KEY,
            hidden_by TEXT NOT NULL,
            hidden_at REAL NOT NULL
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
        log.info("hidden-docs store opened at %s", path)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @staticmethod
    def _strip_prefix(document_id: str) -> str:
        """Accept either `DocumentReference/<uuid>` or `<uuid>` and return
        the bare uuid the schema stores."""
        if "/" in document_id:
            return document_id.split("/", 1)[1]
        return document_id

    def hide(self, *, document_id: str, hidden_by: str) -> HiddenDoc:
        """Mark a document hidden. Idempotent — re-hiding the same doc
        refreshes `hidden_at`/`hidden_by` so the audit log reflects the
        most recent action. Returns the persisted row."""
        doc_id = self._strip_prefix(document_id)
        now = time.time()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO hidden_documents (document_id, hidden_by, hidden_at)
                VALUES (?, ?, ?)
                ON CONFLICT(document_id) DO UPDATE SET
                    hidden_by = excluded.hidden_by,
                    hidden_at = excluded.hidden_at
                """,
                (doc_id, hidden_by, now),
            )
            self._conn.commit()
        return HiddenDoc(document_id=doc_id, hidden_by=hidden_by, hidden_at=now)

    def unhide(self, document_id: str) -> bool:
        """Restore a document to the default view. Returns True if a
        row existed; False if the doc wasn't hidden in the first place."""
        doc_id = self._strip_prefix(document_id)
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM hidden_documents WHERE document_id = ?",
                (doc_id,),
            )
            self._conn.commit()
            return cur.rowcount > 0

    def is_hidden(self, document_id: str) -> bool:
        doc_id = self._strip_prefix(document_id)
        with self._lock:
            cur = self._conn.execute(
                "SELECT 1 FROM hidden_documents WHERE document_id = ? LIMIT 1",
                (doc_id,),
            )
            return cur.fetchone() is not None

    def list_hidden_ids(self) -> set[str]:
        """Return every hidden document_id (bare uuid form) so the docs
        endpoint can filter in a single set lookup. Cheap on small N
        (typical demo: 0-50 hidden); a future deploy with thousands
        would want a per-patient index instead."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT document_id FROM hidden_documents",
            )
            return {row["document_id"] for row in cur.fetchall()}
