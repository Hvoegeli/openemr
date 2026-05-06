"""SQLite-backed (patient_id, fingerprint) -> document_id index for
dedup Layer 2 (content-match dedup).

Layer 1 (`writer.find_document_reference_by_sha`) catches re-uploads of
the same bytes; this store catches re-uploads of the same *content* in
different bytes (rescan of the same paper, re-photograph, etc.).

The schema is intentionally narrow: one row per (patient_id, fingerprint)
holding the first DocumentReference id we ever saw at that
fingerprint. If a later upload extracts to the same fingerprint, we
return the prior `document_id` so the upload endpoint can prompt the
user (Replace / Keep both / Cancel). Re-recording the fingerprint on
"Replace" is the responsibility of the upload endpoint — this store
just holds whatever it's given.

Storage shares the `data/traces.db` SQLite file used by the trace,
auth, assignment, and hidden-docs stores; concurrent uvicorn workers
serialize through the per-instance lock the same way.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("agent.fingerprints.db")


@dataclass
class FingerprintRecord:
    patient_id: str
    fingerprint: str
    document_id: str
    recorded_at: float
    recorded_by: str | None


class DocumentFingerprintStore:
    """`(patient_id, fingerprint) -> document_id` index.

    `document_id` is the bare uuid (no `DocumentReference/` prefix), to
    match the storage convention in `HiddenDocsStore`.
    """

    SCHEMA = """
        CREATE TABLE IF NOT EXISTS document_fingerprints (
            patient_id TEXT NOT NULL,
            fingerprint TEXT NOT NULL,
            document_id TEXT NOT NULL,
            recorded_at REAL NOT NULL,
            recorded_by TEXT,
            PRIMARY KEY (patient_id, fingerprint)
        );
        CREATE INDEX IF NOT EXISTS idx_fingerprints_document
            ON document_fingerprints(document_id);
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(self.SCHEMA)
        self._conn.commit()
        log.info("fingerprints store opened at %s", path)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @staticmethod
    def _strip_prefix(document_id: str) -> str:
        if "/" in document_id:
            return document_id.split("/", 1)[1]
        return document_id

    def find_match(
        self, *, patient_id: str, fingerprint: str,
    ) -> FingerprintRecord | None:
        """Look up the prior document_id with this fingerprint for this
        patient. Returns None when nothing matches."""
        with self._lock:
            cur = self._conn.execute(
                """
                SELECT * FROM document_fingerprints
                WHERE patient_id = ? AND fingerprint = ?
                LIMIT 1
                """,
                (patient_id, fingerprint),
            )
            row = cur.fetchone()
        if row is None:
            return None
        return FingerprintRecord(
            patient_id=row["patient_id"],
            fingerprint=row["fingerprint"],
            document_id=row["document_id"],
            recorded_at=row["recorded_at"],
            recorded_by=row["recorded_by"],
        )

    def record(
        self,
        *,
        patient_id: str,
        fingerprint: str,
        document_id: str,
        recorded_by: str | None = None,
    ) -> FingerprintRecord:
        """Persist a (patient_id, fingerprint) -> document_id row.

        Idempotent: re-recording the same key updates the document_id
        + audit fields. The upload endpoint relies on this when the
        user picks Replace — the new document supersedes the old as
        the canonical fingerprint owner."""
        doc_id = self._strip_prefix(document_id)
        now = time.time()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO document_fingerprints
                    (patient_id, fingerprint, document_id, recorded_at, recorded_by)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(patient_id, fingerprint) DO UPDATE SET
                    document_id = excluded.document_id,
                    recorded_at = excluded.recorded_at,
                    recorded_by = excluded.recorded_by
                """,
                (patient_id, fingerprint, doc_id, now, recorded_by),
            )
            self._conn.commit()
        return FingerprintRecord(
            patient_id=patient_id,
            fingerprint=fingerprint,
            document_id=doc_id,
            recorded_at=now,
            recorded_by=recorded_by,
        )

    def forget_document(self, document_id: str) -> int:
        """Drop every fingerprint pointing at `document_id`. Used when
        the user picks Replace (the old doc's fingerprint is freed so
        the new doc can claim it via `record`). Returns the number of
        rows removed."""
        doc_id = self._strip_prefix(document_id)
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM document_fingerprints WHERE document_id = ?",
                (doc_id,),
            )
            self._conn.commit()
            return cur.rowcount
