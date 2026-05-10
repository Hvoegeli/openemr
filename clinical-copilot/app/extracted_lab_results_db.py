"""SQLite-backed lab-results store: individual lab values extracted from
lab PDFs / fax-packet lab pages, keyed by `(source_doc_id, row_index)`.

Why this exists: OpenEMR's standard REST API doesn't expose a write
endpoint for `procedure_result` (the native lab table) or for FHIR
`Observation` resources. The pragmatic surface available to the writer
is encounter + SOAP-note-objective text, which is human-readable and
flows to the chat agent's `get_notes_24h` but is invisible to anything
that queries `Observation?category=laboratory` — so the Modern
Dashboard's Lab Results tab was permanently empty.

This store captures each extracted lab result as a structured row so
the dashboard can show real values (test_name, value, unit, reference
range, abnormal flag, collection date) without depending on an
OpenEMR-side schema change. Same SQLite file as the other sidecar
stores (auth, traces, extracted-sources, extracted-practitioners) for
one-file deploy.

Composite primary key `(source_doc_id, row_index)` means re-extracting
the same lab PDF idempotently overwrites — the latest extraction wins.
Different lab PDFs produce different rows (no value-based dedup).
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("agent.extracted_lab_results.db")


@dataclass(frozen=True)
class ExtractedLabResult:
    """One lab result row extracted from one source document."""

    patient_uuid: str
    source_doc_id: str
    row_index: int
    test_name: str
    value: str | None       # stringified — labs can be numeric or qualitative
    unit: str | None
    reference_range: str | None
    abnormal_flag: str | None
    collection_date: str | None  # ISO date as stored on the LabResult schema
    recorded_at: float


class ExtractedLabResultsStore:
    """Source-of-truth for lab results derived from extracted documents.

    Methods are thread-safe via a per-instance lock so concurrent
    uvicorn workers serialize writes cleanly.
    """

    SCHEMA = """
        CREATE TABLE IF NOT EXISTS extracted_lab_results (
            patient_uuid TEXT NOT NULL,
            source_doc_id TEXT NOT NULL,
            row_index INTEGER NOT NULL,
            test_name TEXT NOT NULL,
            value TEXT,
            unit TEXT,
            reference_range TEXT,
            abnormal_flag TEXT,
            collection_date TEXT,
            recorded_at REAL NOT NULL,
            PRIMARY KEY (source_doc_id, row_index)
        );
        CREATE INDEX IF NOT EXISTS idx_elr_patient
            ON extracted_lab_results (patient_uuid);
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(self.SCHEMA)
        self._conn.commit()
        log.info("extracted-lab-results store opened at %s", path)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @staticmethod
    def _strip_doc_prefix(source_doc_id: str) -> str:
        """Accept either `DocumentReference/<uuid>` or `<uuid>` form so
        callers can be sloppy. Stored form is the bare uuid."""
        if "/" in source_doc_id:
            return source_doc_id.split("/", 1)[1]
        return source_doc_id

    def upsert_batch(
        self,
        *,
        patient_uuid: str,
        source_doc_id: str,
        rows: list[dict],
    ) -> int:
        """Idempotent upsert of all rows from one source document.

        Each row dict has: test_name, value (any type — stringified),
        unit, reference_range, abnormal_flag, collection_date.
        Re-extraction of the same source replaces all prior rows for
        that doc (delete-then-insert under the lock).

        Returns count written. Returns 0 with no DB write when input
        is empty or required identifiers are missing.
        """
        if not patient_uuid or not source_doc_id or not rows:
            return 0
        doc_id = self._strip_doc_prefix(source_doc_id)
        now = time.time()
        with self._lock:
            # Delete-then-insert under one transaction so the source's
            # rows are always coherent — readers never see a half-replaced
            # state.
            self._conn.execute(
                "DELETE FROM extracted_lab_results WHERE source_doc_id = ?",
                (doc_id,),
            )
            written = 0
            for i, r in enumerate(rows):
                test_name = (r.get("test_name") or "").strip()
                if not test_name:
                    continue
                value = r.get("value")
                value_str = None if value is None else str(value)
                self._conn.execute(
                    """
                    INSERT INTO extracted_lab_results
                        (patient_uuid, source_doc_id, row_index, test_name,
                         value, unit, reference_range, abnormal_flag,
                         collection_date, recorded_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        patient_uuid, doc_id, i, test_name,
                        value_str, r.get("unit"), r.get("reference_range"),
                        r.get("abnormal_flag"),
                        str(r.get("collection_date")) if r.get("collection_date") else None,
                        now,
                    ),
                )
                written += 1
            self._conn.commit()
            return written

    def list_for_patient(self, patient_uuid: str) -> list[ExtractedLabResult]:
        """Every recorded lab result for the patient, newest collection
        date first (then newest extraction first, for ties)."""
        if not patient_uuid:
            return []
        with self._lock:
            cur = self._conn.execute(
                """
                SELECT * FROM extracted_lab_results
                WHERE patient_uuid = ?
                ORDER BY collection_date DESC, recorded_at DESC, row_index ASC
                """,
                (patient_uuid,),
            )
            return [self._row_to_result(r) for r in cur.fetchall()]

    def delete_for_doc(self, source_doc_id: str) -> int:
        """Drop all rows for one source document. Returns count deleted."""
        doc_id = self._strip_doc_prefix(source_doc_id)
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM extracted_lab_results WHERE source_doc_id = ?",
                (doc_id,),
            )
            self._conn.commit()
            return cur.rowcount

    @staticmethod
    def _row_to_result(row: sqlite3.Row) -> ExtractedLabResult:
        return ExtractedLabResult(
            patient_uuid=row["patient_uuid"],
            source_doc_id=row["source_doc_id"],
            row_index=row["row_index"],
            test_name=row["test_name"],
            value=row["value"],
            unit=row["unit"],
            reference_range=row["reference_range"],
            abnormal_flag=row["abnormal_flag"],
            collection_date=row["collection_date"],
            recorded_at=row["recorded_at"],
        )
