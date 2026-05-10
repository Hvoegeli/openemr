"""SQLite-backed care-team store: physicians extracted from referral
letters, keyed by `(patient_uuid, source_doc_id)`.

Why this exists: OpenEMR's `CareTeam` FHIR resource is rarely populated
for demo patients, and `ReferringPhysician` (extracted from referral
letters by the Phase 2 VLM pipeline) was previously discarded after
extraction — the persistence path writes problems / allergies / meds
back to OpenEMR but has no FHIR target for the physician's contact
block. This store captures it so the Modern Dashboard's Care Team tab
has real data to show.

Composite primary key `(patient_uuid, source_doc_id)` means re-uploading
the same referral idempotently overwrites — the latest extraction wins,
no duplicates pile up. Uploading two distinct referrals naming the same
physician will produce two rows; deduplication by physician identity is
deferred (no NPI on most letters → no clean key). The dashboard tolerates
duplicates: it groups by (name, practice) at render time.

Same SQLite file as the trace / auth / hide / extracted-sources stores
to keep deploy simple — one file to back up and migrate.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("agent.extracted_practitioners.db")


@dataclass(frozen=True)
class ExtractedPractitioner:
    """One physician extracted from one source document for one patient."""

    patient_uuid: str
    source_doc_id: str
    name: str
    practice: str | None
    specialty: str | None
    phone: str | None
    address: str | None
    npi: str | None
    recorded_at: float


class ExtractedPractitionersStore:
    """Source-of-truth for care-team entries derived from extracted documents.

    Methods are thread-safe via a per-instance lock so concurrent uvicorn
    workers serialize writes cleanly.
    """

    SCHEMA = """
        CREATE TABLE IF NOT EXISTS extracted_practitioners (
            patient_uuid TEXT NOT NULL,
            source_doc_id TEXT NOT NULL,
            name TEXT NOT NULL,
            practice TEXT,
            specialty TEXT,
            phone TEXT,
            address TEXT,
            npi TEXT,
            recorded_at REAL NOT NULL,
            PRIMARY KEY (patient_uuid, source_doc_id)
        );
        CREATE INDEX IF NOT EXISTS idx_ep_patient
            ON extracted_practitioners (patient_uuid);
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(self.SCHEMA)
        self._conn.commit()
        log.info("extracted-practitioners store opened at %s", path)

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

    def upsert(
        self,
        *,
        patient_uuid: str,
        source_doc_id: str,
        name: str,
        practice: str | None = None,
        specialty: str | None = None,
        phone: str | None = None,
        address: str | None = None,
        npi: str | None = None,
    ) -> None:
        """Idempotent upsert keyed by `(patient_uuid, source_doc_id)`. A
        re-extraction (re-upload of the same referral) replaces the prior
        row and refreshes `recorded_at` so the audit log reflects the
        latest extraction."""
        if not patient_uuid or not source_doc_id or not name:
            log.debug(
                "extracted-practitioners: skipping upsert with missing field "
                "(patient=%r doc=%r name=%r)",
                patient_uuid, source_doc_id, name,
            )
            return
        doc_id = self._strip_doc_prefix(source_doc_id)
        now = time.time()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO extracted_practitioners
                    (patient_uuid, source_doc_id, name, practice, specialty,
                     phone, address, npi, recorded_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(patient_uuid, source_doc_id) DO UPDATE SET
                    name = excluded.name,
                    practice = excluded.practice,
                    specialty = excluded.specialty,
                    phone = excluded.phone,
                    address = excluded.address,
                    npi = excluded.npi,
                    recorded_at = excluded.recorded_at
                """,
                (patient_uuid, doc_id, name, practice, specialty, phone,
                 address, npi, now),
            )
            self._conn.commit()

    def list_for_patient(self, patient_uuid: str) -> list[ExtractedPractitioner]:
        """Every recorded physician for the given patient. Sort is
        newest-first by `recorded_at` so the most recently uploaded
        referral ranks first on the dashboard.
        """
        if not patient_uuid:
            return []
        with self._lock:
            cur = self._conn.execute(
                """
                SELECT * FROM extracted_practitioners
                WHERE patient_uuid = ?
                ORDER BY recorded_at DESC
                """,
                (patient_uuid,),
            )
            return [self._row_to_practitioner(r) for r in cur.fetchall()]

    def delete(self, *, patient_uuid: str, source_doc_id: str) -> bool:
        """Drop a single mapping. Returns True if a row existed."""
        doc_id = self._strip_doc_prefix(source_doc_id)
        with self._lock:
            cur = self._conn.execute(
                """
                DELETE FROM extracted_practitioners
                WHERE patient_uuid = ? AND source_doc_id = ?
                """,
                (patient_uuid, doc_id),
            )
            self._conn.commit()
            return cur.rowcount > 0

    @staticmethod
    def _row_to_practitioner(row: sqlite3.Row) -> ExtractedPractitioner:
        return ExtractedPractitioner(
            patient_uuid=row["patient_uuid"],
            source_doc_id=row["source_doc_id"],
            name=row["name"],
            practice=row["practice"],
            specialty=row["specialty"],
            phone=row["phone"],
            address=row["address"],
            npi=row["npi"],
            recorded_at=row["recorded_at"],
        )
