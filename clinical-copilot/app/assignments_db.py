"""SQLite-backed patient-assignment store.

Authoritative for which patients sit in which provider's panel within the
co-pilot. We keep this here (vs OpenEMR's `Patient.generalPractitioner`
FHIR field) because OpenEMR's Demographics → Provider UI does not write to
that FHIR field — assignments set there don't propagate to a FHIR query for
`?general-practitioner=Practitioner/{id}`. Until the OpenEMR mapping gap
closes, the co-pilot owns assignments end-to-end via this table.

Production path: the admin endpoint that writes this table can also fan out
a FHIR PUT/PATCH on `Patient.generalPractitioner` once OpenEMR's mapping is
fixed (or once we patch the upstream FHIR Patient mapper). Until then,
co-pilot is the single source of truth and the admin UI is the only legal
mutation point.

Storage shares the same SQLite file as the trace store + auth store
(`data/traces.db`) — same `check_same_thread=False` + lock pattern as
`AuthStore` so concurrent uvicorn workers serialize cleanly.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("agent.assignments.db")


@dataclass
class Assignment:
    patient_id: str
    practitioner_id: str
    assigned_at: float
    assigned_by: str


class AssignmentStore:
    """Patient → Practitioner assignment store (1 practitioner per patient)."""

    SCHEMA = """
        CREATE TABLE IF NOT EXISTS patient_assignments (
            patient_id TEXT PRIMARY KEY,
            practitioner_id TEXT NOT NULL,
            assigned_at REAL NOT NULL,
            assigned_by TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_assignments_prac
            ON patient_assignments(practitioner_id);
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(self.SCHEMA)
        self._conn.commit()
        log.info("assignment store opened at %s", path)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def upsert(
        self, *, patient_id: str, practitioner_id: str, assigned_by: str,
    ) -> Assignment:
        """Set the patient's assigned practitioner (replacing any existing)."""
        now = time.time()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO patient_assignments
                    (patient_id, practitioner_id, assigned_at, assigned_by)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(patient_id) DO UPDATE SET
                    practitioner_id = excluded.practitioner_id,
                    assigned_at = excluded.assigned_at,
                    assigned_by = excluded.assigned_by
                """,
                (patient_id, practitioner_id, now, assigned_by),
            )
            self._conn.commit()
        return Assignment(
            patient_id=patient_id,
            practitioner_id=practitioner_id,
            assigned_at=now,
            assigned_by=assigned_by,
        )

    def unassign(self, patient_id: str) -> bool:
        """Remove the assignment for a patient. Returns True if a row existed."""
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM patient_assignments WHERE patient_id = ?",
                (patient_id,),
            )
            self._conn.commit()
            return cur.rowcount > 0

    def get_assignment(self, patient_id: str) -> Assignment | None:
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM patient_assignments WHERE patient_id = ?",
                (patient_id,),
            )
            row = cur.fetchone()
        if row is None:
            return None
        return Assignment(
            patient_id=row["patient_id"],
            practitioner_id=row["practitioner_id"],
            assigned_at=row["assigned_at"],
            assigned_by=row["assigned_by"],
        )

    def patients_for_practitioner(self, practitioner_id: str) -> list[str]:
        """Return the list of patient IDs assigned to `practitioner_id`."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT patient_id FROM patient_assignments WHERE practitioner_id = ?",
                (practitioner_id,),
            )
            rows = cur.fetchall()
        return [r["patient_id"] for r in rows]

    def all_assignments(self) -> dict[str, str]:
        """Return {patient_id: practitioner_id} for every assignment."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT patient_id, practitioner_id FROM patient_assignments"
            )
            rows = cur.fetchall()
        return {r["patient_id"]: r["practitioner_id"] for r in rows}
