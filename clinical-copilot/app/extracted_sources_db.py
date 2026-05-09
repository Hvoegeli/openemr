"""SQLite-backed sidecar mapping (resource_type, resource_id) -> source
DocumentReference + bbox.

Why this exists: OpenEMR's standard REST API and FHIR layer do not
round-trip the `comments` field as `note` for AllergyIntolerance
resources. The writer puts a `[copilot-source: <doc>; bbox=...]` tag
into `comments` for every extracted fact, and that tag survives the
round-trip for medical_problem and medication (so the bbox-manifest
adapter can read it back from FHIR `note`). For allergies, the FHIR
serializer in `FhirAllergyIntoleranceService.php` ignores the
`comments` column entirely — the tag is stored in MySQL but never
visible to FHIR clients. Result: allergy citations cannot deep-link
to their source PDF, and the bbox manifest never lists them.

This store is the round-trip-safe alternative. Every extracted fact
write also INSERTs a row here keyed by `(resource_type, resource_id)`,
with the source document id and (optional) bbox JSON. The manifest
adapter and the resource-source-document adapter both consult it as a
fallback (and a primary source for allergies). Sharing the same SQLite
file as the trace / auth / hide stores keeps deploy simpler — one
file to back up and migrate.

The table is also used to backfill historical resources whose tags
were dropped by OpenEMR (Chen's pre-existing allergies, etc.) — see
`scripts/backfill_extracted_sources.py`.
"""

from __future__ import annotations

import json as _json
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("agent.extracted_sources.db")


@dataclass(frozen=True)
class ResourceSource:
    """A single (resource -> source-doc) mapping the manifest reader
    treats as authoritative when present."""

    resource_type: str
    resource_id: str
    source_doc_id: str
    bbox: dict | None
    label: str | None
    recorded_at: float


class ExtractedSourcesStore:
    """Source-of-truth for `(resource_type, resource_id) -> source-doc + bbox`.

    Methods are thread-safe via a per-instance lock so concurrent
    uvicorn workers serialize writes cleanly. Reads are O(1) on the
    primary key and O(log n) via the source-doc index for manifest
    fan-out.

    `bbox` is stored as JSON text — keeps the schema small and means
    we don't need separate columns for page/x/y/width/height (or to
    migrate the schema if the bbox shape ever changes). NULL means
    "tag exists but no spatial coordinates" — the manifest reader
    surfaces these as bbox=None facts so the frontend page-band
    fallback can highlight at the document level.
    """

    SCHEMA = """
        CREATE TABLE IF NOT EXISTS extracted_resource_sources (
            resource_type TEXT NOT NULL,
            resource_id TEXT NOT NULL,
            source_doc_id TEXT NOT NULL,
            bbox_json TEXT,
            label TEXT,
            recorded_at REAL NOT NULL,
            PRIMARY KEY (resource_type, resource_id)
        );
        CREATE INDEX IF NOT EXISTS idx_ers_doc_id
            ON extracted_resource_sources (source_doc_id);
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(self.SCHEMA)
        self._conn.commit()
        log.info("extracted-sources store opened at %s", path)

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

    def record(
        self,
        *,
        resource_type: str,
        resource_id: str,
        source_doc_id: str,
        bbox: dict | None = None,
        label: str | None = None,
    ) -> None:
        """Upsert a `(resource, source-doc)` mapping. Idempotent — a
        re-write replaces the prior bbox/label and refreshes
        `recorded_at` so the audit log reflects the latest extraction.
        """
        if not resource_type or not resource_id or not source_doc_id:
            log.debug(
                "extracted-sources: skipping record with missing field "
                "(rtype=%r rid=%r doc=%r)",
                resource_type, resource_id, source_doc_id,
            )
            return
        doc_id = self._strip_doc_prefix(source_doc_id)
        bbox_json = _json.dumps(bbox, separators=(",", ":")) if bbox else None
        now = time.time()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO extracted_resource_sources
                    (resource_type, resource_id, source_doc_id, bbox_json,
                     label, recorded_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(resource_type, resource_id) DO UPDATE SET
                    source_doc_id = excluded.source_doc_id,
                    bbox_json = excluded.bbox_json,
                    label = excluded.label,
                    recorded_at = excluded.recorded_at
                """,
                (resource_type, resource_id, doc_id, bbox_json, label, now),
            )
            self._conn.commit()

    def get(self, *, resource_type: str, resource_id: str) -> ResourceSource | None:
        """Look up the source-doc mapping for one resource. Returns
        None when the resource was never extracted (hand-entered, seed
        data, or extracted before this store existed)."""
        with self._lock:
            cur = self._conn.execute(
                """
                SELECT * FROM extracted_resource_sources
                WHERE resource_type = ? AND resource_id = ?
                """,
                (resource_type, resource_id),
            )
            row = cur.fetchone()
        if row is None:
            return None
        return self._row_to_source(row)

    def list_for_doc(self, source_doc_id: str) -> list[ResourceSource]:
        """Every recorded fact extracted from the given DocumentReference,
        for the manifest endpoint to merge with the FHIR-note-derived
        facts. Sort is stable (resource_type, resource_id) so callers
        can rely on order for diffing."""
        doc_id = self._strip_doc_prefix(source_doc_id)
        with self._lock:
            cur = self._conn.execute(
                """
                SELECT * FROM extracted_resource_sources
                WHERE source_doc_id = ?
                ORDER BY resource_type, resource_id
                """,
                (doc_id,),
            )
            return [self._row_to_source(r) for r in cur.fetchall()]

    def delete(self, *, resource_type: str, resource_id: str) -> bool:
        """Drop a single mapping. Returns True if a row existed."""
        with self._lock:
            cur = self._conn.execute(
                """
                DELETE FROM extracted_resource_sources
                WHERE resource_type = ? AND resource_id = ?
                """,
                (resource_type, resource_id),
            )
            self._conn.commit()
            return cur.rowcount > 0

    @staticmethod
    def _row_to_source(row: sqlite3.Row) -> ResourceSource:
        bbox: dict | None = None
        raw = row["bbox_json"]
        if raw:
            try:
                parsed = _json.loads(raw)
                if isinstance(parsed, dict):
                    bbox = parsed
            except (ValueError, TypeError):
                bbox = None
        return ResourceSource(
            resource_type=row["resource_type"],
            resource_id=row["resource_id"],
            source_doc_id=row["source_doc_id"],
            bbox=bbox,
            label=row["label"],
            recorded_at=row["recorded_at"],
        )
