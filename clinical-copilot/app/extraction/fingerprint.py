"""Content-fingerprint helper for dedup Layer 2.

Layer 1 (`writer.find_document_reference_by_sha`) catches re-uploads of
the same bytes. Layer 2 catches re-uploads of the *same content* in
different bytes — a re-scan of the same paper form, a re-photograph,
or the same lab report PDF re-exported from a different EHR. Both
cases produce different SHA-256 hashes but a structurally identical
extraction.

The fingerprint is a hex SHA-256 over a canonical-JSON projection of
the fields a doctor would consider semantically identifying. The
projection deliberately drops bbox coordinates, source-citation
metadata, and any field that varies under re-scan (image quality
markers, OCR confidence, etc.).

Returns `None` for unsupported document types so the caller can
no-op past Layer 2 instead of branching on `isinstance`.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from app.extraction.schemas import ExtractedDocument, IntakeForm, LabReport


def _canonical(payload: Any) -> str:
    """JSON-serialize with sorted keys + no whitespace, so a permutation
    of a dict's key insertion order produces the same hash."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _intake_projection(extracted: IntakeForm) -> dict:
    """Pull the identifying slice out of an `IntakeForm`.

    Sorts every list so the same intake captured in different write
    orders hashes the same. Lowercases free-text strings — handwritten
    forms transcribe with case variation, and "Penicillin" vs
    "penicillin" should not be a different fingerprint."""
    demo = extracted.demographics
    name = " ".join([
        (demo.given_name or "").strip().lower(),
        (demo.family_name or "").strip().lower(),
    ]).strip()
    dob = str(demo.date_of_birth) if demo.date_of_birth else ""
    chief = (extracted.chief_concern or "").strip().lower()

    allergies = sorted(
        ((a.substance or "").strip().lower(), (a.reaction or "").strip().lower())
        for a in extracted.allergies
    )
    meds = sorted(
        ((m.name or "").strip().lower(),
         (m.dose or "").strip().lower(),
         (m.frequency or "").strip().lower())
        for m in extracted.current_medications
    )
    family = sorted(
        ((f.relation or "").strip().lower(),
         (f.condition or "").strip().lower())
        for f in extracted.family_history
    )

    return {
        "doc_type": "intake_form",
        "name": name,
        "dob": dob,
        "chief_concern": chief,
        "allergies": allergies,
        "medications": meds,
        "family_history": family,
    }


def _lab_projection(extracted: LabReport) -> dict:
    """Pull the identifying slice out of a `LabReport`.

    Each lab result contributes (test_name, str(value), unit). The
    collection_date anchors the lab to a specific draw. Two reports
    of the same panel from different draw days hash differently — that
    is intentional: they are semantically different documents even if
    their values happen to coincide.

    `value` is stringified because some result rows are numeric and
    others are qualitative (`"positive"`, `"detected"`); JSON
    serializes them differently in the canonical form, but a numeric
    1 and the string '1' should hash the same here. Matching the
    schema's str-or-number semantics with str() keeps the fingerprint
    stable across schema-equivalent result rows.
    """
    rows = sorted(
        (
            (r.test_name or "").strip().lower(),
            str(r.value if r.value is not None else "").strip().lower(),
            (r.unit or "").strip().lower(),
        )
        for r in extracted.results
    )
    # All rows on the same collection_date — fingerprint anchors on the
    # earliest if the report mixes (it shouldn't, but the schema
    # tolerates per-result dates).
    collection_dates = sorted({
        str(r.collection_date) for r in extracted.results if r.collection_date
    })
    return {
        "doc_type": "lab_pdf",
        "collection_dates": collection_dates,
        "results": rows,
    }


def compute_fingerprint(extracted: ExtractedDocument) -> str | None:
    """Return a stable SHA-256 hex over the doc's identifying content.

    Returns `None` for an unrecognized document type — the caller can
    skip Layer 2 in that case rather than fail the whole upload.
    """
    if isinstance(extracted, IntakeForm):
        projection = _intake_projection(extracted)
    elif isinstance(extracted, LabReport):
        projection = _lab_projection(extracted)
    else:
        return None
    serialized = _canonical(projection)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
