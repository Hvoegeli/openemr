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

from app.extraction.schemas import (
    ExtractedDocument,
    FaxPacket,
    Hl7Message,
    IntakeForm,
    LabReport,
    ReferralLetter,
    Workbook,
)


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


def _referral_projection(extracted: ReferralLetter) -> dict:
    """Pull the identifying slice out of a `ReferralLetter`.

    Two referral letters fingerprint identically when they share the same
    referring physician, reason for referral, and structured chart slice
    (PMH conditions, current meds, allergies). Free-text narrative
    (HPI, requested action) is included normalized — re-typing the same
    referral with whitespace differences should still collide. Pertinent
    labs are included by (test, value, unit) tuples; their dates are
    omitted because the same referral re-sent on a different day should
    still hash the same."""
    rp = extracted.referring_physician
    physician = (rp.name or "").strip().lower()
    reason = " ".join((extracted.reason_for_referral or "").lower().split())
    hpi = " ".join((extracted.history_of_present_illness or "").lower().split())
    action = " ".join((extracted.requested_action or "").lower().split())

    pmh = sorted(
        ((c.name or "").strip().lower(), (c.icd10_code or "").strip().lower())
        for c in extracted.past_medical_history
    )
    meds = sorted(
        ((m.name or "").strip().lower(),
         (m.dose or "").strip().lower(),
         (m.frequency or "").strip().lower())
        for m in extracted.current_medications
    )
    allergies = sorted(
        ((a.substance or "").strip().lower(), (a.reaction or "").strip().lower())
        for a in extracted.allergies
    )
    labs = sorted(
        (
            (r.test_name or "").strip().lower(),
            str(r.value if r.value is not None else "").strip().lower(),
            (r.unit or "").strip().lower(),
        )
        for r in extracted.pertinent_labs
    )
    return {
        "doc_type": "referral_letter",
        "referring_physician": physician,
        "reason_for_referral": reason,
        "history_of_present_illness": hpi,
        "requested_action": action,
        "past_medical_history": pmh,
        "medications": meds,
        "allergies": allergies,
        "pertinent_labs": labs,
    }


def _hl7_projection(extracted: Hl7Message) -> dict:
    """Pull the identifying slice out of an `Hl7Message`.

    HL7's MSH-10 (message_control_id) is supposed to be globally unique
    per message, but in practice senders reuse it for retransmits or
    derive it from a deterministic timestamp template — so we cannot
    rely on it alone. We hash the clinically-relevant payload instead:

    - For ADT-A08: the EVN reason note + the AL1 substance/severity set.
    - For ORU-R01: the per-panel (loinc, collection_date) pair plus each
      panel's (test_name, value, unit) result tuples + the NTE notes.

    Free-text fields are whitespace-normalized + lowercased so a re-typed
    copy with cosmetic differences fingerprints the same. Two ORUs of
    the same panel drawn on different days fingerprint differently, just
    like the LabReport projection — different draw days are different
    documents."""
    if extracted.message_type == "ADT^A08":
        reason = " ".join((extracted.event_reason or "").lower().split())
        allergies = sorted(
            ((a.substance or "").strip().lower(),
             (a.severity or "").strip().lower(),
             (a.reaction or "").strip().lower())
            for a in extracted.allergies
        )
        return {
            "doc_type": "hl7_message",
            "message_type": "ADT^A08",
            "event_reason": reason,
            "allergies": allergies,
        }
    # ORU-R01
    panels: list[dict] = []
    for p in extracted.lab_panels:
        rows = sorted(
            (
                (r.test_name or "").strip().lower(),
                str(r.value if r.value is not None else "").strip().lower(),
                (r.unit or "").strip().lower(),
            )
            for r in p.results
        )
        panels.append({
            "loinc": (p.panel_loinc or "").strip().lower(),
            "collection_date": str(p.collection_date) if p.collection_date else "",
            "results": rows,
            "note": " ".join((p.notes or "").lower().split()),
        })
    return {
        "doc_type": "hl7_message",
        "message_type": "ORU^R01",
        "panels": panels,
    }


def _fax_projection(extracted: FaxPacket) -> dict:
    """Pull the identifying slice out of a `FaxPacket`.

    Two faxes fingerprint identically when they share the same fax_date,
    sender/recipient pair, reason for consultation, the structured chart
    slice (active problems, current medications, allergies), and the
    embedded lab values. The cover-sheet free-text MESSAGE field is
    whitespace-normalized + lowercased so retransmitted fax with a
    cosmetic re-typing still collides. Lab dates ARE included (the fax
    IS the source of those results — different draw days are different
    documents, matching the LabReport projection's behavior).
    """
    cover = extracted.cover_sheet
    fax_date = str(cover.fax_date) if cover.fax_date else ""
    sender = (cover.sender_name or "").strip().lower()
    recipient = (cover.recipient_name or "").strip().lower()
    message = " ".join((cover.message or "").lower().split())
    reason = " ".join((extracted.reason_for_consultation or "").lower().split())
    specialty = (extracted.receiving_specialty or "").strip().lower()

    problems = sorted(
        ((c.name or "").strip().lower(), (c.icd10_code or "").strip().lower())
        for c in extracted.active_problems
    )
    meds = sorted(
        ((m.name or "").strip().lower(),
         (m.dose or "").strip().lower(),
         (m.frequency or "").strip().lower())
        for m in extracted.current_medications
    )
    allergies = sorted(
        ((a.substance or "").strip().lower(), (a.reaction or "").strip().lower())
        for a in extracted.allergies
    )
    labs = sorted(
        (
            (r.test_name or "").strip().lower(),
            str(r.value if r.value is not None else "").strip().lower(),
            (r.unit or "").strip().lower(),
            str(r.collection_date) if r.collection_date else "",
        )
        for r in extracted.lab_results
    )
    return {
        "doc_type": "fax_packet",
        "fax_date": fax_date,
        "sender": sender,
        "recipient": recipient,
        "specialty": specialty,
        "message": message,
        "reason_for_consultation": reason,
        "active_problems": problems,
        "medications": meds,
        "allergies": allergies,
        "lab_results": labs,
    }


def _workbook_projection(extracted: Workbook) -> dict:
    """Pull the identifying slice out of a `Workbook`.

    Two workbooks fingerprint identically when they share the same
    patient (name + DOB + MRN), the same as_of_date, and the same
    structured payload (medication tuples, lab trend matrix, care-gap
    measures). The cover-style metadata fields (PCP, insurance) are
    captured because two workbooks for the same patient pulled at the
    same as_of_date with different PCPs are semantically different
    documents (e.g. a referring-vs-receiving practice copy).

    Lab-trend dates ARE included — the workbook IS the source of those
    historical values, so a re-pull on a later date with new columns
    fingerprints differently (matches the LabReport / FaxPacket
    behavior).
    """
    name = (extracted.patient_name or "").strip().lower()
    dob = str(extracted.patient_dob) if extracted.patient_dob else ""
    mrn = (extracted.patient_mrn or "").strip().lower()
    as_of = str(extracted.as_of_date) if extracted.as_of_date else ""
    pcp = (extracted.pcp_name or "").strip().lower()
    insurance = (extracted.insurance or "").strip().lower()
    allergies = (extracted.allergies_text or "").strip().lower()

    meds = sorted(
        ((m.generic or m.brand or "").strip().lower(),
         (m.strength or "").strip().lower(),
         (m.sig or "").strip().lower(),
         str(m.start_date) if m.start_date else "")
        for m in extracted.medications
    )
    # Lab trend: each row contributes one tuple with all of its dated
    # values sorted; the surrounding list is then sorted across rows.
    labs: list[tuple] = []
    for trend in extracted.lab_trends:
        per_row = sorted(
            (str(v.collection_date), v.value.strip().lower())
            for v in trend.values
        )
        labs.append((
            (trend.test_name or "").strip().lower(),
            (trend.units or "").strip().lower(),
            tuple(per_row),
        ))
    labs.sort()
    care_gaps = sorted(
        (
            (g.measure or "").strip().lower(),
            (g.status or "").strip().lower(),
            str(g.due_date) if g.due_date else "",
        )
        for g in extracted.care_gaps
    )
    return {
        "doc_type": "workbook",
        "patient_name": name,
        "patient_dob": dob,
        "patient_mrn": mrn,
        "as_of_date": as_of,
        "pcp": pcp,
        "insurance": insurance,
        "allergies": allergies,
        "medications": meds,
        "lab_trends": labs,
        "care_gaps": care_gaps,
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
    elif isinstance(extracted, ReferralLetter):
        projection = _referral_projection(extracted)
    elif isinstance(extracted, Hl7Message):
        projection = _hl7_projection(extracted)
    elif isinstance(extracted, FaxPacket):
        projection = _fax_projection(extracted)
    elif isinstance(extracted, Workbook):
        projection = _workbook_projection(extracted)
    else:
        return None
    serialized = _canonical(projection)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


# Layer 1.5 — pre-extraction text-layer dedup. Companion to
# `compute_fingerprint` (Layer 2). Layer 2 needs a vision pass to derive
# the structural fields it hashes; Layer 1.5 hashes the PDF's embedded
# text layer directly so we can short-circuit a duplicate prompt
# *before* paying for a Claude vision call.

_TEXT_FP_NAMESPACE = "text-v1:"


def namespace_text_fingerprint(text_fp: str) -> str:
    """Wrap a raw PDF text-layer fingerprint with the L1.5 namespace prefix.

    The fingerprints store is shared with the post-extraction structural
    fingerprints; the prefix prevents a freak SHA collision between the
    two layers from cross-prompting a user with a doc that matches by
    text but not by extracted facts (or vice versa).
    """
    return f"{_TEXT_FP_NAMESPACE}{text_fp}"


def pdf_text_fingerprint(pdf_bytes: bytes) -> str | None:
    """Return a stable SHA-256 over the normalized text layer of a PDF.

    Used by dedup Layer 1.5 to short-circuit a Layer-2 prompt for a
    re-uploaded text-PDF *before* paying for a Claude vision call. Two
    PDFs with identical visible text but different bytes — the same
    lab report re-exported from a different EHR with new generator
    metadata, the same referral re-printed at a different DPI — collide
    here even though the byte-SHA differs.

    Returns `None` when the PDF carries no extractable text. Pure
    scanned-image PDFs (faxes that arrived as TIFFs and were re-wrapped
    as PDF, photos of paper forms) need vision; the post-extraction
    structural fingerprint catches those duplicates downstream.

    Normalization is intentionally aggressive — Unicode whitespace runs
    collapse to a single space and case is folded — so two PDFs with
    the same visible text but different paragraph wrapping or font
    casing still hash the same.
    """
    if not pdf_bytes:
        return None
    # Lazy import: pypdfium2 ships a sizeable native library and we
    # don't want to pay it on `from app.extraction.fingerprint import …`
    # paths (e.g. agent graph startup) that never touch this helper.
    import pypdfium2 as pdfium  # noqa: PLC0415

    # Open defensively — non-PDF bytes (a mislabelled MIME at the
    # writer, or a `contentType`-less attachment that the resolve
    # endpoint defaulted to "application/pdf") raise PdfiumError.
    # Treat that as "no text-layer fingerprint" so the caller falls
    # through to vision instead of 500-ing the upload.
    try:
        pdf = pdfium.PdfDocument(pdf_bytes)
    except pdfium.PdfiumError:
        return None
    try:
        if len(pdf) == 0:
            return None
        normalized_pages: list[str] = []
        any_text = False
        for page in pdf:
            try:
                tp = page.get_textpage()
                try:
                    text = tp.get_text_range()
                finally:
                    tp.close()
            finally:
                page.close()
            normalized = " ".join(text.split()).lower()
            if normalized:
                any_text = True
            normalized_pages.append(normalized)
        if not any_text:
            return None
        # Per-page newline keeps page boundaries observable in the hash;
        # two PDFs with the same text rearranged across pages (rare but
        # possible) hash differently. Within a page, content is already
        # whitespace-collapsed to one line.
        joined = "\n".join(normalized_pages)
        return hashlib.sha256(joined.encode("utf-8")).hexdigest()
    finally:
        pdf.close()
