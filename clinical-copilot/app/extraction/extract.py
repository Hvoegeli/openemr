"""Top-level `attach_and_extract` orchestrator.

End-to-end pipeline for turning an uploaded clinical document into
(a) a persisted FHIR DocumentReference in OpenEMR and (b) the
structured-typed extraction of its clinical contents.

Two phases:

1. **Persist source.** `OpenEMRWriter.write_document_reference` uploads
   the file and returns a `DocumentReference/{uuid}` id. SHA-256
   idempotency means re-uploading the same bytes returns the existing
   id without a second POST.

2. **Extract.** Render the document to per-page PNGs, send them to
   Claude vision with the `record_<doc_type>` tool forced, and validate
   the returned tool input against the matching Pydantic schema.

The persisted DocumentReference id is threaded through to the
extraction call as `source_document_id` so every Citation in the
returned object points at the real OpenEMR resource — not a synthetic
placeholder.

Both the writer and the Anthropic client are injected so the test
suite can replace them with fakes. Production callers leave both
unset and accept the defaults (`OpenEMRWriter()` + `AsyncAnthropic()`).

This module deliberately does NOT touch the agent graph. The function
is callable from any context (CLI smoke, FastAPI endpoint, unit test).
Wiring it into the agent's tool list happens in Phase 2.4 alongside
the upload endpoint.
"""

from __future__ import annotations

import logging
from typing import Any

from anthropic import AsyncAnthropic

from app.extraction.hl7 import Hl7ParseError, parse_hl7_message
from app.extraction.render import render_to_png_pages
from app.extraction.schemas import (
    Citation,
    DocumentType,
    ExtractedDocument,
    FaxPacket,
    Hl7Message,
    IntakeForm,
    LabReport,
    LabResult,
    ReferralLetter,
    Workbook,
)
from app.extraction.vision import DEFAULT_MODEL, ExtractionError, extract_via_claude
from app.extraction.workbook import WorkbookParseError, parse_workbook
from app.fhir.writer import OpenEMRWriteError, OpenEMRWriter

log = logging.getLogger("agent.extraction.extract")


class AttachAndExtractResult:
    """Lightweight value-object holding both halves of the pipeline output.

    Using a class (not a TypedDict / dict) so the IDE / type-checker
    can see that `extracted` is `ExtractedDocument` and `write_result`
    is a real dict shape from the writer — not just `Any`.
    """

    __slots__ = ("extracted", "write_result", "persistence")

    def __init__(
        self,
        extracted: ExtractedDocument | None,
        write_result: dict[str, Any],
        persistence: dict[str, Any] | None = None,
    ) -> None:
        # `extracted` is `None` only when the caller passed
        # `skip_extraction=True` — the writer ran (Phase 1) but the
        # vision call was deferred so the caller can interpose a
        # pre-extraction check (typically Layer-1.5 text-fingerprint
        # dedup) and decide whether to incur Anthropic credits.
        self.extracted = extracted
        self.write_result = write_result
        # Persistence is the optional Phase-3 summary describing which
        # extracted facts were written into OpenEMR's native tables (and
        # which failed). `None` when Phase 3 was skipped (e.g. unit
        # tests). Each per-fact entry is `{kind, ok, id, error}` so a
        # caller can render a partial-success message in the UI.
        self.persistence = persistence

    @property
    def reference_id(self) -> str:
        """Convenience: the FHIR DocumentReference id from the writer
        result. Same value as `extracted.source_document_id`."""
        return str(self.write_result["reference_id"])

    @property
    def created(self) -> bool:
        """True if the writer actually uploaded the file; False on a
        SHA-256 idempotency hit (re-upload of the same bytes)."""
        return bool(self.write_result["created"])


async def attach_and_extract(
    *,
    file_bytes: bytes,
    filename: str,
    doc_type: DocumentType,
    patient_uuid: str,
    mime_type: str = "application/pdf",
    writer: OpenEMRWriter | None = None,
    anthropic_client: AsyncAnthropic | None = None,
    model: str = DEFAULT_MODEL,
    skip_extraction: bool = False,
    skip_persistence: bool = False,
    practitioners_store: Any | None = None,
) -> AttachAndExtractResult:
    """Persist a clinical document to OpenEMR + extract its structured
    contents via Claude vision.

    Args:
        file_bytes: Raw file bytes (PDF or image).
        filename: Original filename, embedded in the OpenEMR upload
            (with the SHA-256 prefix added by the writer for idempotency).
        doc_type: Which schema/tool to force the model into. One of
            `lab_pdf`, `intake_form`, or `referral_letter`.
        patient_uuid: FHIR Patient UUID the doc attaches to.
        mime_type: MIME of the upload. Defaults to `application/pdf`.
            For PNG/JPEG uploads the caller should set this explicitly.
        writer: An OpenEMRWriter to persist the source document. Defaults
            to a fresh client; tests pass a fake.
        anthropic_client: Anthropic SDK client. Defaults to
            `AsyncAnthropic()`; tests pass a fake.
        model: Claude model id. Defaults to `claude-sonnet-4-6` (matches
            the Week 1 chart-summarizer default).

    Returns:
        AttachAndExtractResult with `.extracted` (LabReport | IntakeForm |
        ReferralLetter) and `.write_result` (the dict returned by the writer).

    Raises:
        OpenEMRWriteError on writer failure (auth, upload, FHIR-GET).
        ValueError on unsupported MIME or empty file bytes.
        ExtractionError on Claude refusing the tool call, calling the
            wrong tool, or returning arguments that fail validation.
    """
    if not file_bytes:
        raise ValueError("attach_and_extract: file_bytes is empty")

    own_writer = writer is None
    w = writer if writer is not None else OpenEMRWriter()
    try:
        # Phase 1 — persist source. Done first so the extraction's
        # `source_document_id` is a real DocumentReference id.
        write_result = await w.write_document_reference(
            patient_uuid=patient_uuid,
            doc_type=doc_type,
            file_bytes=file_bytes,
            filename=filename,
            mime_type=mime_type,
        )
        source_document_id = str(write_result["reference_id"])

        # Caller is interposing a pre-extraction check (typically the
        # Layer-1.5 PDF text-fingerprint dedup) and will run Phase 2 +
        # Phase 3 itself if the check passes. We bail out here so a
        # confirmed text-fingerprint hit doesn't pay for a vision call.
        if skip_extraction:
            return AttachAndExtractResult(
                extracted=None, write_result=write_result, persistence=None,
            )

        # Phase 2 — render + extract. Renderer will raise ValueError for
        # unsupported MIMEs; let it propagate (caller's bug to fix).
        page_pngs = render_to_png_pages(file_bytes, mime_type)
        log.info(
            "attach_and_extract: rendered %d page(s) for doc_type=%s "
            "source_id=%s patient=%s created=%s",
            len(page_pngs), doc_type, source_document_id,
            patient_uuid, write_result["created"],
        )
        extracted = await extract_via_claude(
            page_pngs=page_pngs,
            doc_type=doc_type,
            source_document_id=source_document_id,
            client=anthropic_client,
            model=model,
        )

        # Phase 3 — persist extracted facts into OpenEMR's native tables
        # so the chat agent can read them back via the existing chart
        # surface (allergies/meds/problems on the patient card; lab
        # values via the latest-encounter SOAP note returned by
        # get_notes_24h). Phase-3 failures DO NOT roll back Phase 1/2
        # — the source document is already in OpenEMR and the structured
        # JSON is valid; per-fact write failures are recorded in the
        # `persistence` summary and surfaced in the API response. The
        # alternative (raise on first failure) would leave the source
        # document orphaned and force the user to re-upload.
        #
        # `skip_persistence=True` defers Phase 3 so the caller can run
        # post-extraction checks (dedup Layer 2 — content fingerprint)
        # before deciding whether to persist facts. The caller must
        # either invoke `persist_extracted_facts` themselves once a
        # decision is reached, or accept that no facts have been
        # written (in which case the new DocumentReference is harmless
        # but orphan; soft-hiding it is the recommended cancel path).
        if skip_persistence:
            return AttachAndExtractResult(
                extracted=extracted, write_result=write_result,
                persistence=None,
            )
        persistence = await persist_extracted_facts(
            writer=w,
            patient_uuid=patient_uuid,
            extracted=extracted,
            source_document_id=source_document_id,
            practitioners_store=practitioners_store,
        )
        return AttachAndExtractResult(
            extracted=extracted, write_result=write_result,
            persistence=persistence,
        )
    finally:
        # Only close the writer if we created it — caller-supplied
        # writers are owned by the caller.
        if own_writer:
            await w.aclose()


async def attach_and_extract_hl7(
    *,
    file_bytes: bytes,
    filename: str,
    patient_uuid: str,
    writer: OpenEMRWriter | None = None,
    skip_persistence: bool = False,
) -> AttachAndExtractResult:
    """Persist a .hl7 file to OpenEMR + parse its structured contents.

    Mirrors `attach_and_extract` for HL7 v2 messages. The two phases are:

    1. **Persist source.** `write_document_reference` uploads the .hl7
       bytes and returns a `DocumentReference/{uuid}`. SHA-256 idempotency
       still applies — re-uploading the same bytes returns the existing id.

    2. **Parse.** No vision call; the parser walks the segments
       deterministically and returns a typed `Hl7Message`. Citations are
       segment-indexed and `bbox=None` everywhere — HL7 has no spatial
       layout.

    Phase 3 (persistence into native tables) reuses the shared
    `persist_extracted_facts` dispatcher; the HL7 branch maps EVN to a
    medical_problem, AL1 to allergies, and each OBR group to a separate
    Encounter.

    Args:
        file_bytes: Raw .hl7 bytes (typically a few KB of pipe-delimited
            text).
        filename: Original filename, threaded through to the writer's
            idempotency-prefixing logic.
        patient_uuid: FHIR Patient UUID the message attaches to.
        writer: Optional pre-instantiated OpenEMRWriter; defaults to a
            fresh client. Tests pass a fake.
        skip_persistence: When True, return after parsing without
            invoking `persist_extracted_facts`. The caller is responsible
            for the Phase-3 decision (e.g., dedup Layer-2 prompt before
            commit).

    Returns:
        AttachAndExtractResult with `.extracted` (Hl7Message) and
        `.write_result` (the writer dict).

    Raises:
        OpenEMRWriteError on writer failure.
        Hl7ParseError on missing MSH, unsupported MSH-9, or schema
            validation failure on the parsed message.
        ValueError on empty file bytes.
    """
    if not file_bytes:
        raise ValueError("attach_and_extract_hl7: file_bytes is empty")

    own_writer = writer is None
    w = writer if writer is not None else OpenEMRWriter()
    try:
        # Phase 1 — persist the .hl7 source. text/plain is the safest
        # MIME for OpenEMR's binary storage; the FHIR Binary endpoint
        # streams it back as-is when the source modal opens it later.
        write_result = await w.write_document_reference(
            patient_uuid=patient_uuid,
            doc_type="hl7_message",
            file_bytes=file_bytes,
            filename=filename,
            mime_type="text/plain",
        )
        source_document_id = str(write_result["reference_id"])

        # Phase 2 — parse the HL7 message. Hl7ParseError already
        # subclasses ValueError, so FastAPI's standard error mapping
        # surfaces it as a 400.
        try:
            extracted = parse_hl7_message(
                file_bytes,
                source_document_id=source_document_id,
            )
        except Hl7ParseError:
            raise
        log.info(
            "attach_and_extract_hl7: parsed %s msg=%s panels=%d allergies=%d "
            "patient=%s created=%s",
            extracted.message_type,
            extracted.message_control_id,
            len(extracted.lab_panels),
            len(extracted.allergies),
            patient_uuid, write_result["created"],
        )

        if skip_persistence:
            return AttachAndExtractResult(
                extracted=extracted, write_result=write_result,
                persistence=None,
            )
        persistence = await persist_extracted_facts(
            writer=w,
            patient_uuid=patient_uuid,
            extracted=extracted,
            source_document_id=source_document_id,
        )
        return AttachAndExtractResult(
            extracted=extracted, write_result=write_result,
            persistence=persistence,
        )
    finally:
        if own_writer:
            await w.aclose()


async def attach_and_extract_workbook(
    *,
    file_bytes: bytes,
    filename: str,
    patient_uuid: str,
    writer: OpenEMRWriter | None = None,
    skip_persistence: bool = False,
) -> AttachAndExtractResult:
    """Persist a .xlsx workbook to OpenEMR + parse its sheets.

    Mirrors `attach_and_extract_hl7` for the workbook (clinical
    dashboard) doc type. The two phases are:

    1. **Persist source.** `write_document_reference` uploads the .xlsx
       bytes and returns a `DocumentReference/{uuid}`. SHA-256
       idempotency still applies.

    2. **Parse.** No vision call; openpyxl walks the four sheets
       (Patient / Medications / Labs_Trend / Care_Gaps) deterministically
       and returns a typed `Workbook`. Citations are sheet/row-indexed
       and `bbox=None` everywhere.

    Phase 3 (persistence) reuses `persist_extracted_facts`. The workbook
    branch writes one Encounter per Labs_Trend collection-date column
    (preserves the trend on the chart timeline) and writes only OVERDUE
    Care_Gaps rows as medical_problems (UP TO DATE rows stay reachable
    via document-text retrieval but don't pollute Active Problems).
    """
    if not file_bytes:
        raise ValueError("attach_and_extract_workbook: file_bytes is empty")

    own_writer = writer is None
    w = writer if writer is not None else OpenEMRWriter()
    try:
        # Phase 1 — persist the source. Use the OOXML MIME so the FHIR
        # Binary endpoint stores the workbook with its real content type;
        # the source-modal can't natively render xlsx, but a future
        # in-browser preview path is unblocked.
        write_result = await w.write_document_reference(
            patient_uuid=patient_uuid,
            doc_type="workbook",
            file_bytes=file_bytes,
            filename=filename,
            mime_type=(
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            ),
        )
        source_document_id = str(write_result["reference_id"])

        # Phase 2 — parse. WorkbookParseError subclasses ValueError so
        # FastAPI's standard error mapping returns a 400.
        try:
            extracted = parse_workbook(
                file_bytes,
                source_document_id=source_document_id,
            )
        except WorkbookParseError:
            raise
        log.info(
            "attach_and_extract_workbook: meds=%d lab_tests=%d "
            "lab_dated_values=%d care_gaps=%d patient=%s created=%s",
            len(extracted.medications),
            len(extracted.lab_trends),
            sum(len(t.values) for t in extracted.lab_trends),
            len(extracted.care_gaps),
            patient_uuid, write_result["created"],
        )

        if skip_persistence:
            return AttachAndExtractResult(
                extracted=extracted, write_result=write_result,
                persistence=None,
            )
        persistence = await persist_extracted_facts(
            writer=w,
            patient_uuid=patient_uuid,
            extracted=extracted,
            source_document_id=source_document_id,
        )
        return AttachAndExtractResult(
            extracted=extracted, write_result=write_result,
            persistence=persistence,
        )
    finally:
        if own_writer:
            await w.aclose()


# HL7 v2 AL1-4 severity codes -> OpenEMR's lowercase severity vocabulary.
# Codes outside this map (or absent) become None; the writer's `severity`
# parameter is optional and untyped on the OpenEMR side.
_HL7_SEVERITY_TO_OEMR: dict[str, str] = {
    "SV": "severe",
    "MO": "moderate",
    "MI": "mild",
}


async def persist_extracted_facts(
    *,
    writer: OpenEMRWriter,
    patient_uuid: str,
    extracted: ExtractedDocument,
    source_document_id: str,
    practitioners_store: Any | None = None,
) -> dict[str, Any]:
    """Push the structured Phase-2 output into OpenEMR's native tables.

    Dispatches by document type:

    - **LabReport** -> one encounter + one SOAP note (objective field
      contains the rendered lab values; subjective field carries the
      bbox manifest for the future PDF-overlay UI). All results land in
      a single encounter so the chat agent's `get_notes_24h` returns
      them as a single coherent block instead of N tiny encounters.

    - **IntakeForm** -> one POST per allergy / medication /
      family-history entry into the corresponding OpenEMR list. The
      intake's `chief_concern` is also written as a medical_problem so
      it appears in the chart's Active Problems section. Demographics
      are NOT written: the patient already exists (we used their UUID
      to upload), and re-writing demographics would risk overwriting
      whatever is already on file with whatever the VLM read off the
      paper form.

    Returns a per-fact summary:
        {
            "doc_type": "lab_pdf" | "intake_form",
            "facts_attempted": int,
            "facts_written": int,
            "items": [{kind, ok, id|error}, ...],
        }

    Failures are caught per-fact and reported in `items` rather than
    raised. The caller can render a partial-success message; nothing
    is rolled back.
    """
    items: list[dict[str, Any]] = []

    if isinstance(extracted, LabReport):
        # All lab results -> one encounter + SOAP note.
        try:
            results_payload = [
                r.model_dump(mode="json") for r in extracted.results
            ]
            result = await writer.write_lab_encounter_with_results(
                patient_uuid=patient_uuid,
                results=results_payload,
                source_doc_id=source_document_id,
                encounter_date=(
                    str(extracted.results[0].collection_date)
                    if extracted.results else None
                ),
            )
            items.append({
                "kind": "lab_encounter",
                "ok": True,
                "id": result.get("encounter_id"),
                "result_count": result.get("result_count"),
            })
        except (OpenEMRWriteError, Exception) as e:  # noqa: BLE001
            log.exception("persist: lab encounter write failed")
            items.append({"kind": "lab_encounter", "ok": False, "error": str(e)[:300]})

        return {
            "doc_type": "lab_pdf",
            "facts_attempted": 1,
            "facts_written": sum(1 for it in items if it["ok"]),
            "items": items,
        }

    if isinstance(extracted, IntakeForm):
        # Chief concern -> medical_problem so it shows in the chart's
        # Active Problems list. Note: the IntakeForm schema does NOT
        # carry a per-field citation for `chief_concern` (it's a bare
        # str). Reusing demographics.source_citation.bbox would point
        # Sunday's PDF-overlay UI at the wrong region (the top-of-form
        # name field, not the chief-concern line). Pass bbox=None — the
        # back-reference still links the row to the source document, the
        # UI just won't have a precise rectangle for this field. Honest
        # > misleading.
        if extracted.chief_concern:
            try:
                r = await writer.write_medical_problem(
                    patient_uuid=patient_uuid,
                    title=extracted.chief_concern,
                    source_doc_id=source_document_id,
                    bbox=None,
                )
                items.append({"kind": "chief_concern", "ok": True, "id": r.get("uuid") or r.get("id")})
            except (OpenEMRWriteError, Exception) as e:  # noqa: BLE001
                log.exception("persist: chief_concern write failed")
                items.append({"kind": "chief_concern", "ok": False, "error": str(e)[:300]})

        for allergy in extracted.allergies:
            try:
                r = await writer.write_allergy(
                    patient_uuid=patient_uuid,
                    substance=allergy.substance,
                    reaction=allergy.reaction,
                    severity=allergy.severity,
                    source_doc_id=source_document_id,
                    bbox=(allergy.source_citation.bbox.model_dump()
                          if allergy.source_citation.bbox else None),
                )
                items.append({
                    "kind": "allergy", "ok": True,
                    "label": allergy.substance,
                    "id": r.get("uuid") or r.get("id"),
                })
            except (OpenEMRWriteError, Exception) as e:  # noqa: BLE001
                log.exception("persist: allergy[%s] write failed", allergy.substance)
                items.append({
                    "kind": "allergy", "ok": False,
                    "label": allergy.substance, "error": str(e)[:300],
                })

        for med in extracted.current_medications:
            # Render the prescription line in OpenEMR's expected
            # free-text shape: "Drug 500 mg PO BID".
            parts = [med.name]
            if med.dose:
                parts.append(med.dose)
            if med.route:
                parts.append(med.route)
            if med.frequency:
                parts.append(med.frequency)
            title = " ".join(parts)
            try:
                r = await writer.write_medication(
                    patient_uuid=patient_uuid,
                    title=title,
                    source_doc_id=source_document_id,
                    bbox=(med.source_citation.bbox.model_dump()
                          if med.source_citation.bbox else None),
                )
                items.append({
                    "kind": "medication", "ok": True,
                    "label": title,
                    "id": r.get("uuid") or r.get("id"),
                })
            except (OpenEMRWriteError, Exception) as e:  # noqa: BLE001
                log.exception("persist: medication[%s] write failed", title)
                items.append({
                    "kind": "medication", "ok": False,
                    "label": title, "error": str(e)[:300],
                })

        for fh in extracted.family_history:
            # Render as: "Family history: mother — type 2 diabetes (onset 45)"
            label_parts = [f"Family history ({fh.relation}): {fh.condition}"]
            if fh.age_at_onset is not None:
                label_parts.append(f"onset age {fh.age_at_onset}")
            title = " — ".join(label_parts)
            try:
                r = await writer.write_medical_problem(
                    patient_uuid=patient_uuid,
                    title=title,
                    source_doc_id=source_document_id,
                    bbox=(fh.source_citation.bbox.model_dump()
                          if fh.source_citation.bbox else None),
                )
                items.append({
                    "kind": "family_history", "ok": True,
                    "label": title,
                    "id": r.get("uuid") or r.get("id"),
                })
            except (OpenEMRWriteError, Exception) as e:  # noqa: BLE001
                log.exception("persist: family_history[%s] write failed", title)
                items.append({
                    "kind": "family_history", "ok": False,
                    "label": title, "error": str(e)[:300],
                })

        return {
            "doc_type": "intake_form",
            "facts_attempted": len(items),
            "facts_written": sum(1 for it in items if it["ok"]),
            "items": items,
        }

    if isinstance(extracted, ReferralLetter):
        # Referring physician -> Co-Pilot-side care-team store. OpenEMR's
        # FHIR has no native target for the contact-block fields a
        # referral letter prints (specialty / phone / address) and its
        # CareTeam resource is rarely populated for demo patients. The
        # Modern Dashboard's Care Team tab reads from this store. Write
        # is best-effort; failures are recorded but never abort the rest
        # of the persistence pass (the chart still gets problems / meds /
        # allergies even if the store is unreachable).
        rp = extracted.referring_physician
        if practitioners_store is not None and rp.name:
            try:
                practitioners_store.upsert(
                    patient_uuid=patient_uuid,
                    source_doc_id=source_document_id,
                    name=rp.name,
                    practice=rp.practice,
                    specialty=rp.specialty,
                    phone=rp.phone,
                    address=rp.address,
                    npi=rp.npi,
                )
                items.append({
                    "kind": "referring_physician", "ok": True,
                    "label": rp.name,
                })
            except Exception as e:  # noqa: BLE001
                log.exception("persist: referring_physician store write failed")
                items.append({
                    "kind": "referring_physician", "ok": False,
                    "label": rp.name, "error": str(e)[:300],
                })

        # Reason-for-referral -> medical_problem so it surfaces on the
        # chart's Active Problems list (mirrors how chief_concern is
        # handled for IntakeForm). The referral's reason is the doctor's
        # framing of why the patient was sent — semantically a problem
        # the receiving practice is being asked to address.
        if extracted.reason_for_referral:
            try:
                title = f"Reason for referral: {extracted.reason_for_referral}"
                # OpenEMR truncates long titles silently; keep this
                # bounded so the chart row stays readable.
                title = title[:500]
                r = await writer.write_medical_problem(
                    patient_uuid=patient_uuid,
                    title=title,
                    source_doc_id=source_document_id,
                    bbox=None,
                )
                items.append({
                    "kind": "reason_for_referral", "ok": True,
                    "id": r.get("uuid") or r.get("id"),
                })
            except (OpenEMRWriteError, Exception) as e:  # noqa: BLE001
                log.exception("persist: reason_for_referral write failed")
                items.append({
                    "kind": "reason_for_referral", "ok": False,
                    "error": str(e)[:300],
                })

        for cond in extracted.past_medical_history:
            # Render as "Type 2 diabetes mellitus (E11.9)" when the code
            # is printed, else just the name. Free-text title because
            # OpenEMR's medical_problem writer doesn't take a separate
            # ICD-10 field on this code path.
            title = cond.name
            if cond.icd10_code:
                title = f"{cond.name} ({cond.icd10_code})"
            try:
                r = await writer.write_medical_problem(
                    patient_uuid=patient_uuid,
                    title=title,
                    source_doc_id=source_document_id,
                    bbox=(cond.source_citation.bbox.model_dump()
                          if cond.source_citation.bbox else None),
                )
                items.append({
                    "kind": "past_medical_history", "ok": True,
                    "label": title,
                    "id": r.get("uuid") or r.get("id"),
                })
            except (OpenEMRWriteError, Exception) as e:  # noqa: BLE001
                log.exception("persist: past_medical_history[%s] write failed", title)
                items.append({
                    "kind": "past_medical_history", "ok": False,
                    "label": title, "error": str(e)[:300],
                })

        for allergy in extracted.allergies:
            try:
                r = await writer.write_allergy(
                    patient_uuid=patient_uuid,
                    substance=allergy.substance,
                    reaction=allergy.reaction,
                    severity=allergy.severity,
                    source_doc_id=source_document_id,
                    bbox=(allergy.source_citation.bbox.model_dump()
                          if allergy.source_citation.bbox else None),
                )
                items.append({
                    "kind": "allergy", "ok": True,
                    "label": allergy.substance,
                    "id": r.get("uuid") or r.get("id"),
                })
            except (OpenEMRWriteError, Exception) as e:  # noqa: BLE001
                log.exception("persist: allergy[%s] write failed", allergy.substance)
                items.append({
                    "kind": "allergy", "ok": False,
                    "label": allergy.substance, "error": str(e)[:300],
                })

        for med in extracted.current_medications:
            parts = [med.name]
            if med.dose:
                parts.append(med.dose)
            if med.route:
                parts.append(med.route)
            if med.frequency:
                parts.append(med.frequency)
            title = " ".join(parts)
            try:
                r = await writer.write_medication(
                    patient_uuid=patient_uuid,
                    title=title,
                    source_doc_id=source_document_id,
                    bbox=(med.source_citation.bbox.model_dump()
                          if med.source_citation.bbox else None),
                )
                items.append({
                    "kind": "medication", "ok": True,
                    "label": title,
                    "id": r.get("uuid") or r.get("id"),
                })
            except (OpenEMRWriteError, Exception) as e:  # noqa: BLE001
                log.exception("persist: medication[%s] write failed", title)
                items.append({
                    "kind": "medication", "ok": False,
                    "label": title, "error": str(e)[:300],
                })

        # `pertinent_labs` is intentionally NOT persisted as a separate
        # Encounter — see ReferralLetter docstring. The labs ride along
        # on the source DocumentReference; the chat agent surfaces them
        # via document text retrieval rather than via get_notes_24h.

        return {
            "doc_type": "referral_letter",
            "facts_attempted": len(items),
            "facts_written": sum(1 for it in items if it["ok"]),
            "items": items,
        }

    if isinstance(extracted, Hl7Message):
        # ADT-A08 path: EVN reason note becomes a medical_problem so it
        # surfaces on the chart's Active Problems list (mirrors how
        # chief_concern / reason_for_referral are handled). AL1 segments
        # become allergies. ORU-R01 path: each OBR group becomes a
        # separate Encounter (one panel = one encounter) so panel-level
        # LOINC + collection_date + interpretation note stay attached to
        # the right order.
        if extracted.message_type == "ADT^A08":
            if extracted.event_reason:
                title = f"ADT-A08 reason: {extracted.event_reason}"[:500]
                try:
                    r = await writer.write_medical_problem(
                        patient_uuid=patient_uuid,
                        title=title,
                        source_doc_id=source_document_id,
                        bbox=None,
                    )
                    items.append({
                        "kind": "hl7_event_reason", "ok": True,
                        "id": r.get("uuid") or r.get("id"),
                    })
                except (OpenEMRWriteError, Exception) as e:  # noqa: BLE001
                    log.exception("persist: hl7 event_reason write failed")
                    items.append({
                        "kind": "hl7_event_reason", "ok": False,
                        "error": str(e)[:300],
                    })

            for allergy in extracted.allergies:
                severity = _HL7_SEVERITY_TO_OEMR.get(allergy.severity or "")
                try:
                    r = await writer.write_allergy(
                        patient_uuid=patient_uuid,
                        substance=allergy.substance,
                        reaction=allergy.reaction,
                        severity=severity,
                        source_doc_id=source_document_id,
                        bbox=None,
                    )
                    items.append({
                        "kind": "allergy", "ok": True,
                        "label": allergy.substance,
                        "id": r.get("uuid") or r.get("id"),
                    })
                except (OpenEMRWriteError, Exception) as e:  # noqa: BLE001
                    log.exception("persist: hl7 allergy[%s] write failed", allergy.substance)
                    items.append({
                        "kind": "allergy", "ok": False,
                        "label": allergy.substance,
                        "error": str(e)[:300],
                    })

        elif extracted.message_type == "ORU^R01":
            for panel_idx, panel in enumerate(extracted.lab_panels, start=1):
                if not panel.results:
                    log.info("persist: skipping empty OBR[%d] panel", panel_idx)
                    continue
                try:
                    panel_results = [r.model_dump(mode="json") for r in panel.results]
                    encounter_date = (
                        str(panel.collection_date)
                        if panel.collection_date
                        else (str(panel.results[0].collection_date)
                              if panel.results else None)
                    )
                    panel_result = await writer.write_lab_encounter_with_results(
                        patient_uuid=patient_uuid,
                        results=panel_results,
                        source_doc_id=source_document_id,
                        encounter_date=encounter_date,
                    )
                    items.append({
                        "kind": "lab_encounter", "ok": True,
                        "label": panel.panel_name or panel.panel_loinc or f"OBR[{panel_idx}]",
                        "id": panel_result.get("encounter_id"),
                        "result_count": panel_result.get("result_count"),
                    })
                except (OpenEMRWriteError, Exception) as e:  # noqa: BLE001
                    log.exception(
                        "persist: hl7 lab_encounter OBR[%d] write failed",
                        panel_idx,
                    )
                    items.append({
                        "kind": "lab_encounter", "ok": False,
                        "label": panel.panel_name or f"OBR[{panel_idx}]",
                        "error": str(e)[:300],
                    })

                # NTE on this panel is a clinician interpretation line
                # ("BNP elevated — consistent with HFrEF decompensation").
                # Write as a medical_problem so it shows in Active Problems
                # alongside the lab encounter — that's the chart surface
                # the chat agent reads via the standard problem-list tool.
                if panel.notes:
                    panel_label = panel.panel_name or panel.panel_loinc or f"OBR[{panel_idx}]"
                    note_title = f"Lab interpretation ({panel_label}): {panel.notes}"[:500]
                    try:
                        r = await writer.write_medical_problem(
                            patient_uuid=patient_uuid,
                            title=note_title,
                            source_doc_id=source_document_id,
                            bbox=None,
                        )
                        items.append({
                            "kind": "hl7_panel_note", "ok": True,
                            "label": panel_label,
                            "id": r.get("uuid") or r.get("id"),
                        })
                    except (OpenEMRWriteError, Exception) as e:  # noqa: BLE001
                        log.exception(
                            "persist: hl7 panel_note OBR[%d] write failed",
                            panel_idx,
                        )
                        items.append({
                            "kind": "hl7_panel_note", "ok": False,
                            "label": panel_label,
                            "error": str(e)[:300],
                        })

        return {
            "doc_type": "hl7_message",
            "message_type": extracted.message_type,
            "facts_attempted": len(items),
            "facts_written": sum(1 for it in items if it["ok"]),
            "items": items,
        }

    if isinstance(extracted, FaxPacket):
        # Fax packets bundle a referral + face sheet + (optionally) a
        # lab report under one source DocumentReference. We persist the
        # full chart slice — Active Problems, Medications, Allergies —
        # so the chat agent can answer "what does this fax say about
        # Chen's meds?" via the standard chart surface. The bundled lab
        # page persists as a real lab Encounter (the fax IS the source,
        # not a quotation — distinct from `referral_letter`, where
        # embedded labs are quoted-context only).
        if extracted.reason_for_consultation:
            specialty = extracted.receiving_specialty
            spec_suffix = f" → {specialty}" if specialty else ""
            title = f"Fax consult{spec_suffix}: {extracted.reason_for_consultation}"[:500]
            try:
                r = await writer.write_medical_problem(
                    patient_uuid=patient_uuid,
                    title=title,
                    source_doc_id=source_document_id,
                    bbox=None,
                )
                items.append({
                    "kind": "fax_reason_for_consultation", "ok": True,
                    "id": r.get("uuid") or r.get("id"),
                })
            except (OpenEMRWriteError, Exception) as e:  # noqa: BLE001
                log.exception("persist: fax reason_for_consultation write failed")
                items.append({
                    "kind": "fax_reason_for_consultation", "ok": False,
                    "error": str(e)[:300],
                })

        for cond in extracted.active_problems:
            cond_title = cond.name
            if cond.icd10_code:
                cond_title = f"{cond.name} ({cond.icd10_code})"
            try:
                r = await writer.write_medical_problem(
                    patient_uuid=patient_uuid,
                    title=cond_title,
                    source_doc_id=source_document_id,
                    bbox=(cond.source_citation.bbox.model_dump()
                          if cond.source_citation.bbox else None),
                )
                items.append({
                    "kind": "active_problem", "ok": True,
                    "label": cond_title,
                    "id": r.get("uuid") or r.get("id"),
                })
            except (OpenEMRWriteError, Exception) as e:  # noqa: BLE001
                log.exception("persist: fax active_problem[%s] write failed", cond_title)
                items.append({
                    "kind": "active_problem", "ok": False,
                    "label": cond_title, "error": str(e)[:300],
                })

        for allergy in extracted.allergies:
            try:
                r = await writer.write_allergy(
                    patient_uuid=patient_uuid,
                    substance=allergy.substance,
                    reaction=allergy.reaction,
                    severity=allergy.severity,
                    source_doc_id=source_document_id,
                    bbox=(allergy.source_citation.bbox.model_dump()
                          if allergy.source_citation.bbox else None),
                )
                items.append({
                    "kind": "allergy", "ok": True,
                    "label": allergy.substance,
                    "id": r.get("uuid") or r.get("id"),
                })
            except (OpenEMRWriteError, Exception) as e:  # noqa: BLE001
                log.exception("persist: fax allergy[%s] write failed", allergy.substance)
                items.append({
                    "kind": "allergy", "ok": False,
                    "label": allergy.substance, "error": str(e)[:300],
                })

        for med in extracted.current_medications:
            parts = [med.name]
            if med.dose:
                parts.append(med.dose)
            if med.route:
                parts.append(med.route)
            if med.frequency:
                parts.append(med.frequency)
            med_title = " ".join(parts)
            try:
                r = await writer.write_medication(
                    patient_uuid=patient_uuid,
                    title=med_title,
                    source_doc_id=source_document_id,
                    bbox=(med.source_citation.bbox.model_dump()
                          if med.source_citation.bbox else None),
                )
                items.append({
                    "kind": "medication", "ok": True,
                    "label": med_title,
                    "id": r.get("uuid") or r.get("id"),
                })
            except (OpenEMRWriteError, Exception) as e:  # noqa: BLE001
                log.exception("persist: fax medication[%s] write failed", med_title)
                items.append({
                    "kind": "medication", "ok": False,
                    "label": med_title, "error": str(e)[:300],
                })

        # Lab page → real lab Encounter. Confirmed user intent — the fax
        # IS the transmission carrying these results, not a quotation of
        # a separately-issued lab report. One Encounter for all results
        # on the page (no OBR boundaries to preserve, unlike HL7 ORU).
        if extracted.lab_results:
            try:
                lab_payload = [r.model_dump(mode="json") for r in extracted.lab_results]
                encounter_date = (
                    str(extracted.lab_results[0].collection_date)
                    if extracted.lab_results else None
                )
                lab_result = await writer.write_lab_encounter_with_results(
                    patient_uuid=patient_uuid,
                    results=lab_payload,
                    source_doc_id=source_document_id,
                    encounter_date=encounter_date,
                )
                items.append({
                    "kind": "lab_encounter", "ok": True,
                    "id": lab_result.get("encounter_id"),
                    "result_count": lab_result.get("result_count"),
                })
            except (OpenEMRWriteError, Exception) as e:  # noqa: BLE001
                log.exception("persist: fax lab_encounter write failed")
                items.append({
                    "kind": "lab_encounter", "ok": False,
                    "error": str(e)[:300],
                })

        return {
            "doc_type": "fax_packet",
            "facts_attempted": len(items),
            "facts_written": sum(1 for it in items if it["ok"]),
            "items": items,
        }

    if isinstance(extracted, Workbook):
        # Workbook persistence: skip patient demographics (already
        # known); write a single allergy entry only when the cell is
        # non-NKDA non-empty; medications row-by-row; **one Encounter
        # per Labs_Trend collection-date column** (preserves the trend
        # on the chart timeline); only OVERDUE Care_Gaps rows write as
        # medical_problems (UP TO DATE / NOT APPLICABLE entries stay
        # reachable via document-text retrieval).
        allergies_text = (extracted.allergies_text or "").strip()
        if allergies_text and allergies_text.upper() not in ("NKDA", "NONE", "NKA", "NO KNOWN ALLERGIES", "NONE KNOWN", "N/A"):
            try:
                r = await writer.write_allergy(
                    patient_uuid=patient_uuid,
                    substance=allergies_text,
                    reaction=None,
                    severity=None,
                    source_doc_id=source_document_id,
                    bbox=None,
                )
                items.append({
                    "kind": "allergy", "ok": True,
                    "label": allergies_text,
                    "id": r.get("uuid") or r.get("id"),
                })
            except (OpenEMRWriteError, Exception) as e:  # noqa: BLE001
                log.exception("persist: workbook allergies_text write failed")
                items.append({
                    "kind": "allergy", "ok": False,
                    "label": allergies_text, "error": str(e)[:300],
                })

        for med in extracted.medications:
            # Render as: "Drug 40 mg PO daily"
            name = med.generic or med.brand or ""
            if not name:
                continue
            parts = [name]
            if med.strength:
                parts.append(med.strength)
            if med.route:
                parts.append(med.route)
            if med.sig:
                # The sig may already contain the full "1 tab PO daily"
                # string; appending the route+sig together would
                # double up "PO". If sig already mentions the route,
                # skip the route token to avoid the duplication.
                if med.route and med.route in med.sig:
                    parts = [p for p in parts if p != med.route]
                parts.append(med.sig)
            med_title = " ".join(parts)
            try:
                r = await writer.write_medication(
                    patient_uuid=patient_uuid,
                    title=med_title,
                    begdate=str(med.start_date) if med.start_date else None,
                    source_doc_id=source_document_id,
                    bbox=None,
                )
                items.append({
                    "kind": "medication", "ok": True,
                    "label": med_title,
                    "id": r.get("uuid") or r.get("id"),
                })
            except (OpenEMRWriteError, Exception) as e:  # noqa: BLE001
                log.exception("persist: workbook medication[%s] write failed", med_title)
                items.append({
                    "kind": "medication", "ok": False,
                    "label": med_title, "error": str(e)[:300],
                })

        # Pivot Labs_Trend by collection_date: each date column becomes
        # a separate Encounter holding all the test values for that date.
        # This preserves the trend on the chart timeline (one OBR =
        # one Encounter, applied at column granularity).
        from collections import defaultdict
        labs_by_date: dict = defaultdict(list)
        for trend in extracted.lab_trends:
            for v in trend.values:
                # Try numeric first; fall through to qualitative string.
                try:
                    parsed_value: float | str = float(v.value)
                except ValueError:
                    parsed_value = v.value
                # Synthesize a per-cell Citation so the LabResult row
                # points at this exact (test, date) cell rather than
                # the row-level citation on the parent WorkbookLabTrend.
                cell_citation = Citation(
                    source_type="workbook",
                    source_id=source_document_id,
                    page_or_section=f"Labs_Trend col {v.collection_date} row {trend.test_name}",
                    field_or_chunk_id=f"labs_trend.{v.collection_date}.{trend.test_name.replace(' ', '_').lower()}",
                    quote_or_value=v.value,
                    bbox=None,
                )
                labs_by_date[v.collection_date].append(LabResult(
                    test_name=trend.test_name,
                    value=parsed_value,
                    unit=trend.units or "",
                    reference_range=trend.reference_range,
                    collection_date=v.collection_date,
                    abnormal_flag=None,  # workbook trends don't carry per-cell flags
                    source_citation=cell_citation,
                ))

        for collection_date in sorted(labs_by_date.keys()):
            results_for_date = labs_by_date[collection_date]
            try:
                payload = [r.model_dump(mode="json") for r in results_for_date]
                lab_result = await writer.write_lab_encounter_with_results(
                    patient_uuid=patient_uuid,
                    results=payload,
                    source_doc_id=source_document_id,
                    encounter_date=str(collection_date),
                )
                items.append({
                    "kind": "lab_encounter", "ok": True,
                    "label": f"workbook draw {collection_date}",
                    "id": lab_result.get("encounter_id"),
                    "result_count": lab_result.get("result_count"),
                })
            except (OpenEMRWriteError, Exception) as e:  # noqa: BLE001
                log.exception(
                    "persist: workbook lab_encounter for %s write failed",
                    collection_date,
                )
                items.append({
                    "kind": "lab_encounter", "ok": False,
                    "label": f"workbook draw {collection_date}",
                    "error": str(e)[:300],
                })

        # OVERDUE care gaps -> medical_problem (only).
        for gap in extracted.care_gaps:
            if gap.status != "OVERDUE":
                continue
            since_clause = f" since {gap.last_done}" if gap.last_done else ""
            note_clause = f" — {gap.notes}" if gap.notes else ""
            gap_title = f"Care gap (overdue{since_clause}): {gap.measure}{note_clause}"[:500]
            try:
                r = await writer.write_medical_problem(
                    patient_uuid=patient_uuid,
                    title=gap_title,
                    source_doc_id=source_document_id,
                    bbox=None,
                )
                items.append({
                    "kind": "care_gap_overdue", "ok": True,
                    "label": gap.measure,
                    "id": r.get("uuid") or r.get("id"),
                })
            except (OpenEMRWriteError, Exception) as e:  # noqa: BLE001
                log.exception("persist: workbook care_gap[%s] write failed", gap.measure)
                items.append({
                    "kind": "care_gap_overdue", "ok": False,
                    "label": gap.measure, "error": str(e)[:300],
                })

        return {
            "doc_type": "workbook",
            "facts_attempted": len(items),
            "facts_written": sum(1 for it in items if it["ok"]),
            "items": items,
        }

    # Unknown discriminator value — should be impossible under
    # ExtractedDocument's discriminated union, but guard for forward-
    # compat with new doc types.
    return {
        "doc_type": "unknown",
        "facts_attempted": 0,
        "facts_written": 0,
        "items": [],
    }


async def extract_only(
    *,
    file_bytes: bytes,
    doc_type: DocumentType,
    mime_type: str = "application/pdf",
    anthropic_client: AsyncAnthropic | None = None,
    model: str = DEFAULT_MODEL,
) -> "ExtractedDocument":
    """Run Claude vision extraction on a file WITHOUT touching OpenEMR.

    Use this when you need the structured extraction but don't yet have
    a patient to attach to (the create-patient-from-upload flow). No
    DocumentReference is created; nothing is persisted. The
    `source_document_id` baked into citations is the placeholder
    "pending" — callers should not feed an `extract_only` result into
    `persist_extracted_facts` without first re-running with a real
    DocumentReference id.

    Returns an `ExtractedDocument` (LabReport | IntakeForm).
    """
    if not file_bytes:
        raise ValueError("extract_only: file_bytes is empty")
    page_pngs = render_to_png_pages(file_bytes, mime_type)
    log.info(
        "extract_only: rendered %d page(s) for doc_type=%s (no persistence)",
        len(page_pngs), doc_type,
    )
    return await extract_via_claude(
        page_pngs=page_pngs,
        doc_type=doc_type,
        source_document_id="pending",
        client=anthropic_client,
        model=model,
    )


__all__ = [
    "AttachAndExtractResult",
    "ExtractionError",
    "attach_and_extract",
    "extract_only",
    "persist_extracted_facts",
]
