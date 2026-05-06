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

from app.extraction.render import render_to_png_pages
from app.extraction.schemas import (
    DocumentType,
    ExtractedDocument,
    IntakeForm,
    LabReport,
)
from app.extraction.vision import DEFAULT_MODEL, ExtractionError, extract_via_claude
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
        extracted: ExtractedDocument,
        write_result: dict[str, Any],
        persistence: dict[str, Any] | None = None,
    ) -> None:
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
    skip_persistence: bool = False,
) -> AttachAndExtractResult:
    """Persist a clinical document to OpenEMR + extract its structured
    contents via Claude vision.

    Args:
        file_bytes: Raw file bytes (PDF or image).
        filename: Original filename, embedded in the OpenEMR upload
            (with the SHA-256 prefix added by the writer for idempotency).
        doc_type: Which schema/tool to force the model into. One of
            `lab_pdf` or `intake_form`.
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
        AttachAndExtractResult with `.extracted` (LabReport | IntakeForm)
        and `.write_result` (the dict returned by the writer).

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


async def persist_extracted_facts(
    *,
    writer: OpenEMRWriter,
    patient_uuid: str,
    extracted: ExtractedDocument,
    source_document_id: str,
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

    # Unknown discriminator value — should be impossible under
    # ExtractedDocument's discriminated union, but guard for forward-
    # compat with new doc types.
    return {
        "doc_type": "unknown",
        "facts_attempted": 0,
        "facts_written": 0,
        "items": [],
    }


__all__ = [
    "AttachAndExtractResult",
    "ExtractionError",
    "attach_and_extract",
    "persist_extracted_facts",
]
