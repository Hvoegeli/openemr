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
from app.extraction.schemas import DocumentType, ExtractedDocument
from app.extraction.vision import DEFAULT_MODEL, ExtractionError, extract_via_claude
from app.fhir.writer import OpenEMRWriter

log = logging.getLogger("agent.extraction.extract")


class AttachAndExtractResult:
    """Lightweight value-object holding both halves of the pipeline output.

    Using a class (not a TypedDict / dict) so the IDE / type-checker
    can see that `extracted` is `ExtractedDocument` and `write_result`
    is a real dict shape from the writer — not just `Any`.
    """

    __slots__ = ("extracted", "write_result")

    def __init__(
        self,
        extracted: ExtractedDocument,
        write_result: dict[str, Any],
    ) -> None:
        self.extracted = extracted
        self.write_result = write_result

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
        return AttachAndExtractResult(
            extracted=extracted, write_result=write_result,
        )
    finally:
        # Only close the writer if we created it — caller-supplied
        # writers are owned by the caller.
        if own_writer:
            await w.aclose()


__all__ = [
    "AttachAndExtractResult",
    "ExtractionError",
    "attach_and_extract",
]
