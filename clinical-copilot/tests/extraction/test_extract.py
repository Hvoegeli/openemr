"""Isolated tests for `attach_and_extract` orchestration.

Both the writer and the Anthropic client are stubbed out, so these
tests verify the orchestration shape (writer is called first, the
returned reference_id is threaded into the vision call, the result
object exposes the right convenience properties) without any network
or API access.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from anthropic.types import Message, ToolUseBlock

from app.extraction.extract import AttachAndExtractResult, attach_and_extract
from app.extraction.schemas import LabReport


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
COHEN_LAB_PDF = REPO_ROOT / "data" / "demo_documents" / "cohen_lab_2026-04-30.pdf"
LAB_FIXTURE_PATH = (
    REPO_ROOT / "data" / "demo_documents" / "fixtures" / "cohen_lab_2026-04-30.expected.json"
)
TEST_REF_ID = "DocumentReference/test-doc-xyz"
PATIENT_UUID = "test-patient-uuid"


class _FakeWriter:
    """In-memory stand-in for OpenEMRWriter. Captures the kwargs of the
    write_document_reference call so the test can assert the orchestrator
    forwarded the right arguments. `created` is configurable so the
    dedupe path can also be exercised."""

    def __init__(
        self, reference_id: str = TEST_REF_ID, created: bool = True,
    ) -> None:
        self._reference_id = reference_id
        self._created = created
        self.calls: list[dict[str, Any]] = []
        self.aclose_called = False

    async def write_document_reference(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return {
            "reference_id": self._reference_id,
            "sha256": "deadbeef" * 8,
            "created": self._created,
        }

    async def aclose(self) -> None:
        self.aclose_called = True


def _lab_payload_with(source_id: str) -> dict[str, Any]:
    payload = json.loads(LAB_FIXTURE_PATH.read_text())
    payload["source_document_id"] = source_id
    for r in payload["results"]:
        r["source_citation"]["source_id"] = source_id
    return payload


class _FakeAnthropicClient:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload
        self.last_kwargs: dict[str, Any] | None = None
        self.messages = self

    async def create(self, **kwargs: Any) -> Message:
        self.last_kwargs = kwargs
        return Message(
            id="msg_test",
            type="message",
            role="assistant",
            model="claude-sonnet-4-6",
            content=[ToolUseBlock(
                type="tool_use",
                id="toolu_test",
                name="record_lab_report",
                input=self._payload,
            )],
            stop_reason="tool_use",
            stop_sequence=None,
            usage={"input_tokens": 1, "output_tokens": 1},  # type: ignore[arg-type]
        )


class TestAttachAndExtractResult:
    def test_properties_expose_reference_id_and_created(self) -> None:
        # Build a minimal valid LabReport so the type-check is real
        payload = _lab_payload_with(TEST_REF_ID)
        report = LabReport.model_validate(payload)
        r = AttachAndExtractResult(
            extracted=report,
            write_result={
                "reference_id": TEST_REF_ID, "sha256": "abc", "created": True,
            },
        )
        assert r.reference_id == TEST_REF_ID
        assert r.created is True
        assert r.extracted is report

    def test_dedupe_hit_exposed_via_created_property(self) -> None:
        payload = _lab_payload_with(TEST_REF_ID)
        report = LabReport.model_validate(payload)
        r = AttachAndExtractResult(
            extracted=report,
            write_result={
                "reference_id": TEST_REF_ID, "sha256": "abc", "created": False,
            },
        )
        assert r.created is False


class TestAttachAndExtract:
    @pytest.mark.asyncio
    async def test_orchestration_calls_writer_then_vision(self) -> None:
        writer = _FakeWriter()
        anthropic_client = _FakeAnthropicClient(_lab_payload_with(TEST_REF_ID))

        result = await attach_and_extract(
            file_bytes=COHEN_LAB_PDF.read_bytes(),
            filename="cohen_lab.pdf",
            doc_type="lab_pdf",
            patient_uuid=PATIENT_UUID,
            mime_type="application/pdf",
            writer=writer,  # type: ignore[arg-type]
            anthropic_client=anthropic_client,  # type: ignore[arg-type]
        )

        # Writer was called with the right shape
        assert len(writer.calls) == 1
        wc = writer.calls[0]
        assert wc["patient_uuid"] == PATIENT_UUID
        assert wc["doc_type"] == "lab_pdf"
        assert wc["filename"] == "cohen_lab.pdf"
        assert wc["mime_type"] == "application/pdf"
        assert wc["file_bytes"] == COHEN_LAB_PDF.read_bytes()

        # Vision call was made and source_document_id from the writer
        # was threaded into the user message
        assert anthropic_client.last_kwargs is not None
        user_text = anthropic_client.last_kwargs["messages"][0]["content"][-1]["text"]
        assert TEST_REF_ID in user_text

        # Result wraps both halves cleanly
        assert isinstance(result, AttachAndExtractResult)
        assert result.reference_id == TEST_REF_ID
        assert result.created is True
        assert isinstance(result.extracted, LabReport)
        assert result.extracted.source_document_id == TEST_REF_ID

    @pytest.mark.asyncio
    async def test_caller_supplied_writer_is_not_closed(self) -> None:
        # Writers belong to the caller when supplied; orchestrator must
        # not close them or the next call from the same caller would
        # blow up on a closed httpx client.
        writer = _FakeWriter()
        anthropic_client = _FakeAnthropicClient(_lab_payload_with(TEST_REF_ID))
        await attach_and_extract(
            file_bytes=COHEN_LAB_PDF.read_bytes(),
            filename="cohen_lab.pdf",
            doc_type="lab_pdf",
            patient_uuid=PATIENT_UUID,
            mime_type="application/pdf",
            writer=writer,  # type: ignore[arg-type]
            anthropic_client=anthropic_client,  # type: ignore[arg-type]
        )
        assert writer.aclose_called is False

    @pytest.mark.asyncio
    async def test_dedupe_path_returned_as_created_false(self) -> None:
        writer = _FakeWriter(created=False)  # writer reports a dedupe hit
        anthropic_client = _FakeAnthropicClient(_lab_payload_with(TEST_REF_ID))
        result = await attach_and_extract(
            file_bytes=COHEN_LAB_PDF.read_bytes(),
            filename="cohen_lab.pdf",
            doc_type="lab_pdf",
            patient_uuid=PATIENT_UUID,
            mime_type="application/pdf",
            writer=writer,  # type: ignore[arg-type]
            anthropic_client=anthropic_client,  # type: ignore[arg-type]
        )
        assert result.created is False

    @pytest.mark.asyncio
    async def test_empty_file_bytes_raises(self) -> None:
        # We never want to send a zero-byte file to the writer or to
        # Claude — surface this at the entry point with a clear error.
        with pytest.raises(ValueError, match="file_bytes is empty"):
            await attach_and_extract(
                file_bytes=b"",
                filename="empty.pdf",
                doc_type="lab_pdf",
                patient_uuid=PATIENT_UUID,
                writer=_FakeWriter(),  # type: ignore[arg-type]
                anthropic_client=_FakeAnthropicClient({}),  # type: ignore[arg-type]
            )
