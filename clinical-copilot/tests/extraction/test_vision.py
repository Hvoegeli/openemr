"""Isolated tests for the Claude vision extractor.

The Anthropic SDK is fully mocked — these tests never make a real HTTP
call and never burn API credits. The integration smoke
(`scripts/smoke_extract.py`) exercises the live API path separately.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from anthropic.types import Message, TextBlock, ToolUseBlock

from app.extraction.schemas import IntakeForm, LabReport
from app.extraction.vision import (
    ExtractionError,
    _user_content_blocks,
    extract_via_claude,
)


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
LAB_FIXTURE_PATH = (
    REPO_ROOT / "data" / "demo_documents" / "fixtures" / "cohen_lab_2026-04-30.expected.json"
)
INTAKE_FIXTURE_PATH = (
    REPO_ROOT / "data" / "demo_documents" / "fixtures" / "cohen_intake_2026-04-30.expected.json"
)
SOURCE_DOC_ID = "DocumentReference/test-doc-abc-123"


def _load_lab_payload() -> dict[str, Any]:
    """Load the Cohen lab fixture and rewrite source_document_id /
    citation source_ids to a test value, so the fake-Claude response
    looks like a real extraction targeted at SOURCE_DOC_ID."""
    payload = json.loads(LAB_FIXTURE_PATH.read_text())
    payload["source_document_id"] = SOURCE_DOC_ID
    for r in payload["results"]:
        r["source_citation"]["source_id"] = SOURCE_DOC_ID
    return payload


def _load_intake_payload() -> dict[str, Any]:
    payload = json.loads(INTAKE_FIXTURE_PATH.read_text())
    payload["source_document_id"] = SOURCE_DOC_ID
    payload["demographics"]["source_citation"]["source_id"] = SOURCE_DOC_ID
    for k in ("current_medications", "allergies", "family_history"):
        for item in payload[k]:
            item["source_citation"]["source_id"] = SOURCE_DOC_ID
    return payload


def _fake_message(blocks: list[Any], stop_reason: str = "tool_use") -> Message:
    """Build a real `anthropic.types.Message` (not a MagicMock) so the
    type checks inside `extract_via_claude` see actual ToolUseBlock /
    TextBlock instances and behave the way they would in production."""
    return Message(
        id="msg_test",
        type="message",
        role="assistant",
        model="claude-sonnet-4-6",
        content=blocks,
        stop_reason=stop_reason,  # type: ignore[arg-type]
        stop_sequence=None,
        usage={"input_tokens": 10, "output_tokens": 10},  # type: ignore[arg-type]
    )


class _FakeAnthropicClient:
    """Stand-in for `anthropic.AsyncAnthropic`. Records the kwargs of the
    last `messages.create` call and returns a canned response. Letting
    tests inspect what the extractor sent to Claude (model, tools,
    tool_choice, message content) is what makes contract assertions
    possible without a live API."""

    def __init__(self, response: Message) -> None:
        self._response = response
        self.last_kwargs: dict[str, Any] | None = None
        self.messages = self  # mimic SDK shape: client.messages.create

    async def create(self, **kwargs: Any) -> Message:
        self.last_kwargs = kwargs
        return self._response


# A dummy 1x1 PNG so the extractor's "non-empty pages" check passes. The
# fake client never decodes the image — it only reads the kwargs.
_DUMMY_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc"
    b"\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
)


class TestUserContentBlocks:
    def test_image_block_per_page_then_text(self) -> None:
        blocks = _user_content_blocks(
            [_DUMMY_PNG, _DUMMY_PNG],
            doc_type="lab_pdf",
            source_document_id=SOURCE_DOC_ID,
        )
        # 2 image blocks + 1 text block
        assert len(blocks) == 3
        assert blocks[0]["type"] == "image"
        assert blocks[1]["type"] == "image"
        assert blocks[2]["type"] == "text"
        assert SOURCE_DOC_ID in blocks[2]["text"]
        assert "lab_pdf" in blocks[2]["text"]

    def test_image_block_uses_base64_png(self) -> None:
        blocks = _user_content_blocks(
            [_DUMMY_PNG], doc_type="lab_pdf", source_document_id=SOURCE_DOC_ID,
        )
        src = blocks[0]["source"]
        assert src["type"] == "base64"
        assert src["media_type"] == "image/png"
        # Round-trip the base64 to make sure it actually encoded the bytes
        import base64
        assert base64.standard_b64decode(src["data"]) == _DUMMY_PNG


class TestExtractViaClaude:
    @pytest.mark.asyncio
    async def test_lab_pdf_happy_path_returns_lab_report(self) -> None:
        payload = _load_lab_payload()
        fake = _FakeAnthropicClient(_fake_message([
            ToolUseBlock(
                type="tool_use",
                id="toolu_test",
                name="record_lab_report",
                input=payload,
            ),
        ]))
        result = await extract_via_claude(
            page_pngs=[_DUMMY_PNG],
            doc_type="lab_pdf",
            source_document_id=SOURCE_DOC_ID,
            client=fake,  # type: ignore[arg-type]
        )
        assert isinstance(result, LabReport)
        assert result.source_document_id == SOURCE_DOC_ID
        assert len(result.results) == 10  # matches the fixture
        # Verify the extractor sent the right tool definition and forced
        # tool_choice — these are correctness invariants, not impl details.
        assert fake.last_kwargs is not None
        assert fake.last_kwargs["tool_choice"] == {
            "type": "tool", "name": "record_lab_report",
        }
        assert len(fake.last_kwargs["tools"]) == 1
        assert fake.last_kwargs["tools"][0]["name"] == "record_lab_report"

    @pytest.mark.asyncio
    async def test_intake_form_happy_path_returns_intake_form(self) -> None:
        payload = _load_intake_payload()
        fake = _FakeAnthropicClient(_fake_message([
            ToolUseBlock(
                type="tool_use",
                id="toolu_test",
                name="record_intake_form",
                input=payload,
            ),
        ]))
        result = await extract_via_claude(
            page_pngs=[_DUMMY_PNG],
            doc_type="intake_form",
            source_document_id=SOURCE_DOC_ID,
            client=fake,  # type: ignore[arg-type]
        )
        assert isinstance(result, IntakeForm)
        assert result.source_document_id == SOURCE_DOC_ID
        assert result.demographics.given_name == "Nora"
        assert fake.last_kwargs is not None
        assert fake.last_kwargs["tool_choice"]["name"] == "record_intake_form"

    @pytest.mark.asyncio
    async def test_no_tool_call_raises(self) -> None:
        # Model returned only text (refused or rambled); we should not
        # silently accept that as success.
        fake = _FakeAnthropicClient(_fake_message(
            [TextBlock(type="text", text="I cannot read this document.")],
            stop_reason="end_turn",
        ))
        with pytest.raises(ExtractionError, match="no record_lab_report tool call"):
            await extract_via_claude(
                page_pngs=[_DUMMY_PNG],
                doc_type="lab_pdf",
                source_document_id=SOURCE_DOC_ID,
                client=fake,  # type: ignore[arg-type]
            )

    @pytest.mark.asyncio
    async def test_wrong_tool_call_raises(self) -> None:
        # Model called the intake tool when we asked for the lab tool.
        # Forcing tool_choice should prevent this in production but the
        # guard rail must still exist.
        fake = _FakeAnthropicClient(_fake_message([
            ToolUseBlock(
                type="tool_use",
                id="toolu_test",
                name="record_intake_form",
                input={},
            ),
        ]))
        with pytest.raises(ExtractionError, match="no record_lab_report tool call"):
            await extract_via_claude(
                page_pngs=[_DUMMY_PNG],
                doc_type="lab_pdf",
                source_document_id=SOURCE_DOC_ID,
                client=fake,  # type: ignore[arg-type]
            )

    @pytest.mark.asyncio
    async def test_invalid_payload_raises(self) -> None:
        # Tool was called but the arguments don't fit the schema (missing
        # required `results` field for a LabReport).
        fake = _FakeAnthropicClient(_fake_message([
            ToolUseBlock(
                type="tool_use",
                id="toolu_test",
                name="record_lab_report",
                input={"source_document_id": SOURCE_DOC_ID},  # no results
            ),
        ]))
        with pytest.raises(ExtractionError, match="failed schema validation"):
            await extract_via_claude(
                page_pngs=[_DUMMY_PNG],
                doc_type="lab_pdf",
                source_document_id=SOURCE_DOC_ID,
                client=fake,  # type: ignore[arg-type]
            )

    @pytest.mark.asyncio
    async def test_empty_page_list_raises(self) -> None:
        fake = _FakeAnthropicClient(_fake_message([]))
        with pytest.raises(ExtractionError, match="no pages to send"):
            await extract_via_claude(
                page_pngs=[],
                doc_type="lab_pdf",
                source_document_id=SOURCE_DOC_ID,
                client=fake,  # type: ignore[arg-type]
            )

    @pytest.mark.asyncio
    async def test_unsupported_doc_type_raises(self) -> None:
        fake = _FakeAnthropicClient(_fake_message([]))
        with pytest.raises(ExtractionError, match="unsupported doc_type"):
            await extract_via_claude(
                page_pngs=[_DUMMY_PNG],
                doc_type="referral_fax",  # type: ignore[arg-type]
                source_document_id=SOURCE_DOC_ID,
                client=fake,  # type: ignore[arg-type]
            )
