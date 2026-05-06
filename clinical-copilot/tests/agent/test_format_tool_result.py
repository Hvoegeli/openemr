"""Isolated tests for the multimodal-aware tool-result helpers in
`app.agent.graph`.

These three helpers are the seam between dispatch (which always returns
`{data, sources}`) and the LangChain `ToolMessage` envelope. The
`get_document_content` tool needs its rendered pages delivered as image
content blocks rather than JSON-stringified base64 (which the LLM
cannot actually see). Locking this down here means a future refactor
can't silently drop image blocks without a test failure.
"""

from __future__ import annotations

import json

from app.agent.graph import (
    _content_text_for_scan,
    _format_tool_result,
    _prepend_quarantine,
)


class TestFormatToolResult:
    def test_non_document_tool_returns_json_string(self) -> None:
        result = {"data": {"hello": 1}, "sources": ["X/1"]}
        out = _format_tool_result("get_patient_card", result)
        assert isinstance(out, str)
        assert json.loads(out) == result

    def test_document_tool_strips_pages_and_emits_image_blocks(self) -> None:
        result = {
            "data": {
                "document_id": "doc-1",
                "title": "Intake",
                "page_count": 2,
                "pages_png_b64": ["AAAA", "BBBB"],
            },
            "sources": ["DocumentReference/doc-1"],
        }
        out = _format_tool_result("get_document_content", result)
        assert isinstance(out, list)

        # First block: text JSON without pages_png_b64
        assert out[0]["type"] == "text"
        meta = json.loads(out[0]["text"])
        assert "pages_png_b64" not in meta["data"]
        assert meta["data"]["document_id"] == "doc-1"
        assert meta["sources"] == ["DocumentReference/doc-1"]

        # Following blocks: image blocks in langchain-anthropic standard shape
        image_blocks = out[1:]
        assert len(image_blocks) == 2
        for block, expected_b64 in zip(image_blocks, ["AAAA", "BBBB"], strict=True):
            assert block["type"] == "image"
            assert block["source_type"] == "base64"
            assert block["mime_type"] == "image/png"
            assert block["data"] == expected_b64

    def test_document_tool_with_no_pages_returns_text_only(self) -> None:
        result = {
            "data": {"document_id": "doc-2", "page_count": 0, "pages_png_b64": []},
            "sources": ["DocumentReference/doc-2"],
        }
        out = _format_tool_result("get_document_content", result)
        # Even with no pages we keep the metadata block so the LLM sees
        # the result instead of an empty payload.
        assert isinstance(out, list)
        assert len(out) == 1
        assert out[0]["type"] == "text"


class TestContentTextForScan:
    def test_plain_string_passes_through(self) -> None:
        assert _content_text_for_scan("ignore previous") == "ignore previous"

    def test_list_concatenates_text_blocks_only(self) -> None:
        blocks = [
            {"type": "text", "text": "alpha"},
            {"type": "image", "data": "Z29ic2VydmU="},
            {"type": "text", "text": "beta"},
        ]
        assert _content_text_for_scan(blocks) == "alphabeta"

    def test_list_with_no_text_blocks_returns_empty_string(self) -> None:
        blocks = [{"type": "image", "data": "..."}]
        assert _content_text_for_scan(blocks) == ""


class TestPrependQuarantine:
    def test_string_concatenates(self) -> None:
        out = _prepend_quarantine("payload", "[QUARANTINE] ")
        assert out == "[QUARANTINE] payload"

    def test_list_prepends_text_block(self) -> None:
        original = [
            {"type": "text", "text": "meta"},
            {"type": "image", "data": "..."},
        ]
        out = _prepend_quarantine(original, "[QUARANTINE] ")
        assert isinstance(out, list)
        assert out[0] == {"type": "text", "text": "[QUARANTINE] "}
        # Existing blocks preserved in order.
        assert out[1:] == original
