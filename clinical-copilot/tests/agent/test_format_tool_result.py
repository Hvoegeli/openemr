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

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agent.graph import (
    _content_text_for_scan,
    _format_tool_result,
    _prepend_quarantine,
    _strip_image_blocks_from_messages,
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


class TestStripImageBlocksFromMessages:
    """Locks down the supervisor-side image-trimming behavior.

    The supervisor never reads pixels — its only job is routing — but
    the LangGraph message stream carries image-bearing ToolMessages
    from `get_document_content` calls forward on every hop. Stripping
    those at the supervisor's per-call view (only) is the cost
    optimization documented in the helper's docstring. These tests
    verify the helper drops images, preserves text, doesn't mutate
    inputs, and leaves non-ToolMessages alone.
    """

    def test_human_and_ai_messages_pass_through_unchanged(self) -> None:
        msgs = [
            HumanMessage(content="hello"),
            AIMessage(content="hi"),
        ]
        out = _strip_image_blocks_from_messages(msgs)
        assert out == msgs

    def test_tool_message_with_string_content_passes_through(self) -> None:
        # Non-document tool results land as JSON strings, not lists. The
        # helper should leave them entirely alone.
        tm = ToolMessage(
            content=json.dumps({"data": {"x": 1}, "sources": []}),
            tool_call_id="call-1",
            name="get_patient_card",
        )
        out = _strip_image_blocks_from_messages([tm])
        assert out == [tm]

    def test_tool_message_with_only_text_blocks_passes_through(self) -> None:
        tm = ToolMessage(
            content=[{"type": "text", "text": "metadata only"}],
            tool_call_id="call-2",
            name="get_document_content",
        )
        out = _strip_image_blocks_from_messages([tm])
        # Same instance returned — no rebuild needed when there's no image.
        assert out[0] is tm

    def test_image_blocks_dropped_text_blocks_preserved(self) -> None:
        original = ToolMessage(
            content=[
                {"type": "text", "text": "metadata json here"},
                {"type": "image", "source_type": "base64",
                 "mime_type": "image/png", "data": "AAAA"},
                {"type": "image", "source_type": "base64",
                 "mime_type": "image/png", "data": "BBBB"},
            ],
            tool_call_id="call-3",
            name="get_document_content",
        )
        out = _strip_image_blocks_from_messages([original])
        assert len(out) == 1
        result = out[0]
        assert isinstance(result, ToolMessage)
        # Should not be the same object — we rebuild to avoid mutating state.
        assert result is not original
        # Text block preserved at the front.
        assert result.content[0] == {"type": "text", "text": "metadata json here"}
        # No image blocks remain.
        assert all(
            (not isinstance(b, dict)) or b.get("type") != "image"
            for b in result.content
        )
        # An informational marker is appended exactly once.
        markers = [
            b for b in result.content
            if isinstance(b, dict) and b.get("type") == "text"
            and "omitted from supervisor view" in b.get("text", "")
        ]
        assert len(markers) == 1
        # tool_call_id and name carry through.
        assert result.tool_call_id == "call-3"
        assert result.name == "get_document_content"

    def test_does_not_mutate_input_list_or_messages(self) -> None:
        original_blocks = [
            {"type": "text", "text": "metadata"},
            {"type": "image", "source_type": "base64",
             "mime_type": "image/png", "data": "ZZZZ"},
        ]
        original = ToolMessage(
            content=list(original_blocks),  # copy so identity check below is meaningful
            tool_call_id="call-4",
            name="get_document_content",
        )
        msgs = [HumanMessage(content="hi"), original]
        snapshot_count = len(original.content)
        _ = _strip_image_blocks_from_messages(msgs)
        # Input message list still references the original ToolMessage,
        # whose content is unchanged.
        assert msgs[1] is original
        assert len(original.content) == snapshot_count
        # And the original still has its image block.
        assert any(
            isinstance(b, dict) and b.get("type") == "image"
            for b in original.content
        )

    def test_mixed_message_stream_only_modifies_image_bearing_tool_messages(self) -> None:
        # Realistic shape: user msg, supervisor pick (AIMessage), tool
        # result without images, tool result WITH images, worker reply.
        no_image_tm = ToolMessage(
            content=[{"type": "text", "text": "card json"}],
            tool_call_id="t1",
            name="get_patient_card",
        )
        image_tm = ToolMessage(
            content=[
                {"type": "text", "text": "doc meta"},
                {"type": "image", "source_type": "base64",
                 "mime_type": "image/png", "data": "PG1"},
            ],
            tool_call_id="t2",
            name="get_document_content",
        )
        msgs = [
            HumanMessage(content="summarize"),
            AIMessage(content="ok"),
            no_image_tm,
            image_tm,
            AIMessage(content="here you go"),
        ]
        out = _strip_image_blocks_from_messages(msgs)
        # Same length, same order.
        assert len(out) == len(msgs)
        # Pass-throughs keep identity.
        assert out[0] is msgs[0]
        assert out[1] is msgs[1]
        assert out[2] is no_image_tm  # text-only TM untouched
        assert out[4] is msgs[4]
        # Image-bearing TM is rebuilt without images.
        rebuilt = out[3]
        assert isinstance(rebuilt, ToolMessage)
        assert rebuilt is not image_tm
        assert all(
            (not isinstance(b, dict)) or b.get("type") != "image"
            for b in rebuilt.content
        )
