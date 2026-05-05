"""Tests for the LLM-as-reranker stage.

The pure-logic helpers (`_parse_scores`, `_build_user_prompt`) get
unit-tested without network. The end-to-end `rerank` function makes
a live Anthropic call and is gated behind the `RERANK_LIVE` env var
so the default test pass stays fast and free.

To run the live test:

    RERANK_LIVE=1 uv run pytest tests/guidelines/test_rerank.py::TestRerankLive
"""

from __future__ import annotations

import os

import pytest

from app.guidelines.rerank import (
    RerankedHit,
    _build_user_prompt,
    _parse_scores,
    rerank,
)
from app.guidelines.retrieve import CORPUS


# Pick three real corpus chunks so prompt-shape tests don't drift if the
# corpus YAML is reorganized — we just look up by source.
def _three_chunks():
    return CORPUS[:3]


class TestBuildUserPrompt:
    def test_includes_query_and_all_candidates(self) -> None:
        chunks = _three_chunks()
        prompt = _build_user_prompt("type 2 diabetes hba1c", chunks)
        assert "QUERY: type 2 diabetes hba1c" in prompt
        for i, c in enumerate(chunks):
            # Each candidate prefixed by its 0-based id and includes
            # the title (so the model has both the slug and the
            # human-readable hook to score against).
            assert f"[{i}] {c.title}" in prompt

    def test_truncates_long_text(self) -> None:
        # Hand-craft a chunk with overlong text and confirm the prompt
        # caps it. Using the existing schema so we don't drift if
        # the truncation logic changes.
        from app.guidelines.schema import GuidelineChunk
        long_text = "x" * 1000
        c = GuidelineChunk(
            chunk_id="test",
            source="TEST",
            title="Long",
            year="2026",
            url="https://example.test",
            topic_tags=[],
            text=long_text,
        )
        prompt = _build_user_prompt("anything", [c])
        # The body of the candidate should be at most 600 chars.
        assert "x" * 600 in prompt
        assert "x" * 700 not in prompt

    def test_returns_response_format_marker(self) -> None:
        chunks = _three_chunks()
        prompt = _build_user_prompt("q", chunks)
        # The prompt must remind the model of the JSON shape so it
        # doesn't drift to free-text responses.
        assert '"scores"' in prompt
        assert '"id"' in prompt
        assert '"score"' in prompt


class TestParseScores:
    def test_well_formed_response(self) -> None:
        raw = '{"scores": [{"id": 0, "score": 0.9}, {"id": 1, "score": 0.4}, {"id": 2, "score": 0.1}]}'
        scores = _parse_scores(raw, n_candidates=3)
        assert scores == [0.9, 0.4, 0.1]

    def test_strips_code_fence(self) -> None:
        raw = '```json\n{"scores": [{"id": 0, "score": 1.0}]}\n```'
        scores = _parse_scores(raw, n_candidates=1)
        assert scores == [1.0]

    def test_strips_bare_code_fence(self) -> None:
        raw = '```\n{"scores": [{"id": 0, "score": 0.5}]}\n```'
        scores = _parse_scores(raw, n_candidates=1)
        assert scores == [0.5]

    def test_clamps_overshoot(self) -> None:
        # The model occasionally returns 1.2 or -0.1; we clamp.
        raw = '{"scores": [{"id": 0, "score": 1.5}, {"id": 1, "score": -0.3}]}'
        scores = _parse_scores(raw, n_candidates=2)
        assert scores == [1.0, 0.0]

    def test_missing_entries_default_to_zero(self) -> None:
        # 3 candidates but model only scored 2 — the missing one
        # should be 0.0 (interpreted as "model thought it was unrelated").
        raw = '{"scores": [{"id": 0, "score": 0.8}, {"id": 2, "score": 0.4}]}'
        scores = _parse_scores(raw, n_candidates=3)
        assert scores == [0.8, 0.0, 0.4]

    def test_extra_entries_ignored(self) -> None:
        # Model returns id=99 which doesn't exist in our 3 candidates.
        raw = '{"scores": [{"id": 0, "score": 0.5}, {"id": 99, "score": 0.9}]}'
        scores = _parse_scores(raw, n_candidates=3)
        assert scores == [0.5, 0.0, 0.0]

    def test_duplicate_ids_first_wins(self) -> None:
        raw = '{"scores": [{"id": 0, "score": 0.7}, {"id": 0, "score": 0.2}]}'
        scores = _parse_scores(raw, n_candidates=1)
        assert scores == [0.7]

    def test_non_numeric_score_treated_as_zero(self) -> None:
        raw = '{"scores": [{"id": 0, "score": "high"}]}'
        scores = _parse_scores(raw, n_candidates=1)
        assert scores == [0.0]

    def test_invalid_json_raises(self) -> None:
        with pytest.raises(ValueError, match="not valid JSON"):
            _parse_scores("this is not json", n_candidates=1)

    def test_missing_scores_key_raises(self) -> None:
        with pytest.raises(ValueError, match="missing 'scores' list"):
            _parse_scores('{"foo": 1}', n_candidates=1)


class TestRerankShortCircuits:
    """`rerank` should never make an API call for inputs where the
    answer is trivially empty — empty candidates, k<=0, blank query."""

    def test_empty_candidates_returns_empty(self) -> None:
        assert rerank("anything", [], k=3) == []

    def test_k_zero_returns_empty(self) -> None:
        assert rerank("anything", _three_chunks(), k=0) == []

    def test_k_negative_returns_empty(self) -> None:
        assert rerank("anything", _three_chunks(), k=-1) == []

    def test_blank_query_returns_empty(self) -> None:
        assert rerank("", _three_chunks(), k=3) == []
        assert rerank("   ", _three_chunks(), k=3) == []


# ── Live tests (gated) ──────────────────────────────────────────────────

@pytest.mark.skipif(
    os.environ.get("RERANK_LIVE") != "1",
    reason="set RERANK_LIVE=1 to run rerank tests against the Anthropic API",
)
class TestRerankLive:
    def test_diabetes_query_picks_diabetes_chunk(self) -> None:
        chunks = list(CORPUS)
        hits = rerank("type 2 diabetes glycemic target", chunks, k=3)
        assert len(hits) == 3
        # The reranker should put a diabetes chunk in the top 1 — both
        # ada_glycemic_targets and ada_pharmacotherapy are highly
        # relevant; either is acceptable as the top.
        top = hits[0]
        assert isinstance(top, RerankedHit)
        assert top.score > 0.0
        assert top.rank == 1
        # At least one of the top-3 should be ADA (the diabetes
        # source); a corpus-wide reranker that put zero ADA chunks in
        # the top-3 for a diabetes query would be obviously wrong.
        assert any(h.chunk.source == "ADA" for h in hits)

    def test_returns_at_most_k_results(self) -> None:
        chunks = list(CORPUS)[:5]
        hits = rerank("aspirin primary prevention", chunks, k=3)
        assert len(hits) == 3

    def test_ranks_are_sequential(self) -> None:
        chunks = list(CORPUS)[:5]
        hits = rerank("statin therapy", chunks, k=3)
        assert [h.rank for h in hits] == [1, 2, 3]

    def test_scores_descending(self) -> None:
        chunks = list(CORPUS)[:5]
        hits = rerank("statin therapy", chunks, k=3)
        scores = [h.score for h in hits]
        assert scores == sorted(scores, reverse=True)
