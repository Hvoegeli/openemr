"""Isolated tests for guideline retrieval.

Loads the real corpus (no fixtures, no mocks) — the corpus is small
enough that round-tripping it in tests is fast and proves the YAML
itself is well-formed.

These tests pass `enable_rerank=False` so the BM25 stage is exercised
without burning API credits on the rerank stage. Live rerank behavior
is covered by `scripts/smoke_rerank.py` (network-dependent, not in CI).
"""

from __future__ import annotations

import re

from app.guidelines.retrieve import (
    CORPUS,
    _tokenize,
    retrieve_guidelines,
)


def _bm25_only(query: str, k: int = 3):
    """Convenience wrapper — every test below wants the BM25-only path.

    Disables BOTH the rerank stage (no LLM calls) AND the dense stage
    (no embedding fan-in). The dense stage was added after these tests
    were written; without `enable_dense=False` it pulls cosine-closest
    chunks into the result set with `score=0.0`, which makes the
    "zero-score chunks are dropped" and "all clamped-k hits have
    positive scores" assertions wrong by construction. Live hybrid
    behavior is exercised separately by `tests/guidelines/test_hybrid.py`.
    """
    return retrieve_guidelines(
        query, k=k, enable_rerank=False, enable_dense=False,
    )


# Mirror of `app/agent/validator.py::CITATION_RE` — the regex the response
# validator uses to detect citations of the form `[Type/id]`. Tests below
# assert every chunk_id in the corpus is shaped so the agent can cite it
# as `[Guideline/<chunk_id>]` without the validator rejecting the bracket.
_CITATION_ID_RE = re.compile(r"^[a-zA-Z0-9._-]+$")


class TestCorpusLoad:
    def test_corpus_has_expected_size(self) -> None:
        # Lock in the count so a silent corpus expansion is loud (we
        # might WANT to add a chunk, but not without intent).
        assert len(CORPUS) == 12

    def test_every_chunk_has_unique_id(self) -> None:
        ids = [c.chunk_id for c in CORPUS]
        assert len(ids) == len(set(ids))

    def test_every_chunk_has_text_and_url(self) -> None:
        # Schema enforces min_length=1 so this is belt-and-suspenders;
        # makes the failure obvious if someone weakens the schema.
        for c in CORPUS:
            assert c.text.strip()
            assert c.url.startswith("http")

    def test_corpus_covers_both_sources(self) -> None:
        sources = {c.source for c in CORPUS}
        assert "USPSTF" in sources
        assert "ADA" in sources

    def test_every_chunk_id_passes_citation_regex(self) -> None:
        """The agent cites guideline hits as `[Guideline/<chunk_id>]`. The
        validator's regex (mirrored above as `_CITATION_ID_RE`) only
        accepts `[a-zA-Z0-9._-]+` for the id portion. A chunk_id with a
        space, slash, unicode, or other punctuation would silently make
        the citation unparsable and fail validation, masking what the
        agent actually retrieved."""
        bad = [c.chunk_id for c in CORPUS if not _CITATION_ID_RE.match(c.chunk_id)]
        assert bad == [], (
            f"chunk_id(s) won't pass the citation regex: {bad}. "
            f"Use only [a-zA-Z0-9._-]; prefer snake_case."
        )


class TestTokenize:
    def test_lowercase_and_alphanum_only(self) -> None:
        assert _tokenize("HbA1c >7% — should we add insulin?") == [
            "hba1c", "7", "should", "we", "add", "insulin",
        ]

    def test_empty_input_returns_empty(self) -> None:
        assert _tokenize("") == []
        assert _tokenize("   \n\t  ") == []


class TestRetrieveGuidelines:
    def test_diabetes_query_returns_diabetes_chunk_first(self) -> None:
        hits = _bm25_only("type 2 diabetes hba1c target", k=3)
        assert len(hits) > 0
        # Top hit should be one of the ADA glycemic / pharmacotherapy
        # chunks — both are highly diabetes-tagged.
        top_id = hits[0].chunk.chunk_id
        assert top_id.startswith("ada_") or "diabetes" in top_id

    def test_aspirin_query_surfaces_aspirin_chunks(self) -> None:
        hits = _bm25_only("aspirin primary prevention bleeding", k=3)
        assert hits, "expected at least one hit for aspirin query"
        ids = [h.chunk.chunk_id for h in hits]
        assert any("aspirin" in i for i in ids)

    def test_statin_query_surfaces_statin_chunks(self) -> None:
        hits = _bm25_only("statin LDL cardiovascular risk", k=3)
        assert hits
        ids = [h.chunk.chunk_id for h in hits]
        assert any("statin" in i for i in ids)

    def test_colorectal_query_surfaces_crc_chunk(self) -> None:
        hits = _bm25_only("colorectal cancer screening colonoscopy", k=3)
        assert hits
        assert any("colorectal" in h.chunk.chunk_id for h in hits)

    def test_results_are_ranked_descending_by_score(self) -> None:
        hits = _bm25_only("statin diabetes", k=5)
        scores = [h.score for h in hits]
        assert scores == sorted(scores, reverse=True)

    def test_rank_field_is_one_indexed_and_sequential(self) -> None:
        hits = _bm25_only("statin diabetes", k=5)
        assert [h.rank for h in hits] == list(range(1, len(hits) + 1))

    def test_zero_score_chunks_are_dropped(self) -> None:
        # A query that shares NO tokens with any corpus chunk should
        # return an empty list — including zero-score hits would dilute
        # the result and confuse the caller about what was retrieved.
        hits = _bm25_only("zzzqqq xyzzy quuxquux", k=3)
        assert hits == []

    def test_empty_query_returns_empty(self) -> None:
        assert _bm25_only("", k=3) == []
        assert _bm25_only("   ", k=3) == []

    def test_zero_k_returns_empty(self) -> None:
        assert _bm25_only("statin", k=0) == []

    def test_negative_k_returns_empty(self) -> None:
        assert _bm25_only("statin", k=-1) == []

    def test_k_larger_than_corpus_clamped(self) -> None:
        # Asking for more hits than the corpus has must NOT raise.
        # Caller might naively pass a huge k for "give me everything".
        hits = _bm25_only("diabetes", k=10_000)
        # All returned hits have positive scores; count is at most CORPUS size.
        assert len(hits) <= len(CORPUS)
        for h in hits:
            assert h.score > 0
