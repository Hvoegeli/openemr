"""BM25 + LLM-rerank retrieval over the hand-curated guideline corpus.

The corpus + BM25 index are built once at module import (eager) so the
first `retrieve_guidelines` call has no warm-up cost. The corpus is
small (currently 12 chunks, ~3KB) so the eager load is essentially free.

Public surface:

  retrieve_guidelines(query, k=3) -> list[RetrievalHit]
      Two-stage retrieve: BM25 selects a candidate pool of size
      `BM25_CANDIDATE_POOL`, then `app.guidelines.rerank.rerank` re-orders
      that pool by LLM-judged relevance and returns the top-k. Callers
      see one `list[RetrievalHit]`; the BM25/rerank split is internal.

  CORPUS: list[GuidelineChunk]   The loaded corpus, exposed for tests
                                 and the smoke script.

The two stages do different work:
- BM25 (sparse) is exhaustive across the corpus and serves as a cheap
  recall filter. Tokenization is lowercased + alphanumeric-only.
  Scoring uses the chunk body + title + topic tags so tag matches
  surface relevant chunks even when body wording diverges.
- The reranker is a Claude Haiku call that scores semantic relevance
  to the query 0.0-1.0. It catches the cases where BM25's word-overlap
  heuristic prefers a high-frequency chunk over a chunk that actually
  addresses the query.

What this is NOT (deferred):
- Dense embedding retrieval (BGE-small or text-embedding-3-small
  cosine over the same corpus). On a 12-chunk corpus the recall
  benefit over BM25 is marginal. Tracked in W2_ARCHITECTURE.md §3 as
  the next iteration when the corpus grows past ~100 chunks.
- Per-tag filtering (could be added by accepting a `tags=` kwarg).

The function signature is the contract — the two-stage pipeline lives
behind it.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import yaml
from rank_bm25 import BM25Okapi

from app.guidelines.schema import GuidelineChunk, RetrievalHit

log = logging.getLogger("agent.guidelines.retrieve")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CORPUS_PATH = REPO_ROOT / "data" / "guidelines" / "corpus.yaml"

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase + extract alphanumeric runs. Stable across runs (no
    randomness, no language detection). BM25 wants tokens that appear
    in both query and corpus, so case-folding is mandatory."""
    return _TOKEN_RE.findall(text.lower())


def _load_corpus(path: Path = CORPUS_PATH) -> list[GuidelineChunk]:
    """Load + validate the YAML corpus. Each entry passes through
    Pydantic so a malformed entry fails loudly at module import."""
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, list) or not raw:
        raise RuntimeError(
            f"guideline corpus at {path} is empty or not a list"
        )
    chunks = [GuidelineChunk.model_validate(entry) for entry in raw]
    # Stable identity on chunk_id — silently colliding ids would make
    # results ambiguous when the agent cites a chunk by id.
    seen: set[str] = set()
    for c in chunks:
        if c.chunk_id in seen:
            raise RuntimeError(
                f"duplicate chunk_id {c.chunk_id!r} in {path}"
            )
        seen.add(c.chunk_id)
    return chunks


def _chunk_to_searchable_tokens(chunk: GuidelineChunk) -> list[str]:
    """Concatenate the searchable surface of a chunk before tokenization:
    body text + title + topic tags. Including title and tags means a
    query like 'aspirin primary prevention' will surface the chunk even
    when the body uses synonyms."""
    surface = " ".join([
        chunk.text,
        chunk.title,
        " ".join(chunk.topic_tags),
    ])
    return _tokenize(surface)


# ── Eager corpus + index load (runs once, at import) ────────────────────
CORPUS: list[GuidelineChunk] = _load_corpus()
_TOKENIZED: list[list[str]] = [_chunk_to_searchable_tokens(c) for c in CORPUS]
_BM25_INDEX: BM25Okapi = BM25Okapi(_TOKENIZED)
log.info("guidelines: loaded %d chunks from %s", len(CORPUS), CORPUS_PATH)


# How many BM25 candidates to feed the reranker. Larger pool = better
# recall before rerank but more rerank cost; smaller pool = cheaper but
# risks BM25 dropping the relevant chunk before rerank gets a chance.
# 8 is a comfortable middle for a 12-chunk corpus — the reranker sees
# 2/3 of the corpus on every query.
BM25_CANDIDATE_POOL = 8


def _bm25_candidates(query: str, pool_size: int) -> list[tuple[GuidelineChunk, float]]:
    """Stage-1: BM25 over the full corpus, return up to `pool_size`
    candidates with positive scores.

    Returns list of (chunk, bm25_score) pairs ranked descending by score.
    Empty if the query has no usable tokens.
    """
    tokens = _tokenize(query)
    if not tokens:
        return []
    scores = _BM25_INDEX.get_scores(tokens)
    ranked = sorted(
        range(len(scores)), key=lambda i: scores[i], reverse=True,
    )[:min(pool_size, len(CORPUS))]
    out: list[tuple[GuidelineChunk, float]] = []
    for corpus_idx in ranked:
        s = float(scores[corpus_idx])
        if s <= 0.0:
            break
        out.append((CORPUS[corpus_idx], s))
    return out


def retrieve_guidelines(
    query: str, k: int = 3, *, enable_rerank: bool = True,
) -> list[RetrievalHit]:
    """Two-stage retrieve: BM25 → rerank → top-`k`.

    Stage 1: BM25 selects up to `BM25_CANDIDATE_POOL` chunks with
             positive scores (the recall filter).
    Stage 2: `app.guidelines.rerank.rerank` re-orders those candidates
             by LLM-judged semantic relevance and keeps the top-`k`.

    The returned `RetrievalHit.score` is the rerank score (0.0-1.0)
    when `enable_rerank=True` (the default and only production path).
    With `enable_rerank=False` it falls back to the BM25 score and is
    intended for tests / corpus-shape inspections that don't want a
    network round-trip.

    Empty/whitespace queries return an empty list. `k` is clamped to
    the BM25 pool size so over-asking isn't an error. Zero-score
    BM25 chunks never reach the reranker.

    If the rerank-stage call fails (no API key, network error, malformed
    response), the exception propagates — there is no silent fallback
    to BM25-only ordering at the production path, because that would
    mask a degraded retrieval quality from operators. Use the explicit
    `enable_rerank=False` if you want BM25 ordering on purpose.
    """
    if k <= 0:
        return []
    candidates = _bm25_candidates(query, BM25_CANDIDATE_POOL)
    if not candidates:
        return []

    if not enable_rerank:
        # BM25-only path — no rerank, score == bm25_score.
        return [
            RetrievalHit(
                chunk=chunk,
                score=bm25_score,
                rank=rank_idx,
                bm25_score=bm25_score,
            )
            for rank_idx, (chunk, bm25_score) in enumerate(
                candidates[:k], start=1,
            )
        ]

    # Local import — keeps `retrieve.py` importable in environments
    # where the rerank module's anthropic dep would lazy-init on
    # first use (the corpus tests import this module without an
    # outbound API call).
    from app.guidelines.rerank import rerank

    candidate_chunks = [c for c, _ in candidates]
    bm25_score_by_id = {c.chunk_id: s for c, s in candidates}

    reranked = rerank(query, candidate_chunks, k=k)

    return [
        RetrievalHit(
            chunk=h.chunk,
            score=h.score,
            rank=h.rank,
            bm25_score=bm25_score_by_id.get(h.chunk.chunk_id, 0.0),
        )
        for h in reranked
    ]
