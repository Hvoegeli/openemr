"""Hybrid (sparse + dense) retrieval + LLM-rerank over the guideline corpus.

The corpus + BM25 index are loaded eagerly at module import (essentially
free for 12 chunks). The dense embedding index is built lazily on the
first hybrid retrieval call so importing this module never blocks on
loading the embedding model — useful for tests that only need BM25 or
corpus-shape inspection. The dense matrix is then cached on disk
(see `app/guidelines/embed.py`); subsequent process starts hit the
cache instead of re-embedding.

Public surface:

  retrieve_guidelines(query, k=3) -> list[RetrievalHit]
      Three-stage retrieve: BM25 + dense embedding both produce candidate
      pools; their union is passed to `app.guidelines.rerank.rerank` for
      LLM-judged final ordering, returning the top-`k`. Callers see one
      `list[RetrievalHit]`; the sparse/dense/rerank split is internal.

  CORPUS: list[GuidelineChunk]   The loaded corpus, exposed for tests
                                 and the smoke script.

The three stages do complementary work:
- **BM25 (sparse)** is exhaustive across the corpus and serves as a cheap
  keyword-overlap recall filter. Tokenization is lowercased +
  alphanumeric-only. Scoring uses the chunk body + title + topic tags.
- **Dense embedding (semantic)** uses `BAAI/bge-small-en-v1.5` cosine
  similarity to catch the cases where wording diverges from the corpus
  ("blood thinner for stroke prevention" → warfarin/AF chunk even when
  the body uses different vocabulary). Runs entirely locally on CPU.
- **The reranker** is a Claude Haiku call that scores semantic relevance
  on each surviving candidate 0.0-1.0. It catches the cases where the
  retrieval stages over-recall because of token frequency or shallow
  semantic overlap.

The union of BM25 and dense pools is deduped by `chunk_id` before
hitting the rerank stage. Empty queries return an empty list. `k` is
clamped to the surviving candidate count.

The function signature is the contract — the three-stage pipeline lives
behind it.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import yaml
from rank_bm25 import BM25Okapi

from app.guidelines.embed import (
    CorpusEmbeddings,
    LocalEmbedder,
    build_corpus_embeddings,
    dense_candidates,
)
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
# Dense embedding index. Built lazily on first retrieval call rather
# than at import — keeps `import retrieve` cheap for tests that don't
# need dense (e.g. tokenizer/corpus-shape checks). The first hybrid
# query pays the build cost (~1-3s for the cached model + 12 chunks).
_EMBEDDER: LocalEmbedder | None = None
_EMBEDDINGS: CorpusEmbeddings | None = None
log.info("guidelines: loaded %d chunks from %s", len(CORPUS), CORPUS_PATH)


# How many candidates each stage feeds to the reranker. Larger pool =
# better recall but more rerank cost; smaller pool = cheaper but risks
# the recall stage dropping the relevant chunk before rerank gets a
# chance. With a 12-chunk corpus, BM25 returning up to 8 + dense
# returning up to 8 (deduped → typically 8-10 unique candidates) covers
# 2/3 to all of the corpus on every query.
BM25_CANDIDATE_POOL = 8
DENSE_CANDIDATE_POOL = 8


def _ensure_embeddings() -> tuple[LocalEmbedder, CorpusEmbeddings]:
    """Lazy build of the dense embedding index. Cached on disk by
    `app.guidelines.embed`; first call after a fresh checkout pays
    the embedding cost, subsequent calls hit the cache."""
    global _EMBEDDER, _EMBEDDINGS
    if _EMBEDDER is None:
        _EMBEDDER = LocalEmbedder()
    if _EMBEDDINGS is None:
        _EMBEDDINGS = build_corpus_embeddings(CORPUS, embedder=_EMBEDDER)
    return _EMBEDDER, _EMBEDDINGS


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
    query: str,
    k: int = 3,
    *,
    enable_rerank: bool = True,
    enable_dense: bool = True,
) -> list[RetrievalHit]:
    """Three-stage retrieve: (BM25 ∪ dense) → rerank → top-`k`.

    Stage 1a: BM25 selects up to `BM25_CANDIDATE_POOL` chunks with
              positive scores (sparse keyword recall).
    Stage 1b: Dense embedding cosine similarity returns up to
              `DENSE_CANDIDATE_POOL` chunks (semantic recall). Local
              `BAAI/bge-small-en-v1.5` — see `app/guidelines/embed.py`.
    Stage 2:  Union of 1a + 1b deduped by `chunk_id` is passed to
              `app.guidelines.rerank.rerank` for LLM-judged final
              ordering. Top-`k` by rerank score is returned.

    The returned `RetrievalHit.score` is the rerank score (0.0-1.0)
    when `enable_rerank=True` (the default and only production path).
    With `enable_rerank=False` the function returns the BM25 ∪ dense
    union ordered as "BM25 candidates first (in BM25 score order), then
    any dense-only chunks that didn't appear in BM25." `score` in this
    fallback equals the BM25 score (0.0 for chunks that surfaced via
    the dense path only). Used by the eval A/B harness to compare
    BM25-only vs hybrid-recall vs full-pipeline quality. With
    `enable_dense=False` the dense stage is skipped entirely (also for
    A/B comparisons; production always runs both).

    Empty/whitespace queries return an empty list. `k` is clamped to
    the candidate pool size so over-asking isn't an error.

    If the rerank-stage call fails (no API key, network error, malformed
    response), the exception propagates — there is no silent fallback
    to BM25 ordering at the production path, because that would mask a
    degraded retrieval quality from operators. Use `enable_rerank=False`
    explicitly if you want BM25 ordering on purpose.
    """
    if k <= 0:
        return []
    if not query.strip():
        return []

    # Stage 1a: BM25 sparse candidates.
    bm25_pool = _bm25_candidates(query, BM25_CANDIDATE_POOL)
    bm25_score_by_id: dict[str, float] = {c.chunk_id: s for c, s in bm25_pool}

    # Stage 1b: Dense semantic candidates. Skipped on enable_dense=False
    # (eval A/B path) and when the global flag has been disabled (e.g.
    # for tests that don't want torch on path). Unionized by chunk_id;
    # BM25 wins in the dedupe so a chunk that hits both sources keeps
    # its BM25 score for the informational `bm25_score` field.
    dense_pool: list[tuple[str, float]] = []
    if enable_dense:
        try:
            _, embeddings = _ensure_embeddings()
            dense_pool = dense_candidates(
                query, embeddings, DENSE_CANDIDATE_POOL,
                embedder=_EMBEDDER,
            )
        except Exception as e:  # noqa: BLE001
            # Dense failure (model unavailable, etc.) shouldn't kill the
            # pipeline — degrade to BM25-only with a warning so the
            # operator notices but the user still gets results.
            log.warning("dense stage failed; degrading to BM25-only: %s", e)

    # Build the candidate-chunk list as the union of both pools, deduped
    # by chunk_id, preserving BM25 ordering first (BM25 hits typically
    # have stronger keyword overlap, which is a reasonable seed order
    # for the reranker).
    seen_ids: set[str] = set()
    candidate_chunks: list[GuidelineChunk] = []
    chunk_by_id = {c.chunk_id: c for c in CORPUS}
    for chunk, _score in bm25_pool:
        if chunk.chunk_id in seen_ids:
            continue
        seen_ids.add(chunk.chunk_id)
        candidate_chunks.append(chunk)
    for cid, _score in dense_pool:
        if cid in seen_ids:
            continue
        seen_ids.add(cid)
        chunk = chunk_by_id.get(cid)
        if chunk is not None:
            candidate_chunks.append(chunk)

    if not candidate_chunks:
        return []

    if not enable_rerank:
        # Sparse-or-hybrid recall ordering, no LLM rerank. Used by the
        # eval A/B harness to measure how much the rerank stage adds.
        # Score in this fallback is the BM25 score where available, else 0.0.
        return [
            RetrievalHit(
                chunk=chunk,
                score=bm25_score_by_id.get(chunk.chunk_id, 0.0),
                rank=rank_idx,
                bm25_score=bm25_score_by_id.get(chunk.chunk_id, 0.0),
            )
            for rank_idx, chunk in enumerate(candidate_chunks[:k], start=1)
        ]

    # Local import — keeps `retrieve.py` importable in environments
    # where the rerank module's anthropic dep would lazy-init on
    # first use (the corpus tests import this module without an
    # outbound API call).
    from app.guidelines.rerank import rerank

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
