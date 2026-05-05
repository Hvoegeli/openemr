"""BM25 retrieval over the hand-curated guideline corpus.

The corpus + index are built once at module import (eager) so the first
`retrieve_guidelines` call has no warm-up cost. The corpus is small
(currently 12 chunks, ~3KB) so the eager load is essentially free.

Public surface:

  retrieve_guidelines(query, k=3) -> list[RetrievalHit]
      Score every chunk against the query, return the top-k hits with
      their full GuidelineChunk metadata + BM25 score + 1-indexed rank.

  CORPUS: list[GuidelineChunk]   The loaded corpus, exposed for tests
                                 and the smoke script.

The search is intentionally light:
- Tokenization: lowercased, alphanumerics-only (regex), single whitespace
  on the tag boundary. Drops punctuation but preserves numerics like
  "10-year" → ["10", "year"] which is fine for BM25.
- Index: BM25Okapi from `rank_bm25`, default k1/b parameters.
- Scoring: per-chunk BM25 score over (text + title + topic_tags) —
  including title and tags lets the model surface a chunk by topic
  even when the user query doesn't repeat the exact corpus wording.

What this is NOT (deferred to post-MVP):
- Dense embeddings (would need an embedding model — OpenAI / Voyage /
  local sentence-transformers — and an extra service or weight load).
- Hybrid retrieve (BM25 ∪ dense, then merge / rerank).
- BGE reranker on top of the candidate pool.
- Per-tag filtering (could be added by accepting a `tags=` kwarg).

The function signature is the contract — whichever ranker lives under
the hood, callers see the same `list[RetrievalHit]`.
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


def retrieve_guidelines(query: str, k: int = 3) -> list[RetrievalHit]:
    """Score every chunk against `query`, return the top-`k` as
    RetrievalHit objects.

    Empty/whitespace queries return an empty list (BM25 against a
    zero-token query would score every chunk at 0.0, which is just
    noise — we'd rather the caller see an empty result and decide).

    `k` is clamped to len(CORPUS) so over-asking isn't an error.
    Hits with score 0.0 are dropped — they're chunks that share no
    tokens with the query and including them would dilute the result.
    """
    if k <= 0:
        return []
    tokens = _tokenize(query)
    if not tokens:
        return []
    scores = _BM25_INDEX.get_scores(tokens)
    # argsort descending, take top k indices
    ranked = sorted(
        range(len(scores)), key=lambda i: scores[i], reverse=True,
    )[:min(k, len(CORPUS))]
    hits: list[RetrievalHit] = []
    for rank_idx, corpus_idx in enumerate(ranked, start=1):
        score = float(scores[corpus_idx])
        if score <= 0.0:
            # Once we hit a zero-score, every later chunk is also 0
            # (sorted descending). No need to keep iterating.
            break
        hits.append(RetrievalHit(
            chunk=CORPUS[corpus_idx], score=score, rank=rank_idx,
        ))
    return hits
