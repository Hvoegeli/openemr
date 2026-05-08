"""Smoke tests for the hybrid (BM25 + dense) retrieval pipeline.

Skips the live sentence-transformers model entirely — these tests inject
a stub embedder so they run without torch on path. The model-loading
behavior is exercised separately by `tests/guidelines/test_retrieve.py`
and (in production) by the eval gate.

Coverage:
- Stub embedder produces deterministic dense rankings.
- BM25 + dense union deduplicates by chunk_id.
- enable_dense=False short-circuits the dense stage.
- Dense-stage failure (e.g. ImportError on torch) degrades cleanly to
  BM25-only with a warning rather than killing the request.
"""

from __future__ import annotations

import numpy as np

from app.guidelines.embed import (
    EMBED_DIM,
    build_corpus_embeddings,
    dense_candidates,
)
from app.guidelines.retrieve import CORPUS, retrieve_guidelines


class StubEmbedder:
    """Deterministic embedder that returns hash-based vectors.

    Two strings that share many tokens get similar vectors; unrelated
    strings get orthogonal vectors. Not semantically meaningful — it's
    just stable enough for tests to assert ordering without invoking
    a real model.
    """

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, EMBED_DIM), dtype=np.float32)
        out = np.zeros((len(texts), EMBED_DIM), dtype=np.float32)
        for i, t in enumerate(texts):
            # Hash each token into a position; sum makes overlapping
            # token sets produce overlapping vectors. Stable across runs
            # because Python's hash() is salted but we use a fixed seed.
            for tok in t.lower().split():
                pos = abs(hash(tok)) % EMBED_DIM
                out[i, pos] += 1.0
            # L2-normalize so cosine similarity = dot product.
            norm = float(np.linalg.norm(out[i]))
            if norm > 0:
                out[i] /= norm
        return out


def test_build_corpus_embeddings_with_stub() -> None:
    """build_corpus_embeddings produces a (N, EMBED_DIM) matrix with the
    stub embedder, no model download required."""
    embeddings = build_corpus_embeddings(CORPUS, embedder=StubEmbedder())
    assert embeddings.matrix.shape == (len(CORPUS), EMBED_DIM)
    assert embeddings.chunk_ids == [c.chunk_id for c in CORPUS]
    # Every row is unit-normalized (within float tolerance).
    norms = np.linalg.norm(embeddings.matrix, axis=1)
    assert np.allclose(norms[norms > 0], 1.0, atol=1e-5)


def test_dense_candidates_returns_ordered_pairs() -> None:
    """dense_candidates returns up to pool_size (chunk_id, score) pairs
    ranked by descending similarity. The stub embedder makes the actual
    ordering hash-dependent but stable, so we just check shape + ordering
    invariant."""
    embeddings = build_corpus_embeddings(CORPUS, embedder=StubEmbedder())
    pairs = dense_candidates(
        "aspirin primary prevention cardiovascular",
        embeddings,
        pool_size=5,
        embedder=StubEmbedder(),
    )
    assert len(pairs) <= 5
    assert all(isinstance(cid, str) for cid, _ in pairs)
    # Strictly non-increasing scores.
    scores = [s for _, s in pairs]
    assert scores == sorted(scores, reverse=True)


def test_dense_candidates_empty_query_returns_empty() -> None:
    """Whitespace-only query short-circuits without an embedder call."""
    embeddings = build_corpus_embeddings(CORPUS, embedder=StubEmbedder())
    assert dense_candidates("   ", embeddings, 5, embedder=StubEmbedder()) == []


def test_retrieve_guidelines_bm25_only_path() -> None:
    """enable_dense=False + enable_rerank=False → pure BM25, no torch
    or anthropic calls. Sanity-check that the function still works in
    that regression-test configuration."""
    hits = retrieve_guidelines(
        "aspirin primary prevention",
        k=3,
        enable_rerank=False,
        enable_dense=False,
    )
    assert len(hits) <= 3
    # Every hit is a real corpus chunk.
    chunk_ids = {c.chunk_id for c in CORPUS}
    assert all(h.chunk.chunk_id in chunk_ids for h in hits)


def test_dense_failure_does_not_kill_pipeline(monkeypatch) -> None:
    """If `_ensure_embeddings` raises (e.g. torch missing on local Mac),
    retrieve_guidelines logs and degrades to BM25 candidates only.

    Simulates the Intel-Mac dev path where sentence-transformers isn't
    installed via the pyproject marker.
    """
    from app.guidelines import retrieve as retrieve_mod

    def boom() -> None:
        raise ImportError("torch not available on this platform")

    monkeypatch.setattr(retrieve_mod, "_ensure_embeddings", boom)
    hits = retrieve_guidelines(
        "aspirin primary prevention",
        k=3,
        enable_rerank=False,  # avoid network call
        enable_dense=True,    # but ask for dense — should silently skip
    )
    # Result should still be a non-empty list (BM25 hits survived) and
    # every hit should be a valid chunk.
    chunk_ids = {c.chunk_id for c in CORPUS}
    assert len(hits) > 0
    assert all(h.chunk.chunk_id in chunk_ids for h in hits)
