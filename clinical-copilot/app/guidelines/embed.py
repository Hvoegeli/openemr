"""Dense embedding stage for the guideline retrieval pipeline.

Composes with `app/guidelines/retrieve.py` to provide the "dense" half of
hybrid sparse+dense retrieval. BM25 (sparse) catches keyword-overlap; this
module catches semantic-overlap (a query about "blood thinner for stroke
prevention" surfaces the warfarin/AF chunk even when the body uses
different vocabulary).

## Embedding backend

Defaults to a local `sentence-transformers` model (`BAAI/bge-small-en-v1.5`,
~130MB on disk, 384-dim vectors, CPU-friendly). The model loads lazily on
the first `embed_batch` call (not at module import), so callers that only
need the corpus-cache helpers don't pay the load cost. Subsequent
`embed_batch` calls reuse the loaded model.

Why local rather than AWS Bedrock Titan or OpenAI embeddings:
- No API credentials required — works on the Hetzner box without AWS or
  OpenAI signup, and avoids any PHI-leak concern from sending queries to
  a third-party embedding API.
- ~50ms per query on CPU; fast enough that hybrid retrieval still
  beats the rerank stage for cost and latency.
- The Hetzner CPX21 has the RAM headroom (~150-200MB resident when
  loaded; we measured 2.4GB free + 2GB swap).
- Production-shape swap (Bedrock Titan, Voyage, Cohere via Bedrock) is
  a single new class implementing the `Embedder` protocol — only
  `embed_batch` needs a method body. The interface is provider-agnostic.

## Cache strategy

Corpus embeddings are computed once and cached on disk at
`data/guidelines/corpus_embeddings.npy`, keyed by a SHA-256 of the
serialized corpus content. If the corpus YAML changes, the hash mismatch
triggers a re-embed at startup. Cache misses cost ~2-3 seconds for the
12-chunk corpus; rebuilding the cache is cheap enough that we don't
bother committing the .npy file to git.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from app.guidelines.schema import GuidelineChunk

log = logging.getLogger("agent.guidelines.embed")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = REPO_ROOT / "data" / "guidelines"
CACHE_FILE = CACHE_DIR / "corpus_embeddings.npy"
CACHE_META_FILE = CACHE_DIR / "corpus_embeddings.meta.json"

# BGE-small has been the SOTA tier-1 small-model retriever since 2023; it
# beats most general-purpose embeddings (including OpenAI ada-002) on the
# MTEB retrieval benchmark and runs comfortably on CPU. 384-dim vectors
# keep the corpus matrix small (12 × 384 = ~18KB).
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384

# BGE-family models perform best when queries are prefixed with this
# "instruction" string at retrieval time (a quirk of how BGE was trained
# — short instruction tokens prime the query encoder for retrieval-style
# similarity rather than paraphrase-style similarity). The corpus side
# does NOT get a prefix.
_BGE_QUERY_INSTRUCTION = (
    "Represent this clinical question for retrieving relevant guidelines: "
)


class Embedder(Protocol):
    """Provider-agnostic embedding interface.

    Implementations only need `embed_batch`. Single-string `embed_query`
    is a thin wrapper that adds the BGE-style query prefix; `embed_corpus`
    likewise wraps without a prefix. Swapping to Bedrock Titan / Voyage /
    OpenAI is a new class implementing this Protocol.
    """

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Return a `(len(texts), dim)` float32 matrix of L2-normalized
        embeddings. Normalization is the contract — callers do dot-product
        cosine similarity directly on the rows."""
        ...


@dataclass(frozen=True)
class _LocalEmbedderState:
    """Lazy holder for the loaded sentence-transformers model. Frozen so
    we never accidentally swap the model mid-process."""
    model: object  # sentence_transformers.SentenceTransformer


class LocalEmbedder:
    """`sentence-transformers` backed embedder using BGE-small-en-v1.5.

    The model is loaded lazily on first `embed_batch` call so importing
    this module never blocks on the model download. First call after
    process start does pay the load cost (~1-2 seconds for a cached
    model, longer on first ever download).
    """

    def __init__(self, model_name: str = EMBED_MODEL_NAME) -> None:
        self._model_name = model_name
        self._state: _LocalEmbedderState | None = None

    def _ensure_model(self) -> _LocalEmbedderState:
        if self._state is not None:
            return self._state
        # Local import — keeps `sentence-transformers` from being a hard
        # dep of every module that imports this file. Tests that mock
        # the embedder don't need torch on path.
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415
        log.info("embed: loading %s (first call; subsequent calls reuse)", self._model_name)
        model = SentenceTransformer(self._model_name)
        self._state = _LocalEmbedderState(model=model)
        return self._state

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, EMBED_DIM), dtype=np.float32)
        state = self._ensure_model()
        # `normalize_embeddings=True` returns L2-normalized vectors so
        # downstream cosine similarity is just a dot product.
        vecs = state.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vecs.astype(np.float32, copy=False)


def _corpus_hash(corpus: list[GuidelineChunk]) -> str:
    """Stable hash of the corpus content. Used as the cache invalidation
    key — any edit to a chunk's id/source/text/title bumps the hash and
    forces a re-embed at startup. Topic tags are excluded (they don't
    affect the embedded surface). Sort by chunk_id for determinism."""
    payload = json.dumps(
        [
            {
                "chunk_id": c.chunk_id,
                "source": c.source,
                "title": c.title,
                "text": c.text,
            }
            for c in sorted(corpus, key=lambda c: c.chunk_id)
        ],
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _chunk_to_embed_surface(chunk: GuidelineChunk) -> str:
    """The text that gets embedded for a corpus chunk. Combines title +
    body so a query about the title surfaces the chunk even when the body
    uses different vocabulary. Topic tags are NOT included — BGE-small
    tokenizes hyphenated tags poorly and their semantic value is already
    captured by BM25 on the same surface."""
    return f"{chunk.title}\n\n{chunk.text}"


@dataclass(frozen=True)
class CorpusEmbeddings:
    """Loaded corpus embedding matrix + index metadata.

    `matrix` is a `(N, dim)` float32 array of L2-normalized embeddings;
    row `i` corresponds to `chunk_ids[i]`. `chunk_id_to_row` is the
    reverse lookup. Rows are not necessarily in corpus.yaml order —
    sort by `chunk_ids` if order matters.
    """
    chunk_ids: list[str]
    chunk_id_to_row: dict[str, int]
    matrix: np.ndarray  # shape (N, EMBED_DIM)
    corpus_hash: str


def _load_cache() -> tuple[np.ndarray, dict] | None:
    """Return (matrix, meta) if a valid cache exists, else None."""
    if not (CACHE_FILE.exists() and CACHE_META_FILE.exists()):
        return None
    try:
        meta = json.loads(CACHE_META_FILE.read_text())
        matrix = np.load(CACHE_FILE)
    except Exception as e:  # noqa: BLE001
        log.warning("embed: cache read failed (%s); will rebuild", e)
        return None
    return matrix, meta


def _save_cache(matrix: np.ndarray, meta: dict) -> None:
    """Persist the embedded matrix + meta to disk. Parent dir is created
    on first save in case the data/guidelines/ folder doesn't yet
    exist (e.g. fresh checkout)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(CACHE_FILE, matrix)
    CACHE_META_FILE.write_text(json.dumps(meta, indent=2))


def build_corpus_embeddings(
    corpus: list[GuidelineChunk],
    *,
    embedder: Embedder | None = None,
) -> CorpusEmbeddings:
    """Build (or reuse cached) embeddings for the full corpus.

    On first call, embeds every chunk and persists to disk. On subsequent
    calls (within the same process or across restarts), reads from the
    on-disk cache as long as the corpus hash matches. Edits to the
    corpus YAML invalidate the cache automatically.

    `embedder` is exposed for tests; production code uses the default
    `LocalEmbedder()`.
    """
    if not corpus:
        raise ValueError("build_corpus_embeddings: empty corpus")
    chunk_ids = [c.chunk_id for c in corpus]
    chunk_id_to_row = {cid: i for i, cid in enumerate(chunk_ids)}
    expected_hash = _corpus_hash(corpus)

    cached = _load_cache()
    if cached is not None:
        matrix, meta = cached
        if (
            meta.get("corpus_hash") == expected_hash
            and meta.get("model_name") == EMBED_MODEL_NAME
            and matrix.shape == (len(corpus), EMBED_DIM)
            and meta.get("chunk_ids") == chunk_ids
        ):
            log.info(
                "embed: cache hit (%d chunks, model=%s)",
                len(chunk_ids), EMBED_MODEL_NAME,
            )
            return CorpusEmbeddings(
                chunk_ids=chunk_ids,
                chunk_id_to_row=chunk_id_to_row,
                matrix=matrix,
                corpus_hash=expected_hash,
            )
        log.info(
            "embed: cache mismatch (hash %s vs %s); rebuilding",
            meta.get("corpus_hash", "")[:12], expected_hash[:12],
        )

    emb = embedder if embedder is not None else LocalEmbedder()
    surfaces = [_chunk_to_embed_surface(c) for c in corpus]
    matrix = emb.embed_batch(surfaces)
    if matrix.shape != (len(corpus), EMBED_DIM):
        raise RuntimeError(
            f"embed: model returned shape {matrix.shape}, "
            f"expected ({len(corpus)}, {EMBED_DIM})"
        )
    _save_cache(
        matrix,
        {
            "corpus_hash": expected_hash,
            "model_name": EMBED_MODEL_NAME,
            "chunk_ids": chunk_ids,
            "dim": EMBED_DIM,
        },
    )
    log.info(
        "embed: built corpus embeddings (%d chunks, model=%s)",
        len(chunk_ids), EMBED_MODEL_NAME,
    )
    return CorpusEmbeddings(
        chunk_ids=chunk_ids,
        chunk_id_to_row=chunk_id_to_row,
        matrix=matrix,
        corpus_hash=expected_hash,
    )


def dense_candidates(
    query: str,
    embeddings: CorpusEmbeddings,
    pool_size: int,
    *,
    embedder: Embedder | None = None,
) -> list[tuple[str, float]]:
    """Stage-1 dense recall: cosine similarity over the corpus matrix.

    Returns up to `pool_size` `(chunk_id, similarity)` pairs ranked
    descending. Empty if the query is blank.

    Embedding is done with the BGE-style query instruction prefix so the
    query side gets the right primer for retrieval-style similarity.
    Corpus side was embedded WITHOUT a prefix in `build_corpus_embeddings`.
    """
    if not query.strip() or pool_size <= 0:
        return []
    emb = embedder if embedder is not None else LocalEmbedder()
    qvec = emb.embed_batch([_BGE_QUERY_INSTRUCTION + query])
    if qvec.shape != (1, EMBED_DIM):
        raise RuntimeError(
            f"dense_candidates: query embedding shape {qvec.shape} "
            f"!= (1, {EMBED_DIM})"
        )
    # Cosine similarity = dot product on L2-normalized vectors.
    sims = embeddings.matrix @ qvec[0]
    # Argsort descending; cap at pool_size.
    n = min(pool_size, len(embeddings.chunk_ids))
    if n <= 0:
        return []
    top_idx = np.argpartition(-sims, n - 1)[:n] if n < len(sims) else np.arange(len(sims))
    top_idx = top_idx[np.argsort(-sims[top_idx])]
    return [
        (embeddings.chunk_ids[int(i)], float(sims[int(i)]))
        for i in top_idx
    ]
