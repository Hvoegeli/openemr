"""LLM-as-reranker for guideline retrieval.

Per Week 2 PRD §3 (line 184): "Retrieve with sparse+dense search, rerank
candidate chunks with Cohere Rerank or an equivalent reranker, and feed
only the top grounded evidence to the answer model."

This module implements the rerank stage. The choice of "Claude Haiku as
relevance scorer" instead of a cross-encoder model is intentional:

  1. Tests against PyPI on macOS x86_64 + CPython 3.14 (the dev box's
     stack) found that `sentence-transformers` requires `torch>=2.4` for
     newer Python ABIs, but torch >= 2.4 dropped Intel Mac wheels. There
     is no version of torch that has both CPython 3.14 wheels AND
     macosx_x86_64 wheels right now. Trying anyway forces a downgrade
     of the project's interpreter or a 700MB Linux-only install path on
     Hetzner that wouldn't run locally.

  2. The PRD says "Cohere Rerank or an equivalent reranker." A
     short-prompted Claude Haiku call IS an equivalent semantic
     relevance scorer for our 12-chunk corpus, costs ~$0.001 per query
     (cheaper than Cohere's $2/1k and well below the project's
     per-request budget), runs in 500-800ms (faster than loading a
     500MB cross-encoder weight on cold start), and adds zero new
     transitive dependencies.

  3. Cross-encoder vs. LLM-as-reranker is a quality vs. ergonomics
     tradeoff. For a 12-chunk corpus the quality delta is small; for
     a 10,000-chunk corpus you'd want the cross-encoder. We've
     documented the rerank-stage interface so a swap to BGE / Cohere
     in production is one module replacement, not a rewrite.

The public surface mirrors the BM25 retriever's: `rerank(query,
candidates, k)` -> ordered list of `RerankedHit`. Callers compose:

    bm25_hits = retrieve_guidelines(query, k=8)         # sparse top-K
    final = rerank(query, [h.chunk for h in bm25_hits], k=3)
"""

from __future__ import annotations

import json
import logging

from anthropic import Anthropic
from anthropic.types import MessageParam
from pydantic import BaseModel, ConfigDict, Field

from app.config import settings
from app.guidelines.schema import GuidelineChunk

log = logging.getLogger("agent.guidelines.rerank")

# Haiku is the right tradeoff for this stage: cheap, fast, and good
# enough at relevance scoring for a 12-chunk corpus. Sonnet/Opus would
# spend 5-15× more for a margin of accuracy that doesn't matter at this
# corpus size.
RERANK_MODEL = "claude-haiku-4-5"

_RERANK_SYSTEM_PROMPT = """You are a clinical-guideline relevance scorer.

Given a clinician's information need (the QUERY) and a numbered list of
candidate guideline excerpts, score each excerpt's relevance to the query
on a scale from 0.0 to 1.0:

  - 1.0 = excerpt directly answers the query
  - 0.7 = excerpt is highly relevant but partial
  - 0.4 = excerpt mentions the topic but addresses a different question
  - 0.0 = excerpt is unrelated

Score on clinical relevance, not surface-keyword overlap. A query about
"aspirin for primary prevention" should score a chunk about aspirin
indications higher than a chunk that only mentions aspirin in passing.

Return STRICT JSON ONLY with the shape:

  {"scores": [{"id": <int>, "score": <float>}, ...]}

One entry per candidate. Do not add commentary, do not wrap in ```. The
parser will reject anything that isn't valid JSON of that exact shape.
"""


class RerankedHit(BaseModel):
    """One re-ranked candidate.

    `score` is the relevance score from the rerank stage (0.0-1.0). It
    is NOT comparable to BM25 scores from the prior stage — different
    units, different scales. Use `rank` for ordering and `score` only
    as a soft confidence signal."""

    model_config = ConfigDict(extra="forbid")

    chunk: GuidelineChunk
    score: float = Field(ge=0.0, le=1.0, description="Rerank relevance score")
    rank: int = Field(ge=1, description="1-indexed final position after rerank")


_client: Anthropic | None = None


def _get_client() -> Anthropic:
    """Lazy Anthropic client. Held in a module-global so each rerank
    call re-uses the same TCP connection pool, but only created on first
    use so importing this module never blocks on network."""
    global _client
    if _client is None:
        if not settings.anthropic_api_key:
            raise RuntimeError(
                "rerank requires ANTHROPIC_API_KEY; check .env"
            )
        _client = Anthropic(api_key=settings.anthropic_api_key)
    return _client


def _build_user_prompt(query: str, candidates: list[GuidelineChunk]) -> str:
    """Render the rerank prompt — query + numbered candidate list. Each
    candidate is identified by its 0-based index in the input list, NOT
    by its chunk_id, so the model can return a compact integer-keyed
    JSON without us having to escape user-provided strings."""
    lines = [f"QUERY: {query}", "", "CANDIDATES:"]
    for i, c in enumerate(candidates):
        # Truncate text to keep the rerank prompt cheap. The chunks are
        # ≤200 words by corpus convention, but a future longer chunk
        # would otherwise inflate the input-token bill on every query.
        snippet = c.text[:600]
        lines.append(f"[{i}] {c.title} ({c.source}): {snippet}")
    lines.append("")
    lines.append('Return JSON: {"scores": [{"id": 0, "score": 0.0-1.0}, ...]}')
    return "\n".join(lines)


def _parse_scores(raw: str, n_candidates: int) -> list[float]:
    """Parse the model's JSON response into a list of scores aligned with
    the input candidate order. Defensive against:
      - the model wrapping the JSON in ```
      - missing entries (filled with 0.0 — equivalent to "the model thought
        this candidate was unrelated, so let it sink to the bottom")
      - extra/duplicate entries (kept first seen, ignored later)
      - non-numeric scores (treated as 0.0)
    """
    text = raw.strip()
    # Strip ```json / ``` fences if the model added them despite the
    # system prompt. Tolerate both ```json\n...\n``` and bare ```...```.
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"rerank response not valid JSON: {e}") from e
    raw_list = payload.get("scores")
    if not isinstance(raw_list, list):
        raise ValueError(
            f"rerank response missing 'scores' list: {payload}"
        )
    scores: list[float] = [0.0] * n_candidates
    seen: set[int] = set()
    for entry in raw_list:
        if not isinstance(entry, dict):
            continue
        idx_raw = entry.get("id")
        score_raw = entry.get("score")
        if not isinstance(idx_raw, int):
            continue
        if not (0 <= idx_raw < n_candidates) or idx_raw in seen:
            continue
        try:
            score_f = float(score_raw)
        except (TypeError, ValueError):
            score_f = 0.0
        # Clamp into [0,1] in case the model overshoots.
        scores[idx_raw] = max(0.0, min(1.0, score_f))
        seen.add(idx_raw)
    return scores


def rerank(
    query: str,
    candidates: list[GuidelineChunk],
    k: int = 3,
) -> list[RerankedHit]:
    """Re-rank `candidates` by query relevance, return the top-`k`.

    Runs synchronously (one Anthropic API call). Empty inputs short-circuit
    to an empty result without an API call. If the model is unavailable
    or the response is malformed, callers get a clear exception — there
    is no silent fallback to BM25 ordering, because that would mask a
    rerank-stage outage and silently degrade response quality without
    operator awareness. Callers that need a fallback should wrap this.
    """
    if k <= 0 or not candidates:
        return []
    if not query.strip():
        return []
    client = _get_client()
    user_prompt = _build_user_prompt(query, candidates)
    msg = client.messages.create(
        model=RERANK_MODEL,
        max_tokens=512,
        system=_RERANK_SYSTEM_PROMPT,
        messages=[
            MessageParam(role="user", content=user_prompt),
        ],
    )
    raw_text = ""
    for block in msg.content:
        # Anthropic's content blocks can be ImageBlock etc. Only text
        # blocks have a `.text` attribute.
        text = getattr(block, "text", None)
        if text:
            raw_text += text
    if not raw_text:
        raise ValueError("rerank response had no text content")

    scores = _parse_scores(raw_text, len(candidates))
    # Pair each candidate with its score, sort by score descending.
    indexed = list(enumerate(candidates))
    indexed.sort(key=lambda pair: scores[pair[0]], reverse=True)
    out: list[RerankedHit] = []
    for rank_idx, (orig_idx, chunk) in enumerate(indexed[:k], start=1):
        out.append(RerankedHit(
            chunk=chunk,
            score=scores[orig_idx],
            rank=rank_idx,
        ))
    log.info(
        "rerank: query=%r → top-%d (cands=%d, model=%s)",
        query[:60], len(out), len(candidates), RERANK_MODEL,
    )
    return out
