"""Pydantic schemas for guideline-corpus chunks + retrieval results.

Both shapes use `extra="forbid"` so a typo in `data/guidelines/corpus.yaml`
fails loudly at corpus-load time rather than silently dropping a field.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class GuidelineChunk(BaseModel):
    """One quotable snippet from a clinical practice guideline.

    `text` is the snippet the agent may quote in its response. Citations
    are constructed by combining `source` + `title` + `year` + `url`
    (the same shape as the Phase 1 Citation primitive's
    `quote_or_value` + `source_id` + `page_or_section` triplet, but
    specialized to the guideline source type so the LLM doesn't have
    to infer it from free-text)."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(min_length=1, description="Stable slug — primary key for the corpus")
    source: str = Field(min_length=1, description="Publishing organization, e.g. 'USPSTF', 'ADA'")
    title: str = Field(min_length=1, description="Canonical recommendation/section title")
    year: str = Field(min_length=1, description="Publication year as a string (e.g. '2024')")
    url: str = Field(min_length=1, description="Canonical URL for the citation")
    topic_tags: list[str] = Field(
        default_factory=list,
        description="Short labels for filtering and human inspection",
    )
    text: str = Field(min_length=1, description="Quotable snippet (≤200 words)")


class RetrievalHit(BaseModel):
    """One match returned by `retrieve_guidelines` — after BM25 + rerank.

    `score` is the rerank stage's relevance score (0.0-1.0). It is the
    authoritative ordering signal and what the agent should use when
    breaking ties or filtering low-confidence hits.

    `bm25_score` is the BM25 score from the prior stage, kept for
    debugging and for tools that want to inspect the recall filter's
    ranking. It is NOT the field that determines `rank`.

    Neither score is suitable for forming absolute confidence claims
    in the user-facing response — both are calibrated only within a
    single result set."""

    model_config = ConfigDict(extra="forbid")

    chunk: GuidelineChunk
    score: float = Field(
        ge=0,
        description="Rerank relevance score (0.0-1.0); the field that determines ordering",
    )
    rank: int = Field(ge=1, description="1-indexed position within the result set")
    bm25_score: float = Field(
        default=0.0, ge=0,
        description="Stage-1 BM25 score (informational); not used for ordering",
    )
