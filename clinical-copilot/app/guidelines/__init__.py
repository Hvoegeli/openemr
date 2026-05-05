"""Guideline retrieval (Phase 3 RAG MVP).

Hand-curated USPSTF + ADA corpus + BM25 index + a `retrieve_guidelines`
function the agent can call to ground its recommendations in published
guidance with verifiable citations.

The MVP deliberately uses BM25 (not dense embeddings + reranking) so
the dependency footprint stays tiny (`rank-bm25` is pure Python with
zero transitives) and the index loads in milliseconds. The PRD-faithful
hybrid retrieve + BGE rerank pipeline is a post-MVP swap that keeps the
same `retrieve_guidelines` interface.
"""
