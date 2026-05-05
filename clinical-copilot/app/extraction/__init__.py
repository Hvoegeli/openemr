"""Document extraction layer.

Owns the schemas + pipeline that turn uploaded clinical documents (lab PDFs,
intake forms) into strict-typed JSON with per-field source citations. Every
extracted clinical fact carries a `Citation` pointing back to its origin in
the source document — same citation primitive used by the Week 1 chart-
summarizer agent and the Week 2 evidence retriever.
"""
