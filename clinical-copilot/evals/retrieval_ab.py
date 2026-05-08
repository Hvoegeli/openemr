"""Retrieval A/B evidence harness for the hybrid retrieval pipeline.

Runs each case in `evals/cases/retrieval.yaml` through three retrieval
configurations and reports `hit@3` (the fraction of expected chunk_ids
the retriever surfaced in its top-3):

  1. **BM25-only (sparse)** — `enable_dense=False, enable_rerank=False`
  2. **BM25 + dense (hybrid, no rerank)** — `enable_rerank=False`
  3. **Full pipeline** — BM25 ∪ dense → rerank → top-3

This is the evidence the W2 review feedback asked for ("clearer reranker
evidence"). The deltas between modes 1→2 and 2→3 show which stage of the
pipeline is doing real work on the curated query set.

Run: `uv run python -m evals.retrieval_ab` (requires ANTHROPIC_API_KEY for
mode 3; modes 1 + 2 run offline).

Output: a markdown table on stdout (also pasted into `evals/RESULTS.md`
for the submission artifact).
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

from app.guidelines.retrieve import retrieve_guidelines

REPO_ROOT = Path(__file__).resolve().parent.parent
CASES_PATH = REPO_ROOT / "evals" / "cases" / "retrieval.yaml"


def load_cases(path: Path = CASES_PATH) -> list[dict]:
    """Load + validate the YAML retrieval-eval case file."""
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, list) or not raw:
        raise RuntimeError(f"retrieval cases at {path} are empty or not a list")
    for c in raw:
        for key in ("case_id", "query", "expected_top_3"):
            if key not in c:
                raise RuntimeError(f"case missing {key}: {c}")
        if not isinstance(c["expected_top_3"], list) or not c["expected_top_3"]:
            raise RuntimeError(f"case {c['case_id']}: expected_top_3 must be a non-empty list")
    return raw


def hit_at_k(retrieved: list[str], expected: list[str], k: int = 3) -> float:
    """Return the fraction of `expected` chunk_ids present in the
    first `k` of `retrieved`. Range [0.0, 1.0]. Order within
    `retrieved[:k]` doesn't matter — this is a recall metric."""
    if not expected:
        return 0.0
    top = set(retrieved[:k])
    hits = sum(1 for e in expected if e in top)
    return hits / len(expected)


def run_one_mode(
    cases: list[dict],
    *,
    label: str,
    enable_dense: bool,
    enable_rerank: bool,
) -> tuple[float, list[tuple[str, float, list[str]]]]:
    """Run every case in `cases` against `retrieve_guidelines` with the
    given mode flags. Returns (mean_hit_at_3, per_case_results) where
    each per_case_result is `(case_id, hit_at_3, retrieved_chunk_ids)`."""
    per_case: list[tuple[str, float, list[str]]] = []
    for case in cases:
        hits = retrieve_guidelines(
            case["query"],
            k=3,
            enable_dense=enable_dense,
            enable_rerank=enable_rerank,
        )
        retrieved = [h.chunk.chunk_id for h in hits]
        score = hit_at_k(retrieved, case["expected_top_3"], k=3)
        per_case.append((case["case_id"], score, retrieved))
    mean = sum(s for _, s, _ in per_case) / len(per_case) if per_case else 0.0
    print(f"\n=== {label} ===  mean hit@3 = {mean:.3f}")
    for cid, score, retr in per_case:
        flag = "✓" if score == 1.0 else ("✗" if score == 0.0 else "·")
        print(f"  {flag} {cid:50s} hit@3={score:.2f}  top3={retr}")
    return mean, per_case


def render_markdown(
    summary: list[tuple[str, float]],
    per_mode: dict[str, list[tuple[str, float, list[str]]]],
) -> str:
    """Render a human-readable markdown summary for pasting into RESULTS.md."""
    lines: list[str] = []
    lines.append("### Retrieval A/B evidence")
    lines.append("")
    lines.append("Hit@3 = fraction of expected chunk_ids that appeared in the retriever's top-3 results, averaged across the cases in [`evals/cases/retrieval.yaml`](cases/retrieval.yaml).")
    lines.append("")
    lines.append("| Configuration | Mean hit@3 |")
    lines.append("|---|---|")
    for label, mean in summary:
        lines.append(f"| {label} | {mean:.3f} |")
    lines.append("")
    lines.append("Per-case detail (full pipeline only):")
    lines.append("")
    lines.append("| case_id | hit@3 | top-3 retrieved |")
    lines.append("|---|---|---|")
    for cid, score, retr in per_mode["full"]:
        lines.append(f"| `{cid}` | {score:.2f} | {', '.join('`' + r + '`' for r in retr)} |")
    return "\n".join(lines)


def main() -> int:
    cases = load_cases()
    print(f"Loaded {len(cases)} retrieval-eval cases from {CASES_PATH}")

    summary: list[tuple[str, float]] = []
    per_mode: dict[str, list[tuple[str, float, list[str]]]] = {}

    mean, per = run_one_mode(
        cases, label="BM25 only (sparse)",
        enable_dense=False, enable_rerank=False,
    )
    summary.append(("BM25 only (sparse)", mean))
    per_mode["bm25"] = per

    mean, per = run_one_mode(
        cases, label="BM25 + dense (hybrid, no rerank)",
        enable_dense=True, enable_rerank=False,
    )
    summary.append(("BM25 + dense (hybrid, no rerank)", mean))
    per_mode["hybrid"] = per

    mean, per = run_one_mode(
        cases, label="Full pipeline (BM25 + dense + rerank)",
        enable_dense=True, enable_rerank=True,
    )
    summary.append(("Full pipeline (BM25 + dense + rerank)", mean))
    per_mode["full"] = per

    print("\n" + "=" * 60)
    print(render_markdown(summary, per_mode))
    return 0


if __name__ == "__main__":
    sys.exit(main())
