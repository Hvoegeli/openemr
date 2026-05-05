"""Smoke test for guideline retrieval.

No network, no API credits, no OpenEMR — purely loads the corpus and
runs a handful of representative queries to print what comes back.
Use this for a fast sanity check after editing `data/guidelines/corpus.yaml`
or `app/guidelines/retrieve.py`.

Run:
  cd clinical-copilot && PYTHONPATH=. uv run python scripts/smoke_retrieve.py
  cd clinical-copilot && PYTHONPATH=. uv run python scripts/smoke_retrieve.py "your query"
"""

from __future__ import annotations

import sys

from app.guidelines.retrieve import CORPUS, retrieve_guidelines


DEMO_QUERIES = [
    "type 2 diabetes hba1c target older patient",
    "should i start a statin for primary prevention",
    "aspirin primary prevention 65 year old",
    "colorectal cancer screening colonoscopy",
    "blood pressure target diabetes ckd",
]


def _print_hit(h, body_chars: int = 200) -> None:
    body = h.chunk.text.strip().replace("\n", " ")
    if len(body) > body_chars:
        body = body[:body_chars - 1] + "…"
    print(
        f"  [{h.rank}] {h.chunk.chunk_id}  score={h.score:.2f}\n"
        f"      {h.chunk.source} {h.chunk.year} — {h.chunk.title}\n"
        f"      {body}\n"
        f"      url: {h.chunk.url}"
    )


def _run_query(q: str, k: int = 3) -> None:
    print(f"\nQUERY: {q}")
    hits = retrieve_guidelines(q, k=k)
    if not hits:
        print("  (no hits — try different terms)")
        return
    for h in hits:
        _print_hit(h)


def main() -> int:
    print(f"corpus size: {len(CORPUS)} chunks")
    if len(sys.argv) > 1:
        # User-supplied query
        _run_query(" ".join(sys.argv[1:]), k=5)
    else:
        for q in DEMO_QUERIES:
            _run_query(q, k=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
