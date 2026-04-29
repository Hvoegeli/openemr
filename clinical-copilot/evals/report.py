"""Reporting outputs for an eval run: CLI, local JSON history, LangSmith.

Three sinks, each independent:

  - print_cli_report:    rich-formatted table to stdout, what you see on push
  - write_history_json:  one JSON file per run under evals/history/, gitignored
  - upload_to_langsmith: posts the run as an Experiment to LangSmith if
                         LANGSMITH_API_KEY is set; silently no-ops otherwise

LangSmith is the cloud system of record for trends + diffs. The local JSON
is a fallback so you can `cat` the most recent run when offline or when
LangSmith is having a bad day.
"""

from __future__ import annotations

import json
import logging
import os
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from evals.types import RunResult

log = logging.getLogger("evals.report")


# ── CLI report ──────────────────────────────────────────────────────────


def print_cli_report(result: RunResult, golden_threshold: float, labeled_threshold: float) -> None:
    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        _print_summary_rich(console, result, golden_threshold, labeled_threshold)

        failures = [
            (c, r) for c in result.case_results for r in c.failures
        ]
        if not failures:
            return

        table = Table(title="Failures", show_lines=False)
        table.add_column("Severity", style="bold")
        table.add_column("Case × Patient")
        table.add_column("Rule")
        table.add_column("Why")

        for case, rule_result in failures:
            sev_style = "red" if case.severity == "golden" else "yellow"
            label = f"{case.case_id} × {case.patient_id or '—'}"
            table.add_row(
                f"[{sev_style}]{case.severity}[/{sev_style}]",
                label,
                rule_result.rule_id,
                rule_result.message,
            )
        console.print(table)
        console.print(
            "  [dim]Snapshot files referenced above live in "
            "[blue]clinical-copilot/evals/snapshots/[/blue][/dim]"
        )
    except ImportError:
        # Rich not installed (shouldn't happen post-uv-sync) — fall back to plain.
        _print_summary_plain(result, golden_threshold, labeled_threshold)


def _print_summary_rich(console, result: RunResult, golden_threshold: float, labeled_threshold: float) -> None:
    golden = result.by_severity("golden")
    labeled = result.by_severity("labeled")
    g_pass = sum(1 for r in golden if r.passed)
    l_pass = sum(1 for r in labeled if r.passed)

    g_rate = result.golden_pass_rate()
    l_rate = result.labeled_pass_rate()

    g_marker = "✓" if g_rate >= golden_threshold else "✗"
    l_marker = "✓" if l_rate >= labeled_threshold else "✗"

    g_color = "green" if g_rate >= golden_threshold else "red"
    l_color = "green" if l_rate >= labeled_threshold else "red"

    console.print(
        f"\n[bold]Eval Run[/bold]  ·  [dim]sha={result.git_sha}[/dim]  ·  "
        f"[dim]{result.started_at}[/dim]"
    )
    console.rule()
    console.print(
        f"  [bold]Golden[/bold]   "
        f"[{g_color}]{g_marker} {g_pass}/{len(golden)}  ({g_rate:.0%})[/{g_color}]   "
        f"[dim]threshold {golden_threshold:.0%}[/dim]"
    )
    console.print(
        f"  [bold]Labeled[/bold]  "
        f"[{l_color}]{l_marker} {l_pass}/{len(labeled)}  ({l_rate:.0%})[/{l_color}]   "
        f"[dim]threshold {labeled_threshold:.0%}[/dim]"
    )
    console.print()


def _print_summary_plain(result: RunResult, golden_threshold: float, labeled_threshold: float) -> None:
    golden = result.by_severity("golden")
    labeled = result.by_severity("labeled")
    g_pass = sum(1 for r in golden if r.passed)
    l_pass = sum(1 for r in labeled if r.passed)

    print(f"\nEval Run · sha={result.git_sha} · {result.started_at}")
    print("-" * 70)
    print(f"  Golden   {g_pass}/{len(golden)}  ({result.golden_pass_rate():.0%})  threshold {golden_threshold:.0%}")
    print(f"  Labeled  {l_pass}/{len(labeled)}  ({result.labeled_pass_rate():.0%})  threshold {labeled_threshold:.0%}\n")

    failures = [(c, r) for c in result.case_results for r in c.failures]
    if not failures:
        return
    print(f"Failures ({len(failures)}):")
    for case, rule_result in failures:
        label = f"{case.case_id} × {case.patient_id or '—'}"
        print(f"  [{case.severity}] {label}  {rule_result.rule_id}  {rule_result.message}")


# ── JSON history ────────────────────────────────────────────────────────


def write_history_json(result: RunResult, history_dir: Path) -> Path:
    history_dir.mkdir(parents=True, exist_ok=True)
    name = result.started_at.replace(":", "-").replace("+00-00", "Z") + ".json"
    out = history_dir / name
    out.write_text(json.dumps(_serialize_run(result), indent=2, default=str))
    log.info("wrote history: %s", out)
    return out


def _serialize_run(result: RunResult) -> dict:
    rule_counter: Counter[tuple[str, str]] = Counter()
    for case in result.case_results:
        for rr in case.rule_results:
            rule_counter[(rr.rule_id, rr.status)] += 1
    by_rule: dict[str, dict[str, int]] = {}
    for (rid, status), n in rule_counter.items():
        by_rule.setdefault(rid, {"pass": 0, "fail": 0, "na": 0})[status] = n

    return {
        "started_at": result.started_at,
        "ended_at": result.ended_at,
        "git_sha": result.git_sha,
        "summary": {
            "golden_pass_rate": result.golden_pass_rate(),
            "labeled_pass_rate": result.labeled_pass_rate(),
            "golden_total": len(result.by_severity("golden")),
            "labeled_total": len(result.by_severity("labeled")),
        },
        "by_rule": by_rule,
        "case_results": [
            {
                "case_id": c.case_id,
                "patient_id": c.patient_id,
                "severity": c.severity,
                "passed": c.passed,
                "rule_results": [asdict(rr) for rr in c.rule_results],
            }
            for c in result.case_results
        ],
    }


# ── LangSmith uploader ──────────────────────────────────────────────────


def upload_to_langsmith(result: RunResult) -> None:
    """Upload an experiment-style run to LangSmith. Silent no-op if not configured."""
    api_key = os.environ.get("LANGSMITH_API_KEY")
    project = os.environ.get("LANGSMITH_PROJECT", "clinical-copilot")
    if not api_key:
        log.info("langsmith: no LANGSMITH_API_KEY set; skipping upload.")
        return

    try:
        from langsmith import Client
    except ImportError:
        log.info("langsmith: package not installed; skipping upload.")
        return

    try:
        client = Client(api_key=api_key)
        # Create / reuse a dataset named after the project. Each run becomes
        # one Run on the dataset, with score = pass-rate.
        dataset_name = f"{project}__evals"
        try:
            ds = client.read_dataset(dataset_name=dataset_name)
        except Exception:  # noqa: BLE001 — read fails if dataset doesn't exist yet
            ds = client.create_dataset(
                dataset_name=dataset_name,
                description="Clinical Co-Pilot eval suite — golden + labeled results.",
            )

        run_name = f"sha={result.git_sha} · {result.started_at}"
        # Use create_examples_from_runs / create_run depending on SDK version;
        # both are supported, but the simplest write is `create_run` with a
        # reference to the dataset.
        client.create_run(
            name=run_name,
            run_type="chain",
            project_name=project,
            inputs={"git_sha": result.git_sha, "started_at": result.started_at},
            outputs={
                "golden_pass_rate": result.golden_pass_rate(),
                "labeled_pass_rate": result.labeled_pass_rate(),
                "golden_total": len(result.by_severity("golden")),
                "labeled_total": len(result.by_severity("labeled")),
                "failed_cases": [
                    f"{c.case_id} × {c.patient_id or '—'}"
                    for c in result.case_results if not c.passed
                ],
            },
            extra={
                "tags": ["clinical-copilot", "eval-suite", f"sha:{result.git_sha}"],
                "metadata": {"dataset": dataset_name},
            },
        )
        log.info("langsmith: uploaded run %s to project %s", run_name, project)
    except Exception as e:  # noqa: BLE001 — never fail the gate on LangSmith errors
        log.warning("langsmith: upload failed (%s: %s); continuing.", type(e).__name__, e)
