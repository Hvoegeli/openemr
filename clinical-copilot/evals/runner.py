"""Eval runner: record live agent traces, replay them through deterministic checkers.

Two modes:

  $ python -m evals.runner                # validate (replay all snapshots through rules)
  $ python -m evals.runner --gate         # validate + exit 1 if golden < 100% or labeled < 90%
  $ python -m evals.runner --record CASE  # record a single case (live FHIR + LLM)
  $ python -m evals.runner --record-all   # record every case-run (live)

`--record` and `--record-all` require the OpenEMR docker stack to be up
locally with the demo patients seeded, plus ANTHROPIC_API_KEY set. They also
upload a run to LangSmith if LANGSMITH_API_KEY is set.

`--gate` is what runs in the prek pre-push hook: pure replay, no network
needed beyond a possible LangSmith upload (which silently no-ops on missing
key or network error).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# Load clinical-copilot/.env so LANGSMITH_API_KEY etc. are available when
# running the eval suite as a standalone module.
load_dotenv(Path(__file__).parent.parent / ".env")

from evals.types import (  # noqa: E402
    Case,
    CaseRunResult,
    Patient,
    RuleResult,
    RunResult,
    Snapshot,
    ToolCall,
    Turn,
    TurnSpec,
)

EVALS_DIR = Path(__file__).parent
SNAPSHOT_DIR = EVALS_DIR / "snapshots"
HISTORY_DIR = EVALS_DIR / "history"
CASES_DIR = EVALS_DIR / "cases"
PATIENTS_FILE = EVALS_DIR / "patients.yaml"

GOLDEN_THRESHOLD = 1.0     # 100% — every golden case must pass
LABELED_THRESHOLD = 0.90   # 90% — labeled may have some failures

log = logging.getLogger("evals.runner")


# ── Loading ─────────────────────────────────────────────────────────────


def load_patients() -> list[Patient]:
    data = yaml.safe_load(PATIENTS_FILE.read_text())
    out: list[Patient] = []
    for row in data["patients"]:
        out.append(Patient(
            id=row["id"],
            fname=row["fname"],
            lname=row["lname"],
            dob=row["dob"],
            tags=frozenset(row.get("tags", [])),
            ssn=row.get("ssn"),
            notes=row.get("notes", ""),
        ))
    return out


def load_cases() -> list[Case]:
    out: list[Case] = []
    for path in sorted(CASES_DIR.glob("*.yaml")):
        data = yaml.safe_load(path.read_text())
        severity = data["severity"]
        for row in data["cases"]:
            turns_raw = row["turns"]
            turns = [TurnSpec(user_msg=t["user_msg"]) for t in turns_raw]
            out.append(Case(
                id=row["id"],
                severity=severity,
                description=row.get("description", ""),
                turns=turns,
                patient_selector=row.get("patient_selector", {"tag": "none"}),
                applies_rules=list(row.get("applies_rules", [])),
                must_match_regex=list(row.get("must_match_regex", [])),
                must_not_match_regex=list(row.get("must_not_match_regex", [])),
                notes=row.get("notes", ""),
            ))
    return out


# ── Patient selection ───────────────────────────────────────────────────


def expand_selector(selector: dict[str, Any], all_patients: list[Patient]) -> list[Patient | None]:
    """Return the list of Patients (possibly [None]) this case runs against."""
    if "ids" in selector:
        ids = set(selector["ids"])
        return [p for p in all_patients if p.id in ids]
    tag = selector.get("tag", "none")
    if tag == "none":
        return [None]
    if tag == "all":
        return list(all_patients)
    return [p for p in all_patients if tag in p.tags]


# ── Rule application ────────────────────────────────────────────────────


def apply_rules(snap: Snapshot, case: Case, patient: Patient | None) -> list[RuleResult]:
    from evals.rules import all_rules  # local import: registry populated on import

    results: list[RuleResult] = []
    for rule in all_rules():
        try:
            result = rule.check(snap, case, patient)
        except Exception as e:  # noqa: BLE001
            result = RuleResult(rule_id=rule.id, status="fail", message=f"checker raised: {type(e).__name__}: {e}")
        # Force the rule_id even if the checker forgot to set it
        if not result.rule_id:
            result = RuleResult(rule_id=rule.id, status=result.status, message=result.message)
        results.append(result)

    # Custom regex assertions from the case YAML — folded in as synthetic rule IDs.
    text = "\n\n".join(t.final_response for t in snap.turns) if snap.turns else ""
    for spec in case.must_match_regex:
        if not re.search(spec["pattern"], text, flags=re.IGNORECASE):
            results.append(RuleResult(
                rule_id="case:must_match",
                status="fail",
                message=f"missing required pattern {spec['pattern']!r}: {spec.get('reason', '')}",
            ))
    for spec in case.must_not_match_regex:
        m = re.search(spec["pattern"], text, flags=re.IGNORECASE)
        if m:
            results.append(RuleResult(
                rule_id="case:must_not_match",
                status="fail",
                message=f"found forbidden pattern {spec['pattern']!r} at {m.group(0)!r}: {spec.get('reason', '')}",
            ))

    return results


# ── Snapshot I/O ────────────────────────────────────────────────────────


def snapshot_path(case_id: str, patient: Patient | None) -> Path:
    suffix = patient.id if patient is not None else "no_patient"
    return SNAPSHOT_DIR / f"{case_id}__{suffix}.json"


def load_snapshot(path: Path) -> Snapshot | None:
    if not path.exists():
        return None
    return Snapshot.from_dict(json.loads(path.read_text()))


def save_snapshot(snap: Snapshot, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snap.to_dict(), indent=2, default=str))


# ── Recording (live agent) ──────────────────────────────────────────────


async def record_one(case: Case, patient: Patient | None) -> Snapshot:
    """Run the live agent against this (case × patient) and return a Snapshot."""
    # Local imports — keep module load-time light for the gate path.
    from langchain_core.messages import AIMessage, HumanMessage

    import tempfile
    from pathlib import Path

    from app.agent.graph import build_graph, message_text
    from app.agent.state import AgentState
    from app.clinical_notes import ClinicalNoteStore
    from app.fhir.client import FhirClient

    fhir = FhirClient()
    # Use a temp empty store so eval recordings don't pick up clinical notes
    # left over on the dev box. `get_vital_trends` will see no note vitals
    # and fall back to FHIR-only data, which is what the snapshots assume.
    # Using mkdtemp (no auto-cleanup) instead of TemporaryDirectory because
    # `notes_store` outlives the with-block via `graph`; the leak is bounded
    # to /tmp and `record_one` is a rare manual recording path.
    notes_dir = tempfile.mkdtemp(prefix="eval-notes-")
    notes_store = ClinicalNoteStore(Path(notes_dir) / "clinical_notes.json")
    graph = build_graph(fhir, notes_store)

    state: AgentState = {
        "messages": [],
        "conversation_sources": [],
        "patient_id": None,
        "validation_attempts": 0,
        # Eval replays should see every patient regardless of ACL — they're
        # testing agent behavior, not access control. Pin "admin" so the
        # access_control gate returns panel=None (no filter).
        "username": "admin",
    }

    turns_out: list[Turn] = []

    for spec in case.turns:
        text = render_user_msg(spec.user_msg, patient)
        # Snapshot the message-list length BEFORE this turn so we can
        # extract just the new tool calls.
        prefix_len = len(state["messages"])
        state["messages"].append(HumanMessage(content=text))
        state["validation_attempts"] = 0
        state = await graph.ainvoke(state)
        # Walk the new messages, picking out tool calls + their results.
        new_messages = state["messages"][prefix_len + 1:]
        tool_calls = _extract_tool_calls(new_messages)
        # The final assistant text is the last AIMessage in the new chunk.
        final_text = ""
        for msg in reversed(new_messages):
            if isinstance(msg, AIMessage):
                final_text = message_text(msg)
                break
        turns_out.append(Turn(user_msg=text, tool_calls=tool_calls, final_response=final_text))

    await fhir.aclose()

    return Snapshot(
        case_id=case.id,
        patient_id=patient.id if patient is not None else None,
        recorded_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        agent_sha=_git_sha(),
        model="claude-sonnet-4-6",
        turns=turns_out,
        conversation_sources=list(state["conversation_sources"]),
        validation_attempts=int(state.get("validation_attempts", 0) or 0),
    )


def _extract_tool_calls(new_messages: list) -> list[ToolCall]:
    """Pair each AIMessage's tool_calls with the corresponding ToolMessage results."""
    from langchain_core.messages import AIMessage, ToolMessage

    by_id: dict[str, dict[str, Any]] = {}
    out: list[ToolCall] = []

    for msg in new_messages:
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            for call in msg.tool_calls:
                by_id[call["id"]] = {"name": call["name"], "args": call["args"]}
        elif isinstance(msg, ToolMessage):
            entry = by_id.get(msg.tool_call_id)
            if not entry:
                continue
            try:
                payload = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
            except json.JSONDecodeError:
                payload = {"raw": msg.content}
            if isinstance(payload, dict):
                data = payload.get("data")
                sources = payload.get("sources", [])
            else:
                data, sources = payload, []
            out.append(ToolCall(
                name=entry["name"],
                args=entry["args"],
                result_data=data,
                result_sources=list(sources) if isinstance(sources, list) else [],
            ))
    return out


def render_user_msg(template: str, patient: Patient | None) -> str:
    if patient is None:
        return template
    return (
        template
        .replace("{patient.lname}", patient.lname)
        .replace("{patient.fname}", patient.fname)
        .replace("{patient.dob}", patient.dob)
    )


# ── Replay (deterministic) ──────────────────────────────────────────────


def replay_one(case: Case, patient: Patient | None, snap: Snapshot) -> CaseRunResult:
    rule_results = apply_rules(snap, case, patient)
    return CaseRunResult(
        case_id=case.id,
        patient_id=patient.id if patient is not None else None,
        severity=case.severity,
        rule_results=rule_results,
    )


def run_validate() -> RunResult:
    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    cases = load_cases()
    patients = load_patients()
    case_results: list[CaseRunResult] = []

    for case in cases:
        targets = expand_selector(case.patient_selector, patients)
        for target in targets:
            path = snapshot_path(case.id, target)
            snap = load_snapshot(path)
            if snap is None:
                # Missing snapshot is itself a failure of the eval gate —
                # surface it as a single case-level fail.
                case_results.append(CaseRunResult(
                    case_id=case.id,
                    patient_id=target.id if target is not None else None,
                    severity=case.severity,
                    rule_results=[RuleResult(
                        rule_id="snapshot",
                        status="fail",
                        message=f"missing snapshot {path.name}; run --record {case.id}",
                    )],
                ))
                continue
            case_results.append(replay_one(case, target, snap))

    return RunResult(
        started_at=started_at,
        ended_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        git_sha=_git_sha(),
        case_results=case_results,
    )


# ── Recording orchestration ─────────────────────────────────────────────


async def run_record_all(case_id_filter: str | None = None) -> None:
    cases = load_cases()
    patients = load_patients()
    if case_id_filter:
        cases = [c for c in cases if c.id == case_id_filter]
        if not cases:
            print(f"No case matches id={case_id_filter!r}", file=sys.stderr)
            sys.exit(2)

    for case in cases:
        targets = expand_selector(case.patient_selector, patients)
        for target in targets:
            path = snapshot_path(case.id, target)
            label = f"{case.id} × {target.id if target else 'no_patient'}"
            print(f"  recording {label} → {path.name}")
            try:
                snap = await record_one(case, target)
            except Exception as e:  # noqa: BLE001
                print(f"    ✗ failed: {type(e).__name__}: {e}", file=sys.stderr)
                continue
            save_snapshot(snap, path)
            print(f"    ✓ saved ({len(snap.turns)} turn(s), sources={len(snap.conversation_sources)})")


# ── Helpers ─────────────────────────────────────────────────────────────


def _git_sha() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return "unknown"


# ── CLI ─────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Clinical Co-Pilot eval runner.")
    parser.add_argument("--gate", action="store_true", help="Exit 1 if golden < 100%% or labeled < 90%%.")
    parser.add_argument("--record", metavar="CASE_ID", help="Record a single case live (all its patients).")
    parser.add_argument("--record-all", action="store_true", help="Record every case live.")
    parser.add_argument("--no-langsmith", action="store_true", help="Skip LangSmith upload even if key is set.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.record:
        asyncio.run(run_record_all(args.record))
        return
    if args.record_all:
        asyncio.run(run_record_all(None))
        return

    result = run_validate()

    # Late import — keeps module load-time fast for `--help` and prevents a
    # forced rich/langsmith dependency until they are actually needed.
    from evals.report import print_cli_report, write_history_json, upload_to_langsmith

    print_cli_report(result, GOLDEN_THRESHOLD, LABELED_THRESHOLD)
    write_history_json(result, HISTORY_DIR)

    if not args.no_langsmith:
        upload_to_langsmith(result)

    if args.gate:
        if result.golden_pass_rate() < GOLDEN_THRESHOLD:
            print(f"\n[gate] FAIL — golden pass rate {result.golden_pass_rate():.1%} < {GOLDEN_THRESHOLD:.0%}", file=sys.stderr)
            sys.exit(1)
        if result.labeled_pass_rate() < LABELED_THRESHOLD:
            print(f"\n[gate] FAIL — labeled pass rate {result.labeled_pass_rate():.1%} < {LABELED_THRESHOLD:.0%}", file=sys.stderr)
            sys.exit(1)
        print("\n[gate] OK")


if __name__ == "__main__":
    main()
