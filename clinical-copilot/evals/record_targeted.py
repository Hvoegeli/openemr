"""Surgical eval recording.

The default `--record CASE_ID` re-records every patient matching the case's
selector, which clobbers already-passing snapshots and burns credits. Use
this when you need to record only specific (case_id, patient_id) tuples —
e.g. after a system-prompt change that affects one case × one patient.

Usage:
  cd clinical-copilot
  PYTHONPATH=. uv run python -m evals.record_targeted \\
      --pair politeness_no_role_change patel \\
      --pair politeness_no_role_change hale \\
      --pair pediatric_dose_boundary  hale \\
      --pair lab_normal               cohen

Each `--pair CASE PATIENT` becomes one live agent run; the snapshot is
written to `evals/snapshots/<case>__<patient>.json` (same naming as the
default runner). Patient id `no_patient` records a case with no patient
selector.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from evals.runner import (
    load_cases,
    load_patients,
    record_one,
    save_snapshot,
    snapshot_path,
)


log = logging.getLogger("evals.record_targeted")


async def record_pairs(pairs: list[tuple[str, str]]) -> None:
    cases_by_id = {c.id: c for c in load_cases()}
    patients_by_id = {p.id: p for p in load_patients()}

    for case_id, patient_id in pairs:
        case = cases_by_id.get(case_id)
        if case is None:
            print(f"  ✗ unknown case_id={case_id!r}", file=sys.stderr)
            continue
        patient = None if patient_id == "no_patient" else patients_by_id.get(patient_id)
        if patient_id != "no_patient" and patient is None:
            print(f"  ✗ unknown patient_id={patient_id!r} (for case={case_id})", file=sys.stderr)
            continue

        path = snapshot_path(case_id, patient)
        label = f"{case_id} × {patient_id}"
        print(f"  recording {label} → {path.name}")
        try:
            snap = await record_one(case, patient)
        except Exception as e:  # noqa: BLE001
            print(f"    ✗ failed: {type(e).__name__}: {e}", file=sys.stderr)
            continue
        save_snapshot(snap, path)
        print(f"    ✓ saved ({len(snap.turns)} turn(s), sources={len(snap.conversation_sources)})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Record specific (case, patient) tuples without clobbering siblings.")
    parser.add_argument(
        "--pair",
        action="append",
        nargs=2,
        metavar=("CASE_ID", "PATIENT_ID"),
        required=True,
        help="A (case_id, patient_id) pair to record. Repeatable. Use patient_id='no_patient' for case-without-patient.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    pairs: list[tuple[str, str]] = [(c, p) for c, p in args.pair]
    asyncio.run(record_pairs(pairs))


if __name__ == "__main__":
    main()
