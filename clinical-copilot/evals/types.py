"""Dataclasses shared by the eval runner, rules, and reporter.

A run consists of:
  Case (yaml row) + Patient (registry row) → CaseRun → Snapshot (JSON file)
  Snapshot + Rule → RuleResult → CaseRunResult → RunResult

Everything is plain dataclasses so they round-trip through JSON cleanly and
the reporter can serialize without per-class encoders.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


# ── Inputs ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Patient:
    """One row from evals/patients.yaml — a logical demo patient profile.

    Tags are how cases pick which patients they apply to (e.g. a no-bp-charted
    test runs against patients tagged `no_bp_charted`). When you seed a new
    demo patient, add a row here with appropriate tags.
    """
    id: str
    fname: str
    lname: str
    dob: str
    tags: frozenset[str]
    # Optional fields used by certain rules / cases:
    ssn: str | None = None         # if set, F1 checks the response doesn't echo this exact SSN
    notes: str = ""


@dataclass(frozen=True)
class TurnSpec:
    """A single user message in a (potentially multi-turn) case."""
    user_msg: str  # may contain `{patient.fname}` / `{patient.lname}` / `{patient.dob}`


@dataclass(frozen=True)
class Case:
    """One row from a cases/*.yaml file — a case template, not yet bound to a patient.

    `patient_selector` decides which patients this case runs against:
      - {"tag": "all"}                     → every patient in the registry
      - {"tag": "no_bp_charted"}            → patients with that tag
      - {"ids": ["cohen", "roberts"]}       → just those patient ids
      - {"tag": "none"}                     → run once, with no patient (e.g. out-of-scope asks)

    `applies_rules` is the list of rule IDs this case is meant to exercise.
    Rules outside this list are marked N/A for the case in the report.
    """
    id: str
    severity: str  # "golden" | "labeled"
    description: str
    turns: list[TurnSpec]
    patient_selector: dict[str, Any]
    applies_rules: list[str]
    must_match_regex: list[dict[str, str]] = field(default_factory=list)      # [{pattern, reason}]
    must_not_match_regex: list[dict[str, str]] = field(default_factory=list)  # [{pattern, reason}]
    notes: str = ""


# ── Snapshot (cached agent trace) ───────────────────────────────────────


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any]
    result_data: Any
    result_sources: list[str]


@dataclass
class Turn:
    """One user → assistant exchange within a case-run, plus tool calls in between."""
    user_msg: str
    tool_calls: list[ToolCall]
    final_response: str


@dataclass
class Snapshot:
    """A recorded agent trace for one (case × patient) — the unit of replay.

    The snapshot file lives at:
      evals/snapshots/{case_id}__{patient_id}.json

    The runner replays each turn's `final_response` + cumulative tool calls
    through every applicable rule's checker. `conversation_sources` is the
    cumulative source set after all turns — what the citation validator saw.
    """
    case_id: str
    patient_id: str | None  # None for `tag: none` cases
    recorded_at: str
    agent_sha: str
    model: str
    turns: list[Turn]
    conversation_sources: list[str]
    validation_attempts: int
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Snapshot:
        return cls(
            case_id=data["case_id"],
            patient_id=data.get("patient_id"),
            recorded_at=data["recorded_at"],
            agent_sha=data["agent_sha"],
            model=data["model"],
            turns=[
                Turn(
                    user_msg=t["user_msg"],
                    tool_calls=[ToolCall(**tc) for tc in t["tool_calls"]],
                    final_response=t["final_response"],
                )
                for t in data["turns"]
            ],
            conversation_sources=data["conversation_sources"],
            validation_attempts=data["validation_attempts"],
            schema_version=data.get("schema_version", 1),
        )


# ── Results ─────────────────────────────────────────────────────────────


@dataclass
class RuleResult:
    """Outcome of one rule check against one (case × patient) snapshot."""
    rule_id: str
    status: str  # "pass" | "fail" | "na"
    message: str = ""  # short explanation when status="fail"


@dataclass
class CaseRunResult:
    """Outcome of all rules for one (case × patient) snapshot.

    Note that custom-regex checks declared in the case YAML are folded into
    `rule_results` under synthetic rule IDs `case:must_match` / `case:must_not_match`.
    """
    case_id: str
    patient_id: str | None
    severity: str
    rule_results: list[RuleResult]

    @property
    def passed(self) -> bool:
        return all(r.status != "fail" for r in self.rule_results)

    @property
    def failures(self) -> list[RuleResult]:
        return [r for r in self.rule_results if r.status == "fail"]


@dataclass
class RunResult:
    """Outcome of an entire eval run."""
    started_at: str
    ended_at: str
    git_sha: str
    case_results: list[CaseRunResult]

    def by_severity(self, severity: str) -> list[CaseRunResult]:
        return [c for c in self.case_results if c.severity == severity]

    def golden_pass_rate(self) -> float:
        rows = self.by_severity("golden")
        if not rows:
            return 1.0
        return sum(1 for r in rows if r.passed) / len(rows)

    def labeled_pass_rate(self) -> float:
        rows = self.by_severity("labeled")
        if not rows:
            return 1.0
        return sum(1 for r in rows if r.passed) / len(rows)
