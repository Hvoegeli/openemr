"""Lightweight chart-internal rule engine — surfaces fact pairs, not advice.

The brief calls for "domain constraint enforcement: clinical rules, dosage
thresholds, interaction flags." The user-facing requirement is that the
agent be **aware** of these constraints and refuse responses that violate
them — not that the agent recommend clinical actions.

This module operationalizes that narrowly. Each rule is a pure function of
the chart slice (active meds, recent labs, active problems, allergies) that
returns a `Flag` when a well-known fact pair is present. The flag's
`summary` is factual and cites chart evidence — never a recommendation.

  Example output (for a patient with metformin + eGFR 22):
    "Metformin prescribed [MedicationRequest/abc]; latest eGFR 22 mL/min
     [Observation/xyz]."

  NOT (this is what we explicitly do not produce):
    "Consider holding metformin given eGFR < 30."

The doctor's training tells her what to do with the pair. The agent's job
is to make sure she sees both facts together, with provenance.

# Design constraints

  1. Rules fire only on data already in the chart. No external clinical
     knowledge embedded in the rule body beyond the rule's existence.
  2. Every flag.summary embeds `[ResourceType/ID]` citations matching the
     `evidence` list, so the citation validator accepts them.
  3. Rules are conservative and few. The cost of a false positive (flagging
     a benign chart pattern as concerning) is high — it teaches the doctor
     to ignore flags. Better to flag less than to flag wrong.

# Adding a rule

  1. Add a rule function below — pure function of (meds, labs, problems,
     allergies) → Flag | None.
  2. Add it to `_RULES` so `evaluate_chart` includes it.
  3. Add eval cases: one snapshot where it should fire, one where it
     shouldn't, on the same patient profile.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class Flag:
    rule_id: str
    summary: str
    evidence: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


# Lab-name fragments used for fallback matching when the FHIR resource
# doesn't carry a clean LOINC code. OpenEMR's coding completeness varies by
# data source, so we accept either LOINC or a display-name substring.
_EGFR_LOINCS = {"33914-3", "48642-3", "48643-1", "62238-1", "98979-8"}
_EGFR_FRAGMENTS = ("egfr", "estimated glomerular", "estimated gfr")
_POTASSIUM_LOINCS = {"6298-4", "2823-3"}
_POTASSIUM_FRAGMENTS = ("potassium",)
_INR_LOINCS = {"6301-6", "34714-6"}
_INR_FRAGMENTS = ("inr", "international normalized")
_CREATININE_LOINCS = {"2160-0"}
_CREATININE_FRAGMENTS = ("creatinine",)


def _med_name_contains(meds: list[dict], *fragments: str) -> dict | None:
    """First active-medication entry whose `drug` name contains any fragment.

    Case-insensitive substring match — robust to "Metformin HCl 500mg" vs
    "metformin hydrochloride", which OpenEMR's free-text dose entry produces."""
    for m in meds:
        name = (m.get("drug") or "").lower()
        if name and any(f in name for f in fragments):
            return m
    return None


def _allergy_substance_contains(allergies: list[dict], *fragments: str) -> dict | None:
    for a in allergies:
        substance = (a.get("substance") or "").lower()
        if substance and any(f in substance for f in fragments):
            return a
    return None


def _latest_lab(
    observations: list[dict],
    loincs: set[str] | None = None,
    name_fragments: tuple[str, ...] = (),
) -> dict | None:
    """Most recent Observation matching the LOINC set OR a display-name
    fragment, whichever lands first.

    Observations should already be sorted desc by time; if not, we sort here.
    Returns the row in the format `_format_vital` produces (id, name, value,
    unit, time)."""
    if not observations:
        return None
    matches: list[dict] = []
    for o in observations:
        loinc = o.get("loinc")  # adapter doesn't currently surface this; reserved
        name = (o.get("name") or "").lower()
        if loincs and loinc in loincs:
            matches.append(o)
            continue
        if name_fragments and any(f in name for f in name_fragments):
            matches.append(o)
    if not matches:
        return None
    matches.sort(key=lambda x: x.get("time") or "", reverse=True)
    return matches[0]


# ─── rules ────────────────────────────────────────────────────────────────


def _rule_metformin_low_egfr(
    meds: list[dict], labs: list[dict], problems: list[dict], allergies: list[dict],
) -> Flag | None:
    """Metformin prescribed AND latest eGFR < 30 (FDA black-box threshold).

    Threshold is the well-known FDA boxed-warning cutoff; we surface the pair,
    not the action. eGFR units are mL/min/1.73m² — we don't normalize, just
    report what the lab returned."""
    metformin = _med_name_contains(meds, "metformin")
    if metformin is None:
        return None
    egfr = _latest_lab(labs, _EGFR_LOINCS, _EGFR_FRAGMENTS)
    if egfr is None or egfr.get("value") is None:
        return None
    try:
        if float(egfr["value"]) >= 30:
            return None
    except (TypeError, ValueError):
        return None
    unit = egfr.get("unit") or "mL/min/1.73m²"
    return Flag(
        rule_id="METFORMIN_LOW_EGFR",
        summary=(
            f"Metformin prescribed [MedicationRequest/{metformin['id']}]; "
            f"latest eGFR {egfr['value']} {unit} [Observation/{egfr['id']}]."
        ),
        evidence=[
            f"MedicationRequest/{metformin['id']}",
            f"Observation/{egfr['id']}",
        ],
    )


_ACE_ARB_FRAGMENTS = (
    "lisinopril", "enalapril", "ramipril", "captopril", "benazepril",
    "quinapril", "fosinopril", "perindopril", "trandolapril",
    "losartan", "valsartan", "olmesartan", "irbesartan", "candesartan",
    "telmisartan", "azilsartan",
)


def _rule_ace_arb_hyperkalemia(
    meds: list[dict], labs: list[dict], problems: list[dict], allergies: list[dict],
) -> Flag | None:
    """ACEi or ARB prescribed AND latest K+ > 5.5 mEq/L.

    5.5 is the standard "moderate hyperkalemia" cutoff; below that, RAAS-
    inhibitor + minor K+ elevation is routine. Above it the pair becomes
    actionable — but the action (hold the drug? add kayexalate? recheck?) is
    a clinician decision, not ours."""
    drug = _med_name_contains(meds, *_ACE_ARB_FRAGMENTS)
    if drug is None:
        return None
    k = _latest_lab(labs, _POTASSIUM_LOINCS, _POTASSIUM_FRAGMENTS)
    if k is None or k.get("value") is None:
        return None
    try:
        if float(k["value"]) <= 5.5:
            return None
    except (TypeError, ValueError):
        return None
    unit = k.get("unit") or "mEq/L"
    return Flag(
        rule_id="ACE_ARB_HYPERKALEMIA",
        summary=(
            f"{drug.get('drug', 'ACEi/ARB')} prescribed "
            f"[MedicationRequest/{drug['id']}]; latest potassium "
            f"{k['value']} {unit} [Observation/{k['id']}]."
        ),
        evidence=[
            f"MedicationRequest/{drug['id']}",
            f"Observation/{k['id']}",
        ],
    )


_NSAID_FRAGMENTS = (
    "ibuprofen", "naproxen", "ketorolac", "diclofenac", "indomethacin",
    "meloxicam", "celecoxib", "aspirin",  # high-dose ASA pattern; conservative
    "piroxicam", "etodolac", "nabumetone",
)


def _rule_nsaid_ckd(
    meds: list[dict], labs: list[dict], problems: list[dict], allergies: list[dict],
) -> Flag | None:
    """NSAID prescribed AND latest eGFR < 60 mL/min/1.73m².

    KDIGO defines eGFR <60 as CKD stage 3+; NSAIDs at this level are widely
    flagged in renal pharmacology references. We surface the pair."""
    drug = _med_name_contains(meds, *_NSAID_FRAGMENTS)
    if drug is None:
        return None
    egfr = _latest_lab(labs, _EGFR_LOINCS, _EGFR_FRAGMENTS)
    if egfr is None or egfr.get("value") is None:
        return None
    try:
        if float(egfr["value"]) >= 60:
            return None
    except (TypeError, ValueError):
        return None
    unit = egfr.get("unit") or "mL/min/1.73m²"
    return Flag(
        rule_id="NSAID_CKD",
        summary=(
            f"{drug.get('drug', 'NSAID')} prescribed "
            f"[MedicationRequest/{drug['id']}]; latest eGFR {egfr['value']} "
            f"{unit} [Observation/{egfr['id']}]."
        ),
        evidence=[
            f"MedicationRequest/{drug['id']}",
            f"Observation/{egfr['id']}",
        ],
    )


def _rule_warfarin_high_inr(
    meds: list[dict], labs: list[dict], problems: list[dict], allergies: list[dict],
) -> Flag | None:
    """Warfarin prescribed AND latest INR > 4.0 (supratherapeutic).

    Most AFib protocols target 2.0–3.0; >4.0 is the standard "hold and check"
    threshold across guidelines. We surface the pair; reversal vs hold vs
    next-day recheck is a clinician decision."""
    warfarin = _med_name_contains(meds, "warfarin", "coumadin")
    if warfarin is None:
        return None
    inr = _latest_lab(labs, _INR_LOINCS, _INR_FRAGMENTS)
    if inr is None or inr.get("value") is None:
        return None
    try:
        if float(inr["value"]) <= 4.0:
            return None
    except (TypeError, ValueError):
        return None
    return Flag(
        rule_id="WARFARIN_HIGH_INR",
        summary=(
            f"Warfarin prescribed [MedicationRequest/{warfarin['id']}]; "
            f"latest INR {inr['value']} [Observation/{inr['id']}]."
        ),
        evidence=[
            f"MedicationRequest/{warfarin['id']}",
            f"Observation/{inr['id']}",
        ],
    )


_SULFA_DRUG_FRAGMENTS = (
    "sulfamethoxazole", "trimethoprim-sulfamethoxazole", "bactrim", "septra",
    "sulfasalazine", "sulfadiazine",
)
_SULFA_ALLERGY_FRAGMENTS = ("sulfa", "sulfonamide", "sulfas")


def _rule_sulfa_allergy_conflict(
    meds: list[dict], labs: list[dict], problems: list[dict], allergies: list[dict],
) -> Flag | None:
    """Sulfa-containing antibiotic prescribed AND a documented sulfa allergy.

    The most-common version of this pair (Bactrim + sulfa allergy) is one of
    the highest-frequency real-world prescribing errors — surfacing the pair
    is genuinely useful and unambiguous. We don't tell the doctor whether to
    discontinue or what to substitute; that's their call."""
    drug = _med_name_contains(meds, *_SULFA_DRUG_FRAGMENTS)
    if drug is None:
        return None
    allergy = _allergy_substance_contains(allergies, *_SULFA_ALLERGY_FRAGMENTS)
    if allergy is None:
        return None
    return Flag(
        rule_id="SULFA_ALLERGY_CONFLICT",
        summary=(
            f"{drug.get('drug', 'Sulfa-containing medication')} prescribed "
            f"[MedicationRequest/{drug['id']}]; documented "
            f"{allergy.get('substance', 'sulfa')} allergy "
            f"[AllergyIntolerance/{allergy['id']}]."
        ),
        evidence=[
            f"MedicationRequest/{drug['id']}",
            f"AllergyIntolerance/{allergy['id']}",
        ],
    )


# Add new rules here. Each must (a) cite chart evidence, (b) state facts not
# advice, (c) be defensible against a "this fired wrongly" complaint.
_RULES = (
    _rule_metformin_low_egfr,
    _rule_ace_arb_hyperkalemia,
    _rule_nsaid_ckd,
    _rule_warfarin_high_inr,
    _rule_sulfa_allergy_conflict,
)


def evaluate_chart(
    *,
    active_meds: list[dict],
    recent_labs: list[dict],
    active_problems: list[dict],
    allergies: list[dict],
) -> list[Flag]:
    """Run every rule against the chart slice and return all flags fired.

    Order is rule-definition order; callers should treat the list as a set
    of independent observations, not a ranked priority. Empty list means
    no chart-internal rule fired — NOT that the chart is "safe", since this
    rule set covers a small slice of the total clinical-rules surface."""
    flags: list[Flag] = []
    for rule in _RULES:
        try:
            flag = rule(active_meds, recent_labs, active_problems, allergies)
        except Exception:  # noqa: BLE001
            # A buggy rule must not break chart loading. Log via the trace
            # store at the call site; here we just skip the rule.
            flag = None
        if flag is not None:
            flags.append(flag)
    return flags
