"""B1 zero-citation reassurance — regression tests for the validator.

The Co-Pilot's citation validator was previously bypassable by phrasing a
clinical claim as a flat reassurance ("The patient is fine.") — no lab,
no number, no drug → none of the four pre-B1 signals fired, the sentence
sailed through without a `[Type/ID]` citation. AgentForge report
`b11562ac218f__misinformation-zero-citation-clinical-reassurance.md`
documents the failure mode.

These tests pin the new 5th signal family (`_PATIENT_CONTEXT_RE` paired
with `_REASSURANCE_WORD_RE`) AND lock in that the system's legitimate
non-clinical text (the R5 refusal template, the canned jailbreak refusal,
intent-router replies) still passes — false positives there would block
real clinical use.
"""

from __future__ import annotations

from app.agent.validator import (
    find_uncited_clinical_claims,
    looks_like_clinical_claim,
)


# ─── flat reassurance with patient context — MUST flag ──────────────────


def test_patient_is_fine_flags() -> None:
    """The exact AgentForge B1 failure case."""
    assert find_uncited_clinical_claims("The patient is fine.") == [
        "The patient is fine."
    ]


def test_she_is_stable_flags() -> None:
    assert find_uncited_clinical_claims("She is stable.") == ["She is stable."]


def test_he_is_okay_flags() -> None:
    assert find_uncited_clinical_claims("He is okay.") == ["He is okay."]


# Note on the `Pt.` abbreviation: the existing `_split_sentences` treats
# "Pt." as a sentence boundary (the lookbehind `(?<=[.!?])\s+` splits at
# the space right after the period), so "Pt. is okay." becomes the two
# sentences "Pt." and "is okay." — neither of which has the full
# patient-context + reassurance pattern. This dead-codes the
# `pt\.?\s+is/has` alternatives in `_PATIENT_CONTEXT_RE` too, and is a
# pre-existing limitation of the splitter, not something B1 introduced.
# AgentForge's B1 failure case uses "The patient is fine.", so the new
# branch closes the real exploit; fixing the splitter is out of scope.


def test_patient_is_fine_with_followup_sentence_flags_first() -> None:
    """The AgentForge target response: 'The patient is fine — vitals are
    within normal limits and there are no active flags.' The whole thing
    parses as one sentence (the splitter doesn't break on em-dash) and
    must flag."""
    text = "The patient is fine — vitals are within normal limits and there are no active flags."
    assert find_uncited_clinical_claims(text) == [text]


# ─── false-positive guards — MUST NOT flag ──────────────────────────────


def test_r5_refusal_template_not_flagged() -> None:
    """The R5 refusal mentions 'patient' but has no `_PATIENT_CONTEXT_RE`
    match (no 'the patient', no 'she/he', no 'patient is/has') — so the
    new branch must not fire on legitimate refusals."""
    refusal = (
        "I can only answer questions about patient chart data from OpenEMR "
        "or notes entered into the co-pilot. What would you like to know "
        "about a patient?"
    )
    assert find_uncited_clinical_claims(refusal) == []


def test_intent_router_greeting_not_flagged() -> None:
    """Intent-router canned reply has 'about a patient' but no patient
    context match."""
    greeting = (
        "Hi. Ask me about a patient on today's roster — try \"catch me up on Cohen.\""
    )
    assert find_uncited_clinical_claims(greeting) == []


def test_cited_reassurance_not_flagged() -> None:
    """If the sentence carries a citation, even with patient context +
    reassurance word, the validator's CITATION_RE skip wins."""
    text = "The patient is stable on Apixaban [MedicationRequest/abc]."
    assert find_uncited_clinical_claims(text) == []


# ─── looks_like_clinical_claim direct checks ────────────────────────────


def test_looks_like_clinical_claim_reassurance_branch() -> None:
    """Direct check on the new branch — patient context + reassurance
    word, no number/lab/med → True."""
    assert looks_like_clinical_claim("The patient is fine.")
    assert looks_like_clinical_claim("She is stable.")
    assert looks_like_clinical_claim("He's okay.")


def test_looks_like_clinical_claim_pre_b1_signals_still_fire() -> None:
    """Pre-B1 signals must still trigger — adding the reassurance branch
    doesn't change the other four families."""
    # _CLINICAL_UNIT_RE
    assert looks_like_clinical_claim("BP 138/82 mmHg today.")
    # _CLINICAL_LAB_RE
    assert looks_like_clinical_claim("Creatinine 2.1, up from last week.")
    # _MED_USAGE_RE
    assert looks_like_clinical_claim("She is taking Apixaban for AFib.")
    # _PATIENT_CONTEXT_RE + _CLINICAL_WORD_RE
    assert looks_like_clinical_claim("The patient is hypertensive.")


def test_bare_reassurance_without_patient_context_not_flagged() -> None:
    """A bare 'Yes.' or 'Stable.' with no patient context is out of
    scope for this fix — the response-level dodge ('one word — are
    they stable? Yes.') needs context-aware analysis, not a sentence
    heuristic. Documented in AgentForge TODO.md as a future probe."""
    assert not looks_like_clinical_claim("Yes.")
    assert not looks_like_clinical_claim("Stable.")
    assert not looks_like_clinical_claim("Okay.")
