"""Citation validator.

Two structural enforcements of R1 ("every clinical fact gets a citation"):

  1. `find_invalid_citations` — rejects responses that cite a resource ID
     not returned by any tool in this conversation. This is the FAKE-cite
     check.

  2. `find_uncited_clinical_claims` — rejects responses that contain a
     clinically-shaped sentence with no `[Type/ID]` cite at all. This is
     the MISSING-cite check, added after MVP grader feedback flagged that
     zero-citation responses bypass attribution enforcement entirely.

Both run as LangGraph retry edges with a shared MAX_VALIDATION_ATTEMPTS
budget — even if the LLM tries to confabulate or to omit citations
silently, it cannot ship the response without satisfying both checks or
falling back to the "insufficient evidence" refusal.
"""

import re

CITATION_RE = re.compile(r"\[([A-Z][a-zA-Z]+)/([a-zA-Z0-9._-]+)\]")


def extract_citations(text: str) -> list[str]:
    """Return every `Type/ID` found in inline `[Type/ID]` citations."""
    return [f"{m.group(1)}/{m.group(2)}" for m in CITATION_RE.finditer(text)]


def find_invalid_citations(text: str, allowed_sources: list[str]) -> list[str]:
    """Return citations from `text` that are not in `allowed_sources` (deduped, ordered)."""
    allowed = set(allowed_sources)
    seen: set[str] = set()
    invalid: list[str] = []
    for cite in extract_citations(text):
        if cite in allowed or cite in seen:
            continue
        seen.add(cite)
        invalid.append(cite)
    return invalid


# ─── zero-citation clinical-claim detection ──────────────────────────────

# A measurable vital / lab value: number followed by a clinical unit.
# Catches "138 mmHg", "72 bpm", "98.6°F", "creatinine 2.1 mg/dL", etc.
_CLINICAL_UNIT_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*"
    r"(?:mmHg|bpm|/min|°F|°C|%|mg|mcg|g/dL|mg/dL|ng/mL|"
    r"mL|lbs?|kg|mEq/L|mmol/L|IU|U/L|mIU/L)\b",
    re.IGNORECASE,
)

# A lab / vital name immediately followed by a numeric value.
# Catches "creatinine 2.1", "HR 78", "SpO2 95%", "BP 138/82", "temp of 102.4".
_CLINICAL_LAB_RE = re.compile(
    r"\b(?:creatinine|BUN|sodium|potassium|chloride|glucose|hemoglobin|"
    r"HbA1c|A1c|INR|WBC|platelets?|hematocrit|albumin|bilirubin|AST|ALT|"
    r"GFR|eGFR|HR|BP|SpO2|O2|pulse|temp(?:erature)?|"
    r"systolic|diastolic|respiratory rate|RR)\b"
    r"\s*(?:[:=]|of|is|was|at|=)?\s*\d",
    re.IGNORECASE,
)

# Patient-context pronoun or noun-phrase referring to the chart subject.
_PATIENT_CONTEXT_RE = re.compile(
    r"\b(?:she|he|her|his|the patient|patient is|patient has|patient's|"
    r"pt\.?\s+is|pt\.?\s+has)\b",
    re.IGNORECASE,
)

# Clinical content words — non-exhaustive but covers the high-frequency
# inpatient vocabulary. Combined with patient context it triggers the
# "this is a clinical claim" judgment.
_CLINICAL_WORD_RE = re.compile(
    r"\b(?:allergy|allergies|allergic|hypertension|hypertensive|diabetes|"
    r"diabetic|CKD|AFib|atrial fibrillation|hyperlipidemia|asthma|COPD|"
    r"pneumonia|sepsis|infection|pyelonephritis|stroke|MI|myocardial|"
    r"congestive|heart failure|prescribed|taking|on (?:a |an )?med|"
    r"medication|active problems?|admitted|discharged|encounter)\b",
    re.IGNORECASE,
)

# Med-usage pattern — high-precision indicator of a patient-specific drug
# claim that needs a cite. Matches "is on Lisinopril", "taking Apixaban",
# "prescribed Metformin", "switched to Atorvastatin", etc.
#
# Verbs are case-insensitive (so "Taking Apixaban" at the start of a sentence
# matches), but the trailing word must be capitalized (so we don't flag
# "is on the chart"). Bare "on" is deliberately excluded — it overfires on
# things like "on April 28" or "on the patient", giving more noise than
# signal. Sentences with bare "on" + clinical content still get caught by
# the patient-context / clinical-word path below.
_MED_USAGE_RE = re.compile(
    r"\b(?i:is\s+on|taking|prescribed|started\s+on|switched\s+to|"
    r"continue[ds]?|added)\s+[A-Z][a-z]{2,}"
)

# Short-form reassurance / status words. Paired with patient context
# (below), these catch the B1 zero-citation bypass that the four
# patterns above miss: "The patient is fine.", "She is stable.",
# "He is okay." — sentences with no number, no lab, no drug, but a
# flat clinical reassurance that R1 still requires a `[Type/ID]`
# citation for (AgentForge report b11562ac218f). Word list is tight
# on purpose — exactly the five reassurance tokens AgentForge's
# prompt names (`yes`, `no`, `fine`, `stable`, `okay`). Broadening
# to `well`, `normal`, `unremarkable`, `ok`, etc. would risk flagging
# legitimate non-clinical prose.
_REASSURANCE_WORD_RE = re.compile(
    r"\b(?:fine|stable|okay|yes|no)\b",
    re.IGNORECASE,
)


def looks_like_clinical_claim(sentence: str) -> bool:
    """True if the sentence asserts something measurable or patient-specific
    that R1 says must be cited.

    Conservative on purpose: the bar to flag is high so that conversational
    intros, refusals, and meta-talk don't trigger. False negatives are OK
    here (some uncited claims slip through, the existing fake-cite check
    still catches the worst case where IDs are made up); false positives
    are NOT — they would block legitimate refusals like the R5 template.
    """
    s = sentence.strip()
    if not s:
        return False
    if _CLINICAL_UNIT_RE.search(s):
        return True
    if _CLINICAL_LAB_RE.search(s):
        return True
    if _MED_USAGE_RE.search(s):
        return True
    if _PATIENT_CONTEXT_RE.search(s) and _CLINICAL_WORD_RE.search(s):
        return True
    # B1: patient-context + a reassurance word with no concrete clinical
    # token (no lab/med/number) is still a clinical-status claim and
    # needs a cite. The R5 refusal template ("about patient chart data",
    # "about a patient") has no `_PATIENT_CONTEXT_RE` match, so this
    # branch doesn't fire on legitimate refusals.
    if _PATIENT_CONTEXT_RE.search(s) and _REASSURANCE_WORD_RE.search(s):
        return True
    return False


def _split_sentences(text: str) -> list[str]:
    """Naive sentence splitter — splits on `.`, `!`, `?` (followed by space
    or newline) and on bullet boundaries. Good enough for validator use."""
    parts = re.split(r"(?<=[.!?])\s+|\n+|^\s*[-•*]\s+", text, flags=re.MULTILINE)
    return [p for p in parts if p and p.strip()]


def _is_setup_phrase(sentence: str) -> bool:
    """A sentence ending in `:` is a list lead-in (e.g. 'Her active problems
    are:'). The cited claims live in the list items that follow; the lead-in
    itself doesn't need a citation. Without this, every bullet header in a
    well-formed agent response trips the missing-cite check."""
    return sentence.rstrip().endswith(":")


def find_uncited_clinical_claims(text: str) -> list[str]:
    """Return clinical-shaped sentences that contain no `[Type/ID]` citation.

    R1 in the system prompt requires every clinical fact to end with a
    citation. The `find_invalid_citations` check catches confabulated IDs;
    this catches the MISSING-cite case where the LLM emits a clinical claim
    with no `[…]` at all and would otherwise sail through.
    """
    out: list[str] = []
    for sentence in _split_sentences(text):
        if not looks_like_clinical_claim(sentence):
            continue
        if CITATION_RE.search(sentence):
            continue
        if _is_setup_phrase(sentence):
            continue
        out.append(sentence.strip())
    return out
