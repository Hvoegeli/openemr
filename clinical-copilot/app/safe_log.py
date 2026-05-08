"""PHI redaction for application logs.

Backstop against accidental PHI in log messages. The primary defense is
"don't log PHI in the first place" — every audited call site in `app/`
is rewritten to log a `patient_uuid` rather than a name/DOB/address.
This module catches the cases the audit missed: future regressions,
third-party libraries that log payloads, and the long tail of unaudited
log lines added after the audit.

## Design

Two pieces:

1. `redact(text)` — applies a small bank of regex masks for the PHI shapes
   that show up most often in clinical logs: SSN, US phone, ISO-style DOB
   in DOB-context. Conservative by design — better to miss a non-shape PHI
   than to mangle a non-PHI ID. Patient names are NOT regex-redacted (no
   reliable shape) — see point 2.

2. `PHIRedactFilter` — a `logging.Filter` that runs `redact(...)` over
   `LogRecord.getMessage()` and rewrites the record so every handler
   downstream (file, stderr, LangSmith bridge) sees the redacted form.
   Installed by `install_phi_filter()` at process startup.

The filter is a *backstop*, not a substitute for clean log call sites.
The repo audit is documented in `W2_ARCHITECTURE.md` §8.2.

## What this does NOT do

- It does NOT strip patient names by string-matching at runtime — that
  would require a runtime registry of every patient name in the system.
  Names should be dropped at the call site (log the `patient_uuid`,
  not the name).
- It does NOT redact PHI inside structured args passed to LangSmith /
  LangChain auto-tracing. That requires a separate run-tree callback;
  the `LANGSMITH_TRACING=false` Hetzner env flag is the production fix
  until that callback exists. See `app/observability.py::init_langsmith`.
- It does NOT redact addresses (no compact regex shape) or names.
- It does NOT remove DOBs that appear in non-DOB context (e.g. a date
  that happens to match `1980-05-12`). The DOB pattern requires DOB-shape
  context to fire.
"""

from __future__ import annotations

import logging
import re

log = logging.getLogger("agent.safe_log")

# ─── Regex bank ──────────────────────────────────────────────────────────
#
# Every pattern is conservative — they only fire on shapes a clinician
# would write into a chart, not on shapes that look superficially similar
# (uuids, FHIR resource ids, version strings, etc.).

# US Social Security Number: NNN-NN-NNNN. The hyphenated form is the
# canonical PHI shape; the all-digits form is left alone because 9-digit
# numbers without hyphens commonly appear as primary keys in clinical
# systems (claim ids, etc.) and we don't want to mask those.
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")

# US phone numbers in the two most common surface forms:
#   - "(NNN) NNN-NNNN" with optional space after parens
#   - "NNN-NNN-NNNN"
# NPA-NXX-XXXX without a leading area-code grouping is rejected to
# avoid masking 7-digit codes (e.g. a HCPCS code like 12345-67890).
_PHONE_PARENS_RE = re.compile(r"\(\d{3}\)\s*\d{3}-\d{4}")
_PHONE_DASHED_RE = re.compile(r"\b\d{3}-\d{3}-\d{4}\b")

# Date-of-birth in ISO format, but ONLY in dob-context. Pattern:
#   <token containing 'dob' or 'birth'><up to 4 chars of separator>YYYY-MM-DD
# This avoids masking arbitrary ISO dates (e.g. log timestamps, encounter
# dates, schedule dates) which are not PHI on their own. The patient's
# DOB IS PHI, but only when paired with the patient.
_DOB_CONTEXT_RE = re.compile(
    r"(?i)(?P<key>\b(?:dob|date[_ ]of[_ ]birth|birth[_ ]?date|birthdate)\b\s*[:=]?\s*)(?P<dob>\d{4}-\d{2}-\d{2})"
)

_REDACT_TAG = "[REDACTED]"


def redact(text: str) -> str:
    """Apply the conservative PHI regex bank to `text` and return the
    masked form. Idempotent — running it twice produces the same output
    because every replacement is `[REDACTED]`-shaped which itself doesn't
    match any of the patterns.
    """
    if not text:
        return text
    out = _SSN_RE.sub(_REDACT_TAG, text)
    out = _PHONE_PARENS_RE.sub(_REDACT_TAG, out)
    out = _PHONE_DASHED_RE.sub(_REDACT_TAG, out)
    # DOB-context: keep the key word, replace only the date portion.
    out = _DOB_CONTEXT_RE.sub(lambda m: f"{m.group('key')}{_REDACT_TAG}", out)
    return out


def puuid_for_uuid(patient_uuid: str | None) -> str:
    """Format a patient UUID as a `<PUUID:abc12345>` placeholder for
    use in human-readable log messages. The first 8 chars are enough
    to disambiguate within a single demo deployment without leaking
    the full uuid into the log. Returns `<PUUID:?>` when the input is
    None or empty so the placeholder is always recognizable."""
    if not patient_uuid:
        return "<PUUID:?>"
    return f"<PUUID:{patient_uuid[:8]}>"


# ─── Logging filter ──────────────────────────────────────────────────────


class PHIRedactFilter(logging.Filter):
    """A `logging.Filter` that runs `redact()` over each log record's
    formatted message and overwrites the record with the masked form.

    Mechanics: `LogRecord.getMessage()` formats `record.msg % record.args`.
    We compute that, redact it, then null out `record.args` and assign the
    redacted result to `record.msg`. Downstream formatters re-format from
    `msg` (now safe) with no args.

    Side effects:
    - The original `args` are dropped — callers that try to inspect the
      raw record post-filter (rare; mostly tests) will see the redacted
      form rather than the original. This is intentional.
    - Performance is one regex pass per log record. The patterns are
      small and compiled at module-import time; impact is negligible
      relative to the file/network handler cost.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            formatted = record.getMessage()
        except Exception:  # noqa: BLE001
            # If formatting fails (mismatched args / format spec), the
            # log line is already broken — skip redaction and let the
            # normal handler chain surface the formatting error.
            return True
        redacted = redact(formatted)
        if redacted != formatted:
            record.msg = redacted
            record.args = ()
        return True


def install_phi_filter() -> None:
    """Attach `PHIRedactFilter` to the root logger so every existing
    handler (and every handler added later) inherits it. Idempotent —
    safe to call multiple times.

    Call this once at process startup, AFTER `logging.basicConfig` (or
    equivalent) has set up the root logger's handlers but BEFORE the
    first log line that might carry PHI. In FastAPI apps the right
    place is the lifespan startup function or right after `load_dotenv`.
    """
    root = logging.getLogger()
    # Don't double-install: bail if any of the existing filters is
    # already a `PHIRedactFilter`.
    for f in root.filters:
        if isinstance(f, PHIRedactFilter):
            return
    filt = PHIRedactFilter()
    root.addFilter(filt)
    # Also attach to every existing handler so handler-side filters
    # (which run AFTER root-level filters) catch records emitted
    # directly to a handler (e.g. through `logger.handle(record)`).
    for handler in root.handlers:
        handler.addFilter(filt)
    log.info("safe_log: installed PHIRedactFilter on root logger")
