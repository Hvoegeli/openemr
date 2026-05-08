"""Tests for `app.safe_log` — PHI redactor + logging filter.

Coverage matrix:
- SSN-shaped strings → masked
- Phone (parens + dashed forms) → masked
- DOB in DOB-context → masked; non-context dates left alone
- `puuid_for_uuid` shapes correctly
- `PHIRedactFilter` rewrites records
- Clean log messages pass through unchanged
- Username (clinician login) is NOT masked
"""

from __future__ import annotations

import logging

from app.safe_log import (
    PHIRedactFilter,
    install_phi_filter,
    puuid_for_uuid,
    redact,
)


# ─── redact() ────────────────────────────────────────────────────────────


def test_ssn_dashed_is_masked() -> None:
    assert redact("patient SSN 123-45-6789 on file") == "patient SSN [REDACTED] on file"


def test_phone_parens_is_masked() -> None:
    assert redact("call (555) 123-4567 today") == "call [REDACTED] today"


def test_phone_dashed_is_masked() -> None:
    assert redact("call 555-123-4567 today") == "call [REDACTED] today"


def test_dob_in_context_is_masked() -> None:
    text = "DOB: 1980-05-12 confirmed"
    assert redact(text) == "DOB: [REDACTED] confirmed"


def test_dob_in_birthdate_context_is_masked() -> None:
    text = "birthdate=1980-05-12"
    out = redact(text)
    assert "1980-05-12" not in out
    assert "[REDACTED]" in out


def test_iso_date_outside_dob_context_is_left_alone() -> None:
    """Encounter dates / log timestamps / schedule dates are not PHI
    on their own — only the patient's DOB is. The pattern requires
    a 'dob' / 'birthdate' context word to fire."""
    text = "encounter on 2024-03-15 completed"
    assert redact(text) == "encounter on 2024-03-15 completed"


def test_clean_message_passes_through_unchanged() -> None:
    text = "create_patient: user=admin uuid=a1b2c3d4 created=true"
    assert redact(text) == text


def test_username_is_not_masked() -> None:
    """Clinician login names (admin, smith, etc.) are NOT PHI in the
    HIPAA sense — they're authorized users of the system, not patients.
    The redactor must not mask them."""
    text = "user=smith login successful"
    assert redact(text) == text


def test_uuid_is_not_mistaken_for_phi() -> None:
    """FHIR UUIDs and DocumentReference IDs share some characters with
    SSN/phone shapes but don't match the actual regexes. Sanity-check."""
    text = "patient_uuid=a1b9ec14-0afa-4ad9-a303-4e61305fcb02 ref=DocumentReference/abc-123-def"
    assert redact(text) == text


def test_redact_idempotent() -> None:
    """Running redact twice should be a no-op — `[REDACTED]` itself
    doesn't match any of the patterns."""
    once = redact("phone (555) 123-4567 ssn 123-45-6789")
    twice = redact(once)
    assert once == twice


# ─── puuid_for_uuid() ────────────────────────────────────────────────────


def test_puuid_for_uuid_truncates_to_eight() -> None:
    assert puuid_for_uuid("a1b9ec14-0afa-4ad9-a303-4e61305fcb02") == "<PUUID:a1b9ec14>"


def test_puuid_for_uuid_handles_none() -> None:
    assert puuid_for_uuid(None) == "<PUUID:?>"


def test_puuid_for_uuid_handles_empty() -> None:
    assert puuid_for_uuid("") == "<PUUID:?>"


# ─── PHIRedactFilter ─────────────────────────────────────────────────────


def _make_record(msg: str, *args: object) -> logging.LogRecord:
    """Construct a minimal LogRecord for filter testing."""
    return logging.LogRecord(
        name="agent.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg=msg,
        args=args,
        exc_info=None,
    )


def test_filter_redacts_formatted_message() -> None:
    filt = PHIRedactFilter()
    rec = _make_record("ssn=%s on file", "123-45-6789")
    assert filt.filter(rec) is True
    assert rec.getMessage() == "ssn=[REDACTED] on file"


def test_filter_passes_clean_records_unchanged() -> None:
    filt = PHIRedactFilter()
    rec = _make_record("user=%s login ok", "admin")
    assert filt.filter(rec) is True
    assert rec.getMessage() == "user=admin login ok"


def test_filter_does_not_break_on_format_error() -> None:
    """If a log call has mismatched args, getMessage() raises. The
    filter should catch and let the log line through (the failure
    becomes visible at handler time)."""
    filt = PHIRedactFilter()
    # Mismatched: %s expects one arg, we provide zero.
    rec = _make_record("missing arg %s")
    rec.args = ()
    # Filter returns True (allowing the record through) without raising.
    assert filt.filter(rec) is True


def test_install_phi_filter_is_idempotent() -> None:
    """Calling install twice should not double-attach the filter."""
    root = logging.getLogger()
    # Snapshot pre-existing filters.
    pre_count = sum(1 for f in root.filters if isinstance(f, PHIRedactFilter))
    install_phi_filter()
    install_phi_filter()
    post_count = sum(1 for f in root.filters if isinstance(f, PHIRedactFilter))
    # Net effect: +0 (already installed) or +1 (first install).
    assert post_count - pre_count <= 1
    assert post_count >= 1
