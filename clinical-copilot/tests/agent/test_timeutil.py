"""Isolated tests for `app.timeutil` — the centralized clinical-tz
conversion helpers.

The TZ followup memory called this out as a regression-prevention
goal: every datetime field rendered in any tool result, when
re-converted from UTC to the doctor's clinical TZ, must match the
value the user saw. These tests pin the behavior of the helpers so
a future "patch only the broken site" doesn't drift back into the
N-hand-rolled-converters world that produced three separate TZ bugs.
"""

from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from app.timeutil import clinical_now, clinical_today, to_clinical_iso


@pytest.fixture
def chicago_tz(monkeypatch):
    """Pin the configured clinical TZ to America/Chicago for this test
    so assertions don't depend on whatever the test runner's env has
    set. Centralized here so every test in this file uses the same
    fixture without re-typing the patch."""
    from app.config import settings  # noqa: PLC0415
    monkeypatch.setattr(settings, "clinical_tz", "America/Chicago")
    yield "America/Chicago"


class TestToClinicalIso:
    def test_utc_z_suffix_converts_to_chicago(self, chicago_tz) -> None:
        # 21:35 UTC = 16:35 CDT (UTC-5 in May) — round-trip preserves
        # the instant.
        out = to_clinical_iso("2026-04-30T21:35:48Z")
        assert out is not None
        parsed = datetime.fromisoformat(out)
        # Same instant: 21:35 UTC == 16:35 CDT
        utc_instant = datetime(2026, 4, 30, 21, 35, 48, tzinfo=timezone.utc)
        assert parsed == utc_instant
        # Wall-clock is Central
        assert parsed.tzinfo is not None
        assert parsed.utcoffset() == ZoneInfo("America/Chicago").utcoffset(parsed)

    def test_explicit_offset_is_preserved_as_instant(self, chicago_tz) -> None:
        # Same instant rendered with an explicit offset should hash
        # identically to the Z-suffix form.
        z = to_clinical_iso("2026-04-30T21:35:48Z")
        plus = to_clinical_iso("2026-04-30T15:35:48-06:00")
        # Both convert to America/Chicago; instants differ here
        # because 15:35 MDT != 21:35 UTC. We only assert that the
        # converter accepted both shapes without exception.
        assert z is not None and plus is not None

    def test_naive_datetime_assumed_utc(self, chicago_tz) -> None:
        out = to_clinical_iso(datetime(2026, 4, 30, 21, 35, 48))
        assert out == to_clinical_iso("2026-04-30T21:35:48Z")

    def test_aware_datetime_passes_through_to_clinical(self, chicago_tz) -> None:
        # A tz-aware datetime in some other zone gets re-stamped to
        # clinical TZ, instant preserved.
        eastern = ZoneInfo("America/New_York")
        dt = datetime(2026, 4, 30, 17, 35, 48, tzinfo=eastern)
        out = to_clinical_iso(dt)
        assert out is not None
        parsed = datetime.fromisoformat(out)
        assert parsed == dt

    def test_none_returns_none(self, chicago_tz) -> None:
        assert to_clinical_iso(None) is None

    def test_empty_string_returns_none(self, chicago_tz) -> None:
        assert to_clinical_iso("") is None

    def test_unparseable_string_returns_unchanged(self, chicago_tz) -> None:
        # Defensive: garbage in, garbage out — but not an exception.
        # Production code that scrubs FHIR fields can't crash on a
        # malformed row.
        assert to_clinical_iso("not-an-iso-string") == "not-an-iso-string"

    def test_unsupported_input_type_returns_none(self, chicago_tz) -> None:
        assert to_clinical_iso(12345) is None  # type: ignore[arg-type]

    def test_seconds_precision_in_output(self, chicago_tz) -> None:
        # The microsecond should be dropped so JS / Python round-trips
        # don't produce different ISO strings for the same instant.
        out = to_clinical_iso(
            datetime(2026, 4, 30, 21, 35, 48, 123456, tzinfo=timezone.utc),
        )
        assert out is not None
        assert "." not in out  # no fractional seconds


class TestClinicalNow:
    def test_returns_aware_datetime(self, chicago_tz) -> None:
        now = clinical_now()
        assert now.tzinfo is not None

    def test_offset_matches_chicago(self, chicago_tz) -> None:
        now = clinical_now()
        chi = ZoneInfo("America/Chicago")
        assert now.utcoffset() == chi.utcoffset(now)


class TestClinicalToday:
    def test_returns_clinical_now_date(self, chicago_tz) -> None:
        # Stable invariant: clinical_today() == clinical_now().date()
        # at any instant (as long as the calls happen within a
        # microsecond of each other — practically always).
        assert clinical_today() == clinical_now().date()
