"""Centralized time-zone conversions for the clinical co-pilot.

Three places in the codebase had been hand-rolling the same conversion
("UTC ISO -> clinical-tz ISO" / "what is `today` for the doctor's wall
clock"), and three TZ bugs slipped through over the lifetime of the
project. This module is the single source of truth so a future fix
to one site automatically flows through to the others.

Public surface:

  clinical_now() -> datetime     The current instant, tagged with
                                 the clinical-tz `tzinfo`. Use whenever
                                 server-side code needs "what time is
                                 it for the doctor right now."

  clinical_today() -> date       The clinical-tz wall-clock date right
                                 now. Use whenever server-side code
                                 wants `today_iso` — never `date.today()`,
                                 which returns server-local (Hetzner is
                                 UTC and after ~5pm clinical the dates
                                 disagree).

  to_clinical_iso(value)         Re-stamp a UTC ISO string OR a tz-aware
                                 datetime into a clinical-tz ISO with
                                 the offset on the wire. Tolerates
                                 None / empty (returns None) and
                                 unparsable inputs (returns the
                                 original) so it can be applied
                                 indiscriminately to FHIR fields
                                 without try/excepts at every call site.

The clinical TZ comes from `settings.clinical_tz` so a deployment can
flip it from `America/Chicago` to whatever its panel is in.

Why a module: the previous "patch only what's broken" pattern means
new code keeps reinventing the boundary conversion. Centralizing the
helpers means a future feature touching time can `from app.timeutil
import clinical_now` instead of typing out `datetime.now(ZoneInfo(...))`
yet again — and a future TZ-bug audit only needs to grep for time
ops in this module's callers.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo


def _clinical_zoneinfo() -> ZoneInfo:
    """Resolve the configured clinical TZ. Lazy-import of `settings`
    keeps this module load-safe under the same circular-import dance
    `app.fhir.adapter` already does."""
    from app.config import settings  # noqa: PLC0415
    return ZoneInfo(settings.clinical_tz)


def clinical_now() -> datetime:
    """Current instant, tagged with the clinical-tz tzinfo."""
    return datetime.now(_clinical_zoneinfo())


def clinical_today() -> date:
    """Today's wall-clock date for the doctor — NOT `date.today()`.

    `date.today()` reads the server's local clock. Hetzner runs UTC,
    the user runs clinical-local. Past local-evening they disagree by
    one calendar day, which is why "today's appointments" appeared
    empty after 5pm without this function.
    """
    return clinical_now().date()


def to_clinical_iso(value: str | datetime | None) -> str | None:
    """Re-stamp a UTC instant into a clinical-tz ISO string with the
    offset on the wire.

    Accepts:
      - `None` or empty string -> returns `None` (caller's missing-data
        story flows through unchanged).
      - A timezone-aware `datetime` -> converts to clinical TZ.
      - A naive `datetime` -> assumed UTC (FHIR convention).
      - An ISO string with `Z` suffix or explicit offset -> parsed.
      - An ISO string with no offset -> assumed UTC.
      - Anything that fails to parse -> returned unchanged so a
        partial-data row keeps its source value visible in logs / UI
        rather than disappearing. The caller can decide what to do
        with a non-ISO string downstream.

    Returns ISO with seconds precision (e.g. `2026-04-30T15:35:48-06:00`)
    so JS / Python parsers downstream both reconstruct the same instant.
    """
    if value is None:
        return None
    if isinstance(value, str):
        if not value:
            return None
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return value
    elif isinstance(value, datetime):
        dt = value
    else:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(_clinical_zoneinfo()).isoformat(timespec="seconds")
