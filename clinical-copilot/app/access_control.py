"""Per-user patient-panel access control.

Closes the §4.3 / §8.3 gap from ARCHITECTURE.md: tools and endpoints used to
trust the session, not a per-tool ACL on patient IDs. Every user with a
registered Practitioner record now sees only the patients whose
`Patient.generalPractitioner` references their Practitioner. Admins (the
`ADMIN_USERNAMES` allow-list) bypass the filter and see every patient — they
operate the assignment UI itself, so a self-imposed filter would be a chicken-
and-egg problem.

OpenEMR is the single source of truth for the assignment. The filter resolves
each user's panel by querying `?general-practitioner=Practitioner/{id}` on the
Patient resource — no parallel assignment store in the co-pilot. The
username -> Practitioner mapping below is the only piece we own; it's small
and explicit until the admin-UI lets operators add rows.
"""

from __future__ import annotations

import logging
import time

from app.auth import is_admin
from app.fhir.client import FhirClient

log = logging.getLogger("agent.acl")

# Username -> OpenEMR Practitioner UUID. Admin users (per `is_admin`) skip this
# map entirely and see every patient. Adding a new physician means: (a) create
# the user in OpenEMR's admin UI with Provider=Yes, (b) read the resulting
# Practitioner UUID via `GET /apis/default/fhir/Practitioner?family=Lastname`,
# (c) add a row here.
USERNAME_TO_PRACTITIONER: dict[str, str] = {
    "Smith": "a1afccd2-13c1-47e5-a1f0-a317272dcb12",
}

# In-memory cache: username -> (cached_at_epoch, panel_or_None). 5-minute TTL
# matches the dashboard cache convention. Restarting the process drops this;
# admin-driven assignment changes flush it via `invalidate_panel`.
_PANEL_CACHE: dict[str, tuple[float, frozenset[str] | None]] = {}
_PANEL_TTL_S = 300.0


def get_practitioner_id(username: str | None) -> str | None:
    """Return the Practitioner UUID for `username`, or None if unmapped."""
    if not username:
        return None
    return USERNAME_TO_PRACTITIONER.get(username)


async def get_panel_for_user(
    client: FhirClient, username: str | None,
) -> frozenset[str] | None:
    """Resolve the set of patient IDs `username` is allowed to see.

    Return semantics:
      - `None`  -> no filter; user sees every patient (admin allow-list).
      - `frozenset()` -> empty panel; user has no assigned patients.
      - `frozenset({pid, ...})` -> explicit allow-list of FHIR Patient IDs.

    Failures (FHIR error, missing mapping for a non-admin) fail closed by
    returning an empty panel — the user sees nothing rather than everything.
    """
    if not username:
        return frozenset()
    if is_admin(username):
        return None

    now = time.time()
    cached = _PANEL_CACHE.get(username)
    if cached and (now - cached[0]) < _PANEL_TTL_S:
        return cached[1]

    prac_id = get_practitioner_id(username)
    if not prac_id:
        # Logged-in user with no Practitioner mapping -> sees nothing.
        # Pre-cache so we don't hit FHIR on every request for an unmapped user.
        _PANEL_CACHE[username] = (now, frozenset())
        return frozenset()

    try:
        patients = await client.search(
            "Patient",
            {"general-practitioner": f"Practitioner/{prac_id}", "_count": 200},
        )
        panel: frozenset[str] = frozenset(p["id"] for p in patients if p.get("id"))
    except Exception as e:  # noqa: BLE001 — fail closed, don't leak chart access on a network blip
        log.warning("acl: panel fetch failed for user=%s: %s", username, e)
        panel = frozenset()

    _PANEL_CACHE[username] = (now, panel)
    log.info("acl: panel for user=%s prac=%s size=%d", username, prac_id, len(panel))
    return panel


def is_in_panel(panel: frozenset[str] | None, patient_id: str) -> bool:
    """True if `patient_id` is reachable. `panel=None` (admin) is always True."""
    if panel is None:
        return True
    return patient_id in panel


class PatientAccessDenied(Exception):
    """Raised by tool dispatch when a patient is outside the caller's panel.

    Carries `patient_id` so the caller can record it in the audit trace and
    return a "not found"-shaped tool message to the LLM. Distinct from
    `LookupError` so the existing tool-error path in `execute_tools` can
    treat ACL denials as a non-leaky "patient not found" rather than a
    generic exception (which the LLM might quote as "I got an error").
    """

    def __init__(self, patient_id: str) -> None:
        self.patient_id = patient_id
        super().__init__(f"patient_access_denied:{patient_id}")


def invalidate_panel(username: str | None = None) -> None:
    """Drop a cached panel (or all of them). Call after admin reassigns."""
    if username is None:
        _PANEL_CACHE.clear()
        return
    _PANEL_CACHE.pop(username, None)
