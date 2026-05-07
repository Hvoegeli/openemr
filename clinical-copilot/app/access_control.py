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
from typing import TYPE_CHECKING

from app.auth import is_admin
from app.fhir.client import FhirClient

if TYPE_CHECKING:
    from app.assignments_db import AssignmentStore

log = logging.getLogger("agent.acl")

# Username -> OpenEMR Practitioner UUID. Admin users (per `is_admin`) skip this
# map entirely and see every patient. Adding a new physician means: (a) create
# the user in OpenEMR's admin UI with Provider=Yes, (b) read the resulting
# Practitioner UUID via `GET /apis/default/fhir/Practitioner?family=Lastname`,
# (c) add a row here.
USERNAME_TO_PRACTITIONER: dict[str, str] = {
    # Static overrides win over the dynamic resolver below. Use this for
    # deployments where username ≠ lname or where two providers share a
    # surname. Empty by default — every seeded OpenEMR user we ship with
    # has username == lname (case-insensitive), which the resolver handles.
}

# Username -> OpenEMR legacy `users.id` integer. The appointment-write
# endpoint (POST /api/patient/{pid}/appointment) takes the legacy
# integer id in its `pc_aid` field, NOT the FHIR Practitioner UUID,
# so we maintain a parallel map. Look up via:
#   SELECT id, username FROM users WHERE authorized = 1;
# inside docker compose exec -T mysql ... openemr.
USERNAME_TO_USER_ID: dict[str, int] = {
    # Legacy `users.id` is not exposed on FHIR Practitioner, so this stays
    # static for now. Only the appointment-write path consumes it (`pc_aid`),
    # and the overlay is disabled — the values below are the OpenEMR-shipped
    # seed defaults; deployments with a different `users` table need to
    # override these (env var or per-deploy config) when re-enabling
    # appointments. Tracked as a follow-up.
    "admin": 1,
    "Smith": 5,
}

# Demo allow-list: only these patients are surfaced anywhere the co-pilot
# decides which patients exist (calendar, admin assignment list, upload
# patient picker). Filtering here rather than deleting from OpenEMR's mysql
# keeps the cut reversible and deploy-safe. Empty/None disables the filter
# (admin still sees every Patient the FHIR layer returns).
#
# Decoded from `patient_data.uuid` on 2026-05-06. To add or remove names,
# edit this set — no mysql changes required. Comment per line names the
# patient so future-me can read the file without running a join.
ACTIVE_PATIENTS: frozenset[str] = frozenset({
    "a1b3cb94-4986-4c23-982d-e71c9f44cb17",  # Margaret Chen
    "a1a6044b-c6af-40a4-80aa-4c5ce61014da",  # Nora Cohen
    "a1b417c8-5256-43a3-b139-909373f584a6",  # Robert Kowalski
    "a1b417c6-03ef-4c5e-852f-0c37c17fed39",  # Sofia Reyes
    "a1a962d6-957a-42ae-9814-7ba2d14ca697",  # Liam Hale
    "a1a9628f-90dc-4927-9770-59a89c4b3d95",  # Anjali Patel
    "a1a9619a-ebe4-4c02-b928-3ea800ee15fd",  # Marcus Roberts
    "a1a6044b-c6e3-4aef-87be-cbdc1c2337d7",  # Jason Binder    (empty shell)
    "a1a6044b-c6fa-4655-b213-afb540f58e6e",  # Wallace Buckley (empty shell)
    "a1a6044b-c6eb-4499-8816-99bbe522233b",  # Robert Dickey   (empty shell)
})


def filter_active(patients: list[dict]) -> list[dict]:
    """Drop any Patient resource whose id isn't in `ACTIVE_PATIENTS`.

    Used at the FHIR-fetch boundary in the calendar, admin-assignment, and
    upload-picker paths. A safety net — the panel ACL filter still applies
    on top for non-admin users.
    """
    if not ACTIVE_PATIENTS:
        return patients
    return [p for p in patients if p.get("id") in ACTIVE_PATIENTS]


# In-memory cache: username -> (cached_at_epoch, panel_or_None). 5-minute TTL
# matches the dashboard cache convention. Restarting the process drops this;
# admin-driven assignment changes flush it via `invalidate_panel`.
_PANEL_CACHE: dict[str, tuple[float, frozenset[str] | None]] = {}
_PANEL_TTL_S = 300.0

# Resolved username -> Practitioner UUID, populated lazily by the dynamic
# resolver below. Lowercased keys. Lives for the process lifetime — the FHIR
# user table doesn't change often, and a UUID for an existing username
# is stable. Restart the process to repopulate.
_RESOLVED_PRACTITIONER: dict[str, str | None] = {}


def get_practitioner_id(username: str | None) -> str | None:
    """Return the Practitioner UUID for `username` from the static override map.

    Static-only; the async `resolve_practitioner_id` below adds a FHIR
    fallback. Kept as a sync helper for callers that just want the override
    (none today, but the symbol is part of the module's public surface).
    """
    if not username:
        return None
    return USERNAME_TO_PRACTITIONER.get(username)


async def resolve_practitioner_id(
    client: FhirClient, username: str | None,
) -> str | None:
    """Resolve a username to a Practitioner UUID, with a FHIR-search fallback.

    Resolution order:
      1. Static `USERNAME_TO_PRACTITIONER` override.
      2. FHIR search `Practitioner?family={username}`, taking the first
         non-inactive match (lowercased family-name compare). We don't
         pass `active=true` server-side because OpenEMR's Practitioner
         endpoint returns 0 rows for that filter even when the records
         are active — the active check is done client-side instead.

    The fallback exists so deployments don't have to touch source to add a
    new physician — every OpenEMR user we ship with has `username == lname`
    by convention, so a `family=` search reliably hits the right record.

    Limitations:
      - Two providers with the same surname collide (the static map is the
        escape hatch).
      - A deployment that intentionally sets `username != lname` won't
        resolve via FHIR; populate `USERNAME_TO_PRACTITIONER` instead.

    Cached by lowercased username for the process lifetime; `None` cached
    too (avoids re-hitting FHIR on every request for an unmapped user).
    """
    if not username:
        return None
    static = USERNAME_TO_PRACTITIONER.get(username)
    if static:
        return static
    key = username.strip().lower()
    if key in _RESOLVED_PRACTITIONER:
        return _RESOLVED_PRACTITIONER[key]
    # NOTE: do NOT pass `active=true` — OpenEMR's Practitioner endpoint
    # has a bug where the `active` token search returns 0 rows even when
    # the resource carries `active: true`. We filter client-side below.
    try:
        rows = await client.search(
            "Practitioner", {"family": username, "_count": 5},
        )
    except Exception:  # noqa: BLE001
        rows = []
    resolved: str | None = None
    for r in rows:
        rid = r.get("id")
        if not rid:
            continue
        if r.get("active") is False:
            continue
        # Defensive lname match — OpenEMR's family-name search is case-
        # insensitive, but in case a deploy returns near-matches we double-
        # check the lname equals (case-insensitive) the username.
        family = ((r.get("name") or [{}])[0].get("family") or "").strip().lower()
        if family and family != key:
            continue
        resolved = rid
        break
    _RESOLVED_PRACTITIONER[key] = resolved
    if resolved:
        log.info("acl: resolved username=%s -> practitioner=%s", username, resolved)
    else:
        log.warning("acl: no Practitioner found for username=%s", username)
    return resolved


def get_user_id(username: str | None) -> int | None:
    """Return the legacy `users.id` for `username`, or None if unmapped.

    Used by the appointment-write path which targets the standard REST
    API (POST /api/patient/{pid}/appointment) with `pc_aid` set to the
    integer user id rather than the FHIR Practitioner UUID.
    """
    if not username:
        return None
    return USERNAME_TO_USER_ID.get(username)


async def get_panel_for_user(
    client: FhirClient,
    username: str | None,
    assignments: "AssignmentStore | None" = None,
) -> frozenset[str] | None:
    """Resolve the set of patient IDs `username` is allowed to see.

    Return semantics:
      - `None`  -> no filter; user sees every patient (admin allow-list).
      - `frozenset()` -> empty panel; user has no assigned patients.
      - `frozenset({pid, ...})` -> explicit allow-list of FHIR Patient IDs.

    Source of truth: `assignments` (the co-pilot SQLite assignment store).
    OpenEMR's Demographics → Provider UI does NOT write to FHIR
    `Patient.generalPractitioner`, so a FHIR-search-based resolution would
    return empty for every non-admin user no matter how the OpenEMR UI is
    used. We bypass the FHIR field entirely until that gap closes upstream.

    `client` is unused now but kept for forward-compat with a future hybrid
    mode (read FHIR.generalPractitioner *and* the local store, union them).

    Failures (missing mapping for a non-admin, no assignments_store) fail
    closed by returning an empty panel — the user sees nothing rather than
    everything.
    """
    if not username:
        return frozenset()
    if is_admin(username):
        return None

    now = time.time()
    cached = _PANEL_CACHE.get(username)
    if cached and (now - cached[0]) < _PANEL_TTL_S:
        return cached[1]

    prac_id = await resolve_practitioner_id(client, username)
    if not prac_id or assignments is None:
        # Logged-in user with no Practitioner mapping (or no store wired) ->
        # sees nothing. Pre-cache so we don't recompute on every request.
        _PANEL_CACHE[username] = (now, frozenset())
        return frozenset()

    panel: frozenset[str] = frozenset(assignments.patients_for_practitioner(prac_id))
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
        _RESOLVED_PRACTITIONER.clear()
        return
    _PANEL_CACHE.pop(username, None)
    _RESOLVED_PRACTITIONER.pop(username.strip().lower(), None)
