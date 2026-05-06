"""Batched dashboard prewarm.

Pre-fix: `_prewarm_dashboard` called `get_patient_card(pid)` and
`get_supporting_documents(pid)` for every patient on the roster. Each
fired ~8 FHIR queries; for a 21-patient panel that meant ~168 in-flight
calls. OpenEMR's PHP session locking serializes server-side, so even
with `asyncio.gather` this took 30+ seconds on first boot.

Post-fix (this module): six roster-wide FHIR calls — Patient,
Encounter, AllergyIntolerance, Condition, MedicationRequest,
Observation (vital-signs), DocumentReference — bucketed by patient
client-side. ~7 calls instead of ~168 regardless of panel size.

The bucketed lists are fed into the `format_*_data` halves of the
patient-card and supporting-docs functions so the formatting logic
stays DRY across the on-demand path (still per-patient calls) and
the prewarm path (batched).

What this module does NOT do:
- It does not change `get_patient_card` / `get_supporting_documents`
  semantics. On-demand fetches (cache miss after TTL expiry, or a
  patient outside the prewarmed roster) still fan out per-patient.
- It does not paginate. `_count` is sized for a 30-patient panel
  with ~10 active resources each. A bigger deployment needs a
  panel-scoped panel param + paged search loop.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable

from app.cache import TTLCache
from app.fhir.adapter import format_patient_card_data
from app.fhir.client import FhirClient
from app.fhir.extras import _safe_search, format_supporting_documents_data

log = logging.getLogger("agent.fhir.prewarm")


def _bucket_by_patient_ref(
    rows: list[dict],
    *,
    field: str = "subject",
) -> dict[str, list[dict]]:
    """Group FHIR resources by their patient reference.

    `field` chooses between `subject.reference` (the FHIR R4 default
    for Encounter/Condition/MedicationRequest/Observation/DocumentReference)
    and `patient.reference` (AllergyIntolerance's required field).
    Both are bare `Patient/<uuid>` strings on the wire.
    """
    out: dict[str, list[dict]] = {}
    for row in rows:
        ref = (row.get(field) or {}).get("reference") or ""
        if not ref.startswith("Patient/"):
            continue
        pid = ref.removeprefix("Patient/")
        out.setdefault(pid, []).append(row)
    return out


async def warm_panel_cards_and_docs(
    client: FhirClient,
    cache: TTLCache,
    patients: list[dict],
    *,
    encounters: list[dict] | None = None,
) -> int:
    """Roster-wide batched prewarm.

    Args:
        patients: the Patient resources from the calendar fetch — we
            avoid re-fetching them.
        encounters: optional pre-fetched Encounter list (the calendar
            fetcher already pulls these for the today-row reason-line);
            pass them through to skip a redundant search.

    Returns the number of patients whose `card:{pid}` and `docs:{pid}`
    cache slices were populated.
    """
    if not patients:
        return 0

    # Pre-fetch every roster-wide resource the patient card and
    # supporting-docs view need. `_count` is generous enough that a
    # 30-patient panel with ~10 active rows per resource type doesn't
    # truncate any individual patient. A bigger deploy would page.
    fetches: list[Callable[[], Any]] = [
        lambda: _safe_search(client, "AllergyIntolerance", {"_count": 500}),
        lambda: _safe_search(client, "Condition", {"_count": 500}),
        lambda: _safe_search(client, "MedicationRequest", {"_count": 500}),
        lambda: _safe_search(client, "Observation", {"category": "vital-signs", "_count": 2000}),
        lambda: _safe_search(client, "DocumentReference", {"_count": 500}),
    ]
    if encounters is None:
        # Same `_count` as today's calendar N+1 fix — covers ~30 patients
        # × 10 recent encounters each without truncating any patient's latest.
        fetches.append(lambda: _safe_search(client, "Encounter", {"_sort": "-date", "_count": 300}))

    results = await asyncio.gather(*[f() for f in fetches])
    if encounters is None:
        allergies, conditions, meds, vitals, docs, encounters = results
    else:
        allergies, conditions, meds, vitals, docs = results

    # Bucket each list by patient. AllergyIntolerance uses `patient`,
    # the rest use `subject`.
    allergies_by = _bucket_by_patient_ref(allergies, field="patient")
    conditions_by = _bucket_by_patient_ref(conditions, field="subject")
    meds_by = _bucket_by_patient_ref(meds, field="subject")
    vitals_by = _bucket_by_patient_ref(vitals, field="subject")
    docs_by = _bucket_by_patient_ref(docs, field="subject")
    encounters_by = _bucket_by_patient_ref(encounters, field="subject")

    warmed = 0
    for p in patients:
        pid = p.get("id")
        if not pid:
            continue
        # Patient-card cache slice
        per_pat_encounters = encounters_by.get(pid, [])
        # `format_patient_card_data` expects an `encounters_all` list —
        # the formatter filters to the active one and trims to [:1].
        # Pass the full bucket (sorted -date desc by the global fetch).
        # The same bucket also feeds the supporting-docs view below.
        card_payload = format_patient_card_data(
            patient=p,
            patient_id=pid,
            encounters_all=per_pat_encounters,
            allergies=allergies_by.get(pid, []),
            problems_all=conditions_by.get(pid, []),
            meds=meds_by.get(pid, []),
            vitals=vitals_by.get(pid, []),
        )
        cache.set(f"card:{pid}", card_payload["data"])

        # Supporting-docs cache slice
        docs_payload = format_supporting_documents_data(
            encounters=per_pat_encounters,
            docs=docs_by.get(pid, []),
        )
        cache.set(f"docs:{pid}", docs_payload["data"])
        warmed += 1

    log.info(
        "prewarm batched: warmed %d patient cards + docs slices "
        "from %d roster-wide FHIR calls",
        warmed, len(fetches),
    )
    return warmed
