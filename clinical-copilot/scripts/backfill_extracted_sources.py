"""Backfill the `extracted_resource_sources` SQLite store for chart
resources whose `[copilot-source: ...]` tag was lost in transit.

Two scenarios:
  - **Allergies (any patient).** OpenEMR's FhirAllergyIntoleranceService
    drops the `comments` column on serialization, so the writer's tag
    never round-trips. Even resources written by today's writer are
    invisible to the manifest reader. Backfill maps each allergy to a
    plausible source-doc by matching the substance text (from
    `text.div`) against the patient's intake-form documents.
  - **Pre-fix historical resources (any type).** Resources written
    before this store existed have their tag in FHIR `note` already,
    so the manifest reader works without backfill. We only backfill
    rows where the tag is missing.

The mapping is heuristic — for the Sunday demo, "Chen's three
allergies came from her intake.pdf" is true and recoverable. A
production version would require explicit provenance at write time.

Run:
  PYTHONPATH=/root/openemr/clinical-copilot uv run \
    python scripts/backfill_extracted_sources.py [patient-name]...
"""

import asyncio
import re
import sys
from pathlib import Path

from app.extracted_sources_db import ExtractedSourcesStore
from app.fhir.adapter import _coded_display, _collect_note_text, _narrative_text
from app.fhir.client import FhirClient

DEFAULT_PATIENTS = ["Whitaker", "Chen", "Reyes", "Kowalski", "Cohen"]
DB_PATH = Path("data/traces.db")

# Substance-name → known page-on-Chen-intake. Used as a hint so the
# page-band overlay renders on the page the substance is actually
# printed on. Chen's intake puts allergies on page 2 (after the meds
# table). Other patients won't match this dict; they get bbox=None
# (page-band defaults to page 1).
_KNOWN_PAGE_HINTS: dict[str, dict] = {
    "Penicillin":     {"page": 2},
    "Sulfa drugs":    {"page": 2},
    "shellfish":      {"page": 2},
    "iodine":         {"page": 2},
}

_TAG_RE = re.compile(r"\[copilot-source:\s*[^\]]+\]")


def _match_intake_doc(docs: list[dict]) -> str | None:
    """Pick the most likely intake-form DocumentReference from a
    patient's docs. Heuristic: any doc whose attachment title or
    description matches `intake`. Demo data ships docs with literal
    'intake' in the filename, so the heuristic is reliable for the
    set we care about. Returns the bare doc id (no `DocumentReference/`
    prefix) or None."""
    for d in docs:
        bits: list[str] = []
        descr = d.get("description") or ""
        bits.append(descr)
        for c in d.get("content") or []:
            att = c.get("attachment") or {}
            bits.append(att.get("title") or "")
        haystack = " ".join(bits).lower()
        if "intake" in haystack:
            return d.get("id")
    return None


def _allergy_substance_text(a: dict) -> str:
    """Extract the substance name from an AllergyIntolerance row.

    Tries `code.text` first, then `code.coding[0].display` (skipping
    OpenEMR's `data-absent-reason: unknown` placeholder), then the
    narrative `text.div`. The third path is what catches Chen's three
    allergies — OpenEMR puts the substance only in the narrative when
    no SNOMED code is supplied.
    """
    return (
        _coded_display(a.get("code") or {})
        or _narrative_text(a)
        or "Unknown allergy"
    )


def _bbox_hint_for_substance(substance: str) -> dict | None:
    """Map a substance string to a known page hint (or None if no hint).

    Substring match so "shellfish?? maybe iodine" matches both
    "shellfish" and "iodine" hints — same page either way.
    """
    s = substance.lower()
    for key, hint in _KNOWN_PAGE_HINTS.items():
        if key.lower() in s:
            return hint
    return None


async def backfill_for_patient(
    client: FhirClient, store: ExtractedSourcesStore, name: str,
) -> dict:
    """Walk one patient and backfill every allergy that lacks a tag."""
    rows = await client.search("Patient", {"family": name, "_count": 5})
    if not rows:
        all_p = await client.search("Patient", {"_count": 200})
        rows = [
            p for p in all_p
            if any(name.lower() in (n.get("family") or "").lower()
                   for n in (p.get("name") or []))
        ]
    if not rows:
        return {"name": name, "skipped": "patient not found"}
    pid = rows[0]["id"]

    docs = await client.search("DocumentReference", {"patient": pid, "_count": 100})
    intake_doc_id = _match_intake_doc(docs)

    allergies = await client.search("AllergyIntolerance", {"patient": pid, "_count": 200})
    backfilled: list[dict] = []
    for a in allergies:
        # Skip rows that already have a tag in `note` — they don't
        # need backfill, the manifest reader sees them via FHIR.
        if _TAG_RE.search(_collect_note_text(a)):
            continue
        substance = _allergy_substance_text(a)
        if not intake_doc_id:
            # No intake doc on this patient's chart — nothing to map
            # the allergy to. Skip; the chart-card-scroll fallback
            # still works for these.
            continue
        bbox_hint = _bbox_hint_for_substance(substance)
        store.record(
            resource_type="AllergyIntolerance",
            resource_id=a.get("id") or "",
            source_doc_id=intake_doc_id,
            bbox=bbox_hint,
            label=substance,
        )
        backfilled.append({
            "resource_id": a.get("id"),
            "substance": substance,
            "source_doc_id": intake_doc_id,
            "bbox_hint": bbox_hint,
        })
    return {
        "name": name,
        "pid": pid,
        "intake_doc_id": intake_doc_id,
        "doc_count": len(docs),
        "allergy_count": len(allergies),
        "backfilled_count": len(backfilled),
        "backfilled": backfilled,
    }


async def main() -> None:
    names = sys.argv[1:] or DEFAULT_PATIENTS
    client = FhirClient()
    store = ExtractedSourcesStore(DB_PATH)
    try:
        await client._ensure_token()
        for name in names:
            r = await backfill_for_patient(client, store, name)
            if r.get("skipped"):
                print(f"[{name}] skipped: {r['skipped']}")
                continue
            print(
                f"[{r['name']}] pid={r['pid']} intake_doc={r['intake_doc_id']} "
                f"allergies={r['allergy_count']} backfilled={r['backfilled_count']}"
            )
            for b in r["backfilled"]:
                print(
                    f"    AllergyIntolerance/{b['resource_id']}  "
                    f"substance={b['substance']!r}  bbox={b['bbox_hint']}"
                )
    finally:
        store.close()
        await client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
