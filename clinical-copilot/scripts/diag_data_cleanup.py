"""Read-only advisory: walk every demo patient's chart, flag resources
whose `[copilot-source: <doc>]` tag points at a DocumentReference that
no longer exists or sits on the wrong patient, and emit a concrete
remediation checklist.

This is the demo-prep script — run it before the demo to know exactly
which patients have broken-citation data and what to do about each one.

Run:
  ssh root@<hetzner> '
    export PATH=/root/.local/bin:$PATH PYTHONPATH=/root/openemr/clinical-copilot
    cd /root/openemr/clinical-copilot && uv run python scripts/diag_data_cleanup.py
  '
"""

import asyncio
import json as _json
import re
import sys

from app.fhir.adapter import _collect_note_text
from app.fhir.client import FhirClient

DEFAULT_PATIENTS = ["Whitaker", "Chen", "Reyes", "Kowalski", "Cohen"]

_TAG_RE = re.compile(
    r"\[copilot-source:\s*(?P<ref>[^\];]+)(?:\s*;\s*bbox=(?P<bbox>\{[^}]*\}))?\s*\]"
)


def _all_tag_refs(text: str) -> list[str]:
    return [(m.group("ref") or "").strip() for m in _TAG_RE.finditer(text or "")]


async def _resolve_patient(client: FhirClient, name: str) -> dict | None:
    rows = await client.search("Patient", {"family": name, "_count": 5})
    if rows:
        return rows[0]
    rows = await client.search("Patient", {"_count": 200})
    for p in rows:
        for n in p.get("name") or []:
            if name.lower() in (n.get("family") or "").lower():
                return p
    return None


async def _doc_exists(client: FhirClient, doc_id: str) -> bool:
    try:
        await client.get(f"DocumentReference/{doc_id}")
        return True
    except Exception:  # noqa: BLE001
        return False


async def audit(client: FhirClient, name: str) -> dict:
    patient = await _resolve_patient(client, name)
    if not patient:
        return {"name": name, "error": "patient not found"}
    pid = patient["id"]
    family_in_fhir = ((patient.get("name") or [{}])[0].get("family") or "")

    docs = await client.search("DocumentReference", {"patient": pid, "_count": 200})
    doc_ids_on_chart = {d["id"] for d in docs}
    docs_with_subject_mismatch: list[dict] = []
    for d in docs:
        title = ""
        for c in d.get("content") or []:
            att = c.get("attachment") or {}
            if att.get("title"):
                title = att["title"]
                break
        # Heuristic: if the family name in the doc title doesn't match
        # the patient's family name in FHIR, the doc was probably
        # uploaded onto the wrong patient. (Demo-data-only heuristic;
        # production would have explicit doc<->patient validation.)
        title_lower = title.lower()
        family_lower = family_in_fhir.lower()
        suspicious = (
            family_lower
            and family_lower not in title_lower
            and any(other.lower() in title_lower
                    for other in DEFAULT_PATIENTS if other.lower() != family_lower)
        )
        if suspicious:
            docs_with_subject_mismatch.append({
                "doc_id": d["id"], "title": title, "subject_pid": pid,
            })

    broken_resources: list[dict] = []
    for rtype in ("AllergyIntolerance", "MedicationRequest", "Condition"):
        rows = await client.search(rtype, {"patient": pid, "_count": 200})
        for r in rows:
            refs = _all_tag_refs(_collect_note_text(r))
            for ref in refs:
                doc_id = ref.removeprefix("DocumentReference/")
                if doc_id in doc_ids_on_chart:
                    continue
                exists = await _doc_exists(client, doc_id)
                broken_resources.append({
                    "rtype": rtype,
                    "rid": r.get("id"),
                    "label": (r.get("code") or {}).get("text")
                    or ((r.get("code") or {}).get("coding") or [{}])[0].get("display")
                    or (r.get("medicationCodeableConcept") or {}).get("text")
                    or "?",
                    "phantom_doc_id": doc_id,
                    "doc_exists": exists,
                    "doc_status": "deleted" if not exists else "wrong-patient",
                })

    return {
        "name": name,
        "pid": pid,
        "fhir_family": family_in_fhir,
        "doc_count": len(docs),
        "docs_with_subject_mismatch": docs_with_subject_mismatch,
        "broken_resources": broken_resources,
    }


def _print_report(reports: list[dict]) -> None:
    print("\n" + "=" * 72)
    print("DATA-CLEANUP REPORT — broken citation deep-links per demo patient")
    print("=" * 72)
    for r in reports:
        print(f"\n--- {r['name']} ---")
        if "error" in r:
            print(f"  ERROR: {r['error']}")
            continue
        print(f"  Patient/{r['pid']}  family={r['fhir_family']!r}  docs={r['doc_count']}")
        if not r["broken_resources"] and not r["docs_with_subject_mismatch"]:
            print("  ✓ no data issues — citations should deep-link cleanly")
            continue
        if r["docs_with_subject_mismatch"]:
            print("  ⚠ docs uploaded onto the wrong patient (title mentions another demo patient):")
            for d in r["docs_with_subject_mismatch"]:
                print(f"      DocumentReference/{d['doc_id']}  {d['title']!r}")
            print("    → ACTION: re-upload these onto the correct patient, then "
                  "soft-hide the wrong-patient copies via the docs UI.")
        if r["broken_resources"]:
            phantoms = sorted({b['phantom_doc_id'] for b in r["broken_resources"]
                               if not b['doc_exists']})
            print(f"  ⚠ {len(r['broken_resources'])} resources cite docs that 404 / aren't on this chart:")
            for b in r["broken_resources"][:20]:
                print(f"      {b['rtype']}/{b['rid']}  label={b['label']!r}  "
                      f"doc={b['phantom_doc_id']}  status={b['doc_status']}")
            if len(r["broken_resources"]) > 20:
                print(f"      … (+{len(r['broken_resources']) - 20} more)")
            if phantoms:
                print(f"    → ACTION: these citations show 'doc not loaded' on click. "
                      f"For the demo, either avoid clicking these resources, or "
                      f"delete them from OpenEMR so they don't appear in the chart.")
    print("\n" + "=" * 72)
    print("DEMO DECISION GUIDE")
    print("=" * 72)
    for r in reports:
        if "error" in r:
            verdict = "SKIP — patient not found"
        elif r["doc_count"] == 0:
            verdict = "SKIP — no docs to demo (chart card only, no PDF deep-link possible)"
        elif r["docs_with_subject_mismatch"]:
            verdict = "RISKY — docs are on the wrong patient (re-upload before demo)"
        elif r["broken_resources"]:
            verdict = (
                f"USE WITH CAUTION — {len(r['broken_resources'])} citations 404; "
                "click only meds/conditions that have a green 'with-bbox' tag"
            )
        else:
            verdict = "GOOD — citations deep-link cleanly"
        print(f"  {r['name']:<10} {verdict}")
    print()


async def main() -> None:
    names = sys.argv[1:] or DEFAULT_PATIENTS
    client = FhirClient()
    try:
        await client._ensure_token()
        reports = []
        for n in names:
            reports.append(await audit(client, n))
        _print_report(reports)
        # Also dump as JSON at the end for any downstream tooling.
        print("\n--- JSON ---")
        print(_json.dumps(reports, indent=2, default=str))
    finally:
        await client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
