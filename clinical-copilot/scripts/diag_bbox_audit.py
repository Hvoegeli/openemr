"""One-shot bbox-audit diagnostic.

For each demo patient, walk every AllergyIntolerance / MedicationRequest /
Condition and report:
  - which DocumentReference each row was extracted from (via the
    [copilot-source: ...] tag the writer embeds in `note` / `comments`)
  - whether the tag carries bbox coordinates or not
  - the bbox values themselves (page / x / y / width / height)

Also dumps the full DocumentReference list per patient with their content
type so we can correlate "doc type X always has empty bbox tags" with the
specific writer paths in app/extraction/extract.py.

Usage:
  uv run python scripts/diag_bbox_audit.py [name1] [name2] ...
  (defaults to Whitaker, Chen, Reyes, Kowalski, Cohen)
"""

import asyncio
import json as _json
import re
import sys

from app.fhir.adapter import _collect_note_text
from app.fhir.client import FhirClient

DEFAULT_PATIENTS = ["Whitaker", "Chen", "Reyes", "Kowalski", "Cohen"]

# Match the writer's tag regardless of which doc it points to. Mirrors the
# adapter regex but without the doc-id filter (we want to enumerate every
# tag found on every row, not check for one specific target doc).
_TAG_RE = re.compile(
    r"\[copilot-source:\s*(?P<ref>[^\];]+)(?:\s*;\s*bbox=(?P<bbox>\{[^}]*\}))?\s*\]"
)


def _all_tags(note_text: str) -> list[dict]:
    out: list[dict] = []
    for m in _TAG_RE.finditer(note_text or ""):
        ref = (m.group("ref") or "").strip()
        bbox: dict | None = None
        raw = m.group("bbox")
        if raw:
            try:
                parsed = _json.loads(raw)
                if isinstance(parsed, dict):
                    bbox = parsed
            except (ValueError, TypeError):
                bbox = None
        out.append({"ref": ref, "bbox": bbox, "bbox_raw": raw})
    return out


def _name_of(p: dict) -> str:
    n = (p.get("name") or [{}])[0]
    given = " ".join(n.get("given") or [])
    family = n.get("family") or ""
    return f"{given} {family}".strip()


def _doc_label(d: dict) -> str:
    title = (d.get("description") or "").strip()
    if not title:
        for c in d.get("content") or []:
            att = c.get("attachment") or {}
            if att.get("title"):
                title = att["title"]
                break
    if not title:
        title = (d.get("type") or {}).get("text") or "(untitled)"
    cats = []
    for cat in d.get("category") or []:
        for c in cat.get("coding") or []:
            if c.get("display"):
                cats.append(c["display"])
                break
    return f"{title}" + (f"  [{','.join(cats)}]" if cats else "")


async def audit_patient(client: FhirClient, patient_name: str) -> None:
    print(f"\n=========== {patient_name} ===========")
    rows = await client.search("Patient", {"family": patient_name, "_count": 5})
    if not rows:
        # Server-side family search misses some name variants (Whitaker
        # in particular). Fall back to listing all patients and
        # substring-matching family name client-side.
        all_patients = await client.search("Patient", {"_count": 200})
        rows = [
            p for p in all_patients
            if any(
                patient_name.lower() in (n.get("family") or "").lower()
                for n in (p.get("name") or [])
            )
        ]
    if not rows:
        print(f"  ! no Patient found for family={patient_name}")
        return
    patient = rows[0]
    pid = patient["id"]
    print(f"Patient/{pid}  {_name_of(patient)}")

    # Documents
    docs = await client.search("DocumentReference", {"patient": pid, "_count": 100})
    print(f"\n  DocumentReferences ({len(docs)}):")
    doc_by_id = {d["id"]: d for d in docs}
    for d in docs:
        print(f"    DocumentReference/{d['id']}  {_doc_label(d)}")

    # Per-resource-type extraction tag walk.
    for rtype in ("AllergyIntolerance", "MedicationRequest", "Condition"):
        rows = await client.search(rtype, {"patient": pid, "_count": 200})
        tagged = 0
        with_bbox = 0
        without_bbox = 0
        no_tag = 0
        per_doc: dict[str, dict[str, int]] = {}
        details: list[str] = []
        for r in rows:
            note_text = _collect_note_text(r)
            tags = _all_tags(note_text)
            rid = r.get("id") or "?"
            if not tags:
                no_tag += 1
                continue
            tagged += 1
            for tag in tags:
                doc_id = (tag.get("ref") or "").removeprefix("DocumentReference/")
                bbox = tag.get("bbox")
                slot = per_doc.setdefault(doc_id, {"with_bbox": 0, "without_bbox": 0})
                if bbox:
                    with_bbox += 1
                    slot["with_bbox"] += 1
                    p_ = bbox.get("page", "?")
                    x_ = bbox.get("x", "?")
                    y_ = bbox.get("y", "?")
                    w_ = bbox.get("width", "?")
                    h_ = bbox.get("height", "?")
                    details.append(
                        f"      {rtype}/{rid} -> doc={doc_id}  "
                        f"page={p_} x={x_} y={y_} w={w_} h={h_}"
                    )
                else:
                    without_bbox += 1
                    slot["without_bbox"] += 1
                    details.append(
                        f"      {rtype}/{rid} -> doc={doc_id}  bbox=NONE"
                    )
        print(
            f"\n  {rtype}: total={len(rows)}  tagged={tagged}  "
            f"with_bbox={with_bbox}  without_bbox={without_bbox}  no_tag={no_tag}"
        )
        for doc_id, slot in per_doc.items():
            label = _doc_label(doc_by_id[doc_id]) if doc_id in doc_by_id else "(doc not found in patient roster)"
            print(
                f"    via DocumentReference/{doc_id} ({label}): "
                f"with_bbox={slot['with_bbox']} without_bbox={slot['without_bbox']}"
            )
        for line in details:
            print(line)


async def main() -> None:
    names = sys.argv[1:] or DEFAULT_PATIENTS
    client = FhirClient()
    try:
        await client._ensure_token()
        for name in names:
            await audit_patient(client, name)
    finally:
        await client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
