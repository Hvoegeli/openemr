"""Round-2 bbox diagnostic — verify Chen's bboxes against page dims, dump
raw allergy notes, and resolve the orphaned/mis-routed DocumentReferences
flagged by the round-1 audit.

Run: PYTHONPATH=/root/openemr/clinical-copilot uv run python scripts/diag_bbox_deep.py
"""

import asyncio
import io
import json as _json
import re

from PIL import Image

from app.fhir.adapter import _collect_note_text
from app.fhir.client import FhirClient

CHEN_INTAKE_DOC = "a1b7fb7f-1a41-4e96-a122-8914e3debff2"
REYES_PHANTOM_DOC = "a1b6102a-db27-435f-9485-fa99bd18e527"
KOWALSKI_PHANTOM_DOC = "a1b5ff3f-b080-4bd0-8f60-3ac66d6c04c4"

CHEN_PID = "a1b5833f-be5c-4bb5-b214-f7ad1d3c55a0"
KOWALSKI_PID = "a1b41849-498a-4fd5-9217-668de8bccc60"
WHITAKER_PID = "a1b8077f-6227-419d-9de6-52976c6a63c0"
REYES_PID = "a1b41847-3522-4d89-bda5-5dcd51da7d25"

_TAG_RE = re.compile(
    r"\[copilot-source:\s*(?P<ref>[^\];]+)(?:\s*;\s*bbox=(?P<bbox>\{[^}]*\}))?\s*\]"
)


def _all_tags(text: str) -> list[dict]:
    out = []
    for m in _TAG_RE.finditer(text or ""):
        bbox = None
        raw = m.group("bbox")
        if raw:
            try:
                parsed = _json.loads(raw)
                if isinstance(parsed, dict):
                    bbox = parsed
            except (ValueError, TypeError):
                bbox = None
        out.append({"ref": (m.group("ref") or "").strip(), "bbox": bbox})
    return out


# ─────────────────────── (1) Chen bbox vs page dims ───────────────────────


async def chen_bbox_vs_page_dims(client: FhirClient) -> None:
    print("\n========= (1) Chen intake.pdf bbox vs rendered page dims =========")
    # Render via the same code path the viewer uses, then check page-2 dims.
    from app.fhir import adapter
    result = await adapter.get_document_pages(
        client, document_id=CHEN_INTAKE_DOC, panel=None, max_pages=10,
    )
    pages = result["data"]["pages"]
    print(f"  Rendered {len(pages)} pages from intake.pdf")
    page_dims = {}
    for p in pages:
        page_dims[p["page"]] = (p["width_px"], p["height_px"])
        print(f"    page {p['page']}:  {p['width_px']} x {p['height_px']} px")

    # Walk Chen's tagged meds + conditions, compare bboxes against page dims.
    print("\n  Tagged resources from this doc:")
    for rtype in ("MedicationRequest", "Condition"):
        rows = await client.search(rtype, {"patient": CHEN_PID, "_count": 200})
        for r in rows:
            tags = _all_tags(_collect_note_text(r))
            for tag in tags:
                doc = (tag.get("ref") or "").removeprefix("DocumentReference/")
                if doc != CHEN_INTAKE_DOC:
                    continue
                bbox = tag.get("bbox")
                if not bbox:
                    print(f"    {rtype}/{r.get('id','?')}  bbox=NONE")
                    continue
                page = bbox.get("page")
                x = bbox.get("x")
                y = bbox.get("y")
                w = bbox.get("width")
                h = bbox.get("height")
                w_px, h_px = page_dims.get(page, (None, None))
                fits = "?"
                if isinstance(w_px, int) and isinstance(h_px, int):
                    in_bounds = (
                        isinstance(x, (int, float)) and isinstance(y, (int, float))
                        and isinstance(w, (int, float)) and isinstance(h, (int, float))
                        and 0 <= x and 0 <= y
                        and x + w <= w_px and y + h <= h_px
                    )
                    fits = "OK" if in_bounds else "OUT-OF-BOUNDS"
                    pct_x = round(x / w_px * 100, 1) if isinstance(x, (int, float)) else "?"
                    pct_y = round(y / h_px * 100, 1) if isinstance(y, (int, float)) else "?"
                    pct_w = round(w / w_px * 100, 1) if isinstance(w, (int, float)) else "?"
                    print(
                        f"    {rtype}/{r.get('id','?')}  page={page} "
                        f"bbox=({x},{y},{w},{h}) page_dim=({w_px}x{h_px})  "
                        f"-> {fits}  (left={pct_x}% top={pct_y}% width={pct_w}%)"
                    )
                else:
                    print(
                        f"    {rtype}/{r.get('id','?')}  page={page} "
                        f"NO-RENDERED-PAGE-DIM bbox=({x},{y},{w},{h})"
                    )


# ─────────────────────── (2) raw allergy notes ───────────────────────


async def dump_allergy_notes(client: FhirClient) -> None:
    print("\n========= (2) Raw AllergyIntolerance.note for each demo patient =========")
    targets = [("Chen", CHEN_PID), ("Kowalski", KOWALSKI_PID), ("Reyes", REYES_PID)]
    for label, pid in targets:
        rows = await client.search("AllergyIntolerance", {"patient": pid, "_count": 200})
        print(f"\n  {label} (Patient/{pid}) — {len(rows)} allergies:")
        for i, r in enumerate(rows):
            code = ((r.get("code") or {}).get("text")
                    or ((r.get("code") or {}).get("coding") or [{}])[0].get("display")
                    or "?")
            notes = r.get("note") or []
            note_texts = [(n.get("text") or "") for n in notes]
            print(f"    [{i}] AllergyIntolerance/{r.get('id','?')}  substance={code!r}")
            print(f"        note count: {len(note_texts)}")
            for j, t in enumerate(note_texts):
                # Show note text fully so we can see if a tag is present
                # but malformed (or stripped). PHI risk: low — synthetic
                # demo patients only.
                print(f"        note[{j}] = {t!r}")
            if not note_texts:
                print("        (no notes at all — write_allergy never wrote a back-ref)")


# ─────────────────────── (3) Doc routing audit ───────────────────────


async def doc_routing_audit(client: FhirClient) -> None:
    print("\n========= (3) Document routing audit =========")

    # 3a — Whitaker has 0 docs; Kowalski has docs named p02-whitaker-*.
    print("\n  (3a) Kowalski's docs — check subject + content title:")
    docs = await client.search("DocumentReference", {"patient": KOWALSKI_PID, "_count": 100})
    for d in docs:
        subj = (d.get("subject") or {}).get("reference")
        title = ""
        for c in d.get("content") or []:
            att = c.get("attachment") or {}
            if att.get("title"):
                title = att["title"]
                break
        descr = (d.get("description") or "").strip()
        print(f"    DocumentReference/{d['id']}  subject={subj}")
        print(f"      description: {descr[:160]!r}")
        print(f"      attachment title: {title[:160]!r}")

    # 3b — Reyes's "phantom" doc id: does it exist? Whose patient is it?
    print(f"\n  (3b) Reyes's phantom source doc {REYES_PHANTOM_DOC}:")
    try:
        d = await client.get(f"DocumentReference/{REYES_PHANTOM_DOC}")
        subj = (d.get("subject") or {}).get("reference")
        print(f"    EXISTS — subject={subj}")
        print(f"    description: {(d.get('description') or '')[:200]!r}")
    except Exception as e:
        print(f"    GET failed: {e}")

    # 3c — Kowalski's other "phantom" doc id (referenced by some meds/conds):
    print(f"\n  (3c) Kowalski's phantom source doc {KOWALSKI_PHANTOM_DOC}:")
    try:
        d = await client.get(f"DocumentReference/{KOWALSKI_PHANTOM_DOC}")
        subj = (d.get("subject") or {}).get("reference")
        print(f"    EXISTS — subject={subj}")
        print(f"    description: {(d.get('description') or '')[:200]!r}")
    except Exception as e:
        print(f"    GET failed: {e}")

    # 3d — Whitaker patient roster — confirm 0 docs & list any allergies/meds/conds
    print(f"\n  (3d) Whitaker (Patient/{WHITAKER_PID}) — full chart:")
    for rtype in ("DocumentReference", "AllergyIntolerance", "MedicationRequest", "Condition", "Encounter"):
        rows = await client.search(rtype, {"patient": WHITAKER_PID, "_count": 100})
        print(f"    {rtype}: {len(rows)}")


async def main() -> None:
    client = FhirClient()
    try:
        await client._ensure_token()
        await chen_bbox_vs_page_dims(client)
        await dump_allergy_notes(client)
        await doc_routing_audit(client)
    finally:
        await client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
