"""High-level FHIR fetchers for the clinical co-pilot.

Each function returns the data the agent needs PLUS a `sources` list of FHIR
resource references (`Patient/123`, `Observation/8821`, ...). The verification
node downstream rejects any agent claim not backed by a source ID from a
tool call in the same conversation.
"""

import base64
import io
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import TypedDict
from zoneinfo import ZoneInfo

from app.fhir.client import FhirClient

log = logging.getLogger("agent.fhir.adapter")

_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _clinical_iso(iso_str: str | None) -> str | None:
    """Re-stamp a UTC ISO timestamp with the clinical-local TZ offset baked in.

    FHIR returns `effectiveDateTime` in UTC (e.g. `2026-04-30T21:35:48Z`).
    Surfacing that string directly to the agent makes it talk about "21:35"
    when the doctor's wall clock said 15:35 MDT — confusing in chat output.
    Convert here so the offset is on the wire (`2026-04-30T15:35:48-06:00`)
    and downstream parsers (Python, JS) still get an unambiguous instant.
    """
    if not iso_str:
        return None
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
    except ValueError:
        return iso_str
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    from app.config import settings  # lazy to avoid circular import on module load
    return dt.astimezone(ZoneInfo(settings.clinical_tz)).isoformat(timespec="seconds")


class SourcedResult(TypedDict):
    data: dict | list
    sources: list[str]


def _ref(resource: dict) -> str:
    """Format a FHIR resource as a citation reference, e.g. 'Patient/123'."""
    return f"{resource['resourceType']}/{resource['id']}"


def _narrative_text(resource: dict) -> str | None:
    """Extract human-readable text from a FHIR resource's narrative `text.div`.

    OpenEMR puts free-text titles for allergies/problems/meds into the
    narrative when no SNOMED/RxNorm code is supplied. Strip the HTML wrapper
    so the agent sees a plain string.
    """
    div = (resource.get("text") or {}).get("div")
    if not isinstance(div, str):
        return None
    stripped = _HTML_TAG_RE.sub("", div).strip()
    return stripped or None


def _coded_display(coded: dict) -> str | None:
    """Best-effort label for a FHIR CodeableConcept: text, then first display.

    Skips FHIR `data-absent-reason` placeholder codings (system ends in
    `data-absent-reason`, code `unknown`/`asked-unknown`/`temp-unknown`/etc.) —
    OpenEMR emits those when no SNOMED/RxNorm code is supplied, and the real
    label is in the narrative `text.div` instead.
    """
    if not isinstance(coded, dict):
        return None
    text = coded.get("text")
    if isinstance(text, str) and text.strip():
        return text.strip()
    for coding in coded.get("coding") or []:
        if not isinstance(coding, dict):
            continue
        system = coding.get("system") or ""
        if "data-absent-reason" in system:
            continue
        display = coding.get("display")
        if isinstance(display, str) and display.strip():
            return display.strip()
        code_val = coding.get("code")
        if isinstance(code_val, str) and code_val.strip():
            return code_val.strip()
    return None


async def resolve_patient(
    client: FhirClient,
    *,
    query: str,
    doctor_panel_ids: list[str] | None = None,
) -> SourcedResult:
    """Resolve a free-text patient reference (bed number, last name, MRN) to a Patient.

    Returns the best match with an `alternatives` list when ambiguous so the
    agent can ask the doctor to disambiguate rather than guessing.
    """
    # Try MRN first (most precise), then last name, then bed number lookup
    # via the related Encounter.location. For now this is a simple name search;
    # bed and MRN paths are TODO once we have demo data shaped.
    matches = await client.search("Patient", {"family": query, "_count": 5})

    # `is not None` (not truthiness) so an empty allow-list correctly drops
    # every match — `if doctor_panel_ids:` would treat `[]` as "no filter"
    # and leak patients outside the caller's panel.
    if doctor_panel_ids is not None:
        matches = [p for p in matches if p["id"] in doctor_panel_ids]

    if not matches:
        return {"data": {"found": False, "query": query}, "sources": []}

    best = matches[0]
    return {
        "data": {
            "found": True,
            "patient_id": best["id"],
            "name": _format_name(best),
            "alternatives": [
                {"patient_id": p["id"], "name": _format_name(p)} for p in matches[1:]
            ],
        },
        "sources": [_ref(p) for p in matches],
    }


async def get_patient_card(client: FhirClient, *, patient_id: str) -> SourcedResult:
    """Fetch the structured data that powers the right-side patient card.

    All six FHIR queries fan out in parallel via `asyncio.gather` — sequential
    awaits added ~3s on a local stack and ~5s through cloudflared. They are
    independent (no result feeds the next), so parallelism is safe.
    """
    import asyncio

    # OpenEMR's FHIR layer is selective about which search params it honors.
    # Using broad searches and filtering client-side is safer than passing
    # `status` / `clinical-status` filters that silently return zero.
    patient, encounters_all, allergies, problems_all, meds, vitals = await asyncio.gather(
        client.get(f"Patient/{patient_id}"),
        client.search("Encounter", {"patient": patient_id, "_sort": "-date", "_count": 5}),
        client.search("AllergyIntolerance", {"patient": patient_id}),
        client.search("Condition", {"patient": patient_id}),
        client.search("MedicationRequest", {"patient": patient_id, "_count": 10}),
        client.search(
            "Observation",
            # 50 observations = roughly 5 vitals-panel timepoints (one rounding
            # event encodes as ~6-10 separate Observation resources). 10 was
            # too tight — the agent only ever saw a single timepoint and
            # couldn't form trends. Matches the trends endpoint's _count.
            {"patient": patient_id, "category": "vital-signs", "_count": 50},
        ),
    )

    encounters = [
        e for e in encounters_all
        if e.get("status") in {"in-progress", "arrived", "triaged", "finished"}
    ][:1]
    problems = [
        c for c in problems_all
        if (
            (c.get("clinicalStatus") or {}).get("coding", [{}])[0].get("code") in {"active", "recurrence", "relapse", None}
        )
    ]

    sources = (
        [_ref(patient)]
        + [_ref(e) for e in encounters]
        + [_ref(a) for a in allergies]
        + [_ref(p) for p in problems]
        + [_ref(m) for m in meds]
        + [_ref(v) for v in vitals]
    )

    return {
        "data": {
            "name": _format_name(patient),
            "mrn": patient_id,
            "age": _calc_age(patient.get("birthDate")),
            "sex": patient.get("gender"),
            "current_encounter": encounters[0] if encounters else None,
            "allergies": [_format_allergy(a) for a in allergies],
            "active_problems": [_format_condition(c) for c in problems],
            "active_medications": [_format_med(m) for m in meds],
            "recent_vitals": [r for v in vitals for r in _format_vital(v)],
        },
        "sources": sources,
    }


# ─── formatters ──────────────────────────────────────────────────────────


def _format_name(patient: dict) -> str:
    name = (patient.get("name") or [{}])[0]
    given = " ".join(name.get("given", []))
    family = name.get("family", "")
    return f"{given} {family}".strip() or "(unknown)"


def _calc_age(birth_date: str | None) -> int | None:
    if not birth_date:
        return None
    from datetime import date

    born = date.fromisoformat(birth_date)
    today = date.today()
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))


def _format_allergy(a: dict) -> dict:
    return {
        "id": a["id"],
        "substance": _coded_display(a.get("code", {})) or _narrative_text(a),
        "reaction": [
            r.get("manifestation", [{}])[0].get("text") for r in a.get("reaction", [])
        ],
        "severity": a.get("criticality"),
    }


def _format_condition(c: dict) -> dict:
    return {
        "id": c["id"],
        "name": _coded_display(c.get("code", {})) or _narrative_text(c),
        "onset": c.get("onsetDateTime"),
    }


def _format_med(m: dict) -> dict:
    dosage = (m.get("dosageInstruction") or [{}])[0]
    return {
        "id": m["id"],
        "drug": _coded_display(m.get("medicationCodeableConcept", {})) or _narrative_text(m),
        "dose_text": dosage.get("text"),
        "started": m.get("authoredOn"),
    }


def _format_vital(v: dict) -> list[dict]:
    """Flatten one Observation into one or more vital rows.

    A blood-pressure Observation in FHIR is a single resource with two
    `component` entries (systolic + diastolic) and no top-level
    `valueQuantity`. Reading `valueQuantity` directly returns None, which
    propagates as "not recorded" to every consumer (agent answers, card UI).
    Split composite BP into two flat rows so the values are visible.

    Times are re-stamped in clinical-local TZ — the agent narrates "15:35
    MDT" instead of "21:35" UTC.
    """
    name = _coded_display(v.get("code", {})) or _narrative_text(v) or ""
    name_lower = name.lower()
    eff_time = _clinical_iso(v.get("effectiveDateTime"))
    obs_id = v["id"]
    components = v.get("component") or []

    is_bp_composite = (
        ("blood pressure" in name_lower or name_lower in ("bp", "bp panel"))
        and components
    )
    if is_bp_composite:
        rows: list[dict] = []
        for comp in components:
            comp_name = (_coded_display(comp.get("code", {})) or "").lower()
            qty = comp.get("valueQuantity") or {}
            val = qty.get("value")
            if val is None:
                continue
            if "systolic" in comp_name:
                rows.append({"id": obs_id, "name": "Systolic BP",
                             "value": val, "unit": qty.get("unit"), "time": eff_time})
            elif "diastolic" in comp_name:
                rows.append({"id": obs_id, "name": "Diastolic BP",
                             "value": val, "unit": qty.get("unit"), "time": eff_time})
        return rows

    return [{
        "id": obs_id,
        "name": name or None,
        "value": v.get("valueQuantity", {}).get("value"),
        "unit": v.get("valueQuantity", {}).get("unit"),
        "time": eff_time,
    }]


# ─── time-windowed Use Case A primitives ─────────────────────────────────


def _category_code(o: dict) -> str | None:
    """First non-empty `category.coding.code` on a resource. Used to label
    Observations as `vital-signs` vs `laboratory` vs other. Returns None if
    no code is present (some OpenEMR responses omit category entirely)."""
    for cat in o.get("category") or []:
        for coding in cat.get("coding") or []:
            code = coding.get("code")
            if isinstance(code, str) and code.strip():
                return code.strip()
    return None


def _utc_cutoff_iso(hours: int) -> str:
    """ISO-8601 UTC cutoff `now - hours`, suitable for FHIR `date=ge…`.

    FHIR accepts both `Z` and `+00:00` suffixes; we use `Z` for brevity and
    so OpenEMR's PHP date parser doesn't have to handle the timezone offset.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    return cutoff.strftime("%Y-%m-%dT%H:%M:%SZ")


async def get_observations_24h(
    client: FhirClient, *, patient_id: str, hours: int = 24,
) -> SourcedResult:
    """Observations (labs + vitals) recorded for the patient in the last `hours`.

    Uses FHIR `date=ge<cutoff>` for server-side filtering — narrower than
    `get_patient_card.recent_vitals` and gives the agent a real "what's new"
    answer instead of "here's the last 50, you figure out which are recent".

    Composite BP Observations (single resource, systolic+diastolic in
    `component`) are split via `_format_vital` so values aren't lost. Times
    are re-stamped to clinical-local TZ via `_clinical_iso`. We also apply
    a client-side filter on `effectiveDateTime` as defense-in-depth: if
    the FHIR layer ever silently drops the `date` filter, the agent still
    sees only in-window rows instead of every Observation up to _count.
    """
    cutoff_iso = _utc_cutoff_iso(hours)
    cutoff_dt = datetime.now(timezone.utc) - timedelta(hours=hours)
    obs = await client.search(
        "Observation",
        {"patient": patient_id, "date": f"ge{cutoff_iso}", "_count": 100},
    )
    rows: list[dict] = []
    sources: list[str] = []
    for o in obs:
        if not _is_recent(o.get("effectiveDateTime"), cutoff_dt):
            continue
        sources.append(_ref(o))
        cat = _category_code(o)
        for row in _format_vital(o):
            row["category"] = cat
            rows.append(row)
    rows.sort(key=lambda r: r.get("time") or "", reverse=True)
    return {
        "data": {
            "window_hours": hours,
            "cutoff_iso": cutoff_iso,
            "count": len(rows),
            "observations": rows,
        },
        "sources": sources,
    }


async def get_notes_24h(
    client: FhirClient, *, patient_id: str, hours: int = 24,
) -> SourcedResult:
    """`DocumentReference`s (clinical notes, progress notes, summaries)
    created for the patient in the last `hours`. Metadata only — title,
    type, date, status, attachment titles. Full-text fetch is a separate
    (planned) tool; metadata alone already answers "did the night team
    document anything I should know about?"."""
    cutoff_iso = _utc_cutoff_iso(hours)
    cutoff_dt = datetime.now(timezone.utc) - timedelta(hours=hours)
    docs = await client.search(
        "DocumentReference",
        {"patient": patient_id, "date": f"ge{cutoff_iso}", "_count": 30},
    )
    items: list[dict] = []
    sources: list[str] = []
    for d in docs:
        if not _is_recent(d.get("date"), cutoff_dt):
            continue
        sources.append(_ref(d))
        attachments: list[dict] = []
        for content in d.get("content") or []:
            att = content.get("attachment") or {}
            attachments.append({
                "title": att.get("title"),
                "content_type": att.get("contentType"),
            })
        items.append({
            "id": d["id"],
            "title": _coded_display(d.get("type") or {}) or "Document",
            "date": _clinical_iso(d.get("date")),
            "status": d.get("status"),
            "category": _coded_display((d.get("category") or [{}])[0]),
            "attachments": attachments,
        })
    items.sort(key=lambda x: x.get("date") or "", reverse=True)
    return {
        "data": {
            "window_hours": hours,
            "cutoff_iso": cutoff_iso,
            "count": len(items),
            "documents": items,
        },
        "sources": sources,
    }


async def get_document_content(
    client: FhirClient,
    *,
    document_id: str,
    panel: frozenset[str] | None = None,
    max_pages: int = 10,
) -> SourcedResult:
    """Fetch a `DocumentReference`'s attachment bytes and render them to PNG
    pages so the agent can read the actual document content (PDF text,
    scanned images) rather than just metadata.

    Companion to `get_notes_24h`, which only returns metadata. The agent
    typically discovers a document_id via `get_notes_24h` (or another
    tool) and then calls this when the doctor asks "what does the doc
    say?", "summarize the intake form", etc.

    ACL: when `panel` is non-None, the DocumentReference's `subject.reference`
    must point to a Patient inside the panel; otherwise raises
    `PatientAccessDenied`. This fails CLOSED — a doc with no Patient subject
    is also denied when a panel is enforced, since we have no basis to
    decide it belongs to a patient inside the panel. Prevents a user from
    fetching a doc by guessing its UUID for a patient outside their
    assignment.

    Returns `data` containing metadata + a `pages_png_b64` list (base64
    PNG strings, one per rendered page across all attachments). The graph
    layer special-cases this tool to surface the pages as image blocks in
    a multimodal `ToolMessage` so Claude can actually see them.

    `max_pages` caps total page count across all attachments to keep the
    message under Anthropic's per-request limits; `data.pages_truncated`
    is true when the cap kicked in.
    """
    from app.access_control import PatientAccessDenied  # lazy: avoids circular import
    from app.extraction.render import render_to_png_pages

    doc = await client.get(f"DocumentReference/{document_id}")

    subject_ref = (doc.get("subject") or {}).get("reference") or ""
    patient_id: str | None = None
    if subject_ref.startswith("Patient/"):
        patient_id = subject_ref.removeprefix("Patient/")
    if panel is not None and (patient_id is None or patient_id not in panel):
        # Fail closed: when a panel is enforced and we cannot establish
        # a Patient subject inside it, deny. PatientAccessDenied requires
        # a string `patient_id` field; pass an empty string when the doc
        # has no Patient subject so the audit log shows the denial without
        # inventing a UUID.
        raise PatientAccessDenied(patient_id or "")

    attachments_meta: list[dict] = []
    pages_b64: list[str] = []
    truncated = False

    for idx, content_entry in enumerate(doc.get("content") or []):
        att = content_entry.get("attachment") or {}
        content_type = att.get("contentType") or ""
        title = att.get("title")
        att_meta = {
            "index": idx,
            "title": title,
            "content_type": content_type,
            "page_count": 0,
        }

        file_bytes = await _fetch_attachment_bytes(client, att)
        if file_bytes is None:
            att_meta["error"] = "no inline data or fetchable url"
            attachments_meta.append(att_meta)
            continue

        try:
            pngs = render_to_png_pages(file_bytes, content_type)
        except ValueError as e:
            att_meta["error"] = str(e)
            attachments_meta.append(att_meta)
            continue

        delivered = 0
        for png in pngs:
            if len(pages_b64) >= max_pages:
                truncated = True
                break
            pages_b64.append(_b64_png(png))
            delivered += 1
        att_meta["page_count"] = delivered
        attachments_meta.append(att_meta)
        if truncated:
            break

    return {
        "data": {
            "document_id": document_id,
            "title": _coded_display(doc.get("type") or {}) or "Document",
            "category": _coded_display((doc.get("category") or [{}])[0]),
            "date": _clinical_iso(doc.get("date")),
            "status": doc.get("status"),
            "patient_id": patient_id,
            "attachments": attachments_meta,
            "page_count": len(pages_b64),
            "max_pages": max_pages,
            "pages_truncated": truncated,
            "pages_png_b64": pages_b64,
        },
        "sources": [_ref(doc)],
    }


async def _fetch_attachment_bytes(
    client: FhirClient, attachment: dict,
) -> bytes | None:
    """Materialize the raw bytes for a FHIR `Attachment` element.

    Two paths per FHIR spec, in order of preference:
      - `data` is a base64 string with the bytes inline.
      - `url` references the bytes elsewhere — typically a relative
        `Binary/{id}` on the same OpenEMR server, but absolute URLs are
        also legal in FHIR.
    Returns None if neither path yields bytes (caller surfaces this to the
    agent as a per-attachment error rather than failing the whole call).
    Absolute URLs that don't point at our own FHIR base are refused
    (returned as None) — `client.get_raw` always prepends our base, and
    we don't want to silently fan our system access token out to a
    third-party host even if the FHIR server claims a doc lives there.
    """
    inline = attachment.get("data")
    if isinstance(inline, str) and inline:
        try:
            return base64.b64decode(inline, validate=False)
        except (ValueError, TypeError):
            return None

    url = attachment.get("url")
    if not isinstance(url, str) or not url:
        return None

    if url.startswith(("http://", "https://")):
        # Lazy import: matches the rest of this module's pattern of avoiding
        # an `app.config` import at module load.
        from app.config import settings  # noqa: PLC0415
        base = settings.openemr_fhir_base_url.rstrip("/") + "/"
        if not url.startswith(base):
            return None
        path = url[len(base):]
    else:
        path = url

    bytes_, _ctype = await client.get_raw(path)
    return bytes_


def _b64_png(png_bytes: bytes) -> str:
    """Base64-encode PNG bytes for the Anthropic image-block envelope.
    Standalone helper so the graph layer can re-build image blocks from
    the SourcedResult without re-rendering pages."""
    return base64.standard_b64encode(png_bytes).decode("ascii")


# Bbox source-tag persisted by `OpenEMRWriter._source_back_ref` into a
# resource's `comments` / FHIR `note.text` field. Shape on disk:
#   `[copilot-source: DocumentReference/<id>; bbox={"page":1,"x":..,"y":..,"width":..,"height":..}]`
# The bbox JSON is optional. The pattern tolerates surrounding whitespace
# and the optional `bbox=` segment so rows persisted before bbox support
# (Phase-3 backfill) still parse out their source DocumentReference id.
_SOURCE_TAG_RE = re.compile(
    r"\[copilot-source:\s*(?P<ref>[^\];]+)(?:\s*;\s*bbox=(?P<bbox>\{[^}]*\}))?\s*\]",
)


def _parse_source_tag(text: str | None, document_id: str) -> dict | None:
    """Pull `(ref, bbox_dict)` out of a comments/note string when it
    references the given DocumentReference. Returns None when the text
    is empty, when no source tag is present, or when the tag points at
    a DIFFERENT DocumentReference. Malformed bbox JSON yields None for
    the bbox field but the function still returns a dict — the caller
    can decide whether a citation without coords is worth keeping in
    the manifest (it isn't, today, but the ref is still useful audit).
    """
    if not text:
        return None
    for m in _SOURCE_TAG_RE.finditer(text):
        ref = m.group("ref").strip()
        target = ref if "/" in ref else f"DocumentReference/{ref}"
        wanted = (
            document_id if "/" in document_id else f"DocumentReference/{document_id}"
        )
        if target != wanted:
            continue
        bbox_dict: dict | None = None
        bbox_raw = m.group("bbox")
        if bbox_raw:
            import json as _json  # noqa: PLC0415 — module-private use
            try:
                parsed = _json.loads(bbox_raw)
                if isinstance(parsed, dict):
                    bbox_dict = parsed
            except (ValueError, TypeError):
                bbox_dict = None
        return {"ref": target, "bbox": bbox_dict}
    return None


def _find_any_source_tag(text: str | None) -> dict | None:
    """Pull the first `[copilot-source: ...]` tag out of a comments/note
    string, regardless of which DocumentReference it points at.

    Companion to `_parse_source_tag`, which filters by a known target
    document. This variant is used by the per-citation source lookup
    endpoint where we have a chart resource id and need to discover which
    document it was extracted from. Returns None when no tag is present;
    otherwise `{ref, bbox}` where `bbox` may be None on parse failure.
    """
    if not text:
        return None
    m = _SOURCE_TAG_RE.search(text)
    if m is None:
        return None
    ref = m.group("ref").strip()
    target = ref if "/" in ref else f"DocumentReference/{ref}"
    bbox_dict: dict | None = None
    bbox_raw = m.group("bbox")
    if bbox_raw:
        import json as _json  # noqa: PLC0415
        try:
            parsed = _json.loads(bbox_raw)
            if isinstance(parsed, dict):
                bbox_dict = parsed
        except (ValueError, TypeError):
            bbox_dict = None
    return {"ref": target, "bbox": bbox_dict}


async def get_resource_source_document(
    client: FhirClient,
    *,
    resource_type: str,
    resource_id: str,
    panel: frozenset[str] | None = None,
) -> SourcedResult:
    """Look up the DocumentReference + bbox a chart resource was extracted
    from, by parsing the `[copilot-source: ...]` tag off its note field.

    Returns `data.document_ref = None` when the resource exists but has no
    source tag (e.g. it was hand-entered, not extracted). The frontend
    treats that as "no source view available" and falls back to the
    chart-card scroll behavior.

    ACL gates on the resource's patient subject — same fail-closed shape
    as `get_document_bbox_manifest`.
    """
    from app.access_control import PatientAccessDenied  # noqa: PLC0415

    if resource_type not in {"AllergyIntolerance", "MedicationRequest", "Condition"}:
        # Not a writer-managed resource type — no source tag will ever
        # be present. Return an empty result rather than 404 so the
        # frontend can branch on `document_ref is None` uniformly.
        return {
            "data": {
                "resource_ref": f"{resource_type}/{resource_id}",
                "document_ref": None,
                "bbox": None,
            },
            "sources": [],
        }

    resource = await client.get(f"{resource_type}/{resource_id}")
    # AllergyIntolerance uses `patient`; MedicationRequest + Condition
    # use `subject`. Try both. (FHIR R4: AllergyIntolerance.patient is
    # required; the others use subject.reference.)
    pat_ref = (resource.get("patient") or {}).get("reference") or ""
    subj_ref = (resource.get("subject") or {}).get("reference") or ""
    ref_val = pat_ref or subj_ref
    patient_id: str | None = None
    if ref_val.startswith("Patient/"):
        patient_id = ref_val.removeprefix("Patient/")
    if panel is not None and (patient_id is None or patient_id not in panel):
        raise PatientAccessDenied(patient_id or "")

    tag = _find_any_source_tag(_collect_note_text(resource))
    if tag is None:
        return {
            "data": {
                "resource_ref": f"{resource_type}/{resource_id}",
                "document_ref": None,
                "bbox": None,
                "patient_id": patient_id,
            },
            "sources": [_ref(resource)],
        }
    return {
        "data": {
            "resource_ref": f"{resource_type}/{resource_id}",
            "document_ref": tag["ref"],
            "bbox": tag.get("bbox"),
            "patient_id": patient_id,
        },
        "sources": [_ref(resource), tag["ref"]],
    }


def _collect_note_text(resource: dict) -> str:
    """Concatenate every `note.text` (FHIR Annotation) on a resource into
    a single string. The writer stores its source tag on `note[0].text`
    today, but a defensive concatenation tolerates older or third-party
    rows that may have spread tags across multiple annotations."""
    notes = resource.get("note") or []
    parts: list[str] = []
    for n in notes:
        if isinstance(n, dict):
            t = n.get("text")
            if isinstance(t, str) and t:
                parts.append(t)
    return "\n".join(parts)


async def get_document_bbox_manifest(
    client: FhirClient,
    *,
    document_id: str,
    panel: frozenset[str] | None = None,
) -> SourcedResult:
    """Walk every resource derived from `DocumentReference/<document_id>`
    (allergies, medications, problems, lab observations) and pull the
    bbox metadata each one carries via the writer's `[copilot-source:
    ...; bbox=...]` comment tag.

    Used by the Supporting-Documents PDF-overlay UI (PRD W2 §5 — visual
    PDF bounding-box overlay). The agent itself does not call this; the
    return shape is keyed for the frontend, not for inline LLM ingestion.

    ACL gates on the DocumentReference's `subject.reference` Patient: a
    user whose panel doesn't include that patient gets `PatientAccessDenied`
    even if they somehow guessed a real document id.
    """
    from app.access_control import PatientAccessDenied  # noqa: PLC0415

    doc = await client.get(f"DocumentReference/{document_id}")

    subject_ref = (doc.get("subject") or {}).get("reference") or ""
    patient_id: str | None = None
    if subject_ref.startswith("Patient/"):
        patient_id = subject_ref.removeprefix("Patient/")
    if panel is not None and (patient_id is None or patient_id not in panel):
        raise PatientAccessDenied(patient_id or "")

    facts: list[dict] = []
    sources: list[str] = [_ref(doc)]

    if patient_id is None:
        # Document with no Patient subject — no derived facts to walk.
        return {
            "data": {
                "document_id": document_id,
                "patient_id": None,
                "facts": facts,
            },
            "sources": sources,
        }

    # Three resource lists own the writer's intake-form derivatives.
    # Lab-encounter results live under Observation -> Encounter and are
    # reached via the SOAP note's manifest; v0 covers the intake-form
    # path (allergies / medications / medical problems) which is the
    # primary surface for the demo. Lab bbox manifests come in v1.
    for resource_type, label_fn in (
        ("AllergyIntolerance", lambda r: _coded_display(r.get("code") or {})),
        ("MedicationRequest", lambda r: _coded_display(r.get("medicationCodeableConcept") or {})),
        ("Condition", lambda r: _coded_display(r.get("code") or {})),
    ):
        try:
            rows = await client.search(resource_type, {"patient": patient_id, "_count": 200})
        except Exception as e:  # noqa: BLE001 — partial manifest is better than 500
            log.warning(
                "bbox-manifest: %s search failed for patient=%s document=%s: %s",
                resource_type, patient_id, document_id, e,
            )
            continue
        for row in rows:
            tag = _parse_source_tag(_collect_note_text(row), document_id)
            if tag is None:
                continue
            row_id = row.get("id") or ""
            ref = f"{resource_type}/{row_id}" if row_id else None
            label = (label_fn(row) or "").strip() or resource_type
            facts.append({
                "resource_type": resource_type,
                "resource_id": row_id,
                "resource_ref": ref,
                "label": label,
                "bbox": tag.get("bbox"),
            })
            if ref:
                sources.append(ref)

    return {
        "data": {
            "document_id": document_id,
            "patient_id": patient_id,
            "facts": facts,
        },
        "sources": sources,
    }


async def get_document_pages(
    client: FhirClient,
    *,
    document_id: str,
    panel: frozenset[str] | None = None,
    max_pages: int = 30,
) -> SourcedResult:
    """Render a DocumentReference's attachments to PNG pages and return
    them with per-page pixel dimensions, for the Supporting-Documents
    inline PDF viewer.

    Companion to `get_document_content` (which is the agent-facing tool
    and bounds at 10 pages so token costs stay sane). The viewer can
    afford a higher cap; we still cap at 30 to keep the JSON response
    sub-megabyte for any pathological scan.
    """
    from app.access_control import PatientAccessDenied  # noqa: PLC0415
    from app.extraction.render import render_to_png_pages
    from PIL import Image  # noqa: PLC0415

    doc = await client.get(f"DocumentReference/{document_id}")

    subject_ref = (doc.get("subject") or {}).get("reference") or ""
    patient_id: str | None = None
    if subject_ref.startswith("Patient/"):
        patient_id = subject_ref.removeprefix("Patient/")
    if panel is not None and (patient_id is None or patient_id not in panel):
        raise PatientAccessDenied(patient_id or "")

    pages: list[dict] = []
    truncated = False

    for content_entry in doc.get("content") or []:
        att = content_entry.get("attachment") or {}
        content_type = att.get("contentType") or ""
        file_bytes = await _fetch_attachment_bytes(client, att)
        if file_bytes is None:
            continue
        try:
            pngs = render_to_png_pages(file_bytes, content_type)
        except ValueError:
            continue
        for png in pngs:
            if len(pages) >= max_pages:
                truncated = True
                break
            try:
                with Image.open(io.BytesIO(png)) as im:
                    w, h = im.size
            except Exception:  # noqa: BLE001
                w, h = 0, 0
            pages.append({
                "page": len(pages) + 1,
                "png_b64": _b64_png(png),
                "width_px": w,
                "height_px": h,
            })
        if truncated:
            break

    return {
        "data": {
            "document_id": document_id,
            "patient_id": patient_id,
            "page_count": len(pages),
            "pages_truncated": truncated,
            "pages": pages,
        },
        "sources": [_ref(doc)],
    }


async def get_med_changes_24h(
    client: FhirClient, *, patient_id: str, hours: int = 24,
) -> SourcedResult:
    """`MedicationRequest`s authored or modified for the patient in the
    last `hours` — new orders, dose changes, holds.

    OpenEMR's FHIR layer does NOT expose `authoredon` as a search param
    (only `patient` / `intent` / `status` / `_id` / `_lastUpdated` per
    FhirMedicationRequestService::loadSearchParameters); using
    `authoredon=ge…` would be silently ignored and the search would
    return every MedicationRequest. We use `_lastUpdated=ge…` instead —
    it's supported and semantically broader (captures modifications to
    existing orders too, which the doctor wants to see). A client-side
    filter on either `authoredOn` or `meta.lastUpdated` keeps results
    inside the window if the server's filter ever silently regresses.
    """
    cutoff_iso = _utc_cutoff_iso(hours)
    cutoff_dt = datetime.now(timezone.utc) - timedelta(hours=hours)
    meds = await client.search(
        "MedicationRequest",
        {"patient": patient_id, "_lastUpdated": f"ge{cutoff_iso}", "_count": 30},
    )
    items: list[dict] = []
    sources: list[str] = []
    for m in meds:
        # Defense-in-depth: skip rows that fall outside the window even if
        # the FHIR layer ignored our filter. Compare on the freshest of
        # `meta.lastUpdated` and `authoredOn`. `(m.get("meta") or {})`
        # protects against an explicit `"meta": null` which would otherwise
        # AttributeError on `.get(...)`.
        meta = m.get("meta") or {}
        if not _is_recent(_pick_latest(meta.get("lastUpdated"),
                                       m.get("authoredOn")),
                          cutoff_dt):
            continue
        sources.append(_ref(m))
        dosage = (m.get("dosageInstruction") or [{}])[0]
        items.append({
            "id": m["id"],
            "drug": _coded_display(m.get("medicationCodeableConcept", {}))
                or _narrative_text(m),
            "dose_text": dosage.get("text"),
            "status": m.get("status"),
            "intent": m.get("intent"),
            "authored_on": _clinical_iso(m.get("authoredOn")),
            "last_updated": _clinical_iso(meta.get("lastUpdated")),
        })
    items.sort(key=lambda x: x.get("last_updated") or x.get("authored_on") or "",
               reverse=True)
    return {
        "data": {
            "window_hours": hours,
            "cutoff_iso": cutoff_iso,
            "count": len(items),
            "medications": items,
        },
        "sources": sources,
    }


def _pick_latest(*iso_strs: str | None) -> str | None:
    """Return the lexicographically-greatest non-None ISO string, which is
    also the most recent timestamp (ISO-8601 sorts naturally)."""
    candidates = [s for s in iso_strs if isinstance(s, str) and s]
    return max(candidates) if candidates else None


def _is_recent(iso_str: str | None, cutoff_dt: datetime) -> bool:
    """True if `iso_str` parses as a datetime at or after `cutoff_dt`.
    Returns True for unparsable input (fail-open) so a malformed timestamp
    doesn't silently drop a real record — better to surface than hide.
    `cutoff_dt` must be tz-aware UTC; we coerce naive inputs to UTC."""
    if not iso_str:
        return True  # fail-open
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
    except ValueError:
        return True  # fail-open on unparsable
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt >= cutoff_dt
