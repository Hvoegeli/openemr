"""HL7 v2 message parser — pure structured-text path, no LLM call.

HL7 v2 messages (the .hl7 wire format) are pipe-delimited segments with
encoding characters declared in MSH-2 (`^~\\&` is the de-facto standard).
Unlike the lab-PDF / intake-form path, there is no vision call: we parse
the segments deterministically and map them to a typed `Hl7Message`.

Two message types in scope for Week 2:

- **ADT-A08** (patient information update). We extract EVN-9 (the clinical
  reason note) and any AL1 segments (allergies). PID demographics are
  intentionally NOT mirrored back into the schema — the patient already
  exists in OpenEMR (we used their UUID to attach this message).

- **ORU-R01** (observation results). We group the message into one or
  more OBR panels, each carrying its own OBX rows + an optional NTE
  interpretation note. One panel = one Encounter on persistence —
  collapsing OBR groups loses the panel-level LOINC and breaks the link
  between an interpretive note and the panel it interprets.

Citations are segment-indexed, never bbox'd. `page_or_section` strings
look like `"OBR[2] / OBX[1]"` or `"AL1[2]"`; `field_or_chunk_id` is the
lowercase snake_case equivalent (`"obr_2.obx_1"`). The `bbox` field is
always None — HL7 has no spatial layout.

The parser is hand-rolled (no external dependency). Only ADT-A08 and
ORU-R01 are recognized; any other MSH-9 raises `Hl7ParseError`.
"""

from __future__ import annotations

import logging
from datetime import date, datetime

from app.extraction.schemas import (
    AbnormalFlag,
    Citation,
    Hl7Allergy,
    Hl7AllergySeverity,
    Hl7AllergyTypeCode,
    Hl7LabPanel,
    Hl7Message,
    LabResult,
)

log = logging.getLogger("agent.extraction.hl7")


class Hl7ParseError(ValueError):
    """Raised when an HL7 v2 message can't be parsed: missing MSH, an
    unsupported MSH-9, malformed segments. Subclass of ValueError so
    upstream FastAPI handlers can map it to a 400 the same way they
    handle other render-side errors."""


# ──────────────────────────────────────────────────────────────────────────
# Low-level grammar helpers
# ──────────────────────────────────────────────────────────────────────────

# HL7 v2.x segment separator is ASCII CR (0x0D). Some systems normalize to
# LF or CRLF on the wire; we accept all three.
_SEG_SEPARATORS = ("\r\n", "\r", "\n")


def _decode(raw: bytes) -> str:
    """Decode HL7 bytes. The MSH-18 character-set field can declare the
    charset; we don't read it because real-world v2.x messages are almost
    always ASCII-compatible (UTF-8 is common; ISO-8859-1 is the legacy
    fallback). Try UTF-8 strict first, then UTF-8 with replacement so
    a stray byte doesn't kill an otherwise-parseable message."""
    if not raw:
        raise Hl7ParseError("HL7 message is empty")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        log.info("HL7 decode: falling back to utf-8 with replacement")
        return raw.decode("utf-8", errors="replace")


def _segment_split(text: str) -> list[list[str]]:
    """Split the raw text into segments, then each segment into fields.

    Returns a list of `[segname, field1, field2, ...]`. MSH is special:
    its first field separator IS MSH-1, so for MSH the returned list is
    `["MSH", "^~\\&", msh3, msh4, ...]` — i.e. MSH-N for N>=2 is at index
    N-1 in the list. For non-MSH segments, segment-N is at index N (the
    segment name is index 0, segments[1] is field 1, etc.).
    """
    norm = text.replace("\r\n", "\n").replace("\r", "\n")
    raw_segs = [s for s in norm.split("\n") if s.strip()]
    return [seg.split("|") for seg in raw_segs]


def _component(field: str, idx: int) -> str:
    """Return the 1-indexed component (split on `^`) from an HL7 field.

    `_component("57698-3^Lipid panel^LN", 2)` -> `"Lipid panel"`.
    Returns `""` when the component is missing."""
    if not field:
        return ""
    parts = field.split("^")
    if 1 <= idx <= len(parts):
        return parts[idx - 1]
    return ""


def _parse_hl7_date(value: str) -> date | None:
    """Parse the date prefix of an HL7 timestamp (YYYYMMDD[HHMM[SS]]).

    Returns `None` for empty / malformed input rather than raising — many
    OBR/OBX rows have valid lab values but inconsistent date encodings,
    and a single bad date shouldn't kill the whole message."""
    if not value or len(value) < 8:
        return None
    try:
        return date(int(value[0:4]), int(value[4:6]), int(value[6:8]))
    except (ValueError, IndexError):
        return None


def _parse_hl7_datetime(value: str) -> datetime | None:
    """Parse a full HL7 timestamp (YYYYMMDDHHMMSS). Returns None on
    malformed input. Used for MSH-7 only — clinical-fact dates use the
    date-only form."""
    if not value or len(value) < 8:
        return None
    try:
        y, m, d = int(value[0:4]), int(value[4:6]), int(value[6:8])
        hh = int(value[8:10]) if len(value) >= 10 else 0
        mm = int(value[10:12]) if len(value) >= 12 else 0
        ss = int(value[12:14]) if len(value) >= 14 else 0
        return datetime(y, m, d, hh, mm, ss)
    except (ValueError, IndexError):
        return None


def _msh_field(msh: list[str], n: int) -> str:
    """Return MSH-N (1-indexed by HL7's spec). MSH-1 is the field
    separator itself ("|") which we synthesize. For N >= 2, the value
    lives at `msh[n - 1]` because the segment-name slot eats one index."""
    if n == 1:
        return "|"
    if n - 1 >= len(msh):
        return ""
    return msh[n - 1]


def _seg_field(seg: list[str], n: int) -> str:
    """Return field N (1-indexed) from a non-MSH segment. `seg[0]` is
    the segment name; field 1 is at `seg[1]`."""
    if n >= len(seg):
        return ""
    return seg[n]


def _normalize_message_type(msh9: str) -> str:
    """MSH-9 is `MSGTYPE^TRIGEVT^STRUCT` (e.g. `ADT^A08^ADT_A01`).

    We collapse to the first two components — `ADT^A08` / `ORU^R01` —
    because the third component (structure name) is informational and
    HL7 v2.5 messages with the same MSGTYPE+TRIGEVT can ship under
    different structure names depending on the dialect."""
    parts = (msh9 or "").split("^")
    msgtype = parts[0] if len(parts) > 0 else ""
    trigger = parts[1] if len(parts) > 1 else ""
    if msgtype and trigger:
        return f"{msgtype}^{trigger}"
    return msgtype or msh9 or ""


# ──────────────────────────────────────────────────────────────────────────
# Citation builders — every clinical fact gets one
# ──────────────────────────────────────────────────────────────────────────

def _hl7_citation(
    *,
    source_document_id: str,
    page_or_section: str,
    field_or_chunk_id: str,
    quote_or_value: str,
) -> Citation:
    """Build a Citation rooted at an HL7 segment/field. `bbox=None`
    universally — HL7 has no spatial layout."""
    return Citation(
        source_type="hl7_message",
        source_id=source_document_id,
        page_or_section=page_or_section,
        field_or_chunk_id=field_or_chunk_id,
        quote_or_value=quote_or_value,
        bbox=None,
    )


# ──────────────────────────────────────────────────────────────────────────
# AL1 (allergy) parsing
# ──────────────────────────────────────────────────────────────────────────

def _parse_al1(seg: list[str], idx: int, source_document_id: str) -> Hl7Allergy | None:
    """Extract one Hl7Allergy from an AL1 segment. `idx` is the 1-based
    position among AL1 segments in the message (used for the citation
    locator). Returns None if AL1-3 (the allergen) is missing — without
    a substance name there's nothing to cite."""
    type_code_raw = _seg_field(seg, 2).strip()
    allergen_field = _seg_field(seg, 3)
    severity_raw = _seg_field(seg, 4).strip()
    reaction_raw = _seg_field(seg, 5).strip()

    substance = _component(allergen_field, 1).strip()
    if not substance:
        return None

    # Validate the type/severity codes against the schema's accepted set.
    # An unknown code shows up as `None` rather than failing the whole
    # message — better to log a benign drop than reject an otherwise
    # parseable allergy.
    type_code: Hl7AllergyTypeCode | None = None
    if type_code_raw in {"DA", "FA", "MA", "EA", "MC", "OT", "AA", "PA"}:
        type_code = type_code_raw  # type: ignore[assignment]
    elif type_code_raw:
        log.info("AL1[%d]: unknown type code %r (skipping)", idx, type_code_raw)

    severity: Hl7AllergySeverity | None = None
    if severity_raw in {"SV", "MO", "MI"}:
        severity = severity_raw  # type: ignore[assignment]
    elif severity_raw:
        log.info("AL1[%d]: unknown severity code %r (skipping)", idx, severity_raw)

    return Hl7Allergy(
        substance=substance,
        type_code=type_code,
        severity=severity,
        reaction=reaction_raw or None,
        source_citation=_hl7_citation(
            source_document_id=source_document_id,
            page_or_section=f"AL1[{idx}]",
            field_or_chunk_id=f"al1_{idx}.allergen",
            quote_or_value=substance,
        ),
    )


# ──────────────────────────────────────────────────────────────────────────
# OBR/OBX (lab panel) parsing
# ──────────────────────────────────────────────────────────────────────────

def _parse_obx_to_lab_result(
    seg: list[str],
    *,
    obr_idx: int,
    obx_idx: int,
    fallback_collection_date: date | None,
    source_document_id: str,
) -> LabResult | None:
    """Map one OBX row to a LabResult. Returns None when the row has no
    usable value (OBX-5 blank, e.g. a textual placeholder row)."""
    test_id_field = _seg_field(seg, 3)
    value_raw = _seg_field(seg, 5).strip()
    units = _seg_field(seg, 6).strip()
    ref_range = _seg_field(seg, 7).strip()
    flag_raw = _seg_field(seg, 8).strip()
    obs_dt_raw = _seg_field(seg, 14).strip()

    if not value_raw:
        return None

    test_loinc = _component(test_id_field, 1).strip()
    test_long_name = _component(test_id_field, 2).strip()
    test_name = test_long_name or test_loinc
    if not test_name:
        return None

    # OBX-5 is sometimes numeric, sometimes a qualitative string
    # ("positive", "detected"). Try numeric first; fall through to str.
    try:
        value: float | str = float(value_raw)
    except ValueError:
        value = value_raw

    abnormal_flag: AbnormalFlag | None = None
    if flag_raw in {"H", "L", "N", "C", "HH", "LL"}:
        abnormal_flag = flag_raw  # type: ignore[assignment]
    elif flag_raw:
        log.info(
            "OBR[%d]/OBX[%d]: unknown abnormal flag %r (dropping)",
            obr_idx, obx_idx, flag_raw,
        )

    collection_date = _parse_hl7_date(obs_dt_raw) or fallback_collection_date
    if collection_date is None:
        log.info(
            "OBR[%d]/OBX[%d]: no usable date (OBX-14=%r), skipping row",
            obr_idx, obx_idx, obs_dt_raw,
        )
        return None

    return LabResult(
        test_name=test_name,
        value=value,
        unit=units,
        reference_range=ref_range or None,
        collection_date=collection_date,
        abnormal_flag=abnormal_flag,
        source_citation=_hl7_citation(
            source_document_id=source_document_id,
            page_or_section=f"OBR[{obr_idx}] / OBX[{obx_idx}]",
            field_or_chunk_id=f"obr_{obr_idx}.obx_{obx_idx}",
            quote_or_value=value_raw,
        ),
    )


def _parse_oru_panels(
    segments: list[list[str]],
    source_document_id: str,
) -> list[Hl7LabPanel]:
    """Walk an ORU-R01 message in segment order and build one
    Hl7LabPanel per OBR group.

    Grouping rule: the panel boundary is the OBR segment. Every OBX
    seen until the next OBR (or end of message) belongs to that panel.
    NTE segments attach to the panel they follow. ORC and other
    administrative segments are ignored (they don't carry clinical
    facts we surface)."""
    panels: list[Hl7LabPanel] = []
    obr_idx = 0
    obx_idx = 0
    current_panel: dict | None = None
    current_obr_collection: date | None = None

    def flush() -> None:
        if current_panel is None:
            return
        panels.append(Hl7LabPanel(**current_panel))

    for seg in segments:
        if not seg:
            continue
        name = seg[0]
        if name == "OBR":
            flush()
            obr_idx += 1
            obx_idx = 0
            obr4 = _seg_field(seg, 4)
            obr7 = _seg_field(seg, 7).strip()
            current_obr_collection = _parse_hl7_date(obr7)
            current_panel = {
                "panel_loinc": _component(obr4, 1).strip() or None,
                "panel_name": _component(obr4, 2).strip() or None,
                "collection_date": current_obr_collection,
                "results": [],
                "notes": None,
            }
        elif name == "OBX" and current_panel is not None:
            obx_idx += 1
            row = _parse_obx_to_lab_result(
                seg,
                obr_idx=obr_idx,
                obx_idx=obx_idx,
                fallback_collection_date=current_obr_collection,
                source_document_id=source_document_id,
            )
            if row is not None:
                current_panel["results"].append(row)
        elif name == "NTE" and current_panel is not None:
            note = _seg_field(seg, 3).strip()
            if note:
                # If a panel carries multiple NTE lines, concatenate
                # with newlines so the chart-row stays readable.
                if current_panel["notes"]:
                    current_panel["notes"] = current_panel["notes"] + "\n" + note
                else:
                    current_panel["notes"] = note
        # ORC and other segments fall through

    flush()
    return panels


# ──────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────

def parse_hl7_message(
    raw_bytes: bytes,
    *,
    source_document_id: str,
) -> Hl7Message:
    """Parse a .hl7 file's bytes into a typed `Hl7Message`.

    Args:
        raw_bytes: The raw bytes of the .hl7 file (as uploaded).
        source_document_id: The DocumentReference/{uuid} the source
            file has been persisted under. Baked into every Citation in
            the returned message.

    Returns:
        A fully-validated Hl7Message. Raises Hl7ParseError if the
        message is missing MSH, has an unsupported MSH-9, or fails the
        Hl7Message schema check.
    """
    text = _decode(raw_bytes)
    segments = _segment_split(text)
    if not segments:
        raise Hl7ParseError("HL7 message contains no segments")

    msh = next((s for s in segments if s and s[0] == "MSH"), None)
    if msh is None:
        raise Hl7ParseError("HL7 message has no MSH segment")

    msh9_raw = _msh_field(msh, 9)
    message_type = _normalize_message_type(msh9_raw)
    if message_type not in ("ADT^A08", "ORU^R01"):
        raise Hl7ParseError(
            f"unsupported HL7 message type {msh9_raw!r}; "
            "expected ADT^A08 or ORU^R01"
        )

    sending_application = _msh_field(msh, 3).strip() or None
    sending_facility = _msh_field(msh, 4).strip() or None
    timestamp = _parse_hl7_datetime(_msh_field(msh, 7).strip())
    message_control_id = _msh_field(msh, 10).strip() or None

    pid = next((s for s in segments if s and s[0] == "PID"), None)
    patient_mrn: str | None = None
    if pid is not None:
        # PID-3 is the patient identifier list; first identifier's
        # first component is the MRN. Some implementations stuff the
        # MRN into PID-2 instead, but PID-3 is the v2.5 canonical slot.
        pid3 = _seg_field(pid, 3)
        if pid3:
            patient_mrn = _component(pid3, 1).strip() or None

    if message_type == "ADT^A08":
        evn = next((s for s in segments if s and s[0] == "EVN"), None)
        event_reason: str | None = None
        if evn is not None:
            # The clinical reason note is non-standard in HL7 v2.5.1 — the
            # canonical EVN segment maxes out at EVN-7 (Event Facility) and
            # has no dedicated free-text reason field. Real-world systems
            # typically piggyback the reason on EVN-6 (Event Occurred,
            # technically a TS field but routinely repurposed for free
            # text on registration updates). EVN-7 and EVN-9 are accepted
            # as defensive fallbacks for senders that put the text there.
            evn6 = _seg_field(evn, 6).strip()
            evn7 = _seg_field(evn, 7).strip()
            evn9 = _seg_field(evn, 9).strip()
            event_reason = evn6 or evn7 or evn9 or None
        allergies: list[Hl7Allergy] = []
        al1_idx = 0
        for seg in segments:
            if seg and seg[0] == "AL1":
                al1_idx += 1
                allergy = _parse_al1(seg, al1_idx, source_document_id)
                if allergy is not None:
                    allergies.append(allergy)
        return Hl7Message(
            message_type="ADT^A08",
            sending_application=sending_application,
            sending_facility=sending_facility,
            message_control_id=message_control_id,
            timestamp=timestamp,
            patient_mrn=patient_mrn,
            event_reason=event_reason,
            allergies=allergies,
            lab_panels=[],
            source_document_id=source_document_id,
        )

    # ORU^R01
    panels = _parse_oru_panels(segments, source_document_id)
    return Hl7Message(
        message_type="ORU^R01",
        sending_application=sending_application,
        sending_facility=sending_facility,
        message_control_id=message_control_id,
        timestamp=timestamp,
        patient_mrn=patient_mrn,
        event_reason=None,
        allergies=[],
        lab_panels=panels,
        source_document_id=source_document_id,
    )


__all__ = [
    "Hl7ParseError",
    "parse_hl7_message",
]
