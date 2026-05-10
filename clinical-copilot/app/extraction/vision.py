"""Claude vision call that turns rendered document pages into a strict
typed `ExtractedDocument` (LabReport | IntakeForm).

Implementation choices documented inline so they are easy to audit:

- **Tool-use forced output.** Anthropic's tool-use lets us bind the
  Pydantic JSON schema as a tool definition and force the model to
  return matching JSON via `tool_choice`. This eliminates the
  parse-then-validate-then-pray pattern and turns extraction into a
  single typed call. Free-text "please return JSON" is brittle by
  comparison.

- **One tool per doc_type.** `record_lab_report` for lab PDFs,
  `record_intake_form` for intake forms. The tool schemas are derived
  directly from `LabReport.model_json_schema()` etc., so the model has
  no ambiguity about field shape and can never propose a third
  doc-type unilaterally.

- **`source_document_id` injected, not extracted.** The model does NOT
  decide what the source_document_id is; we set it from the writer's
  return value before the call and validate that every Citation in the
  returned payload uses it. Lets the schema's `min_length=1` field
  succeed even when the document body never mentions an ID.

- **Mockable transport.** The `client` arg is a typed `AsyncAnthropic`
  so unit tests can pass a fake client whose `messages.create`
  returns a canned response. No HTTP calls in unit tests.

- **Model defaults to Sonnet 4.6.** Same default as `app/agent/graph.py`.
  Plenty smart for typed extraction, cheaper than Opus per call.
"""

from __future__ import annotations

import base64
import json
import logging
from typing import Any

from anthropic import AsyncAnthropic
from anthropic.types import ToolUseBlock

from app.extraction.schemas import (
    DocumentType,
    ExtractedDocument,
    FaxPacket,
    IntakeForm,
    LabReport,
    PatientIdentity,
    ReferralLetter,
)

log = logging.getLogger("agent.extraction.vision")

DEFAULT_MODEL = "claude-sonnet-4-6"
# Demographics-only auto-fill on the create-patient flow uses Haiku for the
# preview: the doctor reviews and corrects the result on the form before
# submitting, and the full upload re-extracts with Sonnet over all pages.
# Haiku is roughly 3× faster than Sonnet on vision and dramatically cheaper.
DEMOGRAPHICS_MODEL = "claude-haiku-4-5"

_DEMOGRAPHICS_SYSTEM_PROMPT = """\
You are a precise clinical-document patient-identity extractor. Look at the \
supplied document image(s) and call `record_patient_identity` with whatever \
identity fields the document prints (given_name, family_name, date_of_birth, \
sex, address, phone). Skip any field the document does not state — never \
invent values.

Sex inference: when the document does NOT print an explicit Sex/Gender field \
but DOES carry an honorific next to the patient's name, infer sex from it \
(Mr.→male, Mrs./Ms./Miss→female, Mx.→other). Honorifics carrying no sex \
signal (Dr., Prof., Rev., Hon.) do NOT trigger inference; leave sex null. \
When neither an explicit field nor a sex-carrying honorific is present, \
leave sex null.

Citations: `source_citation` is OPTIONAL on PatientIdentity; this is a \
preview-only extraction that never gets persisted, so leave \
`source_citation` null and skip filling it. The downstream UI displays \
only the identity fields, not the citation.
"""

_SYSTEM_PROMPT = """\
You are a precise clinical-document extraction service. Your only job is \
to look at the supplied document image(s) and call the matching `record_*` \
tool with the structured contents you read.

Strict rules:
- Never invent fields not visible in the document. If something is not on \
the page, omit the optional field entirely (never use empty strings or \
placeholder values).
- Every clinical fact must carry a Citation pointing back to where in the \
document it appeared.
- Use the supplied `source_document_id` verbatim in every Citation. Do \
not invent or alter it.
- For lab values: extract every visible result row. `value` is numeric \
when the result is a number, otherwise the literal qualitative string \
('positive', 'detected', etc.). `unit` is the literal printed unit \
(empty string for unitless results). Use the exact reference range and \
abnormal flag as printed.
- For abnormal flags, normalize to one of {H, L, N, C}. Map \
'Critical High' or 'Critical Low' to 'C'.
- For intake forms: extract demographics, chief concern, current \
medications, allergies, family history. Empty list-fields are fine \
(intake forms with NKDA, no current meds, no recorded family history).
- For referral letters: extract the referring physician, the reason for \
referral (verbatim or close to it), HPI if printed, the requested action \
if printed, plus the structured chart slice the letter exposes — past \
medical history (with ICD-10 codes when printed), current medications, \
allergies, and any pertinent lab values cited inline. Empty list-fields \
are fine (a referral with NKDA, or a referral that omits PMH because it \
folds it into the HPI paragraph). When the letter prints a patient \
header (name + DOB on letterhead, or a 'Re: <patient name>, DOB ...' \
salutation), populate `patient_identity` with whatever fields are \
visible (given_name, family_name, date_of_birth, sex, address, phone). \
Omit `patient_identity` entirely when the letter only references the \
patient anonymously ('the patient', 'this 55yo male'). For the \
referring physician, also populate `specialty`, `phone`, and `address` \
from the letterhead or signature block when printed (e.g. 'Helen Park, \
MD — Cardiology', '(650) 555-0100', '123 Main St, Mountain View, CA \
94040'). Join multi-line addresses into a single comma-separated line. \
Omit each of these three fields when the letter does not print it.
- For fax packets: a single fax bundles a transmittal cover sheet plus \
1+ clinical pages (referral request, patient face sheet, lab report). \
Extract the cover-sheet metadata (date, sender, recipient, urgency, the \
free-text MESSAGE paragraph), the reason-for-consultation from the \
referral page, the receiving specialty, and the structured chart slice \
from the face sheet (active problems with ICD-10 when printed, current \
medications, allergies). When the patient face sheet prints the \
patient's name / DOB / sex / address / phone, populate \
`patient_identity` from those values. Omit `patient_identity` entirely \
on cover-only transmittals that carry no face sheet. When a Laboratory \
Report page is included, extract every result row into `lab_results` — \
these will be persisted as a real lab encounter (the fax IS the \
source, not a quotation). Empty list fields are fine for cover-only or \
referral-only faxes.
- Citations: `page_or_section` is human-readable like 'page 1' or \
'Allergies section'. `field_or_chunk_id` is a snake_case slug like \
'results_table.hba1c' or 'medications.0' that uniquely identifies the \
field within the document. `quote_or_value` is the literal text or \
value as printed (the value, not your interpretation).
- Citation `bbox` (visual overlay coordinates): for every clinical fact \
extracted from a visual document (lab PDFs, intake forms, referral \
letters, fax packets), populate `bbox` with the rectangle that surrounds \
the cited value on the rendered page. Coordinates are in PIXELS of the \
page image you are looking at, with the origin (0,0) at the TOP-LEFT \
corner of that page. `x`,`y` is the upper-left corner of the box; \
`width`,`height` are positive pixel dimensions. `page` is the 1-indexed \
page number within the document. Do NOT use PDF points, do NOT \
normalize to 0-1, do NOT use a 1000x1000 grid — use the literal pixel \
coordinates of the rendered page image. Make the box tight around just \
the cited value (the number itself, the printed allergy entry, etc.), \
not the whole row or section. Omit `bbox` entirely for HL7 messages, \
spreadsheets, or any source without visual page coordinates.
- Sex inference from honorifics: when the document does NOT print an \
explicit Sex/Gender field but DOES carry a courtesy title next to the \
patient's name, infer `sex` from the honorific: Mr.→male, \
Mrs./Ms./Miss→female, Mx.→other. Honorifics carrying no sex signal \
(Dr., Prof., Rev., Hon.) do NOT trigger inference; leave `sex` null. \
When neither an explicit field nor a sex-carrying honorific is \
present, leave `sex` null and let the doctor confirm post-create.
"""


def _tool_for_lab_report() -> dict[str, Any]:
    """Tool definition that forces a LabReport-shaped tool call."""
    return {
        "name": "record_lab_report",
        "description": (
            "Record the structured contents of a clinical lab report "
            "(blood panel, urinalysis, etc.) as a LabReport object."
        ),
        "input_schema": LabReport.model_json_schema(),
    }


def _tool_for_intake_form() -> dict[str, Any]:
    """Tool definition that forces an IntakeForm-shaped tool call."""
    return {
        "name": "record_intake_form",
        "description": (
            "Record the structured contents of a patient intake form "
            "(demographics + history + current meds + allergies) as an "
            "IntakeForm object."
        ),
        "input_schema": IntakeForm.model_json_schema(),
    }


def _tool_for_referral_letter() -> dict[str, Any]:
    """Tool definition that forces a ReferralLetter-shaped tool call."""
    return {
        "name": "record_referral_letter",
        "description": (
            "Record the structured contents of a clinical referral letter "
            "(referring physician + reason for referral + HPI + requested "
            "action + the patient's PMH / current meds / allergies / "
            "pertinent labs as cited in the letter) as a ReferralLetter "
            "object."
        ),
        "input_schema": ReferralLetter.model_json_schema(),
    }


def _tool_for_fax_packet() -> dict[str, Any]:
    """Tool definition that forces a FaxPacket-shaped tool call."""
    return {
        "name": "record_fax_packet",
        "description": (
            "Record the structured contents of a multi-page clinical "
            "fax packet (transmittal cover + referral request + patient "
            "face sheet + optional lab report). Pull cover-sheet "
            "metadata, reason for consultation, the active "
            "problems / medications / allergies from the face sheet, "
            "and any inline lab results."
        ),
        "input_schema": FaxPacket.model_json_schema(),
    }


_TOOL_BUILDERS = {
    "lab_pdf": (_tool_for_lab_report, "record_lab_report", LabReport),
    "intake_form": (_tool_for_intake_form, "record_intake_form", IntakeForm),
    "referral_letter": (_tool_for_referral_letter, "record_referral_letter", ReferralLetter),
    "fax_packet": (_tool_for_fax_packet, "record_fax_packet", FaxPacket),
}


def _user_content_blocks(
    page_pngs: list[bytes],
    *,
    doc_type: DocumentType,
    source_document_id: str,
) -> list[dict[str, Any]]:
    """Build the `messages[0].content` array: page images, then instruction.

    For each page we measure the actual PNG dimensions and tell the
    model exactly how big the image it's looking at is. Claude has been
    observed to fall back to ~612x792 PDF-point coordinates for bbox
    values when not given explicit dimensions; calling out the literal
    pixel size per page anchors the coordinate space to the rendered
    image and keeps overlays from landing 50% of a page off.
    """
    from PIL import Image  # noqa: PLC0415
    import io as _io  # noqa: PLC0415

    blocks: list[dict[str, Any]] = []
    page_dims: list[tuple[int, int]] = []
    for png in page_pngs:
        try:
            with Image.open(_io.BytesIO(png)) as im:
                page_dims.append((int(im.width), int(im.height)))
        except Exception:  # noqa: BLE001
            page_dims.append((0, 0))
        blocks.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": base64.standard_b64encode(png).decode("ascii"),
            },
        })
    dims_str = "; ".join(
        f"page {i + 1}: {w} x {h} px"
        for i, (w, h) in enumerate(page_dims)
    )
    blocks.append({
        "type": "text",
        "text": (
            f"Extract the contents of this {doc_type} into the matching "
            f"tool call. Use source_document_id={source_document_id!r} "
            f"verbatim in every Citation.\n\n"
            f"Page dimensions (for bbox coordinates): {dims_str}.\n"
            f"Every Citation's bbox.x and bbox.width must be in [0, page_width_px]; "
            f"every bbox.y and bbox.height must be in [0, page_height_px] "
            f"of the SPECIFIC page named by bbox.page (1-indexed). "
            f"Do NOT use PDF points (612 x 792); use the literal pixel "
            f"coordinates of the page image you are looking at."
        ),
    })
    return blocks


class ExtractionError(RuntimeError):
    """Raised when Claude returns no tool call, the wrong tool call, or
    a tool call whose arguments fail Pydantic validation."""


async def extract_via_claude(
    *,
    page_pngs: list[bytes],
    doc_type: DocumentType,
    source_document_id: str,
    client: AsyncAnthropic | None = None,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 8192,
) -> ExtractedDocument:
    """Send the rendered pages to Claude and return a validated
    ExtractedDocument.

    Raises ExtractionError if the model declines to call the tool, calls
    the wrong tool, or returns arguments that fail schema validation.
    """
    if not page_pngs:
        raise ExtractionError("extract_via_claude: no pages to send")
    try:
        builder, tool_name, model_cls = _TOOL_BUILDERS[doc_type]
    except KeyError as exc:
        raise ExtractionError(
            f"extract_via_claude: unsupported doc_type {doc_type!r}"
        ) from exc
    tool = builder()

    anthropic = client if client is not None else AsyncAnthropic()
    response = await anthropic.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=_SYSTEM_PROMPT,
        tools=[tool],
        tool_choice={"type": "tool", "name": tool_name},
        messages=[{
            "role": "user",
            "content": _user_content_blocks(
                page_pngs,
                doc_type=doc_type,
                source_document_id=source_document_id,
            ),
        }],
    )

    # tool_choice forces a tool_use block; pull the first matching one.
    tool_block: ToolUseBlock | None = None
    for block in response.content:
        if isinstance(block, ToolUseBlock) and block.name == tool_name:
            tool_block = block
            break
    if tool_block is None:
        raise ExtractionError(
            f"Claude returned no {tool_name} tool call. "
            f"stop_reason={response.stop_reason!r} "
            f"content_types={[type(b).__name__ for b in response.content]}"
        )

    raw = tool_block.input
    if not isinstance(raw, dict):
        raise ExtractionError(
            f"{tool_name} tool input was not a dict: type={type(raw).__name__}"
        )

    try:
        extracted = model_cls.model_validate(raw)
    except Exception as exc:
        raise ExtractionError(
            f"{tool_name} tool input failed schema validation: {exc}\n"
            f"raw={json.dumps(raw)[:1000]}"
        ) from exc

    log.info(
        "extracted via claude doc_type=%s source_id=%s pages=%d",
        doc_type, source_document_id, len(page_pngs),
    )
    return extracted  # type: ignore[return-value]


async def extract_patient_identity(
    *,
    page_pngs: list[bytes],
    client: AsyncAnthropic | None = None,
    model: str = DEMOGRAPHICS_MODEL,
    max_tokens: int = 1024,
) -> PatientIdentity | None:
    """Fast Haiku-based extraction of just patient identity for auto-fill.

    Used by `/api/upload/extract-demographics` on vision-bound doc types
    (intake_form, fax_packet, referral_letter) to preview demographics on
    the create-patient form without paying for a full Sonnet extraction.
    The full upload step still re-extracts every page with Sonnet — this
    function is preview-only and never persisted.

    Returns None if the model declines to call the tool (some docs carry
    no identity at all). Raises ExtractionError on schema validation
    failure so the caller can fall back to manual entry.
    """
    if not page_pngs:
        raise ExtractionError("extract_patient_identity: no pages to send")
    tool = {
        "name": "record_patient_identity",
        "description": (
            "Record the patient identity (name, DOB, sex, address, phone) "
            "from the document header / face sheet. Use null for fields "
            "the document does not state."
        ),
        "input_schema": PatientIdentity.model_json_schema(),
    }

    anthropic = client if client is not None else AsyncAnthropic()
    user_blocks: list[dict[str, Any]] = []
    for png in page_pngs:
        user_blocks.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": base64.standard_b64encode(png).decode("ascii"),
            },
        })
    user_blocks.append({
        "type": "text",
        "text": (
            "Extract the patient identity into the record_patient_identity "
            "tool. Leave source_citation null — this is a preview-only "
            "extraction."
        ),
    })

    response = await anthropic.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=_DEMOGRAPHICS_SYSTEM_PROMPT,
        tools=[tool],
        tool_choice={"type": "tool", "name": "record_patient_identity"},
        messages=[{"role": "user", "content": user_blocks}],
    )

    tool_block: ToolUseBlock | None = None
    for block in response.content:
        if isinstance(block, ToolUseBlock) and block.name == "record_patient_identity":
            tool_block = block
            break
    if tool_block is None:
        log.info("extract_patient_identity: no tool call returned (no identity in doc)")
        return None

    raw = tool_block.input
    if not isinstance(raw, dict):
        raise ExtractionError(
            f"record_patient_identity tool input was not a dict: "
            f"type={type(raw).__name__}"
        )
    try:
        return PatientIdentity.model_validate(raw)
    except Exception as exc:
        raise ExtractionError(
            f"record_patient_identity input failed schema validation: {exc}\n"
            f"raw={json.dumps(raw)[:500]}"
        ) from exc
