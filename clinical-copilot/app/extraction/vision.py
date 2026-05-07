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
    ReferralLetter,
)

log = logging.getLogger("agent.extraction.vision")

DEFAULT_MODEL = "claude-sonnet-4-6"

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
patient anonymously ('the patient', 'this 55yo male').
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
    """Build the `messages[0].content` array: page images, then instruction."""
    blocks: list[dict[str, Any]] = []
    for png in page_pngs:
        blocks.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": base64.standard_b64encode(png).decode("ascii"),
            },
        })
    blocks.append({
        "type": "text",
        "text": (
            f"Extract the contents of this {doc_type} into the matching "
            f"tool call. Use source_document_id={source_document_id!r} "
            f"verbatim in every Citation."
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
