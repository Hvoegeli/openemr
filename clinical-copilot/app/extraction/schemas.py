"""Strict Pydantic schemas for extracted clinical documents.

Two top-level document shapes — `LabReport` (a lab PDF's worth of results)
and `IntakeForm` (one filled intake form). Both are returned by
`attach_and_extract` and round-tripped into OpenEMR as a `DocumentReference`
(the source document) plus the appropriate FHIR resources (`Observation` for
labs; `Patient` / `Condition` / `AllergyIntolerance` / `MedicationStatement`
for intake fields).

Every clinical fact in either shape carries a `Citation` — a structured
extension of the Week 1 source-attribution concept (Week 1 used inline
`[ResourceType/ID]` regex strings; Week 2 needs richer metadata so the UI
can locate values within unstructured documents). The citation shape is the
one mandated by the Week 2 PRD §5:
`{source_type, source_id, page_or_section, field_or_chunk_id, quote_or_value}`,
plus an optional `bbox` for the visual PDF-overlay UI.

All models use `extra="forbid"` so the VLM cannot smuggle hallucinated fields
past the schema. All required string fields require non-empty values via
`min_length=1` — an empty citation is structurally indistinguishable from a
missing one and would silently pass `citation_present` checks downstream.
Required fields are required; optional fields are explicit.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field


# ──────────────────────────────────────────────────────────────────────────
# Shared primitives
# ──────────────────────────────────────────────────────────────────────────

SourceType = Literal["lab_pdf", "intake_form", "referral_letter", "hl7_message", "fax_packet", "workbook", "guideline", "fhir_resource"]
"""Where a citation points. Spans both the document-extraction path
(lab_pdf, intake_form, referral_letter, hl7_message, fax_packet,
workbook), the evidence-retrieval path (guideline), and the existing
chart-summarizer path (fhir_resource)."""


DocumentType = Literal["lab_pdf", "intake_form", "referral_letter", "hl7_message", "fax_packet", "workbook"]
"""Doc types the Phase 2 extractor + writer support. Strict subset of
SourceType (the citation-source vocabulary is broader than the
extractable-doc vocabulary — guideline / fhir_resource citations exist
without ever being uploaded as documents)."""


DOC_TYPE_LABELS: dict[str, str] = {
    "lab_pdf":         "Lab Report (PDF)",
    "intake_form":     "Patient Intake Form",
    "referral_letter": "Referral Letter",
    "hl7_message":     "HL7 v2 Message",
    "fax_packet":      "Fax Packet",
    "workbook":        "Clinical Workbook",
}
"""Human-readable labels for each DocumentType, used by the upload-form
dropdown. Single source of truth — adding a new DocumentType means adding
one row here so the UI dropdown picks it up automatically."""


class BoundingBox(BaseModel):
    """Where on the source page a cited value lives.

    Coordinates are in the source document's native unit space (PDF points
    for PDFs). The frontend's PDF.js overlay layer transforms these into
    screen pixels at render time.
    """

    model_config = ConfigDict(extra="forbid")

    page: int = Field(ge=1, description="1-indexed page number within the source document")
    x: float = Field(ge=0)
    y: float = Field(ge=0)
    width: float = Field(gt=0)
    height: float = Field(gt=0)


class Citation(BaseModel):
    """The mandated Week 2 citation shape, used for every clinical claim.

    Per Week 2 PRD §5: `{source_type, source_id, page_or_section,
    field_or_chunk_id, quote_or_value}`. The `bbox` field is an extension
    populated when the source supports visual overlay (lab PDFs, intake
    forms) and omitted when it does not (guideline text, FHIR resources).
    """

    model_config = ConfigDict(extra="forbid")

    source_type: SourceType
    source_id: str = Field(min_length=1, description="Stable ID for the source — DocumentReference ID, guideline slug, FHIR resource ID")
    page_or_section: str = Field(min_length=1, description="Human-readable locator: 'page 1', 'Recommendation Statement', etc.")
    field_or_chunk_id: str = Field(min_length=1, description="Machine-readable locator within the source: form field name, RAG chunk ID, etc.")
    quote_or_value: str = Field(min_length=1, description="The literal text or value being cited, for verifiability")
    bbox: BoundingBox | None = None


# ──────────────────────────────────────────────────────────────────────────
# Lab PDF schema
# ──────────────────────────────────────────────────────────────────────────

AbnormalFlag = Literal["H", "L", "N", "C", "HH", "LL"]
"""Abnormal-flag values: H = high, L = low, N = normal, C = critical
(direction-collapsed), HH = critical-high, LL = critical-low. Lab PDFs
typically use H/L/N/C; HL7 v2 ORU-R01 messages use the fuller
H/L/N/HH/LL set. Both vocabularies are accepted so the source-fidelity
of the original message is preserved on round-trip. Code that switches
on criticality should test `flag in ("C", "HH", "LL")`. Optional —
many sources omit the flag and only print the reference range."""


class LabResult(BaseModel):
    """A single line on a lab PDF — one test result with its reference context."""

    model_config = ConfigDict(extra="forbid")

    test_name: str = Field(min_length=1, description="LOINC long name or vendor label as printed (e.g. 'Hemoglobin A1c')")
    value: float | str = Field(description="Numeric where applicable; string for qualitative ('positive', 'detected', 'N/A')")
    unit: str = Field(description="As printed; empty string for unitless values")
    reference_range: str | None = Field(default=None, description="As printed (e.g. '4.0-5.6 %')")
    collection_date: date
    abnormal_flag: AbnormalFlag | None = None
    source_citation: Citation


class LabReport(BaseModel):
    """Top-level shape returned by `attach_and_extract(doc_type='lab_pdf')`.

    `source_document_id` is populated after the source PDF has been written
    to OpenEMR as a FHIR `DocumentReference`; the per-result citations
    reference this same ID.
    """

    model_config = ConfigDict(extra="forbid")

    document_type: Literal["lab_pdf"] = "lab_pdf"
    results: list[LabResult] = Field(min_length=1, description="At least one extracted result; an empty lab PDF is a failed extraction")
    source_document_id: str = Field(min_length=1, description="FHIR DocumentReference/{id} for the source PDF after persistence")
    facility: str | None = Field(default=None, description="Lab facility name as printed on the report")
    ordering_provider: str | None = Field(default=None, description="Ordering clinician name as printed on the report")


# ──────────────────────────────────────────────────────────────────────────
# Intake form schema
# ──────────────────────────────────────────────────────────────────────────

Sex = Literal["male", "female", "other", "unknown"]


class Demographics(BaseModel):
    """Patient identity + contact fields extracted from the intake form."""

    model_config = ConfigDict(extra="forbid")

    given_name: str = Field(min_length=1)
    family_name: str = Field(min_length=1)
    date_of_birth: date | None = None
    sex: Sex | None = None
    address: str | None = None
    phone: str | None = None
    source_citation: Citation


class Medication(BaseModel):
    """A single current-medication line.

    Dose / frequency / route are kept as strings rather than parsed because
    intake forms surface them in free-text shapes ('twice daily',
    '500mg PO BID') that aren't worth parsing into structured units at
    extraction time — the downstream FHIR mapper handles normalization.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    dose: str | None = None
    frequency: str | None = None
    route: str | None = None
    source_citation: Citation


class Allergy(BaseModel):
    """A single allergy-list line."""

    model_config = ConfigDict(extra="forbid")

    substance: str = Field(min_length=1)
    reaction: str | None = None
    severity: Literal["mild", "moderate", "severe"] | None = None
    source_citation: Citation


class FamilyHistoryItem(BaseModel):
    """A single family-history line."""

    model_config = ConfigDict(extra="forbid")

    relation: str = Field(min_length=1, description="e.g. 'mother', 'father', 'maternal grandmother'")
    condition: str = Field(min_length=1)
    age_at_onset: int | None = Field(default=None, ge=0, le=120)
    source_citation: Citation


class IntakeForm(BaseModel):
    """Top-level shape returned by `attach_and_extract(doc_type='intake_form')`.

    The four list fields can be empty (a patient with no medications, no
    known allergies, or no recorded family history). They cannot be missing
    — the extractor explicitly returns `[]` rather than omitting the field
    so downstream code can rely on the shape.
    """

    model_config = ConfigDict(extra="forbid")

    document_type: Literal["intake_form"] = "intake_form"
    demographics: Demographics
    chief_concern: str = Field(min_length=1)
    current_medications: list[Medication] = Field(default_factory=list)
    allergies: list[Allergy] = Field(default_factory=list)
    family_history: list[FamilyHistoryItem] = Field(default_factory=list)
    source_document_id: str = Field(min_length=1, description="FHIR DocumentReference/{id} for the source intake form after persistence")


# ──────────────────────────────────────────────────────────────────────────
# Referral letter schema
# ──────────────────────────────────────────────────────────────────────────

class MedicalCondition(BaseModel):
    """A single past-medical-history line on a referral letter.

    `icd10_code` is captured when the referral letter prints it (most do —
    formatted as `(E11.9)`). It's optional because some referrals omit the
    code on the bullet line. The downstream writer falls back to the
    free-text `name` when the code is absent.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, description="Clinical name as printed (e.g. 'Type 2 diabetes mellitus without complications')")
    icd10_code: str | None = Field(default=None, description="ICD-10 code as printed (e.g. 'E11.9'); omitted if not on the page")
    source_citation: Citation


class ReferringPhysician(BaseModel):
    """The physician who authored the referral letter."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, description="Full name including credentials (e.g. 'Helen Park, MD')")
    practice: str | None = Field(default=None, description="Practice or institution name as printed")
    npi: str | None = Field(default=None, description="10-digit NPI as printed; omitted if not visible")
    source_citation: Citation


class ReferralLetter(BaseModel):
    """Top-level shape returned by `attach_and_extract(doc_type='referral_letter')`.

    Referral letters are the union of an intake-form-style chart slice
    (PMH, current meds, allergies) plus referral-specific narrative
    (reason for referral, HPI, requested action) plus the referring
    physician. The discriminated union below tags it with
    `document_type='referral_letter'` so downstream code branches cleanly.

    Empty list fields are allowed — a referral with NKDA, no current meds,
    or no separate PMH (e.g. PMH inlined into HPI) extracts as `[]`. The
    extractor explicitly returns `[]` rather than omitting the field so
    the downstream persistence loop can iterate uniformly.

    `pertinent_labs` is captured for grounding (so the chat agent can
    surface "the referral cited LDL-C 142 on 2026-04-12") but is NOT
    persisted as a separate Encounter — that would manufacture a duplicate
    lab event the lab itself never generated. The labs ride along on the
    DocumentReference and are reachable via document text retrieval.
    """

    model_config = ConfigDict(extra="forbid")

    document_type: Literal["referral_letter"] = "referral_letter"
    referring_physician: ReferringPhysician
    reason_for_referral: str = Field(min_length=1, description="The 'Reason for Referral:' paragraph, verbatim or close to it")
    history_of_present_illness: str | None = Field(default=None, description="HPI paragraph as printed; omitted if the letter has none")
    requested_action: str | None = Field(default=None, description="The 'Specific Question / Requested Action:' paragraph; omitted if absent")
    past_medical_history: list[MedicalCondition] = Field(default_factory=list)
    current_medications: list[Medication] = Field(default_factory=list)
    allergies: list[Allergy] = Field(default_factory=list)
    pertinent_labs: list[LabResult] = Field(default_factory=list, description="Lab values cited inline by the referring physician; not persisted as a separate Encounter")
    source_document_id: str = Field(min_length=1, description="FHIR DocumentReference/{id} for the source referral letter after persistence")


# ──────────────────────────────────────────────────────────────────────────
# HL7 v2 message schema (ADT-A08 + ORU-R01)
# ──────────────────────────────────────────────────────────────────────────

Hl7MessageType = Literal["ADT^A08", "ORU^R01"]
"""HL7 v2 message subtypes the parser supports. MSH-9 carries the value
verbatim from the source message; we keep the caret form so users
familiar with HL7 see the standard wire representation rather than a
re-spelled variant."""


Hl7AllergyTypeCode = Literal["DA", "FA", "MA", "EA", "MC", "OT", "AA", "PA"]
"""AL1-2 allergy type codes from HL7 table 0127. DA = drug allergy,
FA = food allergy, MA = misc allergy, EA = environmental, MC = misc
contraindication, OT = other, AA = animal, PA = plant. Optional —
many AL1 segments omit the type."""


Hl7AllergySeverity = Literal["SV", "MO", "MI"]
"""AL1-4 severity codes from HL7 table 0128. SV = severe, MO = moderate,
MI = mild. Optional. Maps to the lowercase forms on `Allergy.severity`
(severe/moderate/mild) at persistence time."""


class Hl7Allergy(BaseModel):
    """One AL1 segment from an HL7 v2 message.

    Distinct from the intake-form `Allergy` shape because the AL1 segment
    carries HL7-coded type/severity vocabularies that we want to preserve
    on round-trip. The persistence step normalizes these to OpenEMR's
    severity vocabulary before writing.
    """

    model_config = ConfigDict(extra="forbid")

    substance: str = Field(min_length=1, description="AL1-3.1 — allergen name as transmitted (e.g. 'LISINOPRIL', 'PENICILLIN')")
    type_code: Hl7AllergyTypeCode | None = None
    severity: Hl7AllergySeverity | None = None
    reaction: str | None = Field(default=None, description="AL1-5 — free-text reaction description")
    source_citation: Citation


class Hl7LabPanel(BaseModel):
    """One OBR group + its OBX rows + any panel-scoped NTE notes.

    Each Hl7LabPanel persists as a single Encounter (one panel = one
    encounter — an OBR is a distinct order in HL7's data model and the
    panel-level LOINC, collection date, and clinician interpretation note
    must stay attached to that order's results). A multi-OBR ORU
    therefore produces multiple Encounters from a single source HL7
    DocumentReference.
    """

    model_config = ConfigDict(extra="forbid")

    panel_loinc: str | None = Field(default=None, description="OBR-4.1 — LOINC code for the ordered panel (e.g. '24323-8')")
    panel_name: str | None = Field(default=None, description="OBR-4.2 — LOINC long name as printed (e.g. 'Comprehensive metabolic 2000 panel')")
    collection_date: date | None = Field(default=None, description="OBR-7 date portion — when the specimen was collected")
    results: list[LabResult] = Field(default_factory=list, description="OBX rows for this panel; LabResult.source_citation points at OBR[i]/OBX[j]")
    notes: str | None = Field(default=None, description="NTE-3 attached to this panel — a clinician's interpretation line (e.g. 'BNP elevated — consistent with HFrEF decompensation')")


class Hl7Message(BaseModel):
    """Top-level shape returned by `attach_and_extract_hl7(...)`.

    A single Hl7Message represents one .hl7 file. The `message_type`
    discriminator (sourced from MSH-9) tags which subset of fields is
    populated:

    - **ADT-A08** (patient information update): populates `event_reason`
      (EVN-9 clinical note line) and `allergies[]` (AL1 segments). The
      patient demographics on PID are deliberately NOT mirrored into the
      schema — the patient already exists in OpenEMR (we used their
      MRN/UUID to attach the message), and overwriting demographics from
      a registration message would clobber the chart with whatever the
      sending system happened to encode.

    - **ORU-R01** (lab results): populates `lab_panels[]`, one entry per
      OBR group. Each panel becomes a separate Encounter on persistence
      so OBR-level provenance (LOINC, collection date, interpretive note)
      stays attached to the right order.

    `bbox` on every Citation in this message is `None` — HL7 is text, not
    pixels, so there is no spatial overlay. Citations cite by segment
    index instead (`OBR[2] / OBX[1]`, `AL1[2]`, `EVN`).
    """

    model_config = ConfigDict(extra="forbid")

    document_type: Literal["hl7_message"] = "hl7_message"
    message_type: Hl7MessageType
    sending_application: str | None = Field(default=None, description="MSH-3 — the application that generated this message (e.g. 'BHS-LIS', 'REGISTRATION')")
    sending_facility: str | None = Field(default=None, description="MSH-4 — the facility that owns the sending application")
    message_control_id: str | None = Field(default=None, description="MSH-10 — the unique control id, used as a Layer-2 dedup signal")
    timestamp: datetime | None = Field(default=None, description="MSH-7 — when the message was generated by the sender")
    patient_mrn: str | None = Field(default=None, description="PID-3.1 — the medical record number on the message (sanity-checked against the attaching patient_uuid at persistence time)")

    # ADT-A08 fields
    event_reason: str | None = Field(default=None, description="EVN-9 — the clinical reason note carried on a registration update")

    # AL1 segments (most common on ADT-A08, occasionally on ORU-R01)
    allergies: list[Hl7Allergy] = Field(default_factory=list)

    # ORU-R01 fields (one entry per OBR group)
    lab_panels: list[Hl7LabPanel] = Field(default_factory=list)

    source_document_id: str = Field(min_length=1, description="FHIR DocumentReference/{id} for the source .hl7 file after persistence")


# ──────────────────────────────────────────────────────────────────────────
# Fax packet schema (multi-page TIFF: cover + referral + face sheet + lab)
# ──────────────────────────────────────────────────────────────────────────

FaxUrgency = Literal["routine", "urgent", "stat"]
"""Standard fax-cover-sheet urgency tick-box vocabulary. Lowercased so
the model returns the value verbatim regardless of how the cover prints
it (`Routine` / `routine` / `[X] ROUTINE`)."""


class FaxCoverSheet(BaseModel):
    """The transmittal cover sheet that fronts a multi-page fax packet.

    All fields optional because real-world fax cover sheets vary in
    completeness — some omit the message body, some omit the recipient
    fax number, some skip the urgency tick-boxes entirely. The fields
    we do capture are the ones the chat agent surfaces back when a
    clinician asks "who sent that fax and why."
    """

    model_config = ConfigDict(extra="forbid")

    fax_date: date | None = Field(default=None, description="The DATE field on the cover sheet (when the fax was sent)")
    sender_name: str | None = Field(default=None, description="FROM clinician — full name including credentials, e.g. 'Dr. Helen Park, MD'")
    sender_practice: str | None = Field(default=None, description="FROM practice / institution name")
    sender_phone: str | None = None
    sender_fax: str | None = None
    recipient_name: str | None = Field(default=None, description="TO clinician — full name with credentials")
    recipient_practice: str | None = None
    recipient_fax: str | None = None
    page_count: int | None = Field(default=None, ge=1, le=200, description="PAGES field as printed (cover-inclusive count)")
    urgency: FaxUrgency | None = None
    message: str | None = Field(default=None, description="The free-text MESSAGE: paragraph on the cover (a clinical summary the sender wrote, distinct from the formal referral body on later pages)")
    source_citation: Citation


class FaxPacket(BaseModel):
    """Top-level shape returned by `attach_and_extract(doc_type='fax_packet')`.

    A fax packet bundles multiple clinical artifacts in one transmission:
    a cover sheet, a referral request page, a patient face sheet (active
    problems / meds / allergies), and frequently a lab report page. We
    extract the high-value fields across all pages into a single typed
    object — the schema doesn't try to recover the per-page boundary.

    Distinct from `ReferralLetter` in two ways:

    1. The `cover_sheet` carries transmittal metadata that doesn't exist
       on a plain referral letter (recipient fax #, urgency, page count).
    2. `lab_results` IS persisted as a real lab Encounter on this path.
       In a referral letter, embedded labs are merely *quoted context* —
       the referral cites prior labs to justify the consult ask, but a
       separate ORU/lab PDF carries the authoritative results. In a fax
       packet, the lab page IS the source of those results — there is no
       parallel ORU. Skipping persistence would leave the chat agent
       unable to answer "what was Chen's last LDL?" via `get_notes_24h`.

    Empty list fields are allowed — a cover-only fax with no lab page
    extracts as `lab_results=[]`. `bbox` on every Citation IS populated
    here (TIFF pages render through the same image surface as PDFs, so
    the visual-overlay UI applies).
    """

    model_config = ConfigDict(extra="forbid")

    document_type: Literal["fax_packet"] = "fax_packet"
    cover_sheet: FaxCoverSheet
    reason_for_consultation: str | None = Field(default=None, description="The Reason-for-Referral / Reason-for-Consultation paragraph from the referral page, verbatim or close to it")
    receiving_specialty: str | None = Field(default=None, description="Specialty the packet is being sent TO (e.g. 'Cardiology', 'Endocrinology')")
    active_problems: list[MedicalCondition] = Field(default_factory=list, description="From the face sheet's Active Problem List — preserve ICD-10 codes when printed")
    current_medications: list[Medication] = Field(default_factory=list, description="From the face sheet's Active Medication List")
    allergies: list[Allergy] = Field(default_factory=list, description="From the face sheet's Allergies field — `[]` for NKDA")
    lab_results: list[LabResult] = Field(default_factory=list, description="Inline lab results from any embedded Laboratory Report page; persisted as a real lab Encounter (the fax IS the source of these results, not a quotation)")
    source_document_id: str = Field(min_length=1, description="FHIR DocumentReference/{id} for the source TIFF after persistence")


# ──────────────────────────────────────────────────────────────────────────
# Workbook schema (.xlsx — clinical dashboard with 4 sheets)
# ──────────────────────────────────────────────────────────────────────────

WorkbookCareGapStatus = Literal["UP TO DATE", "OVERDUE", "DUE SOON", "NOT APPLICABLE", "DEFERRED"]
"""Care-gap status vocabulary as it appears in the workbook's Care_Gaps
sheet. Uppercase to match the source spreadsheet verbatim. Only
OVERDUE rows are persisted as medical_problem entries; the rest stay
on the source DocumentReference for document-text retrieval."""


class WorkbookMedication(BaseModel):
    """One row from the workbook's Medications sheet.

    Workbook medications are richer than IntakeForm/ReferralLetter
    medications: they carry brand + generic + start_date + last_filled +
    refills_remaining + prescriber. We preserve all of it on the schema
    even though only the title-rendering set (name + dose + route +
    frequency-as-sig) is used at persistence time — the extra fields
    let the chat agent answer "when was Chen's atorvastatin last
    filled?" via document-text retrieval.
    """

    model_config = ConfigDict(extra="forbid")

    brand: str | None = None
    generic: str | None = None
    strength: str | None = None
    route: str | None = None
    sig: str | None = Field(default=None, description="Free-text dosing instruction (e.g. '1 tab PO daily')")
    indication: str | None = None
    start_date: date | None = None
    last_filled: date | None = None
    refills_remaining: int | None = Field(default=None, ge=0, le=99)
    prescriber: str | None = None
    source_citation: Citation


class WorkbookLabValue(BaseModel):
    """A single (collection_date, value) pair within a lab-trend row."""

    model_config = ConfigDict(extra="forbid")

    collection_date: date
    value: str = Field(min_length=1, description="Cell value as printed (numeric or qualitative); kept as a string because workbook columns sometimes mix '142' with 'pending' / 'cancelled' on different rows")


class WorkbookLabTrend(BaseModel):
    """One test row from the workbook's Labs_Trend sheet — with all
    of its date columns. Persistence pivots on `values` so each
    collection-date column lands as its own lab Encounter."""

    model_config = ConfigDict(extra="forbid")

    test_name: str = Field(min_length=1)
    loinc: str | None = None
    units: str | None = None
    reference_range: str | None = None
    values: list[WorkbookLabValue] = Field(default_factory=list, description="One entry per dated column on the row; empty cells are skipped")
    source_citation: Citation


class WorkbookCareGap(BaseModel):
    """One row from the workbook's Care_Gaps sheet."""

    model_config = ConfigDict(extra="forbid")

    measure: str = Field(min_length=1, description="The HEDIS / USPSTF measure label as printed (e.g. 'Diabetic eye exam (annual)')")
    reference: str | None = Field(default=None, description="HEDIS or USPSTF reference code (e.g. 'HEDIS EED', 'USPSTF Grade B')")
    status: WorkbookCareGapStatus | None = None
    last_done: date | None = None
    due_date: date | None = None
    notes: str | None = None
    source_citation: Citation


class Workbook(BaseModel):
    """Top-level shape returned by `attach_and_extract_workbook(...)`.

    A clinical workbook (.xlsx) is a per-patient summary dashboard with
    four sheets: Patient (key-value demographics + allergies), Medications
    (one row per active rx), Labs_Trend (one row per test, one column per
    draw date), Care_Gaps (HEDIS/USPSTF compliance items).

    Persistence dispatches as follows:

    - **Patient demographics** — skipped; the patient already exists.
    - **Allergies cell** — single-string field. `NKDA` / empty → no write;
      anything else writes one allergy entry.
    - **Medications** — one `write_medication` per row.
    - **Labs_Trend** — pivoted to one Encounter per collection-date column
      (preserves the trend on the chart timeline; matches HL7 ORU's
      one-OBR-one-Encounter pattern).
    - **Care_Gaps** — only rows with `status == 'OVERDUE'` write as
      medical_problems with the prefix `Care gap (overdue): ...`.
      UP TO DATE / NOT APPLICABLE entries stay reachable via document
      retrieval but don't pollute Active Problems.

    Citations are sheet/row indexed (`Medications row 3`,
    `Labs_Trend col 2026-04-12 row LDL`); `bbox=None` everywhere — xlsx
    has no spatial overlay layer.
    """

    model_config = ConfigDict(extra="forbid")

    document_type: Literal["workbook"] = "workbook"

    # Patient sheet — key/value demographics. Captured for round-trip
    # fidelity (so the chat agent can confirm "this workbook is about
    # Margaret Chen") but NOT written back to OpenEMR — patient already
    # exists, and overwriting demographics from a summary sheet is
    # exactly the failure mode the IntakeForm path also avoids.
    patient_name: str | None = None
    patient_dob: date | None = None
    patient_mrn: str | None = None
    pcp_name: str | None = None
    insurance: str | None = None
    allergies_text: str | None = Field(default=None, description="Single-string from the Patient sheet's Allergies cell; 'NKDA' is common and parses as no allergies on persistence")
    as_of_date: date | None = Field(default=None, description="The Patient sheet's As_Of_Date — when the workbook was last refreshed")

    # Tabular sheets
    medications: list[WorkbookMedication] = Field(default_factory=list)
    lab_trends: list[WorkbookLabTrend] = Field(default_factory=list)
    care_gaps: list[WorkbookCareGap] = Field(default_factory=list)

    source_document_id: str = Field(min_length=1, description="FHIR DocumentReference/{id} for the source .xlsx after persistence")


# ──────────────────────────────────────────────────────────────────────────
# Discriminated union — the public return type of `attach_and_extract`
# ──────────────────────────────────────────────────────────────────────────

ExtractedDocument = Annotated[
    Union[LabReport, IntakeForm, ReferralLetter, Hl7Message, FaxPacket, Workbook],
    Field(discriminator="document_type"),
]
"""Discriminated union over the supported doc types. Pydantic uses
`document_type` (a `Literal` on each member) to pick the correct model
without having to try-validate-fail through the union. Future doc types
extend this union by adding new literal-tagged shapes; the discriminator
stays the same."""
