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

from datetime import date
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field


# ──────────────────────────────────────────────────────────────────────────
# Shared primitives
# ──────────────────────────────────────────────────────────────────────────

SourceType = Literal["lab_pdf", "intake_form", "guideline", "fhir_resource"]
"""Where a citation points. Spans both the document-extraction path
(lab_pdf, intake_form), the evidence-retrieval path (guideline), and the
existing chart-summarizer path (fhir_resource)."""


DocumentType = Literal["lab_pdf", "intake_form"]
"""Doc types the Phase 2 extractor + writer support. Strict subset of
SourceType (the citation-source vocabulary is broader than the
extractable-doc vocabulary — guideline / fhir_resource citations exist
without ever being uploaded as documents)."""


DOC_TYPE_LABELS: dict[str, str] = {
    "lab_pdf":     "Lab Report (PDF)",
    "intake_form": "Patient Intake Form",
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

AbnormalFlag = Literal["H", "L", "N", "C"]
"""Abnormal-flag values seen on US lab PDFs: H = high, L = low, N = normal,
C = critical. H/L/N follow HL7 v2 (where critical is `HH`/`LL`); we collapse
critical-high and critical-low into a single `C` because the doctor-facing
distinction is the criticality, not the direction. Extending to the full
HL7 set (HH/LL/A/AA/W/B) is a future tightening if real lab PDFs use them.
Optional — many lab PDFs omit the flag and only print the reference range."""


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
# Discriminated union — the public return type of `attach_and_extract`
# ──────────────────────────────────────────────────────────────────────────

ExtractedDocument = Annotated[
    Union[LabReport, IntakeForm],
    Field(discriminator="document_type"),
]
"""Discriminated union over the two MVP doc types. Pydantic uses
`document_type` (a `Literal` on each member) to pick the correct model
without having to try-validate-fail through the union. Future doc types
(referral_fax, medication_list — Week 2 extension) extend this union by
adding new literal-tagged shapes; the discriminator stays the same."""
