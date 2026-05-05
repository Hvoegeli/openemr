"""Validation tests for the extracted-document schemas.

Mirrors the Week 2 PRD eval rubric `schema_valid` — these are the
deterministic, fast checks that gate every extraction. If a real-world
extraction starts producing JSON the schemas reject, the gate fails before
the data reaches the LLM or the persistence layer.

Coverage:
- Required fields are required; missing them raises ValidationError.
- `extra="forbid"` rejects hallucinated fields the VLM might smuggle in.
- Discriminator literals (`document_type`) cannot be mis-set.
- Numeric / enum / date constraints fire correctly.
- Empty container behaviour matches docstring contract (lab results
  cannot be empty; intake list-fields default to empty).
"""

from __future__ import annotations

from datetime import date

import pytest
from pydantic import TypeAdapter, ValidationError

from app.extraction.schemas import (
    DOC_TYPE_LABELS,
    Allergy,
    BoundingBox,
    Citation,
    Demographics,
    DocumentType,
    ExtractedDocument,
    FamilyHistoryItem,
    IntakeForm,
    LabReport,
    LabResult,
    Medication,
)


# ──────────────────────────────────────────────────────────────────────────
# Citation
# ──────────────────────────────────────────────────────────────────────────


def _valid_lab_citation() -> Citation:
    return Citation(
        source_type="lab_pdf",
        source_id="DocumentReference/doc-abc-123",
        page_or_section="page 1",
        field_or_chunk_id="HbA1c",
        quote_or_value="7.4 %",
    )


class TestCitation:
    def test_minimal_lab_citation_parses(self) -> None:
        c = _valid_lab_citation()
        assert c.source_type == "lab_pdf"
        assert c.bbox is None

    def test_with_bbox(self) -> None:
        c = Citation(
            source_type="lab_pdf",
            source_id="DocumentReference/doc-abc",
            page_or_section="page 1",
            field_or_chunk_id="HbA1c",
            quote_or_value="7.4 %",
            bbox=BoundingBox(page=1, x=120.0, y=340.0, width=80.0, height=20.0),
        )
        assert c.bbox is not None
        assert c.bbox.page == 1

    def test_extra_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Citation.model_validate({
                "source_type": "lab_pdf",
                "source_id": "x",
                "page_or_section": "y",
                "field_or_chunk_id": "z",
                "quote_or_value": "v",
                "hallucinated_field": "nope",
            })

    @pytest.mark.parametrize(
        "missing_field",
        ["source_type", "source_id", "page_or_section", "field_or_chunk_id", "quote_or_value"],
    )
    def test_missing_required_field_rejected(self, missing_field: str) -> None:
        payload = {
            "source_type": "lab_pdf",
            "source_id": "x",
            "page_or_section": "y",
            "field_or_chunk_id": "z",
            "quote_or_value": "v",
        }
        del payload[missing_field]
        with pytest.raises(ValidationError):
            Citation.model_validate(payload)

    def test_invalid_source_type_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Citation.model_validate({
                "source_type": "made_up",
                "source_id": "x",
                "page_or_section": "y",
                "field_or_chunk_id": "z",
                "quote_or_value": "v",
            })

    def test_empty_source_id_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Citation.model_validate({
                "source_type": "lab_pdf",
                "source_id": "",
                "page_or_section": "y",
                "field_or_chunk_id": "z",
                "quote_or_value": "v",
            })

    @pytest.mark.parametrize(
        "empty_field",
        ["source_id", "page_or_section", "field_or_chunk_id", "quote_or_value"],
    )
    def test_empty_required_string_field_rejected(self, empty_field: str) -> None:
        """Every string field in the citation must be non-empty.

        An empty string is structurally indistinguishable from missing data
        but would silently pass a `citation_present` check downstream — the
        whole point of the citation is to be locatable and verifiable.
        """
        payload = {
            "source_type": "lab_pdf",
            "source_id": "x",
            "page_or_section": "y",
            "field_or_chunk_id": "z",
            "quote_or_value": "v",
        }
        payload[empty_field] = ""
        with pytest.raises(ValidationError):
            Citation.model_validate(payload)


class TestBoundingBox:
    def test_minimal(self) -> None:
        b = BoundingBox(page=1, x=0.0, y=0.0, width=10.0, height=10.0)
        assert b.page == 1

    def test_zero_page_rejected(self) -> None:
        with pytest.raises(ValidationError):
            BoundingBox(page=0, x=0, y=0, width=10, height=10)

    def test_zero_width_rejected(self) -> None:
        with pytest.raises(ValidationError):
            BoundingBox(page=1, x=0, y=0, width=0, height=10)

    def test_negative_y_rejected(self) -> None:
        with pytest.raises(ValidationError):
            BoundingBox(page=1, x=0, y=-1.0, width=10, height=10)

    def test_negative_x_rejected(self) -> None:
        with pytest.raises(ValidationError):
            BoundingBox(page=1, x=-1.0, y=0, width=10, height=10)

    def test_zero_height_rejected(self) -> None:
        with pytest.raises(ValidationError):
            BoundingBox(page=1, x=0, y=0, width=10, height=0)


# ──────────────────────────────────────────────────────────────────────────
# Lab schemas
# ──────────────────────────────────────────────────────────────────────────


class TestLabResult:
    def test_numeric_value(self) -> None:
        r = LabResult(
            test_name="Hemoglobin A1c",
            value=7.4,
            unit="%",
            reference_range="4.0-5.6 %",
            collection_date=date(2026, 4, 30),
            abnormal_flag="H",
            source_citation=_valid_lab_citation(),
        )
        assert r.value == 7.4
        assert r.abnormal_flag == "H"

    def test_qualitative_value(self) -> None:
        r = LabResult(
            test_name="HIV Antibody",
            value="negative",
            unit="",
            reference_range=None,
            collection_date=date(2026, 4, 30),
            abnormal_flag=None,
            source_citation=_valid_lab_citation(),
        )
        assert r.value == "negative"

    def test_invalid_abnormal_flag_rejected(self) -> None:
        with pytest.raises(ValidationError):
            LabResult.model_validate({
                "test_name": "HbA1c",
                "value": 7.4,
                "unit": "%",
                "collection_date": "2026-04-30",
                "abnormal_flag": "X",
                "source_citation": _valid_lab_citation().model_dump(),
            })

    def test_missing_citation_rejected(self) -> None:
        with pytest.raises(ValidationError):
            LabResult.model_validate({
                "test_name": "HbA1c",
                "value": 7.4,
                "unit": "%",
                "collection_date": "2026-04-30",
            })

    def test_empty_test_name_rejected(self) -> None:
        with pytest.raises(ValidationError):
            LabResult.model_validate({
                "test_name": "",
                "value": 7.4,
                "unit": "%",
                "collection_date": "2026-04-30",
                "source_citation": _valid_lab_citation().model_dump(),
            })


class TestLabReport:
    def test_one_result(self) -> None:
        report = LabReport(
            results=[
                LabResult(
                    test_name="HbA1c",
                    value=7.4,
                    unit="%",
                    collection_date=date(2026, 4, 30),
                    source_citation=_valid_lab_citation(),
                ),
            ],
            source_document_id="DocumentReference/doc-abc-123",
        )
        assert report.document_type == "lab_pdf"
        assert len(report.results) == 1

    def test_empty_results_rejected(self) -> None:
        with pytest.raises(ValidationError):
            LabReport.model_validate({
                "results": [],
                "source_document_id": "DocumentReference/doc-abc-123",
            })

    def test_wrong_document_type_rejected(self) -> None:
        with pytest.raises(ValidationError):
            LabReport.model_validate({
                "document_type": "intake_form",
                "results": [{
                    "test_name": "HbA1c",
                    "value": 7.4,
                    "unit": "%",
                    "collection_date": "2026-04-30",
                    "source_citation": _valid_lab_citation().model_dump(),
                }],
                "source_document_id": "DocumentReference/doc-abc-123",
            })

    def test_empty_source_document_id_rejected(self) -> None:
        with pytest.raises(ValidationError):
            LabReport.model_validate({
                "results": [{
                    "test_name": "HbA1c",
                    "value": 7.4,
                    "unit": "%",
                    "collection_date": "2026-04-30",
                    "source_citation": _valid_lab_citation().model_dump(),
                }],
                "source_document_id": "",
            })


# ──────────────────────────────────────────────────────────────────────────
# Intake schemas
# ──────────────────────────────────────────────────────────────────────────


def _valid_intake_citation(field: str) -> Citation:
    return Citation(
        source_type="intake_form",
        source_id="DocumentReference/intake-xyz-789",
        page_or_section="page 1",
        field_or_chunk_id=field,
        quote_or_value="(...)",
    )


class TestDemographics:
    def test_minimal(self) -> None:
        d = Demographics(
            given_name="Jane",
            family_name="Cohen",
            source_citation=_valid_intake_citation("demographics"),
        )
        assert d.given_name == "Jane"
        assert d.date_of_birth is None

    def test_invalid_sex_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Demographics.model_validate({
                "given_name": "Jane",
                "family_name": "Cohen",
                "sex": "F",  # must be 'female', not 'F'
                "source_citation": _valid_intake_citation("demographics").model_dump(),
            })

    def test_empty_given_name_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Demographics.model_validate({
                "given_name": "",
                "family_name": "Cohen",
                "source_citation": _valid_intake_citation("demographics").model_dump(),
            })


class TestMedication:
    def test_minimal(self) -> None:
        m = Medication(name="Lisinopril", source_citation=_valid_intake_citation("meds"))
        assert m.dose is None

    def test_full(self) -> None:
        m = Medication(
            name="Metformin",
            dose="500mg",
            frequency="BID",
            route="PO",
            source_citation=_valid_intake_citation("meds"),
        )
        assert m.dose == "500mg"


class TestAllergy:
    def test_minimal(self) -> None:
        a = Allergy(substance="penicillin", source_citation=_valid_intake_citation("allergies"))
        assert a.severity is None

    def test_invalid_severity_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Allergy.model_validate({
                "substance": "penicillin",
                "severity": "deadly",  # not in {mild, moderate, severe}
                "source_citation": _valid_intake_citation("allergies").model_dump(),
            })


class TestFamilyHistoryItem:
    def test_minimal(self) -> None:
        f = FamilyHistoryItem(
            relation="mother",
            condition="type 2 diabetes",
            source_citation=_valid_intake_citation("family_history"),
        )
        assert f.age_at_onset is None

    def test_age_out_of_range_rejected(self) -> None:
        with pytest.raises(ValidationError):
            FamilyHistoryItem.model_validate({
                "relation": "mother",
                "condition": "T2DM",
                "age_at_onset": 200,
                "source_citation": _valid_intake_citation("family_history").model_dump(),
            })


class TestIntakeForm:
    def _valid_form(self) -> IntakeForm:
        return IntakeForm(
            demographics=Demographics(
                given_name="Jane",
                family_name="Cohen",
                source_citation=_valid_intake_citation("demographics"),
            ),
            chief_concern="follow-up for diabetes",
            source_document_id="DocumentReference/intake-xyz-789",
        )

    def test_empty_lists_default(self) -> None:
        f = self._valid_form()
        assert f.current_medications == []
        assert f.allergies == []
        assert f.family_history == []

    def test_with_lists_populated(self) -> None:
        f = IntakeForm(
            demographics=Demographics(
                given_name="Jane",
                family_name="Cohen",
                source_citation=_valid_intake_citation("demographics"),
            ),
            chief_concern="follow-up for diabetes",
            current_medications=[
                Medication(name="Metformin", source_citation=_valid_intake_citation("meds-0")),
            ],
            allergies=[
                Allergy(substance="penicillin", source_citation=_valid_intake_citation("allergies-0")),
            ],
            family_history=[
                FamilyHistoryItem(
                    relation="mother",
                    condition="T2DM",
                    source_citation=_valid_intake_citation("fh-0"),
                ),
            ],
            source_document_id="DocumentReference/intake-xyz-789",
        )
        assert len(f.current_medications) == 1
        assert len(f.allergies) == 1
        assert len(f.family_history) == 1

    def test_missing_demographics_rejected(self) -> None:
        with pytest.raises(ValidationError):
            IntakeForm.model_validate({
                "chief_concern": "follow-up",
                "source_document_id": "DocumentReference/x",
            })

    def test_empty_chief_concern_rejected(self) -> None:
        with pytest.raises(ValidationError):
            IntakeForm.model_validate({
                "demographics": Demographics(
                    given_name="Jane",
                    family_name="Cohen",
                    source_citation=_valid_intake_citation("demographics"),
                ).model_dump(),
                "chief_concern": "",
                "source_document_id": "DocumentReference/x",
            })

    def test_wrong_document_type_rejected(self) -> None:
        with pytest.raises(ValidationError):
            IntakeForm.model_validate({
                "document_type": "lab_pdf",
                "demographics": Demographics(
                    given_name="Jane",
                    family_name="Cohen",
                    source_citation=_valid_intake_citation("demographics"),
                ).model_dump(),
                "chief_concern": "follow-up",
                "source_document_id": "DocumentReference/x",
            })

    def test_empty_source_document_id_rejected(self) -> None:
        with pytest.raises(ValidationError):
            IntakeForm.model_validate({
                "demographics": Demographics(
                    given_name="Jane",
                    family_name="Cohen",
                    source_citation=_valid_intake_citation("demographics"),
                ).model_dump(),
                "chief_concern": "follow-up",
                "source_document_id": "",
            })


# ──────────────────────────────────────────────────────────────────────────
# Discriminated union
# ──────────────────────────────────────────────────────────────────────────


class TestDocTypeConstantAlignment:
    """Three constants describe the supported doc-type set:
       - app.extraction.schemas.DocumentType        — the Literal type
       - app.extraction.schemas.DOC_TYPE_LABELS     — UI dropdown labels
       - app.fhir.writer.DOC_CATEGORIES             — OpenEMR category paths
    Adding a new doc type requires updating ALL THREE; this test makes
    silent drift between them loud."""

    def test_keys_match_across_all_three_sources(self) -> None:
        from typing import get_args

        from app.fhir.writer import DOC_CATEGORIES

        type_args = set(get_args(DocumentType))
        labels_keys = set(DOC_TYPE_LABELS)
        cats_keys = set(DOC_CATEGORIES)
        assert type_args == labels_keys, (
            f"DocumentType {type_args} != DOC_TYPE_LABELS keys {labels_keys}"
        )
        assert type_args == cats_keys, (
            f"DocumentType {type_args} != DOC_CATEGORIES keys {cats_keys}"
        )


class TestExtractedDocument:
    """`ExtractedDocument` is a discriminated union — Pydantic should pick
    the right model based on `document_type` without try/fail-through."""

    _adapter: TypeAdapter[ExtractedDocument] = TypeAdapter(ExtractedDocument)

    def test_lab_pdf_dispatched_to_lab_report(self) -> None:
        result = self._adapter.validate_python({
            "document_type": "lab_pdf",
            "results": [{
                "test_name": "HbA1c",
                "value": 7.4,
                "unit": "%",
                "collection_date": "2026-04-30",
                "source_citation": _valid_lab_citation().model_dump(),
            }],
            "source_document_id": "DocumentReference/doc-abc-123",
        })
        assert isinstance(result, LabReport)
        assert result.results[0].test_name == "HbA1c"

    def test_intake_form_dispatched_to_intake_form(self) -> None:
        result = self._adapter.validate_python({
            "document_type": "intake_form",
            "demographics": Demographics(
                given_name="Jane",
                family_name="Cohen",
                source_citation=_valid_intake_citation("demographics"),
            ).model_dump(),
            "chief_concern": "follow-up for diabetes",
            "source_document_id": "DocumentReference/intake-xyz-789",
        })
        assert isinstance(result, IntakeForm)
        assert result.chief_concern == "follow-up for diabetes"

    def test_unknown_document_type_rejected(self) -> None:
        with pytest.raises(ValidationError):
            self._adapter.validate_python({
                "document_type": "referral_fax",  # not a member of the union
                "source_document_id": "x",
            })

    def test_missing_discriminator_rejected(self) -> None:
        with pytest.raises(ValidationError):
            self._adapter.validate_python({
                "results": [{
                    "test_name": "HbA1c",
                    "value": 7.4,
                    "unit": "%",
                    "collection_date": "2026-04-30",
                    "source_citation": _valid_lab_citation().model_dump(),
                }],
                "source_document_id": "DocumentReference/doc-abc-123",
            })
