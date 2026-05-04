"""Validation tests for the synthetic demo-extraction fixtures.

The Phase 1.2 generator (`scripts/generate_demo_documents.py`) writes two
schema-valid expected-extraction JSONs to `data/demo_documents/fixtures/`.
These are the *ground truth* used by Phase 2 to compare what the VLM
actually extracted from the PDFs against what should have been extracted.

This test guarantees the fixtures themselves are schema-valid before the
extractor exists — if a future schema change breaks the fixtures, this test
fails immediately rather than silently masking a regression in Phase 2 work.
It also verifies the lab fixture round-trips through the discriminated
union, exercising the same dispatch path the extractor will use.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import TypeAdapter

from app.extraction.schemas import (
    ExtractedDocument,
    IntakeForm,
    LabReport,
)


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FIXTURE_DIR = REPO_ROOT / "data" / "demo_documents" / "fixtures"


@pytest.fixture(scope="module")
def lab_fixture_json() -> dict:
    path = FIXTURE_DIR / "cohen_lab_2026-04-30.expected.json"
    return json.loads(path.read_text())


@pytest.fixture(scope="module")
def intake_fixture_json() -> dict:
    path = FIXTURE_DIR / "cohen_intake_2026-04-30.expected.json"
    return json.loads(path.read_text())


class TestLabFixture:
    def test_parses_through_lab_report(self, lab_fixture_json: dict) -> None:
        report = LabReport.model_validate(lab_fixture_json)
        assert report.document_type == "lab_pdf"
        # Generator writes 10 lab tests; lock that in so a silent change is loud.
        assert len(report.results) == 10

    def test_dispatches_through_extracted_document(
        self, lab_fixture_json: dict,
    ) -> None:
        adapter = TypeAdapter(ExtractedDocument)
        result = adapter.validate_python(lab_fixture_json)
        assert isinstance(result, LabReport)

    def test_every_result_has_a_citation(self, lab_fixture_json: dict) -> None:
        report = LabReport.model_validate(lab_fixture_json)
        for r in report.results:
            assert r.source_citation.source_type == "lab_pdf"
            assert r.source_citation.source_id == "DocumentReference/cohen_lab_2026-04-30"
            assert r.source_citation.quote_or_value  # non-empty enforced by schema


class TestIntakeFixture:
    def test_parses_through_intake_form(self, intake_fixture_json: dict) -> None:
        form = IntakeForm.model_validate(intake_fixture_json)
        assert form.document_type == "intake_form"
        # Lock the generated counts so silent drift in the generator is caught.
        assert len(form.current_medications) == 4
        assert len(form.allergies) == 2
        assert len(form.family_history) == 3

    def test_dispatches_through_extracted_document(
        self, intake_fixture_json: dict,
    ) -> None:
        adapter = TypeAdapter(ExtractedDocument)
        result = adapter.validate_python(intake_fixture_json)
        assert isinstance(result, IntakeForm)

    def test_every_field_has_a_citation(self, intake_fixture_json: dict) -> None:
        form = IntakeForm.model_validate(intake_fixture_json)
        # Demographics + every list element must carry a citation back to the
        # intake form, with the same source_id.
        expected_source_id = "DocumentReference/cohen_intake_2026-04-30"
        assert form.demographics.source_citation.source_id == expected_source_id
        for item in (
            *form.current_medications,
            *form.allergies,
            *form.family_history,
        ):
            assert item.source_citation.source_id == expected_source_id
            assert item.source_citation.source_type == "intake_form"
