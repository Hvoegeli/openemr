"""Generate the synthetic Week 2 MVP demo documents.

Produces two PDFs and two matching expected-extraction fixture JSONs for the
existing Week 1 demo patient (Nora Cohen):

- `cohen_lab_2026-04-30.pdf`         — comprehensive metabolic + lipid panel
- `cohen_lab_2026-04-30.expected.json` — schema-valid `LabReport`
- `cohen_intake_2026-04-30.pdf`         — annual-visit intake form
- `cohen_intake_2026-04-30.expected.json` — schema-valid `IntakeForm`

The fixtures are the *ground truth* for what `attach_and_extract` should
produce when fed the corresponding PDF. Used by Phase 2 to compare extracted
output against expected output (`factually_consistent` rubric category in the
Week 2 eval gate).

Identity continuity with Week 1: the patient is Nora Cohen (PUUID
`a1a6044b-c6af-40a4-80aa-4c5ce61014da`, the existing seeded demo patient
referenced from `seed_cohen.py`). Conditions, allergies, and medications
mirror what's already in the chart so the lab values + intake fields tell
a coherent story when the agent ties them back to FHIR data.

Run: `cd clinical-copilot && PYTHONPATH=. uv run python scripts/generate_demo_documents.py`
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from app.extraction.schemas import (
    Allergy,
    Citation,
    Demographics,
    FamilyHistoryItem,
    IntakeForm,
    LabReport,
    LabResult,
    Medication,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "data" / "demo_documents"
FIXTURE_DIR = OUT_DIR / "fixtures"

# Stable IDs the extractor will use as `source_document_id` placeholders
# during Phase 2 testing (before the real FHIR DocumentReference ID is
# minted). Phase 1.3 will replace these with the real persistence ID.
LAB_DOC_ID = "DocumentReference/cohen_lab_2026-04-30"
INTAKE_DOC_ID = "DocumentReference/cohen_intake_2026-04-30"


# ──────────────────────────────────────────────────────────────────────────
# Lab PDF
# ──────────────────────────────────────────────────────────────────────────

LAB_FACILITY = "Acme Diagnostic Laboratories"
LAB_PROVIDER = "Sarah Martinez, MD"
LAB_REPORT_DATE = date(2026, 4, 30)
LAB_COLLECTION_DATE = date(2026, 4, 29)

# (test_name, value, unit, reference_range, abnormal_flag)
# Values reflect Cohen's chronic conditions: T2DM (HbA1c high), CKD3
# (eGFR low / Cr high / BUN high / K mildly high on ACEi), reasonable
# lipid control on atorvastatin.
LAB_TESTS: list[tuple[str, float, str, str, str | None]] = [
    ("Hemoglobin A1c",    7.4,  "%",          "4.0-5.6 %",       "H"),
    ("Fasting Glucose",   145,  "mg/dL",      "70-99 mg/dL",     "H"),
    ("Creatinine",        1.4,  "mg/dL",      "0.6-1.1 mg/dL",   "H"),
    ("eGFR",              52,   "mL/min/1.73m^2", ">60 mL/min/1.73m^2", "L"),
    ("BUN",               28,   "mg/dL",      "7-20 mg/dL",      "H"),
    ("Sodium",            138,  "mEq/L",      "135-145 mEq/L",   "N"),
    ("Potassium",         5.2,  "mEq/L",      "3.5-5.0 mEq/L",   "H"),
    ("LDL Cholesterol",   95,   "mg/dL",      "<100 mg/dL",      "N"),
    ("HDL Cholesterol",   38,   "mg/dL",      ">40 mg/dL",       "L"),
    ("Total Cholesterol", 175,  "mg/dL",      "<200 mg/dL",      "N"),
]


def _lab_citation(test_name: str, value_str: str) -> Citation:
    """Build the Citation that the extractor SHOULD return for this row.

    `field_or_chunk_id` follows a predictable convention so future tests can
    assert the extractor used the same convention.
    """
    return Citation(
        source_type="lab_pdf",
        source_id=LAB_DOC_ID,
        page_or_section="page 1",
        field_or_chunk_id=f"results_table.{test_name.lower().replace(' ', '_')}",
        quote_or_value=value_str,
    )


def write_lab_pdf(path: Path) -> None:
    """Render the lab PDF with a header, patient block, results table, footer."""
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "Title", parent=styles["Heading1"], fontSize=18, spaceAfter=6,
    )
    subtitle_style = ParagraphStyle(
        "Subtitle", parent=styles["Normal"], fontSize=11, textColor=colors.grey,
        spaceAfter=12,
    )
    section_style = ParagraphStyle(
        "Section", parent=styles["Heading2"], fontSize=12,
        spaceBefore=12, spaceAfter=6,
    )

    story: list[object] = [
        Paragraph(LAB_FACILITY, title_style),
        Paragraph(
            f"Laboratory Report — Issued {LAB_REPORT_DATE:%B %d, %Y}",
            subtitle_style,
        ),

        Paragraph("Patient Information", section_style),
        Table(
            [
                ["Patient Name:", "Nora Cohen"],
                ["Date of Birth:", "1958-06-22"],
                ["Sex:",           "Female"],
                ["MRN:",           "COHEN-0008"],
                ["Collection Date:", LAB_COLLECTION_DATE.isoformat()],
                ["Ordering Provider:", LAB_PROVIDER],
            ],
            colWidths=[1.6 * inch, 4.0 * inch],
            style=TableStyle([
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]),
        ),

        Paragraph("Results", section_style),
        Table(
            [["Test", "Value", "Unit", "Reference Range", "Flag"]] + [
                [test, str(val), unit, ref_range, (flag or "")]
                for test, val, unit, ref_range, flag in LAB_TESTS
            ],
            colWidths=[2.1 * inch, 0.8 * inch, 1.0 * inch, 1.8 * inch, 0.5 * inch],
            style=TableStyle([
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("ALIGN", (1, 1), (1, -1), "RIGHT"),
                ("ALIGN", (-1, 1), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
            ]),
        ),

        Spacer(1, 0.4 * inch),
        Paragraph(
            "Reference ranges are method-specific. Flag legend: "
            "H = high, L = low, N = normal, C = critical.",
            ParagraphStyle("Footer", parent=styles["Normal"], fontSize=8,
                          textColor=colors.grey),
        ),
    ]

    SimpleDocTemplate(
        str(path), pagesize=LETTER,
        leftMargin=0.75 * inch, rightMargin=0.75 * inch,
        topMargin=0.75 * inch, bottomMargin=0.75 * inch,
    ).build(story)


def lab_fixture() -> LabReport:
    """Build the schema-valid `LabReport` matching what the lab PDF prints."""
    return LabReport(
        results=[
            LabResult(
                test_name=test,
                value=value,
                unit=unit,
                reference_range=ref_range,
                collection_date=LAB_COLLECTION_DATE,
                abnormal_flag=flag,  # type: ignore[arg-type]
                source_citation=_lab_citation(test, str(value)),
            )
            for test, value, unit, ref_range, flag in LAB_TESTS
        ],
        source_document_id=LAB_DOC_ID,
        facility=LAB_FACILITY,
        ordering_provider=LAB_PROVIDER,
    )


# ──────────────────────────────────────────────────────────────────────────
# Intake form PDF
# ──────────────────────────────────────────────────────────────────────────

INTAKE_CLINIC = "Sunrise Family Medicine"
INTAKE_DATE = date(2026, 4, 30)


def _intake_citation(field: str, value: str) -> Citation:
    return Citation(
        source_type="intake_form",
        source_id=INTAKE_DOC_ID,
        page_or_section="page 1",
        field_or_chunk_id=field,
        quote_or_value=value,
    )


# (relation, condition, age_at_onset)
FAMILY_HISTORY: list[tuple[str, str, int | None]] = [
    ("Mother", "Type 2 diabetes mellitus", 55),
    ("Father", "Myocardial infarction",     62),
    ("Sister", "Hypertension",              None),
]

# (substance, reaction, severity)
ALLERGIES: list[tuple[str, str, str]] = [
    ("Penicillin",  "Hives", "moderate"),
    ("Sulfa drugs", "Rash",  "mild"),
]

# (name, dose, frequency, route)
MEDICATIONS: list[tuple[str, str, str, str]] = [
    ("Metformin",    "1000 mg", "BID",   "PO"),
    ("Lisinopril",   "20 mg",   "daily", "PO"),
    ("Apixaban",     "5 mg",    "BID",   "PO"),
    ("Atorvastatin", "40 mg",   "QHS",   "PO"),
]

CHIEF_CONCERN = (
    "Annual follow-up for diabetes, hypertension, and kidney function. "
    "Patient reports increased fatigue over past month."
)


def write_intake_pdf(path: Path) -> None:
    """Render the intake-form PDF with patient info, chief concern, and lists."""
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "Title", parent=styles["Heading1"], fontSize=18, spaceAfter=6,
    )
    subtitle_style = ParagraphStyle(
        "Subtitle", parent=styles["Normal"], fontSize=11, textColor=colors.grey,
        spaceAfter=12,
    )
    section_style = ParagraphStyle(
        "Section", parent=styles["Heading2"], fontSize=12,
        spaceBefore=12, spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "Body", parent=styles["Normal"], fontSize=10, leading=14,
    )

    story: list[object] = [
        Paragraph(INTAKE_CLINIC, title_style),
        Paragraph(
            f"Annual Visit Intake Form — {INTAKE_DATE:%B %d, %Y}",
            subtitle_style,
        ),

        Paragraph("Patient Information", section_style),
        Table(
            [
                ["Given Name:",  "Nora"],
                ["Family Name:", "Cohen"],
                ["Date of Birth:", "1958-06-22"],
                ["Sex:",          "Female"],
                ["Address:",      "1234 Maple Street, Austin, TX 78701"],
                ["Phone:",        "(512) 555-0142"],
            ],
            colWidths=[1.6 * inch, 4.5 * inch],
            style=TableStyle([
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]),
        ),

        Paragraph("Chief Concern", section_style),
        Paragraph(CHIEF_CONCERN, body_style),

        Paragraph("Current Medications", section_style),
        Table(
            [["Medication", "Dose", "Frequency", "Route"]] + [
                [name, dose, freq, route]
                for name, dose, freq, route in MEDICATIONS
            ],
            colWidths=[2.0 * inch, 1.0 * inch, 1.0 * inch, 1.0 * inch],
            style=TableStyle([
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
            ]),
        ),

        Paragraph("Allergies", section_style),
        Table(
            [["Substance", "Reaction", "Severity"]] + [
                [substance, reaction, severity]
                for substance, reaction, severity in ALLERGIES
            ],
            colWidths=[1.8 * inch, 2.0 * inch, 1.2 * inch],
            style=TableStyle([
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
            ]),
        ),

        Paragraph("Family History", section_style),
        Table(
            [["Relation", "Condition", "Age at Onset"]] + [
                [relation, condition, (str(age) if age is not None else "")]
                for relation, condition, age in FAMILY_HISTORY
            ],
            colWidths=[1.4 * inch, 3.2 * inch, 1.2 * inch],
            style=TableStyle([
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
            ]),
        ),
    ]

    SimpleDocTemplate(
        str(path), pagesize=LETTER,
        leftMargin=0.75 * inch, rightMargin=0.75 * inch,
        topMargin=0.75 * inch, bottomMargin=0.75 * inch,
    ).build(story)


def intake_fixture() -> IntakeForm:
    """Build the schema-valid `IntakeForm` matching what the intake PDF prints."""
    return IntakeForm(
        demographics=Demographics(
            given_name="Nora",
            family_name="Cohen",
            date_of_birth=date(1958, 6, 22),
            sex="female",
            address="1234 Maple Street, Austin, TX 78701",
            phone="(512) 555-0142",
            source_citation=_intake_citation("demographics", "Nora Cohen / 1958-06-22"),
        ),
        chief_concern=CHIEF_CONCERN,
        current_medications=[
            Medication(
                name=name,
                dose=dose,
                frequency=freq,
                route=route,
                source_citation=_intake_citation(
                    f"medications.{i}", f"{name} {dose} {route} {freq}",
                ),
            )
            for i, (name, dose, freq, route) in enumerate(MEDICATIONS)
        ],
        allergies=[
            Allergy(
                substance=substance,
                reaction=reaction,
                severity=severity,  # type: ignore[arg-type]
                source_citation=_intake_citation(
                    f"allergies.{i}", f"{substance} - {reaction}",
                ),
            )
            for i, (substance, reaction, severity) in enumerate(ALLERGIES)
        ],
        family_history=[
            FamilyHistoryItem(
                relation=relation,
                condition=condition,
                age_at_onset=age,
                source_citation=_intake_citation(
                    f"family_history.{i}", f"{relation}: {condition}",
                ),
            )
            for i, (relation, condition, age) in enumerate(FAMILY_HISTORY)
        ],
        source_document_id=INTAKE_DOC_ID,
    )


# ──────────────────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────────────────


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

    lab_pdf = OUT_DIR / "cohen_lab_2026-04-30.pdf"
    intake_pdf = OUT_DIR / "cohen_intake_2026-04-30.pdf"
    lab_json = FIXTURE_DIR / "cohen_lab_2026-04-30.expected.json"
    intake_json = FIXTURE_DIR / "cohen_intake_2026-04-30.expected.json"

    write_lab_pdf(lab_pdf)
    write_intake_pdf(intake_pdf)

    # Use Pydantic's own JSON serializer so the fixtures round-trip cleanly
    # back into the schema (dates become "YYYY-MM-DD" strings, etc.).
    lab_json.write_text(lab_fixture().model_dump_json(indent=2) + "\n")
    intake_json.write_text(intake_fixture().model_dump_json(indent=2) + "\n")

    for path in (lab_pdf, intake_pdf, lab_json, intake_json):
        rel = path.relative_to(REPO_ROOT)
        size_kb = path.stat().st_size / 1024
        print(f"  wrote {rel}  ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
