"""Generate a synthetic follow-up lab panel PDF for Margaret L. Chen.

Output: `data/demo_documents/real/p01-chen-lipid-panel-followup.pdf`.

Why this exists: Chen's existing `p01-chen-lipid-panel.pdf` was
uploaded before the `extracted_lab_results` SQLite store existed.
SHA-256 dedup means re-uploading the same bytes won't trigger a fresh
extraction-and-persistence pass, so her existing labs would never
populate the new store.

This generator emits a NEW lab PDF (different bytes → different SHA →
no dedup hit) representing a 12-week follow-up panel after the
referral letter. Same patient, same general lipid theme, slightly
improved values that show the statin started by Cardiology is
working — clinically coherent.

Run: `cd clinical-copilot && uv run python scripts/generate_chen_lab_followup.py`
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    HRFlowable,
    Table,
    TableStyle,
)
from reportlab.lib import colors


OUTPUT_PATH = (
    Path(__file__).resolve().parent.parent
    / "data" / "demo_documents" / "real" / "p01-chen-lipid-panel-followup.pdf"
)


def build() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUTPUT_PATH), pagesize=LETTER,
        leftMargin=0.9 * inch, rightMargin=0.9 * inch,
        topMargin=0.8 * inch, bottomMargin=0.8 * inch,
        title="Lab Report - Margaret L. Chen - Lipid Panel Follow-Up",
        author="East Bay Reference Laboratory",
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="Letterhead", fontName="Helvetica-Bold", fontSize=14,
        alignment=1, spaceAfter=2,
    ))
    styles.add(ParagraphStyle(
        name="LetterheadSub", fontName="Helvetica", fontSize=10,
        alignment=1, textColor="#444444", spaceAfter=2,
    ))
    body = ParagraphStyle(
        name="Body", parent=styles["BodyText"], fontSize=11, leading=15,
        spaceAfter=8,
    )

    story: list = []

    story.append(Paragraph("East Bay Reference Laboratory", styles["Letterhead"]))
    story.append(Paragraph(
        "1450 Telegraph Avenue &nbsp;&middot;&nbsp; Oakland, CA 94612",
        styles["LetterheadSub"],
    ))
    story.append(Paragraph(
        "CLIA #: 05D2061798 &nbsp;&middot;&nbsp; Phone: (510) 555-0177",
        styles["LetterheadSub"],
    ))
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=0.6, color="#cccccc"))
    story.append(Spacer(1, 12))

    story.append(Paragraph("<b>LABORATORY REPORT</b>", body))
    story.append(Spacer(1, 4))

    story.append(Paragraph(
        "<b>Patient:</b> Margaret L. Chen &nbsp;&middot;&nbsp; "
        "<b>DOB:</b> 1967-08-14 &nbsp;&middot;&nbsp; "
        "<b>Sex:</b> Female &nbsp;&middot;&nbsp; "
        "<b>MRN:</b> CHEN-001",
        body,
    ))
    story.append(Paragraph(
        "<b>Collection Date:</b> 2026-05-09 &nbsp;&middot;&nbsp; "
        "<b>Report Date:</b> 2026-05-09 &nbsp;&middot;&nbsp; "
        "<b>Ordering Provider:</b> Helen Park, MD (Cardiology)",
        body,
    ))
    story.append(Paragraph(
        "<b>Specimen:</b> Serum, fasting 12 hours &nbsp;&middot;&nbsp; "
        "<b>Reason for Test:</b> 12-week follow-up post statin initiation",
        body,
    ))
    story.append(Spacer(1, 12))

    story.append(Paragraph("<b>Lipid Panel</b>", body))

    headers = ["Test", "Result", "Unit", "Reference Range", "Flag"]
    rows = [
        ["Total Cholesterol", "188", "mg/dL", "<200", ""],
        ["LDL Cholesterol",   "112", "mg/dL", "<100",         "H"],
        ["HDL Cholesterol",   "44",  "mg/dL", ">=40 (female)", ""],
        ["Triglycerides",     "152", "mg/dL", "<150",         "H"],
        ["Non-HDL Cholesterol", "144", "mg/dL", "<130",       "H"],
        ["Cholesterol/HDL Ratio", "4.3", "", "<5.0",          ""],
    ]
    table_data = [headers] + rows
    table = Table(table_data, colWidths=[2.0*inch, 0.9*inch, 0.9*inch, 1.6*inch, 0.5*inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef2f7")),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN",      (1, 1), (-1, -1), "LEFT"),
        ("BOX",        (0, 0), (-1, -1), 0.5, colors.HexColor("#999999")),
        ("INNERGRID",  (0, 0), (-1, -1), 0.25, colors.HexColor("#cccccc")),
        ("FONTSIZE",   (0, 0), (-1, -1), 10),
        ("LEFTPADDING",  (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING",   (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
    ]))
    story.append(table)
    story.append(Spacer(1, 14))

    story.append(Paragraph(
        "<b>Comments:</b> Compared to lipid panel of 2026-04-12 "
        "(LDL-C 161, HDL-C 38, Trig 198, Total 232): meaningful "
        "improvement in LDL-C and HDL-C consistent with response to "
        "moderate-intensity statin initiation. Triglycerides and "
        "non-HDL still mildly elevated — recommend continued lifestyle "
        "support and recheck in 12 weeks.",
        body,
    ))

    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "<i>Reviewed by: Anne Foster, MD &mdash; Laboratory Director, "
        "East Bay Reference Laboratory</i>",
        body,
    ))

    doc.build(story)
    print(f"wrote {OUTPUT_PATH} ({OUTPUT_PATH.stat().st_size} bytes)")


if __name__ == "__main__":
    build()
