"""Generate a synthetic referral letter PDF for Margaret L. Chen.

Output: `data/demo_documents/real/p01-chen-referral-letter.pdf`.

Why this exists: Chen has an intake form and a lab PDF on file but no
referral letter, which means her Care Team tab on the Modern Dashboard
has no `ReferringPhysician` to surface. This script creates a
plausible-looking referral letter from a Cardiology practice so the
Phase 2 VLM pipeline has something to extract — `name`, `practice`,
`specialty`, `phone`, `address`, `npi` all present and clearly placed
in the signature block + letterhead.

The letter content matches Chen's known clinical context (the lipid
panel is on file; the referral asks the receiving practice to manage
her lipid disorder + assess CV risk) so the rest of the extraction
(reason_for_referral, PMH, current meds, allergies) is also coherent.

Run: `cd clinical-copilot && uv run python scripts/generate_chen_referral.py`
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
)


OUTPUT_PATH = (
    Path(__file__).resolve().parent.parent
    / "data" / "demo_documents" / "real" / "p01-chen-referral-letter.pdf"
)


def build() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUTPUT_PATH), pagesize=LETTER,
        leftMargin=0.9 * inch, rightMargin=0.9 * inch,
        topMargin=0.8 * inch, bottomMargin=0.8 * inch,
        title="Referral Letter - Margaret L. Chen",
        author="Bay Area Cardiology Associates",
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
    label = ParagraphStyle(
        name="Label", parent=body, fontName="Helvetica-Bold",
    )
    sig = ParagraphStyle(
        name="Sig", parent=body, fontName="Helvetica",
    )

    story: list = []

    story.append(Paragraph("Bay Area Cardiology Associates", styles["Letterhead"]))
    story.append(Paragraph(
        "2150 Shattuck Avenue, Suite 800 &nbsp;&middot;&nbsp; Berkeley, CA 94704",
        styles["LetterheadSub"],
    ))
    story.append(Paragraph(
        "Phone: (510) 555-0192 &nbsp;&middot;&nbsp; Fax: (510) 555-0193",
        styles["LetterheadSub"],
    ))
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=0.6, color="#cccccc"))
    story.append(Spacer(1, 12))

    story.append(Paragraph("April 28, 2026", body))
    story.append(Spacer(1, 8))

    story.append(Paragraph("To: Internal Medicine, East Bay Primary Care", body))
    story.append(Paragraph(
        "<b>Re:</b> Margaret L. Chen &nbsp;&middot;&nbsp; <b>DOB:</b> 1967-08-14 "
        "&nbsp;&middot;&nbsp; <b>Sex:</b> Female",
        body,
    ))
    story.append(Spacer(1, 8))

    story.append(Paragraph("Dear Colleague,", body))

    story.append(Paragraph(
        "<b>Reason for Referral:</b> Mixed dyslipidemia with elevated "
        "LDL-C and reduced HDL-C on routine lipid panel; please "
        "co-manage lipid-lowering therapy and assess 10-year ASCVD risk.",
        body,
    ))

    story.append(Paragraph(
        "<b>History of Present Illness:</b> Ms. Chen is a 58-year-old "
        "woman seen in our office for cardiovascular risk assessment "
        "following her primary-care lipid panel of 2026-04-12 "
        "(LDL-C 161 mg/dL, HDL-C 38 mg/dL, triglycerides 198 mg/dL, "
        "total cholesterol 232 mg/dL). She is asymptomatic, with no "
        "chest pain, dyspnea on exertion, palpitations, or syncope. "
        "Family history is notable for a paternal MI at age 62. "
        "She walks 30 minutes most days and follows a Mediterranean-"
        "style diet. Current BP in clinic 132/84.",
        body,
    ))

    story.append(Paragraph(
        "<b>Past Medical History:</b>", label,
    ))
    story.append(Paragraph(
        "&middot; Type 2 diabetes mellitus without complications (E11.9)<br/>"
        "&middot; Essential hypertension (I10)<br/>"
        "&middot; Mixed hyperlipidemia (E78.2)",
        body,
    ))

    story.append(Paragraph(
        "<b>Current Medications:</b>", label,
    ))
    story.append(Paragraph(
        "&middot; Metformin 500 mg PO BID<br/>"
        "&middot; Lisinopril 10 mg PO daily<br/>"
        "&middot; Aspirin 81 mg PO daily",
        body,
    ))

    story.append(Paragraph(
        "<b>Allergies:</b> Penicillin (rash, moderate)",
        body,
    ))

    story.append(Paragraph(
        "<b>Specific Question / Requested Action:</b> Initiate "
        "moderate-intensity statin therapy and follow-up lipid panel "
        "in 12 weeks. Please reach out with any questions regarding "
        "shared management.",
        body,
    ))

    story.append(Spacer(1, 16))
    story.append(Paragraph("Sincerely,", body))
    story.append(Spacer(1, 28))
    story.append(Paragraph("<b>Helen Park, MD</b>", sig))
    story.append(Paragraph("Cardiology", sig))
    story.append(Paragraph("Bay Area Cardiology Associates", sig))
    story.append(Paragraph("2150 Shattuck Avenue, Suite 800, Berkeley, CA 94704", sig))
    story.append(Paragraph("Phone: (510) 555-0192", sig))
    story.append(Paragraph("NPI: 1538291746", sig))

    doc.build(story)
    print(f"wrote {OUTPUT_PATH} ({OUTPUT_PATH.stat().st_size} bytes)")


if __name__ == "__main__":
    build()
