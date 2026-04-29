"""Render ARCHITECTURE.md to a visual PDF.

Generates two diagrams (architecture overview + verification flow) as PNGs
with matplotlib, then assembles ARCHITECTURE.pdf with reportlab.

Run: uv run python scripts/render_architecture_pdf.py
"""

import re
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# Repo root is one level above clinical-copilot/, where ARCHITECTURE.md lives
# alongside USERS.md and AUDIT.md per the brief's "at the root of the repo"
# submission requirement.
ROOT = Path(__file__).resolve().parents[2]
ARCH_MD = ROOT / "ARCHITECTURE.md"
OUT_PDF = ROOT / "ARCHITECTURE.pdf"
DIAGRAM_DIR = ROOT / "build" / "diagrams"
DIAGRAM_DIR.mkdir(parents=True, exist_ok=True)


# ─── Diagram 1: System Architecture ───────────────────────────────────────


def draw_architecture_diagram(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 9))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    layer_color = "#E8F0FE"
    sub_color = "#FFFFFF"
    accent = "#1A73E8"
    arrow_color = "#5F6368"
    text_color = "#202124"

    def box(x, y, w, h, label, fill=sub_color, edge=accent, fontsize=10, weight="normal"):
        rect = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.3",
            linewidth=1.5, edgecolor=edge, facecolor=fill,
        )
        ax.add_patch(rect)
        ax.text(
            x + w / 2, y + h / 2, label,
            ha="center", va="center", fontsize=fontsize,
            fontweight=weight, color=text_color, wrap=True,
        )

    def arrow(x1, y1, x2, y2, label=None):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", color=arrow_color, lw=1.4),
        )
        if label:
            ax.text((x1 + x2) / 2 + 1, (y1 + y2) / 2,
                    label, fontsize=8, color=arrow_color, style="italic")

    # User
    box(35, 92, 30, 6, "Hospitalist physician (browser)",
        fill="#FEF7E0", edge="#F9AB00", fontsize=11, weight="bold")

    # Layer container — UI
    box(10, 78, 80, 10, "", fill=layer_color, edge=accent)
    ax.text(11, 86, "UI", fontsize=9, color=accent, fontweight="bold")
    box(13, 80, 22, 6, "Patient list", fontsize=9)
    box(38, 80, 24, 6, "Chat (streaming)", fontsize=9)
    box(65, 80, 22, 6, "Patient card", fontsize=9)

    # Layer container — FastAPI
    box(5, 32, 90, 42, "", fill=layer_color, edge=accent)
    ax.text(6, 72, "FastAPI backend", fontsize=10, color=accent, fontweight="bold")

    box(8, 64, 84, 6, "Auth: session → user → role (physician/nurse/resident)",
        fontsize=9, weight="bold")

    # LangGraph state machine
    box(8, 46, 84, 16, "", fill="#F1F3F4", edge=accent)
    ax.text(9, 60, "LangGraph state machine", fontsize=9,
            color=accent, fontweight="bold")
    nodes = [
        ("resolve\npatient", 12, 49),
        ("call\nLLM", 28, 49),
        ("tool\ncalls", 44, 49),
        ("validate\ncitations", 60, 49),
        ("stream\nto user", 76, 49),
    ]
    for label, x, y in nodes:
        box(x, y, 12, 7, label, fontsize=8, fill="#FFFFFF")
    for i in range(len(nodes) - 1):
        x1 = nodes[i][1] + 12
        x2 = nodes[i + 1][1]
        arrow(x1, nodes[i][2] + 3.5, x2, nodes[i + 1][2] + 3.5)

    # Tool layer
    box(8, 38, 84, 6,
        "Tool layer: 7 tools, each returns {data, sources: [resource_type/id, ...]}",
        fontsize=9, weight="bold")

    # FHIR adapter
    box(8, 32, 84, 4,
        "FHIR adapter — OAuth2 + authz check + audit log write per call",
        fontsize=9, fill="#FCE8E6", edge="#D93025")

    # OpenEMR
    box(15, 18, 70, 10,
        "OpenEMR FHIR R4  —  /apis/default/fhir/  (Patient, Encounter, "
        "Observation, MedicationRequest,\nCondition, AllergyIntolerance, "
        "DocumentReference, Practitioner)",
        fontsize=9, weight="bold", fill="#E6F4EA", edge="#188038")

    # Side cluster — cross-cutting
    box(2, 4, 96, 10, "", fill="#FFFFFF", edge="#9AA0A6")
    ax.text(3, 12, "Cross-cutting", fontsize=9, color="#5F6368", fontweight="bold")
    box(5, 6, 28, 5, "LangSmith — every node + tool + token traced",
        fontsize=8, fill="#FFFFFF", edge="#9AA0A6")
    box(36, 6, 28, 5, "Postgres (Fly.io) — sessions + audit log",
        fontsize=8, fill="#FFFFFF", edge="#9AA0A6")
    box(67, 6, 28, 5, "Anthropic prompt cache — system + per-patient",
        fontsize=8, fill="#FFFFFF", edge="#9AA0A6")

    # Top arrow user → UI
    arrow(50, 91, 50, 88, label="HTTPS")
    # UI → FastAPI
    arrow(50, 78, 50, 74, label="session cookie")
    # FastAPI → OpenEMR
    arrow(50, 32, 50, 28, label="system OAuth2 token")

    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ─── Diagram 2: Verification Flow ─────────────────────────────────────────


def draw_verification_diagram(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    accent = "#1A73E8"
    danger = "#D93025"
    ok = "#188038"

    def box(x, y, w, h, label, fontsize=10, fill="#FFFFFF", edge=accent, weight="normal"):
        rect = patches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.3",
            linewidth=1.5, edgecolor=edge, facecolor=fill,
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=fontsize, fontweight=weight, color="#202124")

    def arrow(x1, y1, x2, y2, label=None, color="#5F6368"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.4))
        if label:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 2, label,
                    fontsize=9, color=color, ha="center", style="italic")

    # User input
    box(2, 80, 22, 12, "User asks:\n'catch me up on bed 412'",
        fontsize=10, fill="#FEF7E0", edge="#F9AB00")

    # LLM call
    box(38, 80, 22, 12, "LLM (Sonnet 4.6)\nplans tool calls",
        fontsize=10, weight="bold")

    # Tools
    box(72, 80, 26, 12,
        "Tools fetch FHIR data;\neach returns sources=[Observation/8821, ...]",
        fontsize=9, fill="#E6F4EA", edge=ok)

    arrow(24, 86, 38, 86)
    arrow(60, 86, 72, 86)

    # Back to LLM
    box(38, 56, 22, 12,
        "LLM drafts response\nwith {claim, sources} per fact",
        fontsize=10, weight="bold")
    arrow(85, 80, 60, 68, label="tool results")

    # Validator
    box(38, 32, 22, 12, "Citation Validator\n(LangGraph node)",
        fontsize=10, fill="#E8F0FE", edge=accent, weight="bold")
    arrow(49, 56, 49, 44)

    # Branch — pass / fail
    box(8, 8, 22, 12, "✓ Every claim cites\na valid tool source\n→ stream to user",
        fontsize=9, fill="#E6F4EA", edge=ok)
    box(68, 8, 26, 12,
        "✗ Uncited or invalid claim\n→ ask LLM to re-state\nwith valid sources",
        fontsize=9, fill="#FCE8E6", edge=danger)

    arrow(45, 32, 25, 20, label="pass", color=ok)
    arrow(53, 32, 75, 20, label="fail", color=danger)
    arrow(81, 20, 49, 56, color=danger)

    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ─── Markdown → PDF flowables ─────────────────────────────────────────────


def build_styles():
    base = getSampleStyleSheet()
    s = {
        "h1": ParagraphStyle("h1", parent=base["Heading1"], fontSize=22,
                             spaceAfter=12, textColor=colors.HexColor("#1A73E8")),
        "h2": ParagraphStyle("h2", parent=base["Heading2"], fontSize=16,
                             spaceBefore=14, spaceAfter=8,
                             textColor=colors.HexColor("#202124")),
        "h3": ParagraphStyle("h3", parent=base["Heading3"], fontSize=12,
                             spaceBefore=8, spaceAfter=4,
                             textColor=colors.HexColor("#5F6368")),
        "body": ParagraphStyle("body", parent=base["BodyText"], fontSize=10,
                               leading=14, alignment=TA_LEFT, spaceAfter=6),
        "bullet": ParagraphStyle("bullet", parent=base["BodyText"], fontSize=10,
                                 leading=14, leftIndent=18, bulletIndent=6,
                                 spaceAfter=2),
        "code": ParagraphStyle("code", parent=base["Code"], fontSize=8.5,
                               leading=11, backColor=colors.HexColor("#F1F3F4"),
                               borderPadding=6, leftIndent=4, spaceAfter=8),
        "caption": ParagraphStyle("caption", parent=base["BodyText"], fontSize=9,
                                  textColor=colors.HexColor("#5F6368"),
                                  alignment=1, spaceAfter=12),
    }
    return s


_INLINE_BOLD = re.compile(r"\*\*([^*]+)\*\*")
_INLINE_ITALIC = re.compile(r"(?<!\*)\*([^*]+)\*(?!\*)")
_INLINE_CODE = re.compile(r"`([^`]+)`")
_INLINE_LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


def inline_md_to_html(text: str) -> str:
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = _INLINE_BOLD.sub(r"<b>\1</b>", text)
    text = _INLINE_ITALIC.sub(r"<i>\1</i>", text)
    text = _INLINE_CODE.sub(
        r'<font face="Courier" backColor="#F1F3F4">\1</font>', text
    )
    text = _INLINE_LINK.sub(
        r'<font color="#1A73E8"><u>\1</u></font>', text
    )
    return text


def parse_table(lines, i):
    """Parse a markdown table starting at lines[i]. Returns (table_data, next_i)."""
    rows = []
    while i < len(lines) and lines[i].strip().startswith("|"):
        row = [c.strip() for c in lines[i].strip().strip("|").split("|")]
        rows.append(row)
        i += 1
    # Drop the separator row (---|---|---)
    rows = [r for r in rows if not all(re.match(r"^:?-+:?$", c) for c in r)]
    return rows, i


def md_to_flowables(md_text: str, styles: dict, diagrams: dict[str, Path]):
    flow = []
    lines = md_text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Diagram replacement marker (for ASCII block in §3)
        if "┌─────" in line or "│" in line and stripped.startswith("│") and "▼" not in stripped:
            # Skip the entire ASCII art block
            while i < len(lines) and (
                "┌" in lines[i] or "│" in lines[i] or "└" in lines[i]
                or "▼" in lines[i] or lines[i].strip().startswith(("Cross", "•"))
                or lines[i].strip() == ""
                and i + 1 < len(lines) and ("│" in lines[i + 1] or "└" in lines[i + 1])
            ):
                i += 1
            continue

        # Headers
        if stripped.startswith("# "):
            flow.append(Paragraph(inline_md_to_html(stripped[2:]), styles["h1"]))
            i += 1
            continue
        if stripped.startswith("## "):
            heading = stripped[3:]
            flow.append(Paragraph(inline_md_to_html(heading), styles["h2"]))
            # Insert architecture diagram after section 3 heading
            if heading.startswith("3. High-level architecture"):
                flow.append(Spacer(1, 0.1 * inch))
                flow.append(Image(str(diagrams["architecture"]), width=6.5 * inch, height=5.3 * inch))
                flow.append(Paragraph("Figure 1 — System architecture overview", styles["caption"]))
            elif heading.startswith("4.4 Verification") or heading.startswith("4. Layer walkthrough"):
                pass
            i += 1
            continue
        if stripped.startswith("### "):
            heading = stripped[4:]
            flow.append(Paragraph(inline_md_to_html(heading), styles["h3"]))
            # Insert verification diagram after the verification subsection
            if "Verification" in heading:
                flow.append(Spacer(1, 0.1 * inch))
                flow.append(Image(str(diagrams["verification"]), width=6.5 * inch, height=3.5 * inch))
                flow.append(Paragraph("Figure 2 — Verification flow with citation validator", styles["caption"]))
            i += 1
            continue

        # Code block
        if stripped.startswith("```"):
            i += 1
            buf = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                buf.append(lines[i])
                i += 1
            code = "<br/>".join(
                line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace(" ", "&nbsp;")
                for line in buf
            )
            flow.append(Paragraph(code, styles["code"]))
            i += 1
            continue

        # Table
        if stripped.startswith("|"):
            rows, i = parse_table(lines, i)
            if rows:
                wrapped = [
                    [Paragraph(inline_md_to_html(c), styles["body"]) for c in r]
                    for r in rows
                ]
                t = Table(wrapped, repeatRows=1, hAlign="LEFT")
                t.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8F0FE")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#1A73E8")),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#DADCE0")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                ]))
                flow.append(t)
                flow.append(Spacer(1, 0.1 * inch))
            continue

        # Bullets
        if stripped.startswith(("- ", "* ")):
            flow.append(Paragraph(
                "• " + inline_md_to_html(stripped[2:]), styles["bullet"],
            ))
            i += 1
            continue
        if re.match(r"^\d+\.\s", stripped):
            flow.append(Paragraph(
                inline_md_to_html(stripped), styles["bullet"],
            ))
            i += 1
            continue

        # Horizontal rule
        if stripped in ("---", "***", "___"):
            flow.append(Spacer(1, 0.15 * inch))
            i += 1
            continue

        # Italic line meta (Status: ..., Date: ...)
        if stripped.startswith("_") and stripped.endswith("_"):
            flow.append(Paragraph(
                f'<i>{inline_md_to_html(stripped[1:-1])}</i>',
                styles["body"],
            ))
            i += 1
            continue

        # Empty line
        if not stripped:
            i += 1
            continue

        # Paragraph (collect contiguous non-empty non-special lines)
        para_lines = [stripped]
        i += 1
        while i < len(lines):
            nxt = lines[i].strip()
            if not nxt or nxt.startswith(("#", "-", "*", "|", "```", "_")) or re.match(r"^\d+\.\s", nxt):
                break
            para_lines.append(nxt)
            i += 1
        flow.append(Paragraph(inline_md_to_html(" ".join(para_lines)), styles["body"]))

    return flow


def main() -> None:
    print("Drawing diagrams...")
    arch_png = DIAGRAM_DIR / "architecture.png"
    verif_png = DIAGRAM_DIR / "verification.png"
    draw_architecture_diagram(arch_png)
    draw_verification_diagram(verif_png)
    print(f"  ✓ {arch_png}")
    print(f"  ✓ {verif_png}")

    print("Building PDF...")
    md = ARCH_MD.read_text()
    styles = build_styles()
    flow = md_to_flowables(md, styles, {"architecture": arch_png, "verification": verif_png})

    doc = SimpleDocTemplate(
        str(OUT_PDF), pagesize=LETTER,
        leftMargin=0.7 * inch, rightMargin=0.7 * inch,
        topMargin=0.7 * inch, bottomMargin=0.7 * inch,
        title="Architecture — Clinical Co-Pilot",
        author="agent_forge",
    )
    doc.build(flow)
    print(f"  ✓ {OUT_PDF}")


if __name__ == "__main__":
    main()
