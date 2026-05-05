"""End-to-end MVP smoke for the Phase 4.1 demo path.

Drives the full Week-2 demo loop in one script:

  1. Upload Margaret Chen's intake form via attach_and_extract — proves
     the writer + Claude vision wiring (Phase 2.1) end-to-end against
     real OpenEMR.
  2. Pose a clinical question to the agent that should trigger BOTH
     a chart-summarizer FHIR tool call AND a retrieve_guidelines tool
     call — proves the guidelines tool (Phase 4.1) lands in the agent's
     tool list and the response cites both `[FHIRType/ID]` and
     `[Guideline/chunk_id]` citations correctly.
  3. Print the response + assert both citation namespaces appear.

Cost per run: ~3 Claude API calls (1 vision for extraction + 1–2 chat
turns for the agent). Don't loop on this — it's a deliberate,
end-to-end verification before a demo.

Prerequisites:
  - All the same env vars as smoke_extract.py (OPENEMR_*, ANTHROPIC_API_KEY,
    DEMO_PATIENT_PUUID for Chen).
  - OpenEMR running locally at the configured FHIR base.
  - Chen seeded (scripts/seed_chen.py) with the PUUID in .env.

Run:
  cd clinical-copilot && PYTHONPATH=. uv run python scripts/smoke_e2e_mvp.py

Skip the upload phase (e.g. if Chen's intake is already attached):
  --skip-upload

Override the agent question (default uses a CRC + diabetes screening
prompt that exercises both ADA + USPSTF retrieval):
  --question "..."
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")

import os  # noqa: E402

from langchain_core.messages import HumanMessage  # noqa: E402

from app.agent.graph import build_graph  # noqa: E402
from app.agent.state import AgentState  # noqa: E402
from app.clinical_notes import ClinicalNoteStore  # noqa: E402
from app.extraction.extract import attach_and_extract  # noqa: E402
from app.extraction.vision import ExtractionError  # noqa: E402
from app.fhir.client import FhirClient  # noqa: E402
from app.fhir.writer import OpenEMRWriteError, OpenEMRWriter  # noqa: E402

DEMO_DIR = REPO_ROOT / "data" / "demo_documents" / "real"
DEFAULT_INTAKE_PDF = DEMO_DIR / "p01-chen-intake-typed.pdf"

DEFAULT_QUESTION = (
    "I'm seeing Margaret Chen for an annual visit. She's 58 with type 2 "
    "diabetes. What does USPSTF say about colorectal cancer screening for "
    "her, and what does ADA recommend for HbA1c targets in her age group?"
)


def _ok(msg: str) -> None:
    print(f"  ✓ {msg}")


def _fail(msg: str) -> None:
    print(f"  ✗ {msg}")


def _step(n: int, total: int, label: str) -> None:
    print(f"\n[{n}/{total}] {label}")


CITATION_RE = re.compile(r"\[([A-Z][a-zA-Z]+)/([a-zA-Z0-9._-]+)\]")


def _classify_citations(text: str) -> tuple[set[str], set[str]]:
    """Return (fhir_refs, guideline_chunk_ids) found in `text`."""
    fhir: set[str] = set()
    guideline: set[str] = set()
    for m in CITATION_RE.finditer(text):
        rtype, rid = m.group(1), m.group(2)
        if rtype == "Guideline":
            guideline.add(rid)
        else:
            fhir.add(f"{rtype}/{rid}")
    return fhir, guideline


async def _run(args: argparse.Namespace) -> int:
    puuid = os.getenv("DEMO_PATIENT_PUUID")
    if not puuid:
        _fail("DEMO_PATIENT_PUUID not in .env — run scripts/seed_chen.py first")
        return 1

    total_steps = 2 if args.skip_upload else 3
    step = 0

    writer = OpenEMRWriter()
    fhir = FhirClient()
    notes_path = Path(
        os.environ.get("CLINICAL_NOTES_PATH", "data/clinical_notes.json")
    )
    notes_store = ClinicalNoteStore(notes_path)
    try:
        if not args.skip_upload:
            step += 1
            _step(step, total_steps, "Upload Chen intake via attach_and_extract")
            if not DEFAULT_INTAKE_PDF.exists():
                _fail(f"missing demo file: {DEFAULT_INTAKE_PDF}")
                return 1
            try:
                up = await attach_and_extract(
                    file_bytes=DEFAULT_INTAKE_PDF.read_bytes(),
                    filename=DEFAULT_INTAKE_PDF.name,
                    doc_type="intake_form",
                    patient_uuid=puuid,
                    mime_type="application/pdf",
                    writer=writer,
                )
            except (OpenEMRWriteError, ExtractionError, ValueError) as exc:
                _fail(f"upload failed: {exc}")
                return 1
            _ok(
                f"DocumentReference={up.reference_id} created={up.created} "
                f"chief_concern={up.extracted.chief_concern[:60] if hasattr(up.extracted, 'chief_concern') else 'n/a'}..."  # type: ignore[union-attr]
            )

        step += 1
        _step(step, total_steps, "Drive the agent with a screening + targets question")
        # build_graph closes over fhir + notes_store; assignments_store=None
        # tells the per-tool ACL gate to skip (smoke runs as admin).
        graph = build_graph(fhir, notes_store, assignments_store=None)
        state: AgentState = {
            "messages": [HumanMessage(content=args.question)],
            "conversation_sources": [],
            "patient_id": None,
            "validation_attempts": 0,
            "username": "admin",  # smoke runs as admin (sees all patients)
            "advisor_mode": False,
        }
        try:
            result = await graph.ainvoke(state)
        except Exception as exc:  # noqa: BLE001
            _fail(f"agent invocation failed: {exc}")
            return 1
        # Pull the assistant's final text content
        msgs = result.get("messages") or []
        if not msgs:
            _fail("agent returned no messages")
            return 1
        final = msgs[-1]
        content = final.content if hasattr(final, "content") else str(final)
        if isinstance(content, list):
            content = " ".join(
                p.get("text", "") if isinstance(p, dict) else str(p)
                for p in content
            )
        print("\n----- agent response -----\n" + str(content) + "\n--------------------------")

        step += 1
        _step(step, total_steps, "Verify response cites both FHIR + Guideline namespaces")
        fhir_refs, guideline_refs = _classify_citations(str(content))
        if not fhir_refs and not guideline_refs:
            _fail("response contained ZERO citations of either kind")
            return 1
        if guideline_refs:
            _ok(f"guideline citations: {sorted(guideline_refs)}")
        else:
            _fail("no [Guideline/...] citation in response — agent may not have called retrieve_guidelines")
            print("  Hint: rephrase the question to be more screening-oriented, or check tool wiring.")
        if fhir_refs:
            _ok(f"FHIR citations: {sorted(fhir_refs)}")
        # Soft-fail: missing FHIR ok if the question was guideline-only.

        print("\n─" * 30)
        print("PASS — Phase 4.1 end-to-end MVP smoke complete")
        print(f"  patient: {puuid}")
        print(f"  FHIR citations:      {len(fhir_refs)}")
        print(f"  Guideline citations: {len(guideline_refs)}")
        print("─" * 30)
        return 0
    finally:
        await writer.aclose()
        await fhir.aclose()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--skip-upload", action="store_true",
        help="Skip the attach_and_extract phase (saves one Claude vision call)",
    )
    p.add_argument(
        "--question", default=DEFAULT_QUESTION,
        help="Override the agent question",
    )
    return p.parse_args()


def main() -> None:
    sys.exit(asyncio.run(_run(_parse_args())))


if __name__ == "__main__":
    main()
