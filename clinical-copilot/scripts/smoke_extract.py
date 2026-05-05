"""Smoke test for the Phase 2.1 attach_and_extract pipeline.

Drives the full end-to-end path against a running local OpenEMR AND a
real Anthropic API key. Each invocation costs roughly one Claude
vision call per document, so do not loop on this in CI — use it as
the verification step after a deliberate code change.

Two phases per document:

  1. Persist source via OpenEMRWriter.write_document_reference. Either
     creates a new DocumentReference or returns the existing dedupe id.
  2. Render the doc to PNG pages (pypdfium2 for PDFs, Pillow for PNGs)
     and extract via Claude vision tool-use into a strict-typed
     LabReport / IntakeForm.

Defaults to Margaret Chen's typed lipid panel + intake (the cleanest
real documents in `data/demo_documents/real/`). Override via env vars
or CLI args to smoke against any other patient.

Prerequisites (one-time per dev box):
  - `cd docker/development-easy && docker compose up --detach --wait`
  - `cd clinical-copilot && PYTHONPATH=. uv run python scripts/register_seed_client.py`
    (or `add_document_reference_scope.py` if the seed client already exists)
  - `cd clinical-copilot && PYTHONPATH=. uv run python scripts/seed_chen.py`
    → paste printed `DEMO_PATIENT_PUUID=...` into `.env`
  - `.env` populated with OPENEMR_SEED_CLIENT_ID/SECRET, OPENEMR_FHIR_BASE_URL,
    OPENEMR_OAUTH_TOKEN_URL, DEMO_PATIENT_PUUID, ANTHROPIC_API_KEY.

Run: `cd clinical-copilot && PYTHONPATH=. uv run python scripts/smoke_extract.py`

Override defaults:
  - `--puuid <uuid>` or DEMO_PATIENT_PUUID env var
  - `--lab-pdf <path>` and `--intake-pdf <path>` for non-Chen documents
  - `--lab-mime <mime>` and `--intake-mime <mime>` for PNG/non-PDF files
  - `--lab-only` / `--intake-only` to halve the API cost when iterating
  - `--model <id>` to override the Claude model (default sonnet 4.6)
"""

from __future__ import annotations

import argparse
import asyncio
import mimetypes
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")

import os  # noqa: E402

from app.extraction.extract import attach_and_extract  # noqa: E402
from app.extraction.vision import DEFAULT_MODEL, ExtractionError  # noqa: E402
from app.fhir.writer import OpenEMRWriteError, OpenEMRWriter  # noqa: E402

DEMO_DIR = REPO_ROOT / "data" / "demo_documents" / "real"
DEFAULT_LAB_PDF = DEMO_DIR / "p01-chen-lipid-panel.pdf"
DEFAULT_INTAKE_PDF = DEMO_DIR / "p01-chen-intake-typed.pdf"


def _print_step(step: int, total: int, label: str) -> None:
    print(f"\n[{step}/{total}] {label}")


def _ok(msg: str) -> None:
    print(f"  ✓ {msg}")


def _fail(msg: str) -> None:
    print(f"  ✗ {msg}")


def _guess_mime(path: Path, override: str | None) -> str:
    if override:
        return override
    guess, _ = mimetypes.guess_type(str(path))
    return guess or "application/octet-stream"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--puuid",
        default=os.getenv("DEMO_PATIENT_PUUID"),
        help="OpenEMR Patient UUID. Defaults to DEMO_PATIENT_PUUID env var.",
    )
    p.add_argument(
        "--lab-pdf", type=Path, default=DEFAULT_LAB_PDF,
        help=f"Path to lab document. Default: {DEFAULT_LAB_PDF.relative_to(REPO_ROOT)}",
    )
    p.add_argument(
        "--intake-pdf", type=Path, default=DEFAULT_INTAKE_PDF,
        help=f"Path to intake document. Default: {DEFAULT_INTAKE_PDF.relative_to(REPO_ROOT)}",
    )
    p.add_argument("--lab-mime", default=None, help="Mime for lab file (default: guessed)")
    p.add_argument("--intake-mime", default=None, help="Mime for intake file (default: guessed)")
    p.add_argument("--lab-only", action="store_true", help="Skip the intake extraction (saves one API call)")
    p.add_argument("--intake-only", action="store_true", help="Skip the lab extraction (saves one API call)")
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"Claude model id (default: {DEFAULT_MODEL})")
    return p.parse_args()


async def _extract_one(
    *,
    label: str,
    pdf_path: Path,
    mime: str,
    puuid: str,
    doc_type: str,
    writer: OpenEMRWriter,
    model: str,
) -> int:
    """Run one attach_and_extract and print a summary. Returns 0 on success, 1 on fail."""
    if not pdf_path.exists():
        _fail(f"file missing: {pdf_path}")
        return 1
    file_bytes = pdf_path.read_bytes()
    print(
        f"  loading {pdf_path.name} ({len(file_bytes)} bytes, {mime})"
    )
    try:
        result = await attach_and_extract(
            file_bytes=file_bytes,
            filename=pdf_path.name,
            doc_type=doc_type,  # type: ignore[arg-type]
            patient_uuid=puuid,
            mime_type=mime,
            writer=writer,
            model=model,
        )
    except OpenEMRWriteError as exc:
        _fail(f"writer failed: {exc}")
        return 1
    except ExtractionError as exc:
        _fail(f"extraction failed: {exc}")
        return 1

    extracted = result.extracted
    sha = result.write_result["sha256"][:12]
    _ok(
        f"{label}: ref={result.reference_id} sha256={sha}... "
        f"created={result.created}"
    )
    if doc_type == "lab_pdf":
        n = len(extracted.results)  # type: ignore[union-attr]
        _ok(f"  extracted {n} lab result(s)")
    elif doc_type == "intake_form":
        n_meds = len(extracted.current_medications)  # type: ignore[union-attr]
        n_all = len(extracted.allergies)  # type: ignore[union-attr]
        n_fh = len(extracted.family_history)  # type: ignore[union-attr]
        chief = extracted.chief_concern  # type: ignore[union-attr]
        _ok(
            f"  extracted {n_meds} med(s), {n_all} allergy/-ies, "
            f"{n_fh} family history item(s)"
        )
        _ok(f"  chief concern: {chief[:80]}{'…' if len(chief) > 80 else ''}")
    return 0


async def _run(args: argparse.Namespace) -> int:
    if not args.puuid:
        _fail("no patient PUUID — set DEMO_PATIENT_PUUID in .env or pass --puuid")
        print(
            "\nIf this is your first run on a fresh OpenEMR instance:\n"
            "  PYTHONPATH=. uv run python scripts/seed_chen.py\n"
            "and paste the printed DEMO_PATIENT_PUUID line into .env.",
            file=sys.stderr,
        )
        return 1
    if args.lab_only and args.intake_only:
        _fail("--lab-only and --intake-only cannot both be set")
        return 1

    do_lab = not args.intake_only
    do_intake = not args.lab_only
    n_steps = int(do_lab) + int(do_intake)
    if n_steps == 0:
        _fail("nothing to do (both flags would skip everything)")
        return 1

    writer = OpenEMRWriter()
    try:
        step = 0
        if do_lab:
            step += 1
            _print_step(step, n_steps, f"Lab extraction (model={args.model})")
            rc = await _extract_one(
                label="lab",
                pdf_path=args.lab_pdf,
                mime=_guess_mime(args.lab_pdf, args.lab_mime),
                puuid=args.puuid,
                doc_type="lab_pdf",
                writer=writer,
                model=args.model,
            )
            if rc != 0:
                return rc

        if do_intake:
            step += 1
            _print_step(step, n_steps, f"Intake extraction (model={args.model})")
            rc = await _extract_one(
                label="intake",
                pdf_path=args.intake_pdf,
                mime=_guess_mime(args.intake_pdf, args.intake_mime),
                puuid=args.puuid,
                doc_type="intake_form",
                writer=writer,
                model=args.model,
            )
            if rc != 0:
                return rc

        print("\n──────────────────────────────────────────────────────────")
        print("PASS — attach_and_extract smoke OK against OpenEMR + Claude")
        print(f"  patient PUUID: {args.puuid}")
        print(f"  model:         {args.model}")
        print("──────────────────────────────────────────────────────────")
        return 0
    finally:
        await writer.aclose()


def main() -> None:
    sys.exit(asyncio.run(_run(_parse_args())))


if __name__ == "__main__":
    main()
