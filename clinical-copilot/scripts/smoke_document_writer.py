"""Smoke test for the Phase 1.3 DocumentReference writer.

Drives `OpenEMRWriter.write_document_reference` end-to-end against a
running local OpenEMR. Proves three things in order:

  1. The seed client + password grant gets a token with the new
     `user/DocumentReference.{read,write}` scopes attached.
  2. The writer POSTs a real PDF and gets back a real
     DocumentReference/{id} from OpenEMR's FHIR API.
  3. Re-running the same upload returns the SAME id without POSTing
     again (SHA-256 idempotency — Phase 1.3 design choice #3 = A).

Defaults to Margaret Chen's typed lipid panel + intake (the cleanest
real documents in `data/demo_documents/real/`). Override via env vars or
CLI args to smoke against any other patient.

Prerequisites (one-time per dev box):
  - `cd docker/development-easy && docker compose up --detach --wait`
  - `cd clinical-copilot && PYTHONPATH=. uv run python scripts/register_seed_client.py`
    (or `add_document_reference_scope.py` if the seed client already exists)
  - `cd clinical-copilot && PYTHONPATH=. uv run python scripts/seed_chen.py`
    → paste printed `DEMO_PATIENT_PUUID=...` into `.env`
  - `.env` populated with OPENEMR_SEED_CLIENT_ID/SECRET, OPENEMR_FHIR_BASE_URL,
    OPENEMR_OAUTH_TOKEN_URL, DEMO_PATIENT_PUUID.

Run: `cd clinical-copilot && PYTHONPATH=. uv run python scripts/smoke_document_writer.py`

Override defaults:
  - `DEMO_PATIENT_PUUID=...` env var or `--puuid` CLI arg
  - `--lab-pdf <path>` and `--intake-pdf <path>` for non-Chen documents
  - `--lab-mime <mime>` and `--intake-mime <mime>` for PNG/non-PDF files
"""

from __future__ import annotations

import argparse
import asyncio
import mimetypes
import os
import sys
from pathlib import Path

from app.fhir.writer import OpenEMRWriteError, OpenEMRWriter


REPO_ROOT = Path(__file__).resolve().parent.parent
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
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--puuid",
        default=os.getenv("DEMO_PATIENT_PUUID"),
        help="OpenEMR Patient UUID. Defaults to DEMO_PATIENT_PUUID env var.",
    )
    p.add_argument(
        "--lab-pdf",
        type=Path,
        default=DEFAULT_LAB_PDF,
        help=f"Path to a lab document. Default: {DEFAULT_LAB_PDF.relative_to(REPO_ROOT)}",
    )
    p.add_argument(
        "--intake-pdf",
        type=Path,
        default=DEFAULT_INTAKE_PDF,
        help=f"Path to an intake document. Default: {DEFAULT_INTAKE_PDF.relative_to(REPO_ROOT)}",
    )
    p.add_argument("--lab-mime", default=None, help="Mime type for the lab file (default: guessed from extension)")
    p.add_argument("--intake-mime", default=None, help="Mime type for the intake file (default: guessed from extension)")
    return p.parse_args()


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

    writer = OpenEMRWriter()
    try:
        # ── Step 1: token acquisition (proves auth + scope wiring works)
        _print_step(1, 5, "Acquire seed-client token (password grant)")
        try:
            await writer._ensure_token()  # noqa: SLF001 — smoke-test access
        except OpenEMRWriteError as exc:
            _fail(f"token acquisition failed: {exc}")
            print("\nLikely fix: check OPENEMR_SEED_CLIENT_ID / OPENEMR_SEED_CLIENT_SECRET")
            print("in .env, and confirm OpenEMR is running on the configured FHIR base.")
            return 1
        _ok("got password-grant token")

        # ── Step 2: load demo files from disk
        _print_step(2, 5, f"Load demo files for patient {args.puuid}")
        for p in (args.lab_pdf, args.intake_pdf):
            if not p.exists():
                _fail(f"demo file missing: {p}")
                print(
                    "\nReal-document fixtures live in "
                    "data/demo_documents/real/. Confirm they were copied in,"
                    "\nor pass --lab-pdf / --intake-pdf to point elsewhere.",
                    file=sys.stderr,
                )
                return 1
        lab_bytes = args.lab_pdf.read_bytes()
        intake_bytes = args.intake_pdf.read_bytes()
        lab_mime = _guess_mime(args.lab_pdf, args.lab_mime)
        intake_mime = _guess_mime(args.intake_pdf, args.intake_mime)
        _ok(
            f"loaded {args.lab_pdf.name} ({len(lab_bytes)} bytes, {lab_mime}) "
            f"+ {args.intake_pdf.name} ({len(intake_bytes)} bytes, {intake_mime})"
        )

        # ── Step 3: write the lab document (first time — should create)
        _print_step(3, 5, "Write lab document (expect created=True)")
        try:
            lab_result = await writer.write_document_reference(
                patient_uuid=args.puuid,
                doc_type="lab_pdf",
                file_bytes=lab_bytes,
                filename=args.lab_pdf.name,
                mime_type=lab_mime,
            )
        except OpenEMRWriteError as exc:
            _fail(f"lab write failed: {exc}")
            print("\nCommon fixes:")
            print("  - seed client missing user/DocumentReference.{read,write} scope")
            print("    → run scripts/add_document_reference_scope.py")
            print("  - OpenEMR FHIR layer not enabled for DocumentReference write")
            print("    → check OpenEMR config + FHIR audit logs")
            return 1
        _ok(f"created {lab_result['reference_id']} (sha256={lab_result['sha256'][:12]}...)")
        if not lab_result["created"]:
            _fail("expected created=True on first write — got False (existing dedupe?)")
            print("  Hint: a previous smoke run left the same DocumentReference. Either")
            print("  pass a different file, or delete the existing one in OpenEMR.")
            return 1

        # ── Step 4: re-write same lab (idempotency — should return existing)
        _print_step(4, 5, "Re-write same lab document (expect created=False, same id)")
        try:
            lab_redo = await writer.write_document_reference(
                patient_uuid=args.puuid,
                doc_type="lab_pdf",
                file_bytes=lab_bytes,
                filename=args.lab_pdf.name,
                mime_type=lab_mime,
            )
        except OpenEMRWriteError as exc:
            _fail(f"idempotency check failed: {exc}")
            return 1
        if lab_redo["created"]:
            _fail(f"expected created=False (dedupe); got True with new id {lab_redo['reference_id']}")
            return 1
        if lab_redo["reference_id"] != lab_result["reference_id"]:
            _fail(
                f"dedupe returned different id: first={lab_result['reference_id']!r} "
                f"second={lab_redo['reference_id']!r}"
            )
            return 1
        _ok(f"dedupe returned same id {lab_redo['reference_id']}")

        # ── Step 5: write the intake document (different bytes -> different DocRef)
        _print_step(5, 5, "Write intake document (different bytes — expect new id)")
        try:
            intake_result = await writer.write_document_reference(
                patient_uuid=args.puuid,
                doc_type="intake_form",
                file_bytes=intake_bytes,
                filename=args.intake_pdf.name,
                mime_type=intake_mime,
            )
        except OpenEMRWriteError as exc:
            _fail(f"intake write failed: {exc}")
            return 1
        _ok(f"created {intake_result['reference_id']} (sha256={intake_result['sha256'][:12]}...)")
        if intake_result["reference_id"] == lab_result["reference_id"]:
            _fail("intake DocumentReference shares an id with lab — dedupe key collision?")
            return 1
        if not intake_result["created"]:
            _fail("expected created=True for intake (different content) — got False")
            return 1

        print("\n──────────────────────────────────────────────────────────")
        print("PASS — DocumentReference writer smoke OK against OpenEMR")
        print(f"  patient PUUID: {args.puuid}")
        print(f"  lab    DocumentReference: {lab_result['reference_id']}")
        print(f"  intake DocumentReference: {intake_result['reference_id']}")
        print("──────────────────────────────────────────────────────────")
        return 0

    finally:
        await writer.aclose()


def main() -> None:
    code = asyncio.run(_run(_parse_args()))
    sys.exit(code)


if __name__ == "__main__":
    main()
