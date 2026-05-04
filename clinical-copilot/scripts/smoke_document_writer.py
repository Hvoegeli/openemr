"""Smoke test for the Phase 1.3 DocumentReference writer.

Drives `OpenEMRWriter.write_document_reference` end-to-end against a
running local OpenEMR. Proves three things in order:

  1. The seed client + password grant gets a token with the new
     `user/DocumentReference.{read,write}` scopes attached.
  2. The writer POSTs a real PDF and gets back a real
     DocumentReference/{id} from OpenEMR's FHIR API.
  3. Re-running the same upload returns the SAME id without POSTing
     again (SHA-256 idempotency — Phase 1.3 design choice #3 = A).

Prerequisites (one-time per dev box):
  - `cd docker/development-easy && docker compose up --detach --wait`
  - `cd clinical-copilot && PYTHONPATH=. uv run python scripts/seed_cohen.py`
  - `cd clinical-copilot && PYTHONPATH=. uv run python scripts/register_seed_client.py`
    (or `add_document_reference_scope.py` if the seed client already exists)
  - `.env` populated with OPENEMR_SEED_CLIENT_ID/SECRET, OPENEMR_FHIR_BASE_URL,
    OPENEMR_OAUTH_TOKEN_URL.

Run: `cd clinical-copilot && PYTHONPATH=. uv run python scripts/smoke_document_writer.py`

Outputs PASS / FAIL per assertion plus the resulting DocumentReference IDs
so the operator can verify in phpMyAdmin / OpenEMR's documents UI.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from app.fhir.writer import OpenEMRWriteError, OpenEMRWriter


REPO_ROOT = Path(__file__).resolve().parent.parent
DEMO_DIR = REPO_ROOT / "data" / "demo_documents"

# Cohen's PUUID — see seed_cohen.py
COHEN_PUUID = "a1a6044b-c6af-40a4-80aa-4c5ce61014da"


def _print_step(step: int, total: int, label: str) -> None:
    print(f"\n[{step}/{total}] {label}")


def _ok(msg: str) -> None:
    print(f"  ✓ {msg}")


def _fail(msg: str) -> None:
    print(f"  ✗ {msg}")


async def _run() -> int:
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

        # ── Step 2: load demo PDFs from disk
        _print_step(2, 5, "Load synthetic demo PDFs")
        lab_path = DEMO_DIR / "cohen_lab_2026-04-30.pdf"
        intake_path = DEMO_DIR / "cohen_intake_2026-04-30.pdf"
        for p in (lab_path, intake_path):
            if not p.exists():
                _fail(f"demo PDF missing: {p}")
                print("\nRun `PYTHONPATH=. uv run python scripts/generate_demo_documents.py`")
                print("to regenerate the demo data.")
                return 1
        lab_bytes = lab_path.read_bytes()
        intake_bytes = intake_path.read_bytes()
        _ok(f"loaded lab PDF ({len(lab_bytes)} bytes) + intake PDF ({len(intake_bytes)} bytes)")

        # ── Step 3: write the lab PDF (first time — should create)
        _print_step(3, 5, "Write lab PDF (expect created=True)")
        try:
            lab_result = await writer.write_document_reference(
                patient_uuid=COHEN_PUUID,
                doc_type="lab_pdf",
                file_bytes=lab_bytes,
                filename=lab_path.name,
            )
        except OpenEMRWriteError as exc:
            _fail(f"lab PDF write failed: {exc}")
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
            print("  re-generate demo PDFs (different bytes) or delete the existing one.")
            return 1

        # ── Step 4: re-write same lab PDF (idempotency — should return existing)
        _print_step(4, 5, "Re-write lab PDF (expect created=False, same id)")
        try:
            lab_redo = await writer.write_document_reference(
                patient_uuid=COHEN_PUUID,
                doc_type="lab_pdf",
                file_bytes=lab_bytes,
                filename=lab_path.name,
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

        # ── Step 5: write the intake PDF (different bytes -> different DocRef)
        _print_step(5, 5, "Write intake PDF (different bytes — expect new id)")
        try:
            intake_result = await writer.write_document_reference(
                patient_uuid=COHEN_PUUID,
                doc_type="intake_form",
                file_bytes=intake_bytes,
                filename=intake_path.name,
            )
        except OpenEMRWriteError as exc:
            _fail(f"intake PDF write failed: {exc}")
            return 1
        _ok(f"created {intake_result['reference_id']} (sha256={intake_result['sha256'][:12]}...)")
        if intake_result["reference_id"] == lab_result["reference_id"]:
            _fail("intake DocumentReference shares an id with lab — dedupe key collision?")
            return 1
        if not intake_result["created"]:
            _fail("expected created=True for intake (different content) — got False")
            return 1

        print("\n──────────────────────────────────────────────────────────")
        print(f"PASS — DocumentReference writer smoke OK against OpenEMR")
        print(f"  lab    DocumentReference: {lab_result['reference_id']}")
        print(f"  intake DocumentReference: {intake_result['reference_id']}")
        print("──────────────────────────────────────────────────────────")
        return 0

    finally:
        await writer.aclose()


def main() -> None:
    code = asyncio.run(_run())
    sys.exit(code)


if __name__ == "__main__":
    main()
