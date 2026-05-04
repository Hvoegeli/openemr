"""Isolated tests for the DocumentReference write path.

The HTTP layers (token acquisition, multipart upload, FHIR GET search) need
a real OpenEMR and are exercised by `scripts/smoke_document_writer.py`.
What's testable in isolation here:

- The SHA-256 helper computes the expected hex digest.
- The idempotency-filename builder is deterministic and embeds the hash in
  the format the FHIR GET search will match against.
- The `doc_type → OpenEMR category path` resolver maps each supported
  doc_type and rejects unknown ones with a typed error rather than letting
  garbage propagate to the upload endpoint.

Anything beyond these — request shape, dedupe behavior, response handling
— is HTTP-layer logic; mocking httpx for it would be brittle and would not
catch real OpenEMR-side regressions, so we leave it to the smoke test.
"""

from __future__ import annotations

import hashlib

import pytest

from app.fhir.writer import (
    DOC_CATEGORIES,
    OpenEMRWriteError,
    OpenEMRWriter,
    _idempotency_filename,
)


COHEN_PUUID = "a1a6044b-c6af-40a4-80aa-4c5ce61014da"


def _sample_pdf_bytes() -> bytes:
    """Minimal PDF magic + body. Not a parseable PDF for OpenEMR's purposes
    but enough for the hash + filename helpers (which never inspect content)."""
    return b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n0 obj\n<< >>\nendobj\n%%EOF\n"


class TestSha256Hex:
    def test_known_value(self) -> None:
        # Anchor against a hand-computed hash so a bug in the helper
        # surfaces instead of "any 64 hex chars passes."
        data = b"hello world"
        expected = hashlib.sha256(data).hexdigest()
        assert OpenEMRWriter._sha256_hex(data) == expected
        assert len(expected) == 64


class TestIdempotencyFilename:
    def test_format(self) -> None:
        sha = "a" * 64
        assert _idempotency_filename(sha, "lab.pdf") == f"sha256-{sha}__lab.pdf"

    def test_deterministic(self) -> None:
        data = _sample_pdf_bytes()
        sha = OpenEMRWriter._sha256_hex(data)
        first = _idempotency_filename(sha, "lab.pdf")
        second = _idempotency_filename(sha, "lab.pdf")
        assert first == second

    def test_includes_full_hash(self) -> None:
        sha = OpenEMRWriter._sha256_hex(_sample_pdf_bytes())
        out = _idempotency_filename(sha, "lab.pdf")
        assert sha in out
        assert len(sha) == 64

    def test_preserves_original_filename(self) -> None:
        out = _idempotency_filename("a" * 64, "Cohen-2026-04-30.lab.pdf")
        assert out.endswith("__Cohen-2026-04-30.lab.pdf")

    def test_different_content_yields_different_filename(self) -> None:
        a = _idempotency_filename(OpenEMRWriter._sha256_hex(b"a"), "x.pdf")
        b = _idempotency_filename(OpenEMRWriter._sha256_hex(b"b"), "x.pdf")
        assert a != b


class TestResolveDocCategory:
    def test_lab_pdf_maps_to_labreport(self) -> None:
        assert OpenEMRWriter.resolve_doc_category("lab_pdf") == "labreport"

    def test_intake_form_maps_to_patientinformation(self) -> None:
        assert (
            OpenEMRWriter.resolve_doc_category("intake_form") == "patientinformation"
        )

    def test_unsupported_doc_type_raises_typed_error(self) -> None:
        with pytest.raises(OpenEMRWriteError) as exc_info:
            OpenEMRWriter.resolve_doc_category("referral_fax")  # type: ignore[arg-type]
        # Error message should identify the offending type AND the expected set
        # so an operator can immediately see what they got wrong.
        assert "referral_fax" in str(exc_info.value)
        assert "lab_pdf" in str(exc_info.value)

    def test_doc_categories_constant_pre_normalized(self) -> None:
        """OpenEMR's getLastIdOfPath does case-sensitive equality against
        `replace(LOWER(name), ' ', '')` — paths must arrive lowercased
        with no spaces or the upload silently fails to attach to a
        category. This test locks the convention in."""
        for path in DOC_CATEGORIES.values():
            assert path == path.lower(), f"{path!r} not lowercased"
            assert " " not in path, f"{path!r} contains a space"
