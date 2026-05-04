"""Isolated tests for the DocumentReference body builder.

`OpenEMRWriter.build_document_reference_body` is a static method on the
writer specifically so it can be tested without standing up OpenEMR. The
HTTP layers (token acquisition, search, POST) require a real OpenEMR and
are exercised by `scripts/smoke_document_writer.py` instead.

What we lock in here:
- The FHIR body shape matches OpenEMR's R4 expectations.
- The SHA-256 identifier is computed correctly and lands under the
  agent-forge URI we'll search by for idempotency.
- The doc-type → LOINC mapping is one-way deterministic.
- Unsupported doc types raise a typed error rather than silently producing
  malformed FHIR.
- Base64-encoded content round-trips back to the original bytes (so the
  PDF lands in OpenEMR byte-identical to what we sent).
"""

from __future__ import annotations

import base64
import hashlib

import pytest

from app.fhir.writer import (
    DOC_HASH_SYSTEM,
    DOC_TYPE_CODES,
    OpenEMRWriteError,
    OpenEMRWriter,
)


COHEN_PUUID = "a1a6044b-c6af-40a4-80aa-4c5ce61014da"


def _sample_pdf_bytes() -> bytes:
    """4-byte PDF magic + minimal content. Not a valid PDF for OpenEMR's
    parsing purposes, but enough to test byte-for-byte round-trip."""
    return b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n0 obj\n<< >>\nendobj\n%%EOF\n"


class TestSha256Hex:
    def test_known_value(self) -> None:
        # Anchor against a hand-computed hash so a bug in the helper
        # surfaces instead of "any 64 hex chars passes."
        data = b"hello world"
        expected = hashlib.sha256(data).hexdigest()
        assert OpenEMRWriter._sha256_hex(data) == expected
        assert len(expected) == 64


class TestBuildDocumentReferenceBody:
    def test_lab_pdf_minimal(self) -> None:
        body = OpenEMRWriter.build_document_reference_body(
            patient_uuid=COHEN_PUUID,
            doc_type="lab_pdf",
            file_bytes=_sample_pdf_bytes(),
            filename="cohen_lab_2026-04-30.pdf",
            creation_iso="2026-04-30T12:00:00+00:00",
        )

        assert body["resourceType"] == "DocumentReference"
        assert body["status"] == "current"
        assert body["subject"]["reference"] == f"Patient/{COHEN_PUUID}"
        assert body["date"] == "2026-04-30T12:00:00+00:00"
        # type → LOINC for laboratory report
        coding = body["type"]["coding"][0]
        assert coding == DOC_TYPE_CODES["lab_pdf"]

    def test_intake_form_uses_intake_loinc(self) -> None:
        body = OpenEMRWriter.build_document_reference_body(
            patient_uuid=COHEN_PUUID,
            doc_type="intake_form",
            file_bytes=_sample_pdf_bytes(),
            filename="cohen_intake_2026-04-30.pdf",
            creation_iso="2026-04-30T12:00:00+00:00",
        )
        assert body["type"]["coding"][0] == DOC_TYPE_CODES["intake_form"]

    def test_identifier_carries_sha256(self) -> None:
        data = _sample_pdf_bytes()
        body = OpenEMRWriter.build_document_reference_body(
            patient_uuid=COHEN_PUUID,
            doc_type="lab_pdf",
            file_bytes=data,
            filename="x.pdf",
            creation_iso="2026-04-30T12:00:00+00:00",
        )
        ident = body["identifier"][0]
        assert ident["system"] == DOC_HASH_SYSTEM
        assert ident["value"] == hashlib.sha256(data).hexdigest()

    def test_attachment_round_trips_bytes(self) -> None:
        data = _sample_pdf_bytes()
        body = OpenEMRWriter.build_document_reference_body(
            patient_uuid=COHEN_PUUID,
            doc_type="lab_pdf",
            file_bytes=data,
            filename="x.pdf",
            mime_type="application/pdf",
            creation_iso="2026-04-30T12:00:00+00:00",
        )
        att = body["content"][0]["attachment"]
        assert att["contentType"] == "application/pdf"
        assert att["title"] == "x.pdf"
        # Base64 decoding must give us back the exact bytes.
        assert base64.b64decode(att["data"]) == data

    def test_unsupported_doc_type_rejected(self) -> None:
        with pytest.raises(OpenEMRWriteError):
            OpenEMRWriter.build_document_reference_body(
                patient_uuid=COHEN_PUUID,
                doc_type="referral_fax",  # type: ignore[arg-type]  # not in DOC_TYPE_CODES
                file_bytes=_sample_pdf_bytes(),
                filename="x.pdf",
            )

    def test_creation_iso_defaults_to_now_when_omitted(self) -> None:
        # We don't lock the exact value (depends on now()), but it must be
        # populated, an ISO-shaped string, and reflected in both `date` and
        # the attachment's `creation` field (consistency between the two).
        body = OpenEMRWriter.build_document_reference_body(
            patient_uuid=COHEN_PUUID,
            doc_type="lab_pdf",
            file_bytes=_sample_pdf_bytes(),
            filename="x.pdf",
        )
        assert isinstance(body["date"], str) and len(body["date"]) >= 10
        assert body["date"] == body["content"][0]["attachment"]["creation"]

    def test_default_mime_type_is_application_pdf(self) -> None:
        body = OpenEMRWriter.build_document_reference_body(
            patient_uuid=COHEN_PUUID,
            doc_type="lab_pdf",
            file_bytes=_sample_pdf_bytes(),
            filename="x.pdf",
            creation_iso="2026-04-30T12:00:00+00:00",
        )
        assert body["content"][0]["attachment"]["contentType"] == "application/pdf"

    def test_custom_mime_type_honored(self) -> None:
        body = OpenEMRWriter.build_document_reference_body(
            patient_uuid=COHEN_PUUID,
            doc_type="lab_pdf",
            file_bytes=_sample_pdf_bytes(),
            filename="x.png",
            mime_type="image/png",
            creation_iso="2026-04-30T12:00:00+00:00",
        )
        assert body["content"][0]["attachment"]["contentType"] == "image/png"
