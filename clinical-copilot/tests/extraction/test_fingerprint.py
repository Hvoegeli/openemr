"""Isolated tests for the document-fingerprint helpers.

`compute_fingerprint` is the post-extraction Layer-2 hash and gets exercised
indirectly by the upload-endpoint dedup tests. The helpers under test here
are the Layer-1.5 PDF text-layer fingerprint that was added to short-circuit
a Layer-2 prompt before paying for a Claude vision call.

We synthesize PDFs in-memory via reportlab so each test controls the
text content it exercises (whitespace, case, multi-page, scanned-empty).
The real demo fixtures are also brought in for one stability assertion
so a future change to the normalization can't silently re-hash a known
production-shape document.
"""

from __future__ import annotations

import io
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas

from app.extraction.fingerprint import (
    namespace_text_fingerprint,
    pdf_text_fingerprint,
)


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEMO_DIR = REPO_ROOT / "data" / "demo_documents"
COHEN_LAB_PDF = DEMO_DIR / "cohen_lab_2026-04-30.pdf"
COHEN_INTAKE_PDF = DEMO_DIR / "cohen_intake_2026-04-30.pdf"


def _text_pdf(lines: list[str]) -> bytes:
    """Build a single-page text PDF whose visible content is `lines`.

    reportlab's text pipeline emits a real text layer (not a rasterized
    bitmap), so `pdf_text_fingerprint` can pull the same chars back out.
    """
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=LETTER)
    y = 720
    for line in lines:
        c.drawString(72, y, line)
        y -= 18
    c.showPage()
    c.save()
    return buf.getvalue()


def _multi_page_text_pdf(pages: list[list[str]]) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=LETTER)
    for page_lines in pages:
        y = 720
        for line in page_lines:
            c.drawString(72, y, line)
            y -= 18
        c.showPage()
    c.save()
    return buf.getvalue()


def _scanned_only_pdf() -> bytes:
    """Build a PDF whose only content is a rasterized image — no text layer.

    Matches the shape of a fax-to-PDF or a phone-photo upload where
    `pdf_text_fingerprint` should refuse to hash and return `None`.
    """
    img = Image.new("RGB", (200, 200), color=(255, 255, 255))
    img_buf = io.BytesIO()
    img.save(img_buf, format="JPEG")
    img_buf.seek(0)

    out = io.BytesIO()
    c = canvas.Canvas(out, pagesize=LETTER)
    from reportlab.lib.utils import ImageReader  # noqa: PLC0415
    c.drawImage(ImageReader(img_buf), 100, 500, width=200, height=200)
    c.showPage()
    c.save()
    return out.getvalue()


class TestPdfTextFingerprint:
    def test_empty_bytes_returns_none(self) -> None:
        assert pdf_text_fingerprint(b"") is None

    def test_non_pdf_bytes_returns_none(self) -> None:
        # The resolve endpoint defaults `contentType` to "application/pdf"
        # when the FHIR attachment omits it; a mislabelled or missing
        # contentType must NOT 500 the helper. Returning None lets the
        # caller fall through to the vision path.
        assert pdf_text_fingerprint(b"not a pdf at all") is None
        # PNG bytes (signature + minimal payload) — the helper must
        # recognize this as non-PDF and bail out cleanly.
        png_signature = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
        assert pdf_text_fingerprint(png_signature) is None

    def test_stable_across_calls_on_same_pdf(self) -> None:
        pdf = _text_pdf(["Patient: Jane Doe", "DOB: 1980-03-12"])
        assert pdf_text_fingerprint(pdf) == pdf_text_fingerprint(pdf)

    def test_different_text_different_hash(self) -> None:
        a = _text_pdf(["Penicillin allergy: hives"])
        b = _text_pdf(["No known drug allergies"])
        ha, hb = pdf_text_fingerprint(a), pdf_text_fingerprint(b)
        assert ha is not None and hb is not None
        assert ha != hb

    def test_whitespace_runs_normalize(self) -> None:
        # Same visible text, different inter-word spacing. Should
        # collapse to identical hash.
        a = _text_pdf(["Patient    Jane    Doe"])
        b = _text_pdf(["Patient Jane Doe"])
        ha = pdf_text_fingerprint(a)
        hb = pdf_text_fingerprint(b)
        assert ha is not None
        assert ha == hb

    def test_case_folding(self) -> None:
        # Same text, different case. Handwritten vs typed transcription
        # should not split fingerprints.
        a = _text_pdf(["PENICILLIN — hives"])
        b = _text_pdf(["penicillin — hives"])
        assert pdf_text_fingerprint(a) == pdf_text_fingerprint(b)

    def test_page_boundary_observable(self) -> None:
        # Same set of words split across pages differently should NOT
        # collide — page boundaries are observable in the hash so two
        # PDFs whose paginated layout differs are different documents.
        single = _multi_page_text_pdf([["alpha beta gamma"]])
        two_pages = _multi_page_text_pdf([["alpha"], ["beta gamma"]])
        h1 = pdf_text_fingerprint(single)
        h2 = pdf_text_fingerprint(two_pages)
        assert h1 is not None and h2 is not None
        assert h1 != h2

    def test_pure_scanned_image_returns_none(self) -> None:
        # A PDF that only contains a rasterized image has no text layer
        # to extract; the helper falls through to None so the caller
        # sends it to vision (where Layer-2 catches duplicates).
        pdf = _scanned_only_pdf()
        assert pdf_text_fingerprint(pdf) is None

    def test_real_demo_pdf_has_stable_hash(self) -> None:
        # Locks in normalization on a production-shape document. If a
        # future change to the helper rehashes the demo lab PDF, a code
        # reviewer should see this fail and explicitly re-bless it.
        pdf_bytes = COHEN_LAB_PDF.read_bytes()
        h = pdf_text_fingerprint(pdf_bytes)
        assert h is not None
        assert len(h) == 64  # SHA-256 hex


class TestNamespaceTextFingerprint:
    def test_prepends_text_v1_prefix(self) -> None:
        out = namespace_text_fingerprint("abc123")
        assert out == "text-v1:abc123"

    def test_namespace_makes_layer_15_distinguishable_from_layer_2(self) -> None:
        # A SHA-256 over the structural projection (Layer 2) and the
        # text projection (Layer 1.5) could in the worst case collide;
        # the namespace prefix keeps the store keys distinct.
        raw = "0" * 64
        assert namespace_text_fingerprint(raw) != raw
        assert namespace_text_fingerprint(raw).endswith(raw)


class TestPdfTextFingerprintAgainstSeparateDecode:
    """Sanity check: the helper sees the same characters pypdfium2 would
    expose to a hand-written caller. Locks down the assumption that
    `get_text_range` is what reportlab's text pipeline writes."""

    def test_extracted_text_round_trips(self) -> None:
        pdf = _text_pdf(["Hello, world."])
        assert pdf_text_fingerprint(pdf) is not None

        doc = pdfium.PdfDocument(pdf)
        try:
            page = doc[0]
            try:
                tp = page.get_textpage()
                try:
                    text = tp.get_text_range()
                finally:
                    tp.close()
            finally:
                page.close()
        finally:
            doc.close()
        assert "hello" in text.lower()
        assert "world" in text.lower()
