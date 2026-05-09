"""Isolated tests for the PDF/image → PNG renderer.

Uses the real Phase 1.2 fixture PDFs (Cohen lab + intake) and one of the
real-doc PNG fixtures (Reyes hba1c) so the test exercises the actual
content shapes the extractor will see, without needing any network or
Anthropic API access.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image

from app.extraction.render import (
    image_to_png_pages,
    pdf_to_png_pages,
    render_to_png_pages,
)


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEMO_DIR = REPO_ROOT / "data" / "demo_documents"
COHEN_LAB_PDF = DEMO_DIR / "cohen_lab_2026-04-30.pdf"
COHEN_INTAKE_PDF = DEMO_DIR / "cohen_intake_2026-04-30.pdf"
REYES_HBA1C_PNG = DEMO_DIR / "real" / "p03-reyes-hba1c.png"


def _png_signature_ok(b: bytes) -> bool:
    """PNGs always start with the 8-byte signature `\\x89PNG\\r\\n\\x1a\\n`."""
    return b.startswith(b"\x89PNG\r\n\x1a\n")


def _png_dimensions(b: bytes) -> tuple[int, int]:
    img = Image.open(io.BytesIO(b))
    img.load()
    return img.size


class TestPdfToPngPages:
    def test_cohen_lab_pdf_renders_to_one_page(self) -> None:
        # Phase 1.2 generator emits a single-page lab PDF; lock that in
        # so a later layout change with multi-page output is loud.
        pages = pdf_to_png_pages(COHEN_LAB_PDF.read_bytes())
        assert len(pages) == 1
        assert _png_signature_ok(pages[0])

    def test_cohen_intake_pdf_renders_to_one_page(self) -> None:
        pages = pdf_to_png_pages(COHEN_INTAKE_PDF.read_bytes())
        assert len(pages) == 1
        assert _png_signature_ok(pages[0])

    def test_render_resolution_is_legible_for_typed_text(self) -> None:
        # 150 DPI on a Letter page (8.5" x 11") -> 1275 x 1650 nominal,
        # then post-resized in `render.py` so the longest edge fits within
        # `_VISION_MAX_EDGE = 1500` (the bbox coordinate-space anchor for
        # Claude vision). After the resize, Letter renders at ≈1158 x 1500.
        # The original `>= 1200` threshold predates the vision-consistency
        # resize; loosened to ≥1100 so we still catch a thumbnail-scale
        # regression without flagging the intentional downscale.
        pages = pdf_to_png_pages(COHEN_LAB_PDF.read_bytes())
        w, h = _png_dimensions(pages[0])
        assert w >= 1100
        assert h >= 1400

    def test_empty_bytes_rejected(self) -> None:
        with pytest.raises(ValueError, match="empty input"):
            pdf_to_png_pages(b"")


class TestImageToPngPages:
    def test_png_passthrough_normalized(self) -> None:
        original = REYES_HBA1C_PNG.read_bytes()
        pages = image_to_png_pages(original)
        assert len(pages) == 1
        assert _png_signature_ok(pages[0])

    def test_jpeg_converted_to_png(self) -> None:
        # Build a tiny JPEG in-memory so the test is self-contained.
        jpeg_buf = io.BytesIO()
        Image.new("RGB", (50, 50), color=(255, 128, 64)).save(jpeg_buf, "JPEG")
        pages = image_to_png_pages(jpeg_buf.getvalue())
        assert len(pages) == 1
        assert _png_signature_ok(pages[0])

    def test_empty_bytes_rejected(self) -> None:
        with pytest.raises(ValueError, match="empty input"):
            image_to_png_pages(b"")


class TestRenderToPngPagesDispatch:
    def test_pdf_dispatches_to_pdf_renderer(self) -> None:
        pages = render_to_png_pages(
            COHEN_LAB_PDF.read_bytes(), "application/pdf",
        )
        assert len(pages) == 1
        assert _png_signature_ok(pages[0])

    def test_png_dispatches_to_image_renderer(self) -> None:
        pages = render_to_png_pages(
            REYES_HBA1C_PNG.read_bytes(), "image/png",
        )
        assert len(pages) == 1
        assert _png_signature_ok(pages[0])

    def test_jpeg_mime_dispatches_to_image_renderer(self) -> None:
        # Confirms the dispatch covers both image/jpeg and image/jpg
        # without exploding on the synonym.
        jpeg_buf = io.BytesIO()
        Image.new("RGB", (10, 10), color=(0, 0, 0)).save(jpeg_buf, "JPEG")
        pages_a = render_to_png_pages(jpeg_buf.getvalue(), "image/jpeg")
        pages_b = render_to_png_pages(jpeg_buf.getvalue(), "image/jpg")
        assert len(pages_a) == 1
        assert len(pages_b) == 1

    def test_mime_case_insensitive(self) -> None:
        pages = render_to_png_pages(
            COHEN_LAB_PDF.read_bytes(), "APPLICATION/PDF",
        )
        assert len(pages) == 1

    def test_unsupported_mime_rejected(self) -> None:
        # The extractor only knows how to look at PDFs and images;
        # a Word doc (or anything else) must surface here, not as a
        # cryptic Claude-side rejection.
        with pytest.raises(ValueError, match="unsupported mime_type"):
            render_to_png_pages(b"x", "application/msword")
