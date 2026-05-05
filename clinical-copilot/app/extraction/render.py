"""Turn an uploaded clinical document into the per-page PNG bytes the
Claude vision API expects.

Two input shapes the writer/extractor accept:

- **PDFs** are rendered page-by-page via `pypdfium2` (pure-Python wrapper
  around Google's PDFium). 150 DPI is the sweet spot for typed lab PDFs —
  high enough that small typeset numbers stay legible, low enough that a
  10-page document doesn't blow past Anthropic's 5 MB-per-image limit.

- **Image inputs (PNG/JPEG)** are treated as a single-page document and
  re-encoded to PNG so the downstream vision call always sees a uniform
  PNG content block. Re-encoding (rather than passthrough) costs a few
  ms but avoids "image format not supported" surprises if the upload's
  MIME type is mislabelled.

Returned bytes are raw PNG — the caller is responsible for base64-
encoding them into the Anthropic message envelope.
"""

from __future__ import annotations

import io

import pypdfium2 as pdfium
from PIL import Image

# Render scale: pypdfium2's `scale` is in "render units" where 1.0 = 72 DPI
# (PDF native). 150 DPI = 150/72 ≈ 2.08. Bumping to 200 DPI is a one-line
# change if Whitaker stress-test extraction is poor on small text.
_PDF_RENDER_SCALE = 150 / 72


def pdf_to_png_pages(pdf_bytes: bytes) -> list[bytes]:
    """Render every page of a PDF to a PNG bytestring.

    Returns a list of PNG bytes, one per page in document order. Empty
    PDFs (zero pages) raise ValueError — there is nothing for the
    extractor to look at and silently returning an empty list would
    propagate as a confusing downstream extraction failure.
    """
    if not pdf_bytes:
        raise ValueError("pdf_to_png_pages: empty input bytes")
    pdf = pdfium.PdfDocument(pdf_bytes)
    try:
        if len(pdf) == 0:
            raise ValueError("pdf_to_png_pages: PDF has zero pages")
        out: list[bytes] = []
        for page in pdf:
            try:
                bitmap = page.render(scale=_PDF_RENDER_SCALE)
                pil_image = bitmap.to_pil()
                buf = io.BytesIO()
                pil_image.save(buf, format="PNG", optimize=True)
                out.append(buf.getvalue())
            finally:
                page.close()
        return out
    finally:
        pdf.close()


def image_to_png_pages(image_bytes: bytes) -> list[bytes]:
    """Treat a single image upload (PNG/JPEG) as a one-page document.

    Re-encodes to PNG via Pillow to normalize the format and strip any
    EXIF/metadata. Returns a single-element list so the call sites can
    treat PDFs and images uniformly.
    """
    if not image_bytes:
        raise ValueError("image_to_png_pages: empty input bytes")
    pil_image = Image.open(io.BytesIO(image_bytes))
    # PIL's lazy-loading keeps a file handle open until you force a load.
    # `load()` materializes the pixel data so the BytesIO can be closed.
    pil_image.load()
    if pil_image.mode not in ("RGB", "RGBA"):
        # Convert anything exotic (CMYK, P, L, etc.) to RGB so PNG encode
        # produces something Anthropic's vision API will accept.
        pil_image = pil_image.convert("RGB")
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG", optimize=True)
    return [buf.getvalue()]


def render_to_png_pages(file_bytes: bytes, mime_type: str) -> list[bytes]:
    """Dispatch to the right per-page renderer based on the upload MIME.

    Supported MIMEs: `application/pdf`, `image/png`, `image/jpeg`,
    `image/jpg`. Anything else raises ValueError so a misfiled upload
    surfaces here instead of propagating into the vision call as a
    cryptic Claude-side rejection.
    """
    mt = (mime_type or "").lower()
    if mt == "application/pdf":
        return pdf_to_png_pages(file_bytes)
    if mt in ("image/png", "image/jpeg", "image/jpg"):
        return image_to_png_pages(file_bytes)
    raise ValueError(
        f"render_to_png_pages: unsupported mime_type {mime_type!r}; "
        "expected application/pdf, image/png, or image/jpeg"
    )
