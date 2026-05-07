"""Turn an uploaded clinical document into the per-page PNG bytes the
Claude vision API expects.

Four input shapes the writer/extractor accept:

- **PDFs** are rendered page-by-page via `pypdfium2` (pure-Python wrapper
  around Google's PDFium). 150 DPI is the sweet spot for typed lab PDFs —
  high enough that small typeset numbers stay legible, low enough that a
  10-page document doesn't blow past Anthropic's 5 MB-per-image limit.

- **Image inputs (PNG/JPEG)** are treated as a single-page document and
  re-encoded to PNG so the downstream vision call always sees a uniform
  PNG content block. Re-encoding (rather than passthrough) costs a few
  ms but avoids "image format not supported" surprises if the upload's
  MIME type is mislabelled.

- **Multi-page TIFF** (the typical fax-packet format) is unpacked frame-
  by-frame via Pillow's `n_frames` API. Each frame is converted to RGB
  (faxes ship as 1-bit bitmaps; the vision API wants RGB-encoded PNG)
  and re-encoded as PNG. Returns one PNG per page, in document order.

- **Word documents (.docx)** are converted to PDF via LibreOffice
  headless, then fed through the PDF path. Requires `soffice` (LibreOffice)
  to be installed and on PATH — `apt install libreoffice-core
  libreoffice-writer` on Debian/Ubuntu, or the LibreOffice.app bundle on
  macOS. Cold-start latency is ~3-5s on the first conversion of a process
  lifecycle; subsequent conversions are sub-second.

Returned bytes are raw PNG — the caller is responsible for base64-
encoding them into the Anthropic message envelope.
"""

from __future__ import annotations

import io
import os
import shutil
import subprocess
import tempfile

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


_DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
_TIFF_MIMES = ("image/tiff", "image/x-tiff", "image/tif", "image/x-tif")


def tiff_to_png_pages(tiff_bytes: bytes) -> list[bytes]:
    """Render every frame of a multi-page TIFF to a PNG bytestring.

    Faxes typically ship as 1-bit (black-and-white) TIFFs at 200 DPI.
    Pillow exposes the frame count as `Image.n_frames` and lets us seek
    to each frame with `Image.seek(i)`. We convert each frame to RGB
    before saving so the vision API gets a consistent RGB-encoded PNG;
    leaving 1-bit mode in place can confuse Anthropic's image decoder
    on some content paths.

    Empty TIFFs (zero frames) raise ValueError — there is nothing for
    the extractor to look at and silently returning `[]` would propagate
    as a confusing downstream extraction failure.
    """
    if not tiff_bytes:
        raise ValueError("tiff_to_png_pages: empty input bytes")
    pil_image = Image.open(io.BytesIO(tiff_bytes))
    n_frames = getattr(pil_image, "n_frames", 1)
    if n_frames < 1:
        raise ValueError("tiff_to_png_pages: TIFF has zero frames")
    out: list[bytes] = []
    for frame_idx in range(n_frames):
        pil_image.seek(frame_idx)
        # `convert("RGB")` flattens 1-bit / palette / CMYK frames to a
        # PNG-encodable mode without losing the visible content. PNG's
        # own optimizer handles the bitmap nature of fax content
        # gracefully (large flat backgrounds compress well).
        frame_rgb = pil_image.convert("RGB")
        buf = io.BytesIO()
        frame_rgb.save(buf, format="PNG", optimize=True)
        out.append(buf.getvalue())
    return out


def _resolve_soffice_binary() -> str | None:
    """Locate the LibreOffice headless binary, returning None if absent.

    Checks `$LIBREOFFICE_BIN` first (lets ops override the lookup on a
    given host), then PATH, then the macOS bundle path. Returns the
    resolved path or `None`. The caller surfaces a clear ValueError so
    users see "install libreoffice" instead of "FileNotFoundError".
    """
    override = os.environ.get("LIBREOFFICE_BIN")
    if override and os.path.isfile(override) and os.access(override, os.X_OK):
        return override
    on_path = shutil.which("soffice") or shutil.which("libreoffice")
    if on_path:
        return on_path
    macos_bundle = "/Applications/LibreOffice.app/Contents/MacOS/soffice"
    if os.path.isfile(macos_bundle) and os.access(macos_bundle, os.X_OK):
        return macos_bundle
    return None


def docx_to_pdf_bytes(docx_bytes: bytes) -> bytes:
    """Convert a .docx blob to PDF bytes via LibreOffice headless.

    Writes the input to a tempdir, shells out to `soffice --headless
    --convert-to pdf`, reads the resulting PDF back, and returns its
    bytes. The tempdir is wiped on exit even on exception. If
    LibreOffice is not installed, raises ValueError with an actionable
    install hint rather than a cryptic FileNotFoundError.

    Subprocess timeout: 60 seconds. Real-world referral letters convert
    in well under a second once LibreOffice's font cache is warm; the
    timeout is a safety net for a hung or runaway process, not a
    performance budget.
    """
    if not docx_bytes:
        raise ValueError("docx_to_pdf_bytes: empty input bytes")
    soffice = _resolve_soffice_binary()
    if soffice is None:
        raise ValueError(
            "docx_to_pdf_bytes: LibreOffice (soffice) not found on PATH or "
            "in /Applications/LibreOffice.app. Install with "
            "`apt install libreoffice-core libreoffice-writer` on Linux or "
            "`brew install --cask libreoffice` on macOS, or set the "
            "LIBREOFFICE_BIN env var to the binary path."
        )
    with tempfile.TemporaryDirectory(prefix="copilot-docx-") as tmp:
        src = os.path.join(tmp, "input.docx")
        with open(src, "wb") as f:
            f.write(docx_bytes)
        # `--outdir` lands the converted PDF next to the input so we can
        # read it back by deterministic name. LibreOffice rewrites the
        # extension automatically: `input.docx` -> `input.pdf`.
        try:
            result = subprocess.run(
                [
                    soffice,
                    "--headless",
                    "--norestore",
                    "--nolockcheck",
                    "--convert-to", "pdf",
                    "--outdir", tmp,
                    src,
                ],
                capture_output=True,
                timeout=60,
                check=False,
            )
        except subprocess.TimeoutExpired as e:
            raise ValueError(
                "docx_to_pdf_bytes: LibreOffice conversion timed out after "
                f"{e.timeout}s — the input may be malformed or LibreOffice "
                "may be deadlocked"
            ) from e
        if result.returncode != 0:
            stderr = (result.stderr or b"").decode("utf-8", errors="replace")[:500]
            raise ValueError(
                f"docx_to_pdf_bytes: LibreOffice exited {result.returncode}: {stderr}"
            )
        out_pdf = os.path.join(tmp, "input.pdf")
        if not os.path.isfile(out_pdf):
            stdout = (result.stdout or b"").decode("utf-8", errors="replace")[:500]
            raise ValueError(
                "docx_to_pdf_bytes: LibreOffice exited 0 but produced no PDF; "
                f"stdout={stdout!r}"
            )
        with open(out_pdf, "rb") as f:
            return f.read()


def render_to_png_pages(file_bytes: bytes, mime_type: str) -> list[bytes]:
    """Dispatch to the right per-page renderer based on the upload MIME.

    Supported MIMEs: `application/pdf`, `image/png`, `image/jpeg`,
    `image/jpg`, multi-page `image/tiff` (and the `image/x-tiff` /
    `image/tif` variants browsers occasionally send for fax packets),
    and the .docx OOXML MIME. Anything else raises ValueError so a
    misfiled upload surfaces here instead of propagating into the
    vision call as a cryptic Claude-side rejection.

    .docx inputs are routed through LibreOffice headless to produce a
    PDF, which is then rendered through the standard PDF path. Requires
    `soffice` on PATH — see `docx_to_pdf_bytes` for installation.

    .tiff inputs (typically multi-page faxes) iterate every frame via
    Pillow's `n_frames` API and produce one PNG per page in document
    order — see `tiff_to_png_pages`.
    """
    mt = (mime_type or "").lower()
    if mt == "application/pdf":
        return pdf_to_png_pages(file_bytes)
    if mt in ("image/png", "image/jpeg", "image/jpg"):
        return image_to_png_pages(file_bytes)
    if mt in _TIFF_MIMES:
        return tiff_to_png_pages(file_bytes)
    if mt == _DOCX_MIME:
        return pdf_to_png_pages(docx_to_pdf_bytes(file_bytes))
    raise ValueError(
        f"render_to_png_pages: unsupported mime_type {mime_type!r}; "
        "expected application/pdf, image/png, image/jpeg, image/tiff, or "
        f"{_DOCX_MIME}"
    )
