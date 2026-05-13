"""OCR helper for the image-channel jailbreak quarantine (C1).

`get_document_content` returns rendered document pages as base64-encoded
PNG image blocks so Claude's vision API can "see" them. The text-based
Layer-3 jailbreak scanner (`detect_jailbreak`) only sees the JSON
metadata block; it has no visibility into pixels. An attacker who can
upload a document with instruction-shaped text painted onto an image
(forged referral letter, doctored fax) can therefore smuggle a jailbreak
past the text scanner.

This module closes that gap: every PNG page returned by
`get_document_content` is OCR'd with Tesseract and the resulting text is
fed into the same `detect_jailbreak` patterns. A hit triggers the
existing quarantine marker, so the LLM sees the rendered pages PLUS a
header explaining that one of them resembled a jailbreak directive.

Scope: this module is wired ONLY into the `get_document_content` tool
result path. Other tools never produce image content, so there is no
benefit to OCR'ing their output and a real cost (latency, Tesseract
spawns) to doing so unconditionally.

Failure mode: fail closed at startup. `verify_tesseract_available()` is
called from the FastAPI lifespan; if the `tesseract` binary is not
installed (`pytesseract.get_tesseract_version()` raises), the app
refuses to start with an install hint. This guarantees a deploy can
never silently lose the image-channel scan.
"""

from __future__ import annotations

import io
import logging
from typing import Final

import pytesseract
from PIL import Image, UnidentifiedImageError

log = logging.getLogger(__name__)

# Tesseract OCR config:
#   --oem 3  → default LSTM engine
#   --psm 6  → "assume a single uniform block of text"; works well for
#              referral letters / lab reports / fax packets where the
#              document is one column of body text. PSM 3 (auto) gives
#              comparable results on clean PDF renders but is slower and
#              more prone to junk output on noisy 1-bit fax bitmaps.
# These are read-only inside the worker, so a module-level constant is
# fine — no need to take a lock or re-read per call.
_TESSERACT_CONFIG: Final[str] = "--oem 3 --psm 6"


class TesseractUnavailable(RuntimeError):
    """Raised by `verify_tesseract_available` when the `tesseract` system
    binary cannot be invoked. The lifespan handler converts this to a
    fail-closed startup error so a deploy without OCR cannot serve."""


def verify_tesseract_available() -> str:
    """Return the installed Tesseract version string, or raise.

    Called from FastAPI's lifespan during startup. The image-channel
    jailbreak scan depends on Tesseract, so a missing binary is a
    deploy-time configuration error — we surface it immediately rather
    than serving requests with one of the layered defenses silently
    disabled.
    """
    try:
        version = pytesseract.get_tesseract_version()
    except pytesseract.TesseractNotFoundError as exc:
        raise TesseractUnavailable(
            "Tesseract OCR binary not found. The image-channel jailbreak "
            "quarantine (C1) requires `tesseract` to be installed. Install "
            "with `brew install tesseract` on macOS or "
            "`apt install tesseract-ocr` on Debian/Ubuntu, then restart."
        ) from exc
    return str(version)


def ocr_png_text(png_bytes: bytes) -> str:
    """Extract text from a single PNG page via Tesseract.

    Returns the OCR'd text (joined across all detected lines), or an
    empty string on a recoverable error — a corrupted PNG, an empty page
    image, or a Tesseract subprocess failure. We do NOT raise because
    the OCR result is downstream input to the jailbreak SCAN, not the
    document content the LLM reads; failing the whole tool call because
    one page of a fax packet wouldn't decode would be worse than
    quietly skipping the scan on that page (the LLM still gets the
    pixels, which is the legitimate clinical use case).

    Tesseract crashes / missing binary are logged at WARNING so the
    audit trail still captures a partial-scan event. The startup check
    ensures the binary IS available on the happy path; this guard
    catches the edge case where the binary disappears after startup
    (rare but possible during a live `apt upgrade`).
    """
    if not isinstance(png_bytes, bytes) or not png_bytes:
        return ""
    try:
        image = Image.open(io.BytesIO(png_bytes))
        image.load()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        log.warning("ocr: PNG decode failed (%s); skipping image scan", exc)
        return ""
    try:
        text = pytesseract.image_to_string(image, config=_TESSERACT_CONFIG)
    except pytesseract.TesseractNotFoundError as exc:
        log.warning(
            "ocr: tesseract binary disappeared after startup (%s); "
            "image scan degraded — install tesseract and restart", exc,
        )
        return ""
    except pytesseract.TesseractError as exc:
        log.warning("ocr: tesseract failed (%s); skipping image scan", exc)
        return ""
    return text if isinstance(text, str) else ""
