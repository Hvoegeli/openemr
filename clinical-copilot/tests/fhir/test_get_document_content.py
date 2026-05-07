"""Isolated tests for `app.fhir.adapter.get_document_content`.

The HTTP layers (FHIR GET, Binary fetch) are covered by an end-to-end
smoke against real OpenEMR. What's testable here without HTTP:

- The adapter walks `DocumentReference.content[].attachment` and pulls
  bytes from either the inline `data` field or the `url` field.
- The adapter renders bytes via `render_to_png_pages` and returns the
  pages as base64 strings under `data.pages_png_b64`.
- The `max_pages` cap kicks in when total page count exceeds the limit
  and `pages_truncated` is set on the result.
- A per-attachment render failure surfaces as `attachments[].error`
  rather than aborting the whole call.
- ACL: when `panel` does not include the document's patient, the
  adapter raises `PatientAccessDenied` instead of returning content.
"""

from __future__ import annotations

import base64
import io

import pytest
from PIL import Image

from app.fhir.adapter import get_document_content, get_document_pages


def _tiny_png_bytes() -> bytes:
    """Smallest legitimate PNG `image_to_png_pages` will accept — a 1x1
    RGB PIL render. Hand-rolled magic-bytes wouldn't survive Pillow's
    `Image.open(...).load()` call, so we generate via PIL itself."""
    img = Image.new("RGB", (1, 1), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _multi_page_pdf_bytes(num_pages: int) -> bytes:
    """Build an N-page text PDF for multi-page-per-attachment tests."""
    from reportlab.lib.pagesizes import LETTER  # noqa: PLC0415
    from reportlab.pdfgen import canvas  # noqa: PLC0415

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=LETTER)
    for i in range(num_pages):
        c.drawString(72, 720, f"page {i + 1}")
        c.showPage()
    c.save()
    return buf.getvalue()


def _png_data_uri_b64(png_bytes: bytes) -> str:
    """Base64 string for the FHIR Attachment.data field (no data: prefix)."""
    return base64.standard_b64encode(png_bytes).decode("ascii")


class FakeFhirClient:
    """In-memory stand-in for `FhirClient.get` / `get_raw` so the adapter
    runs end-to-end without an OpenEMR server. Records calls so tests can
    assert which path (`Binary/{id}` vs inline) the adapter took."""

    def __init__(
        self,
        *,
        documents: dict[str, dict],
        binaries: dict[str, tuple[bytes, str]] | None = None,
    ) -> None:
        self._documents = documents
        self._binaries = binaries or {}
        self.get_calls: list[str] = []
        self.get_raw_calls: list[str] = []

    async def get(self, path: str, params: dict | None = None) -> dict:
        self.get_calls.append(path)
        if path.startswith("DocumentReference/"):
            doc_id = path.removeprefix("DocumentReference/")
            try:
                return self._documents[doc_id]
            except KeyError as e:
                raise LookupError(f"no fake doc registered for {path!r}") from e
        raise LookupError(f"unmocked GET {path!r}")

    async def get_raw(self, path: str) -> tuple[bytes, str]:
        self.get_raw_calls.append(path)
        try:
            return self._binaries[path]
        except KeyError as e:
            raise LookupError(f"no fake binary registered for {path!r}") from e


def _doc(
    doc_id: str,
    *,
    patient_id: str = "pat-1",
    attachments: list[dict] | None = None,
) -> dict:
    return {
        "resourceType": "DocumentReference",
        "id": doc_id,
        "status": "current",
        "subject": {"reference": f"Patient/{patient_id}"},
        "date": "2026-05-05T12:00:00Z",
        "type": {"text": "Patient Information"},
        "category": [{"text": "Patient Information"}],
        "content": [{"attachment": a} for a in (attachments or [])],
    }


class TestInlineData:
    async def test_inline_png_returns_one_page(self) -> None:
        png = _tiny_png_bytes()
        client = FakeFhirClient(
            documents={
                "doc-1": _doc(
                    "doc-1",
                    attachments=[{
                        "title": "intake.png",
                        "contentType": "image/png",
                        "data": _png_data_uri_b64(png),
                    }],
                ),
            },
        )

        result = await get_document_content(
            client,  # type: ignore[arg-type]
            document_id="doc-1",
        )

        assert result["sources"] == ["DocumentReference/doc-1"]
        data = result["data"]
        assert data["document_id"] == "doc-1"
        assert data["patient_id"] == "pat-1"
        assert data["page_count"] == 1
        assert data["pages_truncated"] is False
        assert len(data["pages_png_b64"]) == 1
        assert isinstance(data["pages_png_b64"][0], str)
        # Re-decoding should produce a real PNG (Pillow re-encoded it).
        decoded = base64.standard_b64decode(data["pages_png_b64"][0])
        assert decoded.startswith(b"\x89PNG"), "page is not a PNG"
        assert data["attachments"][0]["page_count"] == 1
        # Inline path must not hit get_raw.
        assert client.get_raw_calls == []


class TestUrlReference:
    async def test_url_pointing_at_binary_endpoint(self) -> None:
        png = _tiny_png_bytes()
        client = FakeFhirClient(
            documents={
                "doc-2": _doc(
                    "doc-2",
                    attachments=[{
                        "title": "scan.png",
                        "contentType": "image/png",
                        "url": "Binary/bin-99",
                    }],
                ),
            },
            binaries={"Binary/bin-99": (png, "image/png")},
        )

        result = await get_document_content(
            client,  # type: ignore[arg-type]
            document_id="doc-2",
        )

        assert result["data"]["page_count"] == 1
        assert client.get_raw_calls == ["Binary/bin-99"]


class TestMaxPagesTruncation:
    async def test_two_attachments_clamped_to_one_page(self) -> None:
        png = _tiny_png_bytes()
        client = FakeFhirClient(
            documents={
                "doc-3": _doc(
                    "doc-3",
                    attachments=[
                        {
                            "title": "a.png",
                            "contentType": "image/png",
                            "data": _png_data_uri_b64(png),
                        },
                        {
                            "title": "b.png",
                            "contentType": "image/png",
                            "data": _png_data_uri_b64(png),
                        },
                    ],
                ),
            },
        )

        result = await get_document_content(
            client,  # type: ignore[arg-type]
            document_id="doc-3",
            max_pages=1,
        )

        assert result["data"]["page_count"] == 1
        assert result["data"]["pages_truncated"] is True
        # Both attachments are recorded; the second has page_count=0
        # because every page from it was dropped by the `max_pages` cap.
        # Sum of per-attachment page counts always equals the delivered
        # `data.page_count`, even under truncation.
        atts = result["data"]["attachments"]
        assert len(atts) == 2
        assert atts[0]["page_count"] == 1
        assert atts[0]["page_start"] == 1
        assert atts[1]["page_count"] == 0
        # Fully-truncated attachment keeps page_start=None — no global
        # page belongs to it.
        assert atts[1]["page_start"] is None
        assert sum(a["page_count"] for a in atts) == result["data"]["page_count"]


class TestMultiAttachmentPageNumbering:
    """Locks down the `page_start` mapping so a multi-attachment doc can
    surface "Attachment N — page K" labels in the viewer without
    re-walking the attachments list. Single-attachment docs always
    have `page_start == 1` (or None on a render-error attachment)."""

    async def test_two_attachments_two_pages_each(self) -> None:
        png = _tiny_png_bytes()
        client = FakeFhirClient(
            documents={
                "doc-multi": _doc(
                    "doc-multi",
                    attachments=[
                        {
                            "title": "cover.png",
                            "contentType": "image/png",
                            "data": _png_data_uri_b64(png),
                        },
                        {
                            "title": "lab.png",
                            "contentType": "image/png",
                            "data": _png_data_uri_b64(png),
                        },
                    ],
                ),
            },
        )

        result = await get_document_content(
            client,  # type: ignore[arg-type]
            document_id="doc-multi",
        )

        assert result["data"]["page_count"] == 2
        atts = result["data"]["attachments"]
        assert len(atts) == 2
        # First attachment's pages start at global page 1.
        assert atts[0]["page_count"] == 1
        assert atts[0]["page_start"] == 1
        # Second attachment's pages start at global page 2 (since the
        # first contributed 1 page).
        assert atts[1]["page_count"] == 1
        assert atts[1]["page_start"] == 2

    async def test_render_error_attachment_keeps_page_start_none(self) -> None:
        png = _tiny_png_bytes()
        client = FakeFhirClient(
            documents={
                "doc-mid-fail": _doc(
                    "doc-mid-fail",
                    attachments=[
                        {
                            "title": "cover.png",
                            "contentType": "image/png",
                            "data": _png_data_uri_b64(png),
                        },
                        {
                            # Unsupported MIME — render fails, attachment
                            # surfaces as `error` with page_count=0.
                            "title": "broken.docx",
                            "contentType": "application/vnd.docx",
                            "data": _png_data_uri_b64(png),
                        },
                        {
                            "title": "report.png",
                            "contentType": "image/png",
                            "data": _png_data_uri_b64(png),
                        },
                    ],
                ),
            },
        )

        result = await get_document_content(
            client,  # type: ignore[arg-type]
            document_id="doc-mid-fail",
        )

        atts = result["data"]["attachments"]
        assert len(atts) == 3
        assert atts[0]["page_start"] == 1 and atts[0]["page_count"] == 1
        # Failed attachment: zero pages, page_start=None. This
        # disambiguates "the attachment that owns global page X" lookups.
        assert atts[1]["page_start"] is None and atts[1]["page_count"] == 0
        # Third attachment shifts to page 2 — failed-attachment "skips"
        # the global numbering entirely, so the global → local lookup
        # never resolves into the failed attachment.
        assert atts[2]["page_start"] == 2 and atts[2]["page_count"] == 1

    async def test_global_to_local_lookup_round_trips(self) -> None:
        """Spec lock-in: for any global page in `pages_png_b64`, exactly
        one attachment owns it, and `(attachment_index, local_page)`
        recovers cleanly from `(page_start, page_count)`."""
        png = _tiny_png_bytes()
        client = FakeFhirClient(
            documents={
                "doc-multi3": _doc(
                    "doc-multi3",
                    attachments=[
                        {"title": f"a{i}.png", "contentType": "image/png",
                         "data": _png_data_uri_b64(png)}
                        for i in range(3)
                    ],
                ),
            },
        )

        result = await get_document_content(
            client,  # type: ignore[arg-type]
            document_id="doc-multi3",
        )

        atts = result["data"]["attachments"]
        total = result["data"]["page_count"]
        assert total == 3

        for global_page in range(1, total + 1):
            # Find the unique attachment that owns this page.
            owning = [
                a for a in atts
                if a["page_start"] is not None
                and a["page_start"] <= global_page
                < a["page_start"] + a["page_count"]
            ]
            assert len(owning) == 1, (
                f"global page {global_page} should belong to exactly one "
                f"attachment, found {len(owning)}"
            )

    async def test_partial_truncation_keeps_page_start_correct(self) -> None:
        """Partial truncation: a 3-page attachment that delivers only 2
        because `max_pages` cuts off mid-attachment. `page_count`
        reflects the delivered count; `page_start` still points at the
        first delivered page (it was stamped before the cap fired)."""
        # First attachment: 1-page PNG. Second attachment: 3-page PDF.
        # max_pages=3 lets through all of attachment 0 + only the first
        # 2 pages of attachment 1.
        png = _tiny_png_bytes()
        pdf = _multi_page_pdf_bytes(3)
        client = FakeFhirClient(
            documents={
                "doc-partial": _doc(
                    "doc-partial",
                    attachments=[
                        {"title": "cover.png", "contentType": "image/png",
                         "data": _png_data_uri_b64(png)},
                        {"title": "report.pdf", "contentType": "application/pdf",
                         "data": _png_data_uri_b64(pdf)},
                    ],
                ),
            },
        )
        result = await get_document_content(
            client,  # type: ignore[arg-type]
            document_id="doc-partial",
            max_pages=3,
        )

        atts = result["data"]["attachments"]
        assert result["data"]["page_count"] == 3
        assert result["data"]["pages_truncated"] is True
        assert atts[0]["page_count"] == 1
        assert atts[0]["page_start"] == 1
        # Partial: 2 of 3 pages delivered. page_start points at the
        # first page we DID deliver — not at where the third (dropped)
        # page would have been.
        assert atts[1]["page_count"] == 2
        assert atts[1]["page_start"] == 2

    async def test_get_document_pages_exposes_attachment_context(self) -> None:
        """The viewer endpoint should mirror the agent tool: each rendered
        page carries `attachment_index` + `attachment_page` (1-indexed
        local), and the top-level `attachments` summary carries
        `page_start`/`page_count` for each attachment."""
        png = _tiny_png_bytes()
        client = FakeFhirClient(
            documents={
                "doc-viewer": _doc(
                    "doc-viewer",
                    attachments=[
                        {"title": "cover.png", "contentType": "image/png",
                         "data": _png_data_uri_b64(png)},
                        {"title": "lab.png", "contentType": "image/png",
                         "data": _png_data_uri_b64(png)},
                    ],
                ),
            },
        )
        result = await get_document_pages(
            client,  # type: ignore[arg-type]
            document_id="doc-viewer",
        )

        pages = result["data"]["pages"]
        assert len(pages) == 2
        # First page belongs to attachment 0, page 1 within that attachment.
        assert pages[0]["page"] == 1
        assert pages[0]["attachment_index"] == 0
        assert pages[0]["attachment_page"] == 1
        # Second page belongs to attachment 1, page 1 within that attachment.
        assert pages[1]["page"] == 2
        assert pages[1]["attachment_index"] == 1
        assert pages[1]["attachment_page"] == 1

        atts = result["data"]["attachments"]
        assert len(atts) == 2
        assert atts[0]["page_start"] == 1 and atts[0]["page_count"] == 1
        assert atts[1]["page_start"] == 2 and atts[1]["page_count"] == 1


class TestPerAttachmentError:
    async def test_unsupported_mime_does_not_abort_call(self) -> None:
        png = _tiny_png_bytes()
        client = FakeFhirClient(
            documents={
                "doc-4": _doc(
                    "doc-4",
                    attachments=[
                        {
                            "title": "broken.docx",
                            "contentType": "application/vnd.docx",
                            "data": _png_data_uri_b64(png),
                        },
                        {
                            "title": "good.png",
                            "contentType": "image/png",
                            "data": _png_data_uri_b64(png),
                        },
                    ],
                ),
            },
        )

        result = await get_document_content(
            client,  # type: ignore[arg-type]
            document_id="doc-4",
        )

        atts = result["data"]["attachments"]
        assert len(atts) == 2
        assert "error" in atts[0]
        assert atts[0]["page_count"] == 0
        assert "error" not in atts[1]
        assert atts[1]["page_count"] == 1
        assert result["data"]["page_count"] == 1


class TestNoFetchableSource:
    async def test_attachment_with_neither_data_nor_url(self) -> None:
        client = FakeFhirClient(
            documents={
                "doc-5": _doc(
                    "doc-5",
                    attachments=[{
                        "title": "metadata-only.pdf",
                        "contentType": "application/pdf",
                    }],
                ),
            },
        )

        result = await get_document_content(
            client,  # type: ignore[arg-type]
            document_id="doc-5",
        )

        atts = result["data"]["attachments"]
        assert atts[0]["error"] == "no inline data or fetchable url"
        assert result["data"]["page_count"] == 0


class TestAcl:
    async def test_panel_match_returns_content(self) -> None:
        png = _tiny_png_bytes()
        client = FakeFhirClient(
            documents={
                "doc-6": _doc(
                    "doc-6",
                    patient_id="pat-7",
                    attachments=[{
                        "title": "x.png",
                        "contentType": "image/png",
                        "data": _png_data_uri_b64(png),
                    }],
                ),
            },
        )

        result = await get_document_content(
            client,  # type: ignore[arg-type]
            document_id="doc-6",
            panel=frozenset({"pat-7"}),
        )

        assert result["data"]["page_count"] == 1

    async def test_panel_mismatch_raises(self) -> None:
        from app.access_control import PatientAccessDenied

        client = FakeFhirClient(
            documents={
                "doc-7": _doc(
                    "doc-7",
                    patient_id="pat-out",
                    attachments=[],
                ),
            },
        )

        with pytest.raises(PatientAccessDenied):
            await get_document_content(
                client,  # type: ignore[arg-type]
                document_id="doc-7",
                panel=frozenset({"pat-in"}),
            )

    async def test_panel_none_skips_acl(self) -> None:
        client = FakeFhirClient(
            documents={"doc-8": _doc("doc-8", patient_id="pat-anyone")},
        )

        # Should not raise even though panel is unset / no patient match.
        result = await get_document_content(
            client,  # type: ignore[arg-type]
            document_id="doc-8",
            panel=None,
        )
        assert result["data"]["patient_id"] == "pat-anyone"

    async def test_panel_provided_doc_without_patient_subject_fails_closed(
        self,
    ) -> None:
        """A DocumentReference with no `subject.reference` MUST be denied
        when a panel is enforced. We have no basis to attribute it to a
        patient inside the panel, so fail closed."""
        from app.access_control import PatientAccessDenied

        client = FakeFhirClient(
            documents={
                "doc-orphan": {
                    "resourceType": "DocumentReference",
                    "id": "doc-orphan",
                    "status": "current",
                    "type": {"text": "System Document"},
                    "category": [{"text": "System"}],
                    "date": "2026-05-05T12:00:00Z",
                    "content": [],
                    # NB: no `subject` field at all.
                },
            },
        )

        with pytest.raises(PatientAccessDenied):
            await get_document_content(
                client,  # type: ignore[arg-type]
                document_id="doc-orphan",
                panel=frozenset({"pat-1"}),
            )

    async def test_panel_none_doc_without_patient_subject_returns_content(
        self,
    ) -> None:
        """Without a panel (admin / unauthenticated CLI smokes), a doc
        with no Patient subject is still readable. The denial above is
        specifically a panel-enforcement artifact."""
        client = FakeFhirClient(
            documents={
                "doc-orphan2": {
                    "resourceType": "DocumentReference",
                    "id": "doc-orphan2",
                    "status": "current",
                    "type": {"text": "System Document"},
                    "category": [{"text": "System"}],
                    "date": "2026-05-05T12:00:00Z",
                    "content": [],
                },
            },
        )

        result = await get_document_content(
            client,  # type: ignore[arg-type]
            document_id="doc-orphan2",
            panel=None,
        )
        assert result["data"]["patient_id"] is None


class TestAbsoluteUrlRefusal:
    async def test_external_absolute_url_returns_none_bytes(self) -> None:
        """Absolute URLs that don't match our FHIR base must NOT be
        fetched — `client.get_raw` always prepends our base, and we
        don't want to fan a system access token out to a third-party
        host the FHIR server happens to reference. The attachment
        surfaces as a per-attachment fetch error instead."""
        client = FakeFhirClient(
            documents={
                "doc-ext": _doc(
                    "doc-ext",
                    attachments=[{
                        "title": "external.png",
                        "contentType": "image/png",
                        "url": "https://attacker.example.com/Binary/exfil",
                    }],
                ),
            },
        )

        result = await get_document_content(
            client,  # type: ignore[arg-type]
            document_id="doc-ext",
        )

        # No bytes were fetched; the attachment is recorded with an error
        # and the document delivered zero pages.
        assert client.get_raw_calls == []
        atts = result["data"]["attachments"]
        assert len(atts) == 1
        assert "error" in atts[0]
        assert result["data"]["page_count"] == 0
