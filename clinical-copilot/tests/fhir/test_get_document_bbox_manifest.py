"""Isolated tests for `app.fhir.adapter.get_document_bbox_manifest`.

The bbox manifest is what the Supporting-Documents PDF-overlay UI
fetches when the user clicks a chart citation. The contract the
frontend depends on:

  - Every chart resource (allergy / med / condition) carrying a
    `[copilot-source: DocumentReference/<id>; bbox=...]` tag in its
    note/comments field MUST appear in `data.facts`, regardless of
    whether the bbox JSON is present.
  - When bbox is absent, `bbox: null` is emitted (NOT the entry
    omitted) so the frontend can render a page-band fallback overlay.
  - `resource_ref` and `label` are always populated so the frontend can
    map an overlay back to the citation that opened the doc.

These tests pin that contract so a future writer change that
accidentally drops bbox-less facts from the manifest doesn't silently
break the page-band overlay.
"""

from __future__ import annotations

from app.fhir.adapter import get_document_bbox_manifest


class FakeFhirClient:
    """In-memory FHIR stand-in for the manifest adapter.

    The adapter calls `client.get` once for the source DocumentReference
    and `client.search` once per resource type (Allergy / Med /
    Condition). This stub serves both from in-memory dicts.
    """

    def __init__(
        self,
        *,
        documents: dict[str, dict],
        searches: dict[str, list[dict]] | None = None,
    ) -> None:
        self._documents = documents
        self._searches = searches or {}
        self.get_calls: list[str] = []
        self.search_calls: list[tuple[str, dict | None]] = []

    async def get(self, path: str, params: dict | None = None) -> dict:
        self.get_calls.append(path)
        if path.startswith("DocumentReference/"):
            doc_id = path.removeprefix("DocumentReference/")
            try:
                return self._documents[doc_id]
            except KeyError as e:
                raise LookupError(f"no fake doc registered for {path!r}") from e
        raise LookupError(f"unmocked GET {path!r}")

    async def search(self, resource_type: str, params: dict | None = None) -> list[dict]:
        self.search_calls.append((resource_type, params))
        return list(self._searches.get(resource_type, []))


def _doc(doc_id: str, *, patient_id: str = "pat-1") -> dict:
    return {
        "resourceType": "DocumentReference",
        "id": doc_id,
        "status": "current",
        "subject": {"reference": f"Patient/{patient_id}"},
        "date": "2026-05-05T12:00:00Z",
        "type": {"text": "Patient Information"},
        "category": [{"text": "Patient Information"}],
        "content": [],
    }


def _condition_with_tag(
    *,
    cond_id: str,
    title: str,
    note_text: str,
) -> dict:
    return {
        "resourceType": "Condition",
        "id": cond_id,
        "code": {"text": title},
        "note": [{"text": note_text}],
    }


class TestBboxManifestContract:
    async def test_facts_with_bbox_are_emitted_with_coords(self) -> None:
        """Baseline: a condition tagged with full bbox shows up in the
        manifest with all five bbox fields populated.
        """
        client = FakeFhirClient(
            documents={"doc-1": _doc("doc-1")},
            searches={
                "Condition": [
                    _condition_with_tag(
                        cond_id="cond-1",
                        title="Type 2 diabetes",
                        note_text=(
                            "[copilot-source: DocumentReference/doc-1; "
                            'bbox={"page":2,"x":52,"y":430,"width":820,"height":28}]'
                        ),
                    ),
                ],
            },
        )
        result = await get_document_bbox_manifest(
            client, document_id="doc-1", panel=None,  # type: ignore[arg-type]
        )
        facts = result["data"]["facts"]
        assert len(facts) == 1
        f = facts[0]
        assert f["resource_type"] == "Condition"
        assert f["resource_ref"] == "Condition/cond-1"
        assert f["label"] == "Type 2 diabetes"
        assert f["bbox"] == {
            "page": 2, "x": 52, "y": 430, "width": 820, "height": 28,
        }

    async def test_facts_without_bbox_still_appear_with_bbox_null(self) -> None:
        """Page-band-fallback contract: a condition tagged with the
        copilot-source ref but no bbox JSON must still appear in
        `data.facts` with `bbox: null`. The frontend uses this to render
        a page-wide band on page 1 instead of skipping the citation.

        Without this guarantee, every fax_packet / referral_letter /
        hl7_message citation would land on a blank PDF.
        """
        client = FakeFhirClient(
            documents={"doc-1": _doc("doc-1")},
            searches={
                "Condition": [
                    _condition_with_tag(
                        cond_id="cond-1",
                        title="Type 2 diabetes",
                        note_text="[copilot-source: DocumentReference/doc-1]",
                    ),
                ],
            },
        )
        result = await get_document_bbox_manifest(
            client, document_id="doc-1", panel=None,  # type: ignore[arg-type]
        )
        facts = result["data"]["facts"]
        assert len(facts) == 1, (
            "page-band fallback contract: bbox-less facts must NOT be "
            "filtered out of the manifest. Frontend depends on resource_ref + "
            "label to render a page-wide band overlay."
        )
        f = facts[0]
        assert f["resource_type"] == "Condition"
        assert f["resource_ref"] == "Condition/cond-1"
        assert f["label"] == "Type 2 diabetes"
        assert f["bbox"] is None

    async def test_mixed_bbox_and_no_bbox_facts_both_emitted(self) -> None:
        """Realistic Kowalski-shaped case: one doc with a mix of
        with-bbox and bbox=None facts. Both must surface so the frontend
        can render precise overlays for some and page-bands for others.
        """
        client = FakeFhirClient(
            documents={"doc-1": _doc("doc-1")},
            searches={
                "Condition": [
                    _condition_with_tag(
                        cond_id="cond-good",
                        title="Hypertension",
                        note_text=(
                            "[copilot-source: DocumentReference/doc-1; "
                            'bbox={"page":1,"x":10,"y":10,"width":100,"height":20}]'
                        ),
                    ),
                    _condition_with_tag(
                        cond_id="cond-bandable",
                        title="Diabetes",
                        note_text="[copilot-source: DocumentReference/doc-1]",
                    ),
                ],
            },
        )
        result = await get_document_bbox_manifest(
            client, document_id="doc-1", panel=None,  # type: ignore[arg-type]
        )
        facts = {f["resource_ref"]: f for f in result["data"]["facts"]}
        assert "Condition/cond-good" in facts
        assert "Condition/cond-bandable" in facts
        assert facts["Condition/cond-good"]["bbox"] is not None
        assert facts["Condition/cond-bandable"]["bbox"] is None

    async def test_resources_with_no_tag_at_all_are_excluded(self) -> None:
        """Sanity: a hand-entered condition (no tag) must NOT appear in
        the manifest — that would attribute it to a doc it didn't come
        from. Only tagged rows are bbox-manifest fodder.
        """
        client = FakeFhirClient(
            documents={"doc-1": _doc("doc-1")},
            searches={
                "Condition": [
                    {
                        "resourceType": "Condition",
                        "id": "cond-untagged",
                        "code": {"text": "Asthma"},
                        # No note / no comments.
                    },
                ],
            },
        )
        result = await get_document_bbox_manifest(
            client, document_id="doc-1", panel=None,  # type: ignore[arg-type]
        )
        assert result["data"]["facts"] == []
