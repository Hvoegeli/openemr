# W2_ARCHITECTURE — Multimodal Evidence Agent

> Companion to [`docs/prds/week-2-multimodal-evidence.md`](docs/prds/week-2-multimodal-evidence.md).
> The PRD says *what* Week 2 must deliver. This document says *how* we deliver
> it — the architectural decisions, the schemas, the worker graph, the
> retrieval design, the eval gate, the risks, and the explicit reversals (or
> non-reversals) of Week 1's deliberate scope choices.
>
> **Status:** MVP shipped on `copilot--branch-2` 2026-05-04 — extraction
> pipeline + upload UI + BM25 RAG + agent tool wiring all live and verified
> end-to-end against real OpenEMR + real Claude. Sections labeled
> **MVP-shipped** below describe the as-built state; sections labeled
> **deferred post-MVP** describe the PRD-faithful target shape we'll iterate
> toward through Sunday 2026-05-10.

---

## 0. Decisions locked

Quick-reference table of choices already made (rationale in the relevant
section below):

| Area | MVP-shipped decision | PRD-faithful target | Section |
|---|---|---|---|
| VLM | Claude Sonnet 4.6 vision, tool-use forced JSON output | + bounding-box overlay UI (deferred) | §4 |
| Retrieval | **BM25 + LLM-rerank** (`rank-bm25` + Claude Haiku) | + FAISS dense (deferred) | §3 |
| Vector store | none (BM25 only) | FAISS in-memory | §3 |
| Guideline corpus | hand-curated YAML, **12 chunks** USPSTF + ADA | + AHA cardiology corpus (deferred) | §3 |
| Document persistence | OpenEMR multipart upload + FHIR-GET resolve; SHA-256 dedupe | (no change planned) | §4.5 |
| Recommendation tone | guideline-grounded recs permitted under R2 carve-out (always-on; not toggle-gated) | per-conversation toggle (deferred) | §6 |
| Upload UI | minimal HTML form at `/upload` with patient + doc-type dropdowns | wire into chat sidebar | §4.6 |
| Patient-mismatch verification | **deferred** (PRD §5.4) | confirm-patient supervisor node | §5.4 |
| Round-trip extraction → FHIR Observation/Condition/etc. | **deferred** (writer can do `DocumentReference` only today) | adapter layer over standard non-FHIR API | §4.5 |
| Branch | `copilot--branch-2`; no auto-merge to master | — | — |
| Preview deploy | Hetzner cloudflared tunnel on `:8001`, isolated from master `:8000` | — | — |
| Doc location | `W2_ARCHITECTURE.md` at repo root | — | — |

---

## 1. Executive summary

**Scenario.** A doctor in clinic uploads a paper lab report or a filled
intake form to Co-pilot. Within ~10 seconds the document is persisted to
OpenEMR (FHIR `DocumentReference`), its clinical contents are parsed into
strict-typed JSON with per-field source citations, and the chat agent —
already grounded in the patient's chart — can now also cite published
guidelines (USPSTF + ADA) when the doctor asks "what should I screen for"
or "what does the guideline say." Every clinical claim — chart fact OR
guideline reference — carries an inline `[Type/ID]` citation that the
existing Week 1 validator gate enforces.

**What's net-new vs. Week 1.** Week 1 was a chart summarizer over
structured FHIR data. Week 2 adds two capabilities the chart layer can't
provide on its own:
- **Multimodal document extraction.** The Phase 1 Pydantic schemas + the
  Phase 2 `attach_and_extract` pipeline turn an arbitrary PDF or image
  into typed, citation-bearing JSON via Claude Sonnet 4.6 vision +
  forced tool-use. The persisted FHIR `DocumentReference` id flows into
  every per-field citation so the chat can later trace any extracted
  fact back to the page it came from.
- **Evidence retrieval over published guidelines.** A 12-chunk
  hand-curated corpus of USPSTF + ADA recommendations, indexed with
  BM25 (`rank-bm25`, pure Python, ~3KB index). The agent's
  `retrieve_guidelines` tool returns ranked chunks; cited inline as
  `[Guideline/<chunk_id>]` alongside chart citations.

**Core architectural moves.**
1. **Pydantic-validated tool-use** for extraction. The vision call is
   forced into a `record_lab_report` / `record_intake_form` tool whose
   input schema is `LabReport.model_json_schema()` /
   `IntakeForm.model_json_schema()`. The model can't smuggle extra
   fields (`extra="forbid"`) and can't return free-text JSON the parser
   would have to repair.
2. **SHA-256 idempotency in the upload filename** — `sha256-<hex>__<orig>`
   — so re-uploading the same file deduplicates via FHIR-GET without a
   parallel local table. OpenEMR stays the single source of truth.
3. **Two citation namespaces, one validator.** `[FHIRType/ID]` for
   chart facts (Week 1) + `[Guideline/<chunk_id>]` for guideline
   quotes (Week 2). The existing validator regex
   `[A-Z][a-zA-Z]+/[a-zA-Z0-9._-]+` accepts both without modification —
   guideline citations are additive, not a parallel surface.
4. **No new agent.** The Phase 4.1 wiring adds `retrieve_guidelines` as
   one more entry in the existing Week 1 `TOOLS` list. The same
   LangGraph + the same validator + the same observability stack apply.
   The "supervisor + worker" multi-agent shape from the PRD §5 is
   deferred — for the MVP the chat agent calls retrieval directly
   alongside its existing FHIR tools, and the live e2e smoke shows it
   does so correctly with no orchestration code added.

**Verification line.** Three layers, each independently fail-closed:
schema validation at the extraction boundary (Pydantic with
`extra="forbid"`), citation-id validation in the response gate
(`find_invalid_citations` rejects fabricated `Guideline/<chunk_id>`s),
and the live end-to-end smoke (`scripts/smoke_e2e_mvp.py`) which
asserts both citation namespaces appear and refuses to fabricate when
the corpus has no relevant chunk. Live verification on 2026-05-04 saw
the agent correctly cite 3 guideline chunks + 1 FHIR ref and honestly
admit corpus gaps rather than make up an HbA1c-target chunk.

**Deliberately out of scope for the MVP** (deferred, not abandoned —
documented per-section below): dense embeddings (BM25 + LLM-rerank
shipped, see §3.3), AHA corpus, PDF bounding-box overlay UI,
patient-mismatch confirmation flow, extraction → FHIR-Observation/
Condition write-back, `recommendation_mode` toggle UI surface (the R2
prompt carve-out is always-on for guideline-backed recs; we kept the
toggle out of the UI for time).

---

## 2. System context — what changed from Week 1

**TODO** — diff vs. Week 1's §2:

- New stakeholders / surfaces (front desk uploading documents).
- New data shapes (PDFs, intake forms, guideline corpus).
- New trust boundary (Claude vision reads document images).
- What stays the same (auth, sessions, audit log, ACL, observability).

---

## 3. RAG design — the evidence layer

### 3.1 Corpus — MVP-shipped

**Decision:** USPSTF + ADA, hand-curated YAML at
[`clinical-copilot/data/guidelines/corpus.yaml`](clinical-copilot/data/guidelines/corpus.yaml),
**12 chunks** (6 USPSTF + 6 ADA), one chunk per recommendation/section.
Each chunk is ≤200 words and carries `chunk_id` (snake_case stable
slug, doubles as the citation id), `source`, `title`, `year`, `url`,
`topic_tags`, and `text`.

- **USPSTF — 6 chunks** — statin / aspirin primary prevention 2022,
  lipid screening 2016, diabetes screening 2021, colorectal cancer
  screening 2021, blood pressure screening 2021. Public-domain
  language paraphrased to ≤200 words to keep the prompt window cheap;
  every chunk carries the canonical USPSTF URL for the doctor to read
  the full source.
- **ADA Standards of Care 2024 — 6 chunks** — glycemic targets,
  pharmacotherapy first-line, lipid management in diabetes, hypertension
  in diabetes, CKD in diabetes, aspirin in diabetes. Same paraphrase
  policy.

**Why 12, not 100.** Hand-curation makes the corpus small and
inspectable for the demo. Every chunk has been read by a human, every
chunk's BM25-tokenized form has been smoke-checked
([`scripts/smoke_retrieve.py`](clinical-copilot/scripts/smoke_retrieve.py)),
and a clinically-relevant top-1 is returned for every demo query
(CRC scored 11.72 vs next-best 1.24 — wide margin). Per the PRD's
pitfall list: *"trying to support five document types before two work
reliably."*

**Chunking strategy: per-recommendation.** Each chunk corresponds to
one self-contained guideline statement (the same shape USPSTF / ADA
publish). No overlap, no sliding window, no semantic chunker — the
guidelines themselves are already chunked the way a clinician thinks
about them.

### 3.1.1 Corpus extension — deferred post-MVP

- **AHA** (American Heart Association) cardiology guidelines (lipid
  management, ASCVD risk calculator details, BP targets). Fills the
  gap USPSTF only partially covers. Hold off until MVP grades green —
  two corpora reliably indexed beats three corpora half-indexed.
- **Move from hand-curated to PDF ingestion** of full USPSTF / ADA
  publications. Requires a structured-extraction pipeline of its own
  (which the Phase 2.1 `attach_and_extract` could service against
  guideline PDFs — same shape as a lab report extraction).

### 3.2 Indexing pipeline — MVP-shipped

**Decision:** in-process **BM25 only** (`rank-bm25` 0.2.x — pure Python,
zero transitives), built eagerly at module import in
[`clinical-copilot/app/guidelines/retrieve.py`](clinical-copilot/app/guidelines/retrieve.py).

- **Index:** `BM25Okapi` over the concatenated `text + title + topic_tags`
  surface of each chunk. Including title and tags lets a query like
  "aspirin primary prevention" surface a chunk even when the body uses
  synonyms.
- **Tokenization:** `re.findall(r"[a-z0-9]+", text.lower())`. Drops
  punctuation, preserves alphanumerics. Locked in by tests
  ([`tests/guidelines/test_retrieve.py::TestTokenize`](clinical-copilot/tests/guidelines/test_retrieve.py)).
- **Module-level eager load** means a malformed YAML or duplicate
  `chunk_id` fails loudly at import (not at first query). For 12 chunks
  this is essentially free; for thousands it would move to lazy load.
- **No vector store, no embedding model.** The PRD-faithful design (FAISS
  + dense embeddings + BGE rerank, see §3.3 deferred) is post-MVP.

### 3.2.1 Indexing — deferred post-MVP (PRD-faithful target)

- **Embedding model:** local sentence-transformer
  (`all-MiniLM-L6-v2`, ~80 MB) so we don't add a vendor dep / BAA.
  Encode + index at startup, persist to disk.
- **FAISS in-memory** (`faiss-cpu`) for dense vectors. Rebuild
  on-startup is sub-second for corpus sizes well past 12 chunks.
- **Hybrid retrieve** layered over the existing BM25 — see §3.3
  deferred.

### 3.3 Retrieval — MVP-shipped

**Public surface:** `retrieve_guidelines(query: str, k: int = 3)
-> list[RetrievalHit]`. Two-stage pipeline behind one function:

- **Stage 1 (BM25 recall filter):** tokenize the query, score every
  chunk, take the top `BM25_CANDIDATE_POOL = 8` with score > 0. This is
  the cheap exhaustive pass that makes sure no relevant chunk gets
  silently dropped before the rerank sees it.
- **Stage 2 (LLM rerank):** Claude Haiku scores each candidate's
  semantic relevance to the query 0.0–1.0. Top-`k` by rerank score is
  returned. Implementation in [`app/guidelines/rerank.py`](clinical-copilot/app/guidelines/rerank.py).

`RetrievalHit.score` is the rerank score (the authoritative ordering
signal); `RetrievalHit.bm25_score` carries the stage-1 score for
debugging. Empty/whitespace queries return `[]`. `k <= 0` returns `[]`.
A test-only `enable_rerank=False` kwarg skips the API call and falls
back to BM25 ordering — used by the corpus-shape tests.

The agent's tool wrapper
([`app/agent/tools.py::_retrieve_guidelines_impl`](clinical-copilot/app/agent/tools.py))
emits `sources: ["Guideline/<chunk_id>", ...]` so the existing citation
validator accepts inline `[Guideline/<chunk_id>]` references.

**Why LLM-as-reranker (Claude Haiku) instead of a cross-encoder
(BGE / Cohere).** The PRD allows "Cohere Rerank or an equivalent
reranker." We chose LLM-as-reranker for three reasons:

1. **Dependency footprint.** A cross-encoder requires
   `sentence-transformers` + `torch` + ~300MB of model weights.
   PyPI has no torch wheel that is both CPython-3.14-compatible AND
   macOS-x86\_64-compatible right now (the dev box's stack), so
   committing to that path forces either a Python downgrade or a
   Linux-only Hetzner-only retrieval stage. We already pay for a
   Claude API key — adding a Haiku call costs nothing new in setup.
2. **Cost.** Haiku rerank costs ~$0.001 per query (vs. Cohere's $2/1k).
   The per-request budget allows it comfortably.
3. **Latency.** ~500–800ms per rerank call, in the same order of
   magnitude as a hosted cross-encoder API. Local cross-encoder is
   faster but requires the model download and warm-up.

A swap to BGE / Cohere Rerank is a single-module replacement (the
`rerank()` function signature is the contract); the choice is reversible
when corpus size or quality bar warrants it.

### 3.3.1 Retrieval — deferred post-MVP (PRD-faithful target)

Dense embeddings (BGE-small or text-embedding-3-small over the same
corpus, cosine merged with BM25 before rerank) is the next iteration.
For a 12-chunk corpus the recall benefit over BM25-alone is marginal
because keyword overlap is essentially exhaustive — BM25 + rerank
already covers 8/12 candidates per query. Dense pays off when the
corpus grows past ~100 chunks where some clinically-relevant chunks
share no keywords with the query. The retrieval-stage interface
already accommodates a dense layer (we'd union BM25 candidates with
dense candidates before passing to the existing rerank).

### 3.4 Citation contract for retrieved evidence

Each retrieved chunk surfaces with:

```
{
  source_type: "guideline",
  source_id: "uspstf-aspirin-cvd-prevention",
  page_or_section: "Recommendation Statement",
  field_or_chunk_id: "uspstf-aspirin-42",
  quote_or_value: "<the actual quoted sentence>"
}
```

This lines up with the same citation shape the document-extraction tools use
(§4.3) — single citation type across the system.

---

## 4. Document ingestion + extraction

### 4.1 Tool signature — MVP-shipped

**Decision:** [`app/extraction/extract.py::attach_and_extract`](clinical-copilot/app/extraction/extract.py).

```python
async def attach_and_extract(
    *,
    file_bytes: bytes,                 # raw bytes (not file_path — works for HTTP uploads)
    filename: str,                     # original name for the OpenEMR upload (gets SHA-256 prefix)
    doc_type: DocumentType,            # Literal["lab_pdf", "intake_form"]
    patient_uuid: str,                 # FHIR Patient UUID; ACL gate at /api/upload
    mime_type: str = "application/pdf",
    writer: OpenEMRWriter | None = None,        # optional injection for tests
    anthropic_client: AsyncAnthropic | None = None,  # optional injection for tests
    model: str = "claude-sonnet-4-6",
) -> AttachAndExtractResult:           # bundles {extracted, write_result}
```

The file is read once (caller already has bytes from `UploadFile.read()`),
persisted to OpenEMR (Phase 1.3 writer — multipart upload + SHA-256 dedupe
+ FHIR-GET resolve), then sent to Claude vision with the resolved
`DocumentReference/{uuid}` id wired into every Citation as
`source_document_id`. The persisted id is also exposed as
`result.reference_id` for the UI.

**ACL.** The `/api/upload` endpoint
([`app/main.py::api_upload`](clinical-copilot/app/main.py)) gates
`patient_uuid` via the same `_check_patient_access` helper the chat uses
— off-panel patients get 404 (not 403, to avoid leaking existence). The
`attach_and_extract` function itself is ACL-agnostic; the gate is at the
HTTP boundary.

**`doc_type` extension.** Adding a new type means three updates: append
to `DocumentType` Literal in `app/extraction/schemas.py`, append to
`DOC_TYPE_LABELS` (UI dropdown source), append to `DOC_CATEGORIES` in
`app/fhir/writer.py` (OpenEMR category-path map), and add a new
`record_<type>` tool entry to `_TOOL_BUILDERS` in
`app/extraction/vision.py`. The drift test in
`tests/extraction/test_schemas.py::TestDocTypeConstantAlignment` locks
the first three in sync.

### 4.2 Schemas — MVP-shipped

Final shapes live at [`app/extraction/schemas.py`](clinical-copilot/app/extraction/schemas.py)
(216 lines, 64 isolated tests at
[`tests/extraction/test_schemas.py`](clinical-copilot/tests/extraction/test_schemas.py)).
Top-level shape is a discriminated union:

```python
ExtractedDocument = Annotated[
    Union[LabReport, IntakeForm],
    Field(discriminator="document_type"),
]
```

Every model uses `model_config = ConfigDict(extra="forbid")` so the VLM
can't smuggle hallucinated fields. Every required string identifier
uses `Field(min_length=1)` — empty strings are structurally
indistinguishable from missing data and would silently pass downstream
`citation_present` checks.

Sub-types:
- **`Citation`** — exactly the PRD §5 shape:
  `{source_type, source_id, page_or_section, field_or_chunk_id,
  quote_or_value, bbox?}`. `bbox: BoundingBox | None` — populated when
  the source supports visual overlay (lab PDFs, intake forms), omitted
  for sources that don't (guideline text, FHIR resources).
- **`LabResult`** — `test_name`, `value: float | str` (qualitative
  results like "positive" stay strings), `unit`, `reference_range`,
  `collection_date: date`, `abnormal_flag: Literal["H","L","N","C"]
  | None`, `source_citation: Citation`.
- **`LabReport`** — `results: list[LabResult]` with `min_length=1`
  (an empty lab PDF is a failed extraction, not a successful one);
  `source_document_id`; optional `facility`, `ordering_provider`.
- **`Demographics`**, **`Medication`**, **`Allergy`**,
  **`FamilyHistoryItem`** — each carries its own `source_citation`.
- **`IntakeForm`** — discriminator literal + `demographics`,
  `chief_concern`, three list fields with `default_factory=list` (a
  patient with no current meds is fine; the VLM returns `[]` rather
  than omitting the key).

**Optionality policy.** Required fields are required; optional fields
are explicit. The VLM is steered not to invent fields it can't see;
omitting an optional field is the right behavior, returning `""` is
not (the latter would slip past `citation_present` checks).

**Tests:** 64 isolated tests covering required-field-rejection
(parametrized over every required field), empty-string rejection on
every `min_length=1` field, discriminator dispatch + missing-discriminator
+ invalid-discriminator, BoundingBox boundary checks, and a fixture-
validates-schema test that locks the Phase 1.2 generator output to the
schemas.

### 4.3 Citation shape

**Decision:** the PRD's mandated shape:

```
{source_type, source_id, page_or_section, field_or_chunk_id, quote_or_value}
```

Used identically for FHIR resources (Week 1), extracted document fields
(Week 2), and retrieved guideline chunks (Week 2). Single shape across the
system.

### 4.4 VLM choice + bounding boxes

**Decision:** Claude Sonnet 4.6 vision, in-line bounding boxes.

- Same provider as Week 1, no new vendor BAA.
- Bounding boxes returned in the same structured response as the extracted
  fields, in PDF coordinate space.
- **Risk:** Claude vision bounding-box quality varies. Fallback if quality is
  insufficient: two-pass extraction (Tesseract or Textract returns boxes →
  Claude maps fields to OCR token IDs). Decision deferred until measured.

### 4.5 Persistence — round-tripping into OpenEMR

The PRD requires source documents and derived facts to round-trip without
duplicates or untraceable records.

**MVP-shipped:**

- **Source document → FHIR `DocumentReference`.** Implemented in
  [`app/fhir/writer.py::write_document_reference`](clinical-copilot/app/fhir/writer.py).
  OpenEMR's FHIR layer doesn't actually route `POST /DocumentReference`
  despite advertising `create` in its CapabilityStatement, so the write
  path pivots to the standard non-FHIR multipart endpoint at
  `POST /api/patient/{pid}/document?path={category}` and then
  resolves the new resource id via FHIR `GET /DocumentReference?patient=...`
  search on `attachment.title`.
- **Idempotency = SHA-256 in the upload filename.** The writer prepends
  the file's SHA-256 to the upload name (`sha256-<hex>__<original>`).
  Re-uploading the same bytes finds the existing match in the FHIR
  search and returns its id without re-POSTing —
  `result["created"] = False` flags the dedupe path. **No parallel
  local table** — OpenEMR stays the single source of truth for what's
  been persisted. Verified via `scripts/smoke_document_writer.py`
  against real OpenEMR (Chen lipid panel + intake produced distinct
  ids; re-upload returned the same id).
- **`DOC_CATEGORIES` pre-lowercased** — `lab_pdf → labreport`,
  `intake_form → patientinformation`. OpenEMR's `getLastIdOfPath` does
  case-sensitive equality against `replace(LOWER(name), ' ', '')`, so
  anything else silently orphans the upload (no `categories_to_documents`
  bridge row). Locked in by
  `tests/fhir/test_writer_document_reference.py::test_doc_categories_constant_pre_normalized`.

**Deferred post-MVP:**

- **Extraction → FHIR `Observation` per lab result**, linked to the
  source `DocumentReference` via `derivedFrom`. The schemas already
  carry every field the FHIR mapping needs (LOINC test_name, value,
  unit, reference_range, collection_date, abnormal_flag); a thin
  adapter would materialize them into FHIR Observations during the
  same `attach_and_extract` call.
- **Intake fields → `Patient` updates / `Condition` / `AllergyIntolerance`
  / `MedicationStatement`.** Most of these require the standard non-FHIR
  `/apis/default/api/` path (the seed scripts at
  `scripts/seed_demo_patients.py` already exercise the same endpoints
  for create-patient, allergies, problems, meds); a Phase 4.x writer-
  helper would consume an `IntakeForm` and emit the corresponding
  REST POSTs.
- **`derivedFrom` linking** — the FHIR R4 `Observation.derivedFrom`
  reference is the canonical join from extracted lab values back to
  their source `DocumentReference`. Persistence pipeline writes the
  `DocumentReference` first (already done; we have the id), then
  emits Observations with `derivedFrom = [{reference: DocumentReference/{id}}]`.

**Why the round-trip matters for the demo.** Without `derivedFrom`,
extracted Observations would float in the chart unmoored from their
source PDF. With it, a doctor can click an extracted A1c, see its
`derivedFrom` link, and pull up the original lab report — the
audit-trail-by-construction guarantee the PRD wants.

### 4.6 PDF bounding-box overlay (UI)

- **Renderer:** PDF.js (Mozilla's open-source JS PDF renderer).
- **Overlay:** second `<canvas>` layer stacked over the PDF render canvas;
  rectangles drawn from extracted bounding-box coords, transformed for
  current zoom + page.
- **Click handler:** clicking a citation in the chat scrolls the PDF to the
  cited page and highlights the cited region.

**TODO:** scope the JS engineering chunk; UI prototype before committing to
final layout.

---

## 5. Multi-agent graph — supervisor + workers

### 5.1 Worker shape — MVP-shipped

**Decision:** no separate worker agents. The Phase 4.1 wiring adds
`retrieve_guidelines` as one more entry in the existing Week 1 `TOOLS`
list at [`app/agent/tools.py`](clinical-copilot/app/agent/tools.py).
The chat agent calls retrieval directly alongside its existing FHIR
tools (`resolve_patient`, `get_patient_card`, `get_vital_trends`,
`clinical_flags`, etc.).

`attach_and_extract` is **not** in the agent's tool list — it's the
backend of the `/api/upload` HTTP endpoint, called by the upload form
(or any other client). The doctor uploads via the form; the chat
agent sees the resulting `DocumentReference` naturally via the
chart-summarizer's existing `get_notes_24h` and Supporting Documents
flow. Wiring `attach_and_extract` as a chat-callable tool would
require a way for the chat to attach a file mid-conversation — a Phase
2.X UX problem the upload form already solves.

### 5.2 Supervisor — MVP-shipped

**Decision:** the existing Week 1 LangGraph IS the supervisor. The
graph routes between `agent` (LLM call) and `tools` (tool execution)
exactly as before; adding one more tool to the list extended the
graph's reach without adding a graph node or a routing layer.

**Why no dedicated supervisor.** The PRD §5 gestures at a "supervisor +
worker" multi-agent pattern. For the MVP the chat agent already
chooses tools per turn (which is the supervisor's only real job); a
dedicated supervisor LLM call would double per-turn latency for no
visible improvement. If we add later workflows where the routing
policy outgrows what a tool-aware system prompt can express (e.g.
"pre-render the document, then summarize, then ask a follow-up"), a
LangGraph supervisor node remains a low-cost addition behind the same
public API.

**Routing observability.** Per-tool execution remains traced through the
existing observability path (`app/observability.py` + `traces.db`). Tool
calls are visible in the trace timeline; the agent's reasoning for
calling each tool is the assistant message that preceded it.

### 5.2.1 Multi-agent shape — deferred post-MVP (PRD-faithful target)

If/when the workflow grows past the single-graph shape, the natural
split is:
- **`intake_extractor`** worker owns `attach_and_extract`. Triggered by
  upload events, runs out-of-band, posts results to a per-patient
  inbox the chat agent reads from.
- **`evidence_retriever`** worker owns retrieval. The chat agent
  emits a query; the retriever returns ranked chunks. Lets the
  retriever swap its ranker (BM25 → hybrid → BGE-reranked) without
  touching the chat agent.
- **Supervisor** node arbitrates: new document this turn? Question
  requires external evidence? Synthesize.

The interfaces above are designed so this split is reachable without
rewriting the chat agent — both `attach_and_extract` and
`retrieve_guidelines` are already standalone async functions; they
just live behind a tool wrapper today instead of behind a worker
process boundary.

### 5.3 Critic agent (extension, not core)

The PRD lists "critic agent that rejects uncited claims or unsafe action
suggestions" as **extension**. We get it for free conceptually because the
existing Week 1 validator ([app/agent/validator.py](clinical-copilot/app/agent/validator.py))
already enforces citation correctness. **TODO:** decide whether to extend the
existing validator with semantic checks (slow path, LLM-judge) or add it as
a graph node post-MVP.

### 5.4 Patient-mismatch verification — extracted-doc vs. assigned-patient

**Decision:** the extractor returns the patient identifiers it sees inside
the document. The supervisor compares those to the `patient_id` the doctor
assigned at upload time. On mismatch, the supervisor routes to a
`confirm_patient` node that surfaces both views to the doctor before any
persistence happens.

**Why:** the front desk attaches docs by hand. Wrong attachments and
mislabeled scans are real-world failure modes — exactly the kind of
data-integrity bug the PRD's *"round-trip through OpenEMR without creating
duplicate or untraceable records"* requirement is testing for. Catching
this at upload is cheap (the extractor already pulls the identifiers);
leaving it for downstream cleanup is expensive. This is a structural
verification step in the same spirit as the citation validator: the agent
does not trust the upload context blindly, it verifies its own assignment.

**Tool change:** `attach_and_extract` (§4.1) returns an additional field
alongside the schema JSON:

```python
extracted_patient_identifiers: PatientIdentifiers  # {name, dob, mrn}
```

The extractor pulls these from the same VLM pass that produces the
schema JSON. Each identifier carries the same `Citation` shape (page,
bounding box) as the rest of the extracted fields — so when the doctor
sees the mismatch view, they can click straight to the part of the doc
the agent read those identifiers from.

**Match policy:**

- **MRN** — exact match required. MRN mismatch is always treated as a
  hard mismatch (these don't collide by accident).
- **DOB** — exact match required when present. If the doc didn't carry a
  DOB, fall through to name comparison.
- **Name** — normalized fuzzy compare (case + whitespace collapse;
  optional middle-name omission tolerated). Borderline matches escalate
  to confirmation rather than auto-accepting.

A mismatch on **any** field that the document carries → confirmation
prompt. The bar is intentionally low; false positives (asking the doctor
to confirm a real match) are cheap, false negatives (silently misfiling)
are expensive.

**Supervisor flow:**

```
upload → intake_extractor → identifier comparison
                              ├─ all match           → proceed with persistence
                              ├─ doc has no
                              │   identifiers        → log soft warning,
                              │                        proceed
                              └─ mismatch            → confirm_patient node
                                                       ├─ doctor confirms
                                                       │   → proceed, log event
                                                       └─ doctor rejects
                                                           → discard upload,
                                                             log event
```

The `confirm_patient` node is a standard LangGraph node, not a hidden
side-channel. Every path through it is logged with structured fields
(`assigned_patient_id`, `extracted_identifiers`, `mismatch_reason`,
`doctor_decision`) into the existing audit log at `traces.db` (§8.1).

**What the doctor sees:** a single confirmation card on the chat surface:

> *"This document was uploaded for **Cohen, Jane (MRN 1234)**, but the
> document appears to belong to **Patel, Raj (MRN 5678)**.*
>
> *Confirm patient assignment, or cancel and re-upload."*

Both names are clickable to their patient cards (existing Week 1
navigation). The extracted-identifier values link to their bounding
boxes in the document preview (§4.6 overlay).

**What this is not:** this is *not* a substitute for the patient-panel
ACL. The ACL gates *which patients the doctor is even allowed to assign
to*; the mismatch check verifies *that the assignment matches the
document's contents*. Both apply, in series.

**TODO:** decide whether the rejection path leaves the source document
in OpenEMR (orphaned, for an admin to reassign) or deletes it. Defaulting
to keep-with-orphaned-status pending audit-trail review — consistent with
the rest of the system's append-only posture.

---

## 6. The recommendation gradient — where the line sits

This is the explicit design decision on Week 1 vs. Week 2 scope drift.

### 6.1 Three levels

- **Level A — pure surfacing.** Agent states facts and stops. Doctor draws
  the conclusion. Week 1 default.
- **Level B — quoting retrieved guidelines.** Agent quotes a tool-returned
  guideline chunk verbatim. Still summarizing tool output.
- **Level C — applying guideline to patient.** Agent combines patient facts
  + guideline rule and produces a recommendation.

### 6.2 Decision

**Level C with guardrails, behind a per-conversation `recommendation_mode`
toggle.** The toggle is *off* by default; the doctor explicitly opts in.

- When OFF: the agent operates at Level A (Week 1 R2 strict). System prompt
  unchanged.
- When ON: system prompt swaps in `ADVISOR_MODE_ADDENDUM` (already wired in
  the spike commit `acbadd0b3`). R2 is relaxed narrowly: the agent may reason
  about safety for the *currently-selected patient only*, must cite every
  chart-derived fact, and **must end every advisory response with a fixed
  disclaimer block** that pushes decision authority back onto the attending.

### 6.3 Why this is not a reversal of Week 1

Week 1's USERS.md and ARCHITECTURE.md §8.3 said *we deliberately don't ship
Level C reasoning because a confidently-wrong dose recommendation harms
patients.* That reasoning still holds. Week 2's change isn't *"now we ship
Level C unconditionally"* — it's *"Level C is opt-in, gated by an explicit
toggle, accompanied by a mandatory disclaimer, and only with cited
guideline evidence."* The Week 1 essays update from *"deliberately not
shipping"* to *"Week 1 deliberately deferred; Week 2 enables under explicit
opt-in with these structural guardrails."*

**TODO:** update USERS.md §Use Case B and ARCHITECTURE.md §8.3 in the same
commit as the toggle wiring, so the docs and code stay coherent.

### 6.4 What still does not happen, even at Level C

- The agent never tells the doctor what dose to give.
- The agent never tells the doctor what drug to start.
- Every advisory response carries the disclaimer block; the toggle does not
  let the disclaimer be suppressed.
- The agent never reasons about a patient who isn't the currently-selected
  one (per the existing patient-panel ACL).

---

## 7. Eval gate

### 7.1 Existing Week 1 suite

Week 1 ships a snapshot+replay gate at [clinical-copilot/evals/](clinical-copilot/evals/) —
25 rules × 150 snapshots, golden 100% / labeled ≥90% required to push, runs
as the prek pre-push hook. **This stays. Week 2 extends it.**

### 7.2 Week 2 additions

A new tier alongside the existing snapshot+replay tier:

- **50 boolean-rubric cases** covering extraction, evidence retrieval,
  citations, refusals, missing-data behavior.
- Five rubric categories per case:
  - `schema_valid` — deterministic Pydantic parse check.
  - `citation_present` — deterministic regex over the response.
  - `factually_consistent` — LLM-as-judge with closed-form yes/no prompt.
  - `safe_refusal` — LLM-as-judge with closed-form yes/no prompt.
  - `no_phi_in_logs` — deterministic regex over trace JSON.
- **Threshold:** the gate fails if any category regresses by more than 5%
  *or* drops below the per-category pass threshold.

### 7.3 LLM-as-judge — keeping it actionable

- Each judge prompt is a **closed-form yes/no question** with explicit
  pass/fail criteria, not a vague rubric.
- Each judge prompt is **validated against a 20-case hand-labeled set**
  before being wired into the gate. Must agree with human grading on ≥18/20.
- Deterministic checks for the deterministic categories; LLM-judge ONLY on
  `factually_consistent` and `safe_refusal`.

### 7.4 The grader regression test

The PRD says: *"During grading, we will introduce a small regression and
confirm your CI gate fails."* The eval suite must catch regressions
introduced by an outside party, not just our own.

**Implication for design:** the eval cases must exercise behavior the agent
*currently* gets right; a regression means a real change in output. We
specifically *cannot* rely on hand-tuned thresholds that pass only the
current state — the rubrics must be principled enough to catch a genuine
behavioral shift.

---

## 8. Observability + cost tracking

### 8.1 Per-encounter log shape

Each chat turn writes a single row to `data/traces.db` ([app/observability_db.py](clinical-copilot/app/observability_db.py))
with — extending the Week 1 schema:

- Tool sequence (existing) + supervisor routing decisions per turn (new).
- Latency per step (existing) + per-worker breakdown (new).
- Token usage + cost estimate (existing).
- **Retrieval hits** (new): which chunks were retrieved, which survived
  reranking, which were cited in the final response.
- **Extraction confidence** (new): VLM-reported confidence per extracted
  field, plus structural validation outcome (Pydantic parse pass/fail).
- Eval outcome (new): which rubric categories passed/failed for this
  encounter when run against the eval gate.

### 8.2 PHI hygiene

Logs must not contain raw PHI. The existing Week 1 audit-log writer already
strips obvious PHI shapes; Week 2 adds:

- **Document images are never logged.** Only structured extraction output +
  citation pointers.
- **Patient names redacted** in trace narration; replaced with PUUIDs.
- **TODO:** SaaS observability (LangSmith) gets only structural events, not
  document content. Audit the existing LangSmith call sites for compliance.

---

## 9. Tradeoffs and alternatives considered

**TODO** — to fill in as we make decisions. Section format mirrors
[ARCHITECTURE.md §7](ARCHITECTURE.md#7-tradeoffs-and-alternatives-considered).
Initial entries:

| Decision | Chose | Rejected | Why |
|---|---|---|---|
| VLM | Claude Sonnet 4.6 vision | GPT-4V, Gemini | Stack coherence with Week 1; same vendor BAA |
| Structured output | Anthropic tool-use forced JSON | Free-text JSON + `json.loads` | Schema-validated input by construction; eliminates parse-then-repair |
| PDF→image | `pypdfium2` (Google PDFium) | `pdf2image` (Poppler), Tesseract | Pure-Python, no system deps — works on Hetzner without an extra apt install |
| Retrieval (MVP) | BM25 only via `rank-bm25` | FAISS+dense+BGE rerank (deferred) | 12-chunk hand-curated corpus → BM25 returns clinically-relevant top-1; deps stay tiny; PRD-faithful shape lives behind the same public API |
| Reranker (deferred) | BGE open-source | Cohere, Voyage | No new vendor BAA, no API key, runs on Hetzner |
| Vector store (deferred) | FAISS in-memory | Chroma, Qdrant, pgvector | Zero ops at MVP scale; corpus small; rebuild fast |
| Guideline corpus | USPSTF + ADA, 12 hand-curated chunks | "everything", PDF ingestion at MVP | Two-source rule from PRD pitfall list; hand-curate while small + readable; AHA + PDF ingestion deferred |
| DocumentReference write path | OpenEMR multipart `/api/patient/{pid}/document` | FHIR `POST /DocumentReference` | OpenEMR doesn't actually route the FHIR POST despite advertising `create` in CapabilityStatement — discovered live, pivoted same session |
| Idempotency key | SHA-256 prepended to filename | Parallel local hash table | OpenEMR stays single source of truth; no consistency-with-DB bug class to manage |
| Multi-agent shape | Single graph + new tool entry | Supervisor + workers (deferred) | One more LangGraph tool ≈ zero new code; the supervisor LLM call would double per-turn latency for no visible benefit at MVP scope |
| Recommendation tone | R2 carve-out for guideline-cited recs (always-on; toggle UI deferred) | Level A always / unconditional Level C | Per-conversation toggle is post-MVP UX work; the carve-out is enforced by the validator (no `[Guideline/X]` citation = no claim that "USPSTF says…") |
| Upload UI | Plain HTML + vanilla JS | Angular component in chat sidebar | Avoids pulling Angular into a new page just for a 250-line form; chat-sidebar wiring is post-MVP |

---

## 10. Risks

**TODO** — initial cuts:

- **VLM bounding-box quality.** If Claude vision returns bad boxes the
  overlay UI is misleading. Mitigation: two-pass extraction with OCR-derived
  boxes; defer until measured.
- **OCR drift on scanned PDFs.** Real lab PDFs have varied layouts; a single
  template won't cover them. Mitigation: schema validation rejects extracted
  values that don't pass structural checks (e.g. lab value not numeric).
- **Guideline staleness.** USPSTF + ADA update annually. Mitigation: include
  source date in citations; surface a "guideline as of YYYY-MM" note in any
  Level C response.
- **The 5% regression rule cuts both ways.** Tightening rubrics may cause
  *our own* gate to flap. Mitigation: validate judge prompts against a hand-
  labeled set before flipping CI to enforce.
- **Scope creep from Level C.** The toggle exists to bound advisory
  behavior; mission-creep into "always recommend" reverses the Week 1
  story. Mitigation: keep the toggle off by default, surface its state
  visibly in the UI, log every toggle flip in the audit log.

---

## 11. Production-readiness gaps (honest)

**TODO** — extends [ARCHITECTURE.md §8](ARCHITECTURE.md#8-production-readiness-gaps-honest)
with Week 2-specific items:

- **OCR accuracy validation.** Pre-production, the extraction pipeline needs
  a labeled accuracy dataset (real lab forms, hand-graded). Sprint scope is
  demo-quality, not production-quality.
- **Document storage encryption at rest** for source PDFs (currently in
  OpenEMR's filesystem; production needs object storage + KMS).
- **Guideline-corpus update pipeline.** Annual refresh is manual today.
- **Bounding-box accuracy SLA.** Production needs a measurable accuracy
  bound + rollback path if the VLM regresses on a Claude version bump.

---

## 12. Build sequence (this sprint)

**TODO** — to fill in once Architecture Defense ships. Initial sketch:

| Stage | Target gate | Notes |
|---|---|---|
| Architecture Defense | 4h from start | This document, schemas finalized, RAG design locked |
| Schemas + ingestion stub | Mon EOD | Pydantic schemas + `attach_and_extract` skeleton |
| MVP — extraction working | Tue 23:59 | Lab PDF + intake form ingest end-to-end on one demo case each |
| Hybrid RAG + worker graph | Wed | Evidence retriever wired; supervisor routes |
| Eval suite expansion | Thu | 50 cases recorded; LLM-judge prompts validated against hand-labeled set |
| Early Submission | Thu 23:59 | Demo video, deployed app, CI gate green |
| UI polish + observability | Fri–Sat | PDF overlay, trace fields, dashboard updates |
| Final | Sun 12:00 | Cost/latency report, interview prep |

---

## 13. The defense — what to expect from Gauntlet

**TODO** — section parallel to [ARCHITECTURE.md §10](ARCHITECTURE.md#10-the-defense--what-to-expect).
Anticipated questions:

- *"Show me the regression test and prove your gate catches it."* — the
  PRD's hard gate.
- *"Where is the line between Week 1 and Week 2 in your codebase?"* —
  README must answer cleanly.
- *"How do you keep the supervisor inspectable?"* — every routing decision
  in the audit log; demo it.
- *"Why doesn't the agent recommend a dose?"* — Level C scope, disclaimer
  block, advisor-mode toggle UX.
- *"What happens if the VLM hallucinates a field?"* — schema validation
  + citation requirement + bounding-box overlay (the visual "show me where
  this came from").
