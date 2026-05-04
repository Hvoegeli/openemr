# W2_ARCHITECTURE — Multimodal Evidence Agent

> Companion to [`docs/prds/week-2-multimodal-evidence.md`](docs/prds/week-2-multimodal-evidence.md).
> The PRD says *what* Week 2 must deliver. This document says *how* we deliver
> it — the architectural decisions, the schemas, the worker graph, the
> retrieval design, the eval gate, the risks, and the explicit reversals (or
> non-reversals) of Week 1's deliberate scope choices.
>
> **Status:** stub — section skeleton in place; content fills in as decisions
> land during the sprint. Updated continuously through Sunday 2026-05-10.

---

## 0. Decisions locked

Quick-reference table of choices already made (rationale in the relevant
section below):

| Area | Decision | Section |
|---|---|---|
| VLM | Claude Sonnet 4.6 vision, in-line bounding boxes | §1 |
| Reranker | BGE (open-source, runs on Hetzner) | §3 |
| Vector store | FAISS in-memory | §3 |
| Guideline corpus | USPSTF + ADA (start) | §3 |
| "Recommendation" tone | Level C with guardrails behind `recommendation_mode` toggle | §6 |
| Branch | `copilot--branch-3`; no auto-merge to master | — |
| Doc location | `W2_ARCHITECTURE.md` at repo root | — |

---

## 1. Executive summary (~500 words target)

**TODO** — written last, after all the decisions below have settled. Mirrors
the shape of [`ARCHITECTURE.md`](ARCHITECTURE.md) §1: scenario, what's net-new
vs. compounded from Week 1, the core architectural moves, where the
verification line sits, what's deliberately out of scope.

---

## 2. System context — what changed from Week 1

**TODO** — diff vs. Week 1's §2:

- New stakeholders / surfaces (front desk uploading documents).
- New data shapes (PDFs, intake forms, guideline corpus).
- New trust boundary (Claude vision reads document images).
- What stays the same (auth, sessions, audit log, ACL, observability).

---

## 3. RAG design — the evidence layer

### 3.1 Corpus

**Decision:** USPSTF + ADA, two sources, start.

- **USPSTF** — US Preventive Services Task Force recommendations. Public
  domain. Cleanest fit for the PRD's "what should I pay attention to"
  preventive-care framing. ~100 recommendations after structuring.
- **ADA** — American Diabetes Association annual care standards. Bread-and-
  butter PCP work; the PRD's patient scenario likely involves a diabetic.
  Freely distributable for educational use.

Two sources keeps the corpus small and inspectable. Per the PRD's pitfall
list: *"trying to support five document types before two work reliably"* —
same principle for guidelines.

**TODO:** decide chunking strategy (per-recommendation vs. per-section vs.
fixed-size with overlap). Hand-curate the source list before mass-indexing.

### 3.2 Indexing pipeline

**Decision:** in-process, FAISS in-memory, rebuilt at startup.

- Embedding model: **TODO** — likely OpenAI `text-embedding-3-small` or a
  local sentence-transformer. Decision pending.
- Vector store: **FAISS in-memory** (`faiss-cpu`). Index lives in the FastAPI
  process; rebuild on startup is sub-second for the corpus size. Zero ops.
- Sparse index: **TODO** — BM25 via `rank-bm25` (pure Python, in-process) or
  Elasticsearch (overkill for this scale). Default to `rank-bm25`.

### 3.3 Retrieval + rerank

**Decision:** Hybrid sparse+dense → BGE reranker → top 5–10 chunks to LLM.

- Stage 1 (retrieval): query against both BM25 and dense FAISS, merge top-50.
- Stage 2 (rerank): **BGE reranker** (open-source, `BAAI/bge-reranker-base`)
  scores each candidate against the query; keeps top 5–10.
- Stage 3: top chunks fed to the answer model with citation metadata
  attached.

**Why BGE over Cohere Rerank** (which the PRD names): no new vendor BAA, no
API key, runs locally on Hetzner, top-of-MTEB quality at its size, easy swap
to Cohere later by changing one function.

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

### 4.1 Tool signature

**Decision:** new `attach_and_extract(patient_id, file_path, doc_type)` tool.

- `patient_id` — gated by the existing patient-panel ACL ([app/access_control.py](clinical-copilot/app/access_control.py)).
- `file_path` — uploaded file in a server-side temp location.
- `doc_type` — enum: `lab_pdf` | `intake_form`. New types added later as
  fast-follow.

Returns: strict-schema JSON (per §4.2 below) plus the persisted
`DocumentReference` ID for the source document.

### 4.2 Schemas

**Pydantic strict schemas.** Stub:

```python
class LabResult(BaseModel):
    test_name: str
    value: float | str
    unit: str
    reference_range: str | None
    collection_date: date
    abnormal_flag: Literal["H", "L", "N", "C", None]
    source_citation: Citation

class IntakeForm(BaseModel):
    demographics: Demographics
    chief_concern: str
    current_medications: list[Medication]
    allergies: list[Allergy]
    family_history: list[FamilyHistoryItem]
    source_citation: Citation
```

**TODO:** finalize all sub-types; write Pydantic validation tests; decide on
optionality vs. required for fields the VLM might miss.

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

- **Source document** → FHIR `DocumentReference` (one of the 4 writable FHIR
  resources per [ARCHITECTURE.md §4.1](ARCHITECTURE.md#41-data-layer--openemr)).
  Reuses the existing write-capable OAuth client ([scripts/register_seed_client.py](clinical-copilot/scripts/register_seed_client.py)).
- **Lab values** → FHIR `Observation` per result, linked to the source
  `DocumentReference` via `derivedFrom`.
- **Intake fields** → mapped to `Patient` updates, `Condition` (problem list),
  `AllergyIntolerance`, `MedicationStatement`. **TODO:** these mostly require
  the non-FHIR `/apis/default/api/` path documented in Week 1 §4.1; need to
  confirm scope coverage and write an adapter layer.
- **Idempotency:** every ingestion writes a content-hash key into a local
  table; re-uploading the same file is a no-op (returns the existing
  `DocumentReference` ID).

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

### 5.1 Workers

- **`intake_extractor`** — owns `attach_and_extract`. Receives a file +
  patient context, returns strict-schema JSON + persisted IDs.
- **`evidence_retriever`** — owns RAG. Receives a query + patient context,
  returns top-N reranked guideline chunks with citations.

### 5.2 Supervisor

**TODO** — design choice: LangGraph state-machine supervisor (consistent with
Week 1's existing LangGraph) vs. dedicated Supervisor agent (separate LLM
call). Default to LangGraph-as-supervisor for inspectability + lower latency.

The supervisor decides per turn:
1. Is there a new document attached this turn? → route to `intake_extractor`.
2. Does the question require external evidence? → route to `evidence_retriever`.
3. Synthesize the final answer with all gathered context.

Every routing decision is **logged** to the existing `traces.db` audit log,
with `routing_reason` as a structured field. No black-box supervisors.

### 5.3 Critic agent (extension, not core)

The PRD lists "critic agent that rejects uncited claims or unsafe action
suggestions" as **extension**. We get it for free conceptually because the
existing Week 1 validator ([app/agent/validator.py](clinical-copilot/app/agent/validator.py))
already enforces citation correctness. **TODO:** decide whether to extend the
existing validator with semantic checks (slow path, LLM-judge) or add it as
a graph node post-MVP.

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
| Reranker | BGE open-source | Cohere, Voyage | No new vendor BAA, no API key, runs on Hetzner |
| Vector store | FAISS in-memory | Chroma, Qdrant, pgvector | Zero ops at MVP scale; corpus small; rebuild fast |
| Guideline corpus | USPSTF + ADA | "everything" | Two-source rule from PRD pitfall list; matches PCP scope |
| Recommendation tone | Level C with toggle | Level A always / Level C always | Refusing Level C fails the PRD scenario; unconditional Level C reverses Week 1 deliberately-deferred decision |

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
