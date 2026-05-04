---
title: Week 2 — Multimodal Evidence Agent
status: in-progress
sprint: 2 of 3
sprint_dates: 2026-05-04 → 2026-05-10
deadlines:
  architecture_defense: 4 hours from sprint start (Central / Austin)
  mvp: Tuesday 2026-05-05 @ 23:59 CT
  early_submission: Thursday 2026-05-07 @ 23:59 CT
  final: Sunday 2026-05-10 @ 12:00 CT
source: Gauntlet AI — Austin Admission Track (PDF received 2026-05-04)
companion_doc: ../../W2_ARCHITECTURE.md
---

# Week 2 — Multimodal Evidence Agent

> Seeing clinical documents, routing work, and gating changes with evals.

This file is the **Week 2 PRD as received from Gauntlet**, scaffolded into the
repo for reference. It is the *requirements document* — the source of truth for
what is being asked of us this week. All architectural decisions and design
responses to this PRD live in [`W2_ARCHITECTURE.md`](../../W2_ARCHITECTURE.md)
at the repo root.

---

## How to use this assignment

The Week 1 agent already reads structured OpenEMR data, attributes claims, logs
tool behavior, and has a starter eval suite. This week adds two new
capabilities: the agent can read real-world clinical documents, and it can
route work across a small multi-agent graph without losing grounding.

**GATE:** Eval-driven CI is non-negotiable. A working demo that cannot block
regressions has not met the Week 2 standard.

## The scenario

A primary-care physician is prepping for a follow-up visit. The chart has
structured OpenEMR data, but the important recent information is buried in a
scanned lab PDF and a patient intake form uploaded by the front desk. The
physician asks: *What changed, what should I pay attention to, and what
evidence supports the recommendation?*

The Week 2 Clinical Co-Pilot must ingest the lab PDF and intake form, extract
structured facts with citations, retrieve relevant guideline evidence, and
return a grounded answer. The answer should be useful even if the document
scan is imperfect, the patient record is incomplete, or the user asks a
follow-up question.

**WHY THIS MATTERS:** Clinical agents fail when they cannot handle the messy
inputs clinicians actually receive. Week 2 is about making the agent see,
keeping the architecture small enough to reason about, and proving quality
through automated evals.

## The hard problems

### Vision extraction without invention

A VLM can read a scanned form, but it can also hallucinate field labels or
overstate confidence. The schema, source links, and verification strategy must
make unsupported extracted facts visible.

### Evidence grounding

Every answer must separate patient-record facts from guideline evidence. A
medication or lab claim is not acceptable unless it points back to a source.

### Multi-agent architecture

The goal is to give multiple workers clear responsibilities and make the
supervisor's routing decisions inspectable.

### Eval-driven development

A 50-case golden set with boolean rubrics. The CI gate must catch regressions
before they reach the demo.

### FHIR and OpenEMR integrity

Uploaded documents and derived observations must round-trip through OpenEMR
without creating duplicate or untraceable records.

### HIPAA-minded development

Use only demo or synthetic data. Do not log raw PHI. Treat prompts, extracted
fields, document images, traces, and screenshots as sensitive.

## The codebase

Build on the Week 1 fork: auth flow, tool layer, verification strategy,
observability, and eval harness. Good Week 1 architecture should compound
here; technical debt from Week 1 should be documented and resolved before
adding new surface area.

The Week 2 work stays inside the same OpenEMR fork on the existing branch
(`copilot--branch-3`). The README must clearly separate Week 1 baseline
behavior from Week 2 multimodal behavior. Graders should be able to run the
core Week 2 flow without guessing which branch, environment variable, or
service is required.

## Project schedule

Hard gates. One-week sprint with four checkpoints. All times Central (Austin).

| Checkpoint | Deadline | Focus |
|---|---|---|
| Architecture Defense | 4 hours from sprint start | Document schemas, RAG and eval design, security concerns |
| MVP | Tuesday 2026-05-05 @ 23:59 CT | Lab PDF and intake form ingestion working locally; first extraction and first evidence retrieval demo |
| Early Submission | Thursday 2026-05-07 @ 23:59 CT | Supervisor + 2 workers, 50-case eval suite, PR-blocking CI, deployed app, demo video |
| Final | Sunday 2026-05-10 @ 12:00 CT | Production-ready Week 2 agent, source-grounded demo, cost/latency report, interview readiness |

## MVP — recommended steps

The MVP is not a full medical-document AI platform. It is a controlled
expansion of the Week 1 agent into two document types, two workers, and one
regression gate.

| Name | Deliverable |
|---|---|
| Ingest two document types | Upload and extract a lab PDF and an intake form using strict schemas |
| Build basic hybrid RAG | Small guideline corpus indexed with keyword + dense retrieval and Cohere rerank or equivalent |
| Add supervisor + 2 workers | Supervisor routes to intake-extractor and evidence-retriever with logged handoffs |
| Gate with eval-driven CI | 50-case golden set, boolean rubrics, PR-blocking Git Hook |
| Integrate and demo | Deployed app, source-grounded UI, latency/cost report, walkthrough video |

## Stages

### Stage 1 — Ingest lab PDF and intake form

Implement a document ingestion flow that accepts a file, associates it with a
patient, stores the source document in OpenEMR, extracts structured JSON, and
links every derived fact back to the source. Required document types: lab PDF
and intake form.

### Stage 2 — Build basic hybrid RAG

Create a small clinical-guideline corpus relevant to the user profile. Use
keyword + vector retrieval, rerank candidate chunks, and return evidence
snippets with source metadata. ColQwen2 and multi-vector indexing are stretch;
the core requirement is a reliable hybrid retriever.

### Stage 3 — Add supervisor + 2 workers

Implement a small graph: one supervisor, one intake-extractor worker, one
evidence-retriever worker. The supervisor decides when extraction is needed,
when evidence retrieval is needed, and when the final answer is ready. Keep
handoffs explicit.

### Stage 4 — Build the eval gate

Create 50 synthetic or demo cases that exercise extraction, evidence
retrieval, citations, refusals, and missing-data behavior. Use boolean
rubrics, not 1–10 ratings. CI must fail on meaningful regression.

### Stage 5 — Integrate, deploy, defend

Expose the Week 2 flow in the deployed app, capture observability traces,
record a demo, and prepare to explain why each capability maps back to the
Week 1 user and workflow.

## Core agent requirements

### 1. Document ingestion and extraction

Implement `attach_and_extract(patient_id, file_path, doc_type)` or an
equivalent tool. Must support `lab_pdf` and `intake_form`. Must store the
source document in OpenEMR, return strict-schema JSON, and persist derived
facts as appropriate FHIR resources or OpenEMR records.

### 2. Structured schemas

Use Pydantic, Zod, or equivalent strict schemas.

**Required lab fields:** at minimum — test name, value, unit, reference range,
collection date, abnormal flag, source citation.

**Required intake fields:** demographics, chief concern, current medications,
allergies, family history, source citation.

### 3. Basic hybrid RAG plus rerank

Index a small clinical-guideline corpus. Retrieve with sparse+dense search,
rerank candidate chunks with Cohere Rerank or an equivalent reranker, and feed
only the top grounded evidence to the answer model.

### 4. Supervisor plus two workers

Use LangGraph, the OpenAI Agents SDK, or another inspectable orchestration
framework. Required workers: intake-extractor and evidence-retriever. A critic
agent is extension work, not core.

### 5. Citation contract

Every clinical claim in the final response must include machine-readable
citation metadata. Minimum citation shape:

```
{source_type, source_id, page_or_section, field_or_chunk_id, quote_or_value}
```

A visual PDF bounding-box overlay is **required**.

### 6. Eval-driven CI gate

Build a 50-case golden set and a PR-blocking Git Hook. Boolean rubric
categories must include:

- `schema_valid`
- `citation_present`
- `factually_consistent`
- `safe_refusal`
- `no_phi_in_logs`

The build must fail if any category regresses by more than 5% or drops below
the pass threshold.

### 7. Observability and cost tracking

Each encounter must log tool sequence, latency by step, token usage, cost
estimate, retrieval hits, extraction confidence, and eval outcome. Logs must
not contain raw PHI.

**HARD GATE:** During grading, Gauntlet will introduce a small regression and
confirm the CI gate fails. If the eval gate does not block the regression,
the Week 2 build does not pass.

## Core, extension, and stretch detail

### Core

- Two document types: lab PDF and intake form.
- One supervisor and two workers: intake-extractor and evidence-retriever.
- Basic hybrid RAG plus rerank over a small guideline corpus.
- 50-case golden dataset with boolean rubrics.
- PR-blocking eval CI and an observable deployed demo.

### Extension

- Critic agent that rejects uncited claims or unsafe action suggestions.
- Click-to-source UI for citation snippets, with a simple document preview.
- A third document type such as referral fax or medication list.

### Stretch

- Lab trend chart widget that uses extracted Observation data.
- Contextual retrieval improvements: better chunking, query rewriting, or
  domain-specific filters.

## Submission requirements

| Deliverable | Requirements |
|---|---|
| GitLab repository | Week 1 fork with Week 2 changes, setup guide, deployed link, clear environment-variable documentation |
| W2 Architecture Doc | `./W2_ARCHITECTURE.md` explaining document ingestion flow, worker graph, RAG design, eval gate, risks, tradeoffs |
| Schemas | Pydantic schemas for `lab_pdf` and `intake_form`, including source citation fields and validation tests |
| Eval Dataset | 50 synthetic/demo cases with expected behavior, boolean rubrics, judge configuration, results |
| CI Evidence | Git Hook or equivalent that runs the eval suite and blocks regressions |
| Demo Video | 3–5 minutes showing document upload, extraction, evidence retrieval, citations, eval results, observability |
| Cost & Latency Report | Actual dev spend, projected production cost, p50/p95 latency, bottleneck analysis |
| Deployed Application | Publicly accessible deployed app with the Week 2 core flow working |

## Common pitfalls

- Trying to support five document types before two work reliably.
- Using a VLM answer directly without schema validation or source metadata.
- Letting the supervisor become a black box. Handoffs must be logged and
  explainable.
- Using LLM-as-judge without a clear rubric. Use boolean rubrics so failures
  are actionable.
- Logging raw document text, patient identifiers, or screenshots to SaaS
  observability tools.

## Final note (from Gauntlet)

> Week 2 is not a contest to integrate the most AI frameworks. It is a test of
> whether you can add multimodal inputs, keep the agent architecture
> comprehensible, and prove quality with a CI gate. The best submissions will
> feel narrower than the original spec and stronger because of it.

---

## Decisions locked (this conversation, 2026-05-04)

These were decided in the planning conversation that scaffolded this PRD into
the repo. The full reasoning lives in [`W2_ARCHITECTURE.md`](../../W2_ARCHITECTURE.md);
this is a quick-reference summary so the choices are visible from the PRD.

| Question | Decision | Rationale (one line) |
|---|---|---|
| VLM | Claude Sonnet 4.6 vision, in-line bounding boxes | Stays inside the Week 1 stack; no new vendor |
| Reranker | BGE (open-source, on Hetzner) | No new vendor BAA, no API key, runs in-process |
| Vector store | FAISS in-memory | Zero ops; small corpus; rebuild on startup is sub-second |
| Guideline corpus | USPSTF + ADA (start) | Public domain + free-for-education; clean fit for PCP scenario |
| "Recommendation" tone | Level C with guardrails behind `recommendation_mode` toggle | Refusing Level C means failing the PRD scenario; advisor spike already wired for this shape |
| Doc placement | `W2_ARCHITECTURE.md` at repo root, exact name | Matches PRD's literal `./W2_ARCHITECTURE.md` reference |
| Branch | Continue on `copilot--branch-3`; do not auto-merge to master | Standing branch-only rule; master stays at Week 1 submission |

### Followups to remind about

- Additional user-role requirements (doctors / admin / nurses) — to be defined.
  Surface this when touching USERS.md, ACL/permissions code, or any tool with
  a role check.
