# Presearch — agent_forge (Clinical Co-Pilot)

_Date: 2026-04-27_
_Stage: production-bound (Gauntlet AI Austin admission gate)_
_Source brief: Week 1 - AgentForge.pdf_

## Sprint gates (all CT)

| Checkpoint | Deadline | Focus |
|---|---|---|
| Architecture Defense | ~24h | Architecture research and planning |
| MVP | Tue 11:59 PM | Audit, agent plan, deployed app, demo video |
| Early Submission | Thu 11:59 PM | Deployed agent + evals + observability + demo |
| Final | Sun (table says noon, submission section says 10:59 PM — confirm) | Production-ready agent, demo, social post |

Required deliverables: AUDIT.md, USERS.md, ARCHITECTURE.md, eval dataset, deployed app, demo video (3–5 min/submission), AI cost analysis (100 / 1K / 10K / 100K users), social post (final only).

---

## Phase 1: Constraints

### 1. Domain Selection
- **Domain:** healthcare — clinical EHR (locked).
- **User persona:** hospitalist rounding (locked).
- **Anchor use cases (locked):**
  - **A. Pre-round patient summary** — "Catch me up on bed 412 since yesterday." Synthesizes notes/labs/vitals/meds across last 24h; supports follow-up drilldown. Conversational because the right altitude depends on what the doctor asks next.
    - **Patient identification:** mix of bed number and last name in chat (fuzzy lookup against the doctor's panel); MRN supported but not expected. UI shows a persistent **patient card** alongside the chat with demographics, allergies, problem list, code status, attending, and recent vitals — both for quick visual reference and so the doctor can verify the agent's claims at a glance.
  - **B. Medication safety check** — "Is it safe to start drug X on this patient?" Cross-references current meds, renal/hepatic labs, allergies, problem list; returns cited yes/no/with-caveats. Conversational because reasoning spans 4+ chart locations and the doctor will follow up. Showcases verification + domain-constraint enforcement.
  - **C. Sign-out / handoff drafting** — "Draft sign-outs for my 18 patients for the night team." Open-ended generative task with iterative per-patient refinement. Conversational because output is writing, not lookup.
- **Why this trio:** covers the full shift (morning/midday/evening), exercises three different data-access modes (read-heavy synthesis / structured rules / generative writing), and each has a sharp source-attribution story.
- **Explicitly out of scope (v1):** acute/emergent triage, discharge-readiness lists (better as a dashboard), real-time clinical decision support during codes.
- **Verification requirements:** every claim must trace to a record in the patient's file (source attribution); responses must respect clinical rules / dosage thresholds / interaction flags (domain constraint enforcement). Locked.
- **Data sources:** OpenEMR — demographics, encounters, problem list, medications, labs, vitals, clinical notes. Demo data only; treat as PHI under HIPAA.

### 2. Scale & Performance
- **Sprint volume:** small — eval suite + demo only.
- **Production projection:** must defend at 100 / 1K / 10K / 100K users. Interview asks "500-bed hospital, 300 concurrent clinical users."
- **Latency:** brief says "answer in seconds, not minutes." Concrete target TBD (proposed default: p50 first-useful-response < 5s, p95 < 10s).
- **Cost constraints:** TBD — feeds the cost-analysis deliverable.

### 3. Reliability Requirements
- **Cost of wrong answer:** catastrophic — confident hallucination can directly harm a patient. Patient safety is the floor.
- **Verification non-negotiables:** unsourced claims must not be stated as fact; clinical-rule violations must be flagged or rejected.
- **Human in the loop:** physician is always the decision-maker — agent surfaces context, never prescribes. (Confirm with user.)
- **Audit/compliance:** HIPAA — PHI access logs, retention policy, breach notification, BAA implications for any LLM provider.

### 4. Team & Skill Constraints
- TBD.

---

## Phase 2: Architecture Discovery

### Data layer (locked)
- **EHR:** OpenEMR fork at https://github.com/Hvoegeli/openemr.git (cloned to `./openemr/`).
- **Local stack:** `cd openemr/docker/development-easy && docker compose up --detach --wait` → https://localhost:9300/ (admin/pass), phpMyAdmin on :8310.
- **Data API:** FHIR R4 at `/apis/default/fhir/`. 30+ resources implemented under `src/RestControllers/FHIR/`. Standard REST API at `/apis/default/api/` is fallback only.
- **Resources we'll consume:** Patient, Encounter, Observation (labs+vitals), MedicationRequest, Condition (problem list), AllergyIntolerance, DocumentReference (clinical notes), Practitioner.
- **Auth to FHIR:** OAuth2 Client Credentials Grant with `system/*` scopes (asymmetric JWKS supported). Backend service registers once, gets a system token, no per-user OAuth dance.
- **Roles model:** GACL (database-driven, flexible) — roles in `list_options` (physician/nurse/manager/admin). Authz happens at query time; we'll check `caller_user_id` at the tool layer.

### Demo data (open)
- OpenEMR ships 3 example patients (`sql/example_patient_data.sql`) — not enough for hospitalist scenarios.
- Options: (a) generate inpatients with Synthea, (b) hand-craft 5–10 rich hospitalist patients with multi-day encounters, notes, labs, med changes, (c) hybrid: Synthea for breadth + hand-craft 2–3 demo "stars".
- Decision pending.

### Application layer (locked)
- **Backend:** FastAPI (Python).
- **Agent orchestration:** **LangGraph** (state machine — gives us an explicit `validate_citations` node between LLM and user). LangChain `@tool` for tool definitions, `ChatAnthropic` for the model wrapper.
- **LLM (model cascade):**
  - Default: **Claude Sonnet 4.6** (~$3 / $15 per MTok)
  - Use Case B (med safety): escalate to **Claude Opus 4.7** (~$15 / $75)
  - Sub-tasks (parameter extraction, classification): **Claude Haiku 4.5** (~$1 / $5)
- **Tools:** thin Python wrappers around the OpenEMR FHIR adapter; every tool returns `{data, sources: [resource_type/id, ...]}`.
- **Verification:** tool-output-as-source-of-truth + structured-output-with-citations validator (LangGraph node). Post-hoc fact-checker (second LLM call) reserved for Use Case B.
- **Frontend:** simple chat UI with patient list + expandable citation cards (the demo's killer visual).
- **Observability:** **LangSmith** (native LangChain integration; traces every node + tool + token; eval UI doubles as our eval-suite deliverable surface).
- **App database:** **Postgres** (Fly.io managed) — conversation state, audit log, eval results. Separate from OpenEMR's MySQL.
- **Audit log:** append-only Postgres table; written by the FHIR adapter on every call.
- **Hosting:** **Fly.io** — FastAPI app, Postgres, optionally OpenEMR container too.
- **Latency targets:**
  - Time-to-first-token: < 2s (via streaming)
  - Use Case A (summary): full response < 8s
  - Use Case B (med safety): full response < 15s
  - Use Case C (sign-out): full response < 60s for 15–20 patients (stream per patient)
- **Performance levers:** streaming, parallel tool calls, Anthropic prompt caching for system prompt + per-patient context.

## Phase 3: Post-Stack Refinement
### Decision log (resolved)

| Decision | Status | Why |
|---|---|---|
| LLM provider | **Locked: Anthropic** | Best balance of quality, tooling fit (LangChain/LangGraph), and rapid iteration for clinical synthesis + citations. |
| Agent framework | **Locked: LangGraph** | Explicit graph control supports deterministic validation steps (`validate_citations`) before user output. |
| Observability | **Locked: LangSmith** | First-class traces for nodes/tools/tokens plus eval UI that can serve sprint deliverable needs. |
| Hosting | **Locked: Fly.io** | Fast deploy loop, managed Postgres, and acceptable footprint for MVP + early submission. |
| Verification strategy | **Locked: tool-output source of truth + citation validator** | Default safety path with deterministic checks; optional second-pass fact-check only for high-risk med safety flow. |

### Security and compliance controls (production-bound baseline)

- **Data classification:** Treat all OpenEMR-derived fields as PHI, including "demo-only" records.
- **Encryption in transit:** TLS 1.2+ for all network hops (browser <-> API, API <-> OpenEMR, API <-> Postgres, API <-> LLM endpoint).
- **Encryption at rest:** Managed disk encryption for Postgres/OpenEMR volumes; encrypted backups; no unencrypted PHI artifacts on local disk.
- **Secrets management:** Provider-managed secrets store; no credentials in source; rotate API keys and OAuth client secrets on a fixed cadence.
- **Access control:** RBAC enforced in tool layer using `caller_user_id`, role, and patient-panel scope (minimum necessary access).
- **Auditability:** Append-only audit table records actor, patient, tool/action, timestamp, outcome, and source references for every read.
- **Retention:** Define and enforce retention windows for chats, traces, and logs; documented purge process for expired records.
- **Vendor compliance:** Confirm BAAs for production vendors before any non-demo PHI is permitted.

### Safety policy and fallback behavior (operational)

- **Output contract:** Every clinical claim must include at least one citation to a specific source object (`resource_type/id`).
- **Hard fail on missing evidence:** If a claim cannot be sourced, the assistant must respond with "insufficient evidence in chart" and avoid guessing.
- **Medication safety guardrails:** If interaction/contraindication signal is incomplete or conflicting, return "unsafe to determine" with required follow-up data.
- **Rule conflict handling:** Domain rule violations are surfaced as blocking warnings; no affirmative safety recommendation is emitted.
- **Timeout/degraded mode:** If critical tools fail or exceed timeout, assistant provides partial status + explicit caveat; no definitive clinical conclusion.
- **Human authority statement:** Every high-risk response ends with "For clinician judgment; verify before ordering."

### Identity and authorization model (end-to-end)

- **Identity source:** App-authenticated user session maps to OpenEMR user identity (`caller_user_id`).
- **Scope enforcement:** Tools can only query patients on the caller's assigned panel unless explicit elevated role allows broader scope.
- **Role behavior:** Physician, nurse, resident, admin roles map to read/write capabilities and visibility constraints.
- **Cross-user isolation:** Conversation state and cached context are partitioned per org + user to prevent data leakage.
- **Authz tests:** Include deny-by-default tests for out-of-panel access and role-escalation attempts.

### Cost model assumptions (for 100 / 1K / 10K / 100K users)

- **Per-request budget:** Track tokens/request by use case (A/B/C), tool-call count, and model tier used.
- **Escalation policy:** Use Opus only for medication safety ambiguity or explicitly high-risk checks; otherwise Sonnet/Haiku path.
- **Caching policy:** Prompt caching for stable system/policy blocks and patient-context templates to reduce input token costs.
- **Guardrail:** Set max token and max tool-call ceilings per request; enforce graceful truncation with explicit user notice.

### Eval framework and release gates

| Dimension | Metric | Gate (MVP) | Gate (Early/Final) |
|---|---|---|---|
| Citation integrity | % claims with valid citation | >= 95% | >= 99% |
| Hallucination control | Unsourced factual claim rate | <= 5% | <= 1% |
| Med safety recall | Detect known contraindication cases | >= 90% | >= 95% |
| Med safety precision | Correctly flag true unsafe combos | >= 85% | >= 90% |
| Authz safety | Out-of-scope patient access success rate | 0% | 0% |
| Latency A | p95 full response (summary) | <= 12s | <= 10s |
| Latency B | p95 full response (med safety) | <= 18s | <= 15s |
| Latency C | p95 completion (15-20 sign-outs) | <= 75s | <= 60s |
| Availability | Successful request rate (non-user error) | >= 99.0% | >= 99.5% |

### Eval dataset shape (minimum)

- **Use case A (summary):** 40 cases, each with expected key facts + required citations.
- **Use case B (med safety):** 50 cases (safe/unsafe/insufficient-data balanced), with adjudicated expected outcome and rationale anchors.
- **Use case C (sign-out):** 30 multi-patient batches scored for factuality, completeness, and citation coverage.
- **Adversarial set:** 20 prompt-injection and missing-data cases to verify refusal/degraded behavior.

### Go/No-Go release criteria

- All safety-critical gates pass (citation integrity, hallucination, med safety recall, authz).
- No high-severity unresolved security findings.
- p95 latency and availability meet target for the current sprint checkpoint.
- Audit log completeness validated for 100% sampled requests.
- Manual clinician review pass on representative scenarios for A/B/C.

### Architecture defense script (2-3 minutes)

**Opening (20-30s)**
- We are building a multi-turn clinical co-pilot for hospitalists on OpenEMR, focused on three shift-critical workflows: pre-round summary, medication safety checks, and sign-out drafting.
- The architecture is safety-first: every claim must map to chart evidence, and the system refuses to guess when evidence is missing.

**Core architecture (45-60s)**
- Data plane: OpenEMR FHIR R4 is the source of truth. Tool wrappers return structured data plus explicit source IDs.
- Control plane: FastAPI + LangGraph. The graph enforces a deterministic validation step (`validate_citations`) before user-visible output.
- Model strategy: Sonnet by default, Haiku for narrow extraction/classification, Opus only for high-risk medication ambiguity.
- State/ops: Postgres stores conversation state + append-only audits; LangSmith captures traces/evals; Fly.io hosts API and managed services.

**Safety and compliance posture (30-45s)**
- Unsourced statements are blocked; missing/conflicting evidence yields "insufficient evidence" or "unsafe to determine."
- RBAC and patient-panel scoping are enforced at tool boundaries using caller identity.
- PHI handling assumes HIPAA constraints end-to-end: TLS in transit, encrypted storage, audited access, retention controls, BAA gate before non-demo PHI.

**Production-defensibility and scale (30-45s)**
- We defend both quality and ops with measurable gates: citation integrity, hallucination rate, med-safety recall/precision, authz safety, p95 latency, and availability.
- Cost controls are built in via model escalation policy, token/tool-call ceilings, and prompt caching.
- The release is gated by objective go/no-go criteria rather than subjective demo quality.

### Likely pushback and crisp responses

**Q: "Why not use a simpler RAG chatbot?"**
- A generic RAG loop cannot guarantee claim-level provenance or hard gating on unsourced output.
- LangGraph gives explicit control points for deterministic validation and refusal behavior required for clinical risk.

**Q: "How do you prevent hallucinations in high-stakes settings?"**
- We constrain generation to tool-returned facts, require citations per claim, and block outputs that fail citation validation.
- For medication safety, ambiguous cases escalate model quality and still fail closed when evidence is incomplete.

**Q: "What if OpenEMR/FHIR is slow or partially down?"**
- The system degrades safely: partial context + explicit caveat, no definitive clinical conclusion.
- Availability and latency are monitored as release gates; critical-tool timeouts trigger refusal patterns, not silent failure.

**Q: "How do you prove role-based safety?"**
- We run deny-by-default authz tests for out-of-panel and role-escalation attempts; success rate must remain 0%.
- Every access is auditable with actor, patient, action, timestamp, and outcome.

**Q: "Is this cost-feasible at 10K-100K users?"**
- We model per-use-case token/tool budgets, keep Sonnet/Haiku as default path, and reserve Opus for high-risk ambiguity only.
- Caching and request ceilings cap tail-cost behavior while preserving safety policies.

**Q: "What makes this production-bound and not just a demo?"**
- The project is evaluated with explicit reliability/safety SLOs, adversarial tests, audit completeness, and release gates.
- The architecture encodes operational controls (authz, audit, observability, fallback), not just prompt quality.

### Judge-facing close (15s)

- This design prioritizes patient safety over fluency, with evidence-locked outputs, fail-closed behavior, and measurable release gates.
- If a judge asks for proof, we can show traces, eval outcomes, and audit records for any response path end-to-end.

---

## Open questions / unresolved
- Team and staffing constraints (who owns backend, evals, data generation, and demo ops)
- Demo-data strategy final pick: Synthea-only vs hand-crafted vs hybrid
- Final deadline ambiguity in brief (noon vs 10:59 PM Sunday) — confirm and lock
- Legal/compliance sign-off owner for BAA confirmation and retention policy

## Decisions locked in (from brief)
- Domain: clinical EHR (healthcare)
- Codebase: OpenEMR fork (https://github.com/openemr/openemr)
- Multi-turn conversational agent (no search bar / dashboard / report)
- Source-attributed verification + domain-constraint enforcement (required)
- HIPAA-aware architecture, demo data only
- Multi-user auth/authz (physician / nurse / resident roles)
- Observability and eval suite required from day one
- Production-defensible standard ("hospital CTO bar")
