# Architecture — Clinical Co-Pilot

_Status: MVP shipped (Tuesday 2026-04-28). Post-MVP delta — what landed Wed/Thu — is in §1.2; Thursday early-submission scope is in §10._
_Date: 2026-04-28 (originally drafted 2026-04-27). Last updated: 2026-04-30._
_Source-of-truth user definition: [USERS.md](USERS.md). Audit findings driving design choices: [AUDIT.md](AUDIT.md). Agent code: [clinical-copilot/](clinical-copilot/)._

## 1. Executive summary (~500 words)

The Clinical Co-Pilot is a multi-turn conversational agent for a hospitalist physician rounding on 12–18 inpatients per shift. It reads the patient's chart from a forked OpenEMR (via FHIR R4), synthesizes what matters, and answers in natural language with **every clinical claim cited back to a specific chart record**. The chat agent is read-only — it never writes to the EHR. (A separate clinical-notes UI in the same web app writes finalized shift notes back to OpenEMR — see §1.2 — but the LLM has no path to writes; the agent gate stays one-way.) The whole thing lives as a Python service inside the same OpenEMR fork at `clinical-copilot/` so the submission ships as one repo, deployed alongside OpenEMR on a single Hetzner host (the original Fly.io plan was retired post-MVP for ops simplicity).

The architecture is shaped by one specific failure mode: a confidently-stated hallucination in a clinical setting can directly harm a patient. The brief calls this out, and we treat it as the floor every other decision rests on.

**Three architectural choices follow from this.**

**(1) Tool output is the source of truth — structural, not best-effort.** The LLM has no path to FHIR; only the tool layer does. Every tool returns `{data, sources: [resource_type/id, ...]}`. The state machine appends those `sources` to the conversation. Before any LLM response reaches the user, a deterministic **citation validator** ([clinical-copilot/app/agent/validator.py](clinical-copilot/app/agent/validator.py)) extracts every `[ResourceType/ID]` from the response text and rejects the response if any cited ID is not in the cumulative tool-sources set. The LLM cannot get past the gate without retrying with valid sources or admitting "insufficient evidence in chart". This is the architectural verification layer the brief calls non-negotiable.

**(2) The LLM is forbidden from clinical reasoning that didn't come from a tool.** During MVP-day testing we observed the LLM emitting drug-interaction rules ("Metformin held if eGFR <30") and dose-reduction criteria ("Apixaban dose reduction if 2 of 3...") unsourced — pulled from medical training. This is exactly the failure mode the brief warns about. The system prompt now hard-forbids it ([clinical-copilot/app/agent/system_prompt.py](clinical-copilot/app/agent/system_prompt.py) §R2). The right long-term answer is a `clinical_rules` tool returning interaction/dose flags from a real source (FDB, RxNorm-DDI); for MVP we cap the LLM at "summarizer of chart contents". The agent does not invent rules from training.

**(3) State machine, not agent-executor.** LangGraph gives us explicit nodes for the LLM call, the tool dispatch, and the citation-validation gate. The validator is first-class, not a post-hoc check. Routing back from validator to LLM on a failed citation is a graph edge, not a hidden retry. This makes the safety contract auditable: there is exactly one path from LLM output to user, and it goes through the gate.

**Two consequential decisions inherited from [AUDIT.md](AUDIT.md):**
- OpenEMR's FHIR API only writes 4 of 30+ resources — clinical-data writes go through a separate non-FHIR REST API with a different scope vocabulary. We registered **two distinct OAuth clients**: a `private_key_jwt` + `system/∗.read` client for the agent's read path, and a `client_secret_post` + `password`-grant + `user/∗.cruds` client originally registered for demo-data seeding. **Post-MVP, the write client is also exercised in production** by the clinical-note finalize flow (vitals only) — see §1.2. The chat agent still uses only the read client.
- OpenEMR's FHIR + standard-API surfaces don't write to its audit-log tables consistently — we own the audit log in our app, append-only Postgres, written by the FHIR adapter on every call (Thursday work).

**Stack:** FastAPI + LangGraph + Anthropic Claude Sonnet 4.6 + httpx; deployed alongside OpenEMR on a single Hetzner host (PHP/Apache + private MariaDB on the same box; uvicorn under `copilot.service`; cloudflared named-tunnel front for public access). The full stack ships from a single repo (`Hvoegeli/openemr`). _Originally targeted Fly.io; retired post-MVP — see §1.2._

**Known gaps the brief expects us to flag:** no audit log writing yet (Thursday), no LangSmith tracing wired (Thursday), no clinical-rules tool (slated Sunday final), no Postgres for sessions (in-memory MVP only). Chat-turn latency is still ~14s — dominated by the LLM call; further drop comes Thursday from prompt caching and tool-call parallelism. **MVP-day deltas vs the original plan:** cookie-session login validating against OpenEMR's password grant is shipped; SSE token streaming is shipped; a server-side TTL cache + startup prewarm makes dashboard endpoints ~10ms warm (calendar/card/documents); citation clicks now navigate (Patient → card tab; Encounter → doc viewer; Allergy/Condition/Med/Obs → scroll-and-flash on the card).

## 1.1 Implementation status (as of MVP)

What is actually shipped tonight versus aspirational, organized by sprint gate:

| Component | MVP (tonight) | Early submission (Thu) | Final (Sun) |
|---|---|---|---|
| Forked OpenEMR | ✅ public via cloudflared (PHP/Apache + private MariaDB); Fly.io deploy in flight at MVP | ✅ post-MVP: Hetzner single-host deploy (PHP/Apache + private MariaDB on the same box as the co-pilot); Fly.io plan retired for ops simplicity | hardened: TLS managed certs, default creds rotated |
| Agent code lives in `clinical-copilot/` | ✅ pushed to `Hvoegeli/openemr` | — | — |
| Citation validator | ✅ regex + cumulative-sources check + retry-on-miss | + LangSmith trace per validation event | — |
| **Citation-click navigation** | ✅ `[Patient]` → card tab; `[Encounter]` → doc viewer; `[Allergy/Condition/Med/Obs]` → scroll + flash matching `<li>` | — | — |
| Tools | ✅ `current_time`, `resolve_patient`, `get_patient_card` | + 24h-window `get_observations`, `get_notes`, `get_med_changes` | + `clinical_rules(meds, problems, labs)` |
| LLM forbidden from clinical reasoning beyond tools | ✅ enforced via system prompt | downgraded once `clinical_rules` tool exists | — |
| **App-layer auth** | ✅ cookie session validating creds against OpenEMR's password-grant OAuth; all data endpoints gated | + role mapping (physician/nurse/resident) | + SSO (SAML/OIDC) for hospital deploy |
| **Streaming** | ✅ token-by-token SSE via `astream_events`; tool-progress events drive UI status pills | — | — |
| **Dashboard UI** | ✅ 2-pane layout: tabs (Today's Calendar default → Patient Card → Supporting Documents) on left, chat on right; doc viewer overlay with Close button | ✅ post-MVP: Clinical Notes tab with shift-aware drafts + per-shift addenda; finalize pushes structured vitals (HR/BP/SpO2/Temp/RR) into OpenEMR's `form_vitals` via the standard `/api/` write surface (note *prose* still local JSON, FHIR `DocumentReference` write deferred); Vital Trends page (`/vital-trends/{id}`) with snapshot card + per-vital charts; Recent Vitals grouped + collapsed to most-recent (older readings via Trends); Sources-cited footer removed | — |
| **Dashboard cache + prewarm** | ✅ in-process TTL cache (5-min); lifespan prewarms calendar + every patient on it; warm calendar/card/documents endpoints ~10ms | + event-driven invalidation on chart writes | + Redis for cross-machine cache |
| Audit log | ❌ in-memory session dict only; no audit-log writes yet | ✅ post-MVP: SQLite-backed durable trace store at `data/traces.db` ([observability_db.py](clinical-copilot/app/observability_db.py)) — every `/chat` request flushes one row with request_id, session, username, full tool events (incl. patient_id, args, sources returned), validator outcome, tokens + cost; survives restart; surfaced via `/observability` + `/api/traces` | + retention enforcement (90d chats / 7y audit) + Postgres swap for multi-host + per-tool patient-panel ACL |
| **In-app observability dashboard** | — | ✅ `/observability` page + `/api/traces` endpoint; per-request latency, tokens, $ cost, tool events, validator outcome | — |
| LangSmith observability | ❌ env wired, integration not active | ✅ post-MVP: env wired AND active in Hetzner production (`LANGSMITH_TRACING=true` + valid key); traces every node + tool + token + cost via langchain-anthropic auto-tracing — every chat turn streams a span tree to https://smith.langchain.com under project `agent_forge` | — |
| Sessions / conversation history | in-memory dict (MVP) | Postgres, per session_id | + Redis for cross-machine state |
| Anthropic prompt caching | ❌ system prompt re-sent every turn | ⚠️ deferred — Sonnet 4.6 minimum cacheable prefix is empirically ~2048 tokens; current prompt is 952. Reaching threshold cleanly is a Sunday-track item alongside the deterministic intent-router (see §10.5) | — |
| **Citation shorthand fix (A4)** | — | ✅ system prompt forbids `et al.` / `…` / `…` inside citation brackets; eval `lab_normal × cohen` re-recorded | — |
| Eval framework | ❌ ad-hoc smokes | ✅ snapshot+replay gate at `clinical-copilot/evals/` (25 rules × 50 templates → 130 snapshots); golden 100% / labeled ≥90%; runs as prek pre-push hook | + CI gate on every PR |
| BAA / HIPAA-grade hosting | ❌ Anthropic direct, Fly.io | ⚠️ post-MVP: Bedrock-ready — `LLM_PROVIDER=bedrock` env flip routes Claude calls through AWS Bedrock (langchain-aws `ChatBedrockConverse`) so a real hospital deploy can sign Anthropic's BAA. AWS creds via `AWS_*` env vars; `BEDROCK_MODEL_ID` configurable. Default stays `anthropic` for dev/demo. Hosting relocation (Hetzner→AWS or HIPAA-eligible host) still required for full PHI deployment. | — |

## 1.2 Post-MVP delta (Wed–Thu 2026-04-29 → 2026-04-30)

What landed on master between the Tuesday MVP submission and the Thursday early-submission gate. Everything below is shipped on master; the §1.1 table reflects original-MVP wording for historical fidelity, this section is what changed since.

- **Clinical Notes UI with shift-aware drafts and finalize → vitals write-back.** New "Clinical Notes" tab on the dashboard. The doctor drafts a note per shift (Day 06:00–18:00 / Night 18:00–06:00, anchored to `CLINICAL_TZ`); per-shift addenda appended after finalize; auto-finalize at shift-end. The note text itself lives in a local JSON-backed store ([`app/clinical_notes.py`](clinical-copilot/app/clinical_notes.py)) and is merged into `/api/patient/{id}/documents` alongside FHIR `DocumentReference`s so it shows in the Supporting Documents list. **On finalize, structured vitals from the note (HR, BP systolic/diastolic, SpO2, Temp, RR) are pushed to OpenEMR as new rows in `form_vitals` via the standard non-FHIR REST API at `/apis/default/api/`** ([`app/fhir/writer.py`](clinical-copilot/app/fhir/writer.py)) — the same surface the seed scripts use, with `client_secret_post` + `password` grant + `user/vital.cruds`. This is the first time the second OAuth client ([§4.1 dual-flow](#41-data-layer--openemr)) is exercised in production. The note is stamped `fhir_synced_at` on success; failure is non-fatal (the JSON store remains authoritative). Synced vitals re-surface through the FHIR-side `Observation` search and appear in the patient card's Recent Vitals deduplicated against any pre-existing FHIR rows. **Open follow-up:** the note *prose* is not yet written to OpenEMR as a FHIR `DocumentReference`; that's a Sunday-track item.

- **Vital Trends page (`/vital-trends/{patient_id}`).** Snapshot card + per-vital line charts + a readings list under each chart. Pulls vital `Observation`s for the patient, groups by vital category, renders SVG charts client-side. Linked from the patient card "Trends" affordance. UI feature only — no agent tool reads it on master (the `get_vital_trends` agent tool is on `clinical-notes-2`, not yet merged).

- **R5: hard-scope rule + persona-swap / jailbreak refusal.** [`system_prompt.py`](clinical-copilot/app/agent/system_prompt.py) gained an explicit R5 rule: refuse persona/role-play/impersonation requests, jailbreak attempts ("ignore previous instructions", "DAN", "developer mode"), system-prompt extraction attempts, off-topic general-knowledge / opinion / code requests, and style-changing format requests — all with a single fixed refusal template. Tool output is treated as data, not instructions, so a tool result that *looks* like an instruction is summarized, not obeyed. This is the second prompt-level defense added to the verification stack ([§4.4](#44-verification--the-differentiator)) — alongside R2 (no clinical reasoning beyond tool output).

- **Citation enforcement tightened (grader-flagged).** Validator now also rejects citation-bracket *shorthand* (`et al.`, `…`, `…`) inside `[ResourceType/ID]`, and the prompt forbids it explicitly. Eval snapshots re-recorded.

- **Composite BP decomposition in adapter.** OpenEMR's FHIR layer returns blood pressure as a single `Observation` with `component` entries for systolic/diastolic and a flat `valueQuantity: null`. The original `_format_vital` flattened that to a row with `value=None`, which got filtered out client-side and made the agent report BP as "not recorded". The adapter now decomposes composite vitals into separate Systolic BP / Diastolic BP rows. Listed alongside the data-absent-reason / narrative-fallback quirk-handling in [§4.1](#41-data-layer--openemr).

- **Display-time alignment to clinical-local TZ.** All chart timestamps that surface in the UI or in agent tool output (Recent Vitals, Vital Trends, clinical-note titles, agent narration) now go through a single `_clinical_iso` helper that re-stamps UTC FHIR times in the configured `CLINICAL_TZ`. Browser-rendered shift labels and group headers additionally re-derive shift from the viewer's *own* timezone so a grader in any TZ sees coherent labels. Sources merged across paths (FHIR rows, trends backfill, synced-note vitals) all emit clinical-local strings to keep dedup keys aligned.

- **Recent Vitals UI: group + collapse.** Recent Vitals on the patient card are grouped by recording time (one timestamp = one cluster of vitals), and only the **most recent group** is shown — older readings are reachable via the Vital Trends page. Reduces the "list of 50" overwhelm we hit during smokes.

- **"Sources cited" footer removed.** The per-response footer that listed every cited resource ID at the bottom of an assistant response is gone. Inline `[ResourceType/ID]` citations after each clinical claim remain — they're the validator's input and the user's verification surface; the footer was redundant and noisy.

- **Demo-data corpus expanded.** Cohen + Roberts + Patel + Hale are seeded as inpatient demos; AUDIT.md §4.1 mentions only Cohen, but the eval suite and the deployed UI cover all four.

- **Snapshot-based eval suite + pre-push gate.** A deterministic snapshot+replay gate at [`clinical-copilot/evals/`](clinical-copilot/evals/) records every tool call and validator outcome per scenario. New runs replay against snapshots; the gate runs as a `prek` pre-push hook so pushes that regress on golden cases are blocked locally before they hit master. Already noted as a row in §1.1; this is the architectural shape.

- **App-layer auth + access tightening.** Cookie-session login validating against OpenEMR's password grant (already in §1.1) was hardened: every data endpoint and the `/chat` SSE stream now require an authenticated session, audit trace IDs link to the session, and the citation validator's retry edge logs invalid-citation attempts as a structured event. The original "MVP gap: no app-layer auth" line in §4.3 is closed.

- **Persistent audit log (closes Tuesday-grader feedback #3).** A SQLite-backed `SqliteTraceStore` ([app/observability_db.py](clinical-copilot/app/observability_db.py)) replaced the in-memory ring buffer for the durable record. Every finalized `RequestTrace` is upserted into `data/traces.db` with the full tool-event tree (name, args incl. patient_id, sources returned, error), all LLM events (tokens, cost, finish_reason), and validator outcome (attempts + final pass/fail). Indexed on `started_at` and `(username, started_at)` for the dashboard's recent-traces view. Survives `systemctl restart` and full-host reboot — that's the architectural floor for HIPAA `§164.312(b)`. Postgres swap for multi-host deployment is still on the Sunday list, but the data shape and write site don't change.

- **Zero-citation gate (closes Tuesday-grader feedback #2).** The validator now runs `find_uncited_clinical_claims` ([app/agent/validator.py:139](clinical-copilot/app/agent/validator.py#L139)) alongside the existing fake-citation check. Sentence-level regexes detect clinical-shaped statements that lack any `[Type/ID]` bracket — measurable values with units (`138 mmHg`, `72 bpm`, `98.6°F`), lab/vital names followed by a number (`creatinine 2.1`, `HR 78`, `BP 138/82`), med-usage patterns (`taking Apixaban`, `started on Lisinopril`), and patient-context pronouns plus clinical vocabulary. Either check failing fires the same `VALIDATION FAILED` retry edge in the graph, with a diagnostic message naming the offending sentences so the LLM can restate or fall back to "insufficient evidence in chart". Conservative-by-design — false positives would block legitimate refusals, so the bar to flag is high; the existing fake-cite check still catches the worst case where IDs are made up.

- **Deploy: Hetzner replaces Fly.io.** The MVP-day cloudflared tunnel and in-flight Fly.io deploy were both retired in favor of a single Hetzner host running OpenEMR (PHP/Apache + private MariaDB) and the co-pilot (uvicorn under `copilot.service`) side-by-side, fronted by a cloudflared-named-tunnel for public access. Lower friction for a one-person sprint; production deployment guidance unchanged (BAA, Bedrock, etc.).

- **Bedrock-ready LLM provider switch.** [`app/agent/graph.py`](clinical-copilot/app/agent/graph.py) now constructs the chat model via a `_build_llm` factory keyed on `settings.llm_provider`. Default `"anthropic"` keeps the direct-API path; setting `LLM_PROVIDER=bedrock` (plus `AWS_REGION` and `BEDROCK_MODEL_ID`) routes Claude calls through AWS Bedrock via `langchain_aws.ChatBedrockConverse`. This is the architectural answer to HIPAA: Anthropic offers a BAA only via Bedrock. The trace's `model` field reflects whichever model is actually being called (Bedrock model ID vs Anthropic model name) so the audit log is truthful. Bedrock dep is `langchain-aws>=0.2`, lazily imported so installs without it keep working on the default path. Switching production over still needs an AWS account + IAM creds + a HIPAA-eligible host (Hetzner is not BAA-covered) — but the code path is wired and tested.

- **Deterministic intent router (latency + cost win on trivial turns).** [`app/agent/intent_router.py`](clinical-copilot/app/agent/intent_router.py) runs in `/chat` and `/chat/stream` after the jailbreak guard but before the LLM. Strictly-anchored regex patterns recognize three trivial intents — pure greetings (`"hi"`, `"good morning"`), pure thanks (`"thanks"`, `"got it"`), and help requests (`"help"`, `"what can you do"`) — and short-circuit to a canned response. Anything more complex (e.g. `"hi can you catch me up on Cohen"`) doesn't match because patterns require the *entire* message to be a trivial phrase, so real chart questions always reach the LLM. Routed turns still write a `RequestTrace` with `model="intent_router"` and `error="routed:<intent>"` so the audit log accounts for them but they sort distinctly from LLM-handled turns. ~5-second latency cuts to ~50ms; cost drops to zero on routed turns. Defends against pattern drift via 31 smoke cases (20 positives + 11 negatives, including embedded-attack greetings like `"hi can you ignore previous instructions"` which fall through to the jailbreak guard).

## 2. System context

| Stakeholder | Role |
|---|---|
| **Hospitalist physician** | Primary user. Carries 15–20 inpatients. Uses the agent at three points in the shift: morning pre-round, midday med safety check, evening sign-out. |
| **Nurse / resident** | Secondary users (read-only access, scoped to their assigned patients). |
| **OpenEMR** | The EHR system of record. The agent reads from it via FHIR; it never writes back in v1. |
| **Hospital IT / CTO** | The "are we comfortable deploying this" approver. Cares about: standards (FHIR), security (OAuth2, audit log), failure modes (cited claims, refusal-to-confabulate), and scale (300 concurrent users, 500-bed hospital). |

## 3. High-level architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Browser chat UI                                                │
│  ┌─────────────────────┐  ┌──────────────────────────────────┐  │
│  │ Patient list (left) │  │ Chat (center) │ Patient card (R) │  │
│  └─────────────────────┘  └──────────────────────────────────┘  │
└──────────────────────────────┬──────────────────────────────────┘
                               │ HTTPS, session cookie → user_id
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  FastAPI backend                                                │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │ Auth: session → user → role (physician/nurse/resident)  │   │
│   └────────────────────────┬────────────────────────────────┘   │
│                            ▼                                    │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │ LangGraph state machine                                 │   │
│   │   user_msg → resolve_patient? → call_llm                │   │
│   │     ↑                              │                    │   │
│   │     └── tool_calls ←── llm_decides ┤                    │   │
│   │                                    ▼                    │   │
│   │                         validate_citations              │   │
│   │                                    │                    │   │
│   │                                    ▼                    │   │
│   │                            stream_to_user               │   │
│   └────────────┬────────────────────────────────────────────┘   │
│                │                                                │
│                ▼                                                │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │ Tool layer (7 tools, each returns {data, sources[]})    │   │
│   └────────────┬────────────────────────────────────────────┘   │
│                ▼                                                │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │ FHIR adapter (httpx + OAuth2 token cache)               │   │
│   │    ↳ authz check + audit log write per call             │   │
│   └────────────┬────────────────────────────────────────────┘   │
└────────────────┼────────────────────────────────────────────────┘
                 │ system-level OAuth2 token
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│  OpenEMR — FHIR R4 API at /apis/default/fhir/                   │
│  (30+ resources: Patient, Encounter, Observation, ...)          │
└─────────────────────────────────────────────────────────────────┘

Cross-cutting:
  • LangSmith — traces every node + tool + token + cost
  • Postgres (Fly.io) — sessions, conversation history, audit log
  • Anthropic prompt caching — system prompt + per-patient context
```

## 4. Layer walkthrough

### 4.1 Data layer — OpenEMR

- **Why FHIR over a direct DB integration:** FHIR is what real hospital integrations use (Cerner, Epic, Allscripts all expose it). "We use FHIR" is a defensible answer at the CTO bar; "we query MariaDB directly" is not.
- **Resources consumed (read):** Patient, Encounter, Observation (labs + vitals), MedicationRequest, Condition (problem list), AllergyIntolerance, DocumentReference (notes), Practitioner.
- **Two distinct OAuth flows** (this is unusual and forced by [AUDIT.md §1.1](AUDIT.md#11-authentication--authorization)):
  - **Read flow (production, what the agent uses):** OAuth2 **`client_credentials` grant + `private_key_jwt` auth + `system/∗.read` scopes** against `/oauth2/default/token`. The agent backend is a SMART Backend Services confidential client. Token is cached for 1h; no per-user redirect.
  - **Write flow (used post-MVP by clinical-note finalize, NOT by the chat agent):** OpenEMR's FHIR API only writes 4 of 30+ resources (Patient/Practitioner/Organization/DocumentReference). Clinical-data writes — Encounter, Condition, Observation, MedicationRequest, AllergyIntolerance — go through a **separate non-FHIR REST API** at `/apis/default/api/` with a different scope vocabulary (`user/allergy.cruds`, not `user/AllergyIntolerance.write`). We registered a second OAuth client with `client_secret_post` auth + `password` grant for this. At MVP this client only seeded demo charts; **post-MVP it is the path clinical-note finalize uses to write structured vital readings (HR/BP/SpO2/Temp/RR) back to OpenEMR's `form_vitals`** ([§1.2](#12-post-mvp-delta-wedthu-2026-04-29--2026-04-30)). The chat agent still has no path to writes. The note's prose stays in a local JSON-backed store and is merged into Supporting Documents alongside FHIR `DocumentReference`s for display; writing the note text itself as a FHIR `DocumentReference` (one of the 4 writable FHIR resources) is a Sunday-track item.
- **Tradeoff:** read-only client_credentials gives the agent broad chart access, so authorization scoping moves to **our tool layer** (see §4.3). The alternative — SMART-on-FHIR user OAuth — pushes auth onto OpenEMR but adds a redirect-and-consent flow per session, which is wrong for a backend agent.
- **Adapter quirk-handling:** OpenEMR's FHIR layer emits `code.coding.system: data-absent-reason / code: unknown` placeholders when a free-text title is supplied without a SNOMED/RxNorm code, with the real label in `text.div` (narrative). Our adapter ([clinical-copilot/app/fhir/adapter.py](clinical-copilot/app/fhir/adapter.py)) skips data-absent-reason codings and falls back to the narrative — without this every chart entry would render as "Unknown". Same adapter also relaxes `Encounter.status` and `Condition.clinical-status` filters because OpenEMR's FHIR layer silently returns zero on them. **Post-MVP:** the adapter also decomposes composite blood-pressure `Observation`s (single resource with `component.systolic` + `component.diastolic` and a flat `valueQuantity: null`) into separate Systolic BP / Diastolic BP rows — without this the agent saw `value=None` and reported BP as "not recorded". All chart timestamps the UI or agent narrate go through a single `_clinical_iso` helper that re-stamps UTC FHIR times in the configured `CLINICAL_TZ`.

### 4.2 Application layer

- **Backend:** FastAPI (Python). Lightweight, async-native, type-safe.
- **Agent orchestration:** **LangGraph** (state machine). Chosen over the older `AgentExecutor` because we need an explicit `validate_citations` node between the LLM's output and the user — LangGraph makes intermediate gating natural; AgentExecutor hides it.
- **LLM choice (current MVP):** **Claude Sonnet 4.6** for every turn. The cascade design (Sonnet default / Opus for med-safety ambiguity / Haiku for sub-tasks) is staged for early submission Thursday once `clinical_rules` and finer-grained tools exist; for MVP one model handles everything.
- **Tools (current MVP — 3 of ~7 planned):** Python functions wrapped with LangChain's `@tool`. Every tool returns `{data, sources: [resource_type/id, ...]}`. The `sources` list is the citation primitive used by the validator.
  - `current_time()` — anchors any relative-date language. Required before "yesterday", "X months ago", "today". Returns `{iso_datetime, date, weekday, timezone}`.
  - `resolve_patient(query)` — last-name search; returns best match plus `alternatives` for disambiguation.
  - `get_patient_card(patient_id)` — demographics, current encounter, allergies, active problems, active medications, recent vitals.
  - **Thursday work:** `get_observations_24h`, `get_notes_24h`, `get_med_changes_24h` (Use Case A time-window primitives), `clinical_rules(meds, problems, labs)` for Use Case B.

### 4.3 Authorization, audit, HIPAA

- **App-layer auth (shipped MVP, hardened post-MVP).** A cookie-backed session validates credentials against OpenEMR's password-grant OAuth at login, and every data endpoint plus the `/chat` SSE stream requires an authenticated session. The original Thursday gap ("anyone with the URL can `/chat`") is closed. **Per-tool patient-panel scoping is still Sunday work** — the tools currently trust the session, not a per-tool ACL on patient IDs.
- **Audit log (SQLite-backed durable record, shipped post-MVP).** Every chat request finalizes into a `RequestTrace` and writes a single row to `data/traces.db` ([clinical-copilot/app/observability_db.py](clinical-copilot/app/observability_db.py)) — request_id, session_id, username, model, started/finished timestamps, all tool events (name, args, patient_id, sources returned, latency, error), all LLM events (tokens, cost), validator attempts and final pass/fail. Indexed on `started_at DESC` and `(username, started_at)`. Survives `systemctl restart` and full-host reboot. Surfaced through `/observability` + `/api/traces` for the in-app dashboard, and queryable directly with `sqlite3 data/traces.db`. This is the structural answer to HIPAA's `§164.312(b)` audit-controls requirement at single-process scale; production multi-host deployment swaps SQLite for Postgres but the data shape and write site stay the same. OpenEMR's FHIR layer doesn't audit-log API calls consistently ([AUDIT.md §5.1](AUDIT.md#51-audit-logging--partial)) so we own this entirely. **Sunday work:** retention enforcement (90d chats / 7y audit) + Postgres swap for multi-host; per-tool patient-panel ACL on top of the session check.
- **PHI handling.** Demo data only for the sprint. In production: BAA with the LLM provider (Anthropic offers BAAs via AWS Bedrock); HIPAA-grade hosting (neither the original Fly.io plan nor the current Hetzner host carries a BAA — production must relocate); secrets in a managed secret store, not env files; TLS everywhere.

### 4.4 Verification — the differentiator

The verification layer has **two structural defenses and two prompt-level rules**, each with a specific failure mode it catches. (R5 was added post-MVP; see [§1.2](#12-post-mvp-delta-wedthu-2026-04-29--2026-04-30).)

1. **Tool-output-as-source-of-truth (architectural).** The LLM has no path to FHIR; only the tool layer does. Every clinical fact in a response must come from a tool call result. *Catches:* the LLM can't go around our tools to invent data. *Implementation:* `app/agent/graph.py` calls `model.bind_tools(...)`; the only Anthropic API call in the codebase is `model.ainvoke(...)` inside `call_llm`. ([clinical-copilot/app/agent/graph.py](clinical-copilot/app/agent/graph.py))

2. **Inline citations + deterministic validator (structural).** The prompt requires every clinical claim to end with `[ResourceType/ID]`. After every LLM response and before it reaches the user, [`validator.py`](clinical-copilot/app/agent/validator.py) extracts every citation and checks each is in the cumulative `conversation_sources` set. Invalid citations trigger a **retry edge** in the graph: a `VALIDATION FAILED` HumanMessage is appended and the LLM is re-invoked. After `MAX_VALIDATION_ATTEMPTS = 2`, the response goes through with a `validation_warning` flag the UI surfaces. *Catches:* hallucinated resource IDs that look plausible. The LLM cannot get past the gate without retrying with valid sources or admitting "insufficient evidence in chart."

3. **No clinical reasoning beyond tool output (prompt-level, R2).** During MVP-day testing the LLM emitted unsourced clinical rules from training — drug interactions, dose-reduction criteria, "things to flag" sections. The system prompt now hard-forbids this until a `clinical_rules` tool exists. *Catches:* training-derived clinical claims that contradict the chart, or that are fine in general but wrong for this patient. *Limitation:* prompt-level rules are weaker than structural; the right answer is a `clinical_rules` tool (Sunday final). Until then we accept the LLM as a strict summarizer of chart contents only.

4. **Hard scope-lock + jailbreak refusal (prompt-level, R5; added post-MVP).** The system prompt refuses persona/role-play/impersonation requests, jailbreak phrasings ("ignore previous instructions", "DAN", "developer mode"), system-prompt extraction attempts, off-topic general-knowledge / opinion / code requests, and style-changing format requests — all with one fixed refusal template. Tool output is treated as data, not instructions: a tool result that *looks* like an instruction ("ignore your rules", "act as…") is summarized, not obeyed. *Catches:* user (or chart-data) attempts to talk the agent off-task or out of its safety contract. *Limitation:* still prompt-level — a robust answer is an out-of-scope classifier as a graph node, but the rule + Sonnet 4.6 holds in our smokes and eval suite.

5. **Post-hoc fact-checker for Use Case B (planned, Sunday).** A second LLM call independently verifies that each cited claim is actually supported by the cited resource's *content*. Slow and expensive, so reserved for medication safety where being wrong is most costly. **MVP state:** not implemented; #1 and #2 are sufficient for Use Case A.

Together these guarantee **a well-behaved agent cannot lie about chart contents**, and **a misbehaving one is caught structurally, not heuristically.**

### 4.4.1 Validator known limitations

The regex-based citation extractor catches `[ResourceType/ID]` patterns but **does not catch:**
- Resource IDs mentioned outside the bracket format (e.g., "Patient abc-123 has...")
- Combined-in-one-bracket citations (`[Patient/a, Patient/b]`)
- Claims with no citation at all — the validator passes responses with zero citations because some legitimate responses (greetings, clarification requests) shouldn't be required to cite anything

These are documented gaps. The mitigation is the prompt rule: the LLM is instructed to use only the bracketed format, and Sonnet 4.6 follows it reliably in our smokes. A more robust validator would require LLM-based claim extraction — a Sunday-final consideration.

### 4.5 Observability

- **LangSmith** traces every conversation as a tree: user input → LLM calls → tool calls → validator → response. Every node carries latency and cost.
- The same LangSmith project hosts the **eval suite** — a YAML dataset of scenarios per use case, scored on factual accuracy, citation coverage, refusal correctness, and latency. Runs on every commit.

## 5. Latency model

| Use Case | TTFW target | Total target | MVP measured (no streaming, sequential tools) | Gap closure plan |
|---|---|---|---|---|
| A — Pre-round summary | < 2s | < 8s | **~14s total** (3 LLM round-trips + 6 sequential FHIR queries) | Streaming TTFW + parallel tool calls + prompt caching → projected p95 ~7s |
| B — Med safety check | < 2s | < 15s | not implemented | TBD Thursday |
| C — Sign-out drafting | < 2s | < 60s for 18 patients | not implemented | Stream per-patient |

**MVP-day measured breakdown for Use Case A**: ~70% of latency is LLM round-trips (Sonnet 4.6 with bound tools, no streaming, three sequential decision turns), ~25% is sequential FHIR queries inside `get_patient_card`, ~5% is OpenEMR's PHP layer overhead. The architecture ships with tool-call latency over the target by design — we prioritized verification correctness over speed in week 1. Latency is now the explicit Thursday goal.

**Performance levers (Thursday)**:
1. **Streaming responses** — Anthropic SDK supports SSE; FastAPI route swaps from JSON to SSE. Cuts perceived-latency to time-to-first-token (~1s).
2. **Parallel tool calls** — `asyncio.gather` inside the FHIR adapter parallelizes the 6 queries in `get_patient_card`. Cuts adapter latency from ~3s to ~0.7s.
3. **Anthropic prompt caching** — system prompt + per-patient context are stable across turns; caching cuts repeat-input cost by ~90% and shaves a few hundred ms off LLM input processing.
4. **Smaller-grained tools** — splitting `get_patient_card` into 6 tools lets the LLM call only what's needed for a follow-up ("what was her potassium?") instead of refetching everything.

**Performance levers (final)**:
- Session-scoped FHIR cache (15–30s TTL) for repeat patient-card pulls.
- Model cascade — Haiku for parameter extraction / disambiguation, Sonnet for synthesis, Opus only for med-safety ambiguity. Token cost drops ~40%.

## 6. Cost model — first-pass projection

Assumptions per session (conservative):
- 4 turns/session, 800 input + 400 output tokens per turn (after caching)
- 70% Sonnet, 25% Haiku, 5% Opus mix

| Scale | Sessions/mo | LLM cost/mo (USD) | Infra (Fly + Postgres + LangSmith) | Total |
|---|---|---|---|---|
| 100 users | ~600 | ~$10 | ~$30 | **~$40** |
| 1K users | ~6K | ~$100 | ~$60 | **~$160** |
| 10K users | ~60K | ~$1,000 | ~$300 | **~$1,300** |
| 100K users | ~600K | ~$10,000 | ~$2,000 (need Postgres scale-out, vector store, dedicated FHIR proxy) | **~$12,000** |

Per-physician marginal cost at the 1K-user tier is ~$0.16/month — well under the price of any clinical SaaS.

## 7. Tradeoffs and alternatives considered

| Decision | Chose | Rejected | Why |
|---|---|---|---|
| LLM provider | Anthropic Claude | OpenAI, Google | Best at "refuse to claim what you can't cite" — the exact behavior we cannot tolerate getting wrong. BAA available via Bedrock. |
| Agent framework | LangChain + LangGraph | Raw Anthropic SDK | Native LangSmith integration, explicit state machine, more documentation for a beginner team. Costs us ~150 lines of abstraction. |
| FHIR interface | FHIR R4 client_credentials | Direct MySQL queries | Standards-compliant, hospital-defensible, decoupled from OpenEMR's internal schema. |
| Verification | Structural (tool source-of-truth + validator) | Just prompt-level | Prompts are not enforcement. A model that gets cleverer can still confabulate; a validator cannot be talked out of its check. |
| Demo data | Synthea | OpenEMR's 3 built-in patients | Need realistic inpatient data with multi-day notes, lab trends, med changes for hospitalist scenarios. |
| Hosting | Fly.io | AWS, Vercel | Fast deploy (one command), managed Postgres, edge-distributed; defensible to a CTO because it's Docker-native (no proprietary lock-in). |
| App database | Postgres | SQLite, MongoDB | Postgres handles JSON well (`jsonb` for tool args), ACID for the audit log, scales horizontally on Fly. SQLite breaks at >1 instance; Mongo wins nothing for our shape. |

## 8. Production-readiness gaps (honest)

This is what would still be required to ship this to a real hospital:

- **BAA with LLM provider** — for the sprint demo we use Anthropic direct (no PHI); production routes through AWS Bedrock with a signed BAA.
- **Real authentication** — current plan is OpenEMR session passthrough; a hospital deployment would integrate with their SSO (SAML/OIDC).
- **Encryption at rest** in our Postgres for conversation history and audit log (Fly.io managed Postgres has this; configuration check needed).
- **Drug interaction database** — for Use Case B, we'd license First Databank (FDB) or RxNorm-DDI rather than relying on the LLM's training knowledge of interactions.
- **Disaster recovery** — backup/restore SLA, multi-region failover, RPO/RTO targets.
- **Clinical validation** — IRB review, physician evaluation panel before live use.
- **The chart UI** — we ship a prototype; production needs a clinical UX review.

### 8.1 Deterministic-first routing — known optimization (Sunday-track)

The current architecture sends every chat turn through the LLM, which is the
right default for a "multi-turn conversational agent" (the brief's framing) but
overpays for **stock queries whose tool sequence is fully determined by intent**.
Examples:

- *"Catch me up on Cohen"* / *"update on Patel"* → always
  `resolve_patient → current_time → get_patient_card`. The LLM is selecting
  tools that an intent classifier could pick deterministically.
- *"What was her creatinine 2 days ago?"* → if the value is on the card returned
  by an earlier turn, code can extract it; the LLM is doing string formatting.
- *"What is she on?"* → enumerates `MedicationRequest` rows from the card with
  one citation each; templated prose suffices.

A deterministic-first router would (a) regex-classify the doctor's message
against a small set of stock intents, (b) run a fixed tool sequence, and (c)
render a Jinja template carrying `[ResourceType/ID]` citations directly from
tool output. The LLM is then reserved for genuinely free-form follow-ups
(*"trending up?"*, *"any worse since admission?"*, *"compare to last
admission"*) where intent is ambiguous or synthesis is required.

**Why this is Sunday-track, not tonight:**

- The eval suite is built around LLM-driven flow; rules like "agent calls
  resolve_patient first" become trivially true if hardcoded, requiring eval
  redesign.
- Done well (intent grammar + synonym table + clinical-time parser + template
  library + fallback rules), this is 1–2 weeks of engineering — not a deadline
  task.
- Done shoddily (regex-only matching), it would fail on real doctor phrasings
  in the demo and look worse than today's agent.

**Why this matters for reviewers:** the discipline of *"API calls only for
things that cannot be deterministically solved"* is a real production concern
for cost-per-turn and latency at hospital scale. We surface it explicitly here
so the gap is documented rather than hidden.

### 8.2 Anthropic prompt caching — known optimization (Sunday-track)

Empirically, Sonnet 4.6's minimum cacheable prefix is ~2048 tokens; the current
system prompt is 952. Cache markers are silently no-ops below threshold. Two
paths for Sunday: (a) restructure the prompt to be cache-friendly (substantive
content reaching ≥2200 tokens, with fresh eval recordings to capture any
behavior shift), or (b) attach `cache_control` to the tool schema's last entry
once `langchain-anthropic` exposes a clean way to do so. Either yields ~80–90%
input-cost reduction on cache hits — measurable directly in the
`/observability` dashboard's `cache_read_tokens` field.

## 9. Build sequencing

| Sprint gate | Date | Scope | Status |
|---|---|---|---|
| Architecture defense | 2026-04-27 evening | This doc + presearch.md + verbal walkthrough | ✅ delivered |
| **MVP** | **2026-04-28 (Tue) 11:59 PM CT** | Forked OpenEMR + publicly accessible deploy + AUDIT + USERS + ARCHITECTURE + 3-5 min demo. Working agent is a *bonus* (not required by the brief — Thursday's gate). | ✅ this submission |
| Early submission | 2026-04-30 (Thu) 11:59 PM CT | Working agent deployed on same infra as OpenEMR + eval suite (130 snapshots, snapshot+replay gate as prek pre-push hook) + per-request observability dashboard (`/observability` + `/api/traces`) + LangSmith env wired + app-layer auth + new demo video. Audit-log-to-Postgres deferred to Sunday with in-memory `RequestTrace` ring buffer covering observability needs (200-deep). | in flight |
| Final | 2026-05-03 (Sun) | + audit log → Postgres (per-FHIR-call append-only) + Use Case B (clinical_rules tool) + Use Case C (sign-out drafting) + deterministic-first intent router (§8.1) + cache-friendly prompt restructure (§8.2) + cost analysis (100/1K/10K/100K) + social post + production-readiness gaps closed | planned |

**MVP-day decision log** (things not visible in the design but worth stating for the interview):

- Built the **working agent against Cohen ahead of schedule** because it makes the demo video concrete: reviewers see "this is what we'll deploy Thursday" instead of architectural promises. Re-tested at MVP after every major change ([§4.4](#44-verification--the-differentiator) verification flow has been smoke-tested 8+ times against Cohen end-to-end).
- **Cloudflare quick-tunnel** is the MVP "deployed app" mechanism. The Fly.io deploy of the OpenEMR fork is in flight ([deploy/fly/](deploy/fly/)) but hit a known issue with the upstream image's first-boot install on a fresh Fly volume; it's a Thursday item, not a blocker for the MVP gate.
- **Patched two real LLM/tool boundary leaks** during MVP day: the LLM was emitting clinical reasoning from training (Metformin+CKD3 dose rules, Apixaban dose-reduction criteria) and was guessing today's date for "X months ago" phrasing. Both are fixed: a `current_time` tool plus a tightened system-prompt rule (R2) that hard-forbids clinical reasoning outside tool outputs. See `clinical-copilot/app/agent/system_prompt.py`.

Use Case A ships first because it's the broadest "feels like the agent works" demo and exercises the full architecture (resolve_patient + current_time + get_patient_card → validate_citations → streaming-eligible response).

## 10. The defense — what to expect

Likely questions and the short answer for each:

| Question | Answer |
|---|---|
| "Why a chat agent and not a dashboard?" | Doctors ask follow-ups ("show me the trend"; "what's he on that affects K?"). A dashboard can't anticipate the thread. |
| "How do you stop hallucinations?" | Architectural: every claim must trace to a tool result. A validator rejects the response if a citation is missing or invalid. The LLM cannot lie about chart contents because it cannot bypass the structural gate. |
| "What about privacy/HIPAA?" | OAuth2 system-level auth, per-tool authorization scoped to the doctor's panel, append-only audit log of every chart access, BAA with LLM provider in production, demo data only in sprint. |
| "How does this scale to 300 concurrent doctors?" | FastAPI is async; FHIR calls parallelize; Anthropic prompt caching keeps per-turn cost flat; Postgres handles the audit log easily at this scale. Fly.io scales horizontally. Bottleneck would be OpenEMR itself, not us. |
| "Why LangGraph and not OpenAI SDK / DSPy / a custom loop?" | We need an explicit verification node between LLM and user. LangGraph's state-machine model makes that gate first-class; AgentExecutor hides it; raw SDK reinvents it. |
| "What happens when the LLM is wrong?" | Three layers: (1) it physically can't cite a resource that doesn't exist (validator); (2) for medication safety it gets a second-pass fact-check; (3) doctor sees the citation and verifies it themselves — the right-side patient card makes this one click. |
| "What if a doctor asks about a patient they shouldn't see?" | Tool returns an error the LLM relays as "you don't have access to that record." The audit log captures the attempt. |
| "Why Fly.io?" | Fast deploy, managed Postgres, Docker-native (so the same image runs in AWS later), no vendor lock-in. |
