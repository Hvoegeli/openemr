# Clinical Co-Pilot — an OpenEMR fork with an AI agent for hospitalists

[![Forked from openemr/openemr](https://img.shields.io/badge/forked%20from-openemr%2Fopenemr-blue)](https://github.com/openemr/openemr)
[![Final submission: 2026-05-03](https://img.shields.io/badge/Final%20submission-2026--05--03-success)](#final-submission)

> **AgentForge — Clinical Co-Pilot**, Gauntlet AI Austin admission track.
> A multi-turn AI agent that helps a hospitalist physician catch up on inpatients fast — reading the chart from a forked [OpenEMR](https://github.com/openemr/openemr) via FHIR R4, summarizing what matters, and citing every clinical claim back to a specific record. Read-only over real EHR data, structurally verified, designed against the "hospital CTO bar".

---

## ★ Final submission — 2026-05-03

Everything a reviewer needs is in this section. Each link below is at the **top of the repo**.

### Deployed app

| | URL | What it is |
|---|---|---|
| **OpenEMR fork** (system of record) | https://alberta-expansion-pittsburgh-vitamin.trycloudflare.com/ | Our forked OpenEMR. Login: `admin` / `pass` (default — flagged in [AUDIT.md §1.2](AUDIT.md#12-default-credentials-and-secrets)). |
| **Clinical Co-Pilot** (the AI agent) | https://deutschland-air-spreading-wants.trycloudflare.com/ | The agent UI. Sign in with the same `admin` / `pass`. Type *"Catch me up on Cohen."* — the demo patient is seeded. |

### Required documents

All required documents (and supplementary references) are at the **root of the repo**:

| Doc | What it is |
|---|---|
| [USERS.md](USERS.md) | **Stage 4** — target user (hospitalist), workflow, six shipped use cases (A, C, D, E, F, G) plus the deliberate scope-out of B, each with an explicit "why an agent" defense |
| [AUDIT.md](AUDIT.md) | **Stage 3** — five-section audit of OpenEMR with a 500-word summary leading with the highest-impact findings |
| [ARCHITECTURE.md](ARCHITECTURE.md) / [ARCHITECTURE.pdf](ARCHITECTURE.pdf) | **Stage 5** — agent integration plan, 500-word summary, implementation-status table, layer walkthrough, latency + cost models |
| [clinical-copilot/evals/RESULTS.md](clinical-copilot/evals/RESULTS.md) | Eval suite results — last run pass rates, per-rule breakdown, known failures |
| [presearch.md](presearch.md) | Phase 1–3 pre-search constraints + decision log |

### What's new since MVP day

The MVP shipped Tuesday with a working chart-summary agent (Use Case A). Between Tuesday and Sunday we added:

- **Clinical Notes tab + vitals round-trip** (Use Case C) — doctor types the shift note alongside the patient card; on finalize the structured vitals (HR, BP, SpO2, Temp, RR) round-trip into OpenEMR's `form_vitals` chart. The note prose appears in Supporting Documents.
- **`clinical_flags` rule engine** — surfaces well-known fact pairs from the chart (metformin + low eGFR, warfarin + high INR, Bactrim + sulfa allergy, etc.) with citations. Surfaces facts, not advice.
- **Time-windowed FHIR tools** — `get_observations_24h`, `get_notes_24h`, `get_med_changes_24h` so the agent can answer "what changed overnight?" without re-pulling the whole chart.
- **Defense-in-depth jailbreak guard** — pre-LLM regex scrubber (`app/agent/input_guard.py`) catches role-override / prompt-injection attempts before the LLM is even called; logs them in the audit feed.
- **Deterministic intent router** — pure greetings / thanks / help short-circuit to a canned reply in ~50ms with zero LLM cost. Strict full-message anchoring; real chart questions always reach the LLM.
- **Durable server-side sessions** ([app/auth_db.py](clinical-copilot/app/auth_db.py)) — cookie holds an opaque sid; idle (30 min) and absolute (12 h) timeouts enforced server-side; admin can revoke any session; every login / logout / revocation is in the auth-events audit log.
- **Admin oversight page** (`/admin`, gated to `ADMIN_USERNAMES` env allow-list) — three live panels: active sessions, recent chat activity, auth events. The verification + observability story is browsable, not just claimed.
- **Anthropic prompt caching** — system prompt is now cache-eligible; ~80–90% input-cost reduction on cache hits, measurable in `/observability` `cache_read_tokens`.
- **Bedrock-ready provider switch** — `LLM_PROVIDER=bedrock` flips Claude calls onto AWS Bedrock so a real hospital deploy can sign Anthropic's BAA. Default stays `anthropic` for dev.
- **Per-tool patient-panel ACL + admin assignment UI** ([app/access_control.py](clinical-copilot/app/access_control.py), [app/web/admin.html](clinical-copilot/app/web/admin.html)) — calendar, every per-patient endpoint, and every patient-id-taking agent tool gates against a per-user assignment table. Admin UI is the only legal mutation point. Empty panel returns a clean "no patient found" rather than a leak; mismatches are written to the auth-events log.
- **Eval suite + pre-push gate** — 150 snapshots × 25 rules (added ACL boundary cases + Use Case D/E/F coverage); golden 100% / labeled ≥90% required to push; running results in [clinical-copilot/evals/RESULTS.md](clinical-copilot/evals/RESULTS.md).

### Caveat for reviewers

Both URLs are publicly reachable and run on a dedicated **Hetzner Cloud CPX21** in Ashburn, VA — no laptop in the path. OpenEMR is the standard `docker/development-easy` docker-compose stack; the co-pilot runs as a `systemd` service alongside it; both `cloudflared` quick-tunnels are themselves `systemd` services with `Restart=always`. The agent streams responses via SSE — token-by-token output begins in ~2s with progress indicators ("Searching for patient…", "Loading chart…") in between. Our originally-attempted Fly.io deploy of OpenEMR (configs in [`deploy/fly/`](deploy/fly/)) hit a known issue with the upstream image's first-boot install path on a fresh Fly volume; the Hetzner deploy uses the same docker-compose that's verified working locally and sidesteps that bug.

**Scope note on verification.** This is a **chart summarizer + clinical-notes capture tool**, not a clinical advisor. The `clinical_flags` rule engine surfaces well-known fact pairs from the chart (with citations), but it does not recommend actions. We deliberately do not implement advisory medication-safety or drug-interaction checks (the brief's "domain constraint enforcement" requirement) — a confidently-wrong dosage *recommendation* is the exact patient-harm failure the brief warns against. Drug-interaction databases (FDB, RxNorm-DDI) are the right tool for that job. See [ARCHITECTURE.md §8 — Production-readiness gaps](ARCHITECTURE.md#8-production-readiness-gaps-honest) for the deliberate-scope discussion.

---

## What's in this repo

```
openemr/                        # the OpenEMR fork (PHP/Apache, MariaDB)
├── README.md                   # ← you are here
├── README.openemr-upstream.md  # the original upstream OpenEMR README
├── USERS.md                    # ★ Stage 4 — target user, workflow, use cases
├── AUDIT.md                    # ★ Stage 3 — security/perf/arch/data-quality/compliance audit
├── ARCHITECTURE.md             # ★ Stage 5 — agent integration plan + 500-word summary
├── ARCHITECTURE.pdf            # rendered architecture doc
├── presearch.md                # Phase 1-3 pre-search per the brief
├── deploy/fly/                 # Fly.io deploy configs (mariadb + openemr)
│   ├── db.toml
│   └── openemr.toml
└── clinical-copilot/           # ★ The AI agent (this is the new code)
    ├── app/
    │   ├── agent/              # LangGraph state machine + citation validator
    │   ├── fhir/               # OAuth2 + FHIR R4 client + adapter
    │   ├── main.py             # FastAPI entry
    │   └── web/index.html      # minimal browser chat UI
    ├── scripts/
    │   ├── register_oauth_client.py    # one-time: read-only system client
    │   ├── register_seed_client.py     # one-time: write-capable seed client
    │   ├── seed_cohen.py               # populate the demo patient
    │   ├── smoke_fhir.py / smoke_anthropic.py / cli_chat.py
    │   └── fly_set_secrets.sh
    ├── Dockerfile
    └── fly.toml
```

The OpenEMR fork itself (`/src`, `/library`, `/interface`, `/apis`, etc.) is **unmodified** from upstream. All net-new code is in [`clinical-copilot/`](clinical-copilot/).

---

## The agent — what it actually does

A focused tool, not always-on. Six shipped use cases (see [USERS.md](USERS.md) for the full set with "why an agent" defense per case):

- **A — Pre-round patient summary** ("Catch me up on Cohen") — *shipped.*
- **C — End-of-shift clinical notes with vitals round-trip** — *shipped*; doctor charts the shift note alongside the patient card, finalize pushes structured vitals to OpenEMR's `form_vitals`. The doctor still signs their actual handoff in OpenEMR's own workflow — we removed the previously-planned agent-generated sign-out *draft document* on purpose; a parallel agent draft would diverge from the legal record.
- **D — 24-hour lab trend review** ("what's drifting?") — *shipped*; time-windowed Observations with per-result citations.
- **E — Overnight watch handoff brief** ("what does the night team need to keep an eye on?") — *shipped*; surfaces nursing notes, med changes, and observation drift. Never tells the night team what to *do*.
- **F — Time-window delta** ("what's changed since I rounded yesterday?") — *shipped*; doctor-specified window via `hours=N` on the time-windowed tools.
- **G — Daily list / panel overview** ("walk me through my list") — *shipped*; calendar respects per-user panel ACL; no cross-panel leakage.
- **B — Medication safety check (advisory)** — *deliberately scoped out.* The `clinical_flags` tool surfaces chart-internal fact pairs (metformin + low eGFR, warfarin + high INR, Bactrim + sulfa allergy, etc.) but **does not recommend actions**. A confidently-wrong dosage / interaction recommendation is the exact patient-harm failure the brief warns against; the right tool for that job is a licensed drug-interaction database (FDB, RxNorm-DDI), not an LLM's training knowledge. See [ARCHITECTURE.md §8.3](ARCHITECTURE.md#83-deliberate-scope-choices--what-we-are-not-building-and-why).

Verification is **structural, not best-effort**: the LLM has no path to FHIR, every tool returns `{data, sources: [...]}`, and a deterministic citation validator rejects responses that cite resource IDs not in the cumulative tool-output set. The system prompt also forbids the LLM from emitting clinical reasoning (drug interactions, dose-reduction rules) that didn't come from a tool — exactly the "confident hallucination → patient harm" failure mode the brief calls out. A pre-LLM jailbreak guard catches role-override and prompt-injection attempts before the LLM is even called, and every blocked attempt is recorded in the audit feed.

A live demo run against Cohen (HTN / T2DM / CKD3 / AFib, on Lisinopril / Metformin / Apixaban / Atorvastatin) produces 23 cited clinical claims, validator passes 0 retries, the BP question gets a refused "insufficient evidence" rather than a confabulated value. See `ARCHITECTURE.md §1` for the design rationale and `AUDIT.md §1` for the OpenEMR-side findings that shape it.

---

## Quick start (local dev)

### 1. Run OpenEMR locally

```bash
cd docker/development-easy
docker compose up --detach --wait
# → https://localhost:9300/  (admin / pass)
```

### 2. Start the agent

```bash
cd clinical-copilot

# one-time: register the read-only OAuth client
PYTHONPATH=. uv run python scripts/register_oauth_client.py
# (paste the printed OPENEMR_CLIENT_ID into .env, then enable
#  the client in OpenEMR admin → System → API Clients)

# one-time: register the demo-data seed client + seed Cohen
PYTHONPATH=. uv run python scripts/register_seed_client.py
PYTHONPATH=. uv run python scripts/seed_cohen.py

# start the agent
PYTHONPATH=. uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
# → http://127.0.0.1:8000/
```

Try `Catch me up on Cohen.` in the chat. You'll see the agent call `current_time → resolve_patient → get_patient_card`, validate citations, and surface BP-not-charted as `Insufficient evidence` rather than confabulating.

See [`clinical-copilot/.env.example`](clinical-copilot/.env.example) for the full env contract.

### 3. (Optional) Public URL via cloudflared

```bash
brew install cloudflared
cloudflared tunnel --url https://localhost:9300 --no-tls-verify
# → prints a https://*.trycloudflare.com URL
```

This is the same mechanism we used for the MVP "deployed app" link above.

---

## What's shipped (sprint history)

| Sprint gate | Date | What landed | Status |
|---|---|---|---|
| **MVP** | 2026-04-28 | Forked + deployed OpenEMR, Stage 3-5 docs, working agent against Cohen (Use Case A), cookie-session login, SSE streaming, dashboard TTL cache + startup prewarm, citation-click navigation, demo video | ✅ shipped |
| **Early submission** | 2026-04-30 | Hetzner same-host deploy, eval framework (145 snapshots / 25 rules, prek pre-push gate), LangSmith observability live in production, durable SQLite trace store + `/observability` page, app-layer auth, defense-in-depth jailbreak guard, deterministic intent router, Anthropic prompt caching, Bedrock-ready provider switch, Clinical Notes tab with vitals round-trip (Use Case C), time-windowed FHIR tools, vital-trends UI | ✅ shipped |
| **Final** | 2026-05-03 | Durable server-side sessions ([app/auth_db.py](clinical-copilot/app/auth_db.py)) with idle/absolute timeouts + admin-driven revocation + auth-events audit log, admin oversight page (`/admin`), `clinical_flags` rule engine (chart-internal fact-pair surfacing), per-tool **patient-panel ACL** ([app/access_control.py](clinical-copilot/app/access_control.py) + admin assignment UI), eval suite expanded to 150 snapshots (ACL boundary cases + Use Case D/E/F coverage), cost analysis with per-tier architectural notes, scope-defense docs in ARCHITECTURE.md §8.3 | ✅ shipped |

Two deliberate scope changes mid-sprint, both documented in [ARCHITECTURE.md §8](ARCHITECTURE.md#8-production-readiness-gaps-honest):
- **Use Case B (medication safety, advisory)** — out of scope by design. The `clinical_flags` engine surfaces fact pairs but does not recommend actions.
- **Sign-out drafting (the previously-planned agent-generated handoff document)** — removed. The Clinical Notes tab covers the same end-of-shift moment; the doctor signs their actual handoff in OpenEMR's own workflow.

---

## Acknowledgements

This is a fork of [OpenEMR](https://github.com/openemr/openemr) — a 20-year-old open-source EHR with a real codebase and a real user community. The original upstream README is preserved at [README.openemr-upstream.md](README.openemr-upstream.md). All credit for OpenEMR itself goes to the OpenEMR project and its contributors. Net-new code in this fork is in [`clinical-copilot/`](clinical-copilot/).

Built with Claude Code as part of the Gauntlet AI Austin admission track.
