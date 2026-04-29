# Clinical Co-Pilot — an OpenEMR fork with an AI agent for hospitalists

[![Forked from openemr/openemr](https://img.shields.io/badge/forked%20from-openemr%2Fopenemr-blue)](https://github.com/openemr/openemr)
[![MVP submission: 2026-04-28](https://img.shields.io/badge/MVP%20submission-2026--04--28-success)](#mvp-submission)

> **AgentForge — Clinical Co-Pilot**, Gauntlet AI Austin admission track.
> A multi-turn AI agent that helps a hospitalist physician catch up on inpatients fast — reading the chart from a forked [OpenEMR](https://github.com/openemr/openemr) via FHIR R4, summarizing what matters, and citing every clinical claim back to a specific record. Read-only over real EHR data, structurally verified, designed against the "hospital CTO bar".

---

## ★ MVP submission — 2026-04-28

Everything a reviewer needs is in this section. Each link below is at the **top of the repo**.

### Deployed app

| | URL | What it is |
|---|---|---|
| **OpenEMR fork** (system of record) | https://southern-chester-hint-teenage.trycloudflare.com/ | Our forked OpenEMR. Login: `admin` / `pass` (default — flagged in [AUDIT.md §1.2](0-mvp/AUDIT.md#12-default-credentials-and-secrets)). |
| **Clinical Co-Pilot** (the AI agent) | https://comfort-reach-soviet-freight.trycloudflare.com/ | The agent UI. Sign in with the same `admin` / `pass`. Type *"Catch me up on Cohen."* — the demo patient is seeded. |

### Required documents

All four required documents live together under [`0-mvp/`](0-mvp/) at the top of the repo:

| Doc | What it is |
|---|---|
| [0-mvp/USERS.md](0-mvp/USERS.md) | **Stage 4** — target user (hospitalist), workflow, three use cases with explicit "why an agent" defense |
| [0-mvp/AUDIT.md](0-mvp/AUDIT.md) | **Stage 3** — five-section audit of OpenEMR with a 500-word summary leading with the highest-impact findings |
| [0-mvp/ARCHITECTURE.md](0-mvp/ARCHITECTURE.md) / [0-mvp/ARCHITECTURE.pdf](0-mvp/ARCHITECTURE.pdf) | **Stage 5** — agent integration plan, 500-word summary, implementation-status table, layer walkthrough, latency + cost models |
| [0-mvp/presearch.md](0-mvp/presearch.md) | Phase 1–3 pre-search constraints + decision log |

### Caveat for reviewers

The MVP gate calls for "publicly accessible deployment." Both URLs above are live and reachable while the cloudflared tunnel processes are up. The agent streams responses via SSE — token-by-token output begins in ~2s with progress indicators ("Searching for patient…", "Loading chart…") in between. Our Fly.io deploy of OpenEMR is in flight and detailed in [`deploy/fly/`](deploy/fly/) — it hit a known issue with the upstream `openemr/openemr:latest` image's first-boot install path on a fresh Fly volume, deferred to Thursday's early submission per the brief's "deploy final agent to the same infrastructure" requirement.

---

## What's in this repo

```
openemr/                        # the OpenEMR fork (PHP/Apache, MariaDB)
├── README.md                   # ← you are here
├── README.openemr-upstream.md  # the original upstream OpenEMR README
├── 0-mvp/                      # ★ MVP submission — all required Stage 3-5 docs
│   ├── USERS.md                # ★ Stage 4 — target user, workflow, use cases
│   ├── AUDIT.md                # ★ Stage 3 — security/perf/arch/data-quality/compliance audit
│   ├── ARCHITECTURE.md         # ★ Stage 5 — agent integration plan + 500-word summary
│   ├── ARCHITECTURE.pdf        # rendered architecture doc
│   └── presearch.md            # Phase 1-3 pre-search per the brief
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

A focused tool, not always-on. Three intended use cases (see [0-mvp/USERS.md](0-mvp/USERS.md)):

- **A — Pre-round patient summary** ("Catch me up on Cohen") — *implemented for MVP*
- **B — Medication safety check** ("Is it safe to start Bactrim on Cohen?") — *Thursday work*
- **C — Sign-out drafting** — *Sunday-final work*

Verification is **structural, not best-effort**: the LLM has no path to FHIR, every tool returns `{data, sources: [...]}`, and a deterministic citation validator rejects responses that cite resource IDs not in the cumulative tool-output set. The system prompt also forbids the LLM from emitting clinical reasoning (drug interactions, dose-reduction rules) that didn't come from a tool — exactly the "confident hallucination → patient harm" failure mode the brief calls out.

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

## Roadmap

| Sprint gate | Date | What's in |
|---|---|---|
| **MVP (this submission)** | 2026-04-28 | Forked + deployed OpenEMR, Stage 3-5 docs, working agent against Cohen, cookie-session login, SSE streaming, dashboard TTL cache + startup prewarm, citation-click navigation, demo video |
| **Early submission** | 2026-04-30 | Deployed agent on Fly.io same-infra, eval framework (~140 cases), LangSmith observability, audit-log Postgres, prompt caching, role mapping (physician/nurse/resident), Clinical Notes → encounter SOAP note round trip |
| **Final** | 2026-05-03 | `clinical_rules` tool, Use Case C, cost analysis (100/1K/10K/100K), social post, production-readiness gaps closed |

---

## Acknowledgements

This is a fork of [OpenEMR](https://github.com/openemr/openemr) — a 20-year-old open-source EHR with a real codebase and a real user community. The original upstream README is preserved at [README.openemr-upstream.md](README.openemr-upstream.md). All credit for OpenEMR itself goes to the OpenEMR project and its contributors. Net-new code in this fork is in [`clinical-copilot/`](clinical-copilot/).

Built with Claude Code as part of the Gauntlet AI Austin admission track.
