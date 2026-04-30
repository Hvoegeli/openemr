# Long-Term Fixes

Recommended follow-ups deferred during the MVP / Thursday / Sunday push. Each
entry lists **what**, **why**, **where to start**, **estimated effort**, and
**status as of last review**. Revisit after tonight's submission to triage what
still applies and what can be retired.

_Last updated: 2026-04-30, branch `feat/clinical-notes` at `11eb0276b`._

---

## Architecture / Platform

### 1. Persist observability traces to a database (SQLite first, Postgres later)
- **Why:** `TraceStore` is a 200-entry in-memory `deque`. On `systemctl restart copilot` (or any deploy) every trace is lost. Compliance, debugging "yesterday at 2pm what happened?", aggregation queries, multi-worker scaling all break.
- **Where:** [app/observability.py:149-167](clinical-copilot/app/observability.py#L149-L167). Replace `TraceStore` internals; the dataclass interface stays the same. Schema sketch:
  ```sql
  CREATE TABLE request_traces (
    request_id UUID PRIMARY KEY, session_id TEXT, username TEXT,
    user_msg TEXT, model TEXT, started_at TIMESTAMPTZ,
    duration_ms REAL, total_input_tokens INT, total_output_tokens INT,
    cost_usd NUMERIC(10,6), validator_failed BOOLEAN, error TEXT,
    tool_events JSONB, llm_events JSONB
  );
  CREATE INDEX ON request_traces (started_at DESC);
  ```
- **Effort:** ~30 min for SQLite (zero ops), ~2 hours for Postgres + migrations + connection pool.
- **Flagged in:** ARCHITECTURE.md §observability gap; user discussion 2026-04-30.

### 2. Dedicated `get_vital_trends` tool for the agent
- **Why:** Currently the agent only has `get_patient_card`. To reason about trends it has to receive a flat list of FHIR Observations, mentally group them by vital, sort by time, filter junk panels, and parse BP composites. Pre-bucketed structured data would cut tokens ~5×, lower error surface, and reuse the `app/vitals.py` parser already used by the UI.
- **Where:** Add tool wrapping the existing `_vital_trends_compute` ([app/main.py](clinical-copilot/app/main.py)) — return `{trends: {bp_systolic: [{date, value}, ...], ...}, current: ...}`. Register in `app/agent/tools.py`, expose in `app/agent/system_prompt.py`.
- **Effort:** ~30 min.
- **Flagged in:** user discussion 2026-04-30 after Hale heart-rate trend question. Quick fix shipped (`_count: 10 → 50`); architectural fix deferred.

### 3. Time-windowed FHIR tools for Use Case A
- **Why:** USERS.md Use Case A is "Catch me up on Cohen since yesterday" — answering this requires a **delta**, not a full snapshot. Today the agent can dump the chart but can't say "what changed in 24h."
- **Where:** New tools `get_recent_observations(patient_id, hours)`, `get_recent_notes(patient_id, hours)`, `get_med_changes(patient_id, hours)`. Each filters by `effectiveDateTime` / `authoredOn` / encounter date.
- **Effort:** ~2 hours (3 tools × similar shape, plus prompt update).
- **Flagged in:** USERS.md §Use Case A "Status: Thursday work."

### 4. `clinical_rules` tool for Use Case B
- **Why:** USERS.md Use Case B is "Is it safe to start Bactrim on Cohen?" — needs deterministic allergy / renal-dose / drug-interaction lookups. Until this exists, R2 in the system prompt forbids the LLM from inventing rules from training, so the agent has to refuse the question.
- **Where:** New tool source could be a small curated JSON (sulfa class → drugs to avoid; renal-dose thresholds for top inpatient meds; common interaction pairs) for the demo, replaceable with FDB / RxNorm-DDI in production.
- **Effort:** ~4 hours — most of it is curating the JSON.
- **Flagged in:** USERS.md §Use Case B; presearch.md.

### 5. Use Case C (sign-out drafting)
- **Why:** USERS.md §Use Case C — "Draft sign-outs for my list for the night team." Sunday-final work in the brief.
- **Where:** Each per-patient blurb is a constrained Use Case A — once the time-window tools (#3) exist, this is mostly prompting.
- **Effort:** ~3 hours after #3 is done.
- **Flagged in:** USERS.md.

---

## Performance

### 6. Anthropic prompt caching
- **Why:** Cache cuts input-token cost 10×. Sonnet 4.6 has an empirical ~2048-token minimum to qualify. The system prompt was ~952 tokens (below threshold) at MVP; with R5 added today it's now closer to 1500 — re-measure to see if it crosses the threshold or if we should pad strategically.
- **Where:** `app/agent/graph.py` — wrap the `SystemMessage` content in a `cache_control` block per the langchain-anthropic API. ARCHITECTURE.md §8.2 has the implementation note.
- **Effort:** ~1 hour including measurement + verification with a real LLM call.
- **Flagged in:** ARCHITECTURE.md §8.2 (Sunday-track).

### 7. Deterministic-first router
- **Why:** Every chat turn currently routes through the LLM. Many turns ("show me Cohen's allergies") could be templated and answered from a tool result without a model call. Lower cost, lower latency, easier to eval.
- **Where:** New `app/agent/router.py` that classifies the turn (intent + entities) and chooses LLM vs template. ARCHITECTURE.md §8.1 has the design sketch.
- **Effort:** ~6 hours.
- **Flagged in:** ARCHITECTURE.md §8.1 (Sunday-track).

---

## Security

### 8. Defense-in-depth for prompt injection (beyond the prompt-level scope rule)
- **Why:** R5 in `system_prompt.py` is prompt-level only. A determined attacker may find phrasing the LLM doesn't refuse. Defense-in-depth would add (a) a regex pre-check for known patterns ("ignore previous instructions", "DAN", etc.) that returns the refusal template without an LLM call, saving cost, and (b) an output classifier that scores responses for off-topic content.
- **Where:** Pre-check in `main.py` `chat()` / `chat_stream()` before `app.state.graph.ainvoke`. Output classifier as a post-validator node in the LangGraph.
- **Effort:** ~2 hours for the pre-check, ~half a day for a classifier.
- **Flagged in:** scope-lock commit message (`c0eb9ca9a`); 2026-04-30.

### 9. Real authentication / Postgres sessions
- **Why:** Current sessions live in an in-memory dict in `main.py`. Restart loses sessions. No SSO/SAML/OIDC for hospital deployment.
- **Where:** `SESSIONS` dict (currently `dict[str, AgentState]`) → DB-backed; auth → OpenEMR session passthrough or external IdP.
- **Effort:** ~1 day.
- **Flagged in:** AUDIT.md production-readiness gaps.

### 10. BAA-routed LLM (AWS Bedrock vs Anthropic direct)
- **Why:** Production hospital deploy needs a Business Associate Agreement. Bedrock has BAA support; Anthropic direct API does not (today).
- **Where:** Swap `langchain_anthropic.ChatAnthropic` for `langchain_aws.ChatBedrock`. Same interface.
- **Effort:** ~2 hours including IAM + region config.
- **Flagged in:** ARCHITECTURE.md §8 production-readiness.

---

## DevEx / Ops

### 11. Stable public URLs for the deployed instance
- **Why:** Currently using cloudflared quick-tunnels (`*.trycloudflare.com`). URLs rotate when cloudflared restarts. Reviewers may follow stale links.
- **Where:** Either point a real subdomain at the Hetzner box (15-min DNS change), or unblock the Fly.io deploy that hit a first-boot bug.
- **Effort:** ~30 min for DNS; ~half a day for the Fly.io path.
- **Flagged in:** AUDIT triage 2026-04-30.

### 12. Record + link the demo video
- **Why:** README references it; no link in the repo. Brief asks for a 3–5 min walkthrough.
- **Where:** Host on Loom / unlisted YouTube / S3, link in README §MVP.
- **Effort:** ~30 min recording + edit.
- **Flagged in:** AUDIT triage 2026-04-30.

### 13. MCP server wrapping the FHIR + clinical-notes layer
- **Why:** Lets other AI clients (Claude Desktop, Cursor, Jupyter via MCP) reuse the same auth/audit boundary the web app has. Platform play, not feature play.
- **Where:** New `clinical-copilot/mcp/` exposing `lookup_patient`, `get_chart`, `add_clinical_note`, `write_vitals` over the MCP transport. Wraps existing endpoints.
- **Effort:** ~1 day.
- **Flagged in:** user discussion 2026-04-30.

---

## Code-quality / Polish

### 14. Native BP composite handling in `_format_vital`
- **Why:** `_format_vital` flattens FHIR Observations by reading `valueQuantity` directly, which drops BP because OpenEMR returns it as a composite (panel with components). The patient-card endpoint backfills BP from `trends.bp_*` ([main.py:_decorate_card_vitals](clinical-copilot/app/main.py#L477)) which works but duplicates parsing logic.
- **Where:** Modify `app/fhir/adapter.py:_format_vital` to detect composites and emit synthetic systolic/diastolic rows; remove the backfill from `_decorate_card_vitals`.
- **Effort:** ~30 min.
- **Flagged in:** vital-trends commit message (`db3f5b30c`); 2026-04-30.

### 15. Move inline `<script>` to end of `<body>` (or load with `defer`)
- **Why:** The chat UI's inline script runs at parse time before the modal HTML below it has been parsed. Today we work around it by lazy-resolving modal refs ([commit `bd4ba2f39`](clinical-copilot/app/web/index.html)). Moving the script unifies init and removes the workaround.
- **Where:** `app/web/index.html` — relocate the `<script>` tag to just before `</body>`, drop the lazy `ensureCnModalWired` indirection.
- **Effort:** ~15 min including a regression sweep.
- **Flagged in:** modal-fix commit message (`bd4ba2f39`); 2026-04-30.

### 16. Eval coverage for Use Case B
- **Why:** Once `clinical_rules` (#4) lands, USERS.md sets the bar at recall ≥95% on adjudicated unsafe combos and precision ≥90% on flagged unsafe. The eval suite has no labeled cases for B today.
- **Where:** `clinical-copilot/evals/cases/labeled.yaml` — add safe / unsafe / "insufficient evidence" combos, then re-record snapshots.
- **Effort:** ~3 hours of curation per ~30 cases.
- **Flagged in:** USERS.md eval gates table.

---

## How to use this file

When you come back to triage:
1. For each entry, mark it **APPLIES**, **DROP**, or **IN PROGRESS** based on what's already been done since last review.
2. Move dropped items to a `## Retired` section at the bottom (with a one-line reason) so the audit trail is preserved.
3. For applicable items, slot them into a build-order based on dependency (e.g., #3 unblocks #5).
