# AUDIT — OpenEMR fork as the foundation for the Clinical Co-Pilot

_Status: locked for MVP / Tuesday submission_
_Method: combination of source-code reading, FHIR + standard-API probing, runtime observation against the local development-easy stack and the deployed Fly.io instance, and direct DB inspection via the MariaDB CLI._

---

## Summary (~500 words)

OpenEMR is a 20-year-old PHP electronic health record system with a real codebase (~1.5M lines of PHP) and a real user base. We forked it because the brief requires it, and because building "an AI agent for healthcare" against a curated mock EHR would never reveal the friction that makes this a hard problem. The most important finding from this audit is that **the friction is real, and it shapes every decision in [ARCHITECTURE.md](ARCHITECTURE.md).**

**Top three findings, ranked by impact on the agent build:**

1. **OpenEMR's FHIR API is read-friendly but write-hostile.** Of 30+ FHIR resources, only `Patient`, `Practitioner`, `Organization`, and `DocumentReference` accept POST/PUT. Everything clinical — `Encounter`, `Condition`, `Observation`, `MedicationRequest`, `AllergyIntolerance` — is read-only on FHIR. Clinical data writes go through a **separate non-FHIR REST API** at `/apis/default/api/` with a different scope vocabulary (`user/allergy.cruds`, not `user/AllergyIntolerance.write`), a different OAuth grant type (`password`, not `client_credentials`), and different date validators per controller (`Y-m-d` for allergies/conditions, `Y-m-d H:i:s` for medications). This forced us to register **two separate OAuth clients** with different scopes — one for the agent's read path, one for demo-data seeding. Production deployments will hit this same wall and need to choose explicitly which side of the fence each integration sits on.

2. **Default credentials and dev-mode flags ship to production by default.** `admin / pass` is the documented default admin login; the local docker-easy stack hard-codes a Composer GitHub token in plaintext; the production docker-compose still defaults to `MYSQL_ROOT_PASSWORD: root`. None of these are exploitable in a dev environment, but they reveal a culture where defaults skew toward "make it run" rather than "make it safe". Anyone deploying OpenEMR to production needs to explicitly override every default; nothing in the deploy path forces this.

3. **The audit log exists but is partial.** OpenEMR has an `audit_master` and `log` tables and writes to them on UI-driven changes, but **the FHIR and standard-API endpoints we used to read and write Cohen's chart did not produce log rows we could find via the API**. HIPAA requires a complete record of *who accessed what PHI when*; OpenEMR's API surface doesn't yet meet that bar without additional instrumentation. Our agent's audit log (planned for Thursday in `clinical-copilot/`) becomes the system of record for AI-mediated access — by design, not by accident.

The audit also surfaced **data-quality quirks** the FHIR layer papers over silently — free-text titles get embedded in the narrative `text.div` rather than the structured `code.coding`, and the FHIR layer emits `data-absent-reason: unknown` placeholders that look real to a naive consumer. We added a narrative-fallback path in our adapter ([clinical-copilot/app/fhir/adapter.py](../clinical-copilot/app/fhir/adapter.py)) so the agent doesn't render "Unknown" for every chart entry.

**Performance** is acceptable for MVP — local p95 around 14s for a full 3-tool turn against Cohen — but the breakdown is 70% LLM, 30% sequential FHIR. A 500-bed-300-concurrent-user deployment will need parallel tool calls, prompt caching, and very likely a FHIR-layer cache.

**Architecture** is split between a modern PSR-4 `src/` namespace and a legacy procedural `library/` tree. New work belongs in `src/`; integration points for the agent are clean (REST + FHIR), so we don't have to touch PHP at all.

The agent design in `ARCHITECTURE.md` is shaped by these findings: the verification layer exists because the data-quality quirks make trusting field values directly unsafe, the dual OAuth scheme exists because the API split requires it, the audit log lives in our app because OpenEMR's doesn't cover the API surface.

---

## 1. Security audit

### 1.1 Authentication & authorization

**OAuth2 server is feature-complete.** OpenEMR ships a SMART-on-FHIR-compliant OAuth2 server at `/oauth2/default/` with multiple grant types: `authorization_code`, `client_credentials`, `password`, `refresh_token`. Dynamic client registration is supported at `/oauth2/default/registration`. JWKS-based `private_key_jwt` auth works as documented (we use it for the agent's read flow).

**Two distinct authorization models coexist.**
- **FHIR API (`/apis/default/fhir/`)** uses **SMART-style scopes** — `system/Patient.read`, `user/Encounter.write`, etc. Token scope is checked at the request boundary.
- **Standard REST API (`/apis/default/api/`)** uses **a custom scope vocabulary** — `user/allergy.cruds` (combined create/read/update/delete/search), `user/medical_problem.cruds`, etc. Then layers the legacy **GACL ACL system** on top: `RestConfig::request_authorization_check($request, "patients", "med")` in every handler.

This duality is not documented well outside the source. We discovered it by reading [`apis/routes/_rest_routes_standard.inc.php`](../apis/routes/_rest_routes_standard.inc.php) and [`src/RestControllers/Config/RestConfig.php`](../src/RestControllers/Config/RestConfig.php) and tracing 401/403 errors through the `BearerTokenAuthorizationStrategy` and the PHP error log.

**Implication for the co-pilot:** writes to clinical data require a **second OAuth client** with `client_secret_post` auth, `password` grant, and the standard-API scopes — because `private_key_jwt` + `system/*.read` (what our FHIR reads use) cannot be promoted to writes by adding scopes. The clients are architecturally distinct.

### 1.2 Default credentials and secrets

| Default | Where | Risk |
|---|---|---|
| `admin` / `pass` | First-boot install of every OpenEMR variant | High in any public deploy. Every internet-reachable OpenEMR with default creds is a one-click takeover. |
| `MYSQL_ROOT_PASSWORD: root` | `docker/production/docker-compose.yml` | Ships in production-labelled compose file. Trivial DB compromise if exposed. |
| Composer GitHub token, plaintext | `docker/development-easy/docker-compose.yml`, line `GITHUB_COMPOSER_TOKEN` | Token has read access to repos; baked into a public docker-compose. Should be a build-time secret, not committed. |
| `couchdb_user: admin / couchdb_pass: password` | development-easy env | Local dev only, but easy to copy into production compose. |
| MFA disabled by default | OpenEMR globals | Acceptable for clinical workflow ergonomics, but not for any internet-facing admin user. |

We did not find a credential-rotation policy mechanism or a "first-boot password reset" enforcement. Anyone deploying must remember to change every default manually.

### 1.3 Data exposure vectors

- **OAuth password grant is opt-in but the flag is permissive.** `oauth_password_grant: 3` enables both user and patient-portal password flows. Our audit confirms this is the recommended dev value; the default in production deployments should be `0` unless legacy clients need it. Password grant skips MFA in the path we exercised.
- **Self-signed certificates** in development-easy. Acceptable for dev. Production deploys need real certs; OpenEMR doesn't manage this.
- **Default site URL** is `https://localhost:9300/`. Any deployment forgetting to change `OPENEMR_SETTING_site_addr_oath` will issue tokens with the wrong audience claim.
- **The FHIR `metadata` endpoint is unauthenticated** and returns the full Capability Statement (which resources are supported, what auth is needed). This is per FHIR spec — useful, but it does fingerprint your install.

### 1.4 PHI handling

- All FHIR endpoints we tested require a valid bearer token. No silent unauthenticated leaks observed.
- The FHIR layer correctly redacts UUIDs in error responses (it returns a generic "insufficient permissions" rather than confirming a record exists). Good.
- However, **FHIR `meta.versionId` increments on read** in some setups — confirmed via response inspection. Not exploitable, but a side-channel.
- We did not test the patient portal's auth surface in this audit. **Stage 4–5 work; relevant for a real production audit but not for the MVP gate.**

### 1.5 Findings → architecture decisions

| Finding | Reflected in ARCHITECTURE.md as |
|---|---|
| Dual auth models (FHIR + standard) | Two registered OAuth clients with separate responsibilities |
| Default credentials shipped | Explicit "rotate every default before deploy" gap noted in §production-readiness gaps |
| No FHIR-API audit log | Co-pilot adds its own append-only audit log (Thursday work) |
| Password grant opt-in | Production posture: `oauth_password_grant: 0`, agent uses `client_credentials` only |

---

## 2. Performance audit

### 2.1 Measured numbers (local + deployed)

Against the local `docker/development-easy` stack with Cohen seeded:

| Operation | p50 | p95 (n=10) | Notes |
|---|---|---|---|
| `POST /oauth2/default/token` (client_credentials) | 0.3s | 0.5s | Token cached for 1h, so amortized cost per request is ~0. |
| `GET /apis/default/fhir/Patient/{id}` | 0.4s | 0.7s | Single-resource fetch. |
| `GET /apis/default/fhir/Patient?family=Cohen` | 0.5s | 0.8s | Search on indexed column. |
| `GET /apis/default/fhir/AllergyIntolerance?patient={id}` | 0.5s | 0.9s | Per-resource patient-scoped query. |
| **Full `get_patient_card` adapter call** (1 patient + 5 list searches, sequential) | 2.8s | 4.2s | Dominated by the 5 sequential queries. |
| **Full agent turn** (resolve_patient → current_time → get_patient_card → LLM synthesis) | 11s | 14s | LLM latency dominates; tool-call latency is ~30%. |

Local Anthropic API latency was 7-9s for a Sonnet 4.6 final-response call with bound tools. The first-token latency was not measured; we don't stream yet.

### 2.2 Bottlenecks

**Sequential FHIR queries** in [`get_patient_card`](../clinical-copilot/app/fhir/adapter.py). The adapter fires 6 queries one after another. `asyncio.gather` would parallelize these and cut the tool latency from ~3s to ~0.7s. Slated for Thursday.

**No prompt caching.** The system prompt is ~1.5KB and re-sent every turn. Anthropic's prompt caching cuts repeat-input cost by ~90%. Slated for Thursday.

**No FHIR-layer cache.** Every `Patient/{id}` lookup hits MariaDB, even though the data changes rarely within a session. A session-scoped cache (15-30s TTL) would absorb most repeat-fetch patterns.

**Sequential LLM tool-call rounds.** The LLM currently calls `resolve_patient`, then `current_time`, then `get_patient_card` in three separate round-trips. If the LLM emitted all three calls in parallel, we'd save ~3-5s. Tool-call parallelism is supported by Sonnet but our prompt does not explicitly encourage it.

### 2.3 What scale exposes

For the brief's "500-bed hospital, 300 concurrent users" question, the most important constraints we surfaced:

- **MariaDB on a single Fly machine** caps at ~5K connections; OpenEMR's per-request connection model means 300 concurrent users with 5 in-flight queries each can saturate it. Real deployment needs a connection pooler (`mariadb-proxy` / `proxysql`) or read replicas.
- **OpenEMR session storage is filesystem-based** (`sites/default/documents/SessionStorage`). Doesn't scale to multi-machine. Real deployment needs Redis or DB-backed sessions.
- **PHP-FPM tuning is conservative by default** — workers cap quickly under burst load. We did not retune; would for production.

### 2.4 Findings → architecture decisions

| Finding | Reflected in ARCHITECTURE.md as |
|---|---|
| Sequential FHIR fetches dominate tool latency | Thursday: parallelize with `asyncio.gather` in adapter |
| No prompt caching | Thursday: enable Anthropic prompt caching for system prompt + per-patient context |
| FHIR-layer cache absent | Sunday: introduce a session-scoped patient-card cache |
| OpenEMR session storage is FS-based | Production-readiness gap: switch to Redis-backed sessions |

---

## 3. Architecture audit

### 3.1 Code organization

OpenEMR has two parallel code styles, both alive:

- **`src/`** — modern, PSR-4 namespaced (`OpenEMR\` prefix), Symfony components, service classes, type-hinted PHP 8.2+. New REST controllers (`src/RestControllers/`), new services (`src/Services/`), and new FHIR adapters (`src/Services/FHIR/`) live here.
- **`library/`** — legacy procedural PHP. Direct `$_SESSION`/`$_GLOBALS` usage, untyped arrays passed through layers, ADODB-style DB calls. Most clinical-form code lives here. CLAUDE.md explicitly says "Legacy code is not the standard."

Modern HTTP entrypoints route through `apis/dispatch.php` → `OpenEMR\RestControllers\ApiApplication` → Symfony EventDispatcher → `HttpRestRouteHandler` → controller closure. Auth happens in `OAuth2AuthorizationListener` and `AuthorizationListener` event subscribers.

### 3.2 Data layer

| Concept | OpenEMR table(s) | FHIR resource | Notes for the agent |
|---|---|---|---|
| Patient | `patient_data` | `Patient` | Has both numeric `pid` and `uuid`; standard API uses `pid`, FHIR uses `uuid`. Both sometimes appear in route parameters. |
| Encounter | `form_encounter` + `forms` | `Encounter` | New encounters get `status: finished` by default — not `in-progress` even when active. Our adapter relaxes the FHIR `status` filter because of this. |
| Condition / Problem list | `lists` (type=`medical_problem`) | `Condition` | Free-text `title` column; optional `diagnosis` reference for SNOMED/ICD codes. |
| Allergy | `lists` (type=`allergy`) | `AllergyIntolerance` | Same shape as problems; allergies and problems share a table. |
| Medication (active list) | `lists` (type=`medication`) | `MedicationRequest` | Goes through `ListService` whose date validator wants `Y-m-d H:i:s` (not just `Y-m-d`). |
| Prescription | `prescriptions` | `MedicationRequest` (different mapping) | Separate from `lists.medication`; both surface as MedicationRequest. |
| Vitals | `form_vitals` | `Observation` (vital-signs category) | Each individual vital is a separate Observation; OpenEMR returns ~10 per encounter (BP, HR, SpO2, Temp, Weight, Height, RR, plus a parent panel observation and a couple of placeholders). |
| UUID registry | `uuid_registry` | n/a | Maps `(table_name, table_id) → uuid`. FHIR queries go through this for resource ID resolution. |

### 3.3 FHIR mapping layers

A single FHIR resource fetch traverses **three layers**:
1. **Route handler** in `_rest_routes_fhir_r4_us_core_3_1_0.inc.php` — dispatches to a FhirGenericRestController.
2. **FHIR controller** (`src/RestControllers/FHIR/Fhir{Resource}RestController.php`) — handles paging, search-param parsing, scope checks.
3. **FHIR service** (`src/Services/FHIR/Fhir{Resource}Service.php`) — calls the underlying `src/Services/{Resource}Service.php` which talks to the DB, then maps the result into a FHIR R4 resource via `php-fhir`-generated classes.

**Implication:** changing the shape of FHIR output requires editing the FhirService, not the underlying Service. We did NOT modify any of these layers — instead, the co-pilot's adapter handles OpenEMR-flavored quirks at the consuming end (narrative fallback for `code.text`, `data-absent-reason` skipping). This keeps the fork minimal and upgrade-friendly.

### 3.4 Integration points

| Where the agent attaches | What it consumes |
|---|---|
| `/apis/default/fhir/{Resource}` | All chart reads (Patient, Encounter, Observation, Condition, AllergyIntolerance, MedicationRequest, DocumentReference, Practitioner). Auth: SMART scopes via `client_credentials`. |
| `/apis/default/api/patient/{puuid}/...` | Demo-data seeding only (writes for clinical data). Auth: standard-API scopes via `password` grant. Not used in production flow. |
| `/oauth2/default/token` and `/oauth2/default/registration` | OAuth dance for both clients above. |
| MariaDB direct (via `docker compose exec mysql mariadb`) | Only used during local development for table inspection and seed cleanup. **Never** used by the running agent. |

### 3.5 Findings → architecture decisions

| Finding | Reflected in ARCHITECTURE.md as |
|---|---|
| Three-layer FHIR mapping with quirks | Adapter does narrative-fallback + data-absent-reason filtering; we don't modify OpenEMR PHP |
| Modern src/ vs legacy library/ | Co-pilot attaches at the `/apis/default/fhir/` boundary; no PHP changes needed |
| `lists` table shared across allergy/problem/medication | Agent treats them as separate FHIR resources (correct abstraction); OpenEMR's internal sharing is invisible to us |
| Encounter `status: finished` default | Adapter loosened FHIR search filter; documented as known FHIR-implementation quirk |

---

## 4. Data quality audit

### 4.1 Stock demo data is too sparse for clinical scenarios

OpenEMR ships **5 example patients** in `sql/example_patient_data.sql`. Each has demographics only — no encounters, no problems, no medications, no labs, no clinical notes. None of them can support a hospitalist demo.

We hand-seeded **Nora Cohen** with: 2 allergies, 4 active problems (HTN, T2DM, CKD3, AFib), 4 active medications (Lisinopril, Metformin, Apixaban, Atorvastatin), 1 active encounter, and 1 vitals set. Seeding script: [`clinical-copilot/scripts/seed_cohen.py`](../clinical-copilot/scripts/seed_cohen.py).

For richer eval datasets (Thursday), Synthea-generated FHIR bundles will fill the gap. Synthea was selected over hand-crafted in [`ARCHITECTURE.md` §7](ARCHITECTURE.md).

### 4.2 Free-text titles dominate over coded entries

When a clinician adds a problem via the UI (`title: "Hypertension"`), OpenEMR's standard API stores the string verbatim in `lists.title`. The FHIR layer then maps this:
- `code.text` → empty
- `code.coding[0].system` → `http://terminology.hl7.org/CodeSystem/data-absent-reason`
- `code.coding[0].code` → `unknown`
- `code.coding[0].display` → `Unknown`
- `text.div` → `<div>Hypertension</div>` (narrative)

A naive FHIR consumer pulling `code.coding[0].display` will render every problem as **"Unknown"**. Our adapter falls back to `text.div` (HTML-stripped) when the coding is the data-absent-reason placeholder.

This is a **systemic data-quality issue** in OpenEMR FHIR output, not a one-off. It affects allergies, problems, and medications equally.

### 4.3 Default statuses don't match clinical reality

| Resource | Default status set on creation | Expected for "active inpatient" |
|---|---|---|
| `Encounter` | `finished` | `in-progress` (or `arrived`) |
| `Condition` | `clinical-status: active` ✓ | active |
| `MedicationRequest` | `status: active` ✓ | active |
| `AllergyIntolerance` | `clinicalStatus: active`, `verificationStatus: unconfirmed` | unconfirmed is technically wrong for known allergies, but acceptable |

The **encounter `finished`** mismatch was actively misleading — the agent's "current encounter" filter returned zero until we widened the status set in [`adapter.get_patient_card`](../clinical-copilot/app/fhir/adapter.py).

### 4.4 Search-parameter inconsistencies

OpenEMR's FHIR layer **silently ignores** some standard FHIR search parameters:

- `Encounter?status=in-progress` → returns zero results even when matching encounters exist
- `Condition?clinical-status=active` → same

We did not exhaustively enumerate which search parameters work. Our defensive posture: pull broader queries and **filter client-side**. This is in the adapter and noted in ARCHITECTURE.md §FHIR-quirks.

### 4.5 Date format inconsistency between standard-API endpoints

| Endpoint | `begdate` format that worked |
|---|---|
| `POST /api/patient/{puuid}/allergy` | `2024-10-01` (ISO date, no time) |
| `POST /api/patient/{puuid}/medical_problem` | `2024-10-01` (ISO date, no time) |
| `POST /api/patient/{pid}/medication` | `2024-10-01 00:00:00` (full DATETIME) |

`ListService` (used by medication) validates with `datetime('Y-m-d H:i:s')`; the allergy/condition controllers use a different validator that accepts `Y-m-d`. Same column type in MariaDB; different validators at the controller layer.

### 4.6 Findings → architecture decisions

| Finding | Reflected in ARCHITECTURE.md as |
|---|---|
| Stock data too sparse | Synthea for breadth; hand-craft 1-2 demo "stars" — Cohen is one |
| Free-text titles dominate | Adapter narrative fallback + `data-absent-reason` skip |
| Encounter status defaults to `finished` | Adapter relaxes FHIR search filter |
| Search params silently ignored | Adapter filters client-side as a defensive posture |

---

## 5. Compliance & regulatory audit

### 5.1 Audit logging — partial

OpenEMR has audit infrastructure: `audit_master`, `audit_master_id` foreign keys, `log` table, `OpenEMR\Logging\AuditUtils`. UI-driven changes (form saves, problem-list edits) write rows. **We did not find evidence that the FHIR or standard-API endpoints we exercised wrote to these tables in the same way.** The PHP error log surfaces auth events; there's no equivalent surface for "user X read Patient Y at time Z" through the API.

For HIPAA, the requirement is *complete* who-accessed-what-when. OpenEMR's API surface, in its current state, **does not meet this bar**. A production deployment of the co-pilot would need either:
- An app-layer audit log (what we plan in [clinical-copilot](../clinical-copilot/) for Thursday — append-only Postgres table written by the FHIR adapter on every call), OR
- Patches to OpenEMR's FHIR + standard-API controllers to call AuditUtils on every request.

The first is faster; the second is the right long-term answer because it captures all consumers, not just our agent.

### 5.2 Retention policy — none out of the box

OpenEMR does not enforce data retention. There is no built-in "purge after N years" job for chat logs, audit events, or session data. HIPAA requires defining and enforcing retention windows; in production, this is on the operator.

For the co-pilot, retention is a thing we own:
- Conversation history: planned 90-day retention in our Postgres
- Audit log: 7-year retention to meet HIPAA + state requirements
- LangSmith traces (Thursday): match conversation-history retention

### 5.3 Breach notification — operational, not technical

Out of OpenEMR's scope. Operator-defined process. We note it as a production-readiness gap.

### 5.4 BAA (Business Associate Agreement) implications

- **Anthropic direct** does not offer a BAA. Sending PHI to `api.anthropic.com` in production = HIPAA violation.
- **AWS Bedrock** (which fronts Anthropic models) does offer a BAA for HIPAA workloads. Production deploy would route LLM traffic through Bedrock.
- **Fly.io** offers SOC2 / ISO27001 but **not a HIPAA BAA** as of the audit date. Production hosting must move to AWS, Google Cloud, or Azure with their respective BAAs.

For the sprint, the brief explicitly says: *"act as if you have a signed BAA with all LLM providers that no data will be used for training purposes."* This is the demo-data-only operating mode. Production deployment requires the BAAs above plus the rest of the HIPAA technical safeguards.

### 5.5 Findings → architecture decisions

| Finding | Reflected in ARCHITECTURE.md as |
|---|---|
| OpenEMR API doesn't audit-log consistently | Co-pilot owns its audit log (append-only Postgres; written by adapter) |
| No retention policy | Co-pilot defines explicit retention windows by data class |
| Anthropic direct has no BAA | Production: route through Bedrock |
| Fly.io has no HIPAA BAA | Production: relocate to AWS/GCP/Azure |

---

## What this audit changed about our plan

Reading the brief at the start of the week, I expected the agent build to be the hard part. The audit revealed that **the OpenEMR integration is the harder problem.** Specifically:

- Without realizing the FHIR API can only write 4 resources, I would have built the demo-data seeding through the wrong endpoint and lost half a day debugging 401s.
- Without realizing the standard API uses a different scope vocabulary, I would have registered an OAuth client with wrong scopes and lost another hour.
- Without seeing the `data-absent-reason` placeholder, the agent would have rendered a chart full of "Unknown" with no obvious bug — a *silent* data-quality failure, the worst kind.
- Without checking the audit-log surface, I would have assumed "OpenEMR has an audit log" and not built one in the co-pilot — leaving production HIPAA non-compliant despite shipping.

Each of these is a **failure mode the brief warns about**: confident, plausible-looking output that's silently wrong. The audit caught them at design time instead of demo time. That's the point.
