"""FastAPI entry point for the clinical co-pilot.

Endpoints:
  GET  /                                  - chat UI (redirects to /login if anonymous)
  GET  /login                             - login page (uses OpenEMR creds)
  POST /api/login                         - validate creds via OpenEMR password grant; opens a server-side session
  POST /api/logout                        - revoke the current session and clear the cookie
  GET  /api/me                            - who am I? (UI uses this to render header)
  POST /chat                              - send a turn, get the assistant response (atomic)
  POST /chat/stream                       - same, but streams tokens + tool-progress as SSE
  GET  /api/patient/{id}/card             - structured patient-card JSON
  GET  /api/patient/{id}/documents        - past encounters + clinical documents
  GET  /api/calendar/today                - today's roster (Appointments, fallback)
  GET  /api/admin/sessions                - currently-valid logins (admin only)
  POST /api/admin/sessions/{sid}/revoke   - kick a session (admin only)
  GET  /api/admin/auth-events             - login/logout audit log (admin only)

Chat conversation state is held in-memory per `session_id` (a separate
concept from the auth session). Auth sessions are durable in SQLite —
see `app/auth_db.py`. End-of-shift charting (the previous "sign-out
drafting" tab) was removed: the Clinical Notes tab covers the
shift-handoff use case, and the doctor still charts in OpenEMR's own
sign-out workflow regardless.
"""

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel
from starlette.middleware.sessions import SessionMiddleware

load_dotenv()

# Mirror agent.* logs to stdout at INFO so tool calls + validator decisions are
# visible during the demo. Uvicorn's own loggers stay independent.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logging.getLogger("agent").setLevel(logging.INFO)

# Install the PHI redaction filter on the root logger so SSN / phone / DOB
# shapes never reach stdout, files, or the in-process trace store. The
# filter is a backstop — every audited call site in this app drops PHI
# fields explicitly (we log patient_uuid, not patient_name). See
# `app/safe_log.py` and W2_ARCHITECTURE.md §8.2.
from app.safe_log import install_phi_filter  # noqa: E402
install_phi_filter()

from app.agent.graph import MAX_VALIDATION_ATTEMPTS, active_model_label, build_graph, message_text  # noqa: E402
from app.agent.input_guard import JAILBREAK_REFUSAL, detect_jailbreak  # noqa: E402
from app.agent.intent_router import route_intent  # noqa: E402
from app.agent.state import AgentState  # noqa: E402
from app import access_control  # noqa: E402
from app.auth import current_session, current_user, is_admin, require_admin, verify_openemr_credentials  # noqa: E402
from app.assignments_db import AssignmentStore  # noqa: E402
from app.auth_db import AuthStore  # noqa: E402
from app.document_fingerprints_db import DocumentFingerprintStore  # noqa: E402
from app.extracted_sources_db import ExtractedSourcesStore  # noqa: E402
from app.extracted_practitioners_db import ExtractedPractitionersStore  # noqa: E402
from app.extracted_lab_results_db import ExtractedLabResultsStore  # noqa: E402
from app.hidden_docs_db import HiddenDocsStore  # noqa: E402
import hashlib  # noqa: E402

from app.cache import TTLCache  # noqa: E402
from app.config import settings  # noqa: E402
from app.fhir import adapter  # noqa: E402
from app.fhir.client import FhirClient  # noqa: E402
from app.fhir.extras import get_calendar_today, get_supporting_documents  # noqa: E402
from app.fhir.writer import OpenEMRWriter, OpenEMRWriteError  # noqa: E402
from app.extraction.extract import attach_and_extract, persist_extracted_facts  # noqa: E402
from app.extraction.fingerprint import (  # noqa: E402
    compute_fingerprint,
    namespace_text_fingerprint,
    pdf_text_fingerprint,
)
from app.extraction.render import render_to_png_pages  # noqa: E402
from app.extraction.schemas import DOC_TYPE_LABELS  # noqa: E402
from app.extraction.vision import ExtractionError, extract_via_claude  # noqa: E402
from app.observability import (  # noqa: E402
    TokenUsageCallback,
    TurnBudgetExceeded,
    init_langsmith,
    new_request_trace,
    reset_current_trace,
    set_current_trace,
)
from app.clinical_notes import (  # noqa: E402
    ClinicalNoteStore,
    NotFoundError as ClinicalNoteNotFound,
    now_utc,
)
from app.vitals import collect_vital_trends, latest_per_vital  # noqa: E402
from app.observability_db import SqliteTraceStore  # noqa: E402

log = logging.getLogger("agent.main")

WEB_DIR = Path(__file__).parent / "web"

# Conversation state, keyed by (username, session_id) — NOT session_id alone.
# `session_id` is echoed back to the client and re-supplied on the next turn,
# so keying on it alone lets any logged-in user adopt another clinician's
# conversation (and replay its PHI-laden ToolMessages) just by sending their
# session_id. Pairing it with the authenticated username isolates sessions
# per user: an unrecognized (username, session_id) simply starts fresh.
SESSIONS: dict[tuple[str, str], AgentState] = {}

# Cache TTL is generous — clinical chart data doesn't change second-to-second,
# and a stale 5-minute read is preferable to making the user wait 8s. Prewarm
# at startup makes the first read after server boot instant; this TTL keeps
# subsequent reads instant for the duration of a demo or a clinic session.
DATA_CACHE_TTL_S = 300.0

# Rendered PDF pages + bbox manifest are immutable for the lifetime of a
# DocumentReference (we never edit attachment bytes after upload). A long
# TTL keeps clicks on the citation-deep-link path instant after the first
# render. Server-side prefetch (kicked off by the docs-list endpoint) means
# the first click is also typically warm.
PAGES_CACHE_TTL_S = 1800.0

# Cap concurrent background page-renders so prefetch can't pin every
# uvicorn worker thread on a patient with many large PDFs.
_PAGES_PREFETCH_SEMAPHORE = asyncio.Semaphore(2)
# Cap docs prefetched per patient so one chart can't queue 50 renders.
_PAGES_PREFETCH_PER_PATIENT_CAP = 8


def _fresh_state() -> AgentState:
    return {
        "messages": [],
        "conversation_sources": [],
        "patient_id": None,
        "validation_attempts": 0,
        "username": None,
        "advisor_mode": False,
        "worker_route": None,
        "route_count": 0,
    }


def _mark_user_sessions_stale_after_upload(
    *,
    username: str,
    patient_uuid: str,
    doc_type: str,
    reference_id: str,
) -> int:
    """Append a SystemMessage to every session belonging to `username` so
    the agent treats prior tool results as stale and refetches on the next
    user turn.

    Why this is needed even though `app.state.cache` is invalidated on
    upload: the cache only protects *new* fetches. Prior `ToolMessage`s
    for the same patient (chart card, recent notes, document content)
    are baked into `SESSIONS[sid]["messages"]` from earlier turns, and
    the LLM happily reads from those instead of re-calling the tool —
    which is the user-visible "uploaded a doc but Copilot didn't see it
    until I logged out" symptom. Telling the agent in-context that the
    data has changed is the cheapest reliable nudge.

    Match by username (not patient_id) because a user can have multiple
    tabs / pinned patients, and a chart write can ripple into any of
    them. The cost of an unnecessary refetch on an unrelated session is
    one extra tool call; the cost of missing one is the bug we're here
    to fix.

    Returns the count of sessions notified for log visibility.
    """
    # `reference_id` is already prefixed (`DocumentReference/<uuid>`) from
    # the writer; do not re-prefix it here or the agent sees a malformed
    # double-prefixed source.
    notice = SystemMessage(content=(
        f"DATA UPDATE: A new clinical document was just uploaded "
        f"(doc_type={doc_type}, source={reference_id}, "
        f"patient={patient_uuid}). Any earlier tool results in this "
        f"conversation for this patient may be stale. Before answering "
        f"the next question about this patient, refetch the relevant "
        f"tools (get_patient_card, get_notes_24h, get_document_content, "
        f"get_observations_24h, get_med_changes_24h) instead of reusing "
        f"prior ToolMessage content."
    ))
    affected = 0
    for (sess_user, _sid), state in SESSIONS.items():
        if sess_user != username:
            continue
        state["messages"] = [*state["messages"], notice]
        affected += 1
    return affected


async def _prewarm_dashboard(app: FastAPI) -> None:
    """Background task: warm OAuth, load calendar + every patient on it.

    Crucial: every fetch routes through `cache.get_or_compute` — so if a
    user request lands mid-prewarm, both share the same in-flight call
    via the per-key lock. No duplicate work, no thundering herd.

    OpenEMR's FHIR layer serializes responses on the server side (PHP
    session locking + MariaDB), so per-patient parallelism only buys
    us a little. The real win is **batching**: one roster-wide search
    per resource type, bucketed by patient client-side. See
    `app.fhir.prewarm.warm_panel_cards_and_docs`.
    """
    from app.fhir.prewarm import warm_panel_cards_and_docs  # lazy: avoid cycle

    cache: TTLCache = app.state.cache
    fhir: FhirClient = app.state.fhir
    t0 = time.time()
    try:
        await fhir._ensure_token()  # noqa: SLF001
        log.info("prewarm: oauth token ready in %.2fs", time.time() - t0)

        async def _calendar() -> dict:
            # Prewarm the admin/no-filter view — matches `_calendar_cache_key(None)`
            # so the first admin request hits warm. Per-user filtered views are
            # cold on first request (rare in demo; 5-min TTL after).
            r = await get_calendar_today(
                fhir, panel=None,
                active_patients_store=getattr(app.state, "active_patients", None),
            )
            return r["data"]
        cal_data = await cache.get_or_compute("calendar:today:all", _calendar)
        # Pull the raw Patient resources too — the calendar fetch produced
        # them but only kept the trimmed dashboard shape. The batched
        # prewarm wants the full FHIR Patient resource so it can format
        # the patient-card payload (which needs name / DOB / gender).
        try:
            patients = await _safe_search_patients(fhir, panel=None)
        except Exception as e:  # noqa: BLE001
            log.warning("prewarm: Patient search failed: %s", e)
            patients = []
        log.info(
            "prewarm: calendar ready in %.2fs (%d patients)",
            time.time() - t0, len(patients),
        )

        warmed = await warm_panel_cards_and_docs(
            fhir, cache, patients,
            active_patients_store=getattr(app.state, "active_patients", None),
        )
        log.info(
            "prewarm: dashboard fully warm in %.2fs (cards=%d)",
            time.time() - t0, warmed,
        )
    except Exception as e:  # noqa: BLE001
        log.warning("prewarm failed (non-fatal): %s", e)


async def _safe_search_patients(fhir: FhirClient, *, panel: frozenset[str] | None):
    """Wrapper around the Patient search the calendar already does, but
    returning the full resource list (the calendar fetcher trims to
    dashboard shape). Same `_count` as `get_calendar_today` so the two
    paths see the same roster.
    """
    rows = await fhir.search("Patient", {"_count": 50})
    rows = access_control.filter_active(
        rows, dynamic_store=getattr(app.state, "active_patients", None),
    )
    if panel is None:
        return rows
    return [p for p in rows if p.get("id") in panel]


MODEL_NAME = "claude-sonnet-4-6"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    init_langsmith()  # idempotent; no-op when LANGSMITH_TRACING is unset
    app.state.fhir = FhirClient()
    app.state.openemr_writer = OpenEMRWriter()
    # Clinical notes persist to disk so drafts survive `systemctl restart copilot`.
    # Loaded BEFORE `build_graph` because the agent's `get_vital_trends` tool
    # reads notes through this store.
    notes_path = Path(os.environ.get("CLINICAL_NOTES_PATH", "data/clinical_notes.json"))
    app.state.clinical_notes = ClinicalNoteStore(notes_path)
    log.info("clinical-notes store loaded from %s (%d notes)", notes_path, len(app.state.clinical_notes._notes))
    # Audit trail persists to SQLite so traces survive `systemctl restart`
    # and every deploy. Path is overridable for tests / alternate volumes.
    trace_db_path = Path(os.environ.get("TRACE_DB_PATH", "data/traces.db"))
    app.state.traces = SqliteTraceStore(trace_db_path)
    log.info("trace store: sqlite at %s", trace_db_path)
    # Auth store shares the same SQLite file (separate tables: `sessions`
    # and `auth_events`). Same path-override convention as the trace store.
    auth_db_path = Path(os.environ.get("AUTH_DB_PATH", trace_db_path))
    app.state.auth_store = AuthStore(auth_db_path)
    log.info("auth store: sqlite at %s", auth_db_path)
    # Patient-assignment store shares the same SQLite file (separate table:
    # `patient_assignments`). Source of truth for the per-tool ACL gate
    # because OpenEMR's UI doesn't write to FHIR `Patient.generalPractitioner`.
    app.state.assignments = AssignmentStore(auth_db_path)
    log.info("assignment store: sqlite at %s", auth_db_path)
    # Soft-hide store for the supporting-documents view — doctors can
    # tidy duplicates / wrong-patient uploads without deleting the
    # underlying OpenEMR DocumentReference. Shares the same SQLite file
    # via a separate `hidden_documents` table.
    app.state.hidden_docs = HiddenDocsStore(auth_db_path)
    log.info("hidden-docs store: sqlite at %s", auth_db_path)
    # Sidecar mapping (resource_type, resource_id) -> source-doc + bbox.
    # Required because OpenEMR's FHIR allergy serializer drops the
    # `comments` column where the writer puts its `[copilot-source: ...]`
    # tag. Same SQLite file, separate `extracted_resource_sources` table.
    app.state.extracted_sources = ExtractedSourcesStore(auth_db_path)
    log.info("extracted-sources store: sqlite at %s", auth_db_path)
    # Care-team entries derived from referral-letter extractions. Same
    # SQLite file, separate `extracted_practitioners` table. Powers the
    # Modern Dashboard's Care Team tab (FHIR CareTeam is rarely populated
    # for demo patients).
    app.state.extracted_practitioners = ExtractedPractitionersStore(auth_db_path)
    log.info("extracted-practitioners store: sqlite at %s", auth_db_path)
    # Lab values derived from lab-PDF / fax-packet extractions. Same SQLite
    # file, separate `extracted_lab_results` table. Powers the Modern
    # Dashboard's Lab Results tab — OpenEMR's REST API has no
    # `procedure_result` write endpoint and no FHIR Observation write
    # endpoint, so the writer persists labs as SOAP-note objective text;
    # this store is the structured mirror that makes them queryable.
    app.state.extracted_lab_results = ExtractedLabResultsStore(auth_db_path)
    log.info("extracted-lab-results store: sqlite at %s", auth_db_path)
    # Content-fingerprint index for dedup Layer 2 — same SQLite file,
    # separate `document_fingerprints` table.
    app.state.fingerprints = DocumentFingerprintStore(auth_db_path)
    log.info("fingerprints store: sqlite at %s", auth_db_path)
    # Dynamic active-patients overrides — doctors create new patients via
    # the upload-extract-and-create flow; their (family, given) tuples
    # land here so they show up in the dashboard / picker without a
    # source-code change to ACTIVE_PATIENT_NAMES.
    from app.active_patients_db import ActivePatientsStore  # noqa: E402
    app.state.active_patients = ActivePatientsStore(auth_db_path)
    log.info("active-patients store: sqlite at %s", auth_db_path)
    # build_graph AFTER assignment store so the per-tool ACL gate has its
    # source of truth wired in.
    app.state.graph = build_graph(
        app.state.fhir, app.state.clinical_notes,
        assignments_store=app.state.assignments, model_name=MODEL_NAME,
    )
    app.state.cache = TTLCache(ttl_seconds=DATA_CACHE_TTL_S)
    # Separate cache for rendered PDF pages + bbox manifest. Distinct
    # because the TTL is much longer (rendered pages don't drift) and the
    # values are large (multi-MB PNG b64 per doc) — keeping them off the
    # main TTL path means a single doc-prefetch storm can't crowd out the
    # small chart-card / docs-list entries.
    app.state.pages_cache = TTLCache(ttl_seconds=PAGES_CACHE_TTL_S)

    # Don't block startup on the prewarm — let it run while uvicorn binds.
    prewarm_task = asyncio.create_task(_prewarm_dashboard(app))
    try:
        yield
    finally:
        prewarm_task.cancel()
        await app.state.fhir.aclose()
        await app.state.openemr_writer.aclose()
        app.state.traces.close()
        app.state.auth_store.close()
        app.state.assignments.close()
        app.state.hidden_docs.close()
        app.state.extracted_sources.close()
        app.state.extracted_practitioners.close()
        app.state.extracted_lab_results.close()
        app.state.fingerprints.close()
        app.state.active_patients.close()


app = FastAPI(title="Clinical Co-pilot", lifespan=lifespan)

app.add_middleware(
    SessionMiddleware,
    secret_key=settings.copilot_session_secret,
    session_cookie="copilot_session",
    same_site="lax",
    https_only=False,  # cloudflared terminates TLS; cookie still travels over public HTTPS
    max_age=12 * 60 * 60,  # 12-hour shift
)

# Modern Dashboard static bundle. Vite builds into
# `clinical-copilot/app/web/dashboard-build/`; we mount it under
# `/dashboard` so `/dashboard` serves index.html and `/dashboard/assets/*`
# serves the hashed JS+CSS chunks. `html=True` makes the mount serve
# index.html for the bare prefix path (the SPA reads `?pid=...` from
# location.search, so we don't need a SPA-style catch-all).
#
# The bare `/dashboard` and `/dashboard/` paths are intercepted by an
# explicit route handler that injects an
# `<meta name="openemr-classic-base">` tag into the served index.html
# based on `settings.openemr_classic_base`. The dashboard's
# `classicBase()` (in src/api.ts) reads that tag to build "Open in
# classic OpenEMR" out-link URLs. Without the injection the dashboard
# falls back to its built-in https://localhost:9300 default — which is
# correct only for local dev. In production the classic OpenEMR base
# is a separate cloudflared tunnel URL whose value can change between
# deploys, so it MUST come from runtime config rather than be baked
# into the bundle. Mount-served assets (`/dashboard/assets/*`) are
# unaffected by the route handler.
#
# The directory may not exist on a fresh checkout (it's a build
# artifact, gitignored). We fall through to a stub route in that case
# so the app still starts and `/dashboard` returns a clear "build not
# found" message instead of a 500.
_DASHBOARD_BUILD_DIR = WEB_DIR / "dashboard-build"
if _DASHBOARD_BUILD_DIR.is_dir() and (_DASHBOARD_BUILD_DIR / "index.html").is_file():
    _DASHBOARD_INDEX_PATH = _DASHBOARD_BUILD_DIR / "index.html"

    def _render_dashboard_index() -> str:
        """Read index.html and inject the openemr-classic-base meta tag.

        Read on every request rather than cached at startup so a build
        regeneration (npm run build) is picked up without restarting
        the server. The file is small (~500 bytes) and the dashboard
        loads at most once per browser session.
        """
        from html import escape  # noqa: PLC0415
        html = _DASHBOARD_INDEX_PATH.read_text()
        # Strip trailing slash because the frontend's classicLinks
        # builders concatenate `${classicBase()}${path}` where path
        # always starts with "/". Without this, an env value of
        # "https://example.com/" produces "https://example.com//interface/...".
        # Cloudflared and nginx collapse the double slash, but stricter
        # routers don't, and it's ugly in the address bar either way.
        base = (settings.openemr_classic_base or "").strip().rstrip("/")
        if base:
            meta = f'<meta name="openemr-classic-base" content="{escape(base, quote=True)}">'
            # Insert after the opening <head> tag. Falls back to a
            # straight prepend if <head> is missing (shouldn't happen
            # with vite output, but defensively don't lose the tag).
            if "<head>" in html:
                html = html.replace("<head>", f"<head>\n    {meta}", 1)
            else:
                html = meta + html
        return html

    @app.get("/dashboard", response_class=HTMLResponse)
    @app.get("/dashboard/", response_class=HTMLResponse)
    async def _dashboard_index() -> HTMLResponse:
        return HTMLResponse(_render_dashboard_index())

    app.mount(
        "/dashboard",
        StaticFiles(directory=str(_DASHBOARD_BUILD_DIR), html=True),
        name="dashboard",
    )
else:
    @app.get("/dashboard")
    async def _dashboard_not_built() -> Response:
        return Response(
            "Modern Dashboard build not found. From `clinical-copilot/dashboard/` "
            "run `npm install && npm run build` to populate "
            "app/web/dashboard-build/, then restart the server.",
            status_code=503,
            media_type="text/plain",
        )


class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str
    # Per-turn medication-safety advisor toggle (default off). When True the
    # graph swaps in the R2-relaxed system prompt for this turn (and only
    # this turn — the value is re-sent by the client on every request, so
    # turning the UI switch off mid-conversation immediately drops the
    # agent back into chart-summarizer-only mode).
    advisor_mode: bool = False


class ChatResponse(BaseModel):
    session_id: str
    response: str
    patient_id: str | None
    sources: list[str]
    validation_warning: bool


class LoginRequest(BaseModel):
    username: str
    password: str


# ─── auth + shell ────────────────────────────────────────────────────────


_NO_CACHE_HEADERS = {
    # The chat UI ships as a single HTML file with the JS embedded inline.
    # When browsers cache it, a deploy that ships new client-side code
    # (new event handlers, new endpoints) is invisible until the user does
    # a hard refresh — exactly the failure mode we hit during the Sunday
    # branch preview when sign-out drafting silently didn't wire because
    # the user was holding stale HTML. We ship behavior changes often
    # enough that "always revalidate" is the right default. The file is
    # small (~80 KB) and the 304-no-body round-trip is negligible.
    "Cache-Control": "no-cache, must-revalidate",
}


@app.get("/", response_model=None)
async def index(request: Request) -> FileResponse | RedirectResponse:
    if current_session(request) is None:
        return RedirectResponse(url="/login", status_code=302)
    return FileResponse(WEB_DIR / "index.html", headers=_NO_CACHE_HEADERS)


@app.get("/login", response_model=None)
async def login_page(request: Request) -> FileResponse | RedirectResponse:
    """Serve the username+password login form.

    Two parallel login paths are intentionally kept:

      - This form-post path (`/login` → POST `/api/login`) is the
        Co-Pilot-only login. It validates credentials against OpenEMR
        via password-grant, creates a Co-Pilot session, and lands the
        user on the Co-Pilot chat surface. OpenEMR's PHP session is
        NOT seeded; OpenEMR-served pages still require their own
        sign-in.

      - The OAuth2/OIDC path (`/oauth/login` → OpenEMR authorize →
        callback → session) covers the W2 dashboard rubric's
        "Authentication via OAuth2/OpenID Connect" requirement. Use
        this when single-sign-on across Co-Pilot and OpenEMR is
        wanted in the same browser tab.

    Both lead to the same Co-Pilot session shape, so the chat surface
    and the Modern Dashboard work identically regardless of which
    door the user came through. The dashboard reads Copilot's
    session cookie only — no OpenEMR PHP session is required for the
    dashboard surface itself.
    """
    if current_session(request) is not None:
        return RedirectResponse(url="/", status_code=302)
    return FileResponse(WEB_DIR / "login.html", headers=_NO_CACHE_HEADERS)


def _client_ua_ip(request: Request) -> tuple[str | None, str | None]:
    """Pull a best-guess UA + IP off the request for the auth audit log.
    Cloudflared adds X-Forwarded-For; falls back to the direct peer."""
    ua = request.headers.get("user-agent")
    ip = request.headers.get("x-forwarded-for") or (
        request.client.host if request.client else None
    )
    # X-Forwarded-For may carry "client, proxy1, proxy2" — keep the leftmost.
    if ip and "," in ip:
        ip = ip.split(",", 1)[0].strip()
    return ua, ip


@app.post("/api/login")
async def login(req: LoginRequest, request: Request) -> dict:
    ua, ip = _client_ua_ip(request)
    try:
        ok = await verify_openemr_credentials(req.username, req.password)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    if not ok:
        request.app.state.auth_store.log_event(
            "login_failure", username=req.username, user_agent=ua, ip=ip,
        )
        raise HTTPException(status_code=401, detail="Invalid credentials")
    sid = request.app.state.auth_store.create_session(
        username=req.username, user_agent=ua, ip=ip,
    )
    request.session["sid"] = sid
    request.app.state.auth_store.log_event(
        "login_success", username=req.username, sid=sid, user_agent=ua, ip=ip,
    )
    return {"ok": True, "username": req.username}


@app.post("/api/logout")
async def logout(request: Request) -> dict:
    sid = request.session.get("sid")
    if sid:
        ua, ip = _client_ua_ip(request)
        username = request.app.state.auth_store.revoke_session(sid)
        # Log a `logout` event (rather than session_revoked) so the audit
        # feed distinguishes "user clicked sign-out" from "admin kicked".
        # If the sid was already revoked or absent, username is None and
        # we still emit the logout marker — the user clicked the button.
        request.app.state.auth_store.log_event(
            "logout", username=username, sid=sid, user_agent=ua, ip=ip,
        )
    request.session.clear()
    return {"ok": True}


@app.get("/api/me")
async def me(username: str = Depends(current_user)) -> dict:
    return {"username": username, "is_admin": is_admin(username)}


# ─── OAuth2 / OIDC login (replaces password-grant for new dashboard) ────


@app.get("/oauth/login", response_model=None)
async def oauth_login(request: Request, next: str | None = None) -> RedirectResponse:
    """Phase 1 of the auth-code flow. Generates a PKCE verifier+challenge
    and a state nonce, stashes them in the user's signed session cookie,
    then 302s the browser to OpenEMR's hosted login page. After the user
    authenticates there, OpenEMR redirects to /oauth/callback below.

    `?next=<path>` carries a same-origin path the user wanted before
    being bounced to login (e.g. /dashboard). Open-redirect guarded —
    only paths starting with `/` and not `//` are accepted.
    """
    from app.oauth import build_authorize_url  # noqa: PLC0415
    try:
        url = build_authorize_url(request, next_path=next)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    return RedirectResponse(url=url, status_code=302)


@app.get("/oauth/callback", response_model=None)
async def oauth_callback(
    request: Request,
    code: str | None = None,
    state: str | None = None,
    error: str | None = None,
    error_description: str | None = None,
) -> RedirectResponse:
    """Phase 2 of the auth-code flow. OpenEMR redirects here after the
    user signs in. We validate state, exchange the code for an id_token,
    decode the username, then create the same Copilot session the
    legacy /api/login created — so the rest of the app behaves
    identically post-OAuth.
    """
    from app.oauth import consume_next_path, exchange_code_for_username  # noqa: PLC0415

    if error:
        # Provider-side error (user denied, scope rejected, etc.). Keep
        # the message in the audit log but surface a generic message to
        # the user — the description may carry stack-trace-shaped detail
        # we don't want in the response.
        ua, ip = _client_ua_ip(request)
        request.app.state.auth_store.log_event(
            "oauth_provider_error",
            username=None, user_agent=ua, ip=ip,
            detail=f"{error}: {(error_description or '')[:200]}",
        )
        raise HTTPException(status_code=400, detail=f"OAuth error: {error}")

    if not code or not state:
        raise HTTPException(
            status_code=400,
            detail="OAuth callback missing required `code` or `state` parameter.",
        )

    ua, ip = _client_ua_ip(request)
    try:
        username = await exchange_code_for_username(request, code=code, state=state)
    except HTTPException:
        request.app.state.auth_store.log_event(
            "oauth_exchange_failure", username=None, user_agent=ua, ip=ip,
        )
        raise
    sid = request.app.state.auth_store.create_session(
        username=username, user_agent=ua, ip=ip,
    )
    request.session["sid"] = sid
    request.app.state.auth_store.log_event(
        "login_success", username=username, sid=sid, user_agent=ua, ip=ip,
        detail="oauth2_authcode_pkce",
    )
    return RedirectResponse(url=consume_next_path(request), status_code=302)


# ─── modern dashboard API ───────────────────────────────────────────────
#
# Read-only patient-summary endpoints that power the React/Vite dashboard
# at /dashboard. Each endpoint:
#   1. Requires a logged-in Copilot session (`current_user` dependency).
#   2. Resolves the user's patient panel and asserts `pid` is in it. The
#      ACL is fail-closed: a non-admin with no panel mapping sees 404 on
#      every patient.
#   3. Fans out to the FHIR client and formats one card's worth of data.
#
# The dashboard's TypeScript API client (`dashboard/src/api.ts`) is the
# canonical consumer; response shapes here MUST match the interfaces
# declared there.


def _dashboard_header_payload(patient: dict, patient_id: str) -> dict:
    """Format a Patient resource into the dashboard's PatientHeader shape."""
    name = (patient.get("name") or [{}])[0]
    given = list(name.get("given") or [])
    family = name.get("family") or ""
    full_name = " ".join([*given, family]).strip() or "(unnamed patient)"

    # MRN from `identifier[].value` — prefer `type.coding.code == 'MR'` per
    # FHIR convention, fall back to the first identifier with a value.
    mrn: str | None = None
    for ident in patient.get("identifier") or []:
        codings = ((ident.get("type") or {}).get("coding") or [])
        if any((c or {}).get("code") == "MR" for c in codings):
            v = ident.get("value")
            if isinstance(v, str) and v:
                mrn = v
                break
    if mrn is None:
        for ident in patient.get("identifier") or []:
            v = ident.get("value")
            if isinstance(v, str) and v:
                mrn = v
                break

    # OpenEMR's internal numeric pid lives in `identifier[]` under
    # `type.coding.code == 'PT'`. The dashboard surfaces this so the
    # "Open in classic" out-link URLs can carry `set_pid=<numeric>`
    # — `interface/.../setpid()` calls `intval()` on the value and
    # rejects non-numeric input, so the FHIR UUID alone would land
    # users on patient #0 (or an error). Falls back to None when the
    # identifier shape varies across deployments; the frontend then
    # disables the out-link rather than emit a broken URL.
    numeric_pid: str | None = None
    for ident in patient.get("identifier") or []:
        codings = ((ident.get("type") or {}).get("coding") or [])
        if any((c or {}).get("code") == "PT" for c in codings):
            v = ident.get("value")
            if isinstance(v, str) and v.isdigit():
                numeric_pid = v
                break

    primary_phone: str | None = None
    for tel in patient.get("telecom") or []:
        if (tel or {}).get("system") == "phone":
            v = tel.get("value")
            if isinstance(v, str) and v:
                primary_phone = v
                break

    from app.fhir.adapter import _calc_age  # noqa: PLC0415
    return {
        "id": patient_id,
        "full_name": full_name,
        "given": given,
        "family": family,
        "birth_date": patient.get("birthDate"),
        "age": _calc_age(patient.get("birthDate")),
        "gender": patient.get("gender"),
        "mrn": mrn,
        "active": bool(patient.get("active", True)),
        "primary_phone": primary_phone,
        "numeric_pid": numeric_pid,
    }


def _dashboard_allergy_entry(a: dict) -> dict:
    from app.fhir.adapter import _coded_display, _narrative_text  # noqa: PLC0415
    crit = a.get("criticality")
    # FHIR R4 `criticality` is one of low|high|unable-to-assess. Anything
    # else gets normalized to None so the UI's pill renderer can rely on
    # a closed enum.
    if crit not in ("low", "high", "unable-to-assess"):
        crit = None
    clinical_status = ((a.get("clinicalStatus") or {}).get("coding") or [{}])[0].get("code")
    # Fallback chain: prefer a coded display, then narrative text, then
    # "(unspecified)". OpenEMR stores allergies entered via the patient
    # intake flow as narrative text without SNOMED/RxNorm codings — so
    # without the narrative fallback every Co-Pilot-uploaded allergy
    # would render as "(unspecified)" on the dashboard while still
    # showing correctly on the Co-Pilot's chat-surface patient card.
    display = _coded_display(a.get("code") or {}) or _narrative_text(a) or "(unspecified)"
    return {
        "id": a.get("id") or "",
        "display": display,
        "criticality": crit,
        "clinical_status": clinical_status,
        "recorded_date": a.get("recordedDate"),
    }


def _dashboard_condition_entry(c: dict) -> dict:
    from app.fhir.adapter import _coded_display, _narrative_text  # noqa: PLC0415
    clinical_status = ((c.get("clinicalStatus") or {}).get("coding") or [{}])[0].get("code")
    display = _coded_display(c.get("code") or {}) or _narrative_text(c) or "(unspecified)"
    return {
        "id": c.get("id") or "",
        "display": display,
        "clinical_status": clinical_status,
        "onset_date": c.get("onsetDateTime"),
    }


def _dashboard_med_entry(m: dict) -> dict:
    from app.fhir.adapter import _coded_display, _narrative_text  # noqa: PLC0415
    dosage = (m.get("dosageInstruction") or [{}])[0]
    display = (
        _coded_display(m.get("medicationCodeableConcept") or {})
        or _narrative_text(m)
        or "(unspecified)"
    )
    return {
        "id": m.get("id") or "",
        "display": display,
        "status": m.get("status"),
        "authored_date": m.get("authoredOn"),
        "dosage_text": dosage.get("text"),
    }


def _dashboard_prescription_entry(m: dict) -> dict:
    """Format a MedicationRequest with the prescription-order lens.

    Same underlying FHIR resource as `_dashboard_med_entry`, but the
    fields exposed here are the ones a prescriber/admin cares about
    (who wrote it, when, how many refills) rather than the clinical
    summary fields (drug + dosage). The rubric calls for both cards
    distinctly, so we expose the full set as separate views over the
    same data.
    """
    from app.fhir.adapter import _coded_display, _narrative_text  # noqa: PLC0415
    display = (
        _coded_display(m.get("medicationCodeableConcept") or {})
        or _narrative_text(m)
        or "(unspecified)"
    )
    requester = (m.get("requester") or {}).get("display") or None
    dispense = m.get("dispenseRequest") or {}
    refills = dispense.get("numberOfRepeatsAllowed")
    quantity = (dispense.get("quantity") or {}).get("value")
    return {
        "id": m.get("id") or "",
        "display": display,
        "status": m.get("status"),
        "intent": m.get("intent"),
        "authored_date": m.get("authoredOn"),
        "prescriber": requester,
        "refills": refills if isinstance(refills, int) else None,
        "quantity": quantity,
    }


def _dashboard_careteam_entry(ct: dict) -> list[dict]:
    """Flatten one CareTeam into one entry per participant.

    A FHIR CareTeam carries `participant[].member.display` for each
    person on the team; we render one row per participant rather than
    one row per team. Empty `participant` lists yield an empty result.
    Defensive against malformed/null participant entries — OpenEMR's
    FHIR can return `participant: [null]` for half-populated rows.
    """
    out: list[dict] = []
    ct_id = ct.get("id") or ""
    for i, p in enumerate(ct.get("participant") or []):
        if not isinstance(p, dict):
            continue
        member = p.get("member") or {}
        role_list = p.get("role") or []
        role_coding: dict = {}
        if role_list and isinstance(role_list[0], dict):
            coding_list = role_list[0].get("coding") or []
            if coding_list and isinstance(coding_list[0], dict):
                role_coding = coding_list[0]
        role = role_coding.get("display") or role_coding.get("code")
        out.append({
            "id": f"{ct_id}:{i}",
            "name": member.get("display"),
            "specialty": role,
            "practice": None,
            "phone": None,
            "address": None,
            "npi": None,
            "source": "fhir",
        })
    return out


def _dashboard_practitioner_entry(p: Any) -> dict:
    """Render one ExtractedPractitioner row as a dashboard care-team entry.

    Schema mirrors `_dashboard_careteam_entry` so the React card can
    iterate one merged list. `source="extracted"` lets the UI render a
    "from <referral document>" deep-link; `id` is a stable string
    derived from the source-doc id so React keys stay stable across
    refetches.
    """
    return {
        "id": f"prac:{p.source_doc_id}",
        "name": p.name,
        "specialty": p.specialty,
        "practice": p.practice,
        "phone": p.phone,
        "address": p.address,
        "npi": p.npi,
        "source": "extracted",
        "source_doc_id": p.source_doc_id,
    }


# Vitals: LOINC codes the dashboard knows how to label + render. Other
# observations slip through as "Other" only if they have a usable name;
# empty rows are dropped before they reach the client.
_VITAL_LOINC_LABELS: dict[str, str] = {
    "8480-6":  "Systolic BP",
    "8462-4":  "Diastolic BP",
    "8867-4":  "Heart Rate",
    "29463-7": "Weight",
    "8302-2":  "Height",
    "8310-5":  "Body Temperature",
    "9279-1":  "Respiratory Rate",
    "59408-5": "SpO2",
    "39156-5": "BMI",
}


def _dashboard_vitals_series(observations: list[dict]) -> list[dict]:
    """Group vital-signs Observations by LOINC code and emit time-series.

    BP Observations carry two `component` entries (systolic + diastolic)
    instead of a top-level `valueQuantity` — split them into the two
    LOINC-keyed series so the cards can plot each independently.
    """
    series: dict[str, dict] = {}

    def push(loinc: str, display: str, unit: str | None, effective: str | None, value: float | None) -> None:
        if not effective or value is None:
            return
        s = series.setdefault(
            loinc,
            {"loinc": loinc, "display": display, "unit": unit, "readings": []},
        )
        s["readings"].append({"effective": effective, "value": value, "unit": unit})

    for o in observations:
        eff = o.get("effectiveDateTime") or o.get("issued")
        # Top-level value first.
        codings = ((o.get("code") or {}).get("coding") or [])
        primary = next((c for c in codings if (c or {}).get("system") == "http://loinc.org"), {})
        primary_loinc = primary.get("code")
        primary_label = _VITAL_LOINC_LABELS.get(primary_loinc or "", primary.get("display") or "Vital")
        vq = o.get("valueQuantity") or {}
        v = vq.get("value")
        if isinstance(v, (int, float)) and primary_loinc:
            push(primary_loinc, primary_label, vq.get("unit"), eff, float(v))

        # Components for composite vitals (BP).
        for comp in o.get("component") or []:
            comp_codings = ((comp.get("code") or {}).get("coding") or [])
            cl = next((c for c in comp_codings if (c or {}).get("system") == "http://loinc.org"), {})
            cl_code = cl.get("code")
            if not cl_code:
                continue
            cvq = comp.get("valueQuantity") or {}
            cv = cvq.get("value")
            if isinstance(cv, (int, float)):
                push(
                    cl_code,
                    _VITAL_LOINC_LABELS.get(cl_code, cl.get("display") or "Vital"),
                    cvq.get("unit"),
                    eff,
                    float(cv),
                )

    # Sort each series chronologically so the sparkline draws left-to-right.
    out: list[dict] = []
    for loinc, s in series.items():
        s["readings"].sort(key=lambda r: r.get("effective") or "")
        out.append(s)
    # Stable order: BPs first, then HR / Temp / RR / SpO2 / BMI / Weight / Height.
    order = ["8480-6", "8462-4", "8867-4", "8310-5", "9279-1", "59408-5", "39156-5", "29463-7", "8302-2"]
    out.sort(key=lambda s: order.index(s["loinc"]) if s["loinc"] in order else 999)
    return out


async def _dashboard_assert_patient_in_panel(request: Request, username: str, pid: str) -> None:
    """Shared ACL preamble for every dashboard endpoint. Raises 404 (NOT
    403) when the patient is outside the user's panel — leaking
    "this patient exists but you can't see them" is a soft-info-leak we
    explicitly avoid in this app."""
    panel = await access_control.get_panel_for_user(
        request.app.state.fhir, username, request.app.state.assignments,
    )
    if not access_control.is_in_panel(panel, pid):
        raise HTTPException(status_code=404, detail="Patient not found")


@app.get("/api/dashboard/patient/{pid}/header")
async def dashboard_header(
    pid: str, request: Request, username: str = Depends(current_user),
) -> dict:
    await _dashboard_assert_patient_in_panel(request, username, pid)
    patient = await request.app.state.fhir.get(f"Patient/{pid}")
    return _dashboard_header_payload(patient, pid)


@app.get("/api/dashboard/patient/{pid}/allergies")
async def dashboard_allergies(
    pid: str, request: Request, username: str = Depends(current_user),
) -> dict:
    await _dashboard_assert_patient_in_panel(request, username, pid)
    rows = await request.app.state.fhir.search("AllergyIntolerance", {"patient": pid, "_count": 200})
    return {"items": [_dashboard_allergy_entry(a) for a in rows]}


@app.get("/api/dashboard/patient/{pid}/conditions")
async def dashboard_conditions(
    pid: str, request: Request, username: str = Depends(current_user),
) -> dict:
    await _dashboard_assert_patient_in_panel(request, username, pid)
    rows = await request.app.state.fhir.search("Condition", {"patient": pid, "_count": 200})
    # Filter to active/recurrence only — closed/inactive problems
    # belong on a "history" tab, not the live problem list.
    active = [
        c for c in rows
        if ((c.get("clinicalStatus") or {}).get("coding") or [{}])[0].get("code")
        in {"active", "recurrence", "relapse", None}
    ]
    return {"items": [_dashboard_condition_entry(c) for c in active]}


@app.get("/api/dashboard/patient/{pid}/medications")
async def dashboard_medications(
    pid: str, request: Request, username: str = Depends(current_user),
) -> dict:
    await _dashboard_assert_patient_in_panel(request, username, pid)
    rows = await request.app.state.fhir.search("MedicationRequest", {"patient": pid, "_count": 200})
    # Medications view: active list — what the patient is currently
    # taking. Filters out cancelled / stopped / draft entries that
    # belong on a "history" tab, not the live clinical summary.
    active = [
        m for m in rows
        if m.get("status") in {"active", "completed", "intended", None}
    ]
    return {"items": [_dashboard_med_entry(m) for m in active]}


@app.get("/api/dashboard/patient/{pid}/prescriptions")
async def dashboard_prescriptions(
    pid: str, request: Request, username: str = Depends(current_user),
) -> dict:
    """Prescription-order view of MedicationRequest.

    The rubric calls for Medications AND Prescriptions as distinct
    cards. They surface the same FHIR resource with different lenses:
    the medications card focuses on the drug + dose (clinical), this
    card focuses on the order itself (prescriber, date written,
    refills, dispense quantity). Sorted by authoredOn descending so
    the most-recently-written prescription is at the top.
    """
    await _dashboard_assert_patient_in_panel(request, username, pid)
    rows = await request.app.state.fhir.search("MedicationRequest", {"patient": pid, "_count": 200})
    rows.sort(key=lambda r: r.get("authoredOn") or "", reverse=True)
    return {"items": [_dashboard_prescription_entry(m) for m in rows]}


@app.get("/api/dashboard/patient/{pid}/care-team")
async def dashboard_care_team(
    pid: str, request: Request, username: str = Depends(current_user),
) -> dict:
    """Care Team for the dashboard, merged from two sources.

    1. **FHIR CareTeam** — OpenEMR's native resource. In practice rarely
       populated for demo patients; the demo cohort has no rows here.
       Wrapped in try/except because OpenEMR's FHIR layer can 4xx/5xx
       on the CareTeam search path (the resource is partially supported)
       and a fetch failure here should NOT break the whole tab.
    2. **Extracted practitioners** — referring physicians captured by the
       Phase 2 VLM pipeline when a referral letter is uploaded for this
       patient. Stored in the Co-Pilot's SQLite (`extracted_practitioners`
       table) because OpenEMR's FHIR has no native target for the
       contact-block fields a referral prints (specialty / phone /
       address). See `app/extracted_practitioners_db.py` for the schema.

    Empty result is the honest answer for a patient with no FHIR
    CareTeam and no uploaded referrals — the React card renders
    "No care team members on file."
    """
    await _dashboard_assert_patient_in_panel(request, username, pid)
    items: list[dict] = []
    try:
        teams = await request.app.state.fhir.search("CareTeam", {"patient": pid, "_count": 50})
        for ct in teams:
            items.extend(_dashboard_careteam_entry(ct))
    except Exception as e:  # noqa: BLE001
        # OpenEMR's CareTeam FHIR endpoint isn't fully reliable; treat
        # any error here as "no FHIR care team rows" rather than 500ing
        # the whole tab.
        log.warning("dashboard_care_team: FHIR CareTeam fetch failed for pid=%s: %s", pid, e)
    practitioners_store = getattr(request.app.state, "extracted_practitioners", None)
    if practitioners_store is not None:
        try:
            rows = practitioners_store.list_for_patient(pid)
            items.extend(_dashboard_practitioner_entry(p) for p in rows)
        except Exception as e:  # noqa: BLE001
            # Same defensive posture as the FHIR branch above — a SQLite
            # error here (file locked, schema mismatch) shouldn't 500
            # the tab. Worst case the user sees an empty Care Team.
            log.warning("dashboard_care_team: practitioners store read failed for pid=%s: %s", pid, e)
    return {"items": items}


@app.get("/api/dashboard/patient/{pid}/vitals")
async def dashboard_vitals(
    pid: str, request: Request, username: str = Depends(current_user),
) -> dict:
    await _dashboard_assert_patient_in_panel(request, username, pid)
    rows = await request.app.state.fhir.search(
        "Observation", {"patient": pid, "category": "vital-signs", "_count": 200},
    )
    return {"series": _dashboard_vitals_series(rows)}


@app.get("/api/dashboard/patient/{pid}/lab-results-debug")
async def dashboard_lab_results_debug(
    pid: str, request: Request, username: str = Depends(current_user),
) -> dict:
    """Diagnostic for the Lab Results backfill. Returns what each phase
    of the backfill actually sees — encounter count, per-encounter
    reason text, objective-text length, parsed row count — so we can
    pinpoint exactly where the chain breaks for a given patient.

    This bypasses the per-process `lab_backfill_tried` guard so a stale
    failed attempt doesn't mask a fix.
    """
    await _dashboard_assert_patient_in_panel(request, username, pid)
    out: dict = {"pid": pid, "encounters": [], "error": None}
    writer = getattr(request.app.state, "openemr_writer", None)
    store = getattr(request.app.state, "extracted_lab_results", None)
    out["writer_available"] = writer is not None
    out["store_available"] = store is not None
    if writer is None:
        out["error"] = "openemr_writer not on app.state"
        return out
    try:
        encounters = await writer.list_encounters(pid)
    except Exception as e:  # noqa: BLE001
        out["error"] = f"list_encounters failed: {type(e).__name__}: {e}"
        return out
    out["encounter_count_total"] = len(encounters)
    lab_count = 0
    for enc in encounters:
        if not isinstance(enc, dict):
            continue
        reason = str(enc.get("reason") or "")
        eid_raw = enc.get("eid") or enc.get("id") or enc.get("encounter_id")
        try:
            eid = int(eid_raw) if eid_raw is not None else None
        except (TypeError, ValueError):
            eid = None
        is_lab = reason.lower().startswith("lab results")
        info: dict = {
            "eid": eid,
            "date": str(enc.get("date") or ""),
            "reason_prefix": reason[:120],
            "matches_lab_filter": is_lab,
            "available_keys": sorted(list(enc.keys()))[:20],
        }
        if is_lab and eid is not None:
            lab_count += 1
            try:
                soap = await writer.get_soap_note(pid, eid)
            except Exception as e:  # noqa: BLE001
                info["soap_error"] = f"{type(e).__name__}: {e}"
                soap = None
            if soap:
                obj = str(soap.get("objective") or "")
                info["objective_chars"] = len(obj)
                info["objective_first_300"] = obj[:300]
                parsed = 0
                samples = []
                for raw_line in obj.splitlines():
                    p = _parse_lab_objective_line(raw_line)
                    if p:
                        parsed += 1
                        if len(samples) < 3:
                            samples.append({"line": raw_line.strip()[:120], "parsed": p})
                info["parsed_row_count"] = parsed
                info["parsed_samples"] = samples
            else:
                info["soap_present"] = False
        out["encounters"].append(info)
    out["lab_encounter_count"] = lab_count
    return out


@app.get("/api/dashboard/patient/{pid}/lab-results")
async def dashboard_lab_results(
    pid: str, request: Request, username: str = Depends(current_user),
) -> dict:
    """Lab results, merged from two sources.

    1. **FHIR Observations** (`category=laboratory`) — empty for the
       demo cohort because OpenEMR's REST API exposes no
       `procedure_result` write endpoint and no FHIR Observation write
       endpoint, so the writer can't put extracted labs there. Kept
       in the merge so any externally-imported Observations still
       surface. Wrapped in try/except: a fetch failure shouldn't
       break the tab.
    2. **Extracted lab results** — the Co-Pilot's
       `extracted_lab_results` SQLite table, written on every lab-PDF
       upload by `persist_extracted_facts`. This is the data path
       that actually populates the demo. See
       `app/extracted_lab_results_db.py` for the schema.

    Returns one entry per result, ordered most-recent first.
    """
    from app.fhir.adapter import _coded_display, _narrative_text  # noqa: PLC0415
    await _dashboard_assert_patient_in_panel(request, username, pid)
    items: list[dict] = []
    try:
        rows = await request.app.state.fhir.search(
            "Observation", {"patient": pid, "category": "laboratory", "_count": 200},
        )
        for o in rows:
            codings = ((o.get("code") or {}).get("coding") or [])
            primary = next(
                (c for c in codings if (c or {}).get("system") == "http://loinc.org"),
                codings[0] if codings else {},
            )
            vq = o.get("valueQuantity") or {}
            items.append({
                "id": o.get("id") or "",
                "loinc": primary.get("code"),
                "test_name": primary.get("display") or _coded_display(o.get("code") or {}) or _narrative_text(o) or "(unspecified)",
                "value": vq.get("value"),
                "value_string": o.get("valueString"),
                "unit": vq.get("unit"),
                "reference_range_text": ((o.get("referenceRange") or [{}])[0].get("text")),
                "abnormal_flag": _abnormal_flag_from_observation(o),
                "effective": o.get("effectiveDateTime") or o.get("issued"),
                "status": o.get("status"),
                "source": "fhir",
            })
    except Exception as e:  # noqa: BLE001
        log.warning("dashboard_lab_results: FHIR Observation fetch failed for pid=%s: %s", pid, e)

    lab_store = getattr(request.app.state, "extracted_lab_results", None)
    if lab_store is not None:
        # Auto-backfill from SOAP notes when the store is empty for this
        # patient. Lab uploads predating the store live only as
        # SOAP-note objective text — without this, the tab stays empty
        # for patients whose lab PDFs were uploaded before today's fix.
        # Guarded by a per-process set so we attempt the backfill at
        # most once per patient per uvicorn lifetime; subsequent reads
        # use the populated store directly.
        try:
            initial_rows = lab_store.list_for_patient(pid)
        except Exception as e:  # noqa: BLE001
            log.warning("dashboard_lab_results: lab_results store read failed for pid=%s: %s", pid, e)
            initial_rows = []
        if not initial_rows:
            tried: set[str] = getattr(request.app.state, "lab_backfill_tried", set())
            if pid not in tried:
                tried.add(pid)
                request.app.state.lab_backfill_tried = tried
                try:
                    await _backfill_lab_results_from_soap(
                        patient_uuid=pid, request=request,
                    )
                    initial_rows = lab_store.list_for_patient(pid)
                except Exception as e:  # noqa: BLE001
                    log.warning("dashboard_lab_results: backfill failed for pid=%s: %s", pid, e)
        try:
            for r in initial_rows:
                # `value` was stringified on write to accommodate both
                # numeric ('7.4') and qualitative ('positive') labs;
                # try to recover a number for the card's right-aligned
                # value column, else fall back to the string form.
                num: float | None = None
                if r.value is not None:
                    try:
                        num = float(r.value)
                    except (ValueError, TypeError):
                        num = None
                items.append({
                    "id": f"el:{r.source_doc_id}:{r.row_index}",
                    "loinc": None,
                    "test_name": r.test_name,
                    "value": num,
                    "value_string": r.value if num is None else None,
                    "unit": r.unit,
                    "reference_range_text": r.reference_range,
                    "abnormal_flag": r.abnormal_flag if r.abnormal_flag and r.abnormal_flag != "N" else None,
                    "effective": r.collection_date,
                    "status": "final",
                    "source": "extracted",
                    "source_doc_id": r.source_doc_id,
                })
        except Exception as e:  # noqa: BLE001
            log.warning("dashboard_lab_results: lab_results store read failed for pid=%s: %s", pid, e)

    items.sort(key=lambda r: r.get("effective") or "", reverse=True)
    return {"items": items}


def _parse_lab_objective_line(line: str) -> dict | None:
    """Parse one SOAP-note objective line back into a structured dict.

    Inverse of the rendering loop in
    ``writer.write_lab_encounter_with_results``. Lines look like:

        HbA1c: 7.4 % (ref 4.0-5.6 %, abnormal flag H)
        LDL: 161 mg/dL (ref <100, abnormal flag H)
        Total Cholesterol: 232 mg/dL
        Glucose, qualitative: positive (ref negative)
        HDL: 38 mg/dL (ref >=40 (female), abnormal flag L)

    String-ops parsing rather than regex — the extras block can carry
    nested parens (`>=40 (female)`) which a flat `[^)]+` regex would
    truncate at the first inner `)`. Returns None when the line
    doesn't have the expected `name: value …` shape (free-text
    comments, blank lines, the writer's "(no extracted results)"
    placeholder).
    """
    line = (line or "").strip()
    if not line or ":" not in line:
        return None
    name, rest = line.split(":", 1)
    name = name.strip()
    rest = rest.strip()
    if not name or not rest:
        return None

    # Pull off the trailing extras block — between the FIRST `(` and the
    # LAST `)`, so nested parens inside the extras are preserved.
    extras_str = ""
    if rest.endswith(")") and "(" in rest:
        open_pos = rest.find("(")
        extras_str = rest[open_pos + 1:-1].strip()
        rest = rest[:open_pos].strip()

    # Now `rest` is "value [unit]" — first token is value, remainder
    # joined back together is the unit (handles compound units like
    # "x10^9/L"). A bare value with no unit is fine.
    parts = rest.split(None, 1)
    if not parts:
        return None
    value = parts[0]
    unit = parts[1].strip() if len(parts) > 1 else None
    if unit == "":
        unit = None

    ref = None
    flag = None
    for chunk in [c.strip() for c in extras_str.split(",") if c.strip()]:
        low = chunk.lower()
        if low.startswith("ref "):
            ref = chunk[4:].strip()
        elif low.startswith("abnormal flag "):
            flag = chunk[len("abnormal flag "):].strip() or None

    # Filter out the writer's empty-results sentinel.
    if name == "(no extracted results)":
        return None

    return {
        "test_name": name,
        "value": value,
        "unit": unit,
        "reference_range": ref,
        "abnormal_flag": flag,
    }


async def _backfill_lab_results_from_soap(
    *, patient_uuid: str, request: Request,
) -> int:
    """Populate ``extracted_lab_results`` for a patient from existing
    SOAP-note objective text written by prior lab-PDF uploads.

    OpenEMR's REST API has no Observation write surface, so the writer
    persists labs as SOAP-note objective text and a structured
    bbox-manifest in the subjective field. Earlier uploads (before the
    SQLite store existed) populated the SOAP notes but not the store,
    so the dashboard's Lab Results tab would still be empty for those
    patients.

    This function reads the encounters back via the standard REST API,
    pulls each lab encounter's SOAP note, parses the objective text
    into structured rows, and writes them to the store with the
    encounter's date as the collection_date (the manifest doesn't carry
    one). Idempotent: re-running for the same patient will overwrite
    any existing rows for the same source DocumentReference.

    Returns the number of result rows written. Best-effort: any
    error in reading encounters / SOAP notes logs and skips that
    encounter rather than aborting the whole backfill.
    """
    store = getattr(request.app.state, "extracted_lab_results", None)
    writer = getattr(request.app.state, "openemr_writer", None)
    if store is None or writer is None:
        return 0
    try:
        encounters = await writer.list_encounters(patient_uuid)
    except Exception as e:  # noqa: BLE001
        log.warning("backfill_lab_results: list_encounters failed for %s: %s", patient_uuid, e)
        return 0

    written_total = 0
    for enc in encounters:
        if not isinstance(enc, dict):
            continue
        # The writer sets reason="Lab results from uploaded document <ref>"
        # on every lab encounter it creates. Filter on that prefix so we
        # don't accidentally parse arbitrary office-visit notes as lab
        # lines.
        reason = str(enc.get("reason") or "")
        if not reason.lower().startswith("lab results"):
            continue
        eid_raw = enc.get("eid") or enc.get("id") or enc.get("encounter_id")
        try:
            eid = int(eid_raw) if eid_raw is not None else None
        except (TypeError, ValueError):
            eid = None
        if eid is None:
            continue
        # Recover the source DocumentReference id from the reason text
        # so the store row keys back to the original PDF (matches what
        # a fresh upload would produce). Fall back to a synthetic id
        # derived from the encounter id if the reason text is missing
        # the back-reference.
        source_doc_id = f"backfill-encounter-{eid}"
        # "Lab results from uploaded document DocumentReference/<uuid>"
        # — pull the trailing token, strip the prefix.
        tail = reason.split("uploaded document", 1)[1].strip() if "uploaded document" in reason else ""
        if tail and tail not in ("(unknown)",):
            source_doc_id = tail
        try:
            soap = await writer.get_soap_note(patient_uuid, eid)
        except Exception as e:  # noqa: BLE001
            log.warning("backfill_lab_results: get_soap_note failed for %s/%s: %s", patient_uuid, eid, e)
            continue
        if not soap:
            continue
        objective_text = str(soap.get("objective") or "")
        if not objective_text.strip():
            continue
        rows: list[dict] = []
        # Encounter date carries the lab draw date (writer sets it from
        # the first result's collection_date). Backfilled rows inherit it.
        enc_date = str(enc.get("date") or "")[:10] or None
        for raw_line in objective_text.splitlines():
            parsed = _parse_lab_objective_line(raw_line)
            if not parsed:
                continue
            parsed["collection_date"] = enc_date
            rows.append(parsed)
        if not rows:
            continue
        try:
            written = store.upsert_batch(
                patient_uuid=patient_uuid,
                source_doc_id=source_doc_id,
                rows=rows,
            )
            written_total += written
        except Exception as e:  # noqa: BLE001
            log.warning("backfill_lab_results: store write failed for %s: %s", patient_uuid, e)
    if written_total:
        log.info(
            "backfill_lab_results: wrote %d rows for patient=%s from %d encounters",
            written_total, patient_uuid, len(encounters),
        )
    return written_total


def _abnormal_flag_from_observation(o: dict) -> str | None:
    """Pull the V2-style abnormal flag (`H`, `L`, `N`, `C`, `HH`, `LL`)
    out of a FHIR Observation's `interpretation`, or return None when
    no interpretation is recorded."""
    for interp in o.get("interpretation") or []:
        for c in interp.get("coding") or []:
            code = c.get("code")
            if isinstance(code, str) and code in ("H", "L", "N", "C", "HH", "LL", "A"):
                return code
    return None


@app.get("/api/dashboard/patient/{pid}/orders")
async def dashboard_orders(
    pid: str, request: Request, username: str = Depends(current_user),
) -> dict:
    """Orders placed for the patient — `ServiceRequest`.

    Used for lab orders, imaging orders, referrals, procedure orders.
    Empty state is the default until OpenEMR begins surfacing these
    via FHIR (and/or until Co-Pilot's clinical-note flow learns to
    write ServiceRequests for orders the previous shift placed).
    """
    from app.fhir.adapter import _coded_display, _narrative_text  # noqa: PLC0415
    await _dashboard_assert_patient_in_panel(request, username, pid)
    try:
        rows = await request.app.state.fhir.search(
            "ServiceRequest", {"patient": pid, "_count": 100},
        )
    except Exception as e:  # noqa: BLE001
        # OpenEMR may not surface ServiceRequest depending on version;
        # fail soft so the empty state renders rather than the card
        # error-banner.
        log.warning("dashboard orders: ServiceRequest search failed: %s", e)
        rows = []
    items: list[dict] = []
    for r in rows:
        items.append({
            "id": r.get("id") or "",
            "display": _coded_display(r.get("code") or {}) or _narrative_text(r) or "(unspecified)",
            "category": _coded_display((r.get("category") or [{}])[0]) if r.get("category") else None,
            "status": r.get("status"),
            "intent": r.get("intent"),
            "priority": r.get("priority"),
            "authored": r.get("authoredOn"),
            "requester": (r.get("requester") or {}).get("display"),
        })
    items.sort(key=lambda r: r.get("authored") or "", reverse=True)
    return {"items": items}


# ─── chat ────────────────────────────────────────────────────────────────


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request, _user: str = Depends(current_user)) -> ChatResponse:
    session_id = req.session_id or str(uuid4())

    # Layer 2 (defense-in-depth) — pre-LLM jailbreak scrubber. R5 in the
    # system prompt is the prompt-level defense; this short-circuits the
    # known attack patterns BEFORE the LLM is even called. The attempt is
    # still recorded in the audit log so the dashboard shows it.
    jb_label = detect_jailbreak(req.message)
    if jb_label is not None:
        guard_trace = new_request_trace(
            session_id=session_id,
            username=_user,
            user_msg=req.message,
            model=active_model_label(MODEL_NAME),
        )
        guard_trace.error = f"jailbreak_blocked:{jb_label}"
        guard_trace.finalize()
        app.state.traces.add(guard_trace)
        log.warning("jailbreak blocked at input layer: pattern=%s session=%s", jb_label, session_id)
        sess = SESSIONS.get((_user, session_id), {})
        return ChatResponse(
            session_id=session_id,
            response=JAILBREAK_REFUSAL,
            patient_id=sess.get("patient_id"),
            sources=sess.get("conversation_sources", []),
            validation_warning=False,
        )

    # Deterministic intent router — handles trivial messages (greetings,
    # thanks, help) without an LLM call. Strict full-message anchoring
    # means anything more complex falls through to the agent graph below.
    routed = route_intent(req.message)
    if routed is not None:
        router_trace = new_request_trace(
            session_id=session_id,
            username=_user,
            user_msg=req.message,
            model="intent_router",
        )
        router_trace.error = f"routed:{routed.intent}"
        router_trace.finalize()
        app.state.traces.add(router_trace)
        log.info("intent routed: pattern=%s session=%s", routed.intent, session_id)
        sess = SESSIONS.get((_user, session_id), {})
        return ChatResponse(
            session_id=session_id,
            response=routed.response,
            patient_id=sess.get("patient_id"),
            sources=sess.get("conversation_sources", []),
            validation_warning=False,
        )

    state = SESSIONS.get((_user, session_id)) or _fresh_state()
    state["messages"] = [*state["messages"], HumanMessage(content=req.message)]
    state["validation_attempts"] = 0
    # Reset supervisor loop guard per turn — same shape as
    # `validation_attempts`. Without this, a long conversation accumulates
    # `route_count` and the next supervisor invocation trips the loop
    # guard immediately.
    state["route_count"] = 0
    state["worker_route"] = None
    state["username"] = _user
    state["advisor_mode"] = req.advisor_mode

    trace = new_request_trace(
        session_id=session_id,
        username=_user,
        user_msg=req.message,
        model=active_model_label(MODEL_NAME),
    )
    token = set_current_trace(trace)
    try:
        # DoS-B: per-turn wall-clock cap. `asyncio.wait_for` cancels
        # the graph task on breach; the per-LLM-call timeout (60s on
        # the answerer chain) only bounds individual calls, not the
        # sum, so a long sequence of expensive calls could otherwise
        # hold the worker thread beyond the operating envelope.
        result = await asyncio.wait_for(
            app.state.graph.ainvoke(
                state,
                config={"callbacks": [TokenUsageCallback()]},
            ),
            timeout=settings.max_turn_wall_seconds,
        )
    except asyncio.TimeoutError:
        trace.error = f"timeout:wall_seconds>{settings.max_turn_wall_seconds}"
        trace.finalize()
        app.state.traces.add(trace)
        reset_current_trace(token)
        raise HTTPException(
            status_code=504,
            detail=(
                f"This turn took longer than the {settings.max_turn_wall_seconds}s "
                "per-turn budget and was cancelled. Try a more specific question."
            ),
        ) from None
    except TurnBudgetExceeded as e:
        # DoS-A: per-turn token / cost cap breached mid-graph.
        trace.error = f"budget:{e.reason}"
        trace.finalize()
        app.state.traces.add(trace)
        reset_current_trace(token)
        raise HTTPException(
            status_code=429,
            detail=(
                f"This turn exceeded the per-turn budget ({e.reason}; "
                f"total_tokens={e.total_tokens}, cost_usd={e.total_cost_usd:.4f}) "
                "and was cancelled. Try a more specific question."
            ),
        ) from None
    except Exception as e:  # noqa: BLE001
        trace.error = f"{type(e).__name__}: {e}"
        trace.finalize()
        app.state.traces.add(trace)
        reset_current_trace(token)
        raise

    new_state: AgentState = {
        "messages": result["messages"],
        "conversation_sources": result["conversation_sources"],
        "patient_id": result.get("patient_id"),
        "validation_attempts": result.get("validation_attempts", 0),
        "username": _user,
        "advisor_mode": req.advisor_mode,
        "worker_route": result.get("worker_route"),
        "route_count": result.get("route_count", 0),
    }
    SESSIONS[(_user, session_id)] = new_state

    last = new_state["messages"][-1]
    text = message_text(last) if isinstance(last, AIMessage) else ""

    trace.validator_attempts = new_state["validation_attempts"]
    trace.validator_failed = new_state["validation_attempts"] >= MAX_VALIDATION_ATTEMPTS
    trace.route_count = new_state["route_count"]
    trace.conversation_sources = list(new_state["conversation_sources"])
    trace.finalize()
    app.state.traces.add(trace)
    reset_current_trace(token)

    return ChatResponse(
        session_id=session_id,
        response=text,
        patient_id=new_state["patient_id"],
        sources=new_state["conversation_sources"],
        validation_warning=new_state["validation_attempts"] >= MAX_VALIDATION_ATTEMPTS,
    )


@app.post("/chat/stream")
async def chat_stream(
    req: ChatRequest,
    request: Request,
    _user: str = Depends(current_user),
) -> StreamingResponse:
    """SSE-streaming version of /chat.

    Emits four event kinds, all as `data: {json}\\n\\n` lines:
      - {type:'session', session_id} once at start
      - {type:'tool', name, args} when a tool fires (lets UI show "Searching chart...")
      - {type:'token', text} token chunks as the final LLM response streams
      - {type:'done', patient_id, sources, validation_warning} once at end
    """
    session_id = req.session_id or str(uuid4())

    # Layer 2 (defense-in-depth) — pre-LLM jailbreak scrubber. Mirror of
    # the check in /chat above, but emits the refusal via SSE so the UI
    # treats it the same as any other streamed answer.
    jb_label = detect_jailbreak(req.message)
    if jb_label is not None:
        guard_trace = new_request_trace(
            session_id=session_id,
            username=_user,
            user_msg=req.message,
            model=active_model_label(MODEL_NAME),
        )
        guard_trace.error = f"jailbreak_blocked:{jb_label}"
        guard_trace.finalize()
        app.state.traces.add(guard_trace)
        log.warning("jailbreak blocked at input layer (stream): pattern=%s session=%s", jb_label, session_id)
        sess = SESSIONS.get((_user, session_id), {})

        async def guard_stream():
            yield f"data: {json.dumps({'type': 'session', 'session_id': session_id, 'request_id': guard_trace.request_id})}\n\n"
            yield f"data: {json.dumps({'type': 'token', 'text': JAILBREAK_REFUSAL})}\n\n"
            done_payload = {
                "type": "done",
                "patient_id": sess.get("patient_id"),
                "sources": sess.get("conversation_sources", []),
                "validation_warning": False,
                "request_id": guard_trace.request_id,
                "blocked_by_guard": jb_label,
            }
            yield f"data: {json.dumps(done_payload)}\n\n"

        return StreamingResponse(
            guard_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # Deterministic intent router — same logic as /chat above, but emits
    # the canned response over SSE so the UI handles it identically to
    # any streamed answer.
    routed = route_intent(req.message)
    if routed is not None:
        router_trace = new_request_trace(
            session_id=session_id,
            username=_user,
            user_msg=req.message,
            model="intent_router",
        )
        router_trace.error = f"routed:{routed.intent}"
        router_trace.finalize()
        app.state.traces.add(router_trace)
        log.info("intent routed (stream): pattern=%s session=%s", routed.intent, session_id)
        sess = SESSIONS.get((_user, session_id), {})

        async def routed_stream():
            yield f"data: {json.dumps({'type': 'session', 'session_id': session_id, 'request_id': router_trace.request_id})}\n\n"
            yield f"data: {json.dumps({'type': 'token', 'text': routed.response})}\n\n"
            done_payload = {
                "type": "done",
                "patient_id": sess.get("patient_id"),
                "sources": sess.get("conversation_sources", []),
                "validation_warning": False,
                "request_id": router_trace.request_id,
                "routed_intent": routed.intent,
            }
            yield f"data: {json.dumps(done_payload)}\n\n"

        return StreamingResponse(
            routed_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    state = SESSIONS.get((_user, session_id)) or _fresh_state()
    state["messages"] = [*state["messages"], HumanMessage(content=req.message)]
    state["validation_attempts"] = 0
    state["route_count"] = 0
    state["worker_route"] = None
    state["username"] = _user
    state["advisor_mode"] = req.advisor_mode

    trace = new_request_trace(
        session_id=session_id,
        username=_user,
        user_msg=req.message,
        model=active_model_label(MODEL_NAME),
    )

    async def event_stream():
        token = set_current_trace(trace)
        # Count answerer invocations this turn. The validator can send a
        # `VALIDATION FAILED:` retry message back to the answerer when it
        # finds uncited or fake-cited claims — that re-runs the answerer
        # and produces a *new* response, NOT a continuation of the first.
        # If we stream both, the user sees two complete answers
        # concatenated (first answer + footer + second answer's preamble
        # + same data again). Track invocations so we can emit a `reset`
        # event to the frontend on retry; the frontend then discards the
        # partial first-attempt text and starts fresh on the second
        # attempt's tokens.
        answerer_starts = 0
        try:
            yield f"data: {json.dumps({'type': 'session', 'session_id': session_id, 'request_id': trace.request_id})}\n\n"

            # DoS-B: per-turn wall-clock cap on the streaming graph
            # work. Wraps just the event-iteration loop so the post-
            # loop trace-finalization isn't subject to the cap. On
            # breach we emit an `error` SSE event and a `done` event
            # so the client tears down cleanly; the outer `finally`
            # still runs to finalize and store the trace.
            try:
                async with asyncio.timeout(settings.max_turn_wall_seconds):
                    async for event in app.state.graph.astream_events(
                        state,
                        version="v2",
                        config={"callbacks": [TokenUsageCallback()]},
                    ):
                        ev = event.get("event")
                        data = event.get("data") or {}
                        md = event.get("metadata") or {}

                        if (
                            ev == "on_chain_start"
                            and event.get("name") == "answerer"
                            and md.get("langgraph_node") == "answerer"
                        ):
                            # Second+ answerer invocation in a single turn ⇒ the
                            # validator retried. Tell the frontend to discard the
                            # partial first-attempt text. The fresh stream of
                            # tokens that follows will populate the bubble cleanly.
                            # We require BOTH event.name == "answerer" and the
                            # langgraph_node metadata match because nested LLM
                            # chains inside the answerer node inherit the
                            # langgraph_node metadata but have their own event
                            # name (e.g. "ChatAnthropic"). Counting nested starts
                            # would over-trigger the reset.
                            if answerer_starts >= 1:
                                yield f"data: {json.dumps({'type': 'reset'})}\n\n"
                            answerer_starts += 1
                            continue

                        if ev == "on_tool_start":
                            yield f"data: {json.dumps({'type': 'tool', 'name': event.get('name'), 'phase': 'start'})}\n\n"
                        elif ev == "on_chat_model_stream":
                            # Only stream tokens from the answerer node. The
                            # supervisor + worker LLMs (intake_extractor,
                            # evidence_retriever) also fire on_chat_model_stream
                            # events, and when a worker decides to emit prose
                            # instead of a tool call the worker text leaks into
                            # the SSE stream alongside the answerer's final text.
                            # The non-stream /chat endpoint isn't affected
                            # because it returns messages[-1] only.
                            node = md.get("langgraph_node")
                            if node != "answerer":
                                continue
                            chunk = data.get("chunk")
                            if chunk is not None and isinstance(chunk.content, str) and chunk.content:
                                yield f"data: {json.dumps({'type': 'token', 'text': chunk.content})}\n\n"
                            elif chunk is not None and isinstance(chunk.content, list):
                                for blk in chunk.content:
                                    if isinstance(blk, dict) and blk.get("type") == "text":
                                        txt = blk.get("text", "")
                                        if txt:
                                            yield f"data: {json.dumps({'type': 'token', 'text': txt})}\n\n"
                        elif ev == "on_chain_end" and event.get("name") == "LangGraph":
                            output = data.get("output") or {}
                            if isinstance(output, dict) and "messages" in output:
                                msgs = output["messages"]
                                if msgs:
                                    new_state: AgentState = {
                                        "messages": msgs,
                                        "conversation_sources": output.get("conversation_sources", []),
                                        "patient_id": output.get("patient_id"),
                                        "validation_attempts": output.get("validation_attempts", 0),
                                        "username": _user,
                                        "advisor_mode": req.advisor_mode,
                                        "worker_route": output.get("worker_route"),
                                        "route_count": output.get("route_count", 0),
                                    }
                                    SESSIONS[(_user, session_id)] = new_state
            except asyncio.TimeoutError:
                trace.error = f"timeout:wall_seconds>{settings.max_turn_wall_seconds}"
                err_payload = {
                    "type": "error",
                    "detail": (
                        f"This turn took longer than the "
                        f"{settings.max_turn_wall_seconds}s per-turn budget "
                        "and was cancelled. Try a more specific question."
                    ),
                }
                yield f"data: {json.dumps(err_payload)}\n\n"
                _sess_to = SESSIONS.get((_user, session_id), {})
                done_payload = {
                    "type": "done",
                    "patient_id": _sess_to.get("patient_id"),
                    "sources": _sess_to.get("conversation_sources", []),
                    "validation_warning": False,
                    "request_id": trace.request_id,
                    "timed_out": True,
                }
                yield f"data: {json.dumps(done_payload)}\n\n"
                return
            except TurnBudgetExceeded as e:
                # DoS-A: per-turn token / cost cap breached mid-graph.
                trace.error = f"budget:{e.reason}"
                err_payload = {
                    "type": "error",
                    "detail": (
                        f"This turn exceeded the per-turn budget "
                        f"({e.reason}; total_tokens={e.total_tokens}, "
                        f"cost_usd={e.total_cost_usd:.4f}) and was cancelled. "
                        "Try a more specific question."
                    ),
                }
                yield f"data: {json.dumps(err_payload)}\n\n"
                _sess_be = SESSIONS.get((_user, session_id), {})
                done_payload = {
                    "type": "done",
                    "patient_id": _sess_be.get("patient_id"),
                    "sources": _sess_be.get("conversation_sources", []),
                    "validation_warning": False,
                    "request_id": trace.request_id,
                    "budget_exceeded": True,
                }
                yield f"data: {json.dumps(done_payload)}\n\n"
                return

            sess = SESSIONS.get((_user, session_id), {})
            trace.validator_attempts = sess.get("validation_attempts", 0)
            trace.validator_failed = trace.validator_attempts >= MAX_VALIDATION_ATTEMPTS
            trace.route_count = sess.get("route_count", 0)
            trace.conversation_sources = list(sess.get("conversation_sources", []))
            done_payload = {
                "type": "done",
                "patient_id": sess.get("patient_id"),
                "sources": sess.get("conversation_sources", []),
                "validation_warning": trace.validator_failed,
                "request_id": trace.request_id,
            }
            yield f"data: {json.dumps(done_payload)}\n\n"
        except Exception as e:  # noqa: BLE001
            trace.error = f"{type(e).__name__}: {e}"
            raise
        finally:
            trace.finalize()
            app.state.traces.add(trace)
            reset_current_trace(token)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ─── data endpoints (all gated by session) ───────────────────────────────


async def _vital_trends_compute(patient_id: str) -> dict:
    """Pull raw FHIR observations + clinical-note vitals into the trends shape.

    Cached separately from the patient card because the card stays stable on
    a chart that wasn't touched, while trends flip every time a doctor
    finalizes a clinical note.
    """
    observations = await app.state.fhir.search(
        "Observation",
        {"patient": patient_id, "category": "vital-signs", "_count": 50},
    )
    notes = [n.to_doc_item() for n in app.state.clinical_notes.list_for_patient(patient_id, now=now_utc())]
    trends = collect_vital_trends(observations, notes)
    return {"current": latest_per_vital(trends), "trends": trends}


_VITAL_NAME_TO_KEY = {
    "heart rate": "heart_rate", "pulse": "heart_rate",
    "respiratory rate": "respiratory_rate", "resp rate": "respiratory_rate",
    "body temperature": "temp_f", "temperature": "temp_f",
    "oxygen_saturation": "spo2", "oxygen saturation": "spo2", "spo2": "spo2",
    "systolic blood pressure": "bp_systolic", "diastolic blood pressure": "bp_diastolic",
    # Short forms — match `_BP_LABEL` ("Systolic BP"/"Diastolic BP") that
    # `_decorate_card_vitals` emits when backfilling composite-BP rows from
    # the trends store. Without these the dedup check fails and BP renders
    # twice on the card.
    "systolic": "bp_systolic", "diastolic": "bp_diastolic",
}


_BP_LABEL = {"bp_systolic": "Systolic BP", "bp_diastolic": "Diastolic BP"}


def _decorate_card_vitals(card_data: dict, notes: list, trends: dict) -> dict:
    """Attach clinical-note provenance to FHIR vital observations and inject
    any unsynced note readings as additional rows.

    The FHIR adapter's `_format_vital` flattens an Observation by reading
    valueQuantity directly — that drops BP because OpenEMR returns it as a
    composite (panel with two `component` entries). We pull BP rows out of
    the already-parsed trends data so they show up in the list.

    Returns a new dict with:
      - recent_vitals: FHIR rows, each optionally carrying a ``from_note``
        block (author, finalized_at, note_id) when the reading was written
        by a finalized clinical note we already pushed to the EHR.
      - ``note_only_vitals``: readings from notes whose FHIR push failed
        (or hasn't happened yet) — kept so they still surface on the card.
    """
    from datetime import datetime

    from app.fhir.adapter import _clinical_iso

    fhir_rows = list(card_data.get("recent_vitals") or [])

    # Backfill BP rows that the flat-formatter dropped because the
    # underlying Observation was a composite. Times are re-stamped in
    # clinical-local TZ so this fallback path doesn't disagree with the
    # primary `_format_vital` rows (which are already clinical-local) on
    # the minute-key the frontend uses to group readings.
    #
    # Skip the backfill entirely when the snapshot already has a
    # combined "Blood Pressure" row (the path through
    # `_latest_vitals_snapshot`'s BP-merge pass) — adding the legacy
    # systolic+diastolic split rows on top would re-introduce the
    # double-row bug clinicians were complaining about.
    existing_names = {(r.get("name") or "").lower() for r in fhir_rows}
    bp_already_listed = (
        "blood pressure" in existing_names
        or any("systolic" in n or "diastolic" in n for n in existing_names)
    )
    if not bp_already_listed:
        for canonical in ("bp_systolic", "bp_diastolic"):
            for pt in (trends.get(canonical) or []):
                src = pt.get("source") or ""
                if not src.startswith("Observation/"):
                    continue
                fhir_rows.append({
                    "id": src.split("/", 1)[1],
                    "name": _BP_LABEL[canonical],
                    "value": pt.get("value"),
                    "unit": pt.get("unit"),
                    "time": _clinical_iso(pt.get("date")),
                })

    synced_notes = [n for n in notes if n.fhir_synced_at]
    unsynced_finals = [n for n in notes if n.status == "final" and not n.fhir_synced_at]

    # Pair FHIR vitals to synced notes by time proximity (±5 min).
    if synced_notes and fhir_rows:
        note_pairs = [(datetime.fromisoformat(n.finalized_at), n) for n in synced_notes if n.finalized_at]
        for v in fhir_rows:
            t = v.get("time")
            if not t:
                continue
            try:
                v_ts = datetime.fromisoformat(t.replace("Z", "+00:00") if "Z" in t else t)
            except (ValueError, TypeError):
                continue
            for note_ts, n in note_pairs:
                if abs((v_ts - note_ts).total_seconds()) <= 300:
                    v["from_note"] = {
                        "author": n.author,
                        "finalized_at": n.finalized_at,
                        "note_id": n.id,
                    }
                    break

    # Build a (canonical_key, timestamp) coverage list from FHIR rows. We
    # match note-vitals against this with ±5 min tolerance — same window as
    # the from_note pairing above — so a cross-minute clock skew between the
    # note's `finalized_at` and OpenEMR's stored `effectiveDateTime` doesn't
    # cause the same reading to render twice.
    fhir_covered: list[tuple[str, datetime]] = []
    for r in fhir_rows:
        nm = (r.get("name") or "").lower()
        val = r.get("value")
        t = r.get("time")
        if val in (None, "") or not t:
            continue
        try:
            ts = datetime.fromisoformat(t.replace("Z", "+00:00") if "Z" in t else t)
        except (ValueError, TypeError):
            continue
        for fragment, key in _VITAL_NAME_TO_KEY.items():
            if fragment in nm:
                fhir_covered.append((key, ts))
                break

    def _already_in_fhir(canonical: str, when_iso: str | None) -> bool:
        if not when_iso:
            return False
        try:
            note_ts = datetime.fromisoformat(when_iso.replace("Z", "+00:00") if "Z" in when_iso else when_iso)
        except (ValueError, TypeError):
            return False
        return any(
            k == canonical and abs((ts - note_ts).total_seconds()) <= 300
            for k, ts in fhir_covered
        )

    # Surface ALL finalized-note vitals (synced + unsynced) that aren't
    # already represented in FHIR rows. Critical for composite-BP readings:
    # OpenEMR returns BP as one Observation with two `component` entries, and
    # the FHIR adapter's flat formatter emits it with `value=None`, so the
    # client filters it out. The note's vitals dict is the doctor's source of
    # truth either way — surface it regardless of FHIR roundtrip status.
    #
    # Dedupe by `(canonical, time-bucket)` so two clinical notes that
    # happen to record the same vitals at the same minute (e.g. two
    # shift handovers a minute apart with the same observed BP) only
    # contribute ONE row per vital, not two. Without this dedup the
    # patient card shows duplicate "Blood Pressure 141/87" lines etc.
    note_only_vitals: list[dict] = []
    seen_note_keys: set[tuple[str, str]] = set()
    for n in [*unsynced_finals, *synced_notes]:
        # Use clinical-local TZ to match `fhir_rows[].time` so the frontend's
        # minute-precision group key lines up across both sources. Without
        # this conversion, a synced note whose FHIR roundtrip dropped a key
        # would surface here in UTC and split into a separate group on the
        # card — and the slice-to-1 view would render only one of them.
        when = _clinical_iso(n.finalized_at or n.updated_at)
        when_bucket = (when or "")[:16]  # YYYY-MM-DDTHH:MM
        for canonical, value in (n.vitals or {}).items():
            if value in (None, ""):
                continue
            if _already_in_fhir(canonical, when):
                continue
            dedup_key = (canonical, when_bucket)
            if dedup_key in seen_note_keys:
                continue
            seen_note_keys.add(dedup_key)
            note_only_vitals.append({
                "kind": "note-vital",
                "key": canonical,
                "value": value,
                "time": when,
                "from_note": {
                    "author": n.author,
                    "finalized_at": n.finalized_at,
                    "note_id": n.id,
                },
                "synced": bool(n.fhir_synced_at),
            })

    return {
        **card_data,
        "recent_vitals": fhir_rows,
        "note_only_vitals": note_only_vitals,
    }


async def _check_patient_access(
    request: Request, username: str, patient_id: str,
) -> None:
    """Raise 404 if `username` cannot access `patient_id`. Audit-log the denial.

    404 (not 403) is intentional: a 403 confirms the patient exists, which
    leaks chart-roster information across panel boundaries. 404 matches the
    "no patient found" response a typo would get.
    """
    panel = await access_control.get_panel_for_user(
        request.app.state.fhir, username, request.app.state.assignments,
    )
    if access_control.is_in_panel(panel, patient_id):
        return
    request.app.state.auth_store.log_event(
        event_type="patient_access_denied",
        username=username,
        sid=request.session.get("sid"),
        user_agent=request.headers.get("user-agent"),
        ip=request.client.host if request.client else None,
        detail=f"patient={patient_id}",
    )
    log.warning("acl: user=%s denied patient=%s", username, patient_id)
    raise HTTPException(status_code=404, detail="Patient not found")


@app.get("/api/patient/{patient_id}/card")
async def patient_card(
    patient_id: str,
    request: Request,
    username: str = Depends(current_user),
) -> dict:
    await _check_patient_access(request, username, patient_id)
    async def _compute() -> dict:
        result = await adapter.get_patient_card(app.state.fhir, patient_id=patient_id)
        return result["data"]
    try:
        card_data = await app.state.cache.get_or_compute(f"card:{patient_id}", _compute)
        trends_data = await app.state.cache.get_or_compute(
            f"trends:{patient_id}",
            lambda: _vital_trends_compute(patient_id),
        )
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"FHIR fetch failed: {e!s}") from e
    notes = app.state.clinical_notes.list_for_patient(patient_id, now=now_utc())
    decorated = _decorate_card_vitals(card_data, notes, trends_data["trends"])
    return {**decorated, "current_vitals": trends_data["current"]}


@app.put("/api/patient/{patient_id}/sex")
async def api_update_patient_sex(
    patient_id: str,
    request: Request,
    username: str = Depends(current_user),
) -> dict:
    """Set a patient's `sex` field. Used by the patient-header "Set sex"
    banner that surfaces when a patient was created from a document that
    didn't carry a sex value (so the create defaulted to "Unknown").

    Body: `{"sex": "Male" | "Female" | "Other" | "Unknown"}`. Returns
    `{"ok": true, "sex": "..."}` on success, 400 on invalid value, 502
    on OpenEMR rejection.
    """
    await _check_patient_access(request, username, patient_id)
    try:
        body = await request.json()
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"invalid JSON body: {e}") from e
    sex = (body or {}).get("sex")
    if sex not in ("Male", "Female", "Other", "Unknown"):
        raise HTTPException(
            status_code=400,
            detail=f"sex must be one of Male/Female/Other/Unknown, got {sex!r}",
        )
    ok = await app.state.openemr_writer.update_patient_sex(
        patient_uuid=patient_id, sex=sex,
    )
    if not ok:
        raise HTTPException(status_code=502, detail="sex update failed at OpenEMR")
    app.state.cache.invalidate(f"card:{patient_id}")
    return {"ok": True, "sex": sex}


@app.get("/api/patient/{patient_id}/vital-trends")
async def patient_vital_trends(
    patient_id: str,
    request: Request,
    username: str = Depends(current_user),
) -> dict:
    await _check_patient_access(request, username, patient_id)
    try:
        return await app.state.cache.get_or_compute(
            f"trends:{patient_id}",
            lambda: _vital_trends_compute(patient_id),
        )
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"FHIR fetch failed: {e!s}") from e


@app.get("/api/patient/{patient_id}/documents")
async def patient_documents(
    patient_id: str,
    request: Request,
    username: str = Depends(current_user),
    include_hidden: bool = False,
) -> dict:
    """Supporting-Documents tab payload.

    When `include_hidden=False` (default), DocumentReference rows that
    appear in the soft-hide store are stripped — same patient retention
    is preserved on the OpenEMR side, the doctor just gets a cleaner
    list. Pass `?include_hidden=true` to see them again with a `hidden:
    true` marker on each affected item so the UI can render them
    differently and offer an "unhide" action.
    """
    await _check_patient_access(request, username, patient_id)
    async def _compute() -> dict:
        result = await get_supporting_documents(app.state.fhir, patient_id=patient_id)
        return result["data"]
    try:
        data = await app.state.cache.get_or_compute(f"docs:{patient_id}", _compute)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"FHIR fetch failed: {e!s}") from e
    # Merge clinical notes (drafts + finals) alongside FHIR documents +
    # encounters. Re-sort the combined list by date so the most recent
    # item — usually the prior shift's clinical note — is on top.
    cn_items = [n.to_doc_item() for n in app.state.clinical_notes.list_for_patient(patient_id, now=now_utc())]
    items = list(data.get("items") or []) + cn_items
    items.sort(key=lambda x: x.get("date") or "", reverse=True)

    # Soft-hide pass — only DocumentReference items participate.
    # Encounters and clinical notes are out of scope for hide/unhide
    # (they are not user-uploaded). The set lookup is O(1) per item.
    hidden_ids = app.state.hidden_docs.list_hidden_ids()
    if hidden_ids:
        out: list[dict] = []
        for it in items:
            ref = it.get("ref") or ""
            if ref.startswith("DocumentReference/"):
                doc_id = ref.removeprefix("DocumentReference/")
                if doc_id in hidden_ids:
                    if include_hidden:
                        # Re-emit with a marker so the UI can grey it
                        # out and show an Unhide button.
                        merged = dict(it)
                        merged["hidden"] = True
                        out.append(merged)
                    # else: drop entirely
                    continue
            out.append(it)
        items = out

    # Background prefetch: kick off PNG renders + bbox manifests for the
    # most recent DocumentReference items in this list so a follow-up
    # citation click or "open document" lands on a warm cache and
    # returns instantly. Capped per patient and globally bounded by a
    # semaphore so a chart with many large PDFs can't pin every worker.
    # Fire-and-forget — failures are logged but never propagate.
    asyncio.create_task(
        _prefetch_doc_pages_for_patient(items, panel=None),
    )

    return {"items": items}


async def _prefetch_doc_pages_for_patient(
    items: list[dict], *, panel: frozenset[str] | None,
) -> None:
    """Pre-render and pre-summarize the top N DocumentReference items in a
    docs-list response so the next click on the citation deep-link path
    is instant.

    Skips items already in `pages_cache` (per-key locking inside
    `get_or_compute` would coalesce duplicate work anyway, but skipping
    early avoids a no-op task spawn). Hidden docs are excluded by the
    caller before we get here.
    """
    cache: TTLCache = app.state.pages_cache
    targets: list[str] = []
    for it in items:
        ref = it.get("ref") or ""
        if not ref.startswith("DocumentReference/"):
            continue
        if it.get("hidden"):
            continue
        doc_id = ref.removeprefix("DocumentReference/")
        if cache.get(f"pages:{doc_id}") is not None:
            continue
        targets.append(doc_id)
        if len(targets) >= _PAGES_PREFETCH_PER_PATIENT_CAP:
            break
    if not targets:
        return

    async def _one(doc_id: str) -> None:
        async with _PAGES_PREFETCH_SEMAPHORE:
            # Warm the docpid cache FIRST. The pages and manifest
            # endpoints both gate on `_get_doc_patient_id`, which on a
            # cold cache fetches the DocumentReference metadata (~1s
            # against the local docker FHIR, several against Hetzner).
            # If the user clicks during that window we'd pay the
            # latency on the user-triggered request even though the
            # PNG render is already cached. Warming docpid first
            # collapses that window to a hash lookup.
            try:
                await _get_doc_patient_id(app.state.fhir, cache, doc_id)
            except Exception as e:  # noqa: BLE001
                log.debug("docpid prefetch skipped for %s: %s", doc_id, e)
            try:
                await cache.get_or_compute(
                    f"pages:{doc_id}",
                    lambda: _prefetch_pages_compute(doc_id, panel),
                )
            except Exception as e:  # noqa: BLE001 — best-effort
                log.debug("pages prefetch skipped for %s: %s", doc_id, e)
            try:
                await cache.get_or_compute(
                    f"bbox:{doc_id}",
                    lambda: _prefetch_bbox_compute(doc_id, panel),
                )
            except Exception as e:  # noqa: BLE001
                log.debug("bbox prefetch skipped for %s: %s", doc_id, e)

    await asyncio.gather(*[_one(d) for d in targets], return_exceptions=True)


async def _prefetch_pages_compute(
    document_id: str, panel: frozenset[str] | None,
) -> dict:
    result = await adapter.get_document_pages(
        app.state.fhir, document_id=document_id, panel=panel,
    )
    return result["data"]


async def _prefetch_bbox_compute(
    document_id: str, panel: frozenset[str] | None,
) -> dict:
    # Pass the sidecar store so the prefetched manifest already
    # includes any allergies / backfilled rows. Without this, the
    # cached bbox-manifest would be missing those entries and only
    # the user-triggered fetch would see them.
    result = await adapter.get_document_bbox_manifest(
        app.state.fhir, document_id=document_id, panel=panel,
        store=app.state.extracted_sources,
    )
    return result["data"]


# ─── document upload (Phase 2.4 — minimum viable user-facing surface) ──


@app.get("/api/upload/patients")
async def api_upload_patients(
    username: str = Depends(current_user),
) -> dict:
    """Patient roster the upload form uses to populate its patient dropdown.

    Returns only patients the current user is allowed to write to (via the
    same panel ACL the chat uses). Admins see every patient. Output shape
    is `{items: [{id, label}]}` sorted by label so the dropdown renders
    in a stable order across requests.
    """
    panel = await access_control.get_panel_for_user(
        app.state.fhir, username, app.state.assignments,
    )
    try:
        # 200 covers every demo / dev OpenEMR; production deploys would
        # paginate the dropdown, but for MVP a single page is fine.
        patients = await app.state.fhir.search("Patient", {"_count": "200"})
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"FHIR fetch failed: {e!s}") from e
    patients = access_control.filter_active(
        patients, dynamic_store=app.state.active_patients,
    )
    items: list[dict] = []
    for p in patients:
        pid = p.get("id")
        if not pid:
            continue
        if not access_control.is_in_panel(panel, pid):
            continue
        names = p.get("name") or []
        family = ""
        given = ""
        if isinstance(names, list) and names:
            n0 = names[0] if isinstance(names[0], dict) else {}
            family = str(n0.get("family") or "")
            given_list = n0.get("given") or []
            if isinstance(given_list, list) and given_list:
                given = str(given_list[0] or "")
        label = (
            f"{family}, {given}" if family and given
            else family or given or pid
        )
        items.append({"id": pid, "label": label})
    items.sort(key=lambda i: i["label"].lower())
    return {"items": items}


@app.get("/api/upload/doc-types")
async def api_upload_doc_types(
    _user: str = Depends(current_user),
) -> dict:
    """Doc-type values the upload form uses to populate its type dropdown.

    Output shape mirrors `/api/upload/patients` so the form's dropdown
    JS can use one rendering function for both selects.
    """
    return {
        "items": [
            {"id": value, "label": label}
            for value, label in DOC_TYPE_LABELS.items()
        ],
    }


@app.post("/api/upload/extract-demographics")
async def api_extract_demographics(
    file: UploadFile,
    doc_type: str = Form(...),
    _user: str = Depends(current_user),
) -> dict:
    """Run intake-form extraction on an uploaded file WITHOUT persisting.

    Used by the "create new patient from this document" flow in the upload
    UI: the doctor selects a file before picking a patient, the frontend
    POSTs here to pull demographics out of the file, and the user gets a
    populated form they confirm or edit before clicking "Create patient."

    All five extracted doc-types are supported now:

    - **intake_form**: required Demographics on the schema — every
      field populated when on the form.
    - **workbook**: deterministic openpyxl parse; `patient_name` is one
      string the helper splits into given/family.
    - **hl7_message**: deterministic PID-5/7/8/11/13 parse.
    - **fax_packet** / **referral_letter**: vision-extracted optional
      `patient_identity` — populated when the face sheet / letter
      header carries the data, `null` otherwise.

    Lab PDFs (`lab_pdf`) still 400 — the lab report header carries the
    patient name/DOB but the LabReport schema deliberately doesn't
    capture them as structured data, and rebuilding extraction for an
    edge case isn't worth the LLM cost.

    Returns: `{"demographics": {given_name, family_name, date_of_birth,
    sex, address, phone}, "chief_concern": str | null}`. Any field can
    be null when the source document didn't print it; the user fills in
    the rest on the create-patient form.
    """
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="empty file upload")
    raw_ct = (file.content_type or "").lower()
    filename = file.filename or "upload.bin"
    if not raw_ct or raw_ct == "application/octet-stream":
        fl = filename.lower()
        if fl.endswith(".docx"):
            raw_ct = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        elif fl.endswith(".pdf"):
            raw_ct = "application/pdf"
        elif fl.endswith(".hl7"):
            raw_ct = "text/plain"
        elif fl.endswith(".tiff") or fl.endswith(".tif"):
            raw_ct = "image/tiff"
        elif fl.endswith(".xlsx"):
            raw_ct = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    mime_type = raw_ct or "application/pdf"

    if doc_type not in DOC_TYPE_LABELS:
        raise HTTPException(
            status_code=400,
            detail=f"unsupported doc_type {doc_type!r}; "
                   f"expected one of {list(DOC_TYPE_LABELS)}",
        )

    chief_concern: str | None = None
    identity: dict[str, str | None]

    # First-page (or first-2-pages on fax packets) cap on the auto-fill
    # render: identity reliably appears in the first 1–2 pages, and the
    # full upload step still re-renders every page with Sonnet for the
    # authoritative extraction. This trims auto-fill latency on
    # multi-page docs.
    _MAX_AUTOFILL_PAGES = {
        "intake_form": 1,
        "referral_letter": 1,
        "fax_packet": 2,
    }

    try:
        if doc_type == "hl7_message":
            from app.extraction.hl7 import parse_hl7_message  # noqa: PLC0415
            msg = parse_hl7_message(file_bytes, source_document_id="DocumentReference/preview")
            identity = _identity_from_optional(msg.patient_identity)
        elif doc_type == "workbook":
            from app.extraction.workbook import parse_workbook  # noqa: PLC0415
            wb = parse_workbook(file_bytes, source_document_id="DocumentReference/preview")
            given_name, family_name = _split_full_name(wb.patient_name)
            identity = {
                "given_name": given_name,
                "family_name": family_name,
                "date_of_birth": wb.patient_dob.isoformat() if wb.patient_dob else None,
                "sex": None,
                "address": None,
                "phone": None,
            }
        elif doc_type in ("intake_form", "fax_packet", "referral_letter"):
            # Vision-bound preview: render to PNG, cap pages, run a fast
            # Haiku extraction that targets just PatientIdentity. Doctor
            # reviews/corrects on the form before submit; full upload
            # re-extracts with Sonnet over every page.
            from app.extraction.render import render_to_png_pages  # noqa: PLC0415
            from app.extraction.vision import extract_patient_identity  # noqa: PLC0415
            try:
                pages = render_to_png_pages(file_bytes, mime_type)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e)) from e
            cap = _MAX_AUTOFILL_PAGES.get(doc_type, 1)
            pages = pages[:cap]
            ident = await extract_patient_identity(page_pngs=pages)
            identity = _identity_from_optional(ident)
        else:
            # lab_pdf and any future doc_type without a demographics
            # surface land here.
            raise HTTPException(
                status_code=400,
                detail=(
                    f"extract-demographics doesn't support doc_type={doc_type!r}; "
                    "the schema for that document type does not capture patient "
                    "identity. Pick a different file or create the patient via "
                    "the manual form."
                ),
            )
    except ValueError as e:
        # Render-side errors (unsupported MIME, empty file) and
        # Hl7ParseError (subclass of ValueError) both land here.
        raise HTTPException(status_code=400, detail=str(e)) from e
    except ExtractionError as e:
        raise HTTPException(
            status_code=502, detail=f"Document extraction failed: {e}",
        ) from e

    return {
        "demographics": identity,
        "chief_concern": chief_concern,
    }


def _identity_from_optional(p: object) -> dict[str, str | None]:
    """Project an optional `PatientIdentity` (or None) into the
    flat dict shape the create-patient UI expects."""
    given = family = dob = sex = address = phone = None
    if p is not None:
        given = getattr(p, "given_name", None)
        family = getattr(p, "family_name", None)
        dob_obj = getattr(p, "date_of_birth", None)
        dob = dob_obj.isoformat() if dob_obj else None
        sex = getattr(p, "sex", None)
        address = getattr(p, "address", None)
        phone = getattr(p, "phone", None)
    return {
        "given_name": given,
        "family_name": family,
        "date_of_birth": dob,
        "sex": sex,
        "address": address,
        "phone": phone,
    }


def _split_full_name(full: str | None) -> tuple[str | None, str | None]:
    """Best-effort split a single 'patient_name' string into
    `(given_name, family_name)`.

    Workbook authors print the name in either of two conventions:
    `"Last, First Middle"` (commas-separated, last-name-first — the
    typical EHR display) or `"First Middle Last"` (space-separated,
    Western convention). We disambiguate by looking for the comma; on
    a comma-split, the chunk before the comma is family, after is
    given. On a space-split, the LAST token is family and the
    earlier tokens collapse into given. Returns `(None, None)` for
    blank input."""
    if not full or not full.strip():
        return (None, None)
    s = full.strip()
    if "," in s:
        family, _, given = s.partition(",")
        return (given.strip() or None, family.strip() or None)
    parts = s.split()
    if len(parts) == 1:
        # A single token is ambiguous — return as the family name and
        # let the user disambiguate on the form. Family is the more
        # commonly-known half (a chart card showing only "Smith" is
        # informative; only "John" is not).
        return (None, parts[0])
    return (" ".join(parts[:-1]), parts[-1])


def _extracted_sex(extracted: object) -> str | None:
    """Pull a `sex` value out of any extraction shape, lower-cased.

    The four extraction variants that carry sex put it in one of two
    places: `IntakeForm.demographics.sex` (required-ish on intake forms)
    or `<X>.patient_identity.sex` (optional on referral_letter,
    fax_packet, hl7_message). Workbooks and lab reports have no sex
    field. Returns None when no sex is present in any expected slot.
    """
    if extracted is None:
        return None
    demo = getattr(extracted, "demographics", None)
    if demo is not None:
        s = getattr(demo, "sex", None)
        if s:
            return str(s).lower()
    ident = getattr(extracted, "patient_identity", None)
    if ident is not None:
        s = getattr(ident, "sex", None)
        if s:
            return str(s).lower()
    return None


def _normalize_name_case(s: str) -> str:
    """Title-case a name when it's fully uppercase; otherwise leave it alone.

    HL7 v2 senders conventionally print PID-5 in ALL CAPS ("NGUYEN^OLIVIA")
    by legacy convention, and that uppercased value flows through auto-fill
    → form fields → OpenEMR's database → calendar display. Rather than
    title-casing every name (which would mangle "McDonald" → "Mcdonald"),
    only normalize when the input has no lowercase letters at all — names
    that already carry case information stay untouched.
    """
    return s.title() if s and s.isupper() else s


class CreatePatientRequest(BaseModel):
    """Demographics body for `/api/admin/patient` (create-from-upload)."""
    given_name: str
    family_name: str
    date_of_birth: str | None = None  # ISO YYYY-MM-DD; required by OpenEMR for idempotency search
    sex: str | None = None  # one of "Male", "Female", "Other", "Unknown"
    street: str | None = None
    city: str | None = None
    state: str | None = None
    postal_code: str | None = None
    phone: str | None = None


@app.post("/api/admin/patient")
async def api_create_patient(
    body: CreatePatientRequest,
    username: str = Depends(current_user),
) -> dict:
    """Create a new Patient in OpenEMR and add them to the active-patients
    allowlist so they appear immediately on the dashboard.

    The create-from-upload flow calls this AFTER the doctor has reviewed
    the demographics extracted from a doc and clicked "Create." On
    success the response carries the new FHIR Patient UUID, which the
    frontend then feeds back into `/api/upload` to attach the original
    document to the freshly-minted patient.

    Auth: any logged-in user can create a patient (matches the seeded
    OpenEMR convention; the demo doesn't model nurse-vs-doctor roles).
    Audit trail: write_patient logs the create at the writer level;
    `username` is recorded as the `added_by` on the active-patients
    override so we can see who minted each runtime entry.
    """
    if not body.given_name.strip() or not body.family_name.strip():
        raise HTTPException(
            status_code=400,
            detail="given_name and family_name are required to create a patient",
        )
    given_clean = _normalize_name_case(body.given_name.strip())
    family_clean = _normalize_name_case(body.family_name.strip())
    # OpenEMR's standard `/api/patient` rejects creates without `sex`. When
    # the source document didn't print one (.docx with only an honorific,
    # workbook, etc.), default to "Unknown" so the patient is still created
    # and let the post-create banner prompt the doctor to fill it in.
    sex_was_inferred = False
    sex_for_create = (body.sex or "").strip()
    if not sex_for_create:
        sex_for_create = "Unknown"
        sex_was_inferred = True

    # Idempotent create — match the seed-script convention. Search FHIR for
    # an existing Patient with the same family + given + birthdate before
    # minting a new one. Without this guard, a doctor who re-runs the
    # create-from-upload flow on the same intake form (e.g. after a
    # network hiccup or because they hit "back" and resubmitted) ends up
    # with N orphan duplicates of the same person — exactly what
    # populated Hetzner with 4 "James Whitaker" rows during testing.
    try:
        params = {"family": family_clean, "given": given_clean, "_count": 5}
        if body.date_of_birth:
            params["birthdate"] = body.date_of_birth
        existing_rows = await app.state.fhir.search("Patient", params)
    except Exception as e:  # noqa: BLE001
        log.warning("create_patient: pre-check FHIR search failed: %s", e)
        existing_rows = []
    existing_match: dict | None = None
    for row in existing_rows:
        names = (row.get("name") or [{}])[0]
        rfam = (names.get("family") or "").strip().lower()
        rgiv = ""
        given_list = names.get("given") or []
        if isinstance(given_list, list) and given_list:
            rgiv = str(given_list[0]).strip().lower()
        if rfam == family_clean.lower() and rgiv == given_clean.lower():
            # Optional DOB check — match only if the existing row carries
            # the same birthDate (else accept any same-name row, since
            # the doctor explicitly told us this is the patient).
            if body.date_of_birth and row.get("birthDate") and row["birthDate"] != body.date_of_birth:
                continue
            existing_match = row
            break
    if existing_match is not None:
        new_uuid = existing_match.get("id")
        log.info(
            "create_patient: idempotent hit, reusing existing uuid=%s",
            new_uuid,
        )
        # Auto-assign + active-patients add still run below via the
        # shared post-create code path.
        result = {"uuid": new_uuid, "pid": None}
    else:
        try:
            result = await app.state.openemr_writer.write_patient(
                given_name=given_clean,
                family_name=family_clean,
                date_of_birth=body.date_of_birth,
                sex=sex_for_create,
                street=body.street,
                city=body.city,
                state=body.state,
                postal_code=body.postal_code,
                phone=body.phone,
            )
        except OpenEMRWriteError as e:
            raise HTTPException(
                status_code=502, detail=f"OpenEMR patient create failed: {e}",
            ) from e
    new_uuid = result.get("uuid")
    new_pid = result.get("pid")
    if not new_uuid:
        raise HTTPException(
            status_code=502,
            detail=f"OpenEMR did not return a uuid for the new patient: {result}",
        )
    # Add to runtime allow-list so the new patient surfaces immediately.
    # `add` is idempotent (PRIMARY KEY ON CONFLICT IGNORE) so the existing-
    # match path is harmless re-insert.
    app.state.active_patients.add(
        family=family_clean, given=given_clean, added_by=username,
    )
    # Auto-assign the new patient to the creating practitioner so the
    # immediately-following upload (and every subsequent chart fetch by
    # the same doctor) passes the panel ACL. Admins skip — they have no
    # Practitioner UUID and don't need a panel slot.
    if not is_admin(username):
        prac_id = await access_control.resolve_practitioner_id(
            app.state.fhir, username,
        )
        if prac_id:
            app.state.assignments.upsert(
                patient_id=new_uuid,
                practitioner_id=prac_id,
                assigned_by=username,
            )
            access_control.invalidate_panel(username)
            log.info(
                "create_patient: auto-assigned %s to practitioner=%s "
                "for user=%s",
                new_uuid, prac_id, username,
            )
        else:
            log.warning(
                "create_patient: could not resolve practitioner for user=%s; "
                "patient created but not auto-assigned (upload will 404)",
                username,
            )
    # Bust calendar caches so the dashboard re-fetches with the new
    # patient included on the next request.
    app.state.cache.invalidate_prefix("calendar:today:")
    log.info(
        "create_patient: user=%s pid=%s uuid=%s",
        username, new_pid, new_uuid,
    )
    return {
        "patient_uuid": new_uuid,
        "pid": new_pid,
        "given_name": given_clean,
        "family_name": family_clean,
        # When True, the document didn't carry a sex and the server defaulted
        # to "Unknown" to satisfy OpenEMR's required-field validator. The
        # frontend uses this to render a persistent "set sex" banner on the
        # patient header until the field is updated.
        "sex_was_inferred": sex_was_inferred,
    }


@app.post("/api/upload")
async def api_upload(
    request: Request,
    file: UploadFile,
    doc_type: str = Form(...),
    patient_uuid: str = Form(...),
    acknowledge_existing: bool = Form(False),
    username: str = Depends(current_user),
) -> dict:
    """Multipart upload endpoint: persist a clinical document to OpenEMR
    and return its structured-typed extraction.

    Form fields:
      file: the upload (PDF, PNG, or JPEG)
      doc_type: one of the DOC_TYPE_LABELS keys (lab_pdf, intake_form)
      patient_uuid: a FHIR Patient UUID the user has write access to
      acknowledge_existing: when True, accept that the file is already
        on the patient's chart and return its existing reference. The
        upload UI sets this on the second submission after the user
        clicks "Use existing" on the duplicate prompt. Default False
        means a SHA-256 dedup hit returns 409 instead of silently
        de-duplicating, so the user sees the prompt.

    Returns:
      {reference_id, sha256, created, extracted, persistence}

    Errors:
      400 — empty file, missing fields, unsupported doc_type
      404 — patient not in user's panel (prefer 404 over 403 to avoid
            leaking that the patient exists)
      409 — same file bytes are already on this patient's chart and
            `acknowledge_existing=False`. Body shape:
            {"detail": {"code": "duplicate_file", "message": ...,
                        "existing": {"ref","title","date","category"},
                        "sha256": "..."}}
      502 — OpenEMR write failure or extractor failure
    """
    if doc_type not in DOC_TYPE_LABELS:
        raise HTTPException(
            status_code=400,
            detail=f"unsupported doc_type {doc_type!r}; "
                   f"expected one of {list(DOC_TYPE_LABELS)}",
        )
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="empty file upload")
    # Run the panel-access check + SHA-256 dedup precheck concurrently —
    # both are FHIR round-trips (~300 ms each) and share no dependency.
    # The access check raises HTTPException(404) on miss; gather() will
    # propagate it as the first failed task and we'll discard the SHA
    # result. acknowledge_existing=True path skips the SHA check.
    sha_hex = hashlib.sha256(file_bytes).hexdigest() if not acknowledge_existing else None
    async def _sha_precheck() -> str | None:
        if acknowledge_existing or sha_hex is None:
            return None
        try:
            return await app.state.openemr_writer.find_document_reference_by_sha(
                patient_uuid, sha_hex,
            )
        except Exception as e:  # noqa: BLE001
            log.warning("dedup precheck failed; proceeding without it: %s", e)
            return None

    _, existing_ref = await asyncio.gather(
        _check_patient_access(request, username, patient_uuid),
        _sha_precheck(),
    )
    raw_ct = (file.content_type or "").lower()
    filename = file.filename or "upload.bin"
    # Some browsers send `application/octet-stream` for .docx / .hl7 /
    # .tiff / .xlsx instead of the dedicated MIME (HL7 v2 has no
    # widely-supported MIME at all; .tiff handling depends on the OS).
    # Fall back to the filename extension before handing off to the
    # renderer's MIME dispatch / HL7 dispatcher / workbook parser.
    if not raw_ct or raw_ct == "application/octet-stream":
        fl = filename.lower()
        if fl.endswith(".docx"):
            raw_ct = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        elif fl.endswith(".pdf"):
            raw_ct = "application/pdf"
        elif fl.endswith(".hl7"):
            raw_ct = "text/plain"
        elif fl.endswith(".tiff") or fl.endswith(".tif"):
            raw_ct = "image/tiff"
        elif fl.endswith(".xlsx"):
            raw_ct = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    mime_type = raw_ct or "application/octet-stream"

    # Dedup Layer 1 — SHA-256 prompt. The writer will idempotently
    # return the existing reference if we proceed; what's missing today
    # is the user-visible "you already uploaded this" surface. The SHA
    # precheck already ran in parallel with the access check above; we
    # only handle the result here. `existing_ref` is None when
    # acknowledge_existing was set or the precheck found no match.
    if existing_ref:
        existing_meta: dict = {"ref": existing_ref}
        # Best-effort: pull title + date for the prompt body. A 502
        # here would be silly — fall back to the bare ref so the UI
        # can still ask the user to confirm.
        try:
            doc_id_only = existing_ref.split("/", 1)[-1]
            existing_doc = await app.state.fhir.get(
                f"DocumentReference/{doc_id_only}",
            )
            attach = (existing_doc.get("content") or [{}])[0].get("attachment") or {}
            existing_meta.update({
                "title": attach.get("title") or "",
                "filename": attach.get("title") or "",
                "date": existing_doc.get("date"),
                "status": existing_doc.get("status"),
            })
        except Exception as e:  # noqa: BLE001
            log.info("dedup metadata fetch failed: %s", e)
        log.info(
            "upload dedup-prompt: user=%s patient=%s sha256=%s -> %s",
            username, patient_uuid, sha_hex[:12], existing_ref,
        )
        raise HTTPException(
            status_code=409,
            detail={
                "code": "duplicate_file",
                "message": (
                    "This file is already on the patient's chart. "
                    "Use the existing copy or cancel this upload."
                ),
                "existing": existing_meta,
                "sha256": sha_hex,
            },
        )

    try:
        # Skip persistence on the initial extract — we want to interpose
        # Layer-2 dedup (content-fingerprint match against prior docs
        # for this patient) BEFORE writing facts. The writer's Phase 1
        # has already committed the source DocumentReference to OpenEMR,
        # so the new doc id is real and stable; if Layer 2 prompts and
        # the user cancels, the upload endpoint's caller can soft-hide
        # the new doc via /api/document/{id}/resolve-content-match.
        if doc_type == "hl7_message":
            # HL7 v2 messages bypass the render+vision path entirely —
            # the parser walks segments deterministically (zero LLM cost).
            from app.extraction.extract import attach_and_extract_hl7  # noqa: PLC0415
            result = await attach_and_extract_hl7(
                file_bytes=file_bytes,
                filename=filename,
                patient_uuid=patient_uuid,
                writer=app.state.openemr_writer,
                skip_persistence=True,
            )
        elif doc_type == "workbook":
            # Workbooks (.xlsx) also bypass vision — openpyxl walks the
            # 4 sheets deterministically (zero LLM cost).
            from app.extraction.extract import attach_and_extract_workbook  # noqa: PLC0415
            result = await attach_and_extract_workbook(
                file_bytes=file_bytes,
                filename=filename,
                patient_uuid=patient_uuid,
                writer=app.state.openemr_writer,
                skip_persistence=True,
            )
        else:
            # Two-phase split so a Layer-1.5 text-fingerprint match can
            # short-circuit before paying for a Claude vision call.
            # Phase 1 (writer) runs unconditionally; Phase 2 (render +
            # extract_via_claude) is gated below.
            result = await attach_and_extract(
                file_bytes=file_bytes,
                filename=filename,
                doc_type=doc_type,  # type: ignore[arg-type]
                patient_uuid=patient_uuid,
                mime_type=mime_type,
                writer=app.state.openemr_writer,
                skip_extraction=True,
                skip_persistence=True,
                practitioners_store=app.state.extracted_practitioners,
                lab_results_store=app.state.extracted_lab_results,
            )
    except ValueError as e:
        # Render-side errors (unsupported MIME, empty file inside renderer)
        # and Hl7ParseError (subclass of ValueError) both land here.
        raise HTTPException(status_code=400, detail=str(e)) from e
    except OpenEMRWriteError as e:
        raise HTTPException(
            status_code=502, detail=f"OpenEMR write failed: {e}",
        ) from e
    except ExtractionError as e:
        raise HTTPException(
            status_code=502, detail=f"Document extraction failed: {e}",
        ) from e

    new_ref = result.reference_id
    new_doc_id_only = new_ref.split("/", 1)[-1]

    # Dedup Layer 1.5 — pre-extraction PDF text-layer fingerprint.
    # Runs only for non-HL7/non-workbook docs (vision-bound paths). For
    # text-bearing PDFs, hashing the embedded text layer lets us detect
    # a re-uploaded document with different bytes but identical visible
    # content and surface the dedup modal *without* spending Anthropic
    # credits on extraction. Pure scanned-image PDFs return None here
    # and fall through to Layer-2 (post-extraction structural match).
    text_fp_namespaced: str | None = None
    if (
        doc_type not in ("hl7_message", "workbook")
        and not acknowledge_existing
        and mime_type.lower() == "application/pdf"
    ):
        raw_text_fp = pdf_text_fingerprint(file_bytes)
        if raw_text_fp:
            text_fp_namespaced = namespace_text_fingerprint(raw_text_fp)
            text_match = app.state.fingerprints.find_match(
                patient_id=patient_uuid, fingerprint=text_fp_namespaced,
            )
            if text_match is not None and text_match.document_id != new_doc_id_only:
                log.info(
                    "upload text-match (L1.5): user=%s patient=%s "
                    "fp=%s prior=%s new=%s — skipping vision",
                    username, patient_uuid, raw_text_fp[:12],
                    text_match.document_id, new_doc_id_only,
                )
                prior_meta: dict = {
                    "ref": f"DocumentReference/{text_match.document_id}",
                }
                try:
                    prior_doc = await app.state.fhir.get(
                        f"DocumentReference/{text_match.document_id}",
                    )
                    prior_attach = (prior_doc.get("content") or [{}])[0].get("attachment") or {}
                    prior_meta.update({
                        "title": prior_attach.get("title") or "",
                        "filename": prior_attach.get("title") or "",
                        "date": prior_doc.get("date"),
                    })
                except Exception as e:  # noqa: BLE001
                    log.info("text-match prior metadata fetch failed: %s", e)
                # Same `content_match` shape Layer 2 raises — the
                # frontend modal already handles Replace / Keep both /
                # Cancel against the new doc id.
                raise HTTPException(
                    status_code=409,
                    detail={
                        "code": "content_match",
                        "message": (
                            "A different file with the same visible text "
                            "is already on this patient's chart. Choose "
                            "Replace, Keep both, or Cancel."
                        ),
                        "fingerprint": raw_text_fp,
                        "match_layer": "text",
                        "prior": prior_meta,
                        "new": {
                            "ref": new_ref,
                            "filename": filename,
                        },
                    },
                )

    # Phase 2 — render + extract. Deferred from `attach_and_extract`
    # above so the Layer-1.5 short-circuit can skip it. ValueError /
    # ExtractionError translate to the same 400/502 the orchestrator
    # would have produced.
    if doc_type not in ("hl7_message", "workbook"):
        try:
            page_pngs = render_to_png_pages(file_bytes, mime_type)
            extracted = await extract_via_claude(
                page_pngs=page_pngs,
                doc_type=doc_type,  # type: ignore[arg-type]
                source_document_id=new_ref,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except ExtractionError as e:
            raise HTTPException(
                status_code=502, detail=f"Document extraction failed: {e}",
            ) from e
        result.extracted = extracted

    # Dedup Layer 2 — content fingerprint match.
    # Compute the structural fingerprint of the extraction. If a prior
    # DocumentReference for this patient produced the same fingerprint
    # (and it's not the doc we just uploaded — sha-256 dedup handled
    # the bytes-identical case earlier), surface a 3-option modal so
    # the user can pick Replace / Keep both / Cancel before we persist
    # facts. The new doc stays in OpenEMR either way; cancel
    # soft-hides it via the resolve-content-match endpoint.
    fingerprint = compute_fingerprint(result.extracted)
    if fingerprint and not acknowledge_existing:
        match = app.state.fingerprints.find_match(
            patient_id=patient_uuid, fingerprint=fingerprint,
        )
        if match is not None and match.document_id != new_doc_id_only:
            log.info(
                "upload content-match: user=%s patient=%s fp=%s prior=%s new=%s",
                username, patient_uuid, fingerprint[:12],
                match.document_id, new_doc_id_only,
            )
            # Pull metadata for the prior doc so the modal can show
            # the user what they're choosing between.
            prior_meta: dict = {"ref": f"DocumentReference/{match.document_id}"}
            try:
                prior_doc = await app.state.fhir.get(
                    f"DocumentReference/{match.document_id}",
                )
                prior_attach = (prior_doc.get("content") or [{}])[0].get("attachment") or {}
                prior_meta.update({
                    "title": prior_attach.get("title") or "",
                    "filename": prior_attach.get("title") or "",
                    "date": prior_doc.get("date"),
                })
            except Exception as e:  # noqa: BLE001
                log.info("content-match prior metadata fetch failed: %s", e)
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "content_match",
                    "message": (
                        "A different file with the same extracted content "
                        "is already on this patient's chart. Choose "
                        "Replace, Keep both, or Cancel."
                    ),
                    "fingerprint": fingerprint,
                    "prior": prior_meta,
                    "new": {
                        "ref": new_ref,
                        "filename": filename,
                    },
                },
            )

    # No match (or user already acknowledged): persist facts now and
    # record the fingerprint so future uploads of the same content
    # surface the prompt.
    #
    # SHA-256 dedup guard: if the writer returned `created=False` the
    # uploaded bytes are byte-identical to a DocumentReference already
    # on the patient's chart, and Phase 3 (persist_extracted_facts) ran
    # on the original upload. Re-running it now would create a duplicate
    # lab Encounter / Observation / Condition row pointing at the same
    # source doc — the symptom Harrison saw on Hetzner where Kowalski
    # had three lab-encounter rows but only one visible lab document.
    if not result.created:
        log.info(
            "upload SHA-dedup: skipping Phase 3 persistence (doc already on file)"
            " ref=%s patient=%s",
            new_ref, patient_uuid,
        )
        persistence = None
    else:
        persistence = await persist_extracted_facts(
            writer=app.state.openemr_writer,
            patient_uuid=patient_uuid,
            extracted=result.extracted,
            source_document_id=new_ref,
            practitioners_store=app.state.extracted_practitioners,
            lab_results_store=app.state.extracted_lab_results,
        )
    result.persistence = persistence
    if fingerprint:
        # Use `record` so the first-seen fingerprint wins and a Replace
        # decision (handled by the resolve endpoint) overwrites the
        # row. acknowledge_existing on this branch means we already
        # know about the match and the user said keep going.
        existing_owner = app.state.fingerprints.find_match(
            patient_id=patient_uuid, fingerprint=fingerprint,
        )
        if existing_owner is None:
            app.state.fingerprints.record(
                patient_id=patient_uuid,
                fingerprint=fingerprint,
                document_id=new_doc_id_only,
                recorded_by=username,
            )
    # Layer-1.5 text fingerprint recording. Same first-seen-wins
    # semantics as the structural row above: only register when no
    # prior doc on this patient owns the same text-fingerprint, so a
    # subsequent upload of the same visible content prompts against
    # the canonical doc rather than rotating the canonical owner.
    if text_fp_namespaced:
        existing_text_owner = app.state.fingerprints.find_match(
            patient_id=patient_uuid, fingerprint=text_fp_namespaced,
        )
        if existing_text_owner is None:
            app.state.fingerprints.record(
                patient_id=patient_uuid,
                fingerprint=text_fp_namespaced,
                document_id=new_doc_id_only,
                recorded_by=username,
            )

    # Auto-update sex when an earlier upload created the patient with
    # sex="Unknown" because the source doc didn't print one, and this
    # later doc DOES carry a sex. Best-effort: a lookup or PUT failure
    # logs and moves on rather than failing the upload.
    extracted_sex = _extracted_sex(result.extracted)
    if extracted_sex:
        try:
            current_patient = await app.state.fhir.get(f"Patient/{patient_uuid}")
            current_gender = (current_patient.get("gender") or "").lower()
        except Exception as e:  # noqa: BLE001
            log.warning("sex-backfill: patient lookup failed for %s: %s", patient_uuid, e)
            current_gender = ""
        # Skip the no-op when both sides are already Unknown — saves a
        # round-trip and a noisy log line. Backfill triggers only when
        # the patient is currently Unknown/missing AND the new doc has
        # a real sex value.
        if current_gender in ("", "unknown") and extracted_sex != "unknown":
            try:
                await app.state.openemr_writer.update_patient_sex(
                    patient_uuid=patient_uuid,
                    sex=extracted_sex.capitalize(),
                )
            except Exception as e:  # noqa: BLE001
                log.warning(
                    "sex-backfill: PUT failed for %s sex=%s: %s",
                    patient_uuid, extracted_sex, e,
                )

    # Invalidate every cache slice that could now hold stale data:
    #   docs:{pid}    — Supporting Documents tab needs the new
    #                   DocumentReference and the lab-encounter we
    #                   created from extracted lab values.
    #   card:{pid}    — patient card needs the newly-written
    #                   allergies / medications / problems from intake.
    #   trends:{pid}  — vital-trends panel can change after lab writes.
    #   calendar:*    — the lab-encounter creates a new today-row in
    #                   the calendar (bound to the patient's chart),
    #                   and panels are keyed on patient ids so all
    #                   slices are potentially affected.
    app.state.cache.invalidate(f"docs:{patient_uuid}")
    app.state.cache.invalidate(f"card:{patient_uuid}")
    app.state.cache.invalidate(f"trends:{patient_uuid}")
    app.state.cache.invalidate_prefix("calendar:today:")
    # The bbox manifest for the new doc summarizes derived
    # allergies/meds/conditions just persisted by the writer; drop the
    # entry so the next viewer click reads the freshly-written tags.
    new_doc_id = result.reference_id.split("/", 1)[-1]
    app.state.pages_cache.invalidate(f"bbox:{new_doc_id}")
    sessions_notified = _mark_user_sessions_stale_after_upload(
        username=username,
        patient_uuid=patient_uuid,
        doc_type=doc_type,
        reference_id=result.reference_id,
    )
    log.info(
        "upload: user=%s patient=%s doc_type=%s ref=%s created=%s "
        "facts_written=%d/%d sessions_notified=%d",
        username, patient_uuid, doc_type,
        result.reference_id, result.created,
        (result.persistence or {}).get("facts_written", 0),
        (result.persistence or {}).get("facts_attempted", 0),
        sessions_notified,
    )
    return {
        "reference_id": result.reference_id,
        "sha256": result.write_result["sha256"],
        "created": result.created,
        "extracted": result.extracted.model_dump(mode="json"),
        "persistence": result.persistence,
    }


async def _get_doc_patient_id(
    fhir: FhirClient, pages_cache: TTLCache, document_id: str,
) -> str | None:
    """Lookup the Patient subject of a DocumentReference, cached.

    ACL gate for the document endpoints. We can't rely on the cache-hit
    path of `get_or_compute(pages:...)` to gate access — when two
    requests race a cache-miss window with different panels, the second
    waiter inherits the first's ACL pass via the per-key lock. Gating
    on this small lookup before touching the page cache keeps each
    request honest while still being microseconds-fast on the second
    hit (a doc's subject never changes for the lifetime of the doc).
    """
    cached = pages_cache.get(f"docpid:{document_id}")
    if cached is not None:
        return cached.get("patient_id")

    async def _compute() -> dict:
        doc = await fhir.get(f"DocumentReference/{document_id}")
        subject_ref = (doc.get("subject") or {}).get("reference") or ""
        pid = (
            subject_ref.removeprefix("Patient/")
            if subject_ref.startswith("Patient/") else None
        )
        return {"patient_id": pid}

    data = await pages_cache.get_or_compute(f"docpid:{document_id}", _compute)
    return data.get("patient_id")


async def _get_binary_patient_id(
    fhir: FhirClient, pages_cache: TTLCache, binary_id: str,
) -> str | None:
    """Resolve the Patient that owns a `Binary/{id}` by finding the
    DocumentReference whose attachment references it. Cached — a binary's
    owning document is fixed once uploaded.

    FHIR has no search parameter for an attachment URL, so on a cache miss
    we scan the DocumentReference roster (the same roster-wide read the
    dashboard prewarm already does). Demo-scale: a handful of docs per
    patient, fetched at most once per binary id per process. Returns None
    when no DocumentReference references this binary — panel-gated callers
    treat that as access-denied (404), mirroring `_get_doc_patient_id`.
    """
    cached = pages_cache.get(f"binarypid:{binary_id}")
    if cached is not None:
        return cached.get("patient_id")

    async def _compute() -> dict:
        docs = await fhir.search("DocumentReference", {"_count": 1000})
        for doc in docs:
            for content in doc.get("content") or []:
                url = ((content.get("attachment") or {}).get("url")) or ""
                if "/Binary/" not in url:
                    continue
                # Same id extraction as app.fhir.extras._proxy_binary_url.
                this_id = url.rsplit("/Binary/", 1)[-1].split("?", 1)[0].split("#", 1)[0]
                if this_id != binary_id:
                    continue
                subject_ref = (doc.get("subject") or {}).get("reference") or ""
                return {
                    "patient_id": (
                        subject_ref.removeprefix("Patient/")
                        if subject_ref.startswith("Patient/") else None
                    )
                }
        return {"patient_id": None}

    data = await pages_cache.get_or_compute(f"binarypid:{binary_id}", _compute)
    return data.get("patient_id")


@app.get("/api/document/{document_id}/pages")
async def api_document_pages(
    document_id: str,
    request: Request,
    username: str = Depends(current_user),
) -> dict:
    """Return rendered PNG pages of a `DocumentReference` for the
    Supporting-Documents inline viewer (PRD W2 §5 — visual PDF
    bounding-box overlay).

    ACL is gated up-front via the cached `docpid:` lookup so every
    request — cache hit or cache miss — re-validates the panel
    membership. The render itself is panel-agnostic (the rendered PNG
    is the same bytes regardless of who fetches it) and lives in
    `pages_cache` keyed by document_id, with a long TTL since
    DocumentReference attachments are immutable.
    """
    panel = await access_control.get_panel_for_user(
        request.app.state.fhir, username, request.app.state.assignments,
    )
    pages_cache: TTLCache = request.app.state.pages_cache

    try:
        patient_id = await _get_doc_patient_id(
            request.app.state.fhir, pages_cache, document_id,
        )
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=404, detail="Document not found") from e
    if panel is not None and (patient_id is None or patient_id not in panel):
        request.app.state.auth_store.log_event(
            event_type="patient_access_denied",
            username=username,
            sid=request.session.get("sid"),
            user_agent=request.headers.get("user-agent"),
            ip=request.client.host if request.client else None,
            detail=f"document={document_id} patient={patient_id}",
        )
        raise HTTPException(status_code=404, detail="Document not found")

    async def _compute() -> dict:
        # ACL already gated above; pass panel=None so the adapter's own
        # check is a no-op and prefetch (also panel=None) shares the
        # same cached value.
        result = await adapter.get_document_pages(
            request.app.state.fhir, document_id=document_id, panel=None,
        )
        return result["data"]

    try:
        return await pages_cache.get_or_compute(f"pages:{document_id}", _compute)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"FHIR fetch failed: {e!s}") from e


@app.get("/api/document-source/{resource_type}/{resource_id}")
async def api_document_source(
    resource_type: str,
    resource_id: str,
    request: Request,
    username: str = Depends(current_user),
) -> dict:
    """Look up the DocumentReference a chart resource was extracted from,
    so a chat citation click can deep-link into the source PDF with the
    matching bbox highlighted (PRD W2 §5 — bbox v1 per-citation).

    Returns 200 with `document_ref=null` when the resource exists but
    has no `[copilot-source: ...]` tag (hand-entered, not extracted).
    The frontend uses that signal to decide between "open source doc"
    and the existing patient-card scroll behavior. 404 only on ACL
    denial / missing resource.
    """
    panel = await access_control.get_panel_for_user(
        request.app.state.fhir, username, request.app.state.assignments,
    )
    try:
        result = await adapter.get_resource_source_document(
            request.app.state.fhir,
            resource_type=resource_type,
            resource_id=resource_id,
            panel=panel,
            store=request.app.state.extracted_sources,
        )
    except access_control.PatientAccessDenied as e:
        request.app.state.auth_store.log_event(
            event_type="patient_access_denied",
            username=username,
            sid=request.session.get("sid"),
            user_agent=request.headers.get("user-agent"),
            ip=request.client.host if request.client else None,
            detail=f"resource={resource_type}/{resource_id} patient={e.patient_id}",
        )
        raise HTTPException(status_code=404, detail="Resource not found") from e
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"FHIR fetch failed: {e!s}") from e
    return result["data"]


@app.get("/api/document/{document_id}/bbox-manifest")
async def api_document_bbox_manifest(
    document_id: str,
    request: Request,
    username: str = Depends(current_user),
) -> dict:
    """Return every extracted fact persisted from a `DocumentReference`
    along with its bbox metadata, so the Supporting-Documents viewer can
    overlay highlight rectangles on the rendered pages.

    Pairs with `/api/document/{id}/pages`; the frontend fetches both,
    renders the pages, then absolute-positions one overlay per fact at
    its `bbox` coordinates. Facts whose `bbox` is null are still returned
    so the frontend can list them as "extracted but no source rectangle"
    instead of silently hiding them.
    """
    panel = await access_control.get_panel_for_user(
        request.app.state.fhir, username, request.app.state.assignments,
    )
    pages_cache: TTLCache = request.app.state.pages_cache

    try:
        patient_id = await _get_doc_patient_id(
            request.app.state.fhir, pages_cache, document_id,
        )
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=404, detail="Document not found") from e
    if panel is not None and (patient_id is None or patient_id not in panel):
        request.app.state.auth_store.log_event(
            event_type="patient_access_denied",
            username=username,
            sid=request.session.get("sid"),
            user_agent=request.headers.get("user-agent"),
            ip=request.client.host if request.client else None,
            detail=f"document={document_id} patient={patient_id}",
        )
        raise HTTPException(status_code=404, detail="Document not found")

    sidecar_store = request.app.state.extracted_sources

    async def _compute() -> dict:
        result = await adapter.get_document_bbox_manifest(
            request.app.state.fhir, document_id=document_id, panel=None,
            store=sidecar_store,
        )
        return result["data"]

    try:
        return await pages_cache.get_or_compute(f"bbox:{document_id}", _compute)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"FHIR fetch failed: {e!s}") from e


@app.post("/api/document/{document_id}/hide")
async def api_document_hide(
    document_id: str,
    request: Request,
    username: str = Depends(current_user),
) -> dict:
    """Soft-hide a DocumentReference from the supporting-documents view.

    The underlying OpenEMR DocumentReference is preserved — this is a
    view-side toggle backed by the `hidden_documents` SQLite table.
    Idempotent: hiding an already-hidden doc refreshes the audit row
    (`hidden_by` / `hidden_at`).

    ACL: the user must be on a panel that includes the document's
    patient subject — same fail-closed shape as the other document
    endpoints. 404 on denial so a guesser can't probe panel
    boundaries via response codes.
    """
    panel = await access_control.get_panel_for_user(
        request.app.state.fhir, username, request.app.state.assignments,
    )
    try:
        # Light ACL check — fetching the DocumentReference confirms it
        # exists AND lets us inspect `subject.reference` for panel
        # membership before flipping the bit.
        doc = await request.app.state.fhir.get(f"DocumentReference/{document_id}")
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=404, detail="Document not found") from e
    subject_ref = (doc.get("subject") or {}).get("reference") or ""
    patient_id = subject_ref.removeprefix("Patient/") if subject_ref.startswith("Patient/") else None
    if panel is not None and (patient_id is None or patient_id not in panel):
        request.app.state.auth_store.log_event(
            event_type="patient_access_denied",
            username=username,
            sid=request.session.get("sid"),
            user_agent=request.headers.get("user-agent"),
            ip=request.client.host if request.client else None,
            detail=f"document={document_id} patient={patient_id}",
        )
        raise HTTPException(status_code=404, detail="Document not found")
    record = request.app.state.hidden_docs.hide(
        document_id=document_id, hidden_by=username,
    )
    # Bust the docs cache so the supporting-docs list re-renders without
    # the hidden item on the next fetch. Same key shape the upload path
    # invalidates.
    if patient_id:
        app.state.cache.invalidate(f"docs:{patient_id}")
    log.info(
        "document hidden: user=%s document=%s patient=%s",
        username, document_id, patient_id,
    )
    return {
        "document_id": record.document_id,
        "hidden_by": record.hidden_by,
        "hidden_at": record.hidden_at,
    }


@app.post("/api/document/{document_id}/resolve-content-match")
async def api_document_resolve_content_match(
    document_id: str,
    request: Request,
    action: str = Form(...),
    prior_ref: str = Form(...),
    username: str = Depends(current_user),
) -> dict:
    """User's choice from the dedup-Layer-2 modal.

    `document_id` is the **new** doc that triggered the prompt. The
    prior matching doc is passed via `prior_ref` (full
    `DocumentReference/<uuid>` form) because the frontend already
    received it on the 409 payload — re-asking the fingerprint store
    here would race against simultaneous uploads.

    Actions:
      - `replace`: soft-hide the prior doc, forget its fingerprint,
        persist facts on the new doc, claim the fingerprint with the
        new doc id.
      - `keep_both`: persist facts on the new doc; do not touch the
        prior doc's fingerprint (it stays the canonical owner).
      - `cancel`: soft-hide the new doc; do not persist facts.
    """
    if action not in {"replace", "keep_both", "cancel"}:
        raise HTTPException(
            status_code=400,
            detail=f"action must be one of replace/keep_both/cancel, got {action!r}",
        )
    new_doc_id = document_id.split("/", 1)[-1] if "/" in document_id else document_id
    prior_doc_id = prior_ref.split("/", 1)[-1] if "/" in prior_ref else prior_ref

    # Re-fetch the new doc so we know its patient + extraction. The
    # straight-through path here is more roundtrip than ideal; a
    # production version would cache the in-flight extraction on a
    # short-lived server-side context. For demo scope, an extra FHIR
    # GET + a re-extract on `replace` / `keep_both` is acceptable.
    try:
        new_doc = await request.app.state.fhir.get(
            f"DocumentReference/{new_doc_id}",
        )
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=404, detail="New document not found") from e
    subject_ref = (new_doc.get("subject") or {}).get("reference") or ""
    if not subject_ref.startswith("Patient/"):
        raise HTTPException(status_code=400, detail="New document has no patient subject")
    patient_id = subject_ref.removeprefix("Patient/")
    panel = await access_control.get_panel_for_user(
        request.app.state.fhir, username, request.app.state.assignments,
    )
    if panel is not None and patient_id not in panel:
        request.app.state.auth_store.log_event(
            event_type="patient_access_denied",
            username=username,
            sid=request.session.get("sid"),
            user_agent=request.headers.get("user-agent"),
            ip=request.client.host if request.client else None,
            detail=f"document={new_doc_id} patient={patient_id}",
        )
        raise HTTPException(status_code=404, detail="Document not found")

    if action == "cancel":
        # Soft-hide the new doc; don't persist anything. The prior
        # doc continues to be the canonical fingerprint owner.
        app.state.hidden_docs.hide(document_id=new_doc_id, hidden_by=username)
        app.state.cache.invalidate(f"docs:{patient_id}")
        log.info(
            "content-match resolve cancel: user=%s patient=%s new=%s",
            username, patient_id, new_doc_id,
        )
        return {"action": "cancel", "new_document_hidden": True}

    # `replace` and `keep_both` both persist facts on the new doc;
    # the only difference is what happens to the prior doc.
    # The extraction is no longer in memory — we have to re-fetch +
    # re-extract from the new doc's bytes. For demo scope this is
    # acceptable; a production version would memoize.
    from app.fhir.adapter import _fetch_attachment_bytes  # noqa: PLC0415
    from app.extraction.render import render_to_png_pages  # noqa: PLC0415
    from app.extraction.vision import extract_via_claude  # noqa: PLC0415

    att = (new_doc.get("content") or [{}])[0].get("attachment") or {}
    file_bytes = await _fetch_attachment_bytes(request.app.state.fhir, att)
    if file_bytes is None:
        raise HTTPException(status_code=502, detail="Could not fetch new document bytes")
    content_type = att.get("contentType") or "application/pdf"
    title = att.get("title") or ""
    # The doc_type rides on OpenEMR's category mapping; fall back to
    # `intake_form` if we can't infer (the frontend always supplied
    # one on the original upload, so this branch is unlikely).
    category_path = ((new_doc.get("category") or [{}])[0].get("coding") or [{}])[0].get("code") or ""
    cp_lower = category_path.lower()
    title_lower = title.lower()
    if "lab" in cp_lower or "lab" in title_lower:
        doc_type = "lab_pdf"
    elif "referral" in cp_lower or "referral" in title_lower:
        doc_type = "referral_letter"
    elif (
        title_lower.endswith(".hl7")
        or "hl7" in title_lower
        or "adt-a08" in title_lower
        or "oru-r01" in title_lower
    ):
        doc_type = "hl7_message"
    elif (
        title_lower.endswith(".tiff")
        or title_lower.endswith(".tif")
        or "fax" in title_lower
    ):
        doc_type = "fax_packet"
    elif (
        title_lower.endswith(".xlsx")
        or "workbook" in title_lower
    ):
        doc_type = "workbook"
    else:
        doc_type = "intake_form"
    if doc_type == "hl7_message":
        # HL7 v2 doesn't render to PNG — re-parse the segments directly.
        from app.extraction.hl7 import parse_hl7_message  # noqa: PLC0415
        extracted = parse_hl7_message(
            file_bytes,
            source_document_id=f"DocumentReference/{new_doc_id}",
        )
    elif doc_type == "workbook":
        # Workbooks (.xlsx) also bypass vision — re-parse via openpyxl.
        from app.extraction.workbook import parse_workbook  # noqa: PLC0415
        extracted = parse_workbook(
            file_bytes,
            source_document_id=f"DocumentReference/{new_doc_id}",
        )
    else:
        page_pngs = render_to_png_pages(file_bytes, content_type)
        extracted = await extract_via_claude(
            page_pngs=page_pngs,
            doc_type=doc_type,  # type: ignore[arg-type]
            source_document_id=f"DocumentReference/{new_doc_id}",
        )
    persistence = await persist_extracted_facts(
        writer=app.state.openemr_writer,
        patient_uuid=patient_id,
        extracted=extracted,
        source_document_id=f"DocumentReference/{new_doc_id}",
        practitioners_store=app.state.extracted_practitioners,
        lab_results_store=app.state.extracted_lab_results,
    )

    fingerprint = compute_fingerprint(extracted)
    # Layer-1.5 text fingerprint for the new doc — recorded for PDF
    # uploads only (mirrors the upload endpoint's gating). The
    # `forget_document` call below clears BOTH the structural and the
    # text rows the prior doc owned, so re-recording both here keeps
    # future uploads of the same content prompting against the new
    # canonical doc.
    text_fp_namespaced: str | None = None
    if content_type.lower() == "application/pdf":
        raw_text_fp = pdf_text_fingerprint(file_bytes)
        if raw_text_fp:
            text_fp_namespaced = namespace_text_fingerprint(raw_text_fp)
    if action == "replace":
        # Hide the prior doc and reassign its fingerprint to the new doc.
        app.state.hidden_docs.hide(document_id=prior_doc_id, hidden_by=username)
        if fingerprint or text_fp_namespaced:
            app.state.fingerprints.forget_document(prior_doc_id)
        if fingerprint:
            app.state.fingerprints.record(
                patient_id=patient_id,
                fingerprint=fingerprint,
                document_id=new_doc_id,
                recorded_by=username,
            )
        if text_fp_namespaced:
            app.state.fingerprints.record(
                patient_id=patient_id,
                fingerprint=text_fp_namespaced,
                document_id=new_doc_id,
                recorded_by=username,
            )
    # action == "keep_both": leave the prior fingerprint alone; both
    # docs are visible. The next upload of the same content prompts
    # against the prior (still canonical) doc.

    app.state.cache.invalidate(f"docs:{patient_id}")
    app.state.cache.invalidate(f"card:{patient_id}")
    app.state.cache.invalidate(f"trends:{patient_id}")
    app.state.cache.invalidate_prefix("calendar:today:")
    # The new doc's bbox manifest just changed — drop the entry so the
    # viewer's next fetch sees the freshly-written derived facts.
    app.state.pages_cache.invalidate(f"bbox:{new_doc_id}")
    log.info(
        "content-match resolve %s: user=%s patient=%s new=%s prior=%s "
        "facts_written=%d/%d",
        action, username, patient_id, new_doc_id, prior_doc_id,
        persistence.get("facts_written", 0),
        persistence.get("facts_attempted", 0),
    )
    return {
        "action": action,
        "new_reference_id": f"DocumentReference/{new_doc_id}",
        "prior_reference_id": f"DocumentReference/{prior_doc_id}",
        "persistence": persistence,
    }


@app.post("/api/document/{document_id}/unhide")
async def api_document_unhide(
    document_id: str,
    request: Request,
    username: str = Depends(current_user),
) -> dict:
    """Restore a soft-hidden DocumentReference to the default view.

    No ACL fetch on the unhide path: the user must already be authenticated,
    and the only thing they can do is restore visibility of a doc that
    is already in the hidden table. A user without panel access cannot
    *see* the hidden doc to call this on, so the practical attack
    surface is nil. (Hide does check ACL because hiding a doc you can't
    see would be a panel-leak signal otherwise.)
    """
    existed = request.app.state.hidden_docs.unhide(document_id)
    # Best-effort cache bust — we don't always know the patient_id at
    # this point. Fetching the doc to discover it would be wasteful;
    # invalidate every patient's docs slice instead. Cheap (string
    # prefix scan over an in-process dict).
    app.state.cache.invalidate_prefix("docs:")
    log.info(
        "document unhidden: user=%s document=%s existed=%s",
        username, document_id, existed,
    )
    return {"document_id": document_id, "was_hidden": existed}


@app.get("/upload", response_model=None)
async def upload_page(request: Request) -> FileResponse | RedirectResponse:
    """Minimal HTML form for uploading a clinical document.

    Mounted at `/upload` rather than under `/api/*` because it returns
    HTML (not JSON). Login-gated like the rest of the UI — anonymous
    visitors get bounced to `/login`.
    """
    if current_session(request) is None:
        return RedirectResponse(url="/login", status_code=302)
    return FileResponse(WEB_DIR / "upload.html", headers=_NO_CACHE_HEADERS)


def _calendar_cache_key(panel: frozenset[str] | None) -> str:
    """Cache key for `/api/calendar/today` keyed on the panel content, not the
    user. Two users with the same allow-list share the cache; admin (panel=None)
    matches the prewarm key so admin's first request hits warm.
    """
    if panel is None:
        return "calendar:today:all"
    if not panel:
        return "calendar:today:empty"
    return "calendar:today:p=" + ",".join(sorted(panel))


@app.get("/api/calendar/today")
async def calendar_today(username: str = Depends(current_user)) -> dict:
    panel = await access_control.get_panel_for_user(
        app.state.fhir, username, app.state.assignments,
    )
    cache_key = _calendar_cache_key(panel)

    async def _compute() -> dict:
        result = await get_calendar_today(
            app.state.fhir, panel=panel,
            active_patients_store=app.state.active_patients,
        )
        return result["data"]
    try:
        return await app.state.cache.get_or_compute(cache_key, _compute)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"FHIR fetch failed: {e!s}") from e


@app.post("/api/schedule")
async def api_schedule(
    request: Request,
    patient_uuid: str = Form(...),
    event_date: str = Form(...),
    hour: int = Form(...),
    minute: int = Form(...),
    duration_minutes: int = Form(30),
    comments: str = Form(""),
    username: str = Depends(current_user),
) -> dict:
    """Schedule an appointment for `patient_uuid` on the logged-in user's
    calendar. Form fields:

      patient_uuid: FHIR Patient UUID the user has access to
      event_date:   ISO date "YYYY-MM-DD"
      hour:         0-23 (24h)
      minute:       0-59, but the UI dropdown restricts to 5-min increments
      duration_minutes: minutes (default 30)
      comments:     optional free-text note saved to pc_hometext

    Provider assignment: per Option A from the calendar design — the
    appointment is created for the currently-logged-in user (their
    `users.id`). Admins schedule on the admin calendar (id=1); other
    users are resolved dynamically by querying OpenEMR's `/api/user`
    endpoint. Users not resolvable to a `users.id` get HTTP 400.

    Returns: {appointment_id, event_date, start_time, duration_minutes}.
    Errors:
      400 — invalid hour/minute, invalid date, user not provider-mapped
      404 — patient not in caller's panel
      502 — OpenEMR rejected the write
    """
    await _check_patient_access(request, username, patient_uuid)

    if not (0 <= hour <= 23):
        raise HTTPException(
            status_code=400, detail=f"hour must be 0-23, got {hour}",
        )
    if not (0 <= minute <= 59):
        raise HTTPException(
            status_code=400, detail=f"minute must be 0-59, got {minute}",
        )
    if duration_minutes <= 0 or duration_minutes > 480:
        raise HTTPException(
            status_code=400,
            detail=f"duration_minutes must be 1-480, got {duration_minutes}",
        )

    provider_user_id = await access_control.resolve_user_id(
        app.state.openemr_writer, username,
    )
    if provider_user_id is None:
        raise HTTPException(
            status_code=400,
            detail=(
                f"user {username!r} could not be resolved to an OpenEMR "
                f"users.id (neither static override nor /api/user lookup matched)"
            ),
        )

    start_time = f"{hour:02d}:{minute:02d}"
    try:
        result = await app.state.openemr_writer.write_appointment(
            patient_uuid=patient_uuid,
            provider_user_id=provider_user_id,
            event_date=event_date,
            start_time=start_time,
            duration_minutes=duration_minutes,
            comments=comments,
        )
    except OpenEMRWriteError as e:
        raise HTTPException(
            status_code=502, detail=f"OpenEMR appointment write failed: {e}",
        ) from e

    # Drop ALL cached calendar entries — the new appointment is going to
    # change every per-panel slice, and the cache is keyed on panel
    # content, not on a date. Use the same prefix-invalidate pattern as
    # the assignment-changed handler (line ~1390).
    app.state.cache.invalidate_prefix("calendar:today:")

    log.info(
        "schedule: user=%s patient=%s date=%s time=%s eid=%s",
        username, patient_uuid, event_date, start_time, result["appointment_id"],
    )
    return {
        "appointment_id": result["appointment_id"],
        "event_date": event_date,
        "start_time": start_time,
        "duration_minutes": duration_minutes,
    }


# ─── clinical notes ──────────────────────────────────────────────────────


class ClinicalNoteRequest(BaseModel):
    notes_md: str = ""
    recs_md: str = ""
    vitals: dict | None = None


@app.get("/api/patient/{patient_id}/clinical-notes/draft")
async def get_clinical_note_draft(
    patient_id: str,
    request: Request,
    username: str = Depends(current_user),
) -> dict:
    """Return the current author's open draft for this patient × current shift,
    or a stub indicating no draft exists yet."""
    await _check_patient_access(request, username, patient_id)
    note = app.state.clinical_notes.get_draft(patient_id, username, now=now_utc())
    if note is None:
        return {"draft": None}
    return {"draft": note.to_doc_item()}


@app.post("/api/patient/{patient_id}/clinical-notes/draft")
async def upsert_clinical_note_draft(
    patient_id: str,
    body: ClinicalNoteRequest,
    request: Request,
    username: str = Depends(current_user),
) -> dict:
    """Create or update the author's open draft. Multiple saves within a
    shift consolidate into the same draft until it is finalized; once
    finalized, a subsequent save opens a fresh draft (an addendum)."""
    await _check_patient_access(request, username, patient_id)
    note = app.state.clinical_notes.upsert_draft(
        patient_id=patient_id,
        author=username,
        notes_md=body.notes_md or "",
        recs_md=body.recs_md or "",
        vitals=body.vitals,
        now=now_utc(),
    )
    # Invalidate the supporting-docs cache for this patient so the new
    # draft (or its label change) shows immediately on next fetch. Trends
    # only count finalized notes, so an in-progress draft can't move them
    # — but if the doctor *deletes* values from a previously-saved draft
    # we still need a fresh read on the next save.
    app.state.cache.invalidate(f"docs:{patient_id}")
    app.state.cache.invalidate(f"trends:{patient_id}")
    return {"draft": note.to_doc_item()}


@app.post("/api/patient/{patient_id}/clinical-notes/save")
async def finalize_clinical_note(
    patient_id: str,
    request: Request,
    username: str = Depends(current_user),
) -> dict:
    """Explicit Save — promote the open draft to immutable 'final' status,
    then best-effort push the vitals to OpenEMR's vitals chart so the EHR
    sees what the doctor entered. A push failure does not roll back the
    local finalize — the note remains the canonical record either way.
    """
    await _check_patient_access(request, username, patient_id)
    try:
        note = app.state.clinical_notes.finalize(patient_id, username, now=now_utc())
    except ClinicalNoteNotFound as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    push_status: dict = {"attempted": False, "ok": False, "error": None, "vital_id": None}
    if note.vitals:
        push_status["attempted"] = True
        try:
            result = await app.state.openemr_writer.write_vitals(
                patient_uuid=patient_id,
                vitals=note.vitals,
                when_iso=note.finalized_at or note.updated_at,
            )
            app.state.clinical_notes.mark_fhir_synced(
                note.id, vital_id=result.get("vital_id"), now=now_utc(),
            )
            push_status.update(ok=True, vital_id=result.get("vital_id"))
            # The newly written observations need to surface on the next
            # FHIR search — invalidate the card-level FHIR cache too.
            app.state.cache.invalidate(f"card:{patient_id}")
        except OpenEMRWriteError as e:
            log.warning("FHIR vitals push failed for note %s: %s", note.id, e)
            push_status["error"] = str(e)
        except Exception as e:  # noqa: BLE001
            log.exception("unexpected error pushing vitals for note %s", note.id)
            push_status["error"] = f"{type(e).__name__}: {e}"

    app.state.cache.invalidate(f"docs:{patient_id}")
    app.state.cache.invalidate(f"trends:{patient_id}")
    # Re-read the note so we pick up the freshly-stamped fhir_synced_at.
    refreshed = next(
        (n for n in app.state.clinical_notes.list_for_patient(patient_id, now=now_utc())
         if n.id == note.id),
        note,
    )
    return {"note": refreshed.to_doc_item(), "ehr_push": push_status}


@app.get("/api/patient/{patient_id}/clinical-notes/latest-prior-shift")
async def latest_prior_shift_note(
    patient_id: str,
    request: Request,
    username: str = Depends(current_user),
) -> dict:
    """Most recent finalized note from a *prior* shift — what a doctor sees
    first when they click into a patient at the start of their shift."""
    await _check_patient_access(request, username, patient_id)
    note = app.state.clinical_notes.latest_prior_shift(patient_id, now=now_utc())
    return {"note": note.to_doc_item() if note else None}


@app.get("/api/binary/{binary_id}")
async def api_binary(
    binary_id: str,
    request: Request,
    username: str = Depends(current_user),
):
    """Proxy a FHIR Binary resource to the browser as raw bytes.

    OpenEMR returns DocumentReference.content[].attachment.url as
    `https://localhost:9300/.../Binary/{id}` — that points at the
    OpenEMR container behind cloudflared and requires an OAuth bearer
    the browser can't see. This endpoint fetches the Binary server-side
    using the agent's existing OAuth client and streams the decoded
    bytes back, so the front-end can use a normal `<a href>` to open
    the document.

    ACL: logged in AND the caller's panel must include the patient that
    owns the parent DocumentReference (resolved via the cached
    `_get_binary_patient_id` lookup). Out-of-panel — or unknown — binaries
    return 404, not 403, mirroring `/api/document/{id}/pages` so a probe
    can't tell "doesn't exist" from "not yours". Admins (panel is None)
    are unrestricted.

    Implementation: OpenEMR's `GET /fhir/Binary/{id}` returns the raw
    file bytes (PDF/PNG/etc.) with the original Content-Type header,
    not a FHIR JSON envelope with base64 `data`. Use the dedicated
    `FhirClient.get_raw` so we don't try to parse PNG bytes as JSON.
    """
    pages_cache: TTLCache = request.app.state.pages_cache
    try:
        patient_id = await _get_binary_patient_id(
            request.app.state.fhir, pages_cache, binary_id,
        )
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=404, detail="Document not found") from e
    panel = await access_control.get_panel_for_user(
        request.app.state.fhir, username, request.app.state.assignments,
    )
    if panel is not None and (patient_id is None or patient_id not in panel):
        request.app.state.auth_store.log_event(
            event_type="patient_access_denied",
            username=username,
            sid=request.session.get("sid"),
            user_agent=request.headers.get("user-agent"),
            ip=request.client.host if request.client else None,
            detail=f"binary={binary_id} patient={patient_id}",
        )
        raise HTTPException(status_code=404, detail="Document not found")

    try:
        body, content_type = await request.app.state.fhir.get_raw(f"Binary/{binary_id}")
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"FHIR fetch failed: {e!s}") from e
    return Response(content=body, media_type=content_type)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


# ─── observability endpoints ─────────────────────────────────────────────


@app.get("/api/traces")
async def list_traces(
    limit: int = 50,
    _admin: str = Depends(require_admin),
) -> dict:
    """Newest-first list of recent request traces. Admin-only.

    Each entry includes latency, token totals, $ cost, validator outcome,
    tool-call count, and per-tool detail — and the clinician's free-text
    question, patient UUIDs in tool args, and raw exception strings, which
    is PHI/PII across users. Hence `require_admin`, not `current_user`.
    Bounded ring buffer (200) — no pagination beyond `limit`.
    """
    items = [t.to_dict() for t in app.state.traces.list_recent(limit=limit)]
    return {"count": len(items), "items": items}


@app.get("/api/traces/{request_id}")
async def get_trace(request_id: str, _admin: str = Depends(require_admin)) -> dict:
    """Single request trace by id. Admin-only — same PHI/PII exposure as
    `/api/traces`."""
    trace = app.state.traces.get(request_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="trace not found")
    return trace.to_dict()


@app.get("/observability", response_model=None)
async def observability_page(request: Request) -> FileResponse | RedirectResponse:
    """Observability dashboard. Admin-only; non-admins are redirected to /
    (the page surfaces other clinicians' traces — see `/api/traces`). A
    redirect is friendlier than a 403 for a misclick, mirroring /admin."""
    session = current_session(request)
    if session is None:
        return RedirectResponse(url="/login", status_code=302)
    if not is_admin(session.username):
        return RedirectResponse(url="/", status_code=302)
    return FileResponse(WEB_DIR / "observability.html", headers=_NO_CACHE_HEADERS)


# ─── admin endpoints (admin-only oversight) ──────────────────────────────


@app.get("/admin", response_model=None)
async def admin_page(request: Request) -> FileResponse | RedirectResponse:
    """Admin oversight page. Login-gated; non-admins get redirected to /
    rather than getting a 403 — the page wouldn't be useful to them anyway,
    and a redirect is friendlier than an error for a misclick."""
    session = current_session(request)
    if session is None:
        return RedirectResponse(url="/login", status_code=302)
    if not is_admin(session.username):
        return RedirectResponse(url="/", status_code=302)
    return FileResponse(WEB_DIR / "admin.html", headers=_NO_CACHE_HEADERS)


@app.get("/api/admin/recent-activity")
async def admin_recent_activity(
    limit: int = 200,
    _admin: str = Depends(require_admin),
) -> dict:
    """Aggregate recent /chat traces by username so admins can see who's
    been using the app. Pulls from the durable SQLite trace store, so
    activity persists across restarts. Limit defaults to 200 traces (the
    most recent), aggregated client-side into a per-username summary."""
    traces = app.state.traces.list_recent(limit=limit)
    by_user: dict[str, dict] = {}
    for t in traces:
        u = t.username or "(anonymous)"
        slot = by_user.setdefault(u, {
            "username": u,
            "request_count": 0,
            "tool_call_count": 0,
            "first_seen": None,
            "last_seen": None,
            "total_cost_usd": 0.0,
            "blocked_count": 0,
            "validator_failed_count": 0,
        })
        slot["request_count"] += 1
        slot["tool_call_count"] += len(t.tool_events)
        slot["total_cost_usd"] = round(slot["total_cost_usd"] + (t.cost_usd or 0.0), 5)
        if t.error and t.error.startswith("jailbreak_blocked:"):
            slot["blocked_count"] += 1
        if t.validator_failed:
            slot["validator_failed_count"] += 1
        # Emit ISO with explicit UTC offset so the browser's `new Date(iso)`
        # parses an unambiguous instant. `fromtimestamp(ts)` without a tz
        # arg returns a naive local-time datetime — that ends up rendering
        # in the wrong zone whenever the server and browser disagree
        # (Hetzner runs UTC; admins likely view from MDT). The UTC-suffixed
        # ISO sorts correctly as a string and round-trips through JS Date.
        ts_iso = (
            datetime.fromtimestamp(t.started_at, tz=timezone.utc)
            .isoformat(timespec="seconds")
        )
        if slot["first_seen"] is None or ts_iso < slot["first_seen"]:
            slot["first_seen"] = ts_iso
        if slot["last_seen"] is None or ts_iso > slot["last_seen"]:
            slot["last_seen"] = ts_iso
    items = sorted(by_user.values(), key=lambda x: x["last_seen"] or "", reverse=True)
    return {"count": len(items), "items": items, "trace_window": limit}


@app.get("/api/admin/practitioners")
async def admin_practitioners(_admin: str = Depends(require_admin)) -> dict:
    """Read-only list of OpenEMR users via FHIR Practitioner search.

    The Practitioner resource is read-only on the agent's OAuth client
    (system/Practitioner.read). Anyone listed here is a potential login
    user — we don't currently surface which OpenEMR `users_secure` row
    each Practitioner maps to (that's a separate non-FHIR API call), so
    "user X is in this list" is necessary but not sufficient for them
    to actually log in. Sufficient for an admin oversight view; full
    user-creation flow is a follow-up task."""
    try:
        rows = await app.state.fhir.search("Practitioner", {"_count": 50})
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"FHIR fetch failed: {e!s}") from e
    # OpenEMR's Practitioner search runs a compound OR over `username` and
    # `abook_type` (see src/Services/PractitionerService.php). A user matching
    # both branches comes back twice with identical ids — dedupe here so the
    # admin UI dropdown shows each practitioner once.
    items: list[dict] = []
    seen_ids: set[str] = set()
    for p in rows:
        pid = p.get("id")
        if not pid or pid in seen_ids:
            continue
        seen_ids.add(pid)
        name = (p.get("name") or [{}])[0]
        given = " ".join(name.get("given") or [])
        family = name.get("family") or ""
        full = (given + " " + family).strip() or "(unknown)"
        items.append({
            "id": pid,
            "name": full,
            "active": p.get("active"),
            "telecom": [
                {"system": t.get("system"), "value": t.get("value")}
                for t in (p.get("telecom") or [])
            ],
        })
    return {"count": len(items), "items": items}


class PatientAssignmentRequest(BaseModel):
    patient_id: str
    practitioner_id: str | None = None  # None / "" means unassign


@app.get("/api/admin/patient-assignments")
async def admin_patient_assignments(_admin: str = Depends(require_admin)) -> dict:
    """List every patient with their current assigned practitioner (if any).

    Joins the FHIR Patient roster (source of truth for who exists) with
    the co-pilot's local `patient_assignments` table (source of truth for
    who's assigned to whom — see `app.assignments_db` for why we don't
    use `Patient.generalPractitioner` here). Returns one row per patient,
    with `assigned_practitioner_id` null if unassigned.
    """
    try:
        patients = await app.state.fhir.search("Patient", {"_count": 200})
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"FHIR fetch failed: {e!s}") from e
    patients = access_control.filter_active(
        patients, dynamic_store=app.state.active_patients,
    )
    current = app.state.assignments.all_assignments()  # patient_id -> prac_id
    items: list[dict] = []
    for p in patients:
        pid = p.get("id")
        if not pid:
            continue
        name = (p.get("name") or [{}])[0]
        given = " ".join(name.get("given") or [])
        family = name.get("family") or ""
        full = (given + " " + family).strip() or "(unknown)"
        items.append({
            "patient_id": pid,
            "name": full,
            "assigned_practitioner_id": current.get(pid),
        })
    items.sort(key=lambda r: r["name"].lower())
    return {"count": len(items), "items": items}


@app.post("/api/admin/patient-assignments")
async def admin_set_patient_assignment(
    body: PatientAssignmentRequest,
    admin: str = Depends(require_admin),
) -> dict:
    """Assign a patient to a practitioner (or clear the assignment).

    Empty / null `practitioner_id` removes the assignment, restoring the
    "no panel mapping" state. After mutation we invalidate the in-memory
    panel cache so the affected user's next request reflects the change
    immediately rather than waiting up to 5 min for the TTL.
    """
    pid = body.patient_id.strip()
    prac_id = (body.practitioner_id or "").strip()
    if not pid:
        raise HTTPException(status_code=400, detail="patient_id required")

    if prac_id:
        app.state.assignments.upsert(
            patient_id=pid, practitioner_id=prac_id, assigned_by=admin,
        )
        action = "assigned"
    else:
        app.state.assignments.unassign(pid)
        action = "unassigned"

    # Drop ALL cached panels and ALL cached calendar entries — assignment
    # changes can affect multiple users' views (the previous owner loses
    # access; the new owner gains it). Cheap to recompute on next request.
    access_control.invalidate_panel()
    app.state.cache.invalidate_prefix("calendar:today:")
    log.info(
        "assignment %s: patient=%s practitioner=%s by=%s",
        action, pid, prac_id or "(none)", admin,
    )
    return {
        "ok": True,
        "patient_id": pid,
        "assigned_practitioner_id": prac_id or None,
    }


def _ts_iso(ts: float | None) -> str | None:
    """UTC-explicit ISO for JSON responses. None passes through so the
    browser-side renderer can decide how to handle missing values."""
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="seconds")


@app.get("/api/admin/sessions")
async def admin_active_sessions(_admin: str = Depends(require_admin)) -> dict:
    """List sessions that are currently valid (not revoked, not idle-timed-out,
    not absolute-timed-out). The list updates as sessions come and go — for
    a durable history of every login/logout, see /api/admin/auth-events."""
    sessions = app.state.auth_store.list_active_sessions()
    items = [
        {
            "sid": s.sid,
            "username": s.username,
            "created_at": _ts_iso(s.created_at),
            "last_seen": _ts_iso(s.last_seen),
            "expires_at": _ts_iso(s.expires_at),
            "user_agent": s.user_agent,
            "ip": s.ip,
        }
        for s in sessions
    ]
    return {"count": len(items), "items": items}


@app.post("/api/admin/sessions/{sid}/revoke")
async def admin_revoke_session(
    sid: str,
    request: Request,
    admin_user: str = Depends(require_admin),
) -> dict:
    """Revoke a single active session. The doctor's next request gets 401
    and is bounced back to /login. Recorded in `auth_events` as a
    `session_revoked` event with the admin's username in `detail`."""
    revoked_username = app.state.auth_store.revoke_session(sid)
    if revoked_username is None:
        raise HTTPException(status_code=404, detail="session not found or already revoked")
    ua, ip = _client_ua_ip(request)
    app.state.auth_store.log_event(
        "session_revoked",
        username=revoked_username,
        sid=sid,
        user_agent=ua,
        ip=ip,
        detail=f"by:{admin_user}",
    )
    return {"ok": True, "revoked_username": revoked_username}


@app.get("/api/admin/auth-events")
async def admin_auth_events(
    limit: int = 200,
    _admin: str = Depends(require_admin),
) -> dict:
    """Newest-first audit log of authentication events: login_success,
    login_failure, logout, session_revoked, session_expired. This table
    is append-only — events are never overwritten. Bounded ring view via
    `limit` (default 200)."""
    events = app.state.auth_store.list_auth_events(limit=limit)
    items = [
        {
            "id": e.id,
            "event_type": e.event_type,
            "username": e.username,
            "sid": e.sid,
            "at": _ts_iso(e.at),
            "user_agent": e.user_agent,
            "ip": e.ip,
            "detail": e.detail,
        }
        for e in events
    ]
    return {"count": len(items), "items": items}
