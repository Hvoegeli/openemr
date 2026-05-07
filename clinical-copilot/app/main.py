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
from typing import AsyncIterator
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse
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

from app.agent.graph import MAX_VALIDATION_ATTEMPTS, active_model_label, build_graph, message_text  # noqa: E402
from app.agent.input_guard import JAILBREAK_REFUSAL, detect_jailbreak  # noqa: E402
from app.agent.intent_router import route_intent  # noqa: E402
from app.agent.state import AgentState  # noqa: E402
from app import access_control  # noqa: E402
from app.auth import current_session, current_user, is_admin, require_admin, verify_openemr_credentials  # noqa: E402
from app.assignments_db import AssignmentStore  # noqa: E402
from app.auth_db import AuthStore  # noqa: E402
from app.document_fingerprints_db import DocumentFingerprintStore  # noqa: E402
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

SESSIONS: dict[str, AgentState] = {}

# Cache TTL is generous — clinical chart data doesn't change second-to-second,
# and a stale 5-minute read is preferable to making the user wait 8s. Prewarm
# at startup makes the first read after server boot instant; this TTL keeps
# subsequent reads instant for the duration of a demo or a clinic session.
DATA_CACHE_TTL_S = 300.0


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
    for state in SESSIONS.values():
        if state.get("username") != username:
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
        return ChatResponse(
            session_id=session_id,
            response=JAILBREAK_REFUSAL,
            patient_id=SESSIONS.get(session_id, {}).get("patient_id"),
            sources=SESSIONS.get(session_id, {}).get("conversation_sources", []),
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
        sess = SESSIONS.get(session_id, {})
        return ChatResponse(
            session_id=session_id,
            response=routed.response,
            patient_id=sess.get("patient_id"),
            sources=sess.get("conversation_sources", []),
            validation_warning=False,
        )

    state = SESSIONS.get(session_id) or _fresh_state()
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
        result = await app.state.graph.ainvoke(
            state,
            config={"callbacks": [TokenUsageCallback()]},
        )
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
    SESSIONS[session_id] = new_state

    last = new_state["messages"][-1]
    text = message_text(last) if isinstance(last, AIMessage) else ""

    trace.validator_attempts = new_state["validation_attempts"]
    trace.validator_failed = new_state["validation_attempts"] >= MAX_VALIDATION_ATTEMPTS
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
        sess = SESSIONS.get(session_id, {})

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
        sess = SESSIONS.get(session_id, {})

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

    state = SESSIONS.get(session_id) or _fresh_state()
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
        try:
            yield f"data: {json.dumps({'type': 'session', 'session_id': session_id, 'request_id': trace.request_id})}\n\n"

            async for event in app.state.graph.astream_events(
                state,
                version="v2",
                config={"callbacks": [TokenUsageCallback()]},
            ):
                ev = event.get("event")
                data = event.get("data") or {}

                if ev == "on_tool_start":
                    yield f"data: {json.dumps({'type': 'tool', 'name': event.get('name'), 'phase': 'start'})}\n\n"
                elif ev == "on_chat_model_stream":
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
                            SESSIONS[session_id] = new_state

            sess = SESSIONS.get(session_id, {})
            trace.validator_attempts = sess.get("validation_attempts", 0)
            trace.validator_failed = trace.validator_attempts >= MAX_VALIDATION_ATTEMPTS
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
    existing_names = {(r.get("name") or "").lower() for r in fhir_rows}
    bp_already_listed = any("systolic" in n or "diastolic" in n for n in existing_names)
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
    note_only_vitals: list[dict] = []
    for n in [*unsynced_finals, *synced_notes]:
        # Use clinical-local TZ to match `fhir_rows[].time` so the frontend's
        # minute-precision group key lines up across both sources. Without
        # this conversion, a synced note whose FHIR roundtrip dropped a key
        # would surface here in UTC and split into a separate group on the
        # card — and the slice-to-1 view would render only one of them.
        when = _clinical_iso(n.finalized_at or n.updated_at)
        for canonical, value in (n.vitals or {}).items():
            if value in (None, ""):
                continue
            if _already_in_fhir(canonical, when):
                continue
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
    return {"items": items}


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

    try:
        if doc_type == "intake_form":
            from app.extraction.extract import extract_only  # noqa: PLC0415
            extracted = await extract_only(
                file_bytes=file_bytes,
                doc_type="intake_form",
                mime_type=mime_type,
            )
            demo = extracted.demographics
            identity = {
                "given_name": demo.given_name,
                "family_name": demo.family_name,
                "date_of_birth": demo.date_of_birth.isoformat() if demo.date_of_birth else None,
                "sex": demo.sex,
                "address": demo.address,
                "phone": demo.phone,
            }
            chief_concern = extracted.chief_concern
        elif doc_type == "hl7_message":
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
        elif doc_type in ("fax_packet", "referral_letter"):
            from app.extraction.extract import extract_only  # noqa: PLC0415
            extracted = await extract_only(
                file_bytes=file_bytes,
                doc_type=doc_type,  # type: ignore[arg-type]
                mime_type=mime_type,
            )
            identity = _identity_from_optional(
                getattr(extracted, "patient_identity", None),
            )
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
    given_clean = body.given_name.strip()
    family_clean = body.family_name.strip()

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
            "create_patient: idempotent hit, reusing existing uuid=%s for %s, %s",
            new_uuid, family_clean, given_clean,
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
                sex=body.sex,
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
        "create_patient: user=%s pid=%s uuid=%s name=%s, %s",
        username, new_pid, new_uuid, body.family_name, body.given_name,
    )
    return {
        "patient_uuid": new_uuid,
        "pid": new_pid,
        "given_name": body.given_name,
        "family_name": body.family_name,
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
    await _check_patient_access(request, username, patient_uuid)
    if doc_type not in DOC_TYPE_LABELS:
        raise HTTPException(
            status_code=400,
            detail=f"unsupported doc_type {doc_type!r}; "
                   f"expected one of {list(DOC_TYPE_LABELS)}",
        )
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="empty file upload")
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
    # is the user-visible "you already uploaded this" surface. Run the
    # SHA lookup before attach_and_extract so a confirmed-duplicate
    # upload doesn't re-spend Anthropic credits on a re-extraction
    # whose facts are already persisted.
    if not acknowledge_existing:
        sha_hex = hashlib.sha256(file_bytes).hexdigest()
        try:
            existing_ref = await app.state.openemr_writer.find_document_reference_by_sha(
                patient_uuid, sha_hex,
            )
        except Exception as e:  # noqa: BLE001
            log.warning("dedup precheck failed; proceeding without it: %s", e)
            existing_ref = None
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


@app.get("/api/document/{document_id}/pages")
async def api_document_pages(
    document_id: str,
    request: Request,
    username: str = Depends(current_user),
) -> dict:
    """Return rendered PNG pages of a `DocumentReference` for the
    Supporting-Documents inline viewer (PRD W2 §5 — visual PDF
    bounding-box overlay).

    The adapter performs the panel ACL check using the document's
    `subject.reference`; we still resolve the panel here because the
    adapter takes a `panel` arg (same shape it already uses for the
    agent's `get_document_content` tool path). 404 on access denial,
    matching the patient-id-keyed endpoints — no panel-leakage via
    response codes.
    """
    panel = await access_control.get_panel_for_user(
        request.app.state.fhir, username, request.app.state.assignments,
    )
    try:
        result = await adapter.get_document_pages(
            request.app.state.fhir,
            document_id=document_id,
            panel=panel,
        )
    except access_control.PatientAccessDenied as e:
        request.app.state.auth_store.log_event(
            event_type="patient_access_denied",
            username=username,
            sid=request.session.get("sid"),
            user_agent=request.headers.get("user-agent"),
            ip=request.client.host if request.client else None,
            detail=f"document={document_id} patient={e.patient_id}",
        )
        raise HTTPException(status_code=404, detail="Document not found") from e
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"FHIR fetch failed: {e!s}") from e
    return result["data"]


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
    try:
        result = await adapter.get_document_bbox_manifest(
            request.app.state.fhir,
            document_id=document_id,
            panel=panel,
        )
    except access_control.PatientAccessDenied as e:
        request.app.state.auth_store.log_event(
            event_type="patient_access_denied",
            username=username,
            sid=request.session.get("sid"),
            user_agent=request.headers.get("user-agent"),
            ip=request.client.host if request.client else None,
            detail=f"document={document_id} patient={e.patient_id}",
        )
        raise HTTPException(status_code=404, detail="Document not found") from e
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"FHIR fetch failed: {e!s}") from e
    return result["data"]


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
    _user: str = Depends(current_user),
):
    """Proxy a FHIR Binary resource to the browser as raw bytes.

    OpenEMR returns DocumentReference.content[].attachment.url as
    `https://localhost:9300/.../Binary/{id}` — that points at the
    OpenEMR container behind cloudflared and requires an OAuth bearer
    the browser can't see. This endpoint fetches the Binary server-side
    using the agent's existing OAuth client and streams the decoded
    bytes back, so the front-end can use a normal `<a href>` to open
    the document.

    Auth: any logged-in user. The endpoint does not yet ACL-walk back
    to the parent DocumentReference -> patient -> panel; tighten this
    before any non-demo deployment.

    Implementation: OpenEMR's `GET /fhir/Binary/{id}` returns the raw
    file bytes (PDF/PNG/etc.) with the original Content-Type header,
    not a FHIR JSON envelope with base64 `data`. Use the dedicated
    `FhirClient.get_raw` so we don't try to parse PNG bytes as JSON.
    """
    try:
        body, content_type = await app.state.fhir.get_raw(f"Binary/{binary_id}")
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
    _user: str = Depends(current_user),
) -> dict:
    """Newest-first list of recent request traces.

    Each entry includes latency, token totals, $ cost, validator outcome,
    tool-call count, and per-tool detail. Bounded ring buffer (200) — no
    pagination beyond `limit`.
    """
    items = [t.to_dict() for t in app.state.traces.list_recent(limit=limit)]
    return {"count": len(items), "items": items}


@app.get("/api/traces/{request_id}")
async def get_trace(request_id: str, _user: str = Depends(current_user)) -> dict:
    trace = app.state.traces.get(request_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="trace not found")
    return trace.to_dict()


@app.get("/observability", response_model=None)
async def observability_page(request: Request) -> FileResponse | RedirectResponse:
    if current_session(request) is None:
        return RedirectResponse(url="/login", status_code=302)
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
