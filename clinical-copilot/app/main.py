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
see `app/auth_db.py`.
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
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage
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
from app.auth import current_session, current_user, is_admin, require_admin, verify_openemr_credentials  # noqa: E402
from app.auth_db import AuthStore  # noqa: E402
from app.cache import TTLCache  # noqa: E402
from app.config import settings  # noqa: E402
from app.fhir import adapter  # noqa: E402
from app.fhir.client import FhirClient  # noqa: E402
from app.fhir.extras import get_calendar_today, get_supporting_documents  # noqa: E402
from app.fhir.writer import OpenEMRWriter, OpenEMRWriteError  # noqa: E402
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
    }


async def _prewarm_dashboard(app: FastAPI) -> None:
    """Background task: warm OAuth, load calendar + every patient on it.

    Crucial: every fetch routes through `cache.get_or_compute` — so if a
    user request lands mid-prewarm, both share the same in-flight call
    via the per-key lock. No duplicate work, no thundering herd.

    OpenEMR's FHIR layer serializes responses on the server side (PHP
    session locking + MariaDB), so client-side parallelism saves only a
    little. The real win is caching: the user pays this latency once at
    server boot, never on a click.
    """
    cache: TTLCache = app.state.cache
    fhir: FhirClient = app.state.fhir
    t0 = time.time()
    try:
        await fhir._ensure_token()  # noqa: SLF001
        log.info("prewarm: oauth token ready in %.2fs", time.time() - t0)

        async def _calendar() -> dict:
            r = await get_calendar_today(fhir)
            return r["data"]
        cal_data = await cache.get_or_compute("calendar:today", _calendar)
        log.info(
            "prewarm: calendar ready in %.2fs (%d patients)",
            time.time() - t0, len(cal_data.get("patients", [])),
        )

        async def _warm_patient(pid: str) -> None:
            async def _card() -> dict:
                r = await adapter.get_patient_card(fhir, patient_id=pid)
                return r["data"]

            async def _docs() -> dict:
                r = await get_supporting_documents(fhir, patient_id=pid)
                return r["data"]

            await asyncio.gather(
                cache.get_or_compute(f"card:{pid}", _card),
                cache.get_or_compute(f"docs:{pid}", _docs),
            )

        await asyncio.gather(*[
            _warm_patient(p["patient_id"])
            for p in cal_data.get("patients", [])
            if p.get("patient_id")
        ])
        log.info("prewarm: dashboard fully warm in %.2fs", time.time() - t0)
    except Exception as e:  # noqa: BLE001
        log.warning("prewarm failed (non-fatal): %s", e)


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
    app.state.graph = build_graph(
        app.state.fhir, app.state.clinical_notes, model_name=MODEL_NAME,
    )
    app.state.cache = TTLCache(ttl_seconds=DATA_CACHE_TTL_S)
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


class ChatResponse(BaseModel):
    session_id: str
    response: str
    patient_id: str | None
    sources: list[str]
    validation_warning: bool


class LoginRequest(BaseModel):
    username: str
    password: str


class SignOutRequest(BaseModel):
    """Optional `patient_ids`: when omitted, defaults to today's calendar.
    Letting the doctor draft for a subset (e.g. "just the ICU bed-block")
    means we accept an explicit list as well."""
    patient_ids: list[str] | None = None


class SignOutDraft(BaseModel):
    patient_id: str
    name: str | None = None
    draft: str
    sources: list[str]
    validation_warning: bool
    error: str | None = None


class SignOutResponse(BaseModel):
    drafts: list[SignOutDraft]


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


@app.get("/api/patient/{patient_id}/card")
async def patient_card(patient_id: str, _user: str = Depends(current_user)) -> dict:
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
async def patient_vital_trends(patient_id: str, _user: str = Depends(current_user)) -> dict:
    try:
        return await app.state.cache.get_or_compute(
            f"trends:{patient_id}",
            lambda: _vital_trends_compute(patient_id),
        )
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"FHIR fetch failed: {e!s}") from e


@app.get("/api/patient/{patient_id}/documents")
async def patient_documents(patient_id: str, _user: str = Depends(current_user)) -> dict:
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
    return {"items": items}


@app.get("/api/calendar/today")
async def calendar_today(_user: str = Depends(current_user)) -> dict:
    async def _compute() -> dict:
        result = await get_calendar_today(app.state.fhir)
        return result["data"]
    try:
        return await app.state.cache.get_or_compute("calendar:today", _compute)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"FHIR fetch failed: {e!s}") from e


# ─── sign-out drafting (Use Case C) ──────────────────────────────────────


SIGN_OUT_USER_PROMPT = (
    "Draft a sign-out for patient_id={patient_id} ({name}). Use the "
    "exact ONE-LINER / EVENTS TODAY / ACTIVE CONCERNS / OVERNIGHT TO-DOS "
    "format from the system prompt. No preamble, no closing footer. Pull "
    "the chart with get_patient_card before drafting; consult "
    "get_observations_24h, get_med_changes_24h, or get_notes_24h if the "
    "doctor would care about overnight changes."
)


async def _draft_signout_for_patient(
    patient_id: str, name: str | None, username: str,
) -> SignOutDraft:
    """Run one full agent invocation to produce a sign-out for one patient.

    Each draft gets its own RequestTrace so the audit log captures it
    alongside chat traces — same SQLite table, same dashboard view. The
    citation validator still gates the output: a draft with a fake or
    missing citation triggers the same retry edge as any other turn.
    """
    state = _fresh_state()
    state["patient_id"] = patient_id
    user_msg = SIGN_OUT_USER_PROMPT.format(patient_id=patient_id, name=name or "?")
    state["messages"] = [HumanMessage(content=user_msg)]

    trace = new_request_trace(
        session_id=f"signout:{patient_id}",
        username=username,
        user_msg=user_msg,
        model=active_model_label(MODEL_NAME),
    )
    token = set_current_trace(trace)
    try:
        try:
            result = await app.state.graph.ainvoke(
                state,
                config={"callbacks": [TokenUsageCallback()]},
            )
        except Exception as e:  # noqa: BLE001
            trace.error = f"{type(e).__name__}: {e}"
            trace.finalize()
            app.state.traces.add(trace)
            return SignOutDraft(
                patient_id=patient_id, name=name, draft="",
                sources=[], validation_warning=False,
                error=trace.error,
            )

        last = result["messages"][-1]
        text = message_text(last) if isinstance(last, AIMessage) else ""
        attempts = result.get("validation_attempts", 0)

        trace.validator_attempts = attempts
        trace.validator_failed = attempts >= MAX_VALIDATION_ATTEMPTS
        trace.finalize()
        app.state.traces.add(trace)

        return SignOutDraft(
            patient_id=patient_id,
            name=name,
            draft=text,
            sources=list(result["conversation_sources"]),
            validation_warning=attempts >= MAX_VALIDATION_ATTEMPTS,
        )
    finally:
        reset_current_trace(token)


@app.post("/api/sign-out/draft", response_model=SignOutResponse)
async def sign_out_draft(
    req: SignOutRequest,
    user: str = Depends(current_user),
) -> SignOutResponse:
    """Generate per-patient sign-out drafts for the doctor's panel.

    With no body (or empty `patient_ids`), defaults to today's calendar
    so the doctor's "draft sign-outs for my list" lands as a single click.
    Each patient is drafted concurrently — wall-clock time is roughly the
    longest single draft, not the sum.
    """
    cal_data: dict = {}
    if not req.patient_ids:
        async def _compute() -> dict:
            result = await get_calendar_today(app.state.fhir)
            return result["data"]
        try:
            cal_data = await app.state.cache.get_or_compute("calendar:today", _compute)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=502, detail=f"calendar fetch failed: {e!s}") from e
        roster = [
            (p["patient_id"], p.get("name"))
            for p in cal_data.get("patients", [])
            if p.get("patient_id")
        ]
    else:
        # When the caller supplies IDs explicitly, do a quick pass on the
        # cached calendar (if present) just to surface names in the response.
        # If the calendar isn't cached yet we still draft — names just stay None.
        cal_data = app.state.cache.get("calendar:today") or {}
        name_by_id = {
            p["patient_id"]: p.get("name")
            for p in cal_data.get("patients", []) if p.get("patient_id")
        }
        roster = [(pid, name_by_id.get(pid)) for pid in req.patient_ids]

    if not roster:
        return SignOutResponse(drafts=[])

    drafts = await asyncio.gather(*[
        _draft_signout_for_patient(pid, name, user)
        for pid, name in roster
    ])
    return SignOutResponse(drafts=drafts)


# ─── clinical notes ──────────────────────────────────────────────────────


class ClinicalNoteRequest(BaseModel):
    notes_md: str = ""
    recs_md: str = ""
    vitals: dict | None = None


@app.get("/api/patient/{patient_id}/clinical-notes/draft")
async def get_clinical_note_draft(
    patient_id: str,
    username: str = Depends(current_user),
) -> dict:
    """Return the current author's open draft for this patient × current shift,
    or a stub indicating no draft exists yet."""
    note = app.state.clinical_notes.get_draft(patient_id, username, now=now_utc())
    if note is None:
        return {"draft": None}
    return {"draft": note.to_doc_item()}


@app.post("/api/patient/{patient_id}/clinical-notes/draft")
async def upsert_clinical_note_draft(
    patient_id: str,
    body: ClinicalNoteRequest,
    username: str = Depends(current_user),
) -> dict:
    """Create or update the author's open draft. Multiple saves within a
    shift consolidate into the same draft until it is finalized; once
    finalized, a subsequent save opens a fresh draft (an addendum)."""
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
    username: str = Depends(current_user),
) -> dict:
    """Explicit Save — promote the open draft to immutable 'final' status,
    then best-effort push the vitals to OpenEMR's vitals chart so the EHR
    sees what the doctor entered. A push failure does not roll back the
    local finalize — the note remains the canonical record either way.
    """
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
    _user: str = Depends(current_user),
) -> dict:
    """Most recent finalized note from a *prior* shift — what a doctor sees
    first when they click into a patient at the start of their shift."""
    note = app.state.clinical_notes.latest_prior_shift(patient_id, now=now_utc())
    return {"note": note.to_doc_item() if note else None}


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
    items: list[dict] = []
    for p in rows:
        name = (p.get("name") or [{}])[0]
        given = " ".join(name.get("given") or [])
        family = name.get("family") or ""
        full = (given + " " + family).strip() or "(unknown)"
        items.append({
            "id": p.get("id"),
            "name": full,
            "active": p.get("active"),
            "telecom": [
                {"system": t.get("system"), "value": t.get("value")}
                for t in (p.get("telecom") or [])
            ],
        })
    return {"count": len(items), "items": items}


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
