"""FastAPI entry point for the clinical co-pilot.

Endpoints:
  GET  /                        - chat UI (single static page; redirects to /login if anonymous)
  GET  /login                   - login page (uses OpenEMR creds)
  POST /api/login               - validate creds via OpenEMR password grant, set session cookie
  POST /api/logout              - clear session
  GET  /api/me                  - who am I? (UI uses this to render header)
  POST /chat                    - send a turn, get the assistant response (atomic)
  POST /chat/stream             - same, but streams tokens + tool-progress as SSE
  GET  /api/patient/{id}/card        - structured patient-card JSON
  GET  /api/patient/{id}/documents   - past encounters + clinical documents
  GET  /api/calendar/today           - today's roster (Appointments, fallback)

Conversation state is held in-memory per `session_id`. This is fine for the
MVP; production would persist to Postgres alongside the audit log.
"""

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
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

from app.agent.graph import MAX_VALIDATION_ATTEMPTS, build_graph, message_text  # noqa: E402
from app.agent.state import AgentState  # noqa: E402
from app.auth import current_user, verify_openemr_credentials  # noqa: E402
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

    # Don't block startup on the prewarm — let it run while uvicorn binds.
    prewarm_task = asyncio.create_task(_prewarm_dashboard(app))
    try:
        yield
    finally:
        prewarm_task.cancel()
        await app.state.fhir.aclose()
        await app.state.openemr_writer.aclose()
        app.state.traces.close()


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


# ─── auth + shell ────────────────────────────────────────────────────────


@app.get("/", response_model=None)
async def index(request: Request) -> FileResponse | RedirectResponse:
    if not request.session.get("username"):
        return RedirectResponse(url="/login", status_code=302)
    return FileResponse(WEB_DIR / "index.html")


@app.get("/login", response_model=None)
async def login_page(request: Request) -> FileResponse | RedirectResponse:
    if request.session.get("username"):
        return RedirectResponse(url="/", status_code=302)
    return FileResponse(WEB_DIR / "login.html")


@app.post("/api/login")
async def login(req: LoginRequest, request: Request) -> dict:
    try:
        ok = await verify_openemr_credentials(req.username, req.password)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    if not ok:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    request.session["username"] = req.username
    return {"ok": True, "username": req.username}


@app.post("/api/logout")
async def logout(request: Request) -> dict:
    request.session.clear()
    return {"ok": True}


@app.get("/api/me")
async def me(username: str = Depends(current_user)) -> dict:
    return {"username": username}


# ─── chat ────────────────────────────────────────────────────────────────


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request, _user: str = Depends(current_user)) -> ChatResponse:
    session_id = req.session_id or str(uuid4())
    state = SESSIONS.get(session_id) or _fresh_state()
    state["messages"] = [*state["messages"], HumanMessage(content=req.message)]
    state["validation_attempts"] = 0

    trace = new_request_trace(
        session_id=session_id,
        username=request.session.get("username", ""),
        user_msg=req.message,
        model=MODEL_NAME,
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
    state = SESSIONS.get(session_id) or _fresh_state()
    state["messages"] = [*state["messages"], HumanMessage(content=req.message)]
    state["validation_attempts"] = 0

    trace = new_request_trace(
        session_id=session_id,
        username=request.session.get("username", ""),
        user_msg=req.message,
        model=MODEL_NAME,
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
    if not request.session.get("username"):
        return RedirectResponse(url="/login", status_code=302)
    return FileResponse(WEB_DIR / "observability.html")
