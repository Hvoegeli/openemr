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


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.fhir = FhirClient()
    app.state.graph = build_graph(app.state.fhir)
    app.state.cache = TTLCache(ttl_seconds=DATA_CACHE_TTL_S)

    # Don't block startup on the prewarm — let it run while uvicorn binds.
    prewarm_task = asyncio.create_task(_prewarm_dashboard(app))
    try:
        yield
    finally:
        prewarm_task.cancel()
        await app.state.fhir.aclose()


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
async def chat(req: ChatRequest, _user: str = Depends(current_user)) -> ChatResponse:
    session_id = req.session_id or str(uuid4())
    state = SESSIONS.get(session_id) or _fresh_state()
    state["messages"] = [*state["messages"], HumanMessage(content=req.message)]
    state["validation_attempts"] = 0

    result = await app.state.graph.ainvoke(state)

    new_state: AgentState = {
        "messages": result["messages"],
        "conversation_sources": result["conversation_sources"],
        "patient_id": result.get("patient_id"),
        "validation_attempts": result.get("validation_attempts", 0),
    }
    SESSIONS[session_id] = new_state

    last = new_state["messages"][-1]
    text = message_text(last) if isinstance(last, AIMessage) else ""

    return ChatResponse(
        session_id=session_id,
        response=text,
        patient_id=new_state["patient_id"],
        sources=new_state["conversation_sources"],
        validation_warning=new_state["validation_attempts"] >= MAX_VALIDATION_ATTEMPTS,
    )


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest, _user: str = Depends(current_user)) -> StreamingResponse:
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

    async def event_stream():
        yield f"data: {json.dumps({'type': 'session', 'session_id': session_id})}\n\n"

        async for event in app.state.graph.astream_events(state, version="v2"):
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

        done_payload = {
            "type": "done",
            "patient_id": SESSIONS.get(session_id, {}).get("patient_id"),
            "sources": SESSIONS.get(session_id, {}).get("conversation_sources", []),
            "validation_warning": (
                SESSIONS.get(session_id, {}).get("validation_attempts", 0)
                >= MAX_VALIDATION_ATTEMPTS
            ),
        }
        yield f"data: {json.dumps(done_payload)}\n\n"

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


@app.get("/api/patient/{patient_id}/card")
async def patient_card(patient_id: str, _user: str = Depends(current_user)) -> dict:
    async def _compute() -> dict:
        result = await adapter.get_patient_card(app.state.fhir, patient_id=patient_id)
        return result["data"]
    try:
        return await app.state.cache.get_or_compute(f"card:{patient_id}", _compute)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"FHIR fetch failed: {e!s}") from e


@app.get("/api/patient/{patient_id}/documents")
async def patient_documents(patient_id: str, _user: str = Depends(current_user)) -> dict:
    async def _compute() -> dict:
        result = await get_supporting_documents(app.state.fhir, patient_id=patient_id)
        return result["data"]
    try:
        return await app.state.cache.get_or_compute(f"docs:{patient_id}", _compute)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"FHIR fetch failed: {e!s}") from e


@app.get("/api/calendar/today")
async def calendar_today(_user: str = Depends(current_user)) -> dict:
    async def _compute() -> dict:
        result = await get_calendar_today(app.state.fhir)
        return result["data"]
    try:
        return await app.state.cache.get_or_compute("calendar:today", _compute)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"FHIR fetch failed: {e!s}") from e


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}
