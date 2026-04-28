"""FastAPI entry point for the clinical co-pilot.

Endpoints:
  GET  /                        - chat UI (single static page)
  POST /chat                    - send a turn, get the assistant response (atomic)
  POST /chat/stream             - same, but streams tokens + tool-progress as SSE
  GET  /api/patient/{id}/card   - structured patient-card JSON

Conversation state is held in-memory per `session_id`. This is fine for the
MVP; production would persist to Postgres alongside the audit log.
"""

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel

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
from app.fhir import adapter  # noqa: E402
from app.fhir.client import FhirClient  # noqa: E402

WEB_DIR = Path(__file__).parent / "web"

SESSIONS: dict[str, AgentState] = {}


def _fresh_state() -> AgentState:
    return {
        "messages": [],
        "conversation_sources": [],
        "patient_id": None,
        "validation_attempts": 0,
    }


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.fhir = FhirClient()
    app.state.graph = build_graph(app.state.fhir)
    try:
        yield
    finally:
        await app.state.fhir.aclose()


app = FastAPI(title="Clinical Co-pilot", lifespan=lifespan)


class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str


class ChatResponse(BaseModel):
    session_id: str
    response: str
    patient_id: str | None
    sources: list[str]
    validation_warning: bool


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
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
async def chat_stream(req: ChatRequest) -> StreamingResponse:
    """SSE-streaming version of /chat.

    Emits three event kinds, all as `data: {json}\\n\\n` lines:
      - {type:'session', session_id} once at start
      - {type:'tool', name, args} when a tool fires (lets UI show "Searching chart...")
      - {type:'token', text} token chunks as the final LLM response streams
      - {type:'done', patient_id, sources, validation_warning} once at end

    The graph is driven via `astream_events`; only the final non-tool LLM
    output is streamed token-by-token, since intermediate tool-decision
    rounds have no user-visible text.
    """
    session_id = req.session_id or str(uuid4())
    state = SESSIONS.get(session_id) or _fresh_state()
    state["messages"] = [*state["messages"], HumanMessage(content=req.message)]
    state["validation_attempts"] = 0

    async def event_stream():
        yield f"data: {json.dumps({'type': 'session', 'session_id': session_id})}\n\n"

        final_message = None
        async for event in app.state.graph.astream_events(state, version="v2"):
            ev = event.get("event")
            data = event.get("data") or {}

            if ev == "on_tool_start":
                yield f"data: {json.dumps({'type': 'tool', 'name': event.get('name'), 'phase': 'start'})}\n\n"
            elif ev == "on_chat_model_stream":
                chunk = data.get("chunk")
                # Only stream text content; skip tool_use blocks.
                if chunk is not None and isinstance(chunk.content, str) and chunk.content:
                    yield f"data: {json.dumps({'type': 'token', 'text': chunk.content})}\n\n"
                elif chunk is not None and isinstance(chunk.content, list):
                    for blk in chunk.content:
                        if isinstance(blk, dict) and blk.get("type") == "text":
                            txt = blk.get("text", "")
                            if txt:
                                yield f"data: {json.dumps({'type': 'token', 'text': txt})}\n\n"
            elif ev == "on_chain_end" and event.get("name") == "LangGraph":
                # Final state for done payload
                output = data.get("output") or {}
                if isinstance(output, dict) and "messages" in output:
                    msgs = output["messages"]
                    if msgs:
                        final_message = msgs[-1]
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
            "X-Accel-Buffering": "no",  # disable proxy buffering
        },
    )


@app.get("/api/patient/{patient_id}/card")
async def patient_card(patient_id: str) -> dict:
    try:
        result = await adapter.get_patient_card(app.state.fhir, patient_id=patient_id)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"FHIR fetch failed: {e!s}") from e
    return result["data"]


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}
