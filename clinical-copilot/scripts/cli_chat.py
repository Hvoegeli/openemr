"""Interactive CLI driver for the clinical co-pilot agent.

Run: PYTHONPATH=. uv run python scripts/cli_chat.py

Each turn:
  - Reads a line from stdin.
  - Invokes the LangGraph with cumulative state.
  - Prints the assistant's text response.
  - Warns if the validator flagged uncited claims after the retry budget.

Type 'exit' (or Ctrl-D) to quit. Type 'reset' to start a fresh conversation.
"""

import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

from app.agent.graph import MAX_VALIDATION_ATTEMPTS, build_graph
from app.agent.state import AgentState
from app.clinical_notes import ClinicalNoteStore
from app.fhir.client import FhirClient

load_dotenv()


def fresh_state() -> AgentState:
    return {
        "messages": [],
        "conversation_sources": [],
        "patient_id": None,
        "validation_attempts": 0,
    }


async def main() -> None:
    client = FhirClient()
    # Same notes path the FastAPI app uses, so CLI sessions see drafts
    # written via the web UI (and `get_vital_trends` reflects them).
    notes_path = Path(os.environ.get("CLINICAL_NOTES_PATH", "data/clinical_notes.json"))
    notes_store = ClinicalNoteStore(notes_path)
    graph = build_graph(client, notes_store)
    state = fresh_state()
    print("Clinical co-pilot — type 'exit' to quit, 'reset' for new conversation.\n")

    try:
        while True:
            try:
                user = input("you> ").strip()
            except EOFError:
                print()
                break
            if not user:
                continue
            if user.lower() in {"exit", "quit"}:
                break
            if user.lower() == "reset":
                state = fresh_state()
                print("(conversation reset)\n")
                continue

            state["messages"] = [*state["messages"], HumanMessage(content=user)]
            state["validation_attempts"] = 0

            result = await graph.ainvoke(state)
            state = {
                "messages": result["messages"],
                "conversation_sources": result["conversation_sources"],
                "patient_id": result.get("patient_id"),
                "validation_attempts": result.get("validation_attempts", 0),
            }

            last = state["messages"][-1]
            if isinstance(last, AIMessage) and isinstance(last.content, str):
                print(f"\nassistant> {last.content}\n")
            else:
                print("\n(no assistant text in response)\n", file=sys.stderr)

            if state["validation_attempts"] >= MAX_VALIDATION_ATTEMPTS:
                print(
                    "[warning] response retained uncited claims after retry; "
                    "verify against the chart.\n",
                    file=sys.stderr,
                )
    finally:
        await client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
