"""Citation validator.

Scans the latest assistant message for inline `[ResourceType/ID]` citations
and verifies each ID was actually returned by a tool call earlier in the
conversation. Citations not in `state.conversation_sources` are rejected.

The validator is the structural backstop for the prompt-level citation rule:
even if the LLM tries to confabulate a resource ID, it cannot get past this
node without retrying with valid sources or refusing the claim.
"""

import re

CITATION_RE = re.compile(r"\[([A-Z][a-zA-Z]+)/([a-zA-Z0-9._-]+)\]")


def extract_citations(text: str) -> list[str]:
    """Return every `Type/ID` found in inline `[Type/ID]` citations."""
    return [f"{m.group(1)}/{m.group(2)}" for m in CITATION_RE.finditer(text)]


def find_invalid_citations(text: str, allowed_sources: list[str]) -> list[str]:
    """Return citations from `text` that are not in `allowed_sources` (deduped, ordered)."""
    allowed = set(allowed_sources)
    seen: set[str] = set()
    invalid: list[str] = []
    for cite in extract_citations(text):
        if cite in allowed or cite in seen:
            continue
        seen.add(cite)
        invalid.append(cite)
    return invalid
