"""Deterministic intent router — handles trivial messages without an LLM call.

The chat endpoints call ``route_intent`` after the jailbreak guard but before
invoking the agent graph. If the user's message matches one of a small set
of strict, fully-anchored patterns (greetings, thanks, help), we return a
canned response immediately. Anything else falls through to the LLM as today.

Why this is a small list, not a large one:

  - The LLM is the right tool for clinical questions; routing should never
    intercept a real chart question. The cost of a false positive
    (intercepting a real question with a canned greeting) is much higher
    than the cost of a false negative (sending a "thanks" through the LLM).
  - Strict full-message anchoring + small pattern set keeps reasoning
    simple. "hi can you catch me up on Cohen" is NOT a greeting — it has
    chart-relevant content past "hi" — and our patterns won't match.
  - This is a fast path, not a router for every category. Real intent
    classification (which would need ML) is out of scope.

Each routed message still writes a RequestTrace to the audit log so the
dashboard accounts for them — but ``model="intent_router"`` and
``error="routed:<intent>"`` so they sort distinctly from LLM-handled turns.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class RoutedResponse:
    intent: str
    response: str


# Canned responses. Match the system-prompt style — terse, no clinical
# claims, end with a nudge toward the actual product (chart questions).
_GREETING_RESPONSE = (
    "Hi. Ask me about a patient on today's roster — try \"catch me up on "
    "Cohen.\""
)
_THANKS_RESPONSE = "You got it."
_HELP_RESPONSE = (
    "I'm a chart assistant for hospitalists. Ask me about a patient on "
    "your roster — pre-round summaries, vital trends, what changed "
    "overnight, sign-out drafts. Every clinical claim comes back with an "
    "inline `[ResourceType/ID]` citation you can click to jump to the "
    "chart record."
)


# Patterns are FULL-MESSAGE anchored (`^...$`) and case-insensitive. A
# message that has anything other than the greeting + optional punctuation
# falls through. This is on purpose — multi-clause messages might contain
# real clinical content past the greeting.
#
# Trailing punctuation (`!`, `.`, `?`, `,`, `:`) is allowed at most once,
# followed by optional whitespace, then end-of-string.
_TRAILER = r"\s*[!.?,:]?\s*$"

_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("greeting",
     re.compile(
         r"^\s*(?:"
         r"hi|hello|hey|howdy|"
         r"hi\s+there|hello\s+there|hey\s+there|"
         r"good\s+(?:morning|afternoon|evening)|"
         r"morning|afternoon|evening"
         r")" + _TRAILER,
         re.IGNORECASE,
     )),
    ("thanks",
     re.compile(
         r"^\s*(?:"
         r"thanks|thank\s+you|thank\s+you\s+very\s+much|"
         r"thx|ty|"
         r"appreciate\s+it|appreciated|"
         r"got\s+it|cool|"
         r"perfect|great|awesome"
         r")" + _TRAILER,
         re.IGNORECASE,
     )),
    ("help",
     re.compile(
         r"^\s*(?:"
         r"help|"
         r"what\s+can\s+you\s+do|what\s+do\s+you\s+do|"
         r"what\s+are\s+you|who\s+are\s+you|"
         r"how\s+do\s+i\s+use\s+(?:this|you)|"
         r"capabilities|features"
         r")(?:\s*\?)?" + _TRAILER,
         re.IGNORECASE,
     )),
]


_RESPONSES: dict[str, str] = {
    "greeting": _GREETING_RESPONSE,
    "thanks": _THANKS_RESPONSE,
    "help": _HELP_RESPONSE,
}


def route_intent(text: str) -> RoutedResponse | None:
    """Return a deterministic ``RoutedResponse`` when ``text`` matches one
    of the strict-anchored intent patterns; otherwise ``None`` (let the LLM
    handle it).

    Empty / non-string input returns ``None``.
    """
    if not isinstance(text, str) or not text.strip():
        return None
    for label, rx in _PATTERNS:
        if rx.match(text):
            return RoutedResponse(intent=label, response=_RESPONSES[label])
    return None
