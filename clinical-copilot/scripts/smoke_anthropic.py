"""Verify the Anthropic API key in .env works end-to-end through LangChain.

Run: uv run python scripts/smoke_anthropic.py
"""

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

load_dotenv()

model = ChatAnthropic(model_name="claude-sonnet-4-6", timeout=30, stop=None)
result = model.invoke("Reply with exactly: Hello from Claude Sonnet 4.6.")
print(result.content)
