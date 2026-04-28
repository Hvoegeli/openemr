FROM python:3.12-slim AS base

# uv for fast dependency resolution; pin a known-good version
RUN pip install --no-cache-dir uv==0.4.30

WORKDIR /app

# Cache deps separately from source — pyproject + lockfile change less often
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# App + private key (image is private; production should mount this as a
# Fly secret file, see AUDIT.md). Anthropic + OAuth credentials come from
# `fly secrets set` at runtime; we never bake those into the image.
COPY app/ ./app/
COPY secrets/agent_key.pem ./secrets/agent_key.pem

ENV PYTHONPATH=/app \
    OPENEMR_PRIVATE_KEY_PATH=/app/secrets/agent_key.pem \
    OPENEMR_KID=agent_forge_key_1 \
    PORT=8000

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
