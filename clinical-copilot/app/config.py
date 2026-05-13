"""Central env-driven config.

Reads .env at import time. Fail fast (with a clear message) if a required
secret is missing rather than crashing deep in some HTTP call.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM provider — "anthropic" (direct API) or "bedrock" (AWS Bedrock).
    # Bedrock is required for HIPAA-grade deployment because Anthropic only
    # offers a BAA via Bedrock, not via the direct API. Default stays
    # "anthropic" so dev / demo work unchanged; production flips to
    # "bedrock" via env and the AWS creds are picked up by boto3.
    llm_provider: str = "anthropic"

    # Anthropic-direct
    anthropic_api_key: str = Field(..., description="Claude API key, sk-ant-...")

    # AWS Bedrock — only used when llm_provider == "bedrock". boto3 picks
    # AWS credentials up from AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY env
    # vars or from ~/.aws/credentials; we don't read them here.
    aws_region: str = "us-east-1"
    bedrock_model_id: str = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"

    # Observability (optional — disable tracing if no key)
    langsmith_api_key: str | None = None
    langsmith_tracing: bool = False
    langsmith_project: str = "agent_forge"

    # OpenEMR FHIR
    openemr_fhir_base_url: str = "https://localhost:9300/apis/default/fhir"
    openemr_oauth_token_url: str = "https://localhost:9300/oauth2/default/token"
    openemr_oauth_authorize_url: str = "https://localhost:9300/oauth2/default/authorize"
    openemr_oauth_userinfo_url: str = "https://localhost:9300/oauth2/default/userinfo"
    openemr_client_id: str | None = None
    openemr_client_secret: str | None = None
    openemr_private_key_path: str = "secrets/agent_key.pem"
    openemr_kid: str = "agent_forge_key_1"
    # Verify the TLS cert on agent → OpenEMR HTTP calls. Default `True`
    # (production-safe). The local dev stack and the Hetzner box both
    # run OpenEMR behind a self-signed cert on `https://localhost:9300`,
    # so each opts out in its own `.env` (`OPENEMR_TLS_VERIFY=false`)
    # rather than the source tree carrying a "verify off" default.
    openemr_tls_verify: bool = True

    # Document fan-out cap for `get_notes_24h(hours=0)` (the "all docs
    # for this patient" mode). AgentForge's C5 DoS attack drives an
    # unbounded per-turn vision-model fan-out by chaining
    # `get_notes_24h(hours=0)` (returns every uploaded doc) into a
    # `get_document_content` call per document. Truncating the
    # newest-first doc list at this cap keeps the heaviest legitimate
    # case ("catch me up on this patient") within budget while closing
    # the unbounded enumeration. The cap fires only for `hours=0`; an
    # explicit time window leaves the result unbounded by count.
    max_notes_24h_docs: int = 20

    # Per-turn wall-clock budget (seconds). Wraps the LangGraph
    # ainvoke / astream_events call so a single chat turn cannot hold
    # the box hostage even if the per-LLM-call timeout (60s on the
    # answerer chain) lets through a long sequence of expensive calls.
    # 120s matches AgentForge's `evals/thresholds.yaml` envelope and
    # sits comfortably above the observed p95 normal-turn latency
    # (~68s) from `COST_LATENCY_REPORT.md`. On breach the request
    # fails closed (HTTPException(504) for `/chat`; SSE error event
    # then done for `/chat/stream`).
    max_turn_wall_seconds: int = 120

    # Copilot's own externally-reachable base URL — used to build the OAuth2
    # callback redirect_uri (`{copilot_base_url}/oauth/callback`). Local dev
    # default; production overrides via env (Hetzner cloudflared tunnel URL).
    copilot_base_url: str = "http://localhost:8000"

    # Externally-reachable base URL for classic OpenEMR. The Modern
    # Dashboard's "Open in classic OpenEMR" out-links use this as the
    # prefix. Empty string ⇒ the dashboard falls back to its built-in
    # default (https://localhost:9300, suitable for local dev only).
    # Production deploys override via OPENEMR_CLASSIC_BASE — on Hetzner
    # this is a separate cloudflared tunnel pointed at port 9300, since
    # the primary tunnel only exposes Co-Pilot at port 8000.
    openemr_classic_base: str = ""

    # OpenEMR seed client — also used for password-grant credential validation
    # (proves a username/password pair is valid against OpenEMR without us
    # having to hit the legacy login endpoints).
    openemr_seed_client_id: str | None = None
    openemr_seed_client_secret: str | None = None

    # Co-pilot cookie-session signing key. REQUIRED — no default, so a deploy
    # that forgets to set it fails fast at startup instead of silently
    # shipping a key that's published in the source tree. Generate with:
    #   python -c 'import secrets; print(secrets.token_hex(32))'
    copilot_session_secret: str = Field(..., min_length=32)

    # Admin users — comma-separated list of OpenEMR usernames who get
    # access to the /admin oversight page. Default is "admin" so the
    # OpenEMR seed admin user has admin privileges out of the box.
    # Production deploys override via ADMIN_USERNAMES env var.
    admin_usernames: str = "admin"

    # App database
    database_url: str = "postgresql://agent_forge:dev@localhost:5432/agent_forge"

    # Clinical timezone — drives shift detection and the "MMDDYYYY - Day/Night
    # Shift" label on clinical notes. UTC default keeps existing behavior; the
    # Hetzner deploy is set to America/Denver via systemd Environment=. Audit
    # and storage timestamps stay UTC regardless (FHIR-compliant); only the
    # *clinical interpretation* of "what shift was this?" honors this setting.
    clinical_tz: str = Field("UTC", description="IANA timezone name, e.g. America/Denver")


settings = Settings()
