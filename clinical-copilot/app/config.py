"""Central env-driven config.

Reads .env at import time. Fail fast (with a clear message) if a required
secret is missing rather than crashing deep in some HTTP call.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM
    anthropic_api_key: str = Field(..., description="Claude API key, sk-ant-...")

    # Observability (optional — disable tracing if no key)
    langsmith_api_key: str | None = None
    langsmith_tracing: bool = False
    langsmith_project: str = "agent_forge"

    # OpenEMR FHIR
    openemr_fhir_base_url: str = "https://localhost:9300/apis/default/fhir"
    openemr_oauth_token_url: str = "https://localhost:9300/oauth2/default/token"
    openemr_client_id: str | None = None
    openemr_client_secret: str | None = None
    openemr_private_key_path: str = "secrets/agent_key.pem"
    openemr_kid: str = "agent_forge_key_1"

    # OpenEMR seed client — also used for password-grant credential validation
    # (proves a username/password pair is valid against OpenEMR without us
    # having to hit the legacy login endpoints).
    openemr_seed_client_id: str | None = None
    openemr_seed_client_secret: str | None = None

    # Co-pilot cookie sessions. Random fallback OK for dev; set explicitly in prod.
    copilot_session_secret: str = "dev-secret-change-me-please-32bytes"

    # App database
    database_url: str = "postgresql://agent_forge:dev@localhost:5432/agent_forge"

    # Clinical timezone — drives shift detection and the "MMDDYYYY - Day/Night
    # Shift" label on clinical notes. UTC default keeps existing behavior; the
    # Hetzner deploy is set to America/Denver via systemd Environment=. Audit
    # and storage timestamps stay UTC regardless (FHIR-compliant); only the
    # *clinical interpretation* of "what shift was this?" honors this setting.
    clinical_tz: str = Field("UTC", description="IANA timezone name, e.g. America/Denver")


settings = Settings()
