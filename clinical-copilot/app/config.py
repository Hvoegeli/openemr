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

    # App database
    database_url: str = "postgresql://agent_forge:dev@localhost:5432/agent_forge"


settings = Settings()
