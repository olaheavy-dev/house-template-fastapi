"""Settings from the environment (and backend/.env locally), read once per process.

Names are the upper-case field names with no prefix (APP_ENV, LOG_LEVEL, DATABASE_URL, ...).
Secrets are SecretStr so they never print in logs or tracebacks; .env.example lists every setting
with a safe value.
"""

from __future__ import annotations

from decimal import Decimal
from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", frozen=True)

    app_env: Literal["local", "test", "production"] = "local"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    # JSON logs in production; readable console logs everywhere else.
    log_json: bool = False
    # Interactive API docs at /docs. Unset means on everywhere except production.
    docs_enabled: bool | None = None

    # SQLAlchemy URL with the psycopg 3 driver. The default matches backend/compose.yaml's local db.
    database_url: SecretStr = SecretStr("postgresql+psycopg://app:app@localhost:5432/app")
    # Connections per process: keep pool_size + max_overflow times the worker count under
    # Postgres's max_connections.
    db_pool_size: int = 5
    db_max_overflow: int = 10

    # The model the gateway calls, "provider:model id", with its prices in app/llm/prices.py.
    llm_model: str = "anthropic:claude-haiku-5-5"
    # Read by the gateway only; the provider's own variable names, so the usual secret works.
    anthropic_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    # Most output tokens per call; the budget reserves for this many.
    llm_max_tokens: int = Field(default=1024, gt=0)
    llm_timeout_seconds: float = Field(default=60.0, gt=0)
    # The provider SDK's retries (rate limits, overload, connection errors): the only retry layer.
    llm_max_retries: int = Field(default=2, ge=0)
    # Model spend allowed per UTC day across every process; calls past it get a 503.
    llm_daily_budget_usd: Decimal = Field(default=Decimal("5"), gt=0)

    # LangSmith tracing of model calls (https://docs.langchain.com/langsmith). Never on in tests.
    langsmith_tracing: bool = False
    langsmith_api_key: SecretStr | None = None
    langsmith_project: str | None = None
    langsmith_endpoint: str | None = None
    # Share of calls traced, 0 to 1: production rarely needs every call.
    langsmith_tracing_sampling_rate: float = Field(default=1.0, ge=0, le=1)
    # Leave prompts and answers out of traces when they may hold personal data.
    langsmith_hide_inputs: bool = False
    langsmith_hide_outputs: bool = False

    @property
    def docs_on(self) -> bool:
        return self.app_env != "production" if self.docs_enabled is None else self.docs_enabled


@lru_cache
def get_settings() -> Settings:
    return Settings()
