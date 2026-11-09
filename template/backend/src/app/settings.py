"""Settings from the environment (and backend/.env locally), read once per process.

Names are the upper-case field names with no prefix (APP_ENV, LOG_LEVEL, DATABASE_URL, ...).
Secrets are SecretStr so they never print in logs or tracebacks; .env.example lists every setting
with a safe value.
"""

from __future__ import annotations

import os
from decimal import Decimal
from functools import lru_cache
from typing import Any, Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# The local development values (backend/compose/); production must set its own.
LOCAL_DATABASE_URL = "postgresql+psycopg://app:app@localhost:5432/app"
LOCAL_STORAGE_SECRET = "app-storage-secret"  # noqa: S105 - the local RustFS's, refused in production


class Settings(BaseSettings):
    # ENV_FILE="" turns the file off: the tests set it before any app code loads, so a
    # developer's backend/.env (real keys included) never reaches them.
    model_config = SettingsConfigDict(
        env_file=os.environ.get("ENV_FILE", ".env") or None, extra="ignore", frozen=True
    )

    app_env: Literal["local", "test", "production"] = "local"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    # JSON logs in production; readable console logs everywhere else.
    log_json: bool = False
    # Largest request body the API reads, in bytes; larger ones get a 413.
    # Files go to storage through presigned URLs, not through the API.
    max_request_bytes: int = Field(default=1_048_576, gt=0)
    # Interactive API docs at /docs. Unset means on everywhere except production.
    docs_enabled: bool | None = None

    # SQLAlchemy URL with the psycopg 3 driver. The default matches the local db in
    # backend/compose/db/compose.yaml.
    database_url: SecretStr = SecretStr(LOCAL_DATABASE_URL)
    # Connections per process: keep pool_size + max_overflow times the worker count under
    # Postgres's max_connections.
    db_pool_size: int = 5
    db_max_overflow: int = 10
    # Server-side limits on every connection: a runaway query, or a transaction left open while
    # the request waits on something else, is cut off instead of holding a connection (and its
    # locks) indefinitely. Milliseconds.
    db_statement_timeout_ms: int = Field(default=30_000, gt=0)
    db_idle_in_transaction_timeout_ms: int = Field(default=60_000, gt=0)

    # Browser origins allowed to call the API (CORS), e.g. ["https://app.example.com"]. Empty
    # means none: same-origin and server-to-server calls only.
    cors_origins: list[str] = []
    # Host headers the API answers to, e.g. ["api.example.com"]; ["*"] accepts any.
    allowed_hosts: list[str] = ["*"]

    # S3-compatible object storage (app/storage.py). Unset STORAGE_BUCKET means no storage.
    # Locally: RustFS from backend/compose/storage/compose.yaml. Production on R2: the account's
    # https://<account id>.r2.cloudflarestorage.com endpoint, region "auto" and an R2 API token.
    storage_bucket: str | None = None
    storage_endpoint_url: str | None = "http://localhost:9000"
    # Where browsers reach the storage, when that differs from the endpoint above (the Compose
    # API reaches storage:9000; a browser, localhost:9000). Presigned URLs are signed for it.
    storage_public_endpoint_url: str | None = None
    storage_region: str = "us-east-1"
    storage_access_key_id: SecretStr | None = SecretStr("app-storage")
    storage_secret_access_key: SecretStr | None = SecretStr(LOCAL_STORAGE_SECRET)
    # How long presigned URLs work; R2 and S3 allow up to 7 days.
    storage_url_seconds: int = Field(default=900, gt=0, le=604_800)
    # Largest file a presigned upload accepts, in bytes; its exact size is signed into the URL.
    storage_max_upload_bytes: int = Field(default=26_214_400, gt=0)
    # Content types uploads may declare. Never text/html or image/svg+xml: served back from the
    # bucket's domain, they can run scripts. As JSON in the environment: ["image/png", ...].
    storage_upload_types: frozenset[str] = frozenset(
        {
            "image/png",
            "image/jpeg",
            "image/webp",
            "image/gif",
            "application/pdf",
            "text/plain",
            "text/csv",
            "text/markdown",
            "application/json",
            "application/zip",
        }
    )

    # The model the gateway calls, "provider:model id", with its prices in app/llm/prices.py. No
    # default: models and prices change, so you choose one. Unset, the gateway refuses to start with
    # a message saying so (a bare required field would only say "Field required").
    llm_model: str | None = None
    # The embedding model gateway.embeddings() calls, "openai:model id", priced in
    # app/llm/prices.py EMBEDDING_PRICES. OpenAI only (Anthropic has no embeddings), so it needs
    # OPENAI_API_KEY whichever provider LLM_MODEL names. 1,536 dimensions, as the models use.
    # No default, like llm_model.
    embedding_model: str | None = None
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
    # Each user's share of a day, for calls tagged with their id (Gateway.call_config). Unset
    # means no per-user cap; set it whenever users can trigger model calls, so one caller can't
    # spend the whole day's budget.
    llm_user_daily_budget_usd: Decimal | None = Field(default=None, gt=0)
    # Most model calls an agent may make in one run. The last one is made without tools, so the
    # model writes a final answer (app/llm/core/steps.py). The old AGENT_MAX_STEPS counted
    # LangGraph steps: its default of 25 allowed 13 calls.
    agent_max_model_calls: int = Field(default=13, gt=0)
    # Connections for agent checkpoints (LangGraph), opened only while agents run.
    agent_pool_size: int = Field(default=4, gt=0)

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

    @model_validator(mode="before")
    @classmethod
    def no_removed_agent_max_steps(cls, data: Any) -> Any:
        """Refuse to start while the old AGENT_MAX_STEPS is set (environment or backend/.env):
        keeping its number would silently change what it limits (steps became model calls)."""
        if "AGENT_MAX_STEPS" in os.environ or (
            isinstance(data, dict) and "agent_max_steps" in data
        ):
            raise ValueError(
                "AGENT_MAX_STEPS was replaced by AGENT_MAX_MODEL_CALLS. Delete AGENT_MAX_STEPS "
                "from backend/.env and from every environment the app runs in. The old value V "
                "allowed ceil(V / 2) model calls (25 allowed 13); the new setting counts model "
                "calls and defaults to 13, so set it only to change that."
            )
        return data

    @model_validator(mode="after")
    def no_local_defaults_in_production(self) -> Settings:
        """Refuse to start in production on the local development credentials: a missing secret
        would otherwise boot the app against localhost, or leave a known key live."""
        if self.app_env != "production":
            return self
        problems = []
        if self.database_url.get_secret_value() == LOCAL_DATABASE_URL:
            problems.append("DATABASE_URL is the local default")
        if self.storage_bucket and (
            (self.storage_secret_access_key or SecretStr("")).get_secret_value()
            == LOCAL_STORAGE_SECRET
        ):
            problems.append("STORAGE_SECRET_ACCESS_KEY is the local default")
        if problems:
            raise ValueError("production settings: " + "; ".join(problems))
        return self

    @property
    def docs_on(self) -> bool:
        return self.app_env != "production" if self.docs_enabled is None else self.docs_enabled


@lru_cache
def get_settings() -> Settings:
    return Settings()
