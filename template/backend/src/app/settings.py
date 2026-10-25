"""Settings from the environment (and backend/.env locally), read once per process.

Names are the upper-case field names with no prefix (APP_ENV, LOG_LEVEL, LOG_JSON, DOCS_ENABLED).
Secrets are SecretStr so they never print in logs or tracebacks; .env.example lists every setting
with a safe value.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", frozen=True)

    app_env: Literal["local", "test", "production"] = "local"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    # JSON logs in production; readable console logs everywhere else.
    log_json: bool = False
    # Interactive API docs at /docs. Unset means on everywhere except production.
    docs_enabled: bool | None = None

    @property
    def docs_on(self) -> bool:
        return self.app_env != "production" if self.docs_enabled is None else self.docs_enabled


@lru_cache
def get_settings() -> Settings:
    return Settings()
