"""Evals call the real model: `make eval` runs them (never `make check` or CI) with a real key,
under the daily budget, against the throwaway test database. With LANGSMITH_API_KEY set, results are
uploaded as a LangSmith experiment; without it they only run locally.

Write evals with LangSmith's pytest integration (https://docs.langchain.com/langsmith/pytest):
mark them `eval` and `langsmith`, log inputs, outputs and feedback with langsmith.testing, and
assert only what must hold for a release. For larger datasets run langsmith.evaluate() from a
script instead.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator

import pytest
from pydantic import SecretStr
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.llm import checkpoints
from app.llm.gateway import Gateway, key_setting
from app.settings import Settings


@pytest.fixture(scope="session")
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(scope="session", autouse=True)
async def one_event_loop() -> None:
    """All evals share one event loop, as the app does. The provider integrations cache one HTTP
    client per process, and its connections belong to the loop that opened them; a fresh loop
    per test would find them closed ("Event loop is closed")."""


@pytest.fixture
def api_key() -> str:
    """The real key for LLM_MODEL's provider. Under `make eval` a missing key stops the run (an
    all-skipped eval run would look like a pass); elsewhere the evals skip."""
    name = key_setting(Settings(_env_file=None).llm_model).upper()
    key = os.environ.get(name)
    if not key:
        if os.environ.get("LLM_EVAL") == "1":
            pytest.exit(f"evals call the real model: set {name}", returncode=2)
        pytest.skip(f"evals call the real model: set {name} and run make eval")
    return key


@pytest.fixture
async def gateway(api_key: str, database_url: str) -> AsyncIterator[Gateway]:
    """The gateway on the real provider, its budget and agent threads kept in the test database.
    api_key comes first, so a missing key stops before the database starts; database_url is a
    plain dependency, so its migrations run before this fixture's event loop does."""
    settings = Settings(_env_file=None, app_env="test", database_url=SecretStr(database_url))
    settings = settings.model_copy(update={key_setting(settings.llm_model): SecretStr(api_key)})
    engine = create_async_engine(database_url)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    gateway = Gateway(settings, lambda: sessions)
    gateway.checkpointer, pool = await checkpoints.open_checkpointer(settings)
    yield gateway
    await pool.close()
    await engine.dispose()
