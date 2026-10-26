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

from app.llm.gateway import Gateway
from app.settings import Settings


@pytest.fixture
async def gateway(request: pytest.FixtureRequest) -> AsyncIterator[Gateway]:
    """The gateway on the real provider, its budget kept in the test database."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        pytest.skip("evals call the real model: set ANTHROPIC_API_KEY and run make eval")
    database_url: str = request.getfixturevalue("database_url")  # only once a key is set
    settings = Settings(_env_file=None, app_env="test", anthropic_api_key=SecretStr(key))
    engine = create_async_engine(database_url)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    yield Gateway(settings, lambda: sessions)
    await engine.dispose()
