"""Evals call the real model: `make eval` runs them (never `make check` or CI) with a real key,
under the daily budget, against the throwaway test database. With LANGSMITH_API_KEY set, results are
uploaded as a LangSmith experiment (with LangChain, the model calls as its child runs; MAF's
calls are traced only outside tests); without it they only run locally.

They run through the same harness as the shared parity tests (tests/harness/), on the real
provider instead of a scripted one, so the same evals hold for whichever framework the project chose
and import none.

Write evals with LangSmith's pytest integration (https://docs.langchain.com/langsmith/pytest):
mark them `eval` and `langsmith`, log inputs, outputs and feedback with langsmith.testing, and
assert only what must hold for a release. For larger datasets run langsmith.evaluate() from a
script instead.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

import pytest
from pydantic import SecretStr
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.llm.core.budget import LlmCall
from app.llm.core.registry import key_setting
from app.settings import Settings
from tests.harness import Harness

NewCalls = Callable[[], Awaitable[list[LlmCall]]]

# With LangSmith tracing on, LangChain's budget refusal stops the call before it awaits the tracer's
# own chat-model-start callback, and Python warns that coroutine was never awaited. Harmless (the
# call is still refused and nothing is sent), so only those warnings are allowed, and only for that
# test under LangChain: under MAF they would hide a real warning (an unclosed client).
LANGCHAIN_REFUSAL_WARNINGS = (
    "ignore:coroutine 'ahandle_event' was never awaited:RuntimeWarning",
    "ignore::pytest.PytestUnraisableExceptionWarning",
)
REFUSAL_TEST = "test_the_budget_refuses_a_call_before_sending_it"


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        callspec = getattr(item, "callspec", None)
        under_langchain = callspec is not None and callspec.params.get("framework") == "langchain"
        if getattr(item, "originalname", None) == REFUSAL_TEST and under_langchain:
            for allowed in LANGCHAIN_REFUSAL_WARNINGS:
                item.add_marker(pytest.mark.filterwarnings(allowed))


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
    name = key_setting(os.environ["LLM_MODEL"]).upper()
    key = os.environ.get(name)
    if not key:
        if os.environ.get("LLM_EVAL") == "1":
            pytest.exit(f"evals call the real model: set {name}", returncode=2)
        pytest.skip(f"evals call the real model: set {name} and run make eval")
    return key


def eval_settings(api_key: str, database_url: str, **options: Any) -> Settings:
    settings = Settings(
        _env_file=None, app_env="test", database_url=SecretStr(database_url), **options
    )
    return settings.model_copy(update={key_setting(os.environ["LLM_MODEL"]): SecretStr(api_key)})


@pytest.fixture
async def make_gateway(
    framework: Harness, api_key: str, database_url: str
) -> AsyncIterator[Callable[..., Awaitable[Any]]]:
    """Gateways on the real provider, their budget and conversations kept in the test database,
    each opened as the app opens it and closed after the test. api_key is asked for before
    database_url, so a missing key stops before the database starts."""
    engine = create_async_engine(database_url)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    gateways: list[Any] = []
    closers: list[Callable[[], Awaitable[None]]] = []

    async def build(**options: Any) -> Any:
        settings = eval_settings(api_key, database_url, **options)
        gateway = framework.gateway(settings, lambda: sessions)
        gateways.append(gateway)  # closed even if opening its threads fails
        closers.append(await framework.open_threads(gateway, settings))
        return gateway

    try:
        yield build
    finally:
        await close_all([*closers, *(gateway.aclose for gateway in gateways)])
        await engine.dispose()


async def close_all(closers: list[Callable[[], Awaitable[None]]]) -> None:
    """Runs every closer, even after one fails, then raises the first failure."""
    first: BaseException | None = None
    for close in closers:
        try:
            await close()
        except Exception as error:  # the rest must still close
            first = first or error
    if first is not None:
        raise first


@pytest.fixture
async def gateway(make_gateway: Callable[..., Awaitable[Any]]) -> Any:
    """The gateway on the real provider, as the app opens it."""
    return await make_gateway()


@pytest.fixture
async def new_calls(framework: Harness, gateway: Any) -> NewCalls:
    """The llm_call rows written since the test started (the database is shared by the run)."""
    sessions = framework.meter(gateway).sessionmaker()
    async with sessions() as session:
        before = set((await session.scalars(select(LlmCall.id))).all())

    async def read() -> list[LlmCall]:
        async with sessions() as session:
            rows = (await session.scalars(select(LlmCall))).all()
        return [row for row in rows if row.id not in before]

    return read
