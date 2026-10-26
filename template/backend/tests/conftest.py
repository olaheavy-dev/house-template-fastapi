"""Test fixtures: an app per test, driven in-process through httpx with its lifespan running,
a real database for the tests that ask for one (db_session, db_client), and recorded model
responses for the tests marked vcr (cassettes in tests/cassettes/, replayed, never re-recorded:
`make record-llm` records them)."""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx
import pytest
from alembic import command
from asgi_lifespan import LifespanManager
from fastapi import FastAPI
from pydantic import SecretStr
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.db import get_session
from app.llm.gateway import key_setting
from app.main import create_app
from app.settings import Settings
from app.storage import create_bucket
from tests import db, storage

# Before langsmith first reads them (it caches them): tests never send traces or sync results to
# LangSmith, whatever the developer's shell exports. Evals (`make eval`, LLM_EVAL=1) upload an
# experiment and its traces when a LangSmith key is set, from the shell or backend/.env.
_upload = os.environ.get("LLM_EVAL") == "1" and bool(os.environ.get("LANGSMITH_API_KEY"))
os.environ["LANGSMITH_TRACING"] = os.environ["LANGSMITH_TEST_TRACKING"] = str(_upload).lower()


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def provider_key() -> tuple[str, str | None]:
    """The environment variable holding LLM_MODEL's provider key, and its value if set."""
    name = key_setting(Settings(_env_file=None).llm_model).upper()
    return name, os.environ.get(name)


@pytest.fixture(scope="session", autouse=True)
def recording_needs_a_real_key(request: pytest.FixtureRequest) -> None:
    # Replaying needs no key; recording with a placeholder would save the provider's 401.
    if request.config.getoption("--record-mode") in (None, "none"):
        return
    name, value = provider_key()
    if not value:
        pytest.exit(f"recording calls the real model: set {name}", returncode=2)


@pytest.fixture
def settings() -> Settings:
    # Placeholder keys for replay; the real ones, from the environment, only while recording.
    def key(name: str) -> SecretStr:
        return SecretStr(os.environ.get(name, "test-key"))

    return Settings(
        _env_file=None,
        app_env="test",
        openai_api_key=key("OPENAI_API_KEY"),
        anthropic_api_key=key("ANTHROPIC_API_KEY"),
    )


# Sent with every recorded request, so never written to a cassette.
SECRET_HEADERS = ["authorization", "x-api-key", "openai-organization", "openai-project"]


def _keep_response_essentials(response: dict[str, Any]) -> dict[str, Any]:
    # Response headers carry organisation ids and cookies; replay needs only the content type.
    response["headers"] = {
        name: value
        for name, value in response["headers"].items()
        if name.lower() in {"content-type", "content-encoding"}
    }
    return response


@pytest.fixture(scope="session")
def vcr_config() -> dict[str, Any]:
    return {
        "filter_headers": SECRET_HEADERS,
        "before_record_response": _keep_response_essentials,
        "decode_compressed_response": True,
        # The body too: a changed prompt or setting is a cassette miss, not a stale answer.
        "match_on": ["method", "scheme", "host", "port", "path", "query", "body"],
        # The app's own in-process test client goes through httpx too; never record it.
        "ignore_hosts": ["test"],
    }


@pytest.fixture
def allowed_hosts() -> list[str]:
    """Hosts a block_network test may still reach: this machine (the test database, Docker's
    socket). Model providers are blocked unless a test is recording."""
    return ["127.0.0.1", "localhost", "::1", "/.*"]


@pytest.fixture
def app(settings: Settings) -> FastAPI:
    return create_app(settings)


@pytest.fixture
async def client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    async with LifespanManager(app) as manager:
        transport = httpx.ASGITransport(app=manager.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield client


@pytest.fixture(scope="session")
def storage_endpoint() -> Iterator[str]:
    """A running S3-compatible storage for the whole test session (see tests/storage.py)."""
    yield from storage.storage_endpoint()


@pytest.fixture
def storage_settings(storage_endpoint: str) -> Settings:
    """Settings for a fresh bucket of its own on the test storage."""
    settings = Settings(
        _env_file=None,
        app_env="test",
        storage_bucket=f"test-{uuid.uuid4().hex[:12]}",
        storage_endpoint_url=storage_endpoint,
        storage_access_key_id=SecretStr(storage.ACCESS_KEY),
        storage_secret_access_key=SecretStr(storage.SECRET_KEY),
    )
    create_bucket(settings)
    return settings


@pytest.fixture(scope="session")
def database_url() -> Iterator[str]:
    """A migrated database for the whole test session (see tests/db.py)."""
    for url in db.database_url():
        command.upgrade(db.alembic_config(url), "heads")
        yield url


@pytest.fixture
async def db_session(database_url: str) -> AsyncIterator[AsyncSession]:
    """A session whose work, commits included, is rolled back when the test ends."""
    engine = create_async_engine(database_url, poolclass=NullPool)
    async with engine.connect() as connection:
        transaction = await connection.begin()
        session = AsyncSession(
            bind=connection, join_transaction_mode="create_savepoint", expire_on_commit=False
        )
        try:
            yield session
        finally:
            await session.close()
            await transaction.rollback()
    await engine.dispose()


@pytest.fixture
async def db_client(
    app: FastAPI, db_session: AsyncSession, database_url: str
) -> AsyncIterator[httpx.AsyncClient]:
    """Like client, with the whole app on the test database: requests use db_session, and code
    that opens sessions from app.state.sessionmaker joins the same rolled-back transaction."""

    async def session_override() -> AsyncIterator[AsyncSession]:
        yield db_session

    app.dependency_overrides[get_session] = session_override
    # The lifespan builds the engine from these settings: the test database, never a local one.
    app.state.settings = app.state.settings.model_copy(
        update={"database_url": SecretStr(database_url)}
    )
    async with LifespanManager(app) as manager:
        app.state.sessionmaker = app.state.budget_sessionmaker = async_sessionmaker(
            bind=db_session.bind, join_transaction_mode="create_savepoint", expire_on_commit=False
        )
        transport = httpx.ASGITransport(app=manager.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield client
