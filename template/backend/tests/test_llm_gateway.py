"""The gateway end to end. The replayed call goes through the real provider SDK and LangChain
integration, answered from tests/cassettes/ with the network blocked."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI
from langchain_core.messages import HumanMessage
from langchain_core.messages.ai import UsageMetadata
from langchain_core.tracers.langchain import LangChainTracer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.llm import prices
from app.llm.budget import LlmCall
from app.llm.gateway import Gateway, GatewayDep
from app.llm.prompts import load_prompt
from app.settings import Settings

pytestmark = pytest.mark.anyio

CASSETTES = Path(__file__).parent / "cassettes"


def add_ask_route(app: FastAPI) -> None:
    @app.post("/ask")
    async def ask(gateway: GatewayDep) -> dict[str, str]:
        answer = await gateway.model().ainvoke([HumanMessage("Reply with one word: hello.")])
        return {"text": answer.text}


@pytest.mark.vcr
@pytest.mark.block_network
async def test_a_replayed_call_is_answered_and_its_cost_settled(
    app: FastAPI, db_client: httpx.AsyncClient, db_session: AsyncSession
) -> None:
    add_ask_route(app)
    response = await db_client.post("/ask")
    assert response.status_code == 200
    assert response.json()["text"]

    [call] = (await db_session.scalars(select(LlmCall))).all()
    assert call.model == "openai:gpt-6-luna"
    assert call.status == "settled"
    # Whatever the recording's usage, the row settles at exactly its price-table cost (re-recording
    # changes the counts; gpt-6 models count reasoning tokens as output).
    assert call.input_tokens
    assert call.output_tokens
    usage: UsageMetadata = {
        "input_tokens": call.input_tokens,
        "output_tokens": call.output_tokens,
        "total_tokens": call.input_tokens + call.output_tokens,
    }
    assert call.cost_micro_usd == prices.cost(prices.PRICES[call.model], usage) > 0
    assert call.reserved_micro_usd >= call.cost_micro_usd


@pytest.mark.block_network
async def test_over_budget_calls_get_a_503_and_never_reach_the_provider(
    app: FastAPI, db_client: httpx.AsyncClient
) -> None:
    app.state.gateway.guard.cap = 1  # micro-USD: no call fits
    add_ask_route(app)
    response = await db_client.post("/ask")
    assert response.status_code == 503
    assert response.headers["content-type"] == "application/problem+json"
    assert "budget" in response.json()["detail"]


async def test_without_a_key_calls_get_a_503(client: httpx.AsyncClient, app: FastAPI) -> None:
    app.state.gateway = Gateway(Settings(_env_file=None, app_env="test"), lambda: None)  # type: ignore[arg-type,return-value]
    add_ask_route(app)
    response = await client.post("/ask")
    assert response.status_code == 503
    assert response.json()["detail"] == "The language model isn't configured."


def test_a_model_without_a_price_is_refused_at_startup() -> None:
    settings = Settings(_env_file=None, llm_model="anthropic:claude-unpriced")
    with pytest.raises(ValueError, match="no price"):
        Gateway(settings, lambda: None)  # type: ignore[arg-type,return-value]


def test_tracing_follows_the_settings_and_is_never_on_in_tests() -> None:
    on = {
        "langsmith_tracing": True,
        "langsmith_project": "demo",
        "langsmith_api_key": "lsv2-x",
        "database_url": "postgresql+psycopg://app:secret@db.internal:5432/app",
    }
    production = Gateway(Settings(_env_file=None, app_env="production", **on), lambda: None)  # type: ignore[arg-type,return-value]
    assert isinstance(production.tracer, LangChainTracer)
    assert production.tracer.project_name == "demo"
    assert Gateway(Settings(_env_file=None, app_env="test", **on), lambda: None).tracer is None  # type: ignore[arg-type,return-value]


def test_prompts_are_versioned_by_their_content(tmp_path: Path) -> None:
    (tmp_path / "greet.md").write_text("Say hello.")
    prompt = load_prompt("greet", tmp_path)
    assert prompt.text == "Say hello."
    assert len(prompt.sha) == 12
    config = prompt.config()
    assert config["metadata"] == {"prompt": "greet", "prompt_sha": prompt.sha}
    (tmp_path / "greet2.md").write_text("Say hello!")
    assert load_prompt("greet2", tmp_path).sha != prompt.sha


def test_prompt_names_cannot_leave_the_prompts_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not a prompt name"):
        load_prompt("../secrets", tmp_path / "prompts")


def test_cassettes_hold_no_credentials() -> None:
    for cassette in CASSETTES.rglob("*.yaml"):
        text = cassette.read_text().lower()
        for marker in (
            "x-api-key",
            "authorization",
            "sk-ant-",
            "sk-proj-",
            "anthropic-organization-id",
            "openai-organization",
        ):
            assert marker not in text, (cassette.name, marker)
