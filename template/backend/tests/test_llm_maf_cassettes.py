"""MAF's real clients against recorded provider answers (#35): the exact request bytes (store=false,
the output limit, redacted text) and a call settled at the cost its recorded usage gives. Replay
matches on the request body with the network blocked, so a test passes only if the bytes MAF's
client sends are the bytes that were recorded.

The cassettes are recorded once, by hand, never by a normal test run, with a small spend cap and the
keys loaded from a file outside the repository (never echoed). From backend/ in a project rendered
with `llm: maf` (the template's own tests record in a scratch render and copy tests/cassettes/
back):

    (set -a; . <file holding OPENAI_API_KEY and ANTHROPIC_API_KEY>; set +a; \\
     env -u LLM_EVAL LLM_DAILY_BUDGET_USD=1 \\
     uv run --extra db --extra maf --extra storage \\
     pytest tests/test_llm_maf_cassettes.py -m vcr --record-mode=once -p no:randomly)

Until a cassette exists its replay test is skipped (a missing cassette would otherwise fail it).
"""

from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from agent_framework import Message
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.auth import User
from app.llm import prices
from app.llm.core.budget import LlmCall
from app.llm.core.prices import cost
from app.llm.core.usage import Usage
from app.llm.maf.gateway import Gateway
from app.llm.maf.steps import step_limit_reached
from app.settings import Settings
from tests.models import CHAT_MODEL

pytestmark = pytest.mark.anyio

CASSETTES = Path(__file__).parent / "cassettes" / "test_llm_maf_cassettes"
ANTHROPIC_MODEL = "anthropic:claude-haiku-5-5"
DAY = date(2026, 1, 8)
EMAIL = "priya.shah@example.com"
RECORD_COMMAND = "see the module notes of tests/test_llm_maf_cassettes.py"


@pytest.fixture
def cassette_or_skip(request: pytest.FixtureRequest) -> None:
    """Skip only while there is nothing to replay and nobody is recording. Recording an Anthropic
    call needs its key: a placeholder would record the provider's 401."""
    name = request.node.name
    recording = request.config.getoption("--record-mode") not in (None, "none")
    if recording:
        if not os.environ.get("ANTHROPIC_API_KEY", "").strip():
            pytest.exit("recording the MAF cassettes needs ANTHROPIC_API_KEY", returncode=2)
    elif not (CASSETTES / f"{name}.yaml").exists():
        pytest.skip(f"the cassette is not recorded yet: {RECORD_COMMAND}")


async def replayed(
    settings: Settings, db_session: AsyncSession, model: str, vcr: Any
) -> tuple[LlmCall, dict[str, Any]]:
    sessions = async_sessionmaker(
        bind=db_session.bind, join_transaction_mode="create_savepoint", expire_on_commit=False
    )
    gateway = Gateway(settings.model_copy(update={"llm_model": model}), lambda: sessions)
    gateway.meter.cap, gateway.meter.today = 1_000_000, lambda: DAY
    response = await gateway.client().get_response(
        [Message("user", [f"Reply with one word: hello. (Context: write to {EMAIL}.)"])]
    )
    assert response.text
    await gateway.aclose()
    [call] = (await db_session.scalars(select(LlmCall).where(LlmCall.day == DAY))).all()
    [request] = vcr.requests
    return call, json.loads(request.body)


def settled_at_its_price(call: LlmCall) -> None:
    assert call.status == "settled"
    assert call.input_tokens
    assert call.output_tokens
    usage = Usage(
        input_tokens=call.input_tokens,
        output_tokens=call.output_tokens,
        reasoning_tokens=call.reasoning_tokens or 0,
        cache_read_tokens=call.cache_read_tokens or 0,
        cache_write_undivided_tokens=call.cache_write_tokens or 0,  # MAF reports no lifetime
    )
    # Whatever the recording's usage, the row settles at exactly the price table's cost of it.
    assert call.cost_micro_usd == cost(prices.PRICES[call.model], usage) > 0
    assert call.reserved_micro_usd >= call.cost_micro_usd


@pytest.mark.vcr
@pytest.mark.block_network
@pytest.mark.usefixtures("cassette_or_skip")
async def test_an_openai_responses_call_is_stateless_redacted_and_settled(
    settings: Settings, db_session: AsyncSession, vcr: Any
) -> None:
    call, body = await replayed(settings, db_session, CHAT_MODEL, vcr)
    assert body["store"] is False  # the Responses API stores by default
    assert "max_output_tokens" in body
    assert EMAIL not in json.dumps(body)
    assert call.redacted_count == 1
    settled_at_its_price(call)


@pytest.mark.vcr
@pytest.mark.block_network
@pytest.mark.usefixtures("cassette_or_skip")
async def test_an_anthropic_call_is_redacted_sets_its_limit_and_settles(
    settings: Settings, db_session: AsyncSession, vcr: Any
) -> None:
    call, body = await replayed(settings, db_session, ANTHROPIC_MODEL, vcr)
    assert "store" not in body
    assert body["max_tokens"] == settings.llm_max_tokens
    assert EMAIL not in json.dumps(body)
    assert call.redacted_count == 1
    settled_at_its_price(call)


class Lead(BaseModel):
    """A lead to write to."""

    email: str


async def streamed(
    settings: Settings, db_session: AsyncSession, model: str, vcr: Any
) -> tuple[LlmCall, dict[str, Any], str, Any]:
    """One streamed call that asks for a schema (#35 PR 3): the text as the consumer read it, the
    final response's value, the ledger row and the request body."""
    sessions = async_sessionmaker(
        bind=db_session.bind, join_transaction_mode="create_savepoint", expire_on_commit=False
    )
    gateway = Gateway(settings.model_copy(update={"llm_model": model}), lambda: sessions)
    gateway.meter.cap, gateway.meter.today = 1_000_000, lambda: DAY
    pieces: list[str] = []
    stream = gateway.client().get_response(
        [Message("user", [f"Give the lead's email as JSON. The lead is {EMAIL}."])],
        stream=True,
        options={"response_format": Lead},
    )
    async with stream:
        async for update in stream:
            pieces += [
                c.text
                for c in update.contents
                if c.type == "text" and c.text and c.merge_mode != "replace"
            ]
        final = await stream.get_final_response()
    await gateway.aclose()
    [call] = (await db_session.scalars(select(LlmCall).where(LlmCall.day == DAY))).all()
    [request] = vcr.requests
    return call, json.loads(request.body), "".join(pieces), final.value


@pytest.mark.vcr
@pytest.mark.block_network
@pytest.mark.usefixtures("cassette_or_skip")
@pytest.mark.parametrize("model", [CHAT_MODEL, ANTHROPIC_MODEL], ids=["openai", "anthropic"])
async def test_a_streamed_structured_call_is_redacted_restored_and_settled(
    settings: Settings, db_session: AsyncSession, vcr: Any, model: str
) -> None:
    # The real APIs' streams: the request holds the placeholder, the text read and the parsed
    # value hold the original, and the row settles from the usage the stream reported at its end
    # (Anthropic's cumulative snapshots counted once).
    call, body, text, value = await streamed(settings, db_session, model, vcr)
    assert body["stream"] is True
    assert EMAIL not in json.dumps(body)
    assert type(value) is Lead
    assert value.email == EMAIL
    assert EMAIL in text
    assert call.redacted_count == 1
    settled_at_its_price(call)


def add(a: int, b: int) -> int:
    """Add two whole numbers."""
    return a + b


@pytest.mark.vcr
@pytest.mark.block_network
@pytest.mark.usefixtures("cassette_or_skip")
@pytest.mark.parametrize("model", [CHAT_MODEL, ANTHROPIC_MODEL], ids=["openai", "anthropic"])
async def test_an_agent_run_stops_at_its_limit_with_a_real_answer(
    settings: Settings, db_session: AsyncSession, vcr: Any, model: str
) -> None:
    # The real APIs at the step limit (#35 PR 4): with a limit of 2 the model asks for the tool
    # on call 1, call 2 is sent with tool_choice "none" and is answered in text, and the run is
    # flagged. Two requests, two settled rows.
    sessions = async_sessionmaker(
        bind=db_session.bind, join_transaction_mode="create_savepoint", expire_on_commit=False
    )
    gateway = Gateway(
        settings.model_copy(update={"llm_model": model, "agent_max_model_calls": 2}),
        lambda: sessions,
    )
    gateway.meter.cap, gateway.meter.today = 1_000_000, lambda: DAY
    agent = gateway.agent([add], instructions="Always use the add tool for arithmetic.")
    response = await agent.run("Use the add tool to add 2 and 3, then add 4 to that result.")
    await gateway.aclose()
    assert response.text
    assert step_limit_reached(response)
    first, last = [json.loads(request.body) for request in vcr.requests]
    assert first.get("tools")
    choice = last.get("tool_choice")
    assert choice == "none" or (isinstance(choice, dict) and choice.get("type") == "none")
    calls = (await db_session.scalars(select(LlmCall).where(LlmCall.day == DAY))).all()
    assert len(calls) == 2
    for call in calls:
        settled_at_its_price(call)


@pytest.mark.vcr
@pytest.mark.block_network
@pytest.mark.usefixtures("cassette_or_skip")
@pytest.mark.parametrize("model", [CHAT_MODEL, ANTHROPIC_MODEL], ids=["openai", "anthropic"])
async def test_a_stored_conversation_is_resent_after_a_restart(
    settings: Settings, db_session: AsyncSession, vcr: Any, model: str
) -> None:
    # The real APIs with a stored session (#35 PR 5): turn 1 states a code word; a new gateway
    # (a restart) loads the session from Postgres and turn 2 asks for it. The second request
    # carries the first turn, and the model answers from it.
    sessions = async_sessionmaker(
        bind=db_session.bind, join_transaction_mode="create_savepoint", expire_on_commit=False
    )
    who = User(id="u_cassette")

    async def turn(question: str) -> str:
        gateway = Gateway(settings.model_copy(update={"llm_model": model}), lambda: sessions)
        gateway.meter.cap, gateway.meter.today = 1_000_000, lambda: DAY
        agent = gateway.agent([], name="memory", instructions="Answer in one short sentence.")
        session = await gateway.sessions.load(who, "c_cassette", agent)
        response = await agent.run(question, session=session)
        await gateway.sessions.save(who, "c_cassette", agent, session)
        await gateway.aclose()
        return response.text

    await turn("Remember this code word for later: ORCHID. Reply with OK.")
    answer = await turn("What was the code word I gave you?")
    assert "ORCHID" in answer.upper()
    first, second = [json.dumps(json.loads(request.body)) for request in vcr.requests]
    assert "ORCHID" in first
    assert "ORCHID" in second  # the stored history was sent again


def test_the_maf_cassettes_hold_no_credentials_or_personal_data() -> None:
    for cassette in CASSETTES.glob("*.yaml"):
        text = cassette.read_text().lower()
        for marker in (
            "x-api-key",
            "authorization",
            "sk-ant-",
            "sk-proj-",
            "anthropic-organization-id",
            "openai-organization",
            EMAIL,
        ):
            assert marker not in text, (cassette.name, marker)
