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
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.llm import prices
from app.llm.core.budget import LlmCall
from app.llm.core.prices import cost
from app.llm.core.usage import Usage
from app.llm.maf.gateway import Gateway
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
