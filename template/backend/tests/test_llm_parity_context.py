"""The context report of a stored conversation, the same under every framework (#46 PR 1): after a
turn, the gateway reports about how many tokens the next call would send, from the last call's
reported usage, against the model's window and limit, with no model call. The database is real;
the model is scripted by the harness, so each test runs once per framework the project chose and
imports none.

The figures are worked out by hand. The model is Claude Sonnet 5.5: a window of 1,000,000 tokens
(app/llm/prices.py), so the limit is 1,000,000 minus LLM_MAX_TOKENS. A scripted answer reports the
usage it is given; after a turn nothing is stored after the answer, so the count is that call's
input plus its output.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from datetime import date
from typing import Any

import pytest
from pydantic import SecretStr
from sqlalchemy import select

from app.db import create_engine, create_sessionmaker
from app.llm.core.bounds import INPUT_MARGIN_TOKENS
from app.llm.core.budget import LlmCall
from app.settings import Settings
from tests.harness import AsksTool, Harness, Reply, user

pytestmark = pytest.mark.anyio

DAY = date(2026, 1, 13)
MODEL = "anthropic:claude-sonnet-5-5"
WINDOW = 1_000_000
USER, CONVERSATION = "u_context", "c_context"


@pytest.fixture
async def make(
    framework: Harness, settings: Settings, database_url: str
) -> AsyncIterator[Callable[..., Any]]:
    """A gateway on the real (committed) test database, as the app runs."""
    with_db = settings.model_copy(
        update={"llm_model": MODEL, "database_url": SecretStr(database_url)}
    )
    engine = create_engine(with_db, pool_size=2, max_overflow=0)
    sessions = create_sessionmaker(engine)
    closers: list[Any] = []
    built: list[Any] = []

    async def build(*script: Reply | AsksTool, limit: int | None = None) -> Any:
        tuned = with_db
        if limit is not None:
            tuned = with_db.model_copy(update={"agent_max_model_calls": limit})
        gateway = framework.gateway(tuned, lambda: sessions)
        meter = framework.meter(gateway)
        meter.cap, meter.today = 10_000_000, lambda: DAY
        framework.install_scripted(gateway, list(script))
        closers.append(await framework.open_threads(gateway, tuned))
        built.append(gateway)
        return gateway

    yield build
    if built:
        for user_ref in (USER, "u_context_other"):
            await framework.forget_threads(
                built[-1], user_ref=user_ref, conversation_id=CONVERSATION
            )
    for close in closers:
        await close()
    async with sessions() as db, db.begin():
        for row in await db.scalars(select(LlmCall).where(LlmCall.day == DAY)):
            await db.delete(row)
    await engine.dispose()


async def turn(
    framework: Harness,
    gateway: Any,
    text: str,
    *,
    stream: bool = False,
    tools: tuple[Callable[..., Any], ...] = (),
) -> str:
    return await framework.thread_turn(
        gateway,
        user_ref=USER,
        conversation_id=CONVERSATION,
        messages=[user(text)],
        stream=stream,
        tools=tools,
    )


def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@pytest.mark.parametrize("stream", [False, True], ids=["run", "streamed"])
async def test_a_stored_turn_is_reported_from_its_calls_usage(
    framework: Harness, make: Callable[..., Any], settings: Settings, stream: bool
) -> None:
    gateway = await make(Reply("A kitchen.", usage=(1_000, 500)))
    await turn(framework, gateway, "Which room is this?", stream=stream)

    got = await framework.context_report(gateway, user_ref=USER, conversation_id=CONVERSATION)
    limit = WINDOW - settings.llm_max_tokens
    # 1,000 read + 500 written = 1,500. Sonnet's price has no prompt cap, so the share is of the
    # window's limit. (The bound is each framework's own byte count of what it stored.)
    shown = got.as_dict()
    del shown["bound"]
    assert shown == {
        "tokens": 1_500,
        "window": WINDOW,
        "limit": limit,
        "prompt_limit": None,
        "percent": round(1_500 * 100 / limit, 1),
        "trimmed": 0,
        "measured": True,
    }
    assert got.bound >= INPUT_MARGIN_TOKENS + len("Which room is this?") + len("A kitchen.")


async def test_the_latest_call_is_the_one_counted(
    framework: Harness, make: Callable[..., Any]
) -> None:
    gateway = await make(Reply("Noted.", usage=(1_000, 500)), Reply("Yes.", usage=(3_000, 200)))
    await turn(framework, gateway, "My room is the kitchen.")
    await turn(framework, gateway, "Is it large?")
    got = await framework.context_report(gateway, user_ref=USER, conversation_id=CONVERSATION)
    # The second call read the whole conversation (3,000) and wrote 200.
    assert (got.tokens, got.measured) == (3_200, True)


@pytest.mark.parametrize("stream", [False, True], ids=["run", "streamed"])
async def test_a_tool_loop_is_counted_from_its_last_call_not_the_runs_total(
    framework: Harness, make: Callable[..., Any], stream: bool
) -> None:
    # Three calls (the step limit): two ask for `add`, the last answers. They report 100/10,
    # 200/20 and 300/30. The last call read everything before its answer, so the count is
    # 300 + 30 = 330; an earlier call's figures (110, 220) or the run's total (660) are wrong.
    script = [AsksTool("add", {"a": 1, "b": 2}, usage=(n * 100, n * 10)) for n in (1, 2, 3)]
    gateway = await make(*script, limit=3)
    await turn(framework, gateway, "What is 1 + 2?", stream=stream, tools=(add,))
    got = await framework.context_report(gateway, user_ref=USER, conversation_id=CONVERSATION)
    assert (got.tokens, got.measured) == (330, True)


@pytest.mark.parametrize("stream", [False, True], ids=["run", "streamed"])
async def test_a_last_call_without_usage_is_never_counted_from_an_earlier_one(
    framework: Harness, make: Callable[..., Any], stream: bool
) -> None:
    # The first call asks for `add` and reports 100/10; the last answers with 4,000 characters and
    # reports nothing. The next call sends that answer, so the count must be at least 4,000 (an
    # earlier call's 110 would under-count it).
    gateway = await make(AsksTool("add", {"a": 1, "b": 2}), Reply("x" * 4_000, usage=None), limit=2)
    await turn(framework, gateway, "What is 1 + 2?", stream=stream, tools=(add,))
    got = await framework.context_report(gateway, user_ref=USER, conversation_id=CONVERSATION)
    assert got.tokens >= 4_000


def describe(topic: str) -> str:
    """Describe a topic."""
    return topic


# A tool whose definition (3,000 characters) is larger than any answer here, so a report that
# left it out would fall below the budget's bound by more than the answer it adds.
describe.__doc__ = "Describe a topic. " + "x" * 3_000


@pytest.mark.parametrize("stream", [False, True], ids=["run", "streamed"])
@pytest.mark.parametrize("told", [True, False], ids=["told-tools", "not-told"])
async def test_the_reports_bound_is_never_below_the_one_the_budget_checks(
    framework: Harness, make: Callable[..., Any], stream: bool, told: bool
) -> None:
    # The budget refuses a prompt whose byte bound passes the price's max_prompt_tokens (OpenAI's
    # entries), so the report's 100% is only right if its `bound` is at least the budget's. Record
    # the bound the budget reserved for the second call, which sent the first turn, the second
    # question and the tools; the report after that turn also holds the second answer. The budget
    # records that bound on the answer, so the report holds it even when it isn't told what the
    # agent sends (its tools here; a response_format schema likewise).
    gateway = await make(Reply("Noted."), Reply("Yes."))
    meter = framework.meter(gateway)
    checked: list[int] = []
    reserve = meter.reserve

    async def recording(call_id: Any, *, input_tokens: int, **kwargs: Any) -> int:
        checked.append(input_tokens)
        return await reserve(call_id, input_tokens=input_tokens, **kwargs)

    meter.reserve = recording  # type: ignore[method-assign]
    await turn(framework, gateway, "My room is the kitchen.", tools=(describe,), stream=stream)
    await turn(framework, gateway, "Is it large?", tools=(describe,), stream=stream)
    got = await framework.context_report(
        gateway, user_ref=USER, conversation_id=CONVERSATION, tools=(describe,) if told else ()
    )
    assert len(checked) == 2
    # At least the budget's figure, and measured the same way: the only extra is the answer.
    assert checked[-1] <= got.bound < checked[-1] + 1_000


async def test_without_usage_the_report_estimates_and_says_so(
    framework: Harness, make: Callable[..., Any]
) -> None:
    gateway = await make(Reply("A kitchen.", usage=None))
    await turn(framework, gateway, "Which room is this?")
    got = await framework.context_report(gateway, user_ref=USER, conversation_id=CONVERSATION)
    assert not got.measured
    # The two messages' bytes, at least their text, plus the margin. Each framework stores the
    # messages in its own shape, so only the lower bound is the same for both.
    assert got.tokens >= INPUT_MARGIN_TOKENS + len("Which room is this?") + len("A kitchen.")


async def test_a_conversation_never_stored_and_another_users_are_empty(
    framework: Harness, make: Callable[..., Any]
) -> None:
    gateway = await make(Reply("A kitchen.", usage=(1_000, 500)))
    empty = await framework.context_report(gateway, user_ref=USER, conversation_id=CONVERSATION)
    # Nothing stored: only the agent's own part and the margin, estimated.
    assert not empty.measured
    assert empty.tokens == empty.bound >= INPUT_MARGIN_TOKENS

    await turn(framework, gateway, "Which room is this?")
    other = await framework.context_report(
        gateway, user_ref="u_context_other", conversation_id=CONVERSATION
    )
    assert other == empty


async def test_the_report_makes_no_model_call(framework: Harness, make: Callable[..., Any]) -> None:
    gateway = await make(Reply("A kitchen."), Reply("never sent"))
    await turn(framework, gateway, "Which room is this?")
    assert framework.scripted_left(gateway) == 1
    await framework.context_report(gateway, user_ref=USER, conversation_id=CONVERSATION)
    assert framework.scripted_left(gateway) == 1
