"""What the MAF gateway refuses, sizes and caps (#35 review): per-call and per-agent chat
middleware, options it does not allow, the full size of tools and schemas, the tool loop's limit,
and a cancelled call."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable
from datetime import date
from typing import Any

import pytest
from agent_framework import (
    Agent,
    ChatContext,
    ChatResponse,
    Content,
    FunctionTool,
    Message,
    chat_middleware,
)
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.llm.core.budget import LlmCall
from app.llm.core.errors import SchemaMismatch
from app.llm.maf.gateway import Gateway
from app.llm.maf.middleware import ALLOWED_OPTIONS, OptionNotAllowed
from app.llm.maf.redaction import RedactionUnavailable
from app.settings import Settings
from tests.harness import Reply
from tests.harness.maf import HARNESS, ScriptedClient
from tests.maf_wire import FakeProviderHttp

pytestmark = pytest.mark.anyio

DAY = date(2026, 1, 9)
EMAIL = "priya.shah@example.com"


@pytest.fixture
def make(settings: Settings, db_session: AsyncSession) -> Any:
    sessions = async_sessionmaker(
        bind=db_session.bind, join_transaction_mode="create_savepoint", expire_on_commit=False
    )

    def build(
        model: str = "anthropic:claude-sonnet-5-5", http: Any = None, **options: Any
    ) -> Gateway:
        gateway = Gateway(
            settings.model_copy(update={"llm_model": model, **options}),
            lambda: sessions,
            http_client=http,
        )
        gateway.meter.cap, gateway.meter.today = 1_000_000, lambda: DAY
        if http is None:
            HARNESS.install_scripted(gateway, [Reply("ok")] * 12)
        return gateway

    return build


async def rows(session: AsyncSession) -> list[LlmCall]:
    return list(
        await session.scalars(
            select(LlmCall).where(LlmCall.day == DAY).order_by(LlmCall.created_at)
        )
    )


@chat_middleware
async def adds_memory(context: ChatContext, call_next: Callable[[], Awaitable[None]]) -> None:
    context.messages = [*context.messages, Message("user", [f"memory: {EMAIL}"])]
    await call_next()


# --- extra chat middleware ------------------------------------------------------------------


async def test_chat_middleware_added_per_call_is_refused_and_nothing_is_sent(make: Any) -> None:
    fake = FakeProviderHttp("openai")
    gateway = make("openai:gpt-6-luna", fake.client)
    with pytest.raises(RedactionUnavailable):
        await gateway.client().get_response(
            [Message("user", ["hi"])], client_kwargs={"middleware": [adds_memory]}
        )
    assert fake.attempts == []


async def test_chat_middleware_added_to_an_agent_is_refused_and_nothing_is_sent(
    make: Any,
) -> None:
    fake = FakeProviderHttp("openai")
    gateway = make("openai:gpt-6-luna", fake.client)
    agent = Agent(client=gateway.client(), middleware=[adds_memory])
    with pytest.raises(Exception) as raised:  # noqa: PT011 - MAF may wrap it; found in the chain
        await agent.run("hi")
    from app.llm.maf.errors import chain

    assert any(isinstance(error, RedactionUnavailable) for error in chain(raised.value))
    assert fake.attempts == []


async def test_function_middleware_on_a_call_is_not_refused(make: Any) -> None:
    from agent_framework import FunctionInvocationContext, function_middleware

    @function_middleware
    async def noop(
        context: FunctionInvocationContext, call_next: Callable[[], Awaitable[None]]
    ) -> None:
        await call_next()

    gateway = make()
    await gateway.client().get_response(
        [Message("user", ["hi"])], client_kwargs={"middleware": [noop]}
    )


# --- options ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "option",
    [
        {"extra_body": {"store": True, "input": "x"}},
        {"conversation_id": "conv_1"},
        {"previous_response_id": "resp_1"},
        {"conversation": "conv_1"},
        {"background": True},
        {"continuation_token": {"x": 1}},
        {"user": EMAIL},
        {"metadata": {"who": EMAIL}},
        {"something_new": 1},
    ],
)
async def test_an_option_the_gateway_does_not_allow_is_refused_before_anything_is_sent(
    make: Any, db_session: AsyncSession, option: dict[str, Any]
) -> None:
    fake = FakeProviderHttp("openai")
    gateway = make("openai:gpt-6-luna", fake.client)
    with pytest.raises(OptionNotAllowed) as raised:
        await gateway.client().get_response([Message("user", ["hi"])], options=option)
    assert fake.attempts == []
    assert await rows(db_session) == []
    assert EMAIL not in str(raised.value)  # the name of the option, never its value


async def test_the_options_the_gateway_allows_pass_and_another_model_is_refused(
    make: Any,
) -> None:
    gateway = make("openai:gpt-6-luna")
    for name, value in (("temperature", 0.2), ("top_p", 0.9), ("seed", 3), ("model", "gpt-6-luna")):
        await gateway.client().get_response([Message("user", ["hi"])], options={name: value})
    with pytest.raises(OptionNotAllowed):
        await gateway.client().get_response(
            [Message("user", ["hi"])], options={"model": "gpt-6.1-sol"}
        )
    assert "extra_body" not in ALLOWED_OPTIONS


@pytest.mark.parametrize("given", [True, 0, -5])
async def test_a_bool_or_non_positive_output_limit_is_treated_as_unset(
    make: Any, given: Any
) -> None:
    gateway = make(llm_max_tokens=500)
    await gateway.client().get_response([Message("user", ["hi"])], options={"max_tokens": given})
    client = gateway._client
    assert isinstance(client, ScriptedClient)
    assert client.seen_options[0]["max_tokens"] == 500


# --- sizing ----------------------------------------------------------------------------------


class Contact(BaseModel):
    email: str = Field(description="d" * 40_000)


def long_tool() -> FunctionTool:
    def lookup(name: str) -> str:
        return name

    return FunctionTool(name="lookup", description="t" * 50_000, func=lookup)


async def reserved(gateway: Gateway, db_session: AsyncSession, **options: Any) -> int:
    # The new row, found by id: the rows of one test share created_at (now() is the transaction's
    # start), so "the last one" is not defined.
    before = {row.id for row in await rows(db_session)}
    # The scripted answer is plain text, so a call that asks for a schema ends in the 502 after it
    # was reserved; only the reservation is measured here.
    with contextlib.suppress(SchemaMismatch):
        await gateway.client().get_response([Message("user", ["hi"])], options=options or None)
    [new] = [row for row in await rows(db_session) if row.id not in before]
    return new.reserved_micro_usd


async def test_a_long_tool_description_and_a_long_schema_are_reserved_for_in_full(
    make: Any, db_session: AsyncSession
) -> None:
    gateway = make()
    plain = await reserved(gateway, db_session)
    with_tool = await reserved(gateway, db_session, tools=[long_tool()])
    with_schema = await reserved(gateway, db_session, response_format=Contact)
    # The reservation prices every input token at max(input, longer cache write) = $4 per million
    # for this model: 4 micro-USD per byte-token, so 50,000 bytes reserve at least 200,000.
    assert with_tool - plain >= 50_000 * 4
    assert with_schema - plain >= 40_000 * 4
    # A schema given as a dict is sized as given.
    as_dict = await reserved(
        gateway, db_session, response_format={"type": "object", "description": "s" * 30_000}
    )
    assert as_dict - plain >= 30_000 * 4


async def test_a_prompt_that_only_its_tools_make_too_long_is_refused(
    make: Any, db_session: AsyncSession
) -> None:
    from app.llm.core import bounds
    from app.llm.core.prices import Price

    gateway = make("openai:gpt-6-luna")
    price = gateway.meter.price
    gateway.meter.price = Price(
        price.rates, checked=price.checked, source=price.source, max_prompt_tokens=10_000
    )
    with pytest.raises(bounds.PromptTooLong):
        await gateway.client().get_response(
            [Message("user", ["hi"])], options={"tools": [long_tool()]}
        )
    assert await rows(db_session) == []


# --- the tool loop's limit ------------------------------------------------------------------


@pytest.mark.parametrize(("limit", "calls"), [(1, 1), (2, 2), (3, 3), (5, 5)])
async def test_agent_max_model_calls_caps_the_tool_loop_of_the_client(
    make: Any, db_session: AsyncSession, limit: int, calls: int
) -> None:
    gateway = make(agent_max_model_calls=limit)
    ran: list[str] = []

    def lookup(name: str) -> str:
        ran.append(name)
        return "found"

    tool = FunctionTool(name="lookup", description="d", func=lookup)

    def asks_for_a_tool(number: int) -> ChatResponse[Any]:
        call = Content.from_function_call(f"c{number}", "lookup", arguments={"name": "x"})
        return ChatResponse(messages=[Message("assistant", [call])])

    HARNESS.install_scripted(gateway, [asks_for_a_tool(n) for n in range(30)] + [Reply("done")])
    await gateway.client().get_response([Message("user", ["go"])], options={"tools": [tool]})
    assert len(await rows(db_session)) == calls  # exactly N model calls (decision 0003)
    client = gateway._client
    assert isinstance(client, ScriptedClient)
    choices = [options.get("tool_choice") for options in client.seen_options]
    # The last call is made with tool_choice "none" (MAF's own final call); at a limit of 1 the
    # only call is that one, so no tools are offered on any call.
    assert choices[-1] == "none"
    assert choices[:-1] == [None] * (calls - 1)
    assert len(ran) == calls - 1


# --- a cancelled call ------------------------------------------------------------------------


async def test_a_cancelled_call_stays_reserved_and_the_meter_forgets_it(
    make: Any, db_session: AsyncSession
) -> None:
    gateway = make()
    started = asyncio.Event()

    class Hangs(ScriptedClient):
        def _inner_get_response(self, **kwargs: Any) -> Any:
            async def wait() -> Any:
                started.set()
                await asyncio.sleep(60)

            return wait()

    client = Hangs(script=[], middleware=gateway.middleware)
    gateway._client = client
    task = asyncio.ensure_future(gateway.client().get_response([Message("user", ["hi"])]))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    [row] = await rows(db_session)
    # Whether it was billed is unknown: the row stays reserved (counted at its reservation) until
    # expire_stale closes it, and the meter no longer tracks it.
    assert row.status == "reserved"
    assert gateway.meter._started == {}


# --- the tool loop's limit on the gateway's real clients -------------------------------------


@pytest.mark.parametrize(
    ("model", "provider"),
    [("openai:gpt-6-luna", "openai"), ("anthropic:claude-haiku-5-5", "anthropic")],
)
@pytest.mark.parametrize(("limit", "iterations"), [(1, 1), (2, 1), (3, 2), (13, 12)])
async def test_the_real_clients_carry_the_tool_loop_limit(
    make: Any, model: str, provider: str, limit: int, iterations: int
) -> None:
    gateway = make(model, FakeProviderHttp(provider).client, agent_max_model_calls=limit)
    client = gateway._client
    assert client is not None
    assert client.function_invocation_configuration["max_iterations"] == iterations
    assert gateway.budget.step_limit.max_model_calls == limit


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
async def test_with_a_limit_of_one_the_real_client_sends_tool_choice_none(
    make: Any, provider: str
) -> None:
    fake = FakeProviderHttp(provider)
    model = "openai:gpt-6-luna" if provider == "openai" else "anthropic:claude-haiku-5-5"
    gateway = make(model, fake.client, agent_max_model_calls=1)
    tool = FunctionTool(name="lookup", description="d", func=lambda name: name)
    await gateway.client().get_response([Message("user", ["go"])], options={"tools": [tool]})
    [body] = fake.bodies
    choice = body["tool_choice"]
    assert choice == "none" or choice == {"type": "none"}, choice


# --- reasoning options -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("provider", "option"),
    [
        ("openai", {"reasoning": {"effort": "low"}}),
        ("openai", {"verbosity": "low"}),
        ("anthropic", {"thinking": {"type": "enabled", "budget_tokens": 1_024}}),
    ],
)
async def test_reasoning_options_pass_for_the_provider_that_has_them(
    make: Any, provider: str, option: dict[str, Any]
) -> None:
    fake = FakeProviderHttp(provider)
    model = "openai:gpt-6-luna" if provider == "openai" else "anthropic:claude-haiku-5-5"
    gateway = make(model, fake.client, llm_max_tokens=2_000)
    await gateway.client().get_response([Message("user", ["hi"])], options=option)
    assert len(fake.attempts) == 1
    [value] = option.values()
    assert str(value) in str(fake.bodies[0]).replace('"', "'")


async def test_a_thinking_budget_the_provider_rejects_is_refused_and_thinking_stays_in_bound(
    make: Any, db_session: AsyncSession
) -> None:
    fake = FakeProviderHttp("anthropic")
    gateway = make("anthropic:claude-haiku-5-5", fake.client, llm_max_tokens=2_000)
    for budget in (2_000, 5_000):
        with pytest.raises(OptionNotAllowed, match="budget_tokens"):
            await gateway.client().get_response(
                [Message("user", ["hi"])],
                options={"thinking": {"type": "enabled", "budget_tokens": budget}},
            )
    assert fake.attempts == []
    # A budget under max_tokens passes, and the reservation covers max_tokens (thinking counts
    # toward it), not max_tokens plus the budget.
    await gateway.client().get_response(
        [Message("user", ["hi"])],
        options={"thinking": {"type": "enabled", "budget_tokens": 1_500}},
    )
    await gateway.client().get_response([Message("user", ["hi"])])
    with_thinking, without = [row.reserved_micro_usd for row in await rows(db_session)]
    assert abs(with_thinking - without) < 200  # the option's few bytes, not 1,500 tokens
