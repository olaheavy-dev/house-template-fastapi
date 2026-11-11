"""The real provider SDKs under MAF's real clients, over a fake HTTP API (#35): what goes out on
the wire (store, the output limit, redacted text), what a call costs from the usage the provider
reports, and who retries (only the SDK, with the gateway's timeout and `max_retries`). No network.

The costs are worked out by hand: 11 input and 3 output tokens at the model's micro-USD rates per
million (gpt-6-luna and claude-haiku-5-5 both: input 100,000, output 500,000):
11 x 100,000 + 3 x 500,000 = 2,600,000, divided by 1,000,000 and rounded UP = 3 micro-USD.
"""

from __future__ import annotations

import json
from datetime import date
from typing import Any

import pytest
from agent_framework import Message
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.llm.core.budget import LlmCall
from app.llm.core.errors import FailureKind, ProviderError
from app.llm.maf.gateway import Gateway
from app.llm.maf.middleware import StoreNotAllowed
from app.settings import Settings
from tests.maf_wire import Attempt, FakeProviderHttp, Fault, sse

pytestmark = pytest.mark.anyio

DAY = date(2026, 1, 5)
EMAIL = "priya.shah@example.com"
OPENAI = "openai:gpt-6-luna"
ANTHROPIC = "anthropic:claude-haiku-5-5"


@pytest.fixture
def make(settings: Settings, db_session: AsyncSession) -> Any:
    sessions = async_sessionmaker(
        bind=db_session.bind, join_transaction_mode="create_savepoint", expire_on_commit=False
    )

    def build(model: str, fake: FakeProviderHttp, **options: Any) -> Gateway:
        gateway = Gateway(
            settings.model_copy(update={"llm_model": model, **options}),
            lambda: sessions,
            http_client=fake.client,
        )
        gateway.meter.cap, gateway.meter.today = 1_000_000, lambda: DAY
        return gateway

    return build


async def rows(session: AsyncSession) -> list[LlmCall]:
    return list(await session.scalars(select(LlmCall).where(LlmCall.day == DAY)))


async def test_an_openai_call_sends_store_false_the_output_limit_and_redacted_text(
    make: Any, db_session: AsyncSession
) -> None:
    fake = FakeProviderHttp("openai")
    gateway = make(OPENAI, fake, llm_max_tokens=300)
    response = await gateway.client().get_response([Message("user", [f"Mail {EMAIL}"])])
    assert response.text == "4"
    [body] = fake.bodies
    # The Responses API stores the conversation unless told not to: we keep history ourselves.
    assert body["store"] is False
    assert body["max_output_tokens"] == 300
    assert EMAIL not in str(body)
    assert body["input"][0]["content"][0]["text"] == "Mail [EMAIL_1]"
    [row] = await rows(db_session)
    assert (row.status, row.cost_micro_usd, row.redacted_count) == ("settled", 3, 1)
    assert (row.input_tokens, row.output_tokens) == (11, 3)


async def test_an_anthropic_call_sends_the_output_limit_it_would_otherwise_default(
    make: Any, db_session: AsyncSession
) -> None:
    fake = FakeProviderHttp("anthropic")
    gateway = make(ANTHROPIC, fake, llm_max_tokens=300)
    await gateway.client().get_response([Message("user", ["hi"])])
    # MAF's Anthropic client would send 1,024 when none is set; the gateway sets its own.
    assert fake.bodies[0]["max_tokens"] == 300
    [row] = await rows(db_session)
    assert (row.status, row.cost_micro_usd) == ("settled", 3)


async def test_store_true_is_refused_before_anything_is_reserved_or_sent(
    make: Any, db_session: AsyncSession
) -> None:
    fake = FakeProviderHttp("openai")
    gateway = make(OPENAI, fake)
    with pytest.raises(StoreNotAllowed):
        await gateway.client().get_response([Message("user", ["hi"])], options={"store": True})
    assert fake.attempts == []
    assert await rows(db_session) == []


async def test_a_streamed_call_is_refused_before_it_reaches_the_provider(
    make: Any, db_session: AsyncSession
) -> None:
    fake = FakeProviderHttp("anthropic")
    gateway = make(ANTHROPIC, fake)
    stream = gateway.client().get_response([Message("user", ["hi"])], stream=True)
    with pytest.raises(Exception, match="not supported"):
        await stream.get_final_response()
    assert fake.attempts == []
    assert await rows(db_session) == []


async def test_the_fake_streams_usage_the_way_anthropic_does() -> None:
    # Used by the streaming PR: message_start seeds the usage and message_delta carries the final
    # output count. Here only that the fake's bytes are what the real SDK parses.
    fake = FakeProviderHttp("anthropic", text="four")
    from anthropic import AsyncAnthropic

    sdk = AsyncAnthropic(api_key="k", http_client=fake.client)
    async with sdk.messages.stream(
        model="claude-haiku-5-5", max_tokens=10, messages=[{"role": "user", "content": "hi"}]
    ) as stream:
        final = await stream.get_final_message()
    assert final.content[0].text == "four"  # type: ignore[union-attr]
    assert sse("x", {"input_tokens": 1, "output_tokens": 2}).startswith(b"event: message_start")


def slow_hint(attempt: Attempt, number: int) -> Fault | None:
    # Every attempt is told to retry after 1 ms, so the SDK's own wait stays short.
    return Fault(status=429, headers={"retry-after-ms": "1"})


async def test_the_sdk_retries_a_rate_limit_and_the_call_settles_once(
    make: Any, db_session: AsyncSession
) -> None:
    fake = FakeProviderHttp(
        "anthropic",
        plan=lambda attempt, number: slow_hint(attempt, number) if number == 1 else None,
    )
    gateway = make(ANTHROPIC, fake, llm_max_retries=2)
    assert (await gateway.client().get_response([Message("user", ["hi"])])).text == "4"
    assert len(fake.attempts) == 2  # one retry, by the SDK
    [row] = await rows(db_session)  # one call, one row, however many attempts
    assert row.status == "settled"


async def test_a_rate_limit_that_outlasts_the_retries_is_released_with_the_providers_hint(
    make: Any, db_session: AsyncSession
) -> None:
    fake = FakeProviderHttp("anthropic", plan=slow_hint)
    gateway = make(ANTHROPIC, fake, llm_max_retries=2)
    with pytest.raises(ProviderError) as raised:
        await gateway.client().get_response([Message("user", ["hi"])])
    assert len(fake.attempts) == 3  # the first try and two retries: the SDK is the only retry layer
    # MAF's Anthropic client calls a 429 an invalid request; the SDK error behind it decides.
    assert (raised.value.kind, raised.value.retry_after) == (FailureKind.RATE_LIMITED, "1")
    [row] = await rows(db_session)
    assert (row.status, row.cost_micro_usd) == ("failed", 0)


async def test_a_bad_request_is_not_retried(make: Any, db_session: AsyncSession) -> None:
    fake = FakeProviderHttp("anthropic", plan=lambda _a, _n: Fault(status=400))
    gateway = make(ANTHROPIC, fake, llm_max_retries=2)
    with pytest.raises(ProviderError) as raised:
        await gateway.client().get_response([Message("user", ["hi"])])
    assert len(fake.attempts) == 1
    assert raised.value.kind is FailureKind.INVALID_REQUEST


async def test_the_gateways_timeout_applies_to_each_attempt_and_the_reservation_is_kept(
    make: Any, db_session: AsyncSession
) -> None:
    fake = FakeProviderHttp("openai", plan=lambda _a, _n: Fault(status=200, delay=2.0))
    gateway = make(OPENAI, fake, llm_timeout_seconds=0.05, llm_max_retries=1)
    with pytest.raises(ProviderError) as raised:
        await gateway.client().get_response([Message("user", ["hi"])])
    assert len(fake.attempts) == 2
    assert raised.value.kind is FailureKind.TIMED_OUT
    [row] = await rows(db_session)
    # A timeout can come after the provider generated (and billed): the reservation stays.
    assert row.status == "failed"
    assert row.cost_micro_usd == row.reserved_micro_usd > 0


async def test_closing_the_gateway_closes_the_sdk_client(make: Any) -> None:
    fake = FakeProviderHttp("openai")
    gateway = make(OPENAI, fake)
    await gateway.aclose()
    assert gateway._sdk.is_closed()


class Contact(BaseModel):
    email: str


SCHEMA = {
    "type": "object",
    "properties": {"email": {"type": "string"}},
    "required": ["email"],
    "additionalProperties": False,
}


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
@pytest.mark.parametrize("schema", [Contact, SCHEMA], ids=["pydantic", "dict"])
async def test_a_structured_answer_is_restored_in_its_text_and_in_its_value(
    make: Any, provider: str, schema: Any
) -> None:
    fake = FakeProviderHttp(provider, text='{"email": "[EMAIL_1]"}')
    gateway = make(OPENAI if provider == "openai" else ANTHROPIC, fake)
    response = await gateway.client().get_response(
        [Message("user", [f"Mail {EMAIL}"])], options={"response_format": schema}
    )
    assert EMAIL not in str(fake.bodies)
    assert json.loads(response.text) == {"email": EMAIL}
    value = response.value
    assert (value.email if isinstance(value, BaseModel) else value["email"]) == EMAIL


def test_the_response_attributes_the_value_fix_resets_exist_in_the_pinned_core() -> None:
    # redaction._reparse resets private attributes of agent-framework-core 1.25.0: a bump that
    # renames them must fail here, not leave placeholders in `value`.
    from agent_framework import ChatResponse

    response: Any = ChatResponse(messages=[Message("assistant", ["{}"])], response_format=Contact)
    for name in ("_value", "_value_parsed", "_response_format"):
        assert hasattr(response, name), name
