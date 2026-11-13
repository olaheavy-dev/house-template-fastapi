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
from typing import Annotated, Any

import pytest
from agent_framework import Message
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.llm.core.budget import LlmCall
from app.llm.core.context import USAGE_KEY
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


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
async def test_a_streamed_call_is_restored_as_it_arrives_and_settled_from_its_final_usage(
    make: Any, db_session: AsyncSession, provider: str
) -> None:
    # The placeholder is split across three deltas; the consumer never sees a piece of it, the
    # provider never sees the email, and the call costs what the same unstreamed call costs:
    # 11 input and 3 output tokens = 3 micro-USD (module notes).
    pieces = ["Mailed [EM", "AIL", "_1]."]
    fake = FakeProviderHttp(provider, text="".join(pieces), pieces=pieces)
    gateway = make(OPENAI if provider == "openai" else ANTHROPIC, fake, llm_max_tokens=300)
    shown: list[str] = []
    whole: list[str] = []
    stream = gateway.client().get_response([Message("user", [f"Mail {EMAIL}"])], stream=True)
    async with stream:
        async for update in stream:
            for content in update.contents:
                if content.type != "text" or not content.text:
                    continue
                # MAF's merge rule: "append" text is a piece, "replace" text is the whole text
                # (OpenAI ends its text with response.output_text.done, which MAF passes on so).
                (whole if content.merge_mode == "replace" else shown).append(content.text)
        final = await stream.get_final_response()
    assert "".join(shown) == final.text == f"Mailed {EMAIL}."
    assert whole in ([], [f"Mailed {EMAIL}."])
    assert not any("[EM" in piece for piece in shown)
    [body] = fake.bodies
    assert body["stream"] is True
    assert EMAIL not in str(body)
    if provider == "openai":
        assert body["store"] is False
        assert body["max_output_tokens"] == 300
    else:
        assert body["max_tokens"] == 300
    [row] = await rows(db_session)
    assert (row.status, row.cost_micro_usd, row.redacted_count) == ("settled", 3, 1)
    assert (row.input_tokens, row.output_tokens) == (11, 3)


async def test_openais_whole_text_at_the_end_replaces_what_was_held_back(
    make: Any, db_session: AsyncSession
) -> None:
    # The answer ends in a lone `[`, which the stream holds back (it could open a placeholder).
    # OpenAI then sends the whole text (response.output_text.done, MAF's merge_mode "replace"):
    # it supersedes what was held, so the whole text is restored once and nothing is doubled.
    pieces = ["Mailed [EM", "AIL_1] ["]
    fake = FakeProviderHttp("openai", text="".join(pieces), pieces=pieces)
    gateway = make(OPENAI, fake)
    whole: list[str] = []
    stream = gateway.client().get_response([Message("user", [f"Mail {EMAIL}"])], stream=True)
    async with stream:
        async for update in stream:
            whole += [
                c.text for c in update.contents if c.type == "text" and c.merge_mode == "replace"
            ]
        final = await stream.get_final_response()
    assert whole == [f"Mailed {EMAIL} ["]
    assert final.text == f"Mailed {EMAIL} ["


async def test_anthropics_cumulative_stream_usage_is_counted_once(
    make: Any, db_session: AsyncSession
) -> None:
    # message_delta repeats the input and cache-read counts already sent in message_start, as the
    # docs' example does. Counted once it is 11 input, 5 cache reads and 3 output tokens:
    # 11 x 100,000 + 5 x 10,000 + 3 x 500,000 = 2,650,000 / 1,000,000 = 2.65, rounded up = 3
    # micro-USD, and the row's input count is 16 (cache reads are part of the input total, as for
    # an unstreamed call, app/llm/maf/usage.py); added up twice it would be 32.
    usage = {"input_tokens": 11, "output_tokens": 3, "cache_read_input_tokens": 5}
    fake = FakeProviderHttp("anthropic", text="four", usage=usage, repeat_input_in_delta=True)
    gateway = make(ANTHROPIC, fake)
    stream = gateway.client().get_response([Message("user", ["hi"])], stream=True)
    final = await stream.get_final_response()
    assert final.text == "four"
    [row] = await rows(db_session)
    assert (row.input_tokens, row.output_tokens, row.cost_micro_usd) == (16, 3, 3)


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
async def test_a_stream_closed_early_keeps_its_reservation(
    make: Any, db_session: AsyncSession, provider: str
) -> None:
    pieces = ["one ", "two ", "three"]
    fake = FakeProviderHttp(provider, text="".join(pieces), pieces=pieces)
    gateway = make(OPENAI if provider == "openai" else ANTHROPIC, fake)
    stream = gateway.client().get_response([Message("user", ["hi"])], stream=True)
    async with stream:
        async for _update in stream:
            break
    [row] = await rows(db_session)
    assert row.status == "failed"
    assert row.cost_micro_usd == row.reserved_micro_usd


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
    assert (getattr(value, "email", None) or value["email"]) == EMAIL


def test_the_response_attributes_the_value_fix_resets_exist_in_the_pinned_core() -> None:
    # redaction._reparse resets private attributes of agent-framework-core 1.25.0: a bump that
    # renames them must fail here, not leave placeholders in `value`.
    from agent_framework import ChatResponse

    response: Any = ChatResponse(messages=[Message("assistant", ["{}"])], response_format=Contact)
    for name in ("_value", "_value_parsed", "_response_format"):
        assert hasattr(response, name), name


class Lead(BaseModel):
    """A lead. Never write to priya.shah@example.com."""

    email: str = Field(description="The lead's address", examples=["priya.shah@example.com"])


def lookup_lead(
    who: Annotated[str, Field(description="Their address, as priya.shah@example.com is")],
) -> str:
    """Look up a lead. Leads for priya.shah@example.com go to the office."""
    return who


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
async def test_tool_definitions_and_the_output_schema_reach_the_wire_with_placeholders(
    make: Any, provider: str
) -> None:
    # #112's review, through each real client and SDK: the tool and the schema are built by the
    # client's own code from the redacted copy and the redacted subclass, so the body holds
    # placeholders, the names as they were, and the value is the caller's class again.
    fake = FakeProviderHttp(provider, text='{"email": "[EMAIL_1]"}')
    gateway = make(OPENAI if provider == "openai" else ANTHROPIC, fake)
    response = await gateway.client().get_response(
        [Message("user", ["Extract the lead."])],
        options={"tools": [lookup_lead], "response_format": Lead, "tool_choice": "none"},
    )
    [body] = fake.bodies
    sent = json.dumps(body)
    assert EMAIL not in sent
    # The tool's description and its parameter's, the schema's description and its example.
    assert sent.count("[EMAIL_1]") == 4
    assert '"lookup_lead"' in sent
    if provider == "openai":
        assert body["text"]["format"]["name"] == "Lead"
    assert type(response.value) is Lead
    assert response.value.email == EMAIL


@pytest.mark.parametrize("model", [OPENAI, ANTHROPIC])
async def test_the_usage_recorded_on_an_answer_is_never_sent_back(model: str, make: Any) -> None:
    # The budget middleware records each call's usage on its answer for the context report (#46);
    # the answer goes back to the provider as history on the next call, its record must not.
    fake = FakeProviderHttp(model.split(":", 1)[0])
    gateway = make(model, fake)
    first = await gateway.client().get_response([Message("user", ["What is 2 + 2?"])])
    answer = first.messages[-1]
    assert USAGE_KEY in answer.additional_properties
    await gateway.client().get_response(
        [Message("user", ["What is 2 + 2?"]), answer, Message("user", ["And 3 + 3?"])]
    )
    sent = json.dumps(fake.bodies[1])
    assert "And 3 + 3?" in sent
    assert USAGE_KEY not in sent
    assert "input_tokens" not in sent
