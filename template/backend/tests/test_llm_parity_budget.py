"""What every model framework must do with the daily budget (#35): reserve the most a call can cost
before it is sent, settle it at its real cost, give the reservation back only when the provider
refused the call before doing any work, refuse what the budget cannot bound, and answer a failed
call without the provider's words. The database is real; the provider is scripted by the harness
(tests/harness/), so each test runs once per framework the project chose and imports none.

A test here states a rule of the budget for ALL frameworks. Anything that holds for one framework
only (LangChain's callbacks, its input_bound) is in that framework's own test file."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from datetime import date
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.llm import prices
from app.llm.core import bounds
from app.llm.core.bounds import PromptTooLong, UnboundedMedia, UnpricedRequest
from app.llm.core.budget import (
    BudgetExhausted,
    LlmCall,
    LlmSpendDay,
    LlmSpendUserDay,
    UserBudgetExhausted,
)
from app.settings import Settings
from tests.harness import Failure, FakeEmbeddingsApi, Harness, Message, Reply, user
from tests.models import EMBEDDING_MODEL

pytestmark = pytest.mark.anyio

MODEL = "anthropic:claude-sonnet-5-5"
PRICE = prices.PRICES[MODEL]
DAY = date(2026, 1, 2)
EMBEDDING_DAY = date(2026, 1, 3)
QUESTION: list[Message] = [user("What is 2 + 2?")]
ANSWER = Reply("4")  # 1,000 input and 500 output tokens


@pytest.fixture
def sessions(db_session: AsyncSession) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(
        bind=db_session.bind, join_transaction_mode="create_savepoint", expire_on_commit=False
    )


@pytest.fixture
def gateway(
    framework: Harness, settings: Settings, sessions: async_sessionmaker[AsyncSession]
) -> Any:
    """The project's gateway on this test's transaction, for a model priced at $2 and $10 per
    million tokens (input and output), with a 500-token answer limit and a $1 day."""
    built = framework.gateway(
        settings.model_copy(update={"llm_model": MODEL, "llm_max_tokens": 500}), lambda: sessions
    )
    chat = framework.meter(built)
    chat.cap, chat.today = 1_000_000, lambda: DAY
    return built


async def calls(session: AsyncSession) -> list[LlmCall]:
    return list((await session.scalars(select(LlmCall).where(LlmCall.day == DAY))).all())


async def committed(session: AsyncSession) -> int | None:
    return await session.scalar(
        select(LlmSpendDay.committed_micro_usd).where(LlmSpendDay.day == DAY)
    )


async def test_a_call_is_settled_at_its_real_cost(
    framework: Harness, gateway: Any, db_session: AsyncSession
) -> None:
    framework.install_scripted(gateway, [ANSWER])
    assert await framework.ask(gateway, QUESTION) == "4"

    [call] = await calls(db_session)
    assert (call.status, call.cost_micro_usd) == ("settled", 7_000)  # 1,000 x $2/M + 500 x $10/M
    assert (call.input_tokens, call.output_tokens) == (1_000, 500)
    assert call.reserved_micro_usd >= (call.cost_micro_usd or 0)
    assert await committed(db_session) == 7_000


async def test_prompt_metadata_is_recorded_with_the_call(
    framework: Harness, gateway: Any, db_session: AsyncSession
) -> None:
    framework.install_scripted(gateway, [ANSWER])
    await framework.ask(gateway, QUESTION, tags={"prompt": "sums", "prompt_sha": "abc"})
    [call] = await calls(db_session)
    assert (call.prompt, call.prompt_sha) == ("sums", "abc")


async def test_a_call_that_could_cross_the_cap_is_never_sent(
    framework: Harness, gateway: Any, db_session: AsyncSession
) -> None:
    framework.meter(gateway).cap = 1_000
    framework.install_scripted(gateway, [ANSWER])
    with pytest.raises(BudgetExhausted):
        await framework.ask(gateway, QUESTION)
    assert framework.scripted_left(gateway) == 1  # the model never answered
    assert await calls(db_session) == []
    assert await committed(db_session) is None


@pytest.mark.parametrize("kind", ["rate_limit", "context_overflow", "authentication"])
async def test_a_call_the_provider_rejected_releases_its_reservation(
    framework: Harness, gateway: Any, db_session: AsyncSession, kind: str
) -> None:
    failure = Failure(kind, "slow down")
    framework.install_scripted(gateway, [failure])
    with pytest.raises(framework.raised_by(failure)):
        await framework.ask(gateway, QUESTION)
    [call] = await calls(db_session)
    assert (call.status, call.cost_micro_usd) == ("failed", 0)
    assert await committed(db_session) == 0


@pytest.mark.parametrize("kind", ["timeout", "connection", "server", "unknown"])
async def test_a_call_that_may_have_been_processed_keeps_its_reservation(
    framework: Harness, gateway: Any, db_session: AsyncSession, kind: str
) -> None:
    # A timeout or a dropped connection can come after the provider generated (and billed).
    failure = Failure(kind, "?")
    framework.install_scripted(gateway, [failure])
    with pytest.raises(framework.raised_by(failure)):
        await framework.ask(gateway, QUESTION)
    [call] = await calls(db_session)
    assert call.status == "failed"
    assert call.cost_micro_usd == call.reserved_micro_usd == await committed(db_session)


async def test_a_call_without_usage_keeps_its_reservation(
    framework: Harness, gateway: Any, db_session: AsyncSession
) -> None:
    framework.install_scripted(gateway, [Reply("4", usage=None)])
    await framework.ask(gateway, QUESTION)
    [call] = await calls(db_session)
    assert call.status == "settled"
    assert call.cost_micro_usd == call.reserved_micro_usd == await committed(db_session)


# --- streamed calls (#35 PR 3) -----------------------------------------------------------------


async def test_a_streamed_call_is_settled_from_its_final_usage(
    framework: Harness, gateway: Any, db_session: AsyncSession
) -> None:
    framework.install_scripted(gateway, [Reply("four", pieces=("fo", "ur"))])
    assert await framework.stream(gateway, QUESTION) == ["fo", "ur"]
    [call] = await calls(db_session)
    # The same cost as the unstreamed call: the usage the provider reported at the end.
    assert (call.status, call.cost_micro_usd) == ("settled", 7_000)
    assert (call.input_tokens, call.output_tokens) == (1_000, 500)


async def test_a_streamed_call_without_usage_keeps_its_reservation(
    framework: Harness, gateway: Any, db_session: AsyncSession
) -> None:
    framework.install_scripted(gateway, [Reply("four", usage=None, pieces=("fo", "ur"))])
    await framework.stream(gateway, QUESTION)
    [call] = await calls(db_session)
    assert call.status == "settled"
    assert call.cost_micro_usd == call.reserved_micro_usd == await committed(db_session)


async def test_a_stream_the_consumer_stops_reading_keeps_its_reservation(
    framework: Harness, gateway: Any, db_session: AsyncSession
) -> None:
    # The provider may bill what it generated before the stream was closed, and reports nothing.
    framework.install_scripted(gateway, [Reply("one two three", pieces=("one ", "two ", "three"))])
    assert await framework.stream(gateway, QUESTION, take=1) == ["one "]
    [call] = await calls(db_session)
    assert call.status == "failed"
    assert call.cost_micro_usd == call.reserved_micro_usd == await committed(db_session)


async def test_a_streamed_call_the_provider_rejected_releases_its_reservation(
    framework: Harness, gateway: Any, db_session: AsyncSession
) -> None:
    framework.install_scripted(gateway, [Failure("rate_limit")])
    with pytest.raises(framework.raised_by(Failure("rate_limit"))):
        await framework.stream(gateway, QUESTION)
    [call] = await calls(db_session)
    assert (call.status, call.cost_micro_usd) == ("failed", 0)
    assert await committed(db_session) == 0


async def test_a_streamed_call_that_could_cross_the_cap_is_never_sent(
    framework: Harness, gateway: Any, db_session: AsyncSession
) -> None:
    framework.meter(gateway).cap = 1_000
    framework.install_scripted(gateway, [ANSWER])
    with pytest.raises(BudgetExhausted):
        await framework.stream(gateway, QUESTION)
    assert framework.scripted_left(gateway) == 1
    assert await calls(db_session) == []


async def test_a_streamed_call_records_its_tags(
    framework: Harness, gateway: Any, db_session: AsyncSession
) -> None:
    framework.install_scripted(gateway, [ANSWER])
    await framework.stream(gateway, QUESTION, tags={"feature": "chat", "user_ref": "u_1"})
    [call] = await calls(db_session)
    assert (call.feature, call.user_ref) == ("chat", "u_1")


async def test_a_larger_bound_limit_is_reserved_for(
    framework: Harness, gateway: Any, db_session: AsyncSession
) -> None:
    # Before each ask the cap is set to what is already spent plus what a plain call reserves
    # (plus one micro-USD of room), so the only difference between the asks is the bound: the
    # plain one fits, the 5,000-token one must not slip through. It cannot go vacuous if the
    # input bound changes, and the control call's own spend is counted before the final ask.
    framework.install_scripted(gateway, [ANSWER, ANSWER, ANSWER])
    await framework.ask(gateway, QUESTION)
    [probe] = await calls(db_session)

    async def room_for_one_plain_call() -> None:
        spent = await committed(db_session)
        assert spent is not None
        framework.meter(gateway).cap = spent + probe.reserved_micro_usd + 1

    await room_for_one_plain_call()
    await framework.ask(gateway, QUESTION)  # control: the same ask without options is allowed
    await room_for_one_plain_call()
    with pytest.raises(BudgetExhausted):
        await framework.ask(
            gateway, QUESTION, options=framework.params_for_tools(max_output_tokens=5_000)
        )


async def test_a_long_tool_definition_is_reserved_for_in_full(
    framework: Harness, gateway: Any, db_session: AsyncSession
) -> None:
    # A tool's description and parameter schema are sent with the call: they count. 50,000
    # characters are at least 50,000 bytes, which the reservation must cover at the lowest input
    # rate it ever charges ($2 per million: 2 micro-USD a token).
    tool = {
        "type": "function",
        "function": {
            "name": "look_up",
            "description": "x" * 50_000,
            "parameters": {"type": "object", "properties": {}},
        },
    }
    framework.install_scripted(gateway, [ANSWER, ANSWER])
    await framework.ask(gateway, QUESTION)
    # Each row is told apart by when it appeared, not by its place in the result: the query has
    # no order, and created_at is the transaction's start, the same for both.
    [plain] = await calls(db_session)
    await framework.ask(gateway, QUESTION, options=framework.params_for_tools(tools=[tool]))
    [with_tool] = [c for c in await calls(db_session) if c.id != plain.id]
    assert with_tool.reserved_micro_usd - plain.reserved_micro_usd >= 50_000 * 2


async def test_images_are_reserved_at_their_maximum_however_they_are_sent(
    framework: Harness, gateway: Any, db_session: AsyncSession
) -> None:
    image = {"type": "image", "url": "https://example.com/chart.png"}
    framework.install_scripted(gateway, [ANSWER, ANSWER])
    await framework.ask(gateway, QUESTION)
    await framework.ask(gateway, [user([image, {"type": "text", "text": "What is 2 + 2?"}])])
    text_only, with_image = sorted(c.reserved_micro_usd for c in await calls(db_session))
    # At least IMAGE_TOKENS more input at the highest input rate ($4/M for 1h cache writes).
    assert with_image - text_only >= bounds.IMAGE_TOKENS * 4


async def test_media_the_budget_cant_bound_is_refused_before_sending(
    framework: Harness, gateway: Any, db_session: AsyncSession
) -> None:
    pdf = {"type": "file", "url": "https://example.com/report.pdf", "mime_type": "application/pdf"}
    framework.install_scripted(gateway, [ANSWER])
    with pytest.raises(UnboundedMedia) as refused:
        await framework.ask(gateway, [user([pdf])])
    assert (refused.value.status, refused.value.kind) == (422, "file")
    assert framework.scripted_left(gateway) == 1
    assert await calls(db_session) == []


async def user_total(session: AsyncSession, user_ref: str) -> int | None:
    return await session.scalar(
        select(LlmSpendUserDay.committed_micro_usd).where(
            LlmSpendUserDay.day == DAY, LlmSpendUserDay.user_ref == user_ref
        )
    )


async def test_one_users_cap_refuses_only_that_user(
    framework: Harness, gateway: Any, db_session: AsyncSession
) -> None:
    capped = framework.meter(gateway)
    capped.user_cap = 1_000_000
    as_user = {"user_ref": "u_1"}
    framework.install_scripted(gateway, [ANSWER, ANSWER])
    await framework.ask(gateway, QUESTION, tags=as_user)
    # The first call settled at 7,000; a cap just short of 7,000 plus a fresh reservation
    # leaves no room for a second call.
    reservation = await db_session.scalar(select(LlmCall.reserved_micro_usd))
    assert reservation is not None
    capped.user_cap = 7_000 + reservation - 1
    with pytest.raises(UserBudgetExhausted) as refused:
        await framework.ask(gateway, QUESTION, tags=as_user)
    assert (refused.value.status, int(refused.value.headers["Retry-After"]) > 0) == (429, True)
    # Someone else is unaffected, and the refusal left the day's total as it was.
    before = await committed(db_session)
    await framework.ask(gateway, QUESTION, tags={"user_ref": "u_2"})
    assert await user_total(db_session, "u_1") == 7_000
    assert await user_total(db_session, "u_2") == 7_000
    assert await committed(db_session) == (before or 0) + 7_000


def limited(framework: Harness, gateway: Any, max_prompt_tokens: int) -> None:
    framework.meter(gateway).price = prices.Price(
        PRICE.rates,
        checked="2026-11-06",
        source="https://example.com/pricing",
        max_prompt_tokens=max_prompt_tokens,
    )


async def test_a_prompt_exactly_at_the_priced_length_is_allowed_and_one_more_is_refused(
    framework: Harness,
    gateway: Any,
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The limit applies to the budget's input bound b (an over-estimate of tokens, core/bounds.py),
    # and a call is refused only when b > max_prompt_tokens. So a limit equal to b lets the call
    # through, and a limit of b - 1 (the prompt being one more than allowed) refuses it. For OpenAI
    # the shipped limit is 272,000 (see test_llm_prices): a prompt whose input bound is 272,000 is
    # allowed, one whose bound is 272,001 is refused. The bound is a byte-based over-estimate
    # (UTF-8 bytes of the JSON payload plus a 1,024 margin), so real refusals start near ~68K
    # English tokens. That reads the page's "272K" as 1,000s. The bound is read from the Meter,
    # which every framework reserves through.
    meter = framework.meter(gateway)
    bound_of_the_call: list[int] = []
    reserve = meter.reserve

    async def watching(call_id: uuid.UUID, *, input_tokens: int, **rest: Any) -> int:
        bound_of_the_call.append(input_tokens)
        return await reserve(call_id, input_tokens=input_tokens, **rest)

    monkeypatch.setattr(meter, "reserve", watching)
    framework.install_scripted(gateway, [ANSWER, ANSWER])
    await framework.ask(gateway, QUESTION)
    bound = bound_of_the_call[-1]
    assert [call.status for call in await calls(db_session)] == ["settled"]

    limited(framework, gateway, bound)
    await framework.ask(gateway, QUESTION)
    assert [call.status for call in await calls(db_session)] == ["settled", "settled"]

    limited(framework, gateway, bound - 1)
    with pytest.raises(PromptTooLong):
        await framework.ask(gateway, QUESTION)
    assert len(await calls(db_session)) == 2  # the refused call reserved nothing


async def test_a_prompt_past_the_priced_length_is_refused_before_reserving(
    framework: Harness, gateway: Any, db_session: AsyncSession
) -> None:
    limited(framework, gateway, 10)
    framework.install_scripted(gateway, [ANSWER])
    with pytest.raises(PromptTooLong) as refused:
        await framework.ask(gateway, QUESTION)
    assert refused.value.status == 422
    assert await calls(db_session) == []


@pytest.mark.parametrize(
    "tool",
    [
        {"type": "web_search"},  # OpenAI, billed per search
        {"type": "web_search_20250305", "name": "web_search", "max_uses": 3},  # Anthropic
        {"type": "file_search", "vector_store_ids": ["vs_1"]},
        {"type": "code_interpreter", "container": {"type": "auto"}},
    ],
)
async def test_a_tool_the_provider_runs_and_bills_per_use_is_refused_before_reserving(
    framework: Harness, gateway: Any, db_session: AsyncSession, tool: dict[str, Any]
) -> None:
    framework.install_scripted(gateway, [ANSWER])
    with pytest.raises(UnpricedRequest, match=str(tool["type"])):
        await framework.ask(gateway, QUESTION, options=framework.params_for_tools(tools=[tool]))
    assert await calls(db_session) == []


async def test_tools_the_app_runs_cost_only_tokens_and_are_allowed(
    framework: Harness, gateway: Any, db_session: AsyncSession
) -> None:
    openai_style = {"type": "function", "function": {"name": "f", "parameters": {}}}
    anthropic_style = {"name": "f", "description": "d", "input_schema": {"type": "object"}}
    framework.install_scripted(gateway, [ANSWER])
    await framework.ask(
        gateway,
        QUESTION,
        options=framework.params_for_tools(tools=[openai_style, anthropic_style]),
    )
    assert [c.status for c in await calls(db_session)] == ["settled"]


@pytest.mark.parametrize("tier", ["priority", "scale"])
async def test_a_service_tier_priced_above_the_standard_rates_is_refused(
    framework: Harness, gateway: Any, db_session: AsyncSession, tier: str
) -> None:
    framework.install_scripted(gateway, [ANSWER])
    with pytest.raises(UnpricedRequest, match=tier):
        await framework.ask(
            gateway, QUESTION, options=framework.params_for_tools(service_tier=tier)
        )
    assert await calls(db_session) == []


@pytest.mark.parametrize("tier", ["auto", "default", "flex", "standard_only"])
async def test_service_tiers_at_or_below_the_standard_rates_are_allowed(
    framework: Harness, gateway: Any, tier: str
) -> None:
    framework.install_scripted(gateway, [ANSWER])
    await framework.ask(gateway, QUESTION, options=framework.params_for_tools(service_tier=tier))


# --- how a failed call reaches the client: problem responses without the provider's words ---------


@pytest.mark.parametrize(
    ("failure", "status", "retry_after"),
    [
        (Failure("rate_limit", "slow down: prompt was SECRET"), 503, "30"),
        (Failure("timeout", "timed out"), 504, None),
        (Failure("context_overflow", "context: 900k tokens"), 422, None),
        (Failure("connection", "reset"), 502, "30"),
        (Failure("server", "500 from provider"), 502, "30"),
        (Failure("authentication", "bad key"), 503, None),
    ],
)
async def test_model_failures_become_problems_without_the_providers_message(
    framework: Harness,
    gateway: Any,
    app: FastAPI,
    client: httpx.AsyncClient,
    failure: Failure,
    status: int,
    retry_after: str | None,
) -> None:
    framework.install_scripted(gateway, [failure])

    @app.get("/model")
    async def model() -> None:
        await framework.ask(gateway, QUESTION)

    response = await client.get("/model")
    assert response.status_code == status
    assert response.headers["content-type"] == "application/problem+json"
    assert response.headers.get("Retry-After") == retry_after
    assert "SECRET" not in response.text
    assert failure.message not in response.text


async def test_the_providers_retry_after_is_passed_on(
    framework: Harness, gateway: Any, app: FastAPI, client: httpx.AsyncClient
) -> None:
    framework.install_scripted(gateway, [Failure("rate_limit", "slow down", retry_after="7")])

    @app.get("/limited")
    async def limited_route() -> None:
        await framework.ask(gateway, QUESTION)

    assert (await client.get("/limited")).headers["Retry-After"] == "7"


async def test_an_invalid_request_the_app_built_is_a_plain_500(
    framework: Harness, gateway: Any, app: FastAPI, client: httpx.AsyncClient
) -> None:
    framework.install_scripted(gateway, [Failure("invalid_request", "unknown parameter")])

    @app.get("/bug")
    async def bug() -> None:
        await framework.ask(gateway, QUESTION)

    response = await client.get("/bug")
    assert response.status_code == 500
    assert "unknown parameter" not in response.text


# --- embeddings share the same budget -------------------------------------------------------------


@pytest.fixture
def embeddings(framework: Harness, gateway: Any) -> Callable[..., FakeEmbeddingsApi]:
    """The gateway's embeddings on a fake of OpenAI's endpoint; the fake records its requests."""

    def make(cap: int = 1_000_000, user_cap: int | None = None, **api: Any) -> FakeEmbeddingsApi:
        fake = FakeEmbeddingsApi(**api)
        framework.install_embeddings(gateway, fake)
        meter = framework.embedding_meter(gateway)
        meter.cap, meter.user_cap, meter.today = cap, user_cap, lambda: EMBEDDING_DAY
        return fake

    return make


async def day_calls(session: AsyncSession) -> list[LlmCall]:
    return list((await session.scalars(select(LlmCall).where(LlmCall.day == EMBEDDING_DAY))).all())


async def test_documents_are_embedded_in_order_and_settled_at_their_real_cost(
    framework: Harness,
    gateway: Any,
    db_session: AsyncSession,
    embeddings: Callable[..., FakeEmbeddingsApi],
) -> None:
    embeddings()
    vectors = await framework.embed_documents(gateway, ["a", "b", "c"], feature="ingest")
    assert vectors == [[0.0] * 3, [1.0] * 3, [2.0] * 3]
    [call] = await day_calls(db_session)
    assert (call.status, call.model, call.feature) == ("settled", EMBEDDING_MODEL, "ingest")
    assert (call.input_tokens, call.output_tokens) == (21, 0)
    # 21 tokens at $0.02 per million, rounded up to a whole micro-USD.
    assert call.cost_micro_usd == 1
    assert (
        await db_session.scalar(
            select(LlmSpendDay.committed_micro_usd).where(LlmSpendDay.day == EMBEDDING_DAY)
        )
        == 1
    )


async def test_a_query_is_one_metered_call(
    framework: Harness,
    gateway: Any,
    db_session: AsyncSession,
    embeddings: Callable[..., FakeEmbeddingsApi],
) -> None:
    fake = embeddings()
    assert await framework.embed_query(gateway, "what is rag?") == [0.0] * 3
    assert fake.requests == [["what is rag?"]]
    assert len(await day_calls(db_session)) == 1


async def test_an_embedding_the_budget_cant_cover_is_never_sent(
    framework: Harness,
    gateway: Any,
    db_session: AsyncSession,
    embeddings: Callable[..., FakeEmbeddingsApi],
) -> None:
    fake = embeddings(cap=0)
    with pytest.raises(BudgetExhausted):
        await framework.embed_documents(gateway, ["a"])
    assert fake.requests == []
    assert await day_calls(db_session) == []


async def test_one_users_cap_applies_to_embeddings(
    framework: Harness, gateway: Any, embeddings: Callable[..., FakeEmbeddingsApi]
) -> None:
    fake = embeddings(user_cap=0)
    with pytest.raises(UserBudgetExhausted):
        await framework.embed_documents(gateway, ["a"], user_ref="u_1")
    assert fake.requests == []


async def test_a_request_the_provider_rejected_releases_its_reservation(
    framework: Harness,
    gateway: Any,
    db_session: AsyncSession,
    embeddings: Callable[..., FakeEmbeddingsApi],
) -> None:
    rejected = Failure("invalid_request", "too long")
    embeddings(error=rejected)
    # The same family of errors as chat calls, so the framework's error handlers answer it the
    # same way.
    with pytest.raises(framework.raised_by_embeddings(rejected)):
        await framework.embed_documents(gateway, ["a"])
    [call] = await day_calls(db_session)
    assert (call.status, call.cost_micro_usd) == ("failed", 0)


async def test_a_request_that_may_have_been_processed_keeps_its_reservation(
    framework: Harness,
    gateway: Any,
    db_session: AsyncSession,
    embeddings: Callable[..., FakeEmbeddingsApi],
) -> None:
    timeout = Failure("timeout")
    embeddings(error=timeout)
    with pytest.raises(framework.raised_by_embeddings(timeout)):
        await framework.embed_documents(gateway, ["a"])
    [call] = await day_calls(db_session)
    assert call.status == "failed"
    assert call.cost_micro_usd == call.reserved_micro_usd > 0
