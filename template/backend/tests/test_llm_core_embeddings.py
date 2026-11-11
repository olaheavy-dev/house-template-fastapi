"""The framework-neutral embeddings (#33 T4): batches metered on a Meter (core/embeddings.py) and
OpenAI's call behind them (core/openai_embeddings.py), with no LangChain involved. Money is worked
out by hand in the comments, never by the code under test: EMBED costs 20,000 micro-USD per million
tokens, which is 0.02 micro-USD per token; a reservation is the request's UTF-8 bytes plus the
1,024-token margin, a settled cost is the provider's token count, and both round UP to a whole
micro-USD. The provider bills 7 tokens per text in these tests.
"""

from __future__ import annotations

import subprocess
import sys
import uuid
from collections.abc import Callable, Iterable
from datetime import date
from types import SimpleNamespace
from typing import Any

import anyio
import httpx2
import openai
import pytest
from pydantic import SecretStr
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.llm.core import embeddings, metrics, openai_embeddings
from app.llm.core.bounds import PromptTooLong
from app.llm.core.budget import BudgetExhausted, CallTags, LlmCall, LlmSpendDay, Meter
from app.llm.core.errors import FailureKind, ProviderError
from app.llm.core.prices import Price, Rates
from app.llm.core.redaction import RedactionFailed, Redactor, Span, redaction_scope
from app.llm.core.usage import Usage
from app.llm.redaction_rules import DETECTORS
from tests.test_llm_core import FRAMEWORKS

pytestmark = pytest.mark.anyio

SOURCE = "https://example.com/pricing"
EMBED = Price(Rates(20_000, 0, 0, 0, 0), checked="2026-11-06", source=SOURCE)
MODEL = "openai:pin-embeddings"
DAY = date(2026, 1, 3)
K = FailureKind


def limited(tokens: int) -> Price:
    return Price(EMBED.rates, checked="2026-11-06", source=SOURCE, max_prompt_tokens=tokens)


async def embed_plain(*args: Any, **kwargs: Any) -> list[list[float]]:
    """embed() with redaction off, for the tests of batching, budget and errors."""
    return await embeddings.embed(*args, redactor=None, **kwargs)


class Provider:
    """Stands in for the code that sends a batch: records each one, bills 7 tokens per text."""

    def __init__(self, error: BaseException | None = None) -> None:
        self.requests: list[list[str]] = []
        self.error = error

    async def create(self, batch: list[str]) -> tuple[list[list[float]], Usage]:
        self.requests.append(list(batch))
        if self.error:
            raise self.error
        return [[float(len(text))] for text in batch], Usage(
            input_tokens=7 * len(batch), output_tokens=0
        )


# --- batching (plain Python) ----------------------------------------------------------------------


def test_the_limits_are_openais_per_request_minus_the_margin() -> None:
    assert embeddings.MAX_INPUTS == 2048
    assert embeddings.MAX_BYTES == 300_000 - 1_024 == 298_976


def test_batches_split_by_count() -> None:
    assert list(embeddings.batches(["ab", "cd", "e"], max_inputs=2, max_bytes=10)) == [
        ["ab", "cd"],
        ["e"],
    ]


def test_batches_split_by_bytes_and_allow_exactly_the_limit() -> None:
    # 4 + 4 = 8 is allowed (not over 8); adding 2 makes 10 > 8, so "cc" starts a new batch.
    assert list(embeddings.batches(["aaaa", "bbbb", "cc"], max_bytes=8)) == [
        ["aaaa", "bbbb"],
        ["cc"],
    ]


def test_batches_count_utf8_bytes_not_characters() -> None:
    # "é" is 2 bytes: 2 + 2 = 4 is allowed, a third makes 6 > 4.
    assert list(embeddings.batches(["é", "é", "é"], max_bytes=4)) == [["é", "é"], ["é"]]


def test_a_text_past_the_byte_limit_goes_alone() -> None:
    assert list(embeddings.batches(["x" * 20, "y"], max_bytes=8)) == [["x" * 20], ["y"]]


def test_batches_yield_nothing_for_no_texts_and_never_an_empty_batch() -> None:
    assert list(embeddings.batches([])) == []
    # A limit of zero or below puts every text alone and neither loops nor yields an empty batch.
    assert list(embeddings.batches(["a", "b"], max_bytes=-24)) == [["a"], ["b"]]


# --- embed(): reserve, send, settle --------------------------------------------------------------


@pytest.fixture
def sessions(db_session: AsyncSession) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(
        bind=db_session.bind, join_transaction_mode="create_savepoint", expire_on_commit=False
    )


@pytest.fixture
def meter(sessions: async_sessionmaker[AsyncSession]) -> Callable[..., Meter]:
    def make(price: Price = EMBED, cap: int = 10**9, **options: Any) -> Meter:
        return Meter(
            model=MODEL,
            price=price,
            cap_micro_usd=cap,
            sessionmaker=lambda: sessions,
            today=lambda: DAY,
            **options,
        )

    return make


async def rows(session: AsyncSession) -> list[LlmCall]:
    return list((await session.scalars(select(LlmCall).where(LlmCall.day == DAY))).all())


async def total(session: AsyncSession) -> int | None:
    return await session.scalar(
        select(LlmSpendDay.committed_micro_usd).where(LlmSpendDay.day == DAY)
    )


async def test_no_texts_send_no_request_and_leave_no_row(
    meter: Callable[..., Meter], db_session: AsyncSession
) -> None:
    provider = Provider()
    assert await embed_plain(meter(), provider.create, []) == []
    assert provider.requests == []
    assert (await rows(db_session), await total(db_session)) == ([], None)


async def test_a_batch_reserves_before_it_is_sent_and_settles_at_the_providers_usage(
    meter: Callable[..., Meter], db_session: AsyncSession
) -> None:
    clock = iter([10.0, 10.5])
    provider = Provider()
    tags = CallTags(feature="ingest", user_ref="u_1")
    vectors = await embed_plain(
        meter(clock=lambda: next(clock)), provider.create, ["a", "b", "c"], tags=tags
    )
    assert vectors == [[1.0], [1.0], [1.0]]
    assert provider.requests == [["a", "b", "c"]]
    [row] = await rows(db_session)
    # Reserved: (3 bytes + 1,024) = 1,027 tokens x 0.02 = 20.54 -> 21. Settled: 3 x 7 = 21 tokens
    # x 0.02 = 0.42 -> 1, so the day holds 1.
    assert (row.status, row.reserved_micro_usd, row.cost_micro_usd) == ("settled", 21, 1)
    assert (row.input_tokens, row.output_tokens) == (21, 0)
    assert row.latency_ms == 500  # (10.5 - 10.0) s
    assert (row.feature, row.user_ref, row.prompt) == ("ingest", "u_1", None)
    assert await total(db_session) == 1


@pytest.mark.parametrize(
    ("kind", "cost"),
    [
        # A provider rejection never processed the request: its reservation is given back.
        (K.RATE_LIMITED, 0),
        (K.INVALID_REQUEST, 0),
        # These may still have been billed: the reservation of 21 stays as the cost.
        (K.PROVIDER_FAILED, 21),
        (K.TIMED_OUT, 21),
    ],
)
async def test_a_provider_error_settles_by_its_kind_and_is_raised_as_it_was(
    meter: Callable[..., Meter], db_session: AsyncSession, kind: FailureKind, cost: int
) -> None:
    error = ProviderError(kind, source="X")
    provider = Provider(error)
    with pytest.raises(ProviderError) as raised:
        await embed_plain(meter(), provider.create, ["a", "b", "c"])
    assert raised.value is error
    [row] = await rows(db_session)
    # Reserved 21 as above.
    assert (row.status, row.reserved_micro_usd, row.cost_micro_usd) == ("failed", 21, cost)
    assert await total(db_session) == cost


async def test_an_unknown_error_keeps_the_reservation_and_is_raised_untouched(
    meter: Callable[..., Meter], db_session: AsyncSession
) -> None:
    error = RuntimeError("boom")
    with pytest.raises(RuntimeError) as raised:
        await embed_plain(meter(), Provider(error).create, ["a", "b", "c"])
    assert raised.value is error
    # Unknown: it may have been billed, so it costs its reservation (21).
    [row] = await rows(db_session)
    assert (row.status, row.cost_micro_usd) == ("failed", 21)
    # Not chained to itself or to anything else.
    assert (raised.value.__cause__, raised.value.__context__) == (None, None)


async def test_a_later_batch_the_budget_cant_cover_is_refused_and_never_sent(
    meter: Callable[..., Meter], db_session: AsyncSession
) -> None:
    provider = Provider()
    with pytest.raises(BudgetExhausted):
        await embed_plain(meter(cap=21), provider.create, ["a", "b"], max_inputs=1)
    # Batch 1: (1 + 1,024) = 1,025 x 0.02 = 20.5 -> 21 <= cap 21, reserved. It settles at
    # 7 x 0.02 = 0.14 -> 1. Batch 2 reserves 21 more: 1 + 21 = 22 > 21, refused.
    assert provider.requests == [["a"]]
    [row] = await rows(db_session)
    assert (row.status, row.cost_micro_usd) == ("settled", 1)
    assert await total(db_session) == 1


# --- embed(): the price's max_prompt_tokens applies to each request -------------------------------


async def test_a_request_at_the_price_limit_is_sent_and_one_byte_more_is_refused(
    meter: Callable[..., Meter], db_session: AsyncSession
) -> None:
    provider = Provider()
    m = meter(limited(1_030))
    # "abcdef": 6 + 1,024 = 1,030 tokens, not over the limit: sent. Reserved 1,030 x 0.02 = 20.6
    # -> 21.
    await embed_plain(m, provider.create, ["abcdef"])
    [row] = await rows(db_session)
    assert (provider.requests, row.reserved_micro_usd) == ([["abcdef"]], 21)
    # "abcdefg": 7 + 1,024 = 1,031 > 1,030: refused before anything is reserved or sent.
    with pytest.raises(PromptTooLong):
        await embed_plain(m, provider.create, ["abcdefg"])
    assert provider.requests == [["abcdef"]]
    assert len(await rows(db_session)) == 1


async def test_a_text_past_the_limit_refuses_the_whole_call_before_any_batch_is_sent(
    meter: Callable[..., Meter], db_session: AsyncSession
) -> None:
    provider = Provider()
    # "abc" alone would be fine (3 + 1,024 = 1,027 <= 1,030), but "abcdefg" is 1,031 > 1,030: the
    # call is refused up front, so "abc" is not sent and paid for first.
    with pytest.raises(PromptTooLong) as raised:
        await embed_plain(meter(limited(1_030)), provider.create, ["abc", "abcdefg"])
    assert raised.value.status == 422
    assert provider.requests == []
    assert (await rows(db_session), await total(db_session)) == ([], None)


async def test_the_reservation_is_the_sum_of_the_utf8_bytes_plus_the_margin(
    sessions: async_sessionmaker[AsyncSession], db_session: AsyncSession
) -> None:
    # One micro-USD per token (1,000,000 per million), so the margin does not hide the bytes.
    one = Price(Rates(1_000_000, 0, 0, 0, 0), checked="2026-11-06", source=SOURCE)
    m = Meter(
        model=MODEL,
        price=one,
        cap_micro_usd=10**9,
        sessionmaker=lambda: sessions,
        today=lambda: DAY,
    )
    await embed_plain(m, Provider().create, ["é", "ab"])
    [row] = await rows(db_session)
    # Reserved: "é" is 2 bytes, "ab" is 2: 2 + 2 + 1,024 margin = 1,028 tokens x 1 = 1,028.
    # Settled: 2 texts x 7 = 14 tokens x 1 = 14.
    assert (row.reserved_micro_usd, row.cost_micro_usd) == (1_028, 14)


async def test_texts_are_batched_under_the_price_limit(
    meter: Callable[..., Meter], db_session: AsyncSession
) -> None:
    provider = Provider()
    await embed_plain(meter(limited(1_030)), provider.create, ["abc", "abc", "abc"])
    # The byte budget is 1,030 - 1,024 = 6: two texts (3 + 3) fit, a third would make 9.
    assert provider.requests == [["abc", "abc"], ["abc"]]
    # Reserved: (6 + 1,024) = 1,030 x 0.02 = 20.6 -> 21; (3 + 1,024) = 1,027 x 0.02 = 20.54 -> 21.
    assert sorted(row.reserved_micro_usd for row in await rows(db_session)) == [21, 21]


async def test_a_limit_of_zero_means_no_limit(meter: Callable[..., Meter]) -> None:
    provider = Provider()
    # With a limit of 0 taken literally the byte budget would be -1,024 and every text alone.
    await embed_plain(meter(limited(0)), provider.create, ["abcdefg", "hi"])
    assert provider.requests == [["abcdefg", "hi"]]


async def test_a_limit_below_the_margin_refuses_every_text(
    meter: Callable[..., Meter], db_session: AsyncSession
) -> None:
    provider = Provider()
    with pytest.raises(PromptTooLong):
        await embed_plain(meter(limited(1_000)), provider.create, ["a", "b"])
    # "a": 1 + 1,024 = 1,025 > 1,000.
    assert provider.requests == []
    assert (await rows(db_session), await total(db_session)) == ([], None)


# --- embed(): a call that is cancelled ------------------------------------------------------------


async def test_a_cancelled_call_stays_reserved_and_the_meter_forgets_it(
    meter: Callable[..., Meter], db_session: AsyncSession
) -> None:
    m = meter()
    cancelled = anyio.get_cancelled_exc_class()()
    with pytest.raises(type(cancelled)) as raised:
        await embed_plain(m, Provider(cancelled).create, ["a"])
    assert raised.value is cancelled
    # Whether it was billed is unknown: the row stays reserved (counted at its reservation, 21:
    # 1,025 x 0.02 = 20.5 -> 21) until expire_stale closes it.
    [row] = await rows(db_session)
    assert (row.status, row.reserved_micro_usd, row.cost_micro_usd) == ("reserved", 21, None)
    assert await total(db_session) == 21
    assert m._started == {}  # the Meter keeps nothing for it (private on purpose)


async def test_abandoning_a_call_the_meter_does_not_know_changes_nothing(
    meter: Callable[..., Meter],
) -> None:
    m = meter()
    m.abandon(uuid.uuid4())
    assert m._started == {}


# --- embed(): redaction, one way (#78 part 3) -----------------------------------------------------
#
# The values below are worked out by hand. A placeholder is [KIND_n]: "[EMAIL_1]" is 9 bytes, the
# email a@example.com is 13, and "[NAME_1]" is 8 bytes. "one" charges 1 micro-USD per token, so a
# reservation is the request's bytes plus the 1,024-token margin, with no rounding.

REDACTOR = Redactor(DETECTORS)
MAIL = "a@example.com"  # 13 bytes; redacted "[EMAIL_1]" is 9
ONE = Price(Rates(1_000_000, 0, 0, 0, 0), checked="2026-11-06", source=SOURCE)


class ExplodesOn:
    kind = "EMAIL"

    def __init__(self, word: str) -> None:
        self._word = word

    def find(self, text: str) -> Iterable[Span]:
        if self._word in text:
            raise ValueError(f"cannot read {text}")
        return []


async def test_the_provider_gets_the_redacted_texts_and_the_vectors_keep_their_order(
    meter: Callable[..., Meter],
) -> None:
    provider = Provider()
    vectors = await embeddings.embed(
        meter(), provider.create, [MAIL, "hi", f"{MAIL} and b@example.com"], redactor=REDACTOR
    )
    # Each text is numbered on its own: the second email of the third text is [EMAIL_2].
    assert provider.requests == [["[EMAIL_1]", "hi", "[EMAIL_1] and [EMAIL_2]"]]
    assert vectors == [[9.0], [2.0], [23.0]]  # the provider's vector is the length of what it got


async def test_the_size_check_and_the_reservation_use_the_redacted_bytes(
    meter: Callable[..., Meter], db_session: AsyncSession
) -> None:
    provider = Provider()
    # The limit is 1,033 tokens. The original is 13 + 1,024 = 1,037 (over it); the redacted text is
    # 9 + 1,024 = 1,033 (exactly at it), so it is sent, and reserved at 1,033 x 1 micro-USD.
    await embeddings.embed(meter(limited_one(1_033)), provider.create, [MAIL], redactor=REDACTOR)
    [row] = await rows(db_session)
    assert (provider.requests, row.reserved_micro_usd) == ([["[EMAIL_1]"]], 1_033)


def limited_one(tokens: int) -> Price:
    return Price(ONE.rates, checked="2026-11-06", source=SOURCE, max_prompt_tokens=tokens)


async def test_a_placeholder_longer_than_its_original_can_push_a_text_past_the_limit(
    meter: Callable[..., Meter], db_session: AsyncSession
) -> None:
    provider = Provider()
    # The limit 1,030 leaves 6 bytes. "Al" is 2 bytes, but its placeholder "[NAME_1]" is 8: 8 +
    # 1,024 = 1,032 > 1,030, so a 422, with nothing sent and no row.
    with patch_redactions() as recorded:
        async with redaction_scope(known={"NAME": ["Al"]}):
            with pytest.raises(PromptTooLong) as raised:
                await embeddings.embed(
                    meter(limited(1_030)), provider.create, ["Al"], redactor=REDACTOR
                )
    assert raised.value.status == 422
    assert provider.requests == []
    assert (await rows(db_session), await total(db_session)) == ([], None)
    assert recorded == [{"NAME": 1}]  # counted once, before the size check


async def test_batches_are_cut_on_the_redacted_texts(meter: Callable[..., Meter]) -> None:
    provider = Provider()
    # max_bytes 20: the originals (13 + 13 = 26) would be one per batch; the redacted 9 + 9 = 18
    # fit two to a batch, and the third text starts the next.
    await embeddings.embed(meter(), provider.create, [MAIL] * 3, redactor=REDACTOR, max_bytes=20)
    assert provider.requests == [["[EMAIL_1]", "[EMAIL_1]"], ["[EMAIL_1]"]]


async def test_each_batch_row_counts_only_the_values_replaced_in_its_own_texts(
    meter: Callable[..., Meter], db_session: AsyncSession
) -> None:
    provider = Provider()
    texts = [f"{MAIL} 07700 900123", "plain", "b@example.com"]  # 2, 0 and 1 values
    vectors = await embeddings.embed(
        meter(), provider.create, texts, redactor=REDACTOR, max_inputs=2
    )
    # Two batches: the first holds texts 1 and 2 (2 + 0 = 2 values), the second text 3 (1 value).
    assert provider.requests == [["[EMAIL_1] [PHONE_1]", "plain"], ["[EMAIL_1]"]]
    assert vectors == [[19.0], [5.0], [9.0]]
    assert sorted(row.redacted_count or 0 for row in await rows(db_session)) == [1, 2]


async def test_redaction_is_counted_once_per_call_however_many_batches(
    meter: Callable[..., Meter],
) -> None:
    with patch_redactions() as recorded:
        await embeddings.embed(
            meter(), Provider().create, [MAIL, "x", MAIL], redactor=REDACTOR, max_inputs=1
        )
    assert recorded == [{"EMAIL": 2}]


async def test_a_text_that_cannot_be_redacted_blocks_every_batch_and_writes_no_row(
    meter: Callable[..., Meter], db_session: AsyncSession
) -> None:
    provider = Provider()
    with pytest.raises(RedactionFailed):
        await embeddings.embed(
            meter(),
            provider.create,
            ["first", "second", "boom"],
            redactor=Redactor([ExplodesOn("boom")]),
            max_inputs=1,
        )
    # The third text fails, and the first two (batches of their own) were never sent either.
    assert provider.requests == []
    assert (await rows(db_session), await total(db_session)) == ([], None)


async def test_with_redaction_off_the_texts_are_untouched_and_the_row_count_is_null(
    meter: Callable[..., Meter], db_session: AsyncSession
) -> None:
    provider = Provider()
    with patch_redactions() as recorded:
        await embeddings.embed(meter(), provider.create, [MAIL], redactor=None)
    assert provider.requests == [[MAIL]]
    [row] = await rows(db_session)
    assert row.redacted_count is None
    assert recorded == []


async def test_with_redaction_on_and_nothing_found_the_row_count_is_zero(
    meter: Callable[..., Meter], db_session: AsyncSession
) -> None:
    provider = Provider()
    await embeddings.embed(meter(), provider.create, ["a freehold"], redactor=REDACTOR)
    assert provider.requests == [["a freehold"]]
    [row] = await rows(db_session)
    assert row.redacted_count == 0  # redaction ran and replaced nothing: zero, not NULL


async def test_the_callers_tags_are_kept_and_only_the_count_is_added(
    meter: Callable[..., Meter], db_session: AsyncSession
) -> None:
    tags = CallTags(feature="ingest", user_ref="u_1")
    await embeddings.embed(meter(), Provider().create, [MAIL], redactor=REDACTOR, tags=tags)
    [row] = await rows(db_session)
    assert (row.feature, row.user_ref, row.redacted_count) == ("ingest", "u_1", 1)
    assert tags.redacted_count is None  # the caller's tags are not changed


async def test_no_texts_with_redaction_on_send_nothing_and_write_no_row(
    meter: Callable[..., Meter], db_session: AsyncSession
) -> None:
    provider = Provider()
    assert await embeddings.embed(meter(), provider.create, [], redactor=REDACTOR) == []
    assert provider.requests == []
    assert await rows(db_session) == []


async def test_the_redactor_is_a_required_keyword(meter: Callable[..., Meter]) -> None:
    with pytest.raises(TypeError, match="redactor"):
        await embeddings.embed(meter(), Provider().create, ["a"])  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        await embeddings.embed(meter(), Provider().create, ["a"], REDACTOR)  # type: ignore[call-arg]


class patch_redactions:
    """Records what core.metrics.record_redactions is given."""

    def __enter__(self) -> list[dict[str, int]]:
        self._patch = pytest.MonkeyPatch()
        self.recorded: list[dict[str, int]] = []
        self._patch.setattr(
            metrics, "record_redactions", lambda counts: self.recorded.append(dict(counts))
        )
        return self.recorded

    def __exit__(self, *exc: object) -> None:
        self._patch.undo()


# --- OpenAI's call --------------------------------------------------------------------------------


def request() -> httpx2.Request:
    return httpx2.Request("POST", "http://x")


def status(code: int) -> httpx2.Response:
    return httpx2.Response(code, request=request())


async def test_openai_vectors_come_back_in_input_order_with_the_providers_usage() -> None:
    items = [SimpleNamespace(index=i, embedding=[float(i)]) for i in range(3)]
    seen: list[dict[str, Any]] = []

    async def create(**kwargs: Any) -> Any:
        seen.append(kwargs)
        return SimpleNamespace(data=list(reversed(items)), usage=SimpleNamespace(prompt_tokens=21))

    client = SimpleNamespace(embeddings=SimpleNamespace(create=create))
    vectors, usage = await openai_embeddings.create(client, "pin-embeddings", ["a", "b", "c"])
    assert vectors == [[0.0], [1.0], [2.0]]
    assert usage == Usage(input_tokens=21, output_tokens=0)
    assert seen == [{"model": "pin-embeddings", "input": ["a", "b", "c"]}]


def openai_errors() -> list[tuple[Exception, FailureKind]]:
    return [
        (
            openai.AuthenticationError("secret prompt", response=status(401), body=None),
            K.AUTHENTICATION,
        ),
        (
            openai.PermissionDeniedError("secret prompt", response=status(403), body=None),
            K.PERMISSION_DENIED,
        ),
        (openai.NotFoundError("secret prompt", response=status(404), body=None), K.MODEL_NOT_FOUND),
        (openai.RateLimitError("secret prompt", response=status(429), body=None), K.RATE_LIMITED),
        (
            openai.BadRequestError("secret prompt", response=status(400), body=None),
            K.INVALID_REQUEST,
        ),
        (openai.ConflictError("secret prompt", response=status(409), body=None), K.PROVIDER_FAILED),
        (
            openai.UnprocessableEntityError("secret prompt", response=status(422), body=None),
            K.PROVIDER_FAILED,
        ),
        (
            openai.InternalServerError("secret prompt", response=status(500), body=None),
            K.PROVIDER_FAILED,
        ),
        (openai.APITimeoutError(request=request()), K.TIMED_OUT),
        (openai.APIConnectionError(message="secret prompt", request=request()), K.UNREACHABLE),
        (openai.APIError("secret prompt", request=request(), body=None), K.PROVIDER_FAILED),
    ]


@pytest.mark.parametrize("case", range(11))
async def test_an_openai_error_becomes_a_provider_error_with_its_kind_and_type_name(
    case: int,
) -> None:
    source, kind = openai_errors()[case]

    async def create(**kwargs: Any) -> Any:
        raise source

    client = SimpleNamespace(embeddings=SimpleNamespace(create=create))
    with pytest.raises(ProviderError) as raised:
        await openai_embeddings.create(client, "pin-embeddings", ["a"])
    error = raised.value
    assert (error.kind, error.source) == (kind, type(source).__name__)
    assert error.__cause__ is source
    assert "secret prompt" not in str(error)  # a provider's message can echo the input
    assert error.retry_after is None


async def test_an_error_that_is_not_openais_passes_through_as_itself() -> None:
    other = RuntimeError("not openai")

    async def create(**kwargs: Any) -> Any:
        raise other

    client = SimpleNamespace(embeddings=SimpleNamespace(create=create))
    with pytest.raises(RuntimeError) as raised:
        await openai_embeddings.create(client, "pin-embeddings", ["a"])
    assert raised.value is other
    assert raised.value.__cause__ is None


def test_the_openai_client_has_the_gateways_timeout_and_retries() -> None:
    client = openai_embeddings.openai_client(SecretStr("sk-x"), timeout=7.5, max_retries=3)
    assert (client.timeout, client.max_retries, client.api_key) == (7.5, 3, "sk-x")


def test_kinds_are_listed_most_specific_first() -> None:
    # APITimeoutError is an APIConnectionError, and each is an APIError: the specific one first.
    assert openai_embeddings.kind_of(openai.APITimeoutError(request=request())) is K.TIMED_OUT
    assert openai_embeddings.kind_of(openai.APIConnectionError(request=request())) is K.UNREACHABLE
    assert openai_embeddings.kind_of(RuntimeError()) is None


# --- the core stays framework-free ----------------------------------------------------------------


def loaded_by(module: str) -> str:
    # A fresh interpreter: the test process has LangChain loaded already.
    code = (
        f"import sys, {module}\n"
        f"print(sorted(set(m.split('.')[0] for m in sys.modules) & set({sorted(FRAMEWORKS)!r})))"
    )
    result = subprocess.run(  # noqa: S603 - this interpreter, running code written above
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def test_the_metered_batching_loads_no_framework_and_no_sdk() -> None:
    assert loaded_by("app.llm.core.embeddings") == "[]"


def test_the_openai_module_loads_openai_and_nothing_else_of_the_listed_frameworks() -> None:
    assert loaded_by("app.llm.core.openai_embeddings") == "['openai']"
