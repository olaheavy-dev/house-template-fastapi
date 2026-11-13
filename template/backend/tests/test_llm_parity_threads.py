"""A stored conversation, the same under every framework (#73, decision 0004): an image in a thread
is a pointer to a stored file, the next turn still sends it to the model, the conversation
survives a restart (it comes back from Postgres), and the budget reserves the image again on
every turn that sends it. The database is real; the model is scripted by the harness, so each
test runs once per framework the project chose and imports none.

The thread is keyed by the user and the conversation id, derived on the server (never a
caller-supplied thread id). A restart is a new gateway on the same database.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from datetime import date
from typing import Any

import pytest
from pydantic import SecretStr
from sqlalchemy import select

from app.db import create_engine, create_sessionmaker
from app.llm import prices
from app.llm.core.bounds import IMAGE_TOKENS
from app.llm.core.budget import LlmCall
from app.settings import Settings
from tests.harness import Harness, Reply, user

pytestmark = pytest.mark.anyio

DAY = date(2026, 1, 11)
MODEL = "anthropic:claude-sonnet-5-5"
# A pointer to a file in the project's storage, never the image's bytes (#73).
IMAGE = {"type": "image", "url": "https://files.example/rooms/kitchen-1.png"}
USER, CONVERSATION = "u_threads", "c_kitchen"


@pytest.fixture
async def make(
    framework: Harness, settings: Settings, database_url: str
) -> AsyncIterator[Callable[..., Any]]:
    """Gateways on the real (committed) test database, as the app runs: each one a fresh start."""
    with_db = settings.model_copy(
        update={"llm_model": MODEL, "database_url": SecretStr(database_url)}
    )
    engine = create_engine(with_db, pool_size=2, max_overflow=0)
    sessions = create_sessionmaker(engine)
    closers: list[Any] = []
    built: list[Any] = []

    async def build(*script: Reply) -> Any:
        gateway = framework.gateway(with_db, lambda: sessions)
        meter = framework.meter(gateway)
        meter.cap, meter.today = 10_000_000, lambda: DAY
        framework.install_scripted(gateway, list(script))
        closers.append(await framework.open_threads(gateway, with_db))
        built.append(gateway)
        return gateway

    yield build
    if built:
        await framework.forget_threads(built[-1], user_ref=USER, conversation_id=CONVERSATION)
    for close in closers:
        await close()
    async with sessions() as db, db.begin():
        rows = await db.scalars(select(LlmCall).where(LlmCall.day == DAY))
        for row in rows:
            await db.delete(row)
    await engine.dispose()


async def test_an_image_in_a_thread_is_sent_again_after_a_restart_and_reserved_each_turn(
    framework: Harness, make: Callable[..., Any], settings: Settings, database_url: str
) -> None:
    first = await make(Reply("A kitchen."))
    answer = await framework.thread_turn(
        first,
        user_ref=USER,
        conversation_id=CONVERSATION,
        messages=[user([IMAGE, {"type": "text", "text": "Which room is this?"}])],
    )
    assert answer == "A kitchen."

    restarted = await make(Reply("Yes, a large one."))
    await framework.thread_turn(
        restarted,
        user_ref=USER,
        conversation_id=CONVERSATION,
        messages=[user("Is it a large room?")],
    )
    sent = framework.last_request_messages(restarted) or []
    # The second turn sends the stored history: the first question with its image pointer, the
    # first answer, then the new question.
    assert IMAGE["url"] in repr(sent)
    assert "Which room is this?" in repr(sent)
    assert "A kitchen." in repr(sent)
    assert "Is it a large room?" in repr(sent)

    # Both turns sent the image, so both reserved at least its bound (IMAGE_TOKENS at the model's
    # highest input rate).
    rates = prices.PRICES[MODEL].highest
    image_floor = -(-IMAGE_TOKENS * max(rates.input, rates.cache_write_long) // 1_000_000)
    engine = create_engine(settings.model_copy(update={"database_url": SecretStr(database_url)}))
    try:
        async with create_sessionmaker(engine)() as db:
            reserved = list(
                await db.scalars(select(LlmCall.reserved_micro_usd).where(LlmCall.day == DAY))
            )
    finally:
        await engine.dispose()
    assert len(reserved) == 2
    assert all(amount >= image_floor for amount in reserved)


async def test_another_users_conversation_with_the_same_id_starts_empty(
    framework: Harness, make: Callable[..., Any]
) -> None:
    gateway = await make(Reply("Noted."), Reply("Hello."))
    await framework.thread_turn(
        gateway,
        user_ref=USER,
        conversation_id=CONVERSATION,
        messages=[user("My room is the kitchen.")],
    )
    await framework.thread_turn(
        gateway,
        user_ref="u_someone_else",
        conversation_id=CONVERSATION,
        messages=[user("Hi")],
    )
    sent = repr(framework.last_request_messages(gateway) or [])
    assert "kitchen" not in sent
    await framework.forget_threads(gateway, user_ref="u_someone_else", conversation_id=CONVERSATION)
