"""The MAF gateway's own rules (#35): the middleware chain it insists on, what it forces on every
call (store, the output limit), the tags that reach the ledger, and its refusals to start."""

from __future__ import annotations

from datetime import date
from typing import Any

import pytest
from agent_framework import Message
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.auth import User
from app.llm.core.budget import LlmCall
from app.llm.core.prompts import Prompt
from app.llm.maf.gateway import Gateway
from app.llm.maf.middleware import current_tags
from app.llm.maf.redaction import RedactionUnavailable
from app.problems import Problem
from app.settings import Settings
from tests.harness import Reply
from tests.harness.maf import HARNESS, ScriptedClient

pytestmark = pytest.mark.anyio

DAY = date(2026, 1, 6)


@pytest.fixture
def make(settings: Settings, db_session: AsyncSession) -> Any:
    sessions = async_sessionmaker(
        bind=db_session.bind, join_transaction_mode="create_savepoint", expire_on_commit=False
    )

    def build(model: str = "anthropic:claude-sonnet-5-5", **options: Any) -> Gateway:
        gateway = Gateway(
            settings.model_copy(update={"llm_model": model, "llm_max_tokens": 500, **options}),
            lambda: sessions,
        )
        gateway.meter.cap, gateway.meter.today = 1_000_000, lambda: DAY
        HARNESS.install_scripted(gateway, [Reply("ok")] * 5)
        return gateway

    return build


async def ask(gateway: Gateway, options: dict[str, Any] | None = None) -> None:
    await gateway.client().get_response([Message("user", ["hi"])], options=options)  # type: ignore[arg-type]


def sent(gateway: Gateway) -> list[Any]:
    client = gateway._client
    assert isinstance(client, ScriptedClient)
    return client.seen_options


# --- the chain ------------------------------------------------------------------------------


@pytest.mark.parametrize("which", ["no redaction", "wrong order", "empty", "someone else's"])
async def test_a_client_whose_chain_is_not_the_gateways_is_refused(make: Any, which: str) -> None:
    gateway = make()
    chains = {
        "no redaction": [gateway.budget],
        "wrong order": [gateway.budget, gateway.redaction],
        "empty": [],
        "someone else's": [make().redaction, make().budget],
    }
    gateway._client = ScriptedClient(script=[], middleware=chains[which])
    with pytest.raises(RedactionUnavailable):
        gateway.client()


async def test_a_client_with_no_middleware_list_at_all_is_refused(make: Any) -> None:
    gateway = make()
    gateway._client = object()  # type: ignore[assignment]
    with pytest.raises(RedactionUnavailable):
        gateway.client()


async def test_the_gateways_own_client_passes_the_check(make: Any) -> None:
    gateway = make()
    assert gateway.client() is gateway._client


# --- what every call carries ---------------------------------------------------------------


async def test_openai_is_told_store_false_on_every_call_and_anthropic_gets_no_store_option(
    make: Any,
) -> None:
    openai_gateway = make("openai:gpt-6-luna")
    await ask(openai_gateway)
    assert sent(openai_gateway)[0]["store"] is False
    anthropic_gateway = make()
    await ask(anthropic_gateway, {"store": False})
    assert "store" not in sent(anthropic_gateway)[0]


@pytest.mark.parametrize("given", [None, 100, 5_000])
async def test_the_output_limit_is_the_gateways_unless_the_call_sets_its_own(
    make: Any, given: int | None
) -> None:
    gateway = make()
    await ask(gateway, {"max_tokens": given} if given else None)
    assert sent(gateway)[0]["max_tokens"] == (given or 500)


async def test_the_callers_options_are_not_changed(make: Any) -> None:
    gateway = make()
    options: dict[str, Any] = {"temperature": 0.2}
    await ask(gateway, options)
    assert options == {"temperature": 0.2}


# --- tags -----------------------------------------------------------------------------------


async def test_call_options_tags_the_ledger_row_and_only_inside_the_block(
    make: Any, db_session: AsyncSession
) -> None:
    gateway = make()
    prompt = Prompt(name="sums", text="Add numbers.")
    with gateway.call_options(User(id="u_1"), "answer", prompt):
        assert current_tags().feature == "answer"
        await ask(gateway)
    assert current_tags().feature is None  # the block restored what was there
    await ask(gateway)
    first, second = list(
        await db_session.scalars(
            select(LlmCall).where(LlmCall.day == DAY).order_by(LlmCall.created_at)
        )
    )
    assert (first.prompt, first.prompt_sha, first.feature, first.user_ref) == (
        "sums",
        prompt.sha,
        "answer",
        "u_1",
    )
    assert (second.prompt, second.feature, second.user_ref) == (None, None, None)


# --- starting -------------------------------------------------------------------------------


async def test_without_a_key_the_client_is_a_503_and_nothing_is_built(
    settings: Settings, db_session: AsyncSession
) -> None:
    sessions = async_sessionmaker(bind=db_session.bind, expire_on_commit=False)
    gateway = Gateway(
        settings.model_copy(update={"llm_model": "openai:gpt-6-luna", "openai_api_key": None}),
        lambda: sessions,
    )
    with pytest.raises(Problem) as refused:
        gateway.client()
    assert refused.value.status == 503


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"llm_model": None}, "LLM_MODEL is not set"),
        ({"llm_model": "openai:no-such-model"}, "has no price"),
        ({"embedding_model": None}, "EMBEDDING_MODEL is not set"),
        ({"llm_redact_kinds": "EMAIL,NOPE"}, "NOPE"),
    ],
)
async def test_the_gateway_refuses_to_start_on_a_bad_setting(
    settings: Settings, db_session: AsyncSession, update: dict[str, Any], message: str
) -> None:
    sessions = async_sessionmaker(bind=db_session.bind, expire_on_commit=False)
    with pytest.raises(Exception, match=message):
        Gateway(
            settings.model_copy(update={"llm_model": "openai:gpt-6-luna", **update}),
            lambda: sessions,
        )


def test_a_validation_error_never_echoes_the_value_given() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError) as raised:
        Settings(_env_file=None, llm_max_tokens="sk-secret-value")  # type: ignore[arg-type]
    assert "sk-secret-value" not in str(raised.value)
