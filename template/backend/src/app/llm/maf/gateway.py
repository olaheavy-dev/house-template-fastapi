"""Builds the app's MAF chat client: the one place a model provider is configured.

The model comes from LLM_MODEL with a pinned id and its price must be in app/llm/prices.py. The
provider's own SDK client is built here, with the timeout and the retries as the only retry layer
(MAF's clients have neither parameter; its Anthropic client never retries), handed to MAF's client,
and closed by the gateway at shutdown (MAF leaves a client it was given open). The chat client is
MAF's own with its real layers (function invocation, chat middleware, telemetry) and the gateway's
two chat middleware in front: personal-data redaction, then the daily budget. Code asks for it
through GatewayDep (`from app.llm import GatewayDep`) and always awaits it:

    @router.post("/summaries")
    async def summarise(body: SummaryIn, user: CurrentUser, gateway: GatewayDep) -> SummaryOut:
        prompt = load_prompt("summarise")
        with gateway.call_options(user, "summarise", prompt):
            response = await gateway.client().get_response(
                [Message("system", [prompt.text]), Message("user", [body.text])]
            )
        return SummaryOut(text=response.text)

Never build a MAF client elsewhere: that skips the budget and the redaction. lint-imports allows
`agent_framework` only under app/llm/maf and the provider SDKs only under app/llm. `client()`
refuses a client whose middleware chain is not exactly the gateway's two, in that order.

Use the client from one event loop per process, as the app does: the SDK clients keep HTTP
connections that belong to the loop that opened them.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import Annotated, Any

import anthropic
import structlog
from agent_framework import (
    Agent,
    FunctionInvocationConfiguration,
    InMemoryHistoryProvider,
    SupportsChatGetResponse,
)

# Not exported by agent-framework-core 1.25.0; it is the function MAF itself sorts middleware with.
# The pins move together by hand (decision 0002), and a test calls it, so a rename fails loudly.
from agent_framework._middleware import categorize_middleware
from agent_framework.anthropic import AnthropicClient
from agent_framework.openai import OpenAIChatClient
from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.auth import User
from app.llm import prices
from app.llm.core import registry
from app.llm.core.budget import CallTags, Meter
from app.llm.core.openai_embeddings import openai_client
from app.llm.core.prices import MICRO_USD
from app.llm.core.prompts import Prompt
from app.llm.core.redaction import Redactor, choose_detectors
from app.llm.core.steps import StepLimit
from app.llm.maf import tracing
from app.llm.maf.checkpoints import PostgresCheckpointStorage
from app.llm.maf.embeddings import GatewayEmbeddings
from app.llm.maf.middleware import BudgetMiddleware, tagged
from app.llm.maf.redaction import RedactionMiddleware, RedactionUnavailable
from app.llm.maf.sessions import PostgresSessionStore
from app.llm.maf.steps import RUN_OPTION, StepRun, flagged
from app.llm.redaction_rules import DETECTORS
from app.problems import Problem
from app.settings import Settings

__all__ = ["Gateway", "GatewayDep", "check_chain", "get_gateway"]

log = structlog.get_logger()


class NoExtraChatMiddleware:
    """Mixed into the gateway's clients. MAF appends chat middleware that arrives per call
    (`client_kwargs={"middleware": ...}`) or per agent (`Agent(..., middleware=[...])`) AFTER the
    client's own chain, so it would run closer to the provider than redaction and the budget and
    could add personal data or spend unmetered. The gateway's clients refuse it instead (fail
    closed): add chat middleware to the gateway's chain, where it is deliberate. Function
    middleware (around tools) is not affected.

    Each `get_response` is one run of the agent step limit (app/llm/maf/steps.py): it gets a fresh
    StepRun in its options, which the budget middleware counts against, and its result is flagged
    if its last call was the forced final answer."""

    def get_response(
        self, messages: Any, *, stream: bool = False, options: Any = None, **kwargs: Any
    ) -> Any:
        run = StepRun()
        with_run = {**(options or {}), RUN_OPTION: run}
        result = super().get_response(  # type: ignore[misc]
            messages, stream=stream, options=with_run, **kwargs
        )
        return flagged(result, run)

    def _get_chat_middleware_pipeline(self, middleware: Any) -> Any:
        if categorize_middleware(list(middleware or []))["chat"]:
            raise RedactionUnavailable(
                "chat middleware added per call or per agent would run after redaction and the "
                "budget; add it to the gateway's own chain"
            )
        return super()._get_chat_middleware_pipeline(middleware)  # type: ignore[misc]


class GatewayOpenAIChatClient(NoExtraChatMiddleware, OpenAIChatClient):
    """MAF's OpenAI Responses client, refusing per-call and per-agent chat middleware."""


class GatewayAnthropicClient(NoExtraChatMiddleware, AnthropicClient):
    """MAF's Anthropic client, refusing per-call and per-agent chat middleware."""


def check_chain(client: Any, expected: list[Any]) -> None:
    """Refuse a client whose chat middleware is not exactly `expected`, the gateway's own
    redaction then budget middleware, in that order, or one that would accept more (fail closed: a
    client without them would send personal data and spend without a reservation)."""
    if not isinstance(client, NoExtraChatMiddleware):
        raise RedactionUnavailable("the chat client is not one the gateway built")
    chain = getattr(client, "chat_middleware", None)
    if not isinstance(chain, list) or len(chain) != len(expected):
        raise RedactionUnavailable("the chat client's middleware is not the gateway's")
    if any(found is not wanted for found, wanted in zip(chain, expected, strict=True)):
        raise RedactionUnavailable("the chat client's middleware is not the gateway's")


class Gateway:
    def __init__(
        self,
        settings: Settings,
        sessionmaker: Callable[[], async_sessionmaker[AsyncSession]],
        *,
        http_client: Any = None,
    ) -> None:
        """`http_client` is a test seam: an httpx client over a fake transport, given to the chat
        provider's SDK so its real retries and timeouts run with no network. Production passes
        none."""
        llm_model = (settings.llm_model or "").strip()
        if not llm_model:
            raise ValueError(
                "LLM_MODEL is not set. Set it to a '<provider>:<model-id>' whose prices are in "
                "app/llm/prices.py (add them there first, from the provider's pricing page)"
            )
        embedding_model = (settings.embedding_model or "").strip()
        if not embedding_model:
            raise ValueError(
                "EMBEDDING_MODEL is not set. Set it to an '<provider>:<model-id>' whose prices are "
                "in app/llm/prices.py EMBEDDING_PRICES (add them there first, from the provider's "
                "pricing page)"
            )
        price = registry.chat_price(llm_model, prices.PRICES)
        self.name = llm_model
        self.step_limit = StepLimit(settings.agent_max_model_calls)
        # AGENT_MAX_MODEL_CALLS counts model calls, the last made without tools; MAF's
        # `max_iterations` counts the tool rounds before that last call (and must be at least 1:
        # with AGENT_MAX_MODEL_CALLS=1 the budget middleware offers no tools on any call, so MAF
        # returns after the one call). Measured on 1.25.0: model calls = max_iterations + 1.
        self.tool_loop: FunctionInvocationConfiguration = {
            "max_iterations": max(settings.agent_max_model_calls - 1, 1)
        }
        self.meter = Meter(
            model=llm_model,
            price=price,
            cap_micro_usd=int(settings.llm_daily_budget_usd * MICRO_USD),
            user_cap_micro_usd=(
                int(settings.llm_user_daily_budget_usd * MICRO_USD)
                if settings.llm_user_daily_budget_usd
                else None
            ),
            sessionmaker=sessionmaker,
        )
        # Personal-data redaction (#78): None when LLM_REDACTION=off. An unknown kind raises here,
        # so the app does not start.
        self.redactor: Redactor | None = None
        if settings.llm_redaction == "on":
            detectors = choose_detectors(DETECTORS, settings.llm_redact_kinds)
            self.redactor = Redactor(detectors)
            log.info(
                "llm_redaction", state="on", kinds=sorted({detector.kind for detector in detectors})
            )
        else:
            log.info("llm_redaction", state="off", kinds=[])
        # Redaction outside, budget inside: the budget sizes and records the redacted request.
        self.redaction = RedactionMiddleware(self.redactor)
        self.budget = BudgetMiddleware(
            self.meter,
            max_tokens=settings.llm_max_tokens,
            sends_store=llm_model.startswith("openai:"),
            step_limit=self.step_limit,
        )
        self.middleware: list[Any] = [self.redaction, self.budget]
        # MAF's instrumentation, and LangSmith over OpenTelemetry when configured (never in tests).
        self.tracing = tracing.configure(settings)
        # Conversations stored in Postgres, keyed by the current user (app/llm/maf/sessions.py).
        self.sessions = PostgresSessionStore(sessionmaker)
        self._sessionmaker = sessionmaker

        # Without OpenAI's key there are no embeddings, and asking for them gets a 503.
        self._embeddings: GatewayEmbeddings | None = None
        if settings.openai_api_key is not None:
            self._embeddings = GatewayEmbeddings(
                model=embedding_model,
                price=registry.embedding_price(embedding_model, prices.EMBEDDING_PRICES),
                client=openai_client(
                    settings.openai_api_key,
                    timeout=settings.llm_timeout_seconds,
                    max_retries=settings.llm_max_retries,
                ),
                redactor=self.redactor,
                cap_micro_usd=self.meter.cap,
                user_cap_micro_usd=self.meter.user_cap,
                sessionmaker=sessionmaker,
            )

        # Without a key there is no model, and calls get a 503 instead of reaching the provider.
        self._client: SupportsChatGetResponse[Any] | None = None
        self._sdk: Any = None
        api_key = getattr(settings, registry.key_setting(llm_model))
        if api_key is not None:
            provider, model_id = llm_model.split(":", 1)
            if provider == "anthropic":
                self._sdk = anthropic.AsyncAnthropic(
                    api_key=api_key.get_secret_value(),
                    timeout=settings.llm_timeout_seconds,
                    max_retries=settings.llm_max_retries,
                    http_client=http_client,
                )
                self._client = GatewayAnthropicClient(
                    model=model_id,
                    anthropic_client=self._sdk,
                    middleware=self.middleware,
                    function_invocation_configuration=self.tool_loop,
                )
            else:
                self._sdk = openai_client(
                    api_key,
                    timeout=settings.llm_timeout_seconds,
                    max_retries=settings.llm_max_retries,
                    http_client=http_client,
                )
                # OpenAI's Responses API; the middleware sends store=False on every call.
                self._client = GatewayOpenAIChatClient(
                    model=model_id,
                    async_client=self._sdk,
                    middleware=self.middleware,
                    function_invocation_configuration=self.tool_loop,
                )
            check_chain(self._client, self.middleware)

    def client(self) -> SupportsChatGetResponse[Any]:
        """The chat client. Its middleware chain is the gateway's own (redaction, then budget):
        a client without it is refused."""
        if self._client is None:
            raise Problem(503, detail="The language model isn't configured.")
        check_chain(self._client, self.middleware)
        return self._client

    def agent(
        self,
        tools: Sequence[Any] = (),
        *,
        instructions: str | None = None,
        **options: Any,
    ) -> Agent[Any]:
        """A tool-calling agent (MAF's Agent) on the gateway's client, so every model call of its
        tool loop is redacted and budgeted:

            agent = gateway.agent([search_courses], instructions=prompt.text)
            with gateway.call_options(user, "answer", prompt):
                response = await agent.run(question)
            if step_limit_reached(response):  # app.llm.maf.steps
                ...  # the answer was cut short at the model-call limit

        A run makes at most AGENT_MAX_MODEL_CALLS model calls (default 13); the last one is made
        without tools, so the model writes a final answer, and the response is flagged. The count
        restarts on every run. `options` go to MAF's Agent (`name`, `default_options`,
        `context_providers`, agent and function `middleware`); chat middleware is refused (it
        would run after redaction and the budget).

        To keep a conversation, give the agent a stable `name` and load and save its session with
        `gateway.sessions` (app/llm/maf/sessions.py). With no context providers given, the agent
        keeps its history in the session with MAF's InMemoryHistoryProvider, added here
        explicitly (MAF's docs: "add it explicitly when you want deterministic local memory
        behavior")."""
        if not options.get("context_providers"):
            options["context_providers"] = [InMemoryHistoryProvider()]
        return Agent(self.client(), instructions, tools=list(tools), **options)

    def checkpoints(self, user: User) -> PostgresCheckpointStorage:
        """MAF's CheckpointStorage for this user's workflow runs, in Postgres
        (app/llm/maf/checkpoints.py): pass it as `checkpoint_storage` to a WorkflowBuilder, or to
        `workflow.run(checkpoint_id=..., checkpoint_storage=...)` to resume. A user only ever sees
        their own checkpoints."""
        return PostgresCheckpointStorage(self._sessionmaker, user)

    async def delete_user_data(self, user_ref: str) -> tuple[int, int]:
        """Delete everything the gateway stored for a user: their agent sessions and workflow
        checkpoints (both hold conversation text). Returns how many of each went."""
        sessions = await self.sessions.delete_user(user_ref)
        checkpoints = await PostgresCheckpointStorage.delete_user(self._sessionmaker, user_ref)
        return sessions, checkpoints

    def embeddings(self) -> GatewayEmbeddings:
        """Embeddings under the same daily budget; always await them:

        vectors = await gateway.embeddings().embed_documents(texts, feature="ingest")
        query = await gateway.embeddings().embed_query(question, user_ref=user.id)
        """
        if self._embeddings is None:
            raise Problem(503, detail="The embedding model isn't configured.")
        return self._embeddings

    @contextmanager
    def call_options(
        self, user: User | None, feature: str, prompt: Prompt | None = None
    ) -> Iterator[None]:
        """Tags the calls made inside the block with what they are for (feature), whom for (counted
        against their daily cap) and the prompt's version:

            with gateway.call_options(user, "answer", prompt):
                response = await gateway.client().get_response(messages)

        The tags travel in context variables, so they follow the call into an agent's tool loop.
        Never put personal data in `feature`: it goes to logs and metrics."""
        with tagged(
            CallTags(
                prompt=prompt.name if prompt else None,
                prompt_sha=prompt.sha if prompt else None,
                feature=feature,
                user_ref=user.id if user is not None else None,
            )
        ):
            yield

    async def aclose(self) -> None:
        """Closes the SDK clients (MAF leaves a client it was given open) and flushes the spans
        still to send, within a time limit."""
        self.tracing.shutdown()
        if self._sdk is not None:
            await self._sdk.close()
        if self._embeddings is not None:
            await self._embeddings.aclose()


def get_gateway(request: Request) -> Gateway:
    gateway: Gateway = request.app.state.gateway
    return gateway


GatewayDep = Annotated[Gateway, Depends(get_gateway)]
