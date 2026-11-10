"""Builds the app's chat model: the one place a model provider is configured.

The model comes from LLM_MODEL with a pinned id, its price must be in app/llm/prices.py, and it is
built with init_chat_model with a token limit, a timeout, the SDK's retries as the only retry
layer, usage reporting on streams, the daily budget and, when on, LangSmith tracing. Code asks for
it through GatewayDep (`from app.llm import GatewayDep`) and always awaits it:

    @router.post("/summaries")
    async def summarise(body: SummaryIn, gateway: GatewayDep) -> SummaryOut:
        prompt = load_prompt("summarise")
        answer = await gateway.model().ainvoke(
            [SystemMessage(prompt.text), HumanMessage(body.text)], config=prompt.config()
        )
        return SummaryOut(text=answer.text)

Never build a model elsewhere or hand a "provider:model" string to LangChain: that skips the
budget. lint-imports allows `langchain` and the provider SDKs only under app/llm.

Use the model from one event loop per process, as the app does. langchain-openai and
langchain-anthropic cache one HTTP client per process whose connections belong to the loop that
opened them, so a worker that calls asyncio.run() per job fails on its second job with "Event loop
is closed": run one loop and process jobs inside it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Annotated, Any, cast

import anyio
import structlog
from fastapi import Depends, Request
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langchain_core.tracers.langchain import LangChainTracer, wait_for_all_tracers
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph
from langsmith import Client
from langsmith import configure as configure_langsmith
from pydantic import SecretStr
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.auth import User
from app.llm import prices
from app.llm.core import registry
from app.llm.core.openai_embeddings import openai_client
from app.llm.core.prices import MICRO_USD
from app.llm.core.prompts import Prompt
from app.llm.core.redaction import Redactor, choose_detectors
from app.llm.core.steps import StepLimit
from app.llm.langchain.callbacks import BudgetGuard
from app.llm.langchain.embeddings import GatewayEmbeddings
from app.llm.langchain.prompts import prompt_config
from app.llm.langchain.redaction import RedactingChatModel, masking_anonymizer
from app.llm.langchain.steps import StepLimitMiddleware, recursion_limit
from app.llm.redaction_rules import DETECTORS
from app.problems import Problem
from app.settings import Settings

__all__ = ["Gateway", "GatewayDep", "get_gateway"]

log = structlog.get_logger()


class Gateway:
    def __init__(
        self, settings: Settings, sessionmaker: Callable[[], async_sessionmaker[AsyncSession]]
    ) -> None:
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
        # Set in the lifespan once the checkpoint pool is open (app/llm/langchain/checkpoints.py).
        self.checkpointer: BaseCheckpointSaver[Any] | None = None
        self.guard = BudgetGuard(
            model=llm_model,
            price=price,
            cap_micro_usd=int(settings.llm_daily_budget_usd * MICRO_USD),
            user_cap_micro_usd=(
                int(settings.llm_user_daily_budget_usd * MICRO_USD)
                if settings.llm_user_daily_budget_usd
                else None
            ),
            max_tokens=settings.llm_max_tokens,
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
        callbacks: list[BaseCallbackHandler] = [self.guard]
        # One LangSmith client for the process: it masks what traces carry whenever redaction is
        # on, so it is also the client of tracing that starts from the environment or from a graph,
        # not only of this gateway's tracer. (It is also what shutdown flushes.)
        client = _langsmith_client(settings, self.redactor)
        if client is not None and self.redactor is not None:
            configure_langsmith(client=client)
        self.tracer = _tracer(settings, client)
        if self.tracer:
            callbacks.append(self.tracer)

        embedding_price = registry.embedding_price(embedding_model, prices.EMBEDDING_PRICES)
        # Without OpenAI's key there are no embeddings, and asking for them gets a 503.
        self._embeddings: GatewayEmbeddings | None = None
        if settings.openai_api_key is not None:
            self._embeddings = GatewayEmbeddings(
                model=embedding_model,
                price=embedding_price,
                client=openai_client(
                    settings.openai_api_key,
                    timeout=settings.llm_timeout_seconds,
                    max_retries=settings.llm_max_retries,
                ),
                cap_micro_usd=self.guard.cap,
                user_cap_micro_usd=self.guard.user_cap,
                sessionmaker=sessionmaker,
            )

        api_key: SecretStr | None = getattr(settings, registry.key_setting(llm_model))
        # Without a key there is no model, and calls get a 503 instead of reaching the provider.
        self._model: BaseChatModel | None = None
        self._wrapped: RedactingChatModel | None = None
        if api_key is not None:
            self._model = init_chat_model(
                llm_model,
                api_key=api_key,
                max_tokens=settings.llm_max_tokens,
                timeout=settings.llm_timeout_seconds,
                max_retries=settings.llm_max_retries,
                stream_usage=True,
                callbacks=callbacks,
            )

    def model(self) -> BaseChatModel:
        """The chat model. With redaction on (the default) it is a RedactingChatModel around the
        provider's, so `isinstance(model, ChatOpenAI)` is False (app/llm/langchain/redaction.py)."""
        if self._model is None:
            raise Problem(503, detail="The language model isn't configured.")
        if self.redactor is None:
            return self._model
        # Wrapped around whatever _model is now (a test may swap it), and kept so every caller
        # gets the same object.
        if self._wrapped is None or self._wrapped.inner is not self._model:
            self._wrapped = RedactingChatModel(self._model, self.redactor)
        return self._wrapped

    def embeddings(self) -> GatewayEmbeddings:
        """Embeddings under the same daily budget; always await them:

        vectors = await gateway.embeddings().aembed_documents(texts, feature="ingest")
        query = await gateway.embeddings().aembed_query(question, user_ref=user.id)
        """
        if self._embeddings is None:
            raise Problem(503, detail="The embedding model isn't configured.")
        return self._embeddings

    def agent(
        self,
        tools: Sequence[BaseTool | Callable[..., Any]] = (),
        *,
        system_prompt: str | None = None,
        **options: Any,
    ) -> CompiledStateGraph[Any, Any, Any, Any]:
        """A tool-calling agent (LangChain's create_agent) on the gateway's model, its threads
        saved in Postgres. Pass a thread id with each call:

            agent = gateway.agent([search_courses], system_prompt=prompt.text)
            state = await agent.ainvoke(
                {"messages": [HumanMessage(question)]},
                config={**prompt.config(), "configurable": {"thread_id": thread_id}},
            )

        A run makes at most AGENT_MAX_MODEL_CALLS model calls (default 13). The last one is made
        without tools, so the model writes a final answer, and the result has `step_limit_reached`
        set (app.llm.langchain.steps.step_limit_reached(state)): tell users the answer was cut
        short. A run cut short with a response_format may have no structured_response. The count
        restarts on every run, also on a saved thread. Middleware you pass runs outside this limit
        and cannot offer tools again on the last call; one whose wrap_model_call retries or falls
        back makes extra provider calls that count as one call here, each has its own llm_call
        row, and step_limit_reached may be logged more than once.

        For anything beyond a plain tool loop, build a LangGraph StateGraph whose nodes call
        gateway.model(), and compile it with checkpointer=gateway.checkpointer.
        """
        # Ours is last, so it runs closest to the model: a caller's middleware cannot offer tools
        # again on the last call.
        middleware = list(options.pop("middleware", ()))
        agent = create_agent(
            self.model(),
            list(tools),
            system_prompt=system_prompt,
            middleware=[*middleware, StepLimitMiddleware(self.step_limit)],
            checkpointer=self.checkpointer,
            **options,
        )
        # LangGraph's own limit is only a backstop (10,007 by default): the run ends by itself at
        # the model-call limit, so this stops a graph that loops without calling the model.
        return agent.with_config(recursion_limit=recursion_limit(self.step_limit, middleware))

    def call_config(
        self, user: User | None, feature: str, prompt: Prompt | None = None, **config: Any
    ) -> RunnableConfig:
        """The config for a model or agent call: what it is for (feature), whom for (counted
        against their daily cap), and the prompt's version.

            answer = await gateway.model().ainvoke(
                messages, config=gateway.call_config(user, "answer", prompt)
            )
        """
        merged: dict[str, Any] = {**(prompt_config(prompt) if prompt else {}), **config}
        metadata = {**(prompt.metadata() if prompt else {}), "feature": feature}
        if user is not None:
            metadata["user_ref"] = user.id
        merged["metadata"] = {**metadata, **config.get("metadata", {})}
        return cast(RunnableConfig, merged)

    async def aclose(self) -> None:
        if self.tracer or self.redactor is not None:
            # Traces are sent from a background thread; send what's left before the process ends.
            # (It flushes the process-wide client, which is this gateway's own client when
            # redaction is on.)
            await anyio.to_thread.run_sync(wait_for_all_tracers)


def _tracing_on(settings: Settings) -> bool:
    # Attaching a tracer also stops LangChain adding its own from LANGSMITH_* variables, so
    # tracing follows these settings alone.
    return settings.langsmith_tracing and settings.app_env != "test"


def _langsmith_client(settings: Settings, redactor: Redactor | None) -> Client | None:
    """The LangSmith client: with the masking anonymizer when redaction is on (also when tracing is
    off, for tracing that starts from the environment), and with `hide_inputs`/`hide_outputs`,
    which still win over masking. None when there is nothing to trace or mask. Batching, and its
    background thread, only when this gateway traces."""
    tracing = _tracing_on(settings)
    if not tracing and redactor is None:
        return None
    api_key = settings.langsmith_api_key
    return Client(
        api_url=settings.langsmith_endpoint,
        api_key=api_key.get_secret_value() if api_key else None,
        hide_inputs=settings.langsmith_hide_inputs,
        hide_outputs=settings.langsmith_hide_outputs,
        tracing_sampling_rate=settings.langsmith_tracing_sampling_rate,
        anonymizer=masking_anonymizer(redactor) if redactor is not None else None,
        auto_batch_tracing=tracing,
    )


def _tracer(settings: Settings, client: Client | None) -> LangChainTracer | None:
    if not _tracing_on(settings) or client is None:
        return None
    return LangChainTracer(project_name=settings.langsmith_project, client=client)


def get_gateway(request: Request) -> Gateway:
    gateway: Gateway = request.app.state.gateway
    return gateway


GatewayDep = Annotated[Gateway, Depends(get_gateway)]
