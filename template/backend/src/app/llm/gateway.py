"""Builds the app's chat model: the one place a model provider is configured.

The model comes from LLM_MODEL with a pinned id, its price must be in app/llm/prices.py, and it is
built with init_chat_model with a token limit, a timeout, the SDK's retries as the only retry
layer, usage reporting on streams, the daily budget and, when on, LangSmith tracing. Code asks for
it through GatewayDep and always awaits it:

    @router.post("/summaries")
    async def summarise(body: SummaryIn, gateway: GatewayDep) -> SummaryOut:
        prompt = load_prompt("summarise")
        answer = await gateway.model().ainvoke(
            [SystemMessage(prompt.text), HumanMessage(body.text)], config=prompt.config()
        )
        return SummaryOut(text=answer.text)

Never build a model elsewhere or hand a "provider:model" string to LangChain: that skips the
budget. lint-imports allows `langchain` and the provider SDKs only in this package.

Use the model from one event loop per process, as the app does. langchain-openai and
langchain-anthropic cache one HTTP client per process whose connections belong to the loop that
opened them, so a worker that calls asyncio.run() per job fails on its second job with "Event loop
is closed": run one loop and process jobs inside it.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Annotated, Any, cast

import anyio
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
from pydantic import SecretStr
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.auth import User
from app.llm import prices
from app.llm.budget import BudgetGuard
from app.llm.core import registry
from app.llm.core.prices import MICRO_USD
from app.llm.embeddings import GatewayEmbeddings, openai_client
from app.llm.prompts import Prompt
from app.problems import Problem
from app.settings import Settings

if TYPE_CHECKING:
    from app.llm.core.registry import KEY_SETTINGS as KEY_SETTINGS
    from app.llm.core.registry import key_setting as key_setting

__all__ = ["KEY_SETTINGS", "Gateway", "GatewayDep", "get_gateway", "key_setting"]


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
        self.max_steps = settings.agent_max_steps
        # Set in the lifespan once the checkpoint pool is open (app/llm/checkpoints.py).
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
        callbacks: list[BaseCallbackHandler] = [self.guard]
        self.tracer = _tracer(settings)
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
        if self._model is None:
            raise Problem(503, detail="The language model isn't configured.")
        return self._model

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

        For anything beyond a plain tool loop, build a LangGraph StateGraph whose nodes call
        gateway.model(), and compile it with checkpointer=gateway.checkpointer.
        """
        agent = create_agent(
            self.model(),
            list(tools),
            system_prompt=system_prompt,
            checkpointer=self.checkpointer,
            **options,
        )
        # A loop that never settles stops here (AGENT_MAX_STEPS) instead of at LangGraph's 10,007.
        return agent.with_config(recursion_limit=self.max_steps)

    def call_config(
        self, user: User | None, feature: str, prompt: Prompt | None = None, **config: Any
    ) -> RunnableConfig:
        """The config for a model or agent call: what it is for (feature), whom for (counted
        against their daily cap), and the prompt's version.

            answer = await gateway.model().ainvoke(
                messages, config=gateway.call_config(user, "answer", prompt)
            )
        """
        merged: dict[str, Any] = {**(prompt.config() if prompt else {}), **config}
        metadata = {**(prompt.config().get("metadata", {}) if prompt else {}), "feature": feature}
        if user is not None:
            metadata["user_ref"] = user.id
        merged["metadata"] = {**metadata, **config.get("metadata", {})}
        return cast(RunnableConfig, merged)

    async def aclose(self) -> None:
        if self.tracer:
            # Traces are sent from a background thread; send what's left before the process ends.
            await anyio.to_thread.run_sync(wait_for_all_tracers)


def _tracer(settings: Settings) -> LangChainTracer | None:
    # Attaching a tracer also stops LangChain adding its own from LANGSMITH_* variables, so
    # tracing follows these settings alone.
    if not settings.langsmith_tracing or settings.app_env == "test":
        return None
    api_key = settings.langsmith_api_key
    client = Client(
        api_url=settings.langsmith_endpoint,
        api_key=api_key.get_secret_value() if api_key else None,
        hide_inputs=settings.langsmith_hide_inputs,
        hide_outputs=settings.langsmith_hide_outputs,
        tracing_sampling_rate=settings.langsmith_tracing_sampling_rate,
    )
    return LangChainTracer(project_name=settings.langsmith_project, client=client)


def get_gateway(request: Request) -> Gateway:
    gateway: Gateway = request.app.state.gateway
    return gateway


GatewayDep = Annotated[Gateway, Depends(get_gateway)]


# Moved to app/llm/core/registry.py (#33). Until 0.8.0 the old names still answer here, with a
# DeprecationWarning, and are the same objects as the core's.
_MOVED = frozenset({"KEY_SETTINGS", "key_setting"})

if not TYPE_CHECKING:

    def __getattr__(name: str) -> object:
        if name in _MOVED:
            warnings.warn(
                f"app.llm.gateway.{name} moved to app.llm.core.registry.{name}: import it from "
                "there (the old name goes in 0.8.0)",
                DeprecationWarning,
                stacklevel=2,
            )
            return getattr(registry, name)
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
