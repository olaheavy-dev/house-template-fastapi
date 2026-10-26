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
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated

import anyio
from fastapi import Depends, Request
from langchain.chat_models import init_chat_model
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.tracers.langchain import LangChainTracer, wait_for_all_tracers
from langsmith import Client
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.llm import prices
from app.llm.budget import BudgetGuard
from app.problems import Problem
from app.settings import Settings


class Gateway:
    def __init__(
        self, settings: Settings, sessionmaker: Callable[[], async_sessionmaker[AsyncSession]]
    ) -> None:
        price = prices.PRICES.get(settings.llm_model)
        if price is None:
            raise ValueError(f"LLM_MODEL {settings.llm_model!r} has no price in app/llm/prices.py")
        self.name = settings.llm_model
        self.guard = BudgetGuard(
            model=settings.llm_model,
            price=price,
            cap_micro_usd=int(settings.llm_daily_budget_usd * prices.MICRO_USD),
            max_tokens=settings.llm_max_tokens,
            sessionmaker=sessionmaker,
        )
        callbacks: list[BaseCallbackHandler] = [self.guard]
        self.tracer = _tracer(settings)
        if self.tracer:
            callbacks.append(self.tracer)

        provider = settings.llm_model.split(":", 1)[0]
        api_key = {
            "anthropic": settings.anthropic_api_key,
            "openai": settings.openai_api_key,
        }.get(provider)
        # Without a key there is no model, and calls get a 503 instead of reaching the provider.
        self._model: BaseChatModel | None = None
        if api_key is not None:
            self._model = init_chat_model(
                settings.llm_model,
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
