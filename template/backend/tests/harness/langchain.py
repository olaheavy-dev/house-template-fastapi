"""The LangChain harness: the real Gateway, with its chat model swapped for a scripted fake that
carries the gateway's own BudgetGuard, so a call goes through the redaction wrapper and the budget
exactly as in the app and nothing reaches a provider."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import Any

from langchain_core.callbacks import AsyncCallbackManagerForLLMRun
from langchain_core.exceptions import (
    ContextOverflowError,
    ModelAPIError,
    ModelAuthenticationError,
    ModelConnectionError,
    ModelInvalidRequestError,
    ModelRateLimitError,
    ModelTimeoutError,
)
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage, convert_to_messages
from langchain_core.messages.ai import UsageMetadata
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.llm.core.budget import Meter
from app.llm.core.redaction import Redactor
from app.llm.langchain.gateway import Gateway
from app.settings import Settings
from tests.harness import Failure, FakeEmbeddingsApi, Message, Reply

# The LangChain exception each failure kind raises (a plain RuntimeError for "unknown").
ERRORS: dict[str, type[BaseException]] = {
    "rate_limit": ModelRateLimitError,
    "timeout": ModelTimeoutError,
    "connection": ModelConnectionError,
    "server": ModelAPIError,
    "context_overflow": ContextOverflowError,
    "authentication": ModelAuthenticationError,
    "invalid_request": ModelInvalidRequestError,
    "unknown": RuntimeError,
}


def _error(failure: Failure) -> BaseException:
    error = ERRORS[failure.kind](failure.message)
    if failure.retry_after is not None:
        # Where the provider's Retry-After header is found (app/llm/langchain/errors.py).
        error.response = SimpleNamespace(  # type: ignore[attr-defined]
            headers={"retry-after": failure.retry_after}
        )
    return error


def _answer(reply: Reply) -> AIMessage:
    if reply.usage is None:
        return AIMessage(reply.text)
    given, produced = reply.usage
    usage: UsageMetadata = {
        "input_tokens": given,
        "output_tokens": produced,
        "total_tokens": given + produced,
    }
    return AIMessage(reply.text, usage_metadata=usage)


class ScriptedModel(GenericFakeChatModel):
    """Answers (or fails) from a script, one item per call, and remembers what it was given."""

    script: list[Reply | Failure] = Field(default_factory=list)
    seen: list[list[BaseMessage]] = Field(default_factory=list)

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.seen.append(list(messages))
        item = self.script.pop(0)
        if isinstance(item, Failure):
            raise _error(item)
        return ChatResult(generations=[ChatGeneration(message=_answer(item))])


class LangChainHarness:
    name = "langchain"

    def gateway(
        self, settings: Settings, sessionmaker: Callable[[], async_sessionmaker[AsyncSession]]
    ) -> Gateway:
        return Gateway(settings, sessionmaker)

    def meter(self, gateway: Gateway) -> Meter:
        return gateway.guard.meter

    def install_scripted(self, gateway: Gateway, script: Sequence[Reply | Failure]) -> None:
        gateway._model = ScriptedModel(
            messages=iter([]), script=list(script), seen=[], callbacks=[gateway.guard]
        )

    def _scripted(self, gateway: Gateway) -> ScriptedModel:
        model = gateway._model
        assert isinstance(model, ScriptedModel)
        return model

    def scripted_left(self, gateway: Gateway) -> int:
        return len(self._scripted(gateway).script)

    def params_for_tools(
        self,
        *,
        tools: Sequence[dict[str, Any]] | None = None,
        service_tier: str | None = None,
        max_output_tokens: int | None = None,
    ) -> dict[str, Any]:
        options: dict[str, Any] = {}
        if tools is not None:
            options["tools"] = list(tools)
        if service_tier is not None:
            options["service_tier"] = service_tier
        if max_output_tokens is not None:
            options["max_tokens"] = max_output_tokens
        return options

    async def ask(
        self,
        gateway: Gateway,
        messages: Sequence[Message],
        tags: dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
    ) -> str:
        model = gateway.model()
        if options:
            model = model.bind(**options)  # type: ignore[assignment]
        config: Any = {"metadata": dict(tags)} if tags else None
        answer = await model.ainvoke(convert_to_messages(list(messages)), config=config)
        return answer.text

    def last_request_messages(self, gateway: Gateway) -> list[Message] | None:
        seen = self._scripted(gateway).seen
        if not seen:
            return None
        roles = {"human": "user", "ai": "assistant", "system": "system", "tool": "tool"}
        return [
            {"role": roles.get(message.type, message.type), "content": message.content}
            for message in seen[-1]
        ]

    def raised_by(self, failure: Failure) -> type[BaseException]:
        return ERRORS[failure.kind]

    def install_embeddings(self, gateway: Gateway, api: FakeEmbeddingsApi) -> None:
        gateway.embeddings()._client = SimpleNamespace(embeddings=api)

    def set_redactor(self, gateway: Gateway, redactor: Redactor | None) -> None:
        gateway.redactor = redactor
        gateway.embeddings()._redactor = redactor
        gateway._wrapped = None  # the chat model is wrapped again with the new redactor

    def embedding_meter(self, gateway: Gateway) -> Meter:
        return gateway.embeddings().meter

    async def embed_documents(
        self,
        gateway: Gateway,
        texts: list[str],
        *,
        feature: str | None = None,
        user_ref: str | None = None,
    ) -> list[list[float]]:
        return await gateway.embeddings().aembed_documents(
            texts, feature=feature, user_ref=user_ref
        )

    async def embed_query(
        self, gateway: Gateway, text: str, *, user_ref: str | None = None
    ) -> list[float]:
        return await gateway.embeddings().aembed_query(text, user_ref=user_ref)

    def raised_by_embeddings(self, failure: Failure) -> type[BaseException]:
        # The same family as chat calls: the embeddings adapter turns the core's ProviderError
        # into LangChain's (app/llm/langchain/embeddings.py).
        return ERRORS[failure.kind]


HARNESS = LangChainHarness()
