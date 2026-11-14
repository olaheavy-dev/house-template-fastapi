"""The LangChain harness: the real Gateway, with its chat model swapped for a scripted fake that
carries the gateway's own BudgetGuard, so a call goes through the redaction wrapper and the budget
exactly as in the app and nothing reaches a provider."""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Sequence
from contextlib import aclosing
from types import SimpleNamespace
from typing import Any, cast

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
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    convert_to_messages,
)
from langchain_core.messages.ai import UsageMetadata
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel as PydanticModel
from pydantic import Field
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.llm.core.budget import Meter
from app.llm.core.context import ContextReport
from app.llm.core.redaction import Redactor
from app.llm.langchain import checkpoints, steps
from app.llm.langchain.gateway import Gateway
from app.settings import Settings
from tests.harness import AsksTool, Failure, FakeEmbeddingsApi, Message, Reply

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


def thread_id(user_ref: str, conversation_id: str) -> str:
    """The thread id an app derives on the server from the user and the conversation."""
    return f"{user_ref}:{conversation_id}"


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


def offers_tools(kwargs: dict[str, Any]) -> bool:
    return bool(kwargs.get("tools")) and kwargs.get("tool_choice") != "none"


def _asks(item: AsksTool, kwargs: dict[str, Any]) -> AIMessage:
    given, produced = item.usage
    usage: UsageMetadata = {
        "input_tokens": given,
        "output_tokens": produced,
        "total_tokens": given + produced,
    }
    if not offers_tools(kwargs):
        return AIMessage(item.text, usage_metadata=usage)
    call = {"name": item.name, "args": item.args, "id": f"call_{uuid.uuid4().hex[:8]}"}
    return AIMessage("", tool_calls=[call], usage_metadata=usage)


class ScriptedModel(GenericFakeChatModel):
    """Answers (or fails) from a script, one item per call, and remembers what it was given."""

    script: list[Reply | Failure | AsksTool] = Field(default_factory=list)
    seen: list[list[BaseMessage]] = Field(default_factory=list)
    seen_kwargs: list[dict[str, Any]] = Field(default_factory=list)

    def bind_tools(
        self, tools: Sequence[Any], *, tool_choice: Any = None, **kwargs: Any
    ) -> Runnable[Any, Any]:
        """As ChatOpenAI binds tools: OpenAI-shaped function definitions."""
        return self.bind(tools=[convert_to_openai_tool(tool) for tool in tools], **kwargs)

    def with_structured_output(  # type: ignore[override]
        self, schema: type[PydanticModel], *, include_raw: bool = False, **kwargs: Any
    ) -> Runnable[Any, Any]:
        """As ChatOpenAI's json_schema method: the model class itself is bound (the provider SDK
        builds the schema from it), and the answer's text is parsed into it."""
        return self.bind(response_format=schema) | RunnableLambda(
            lambda message: schema.model_validate_json(message.text)
        )

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.seen.append(list(messages))
        self.seen_kwargs.append(dict(kwargs))
        item = self.script.pop(0)
        if isinstance(item, Failure):
            raise _error(item)
        if isinstance(item, AsksTool):
            return ChatResult(generations=[ChatGeneration(message=_asks(item, kwargs))])
        return ChatResult(generations=[ChatGeneration(message=_answer(item))])

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """The scripted reply in its pieces, the usage on the last one (as providers report it)."""
        self.seen.append(list(messages))
        self.seen_kwargs.append(dict(kwargs))
        item = self.script.pop(0)
        if isinstance(item, Failure):
            raise _error(item)
        if isinstance(item, AsksTool):
            # One chunk holding the whole call, as its fragments merge (with the call's usage).
            asked = _asks(item, kwargs)
            chunk = ChatGenerationChunk(
                message=AIMessageChunk(
                    content="",
                    tool_call_chunks=[
                        {
                            "name": call["name"],
                            "args": json.dumps(call["args"]),
                            "id": call["id"],
                            "index": index,
                            "type": "tool_call_chunk",
                        }
                        for index, call in enumerate(asked.tool_calls)
                    ],
                    usage_metadata=asked.usage_metadata,
                )
            )
            if run_manager is not None:
                await run_manager.on_llm_new_token("", chunk=chunk)
            yield chunk
            return
        pieces = item.streamed()
        for number, piece in enumerate(pieces, start=1):
            usage = _answer(item).usage_metadata if number == len(pieces) else None
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=piece, usage_metadata=usage))
            if run_manager is not None:
                await run_manager.on_llm_new_token(piece, chunk=chunk)
            yield chunk


class LangChainHarness:
    name = "langchain"

    def gateway(
        self, settings: Settings, sessionmaker: Callable[[], async_sessionmaker[AsyncSession]]
    ) -> Gateway:
        return Gateway(settings, sessionmaker)

    def meter(self, gateway: Gateway) -> Meter:
        return gateway.guard.meter

    def install_scripted(
        self, gateway: Gateway, script: Sequence[Reply | Failure | AsksTool]
    ) -> None:
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

    async def ask_with_tool(
        self, gateway: Gateway, messages: Sequence[Message], tool: Callable[..., Any]
    ) -> str:
        model = gateway.model().bind_tools([tool])
        answer = await model.ainvoke(convert_to_messages(list(messages)))
        return str(answer.text)

    def last_request_tools(self, gateway: Gateway) -> list[dict[str, Any]]:
        kwargs = self._scripted(gateway).seen_kwargs[-1]
        return [
            {
                "name": tool["function"]["name"],
                "description": tool["function"].get("description"),
                "parameters": tool["function"].get("parameters"),
            }
            for tool in kwargs.get("tools") or []
        ]

    async def ask_structured(
        self, gateway: Gateway, messages: Sequence[Message], schema: type[PydanticModel]
    ) -> Any:
        runnable = gateway.model().with_structured_output(schema)
        return await runnable.ainvoke(convert_to_messages(list(messages)))

    def last_request_schema(self, gateway: Gateway) -> dict[str, Any] | None:
        response_format = self._scripted(gateway).seen_kwargs[-1].get("response_format")
        if isinstance(response_format, type) and issubclass(response_format, PydanticModel):
            return response_format.model_json_schema()  # what the provider SDK would send
        return cast("dict[str, Any] | None", response_format)

    async def stream(
        self,
        gateway: Gateway,
        messages: Sequence[Message],
        *,
        take: int | None = None,
        tags: dict[str, Any] | None = None,
    ) -> list[str]:
        config: Any = {"metadata": dict(tags)} if tags else None
        out: list[str] = []
        chunks = gateway.model().astream(convert_to_messages(list(messages)), config=config)
        async with aclosing(cast(AsyncGenerator[Any, None], chunks)) as stream:
            async for chunk in stream:
                if chunk.text:
                    out.append(chunk.text)
                if take is not None and len(out) >= take:
                    break
        return out

    async def run_agent(
        self,
        gateway: Gateway,
        question: str,
        tools: Sequence[Callable[..., Any]],
        *,
        tags: dict[str, Any] | None = None,
        instructions: str | None = None,
    ) -> tuple[str, bool]:
        agent = gateway.agent(list(tools), system_prompt=instructions)
        thread_id = f"parity-{uuid.uuid4()}"
        try:
            state = await agent.ainvoke(
                {"messages": [HumanMessage(question)]},
                config={"configurable": {"thread_id": thread_id}, "metadata": dict(tags or {})},
            )
        finally:
            # The test database can outlive the run (TEST_DATABASE_URL): keep no thread.
            if gateway.checkpointer is not None:
                await gateway.checkpointer.adelete_thread(thread_id)
        return str(state["messages"][-1].text), steps.step_limit_reached(state)

    async def open_threads(
        self, gateway: Gateway, settings: Settings
    ) -> Callable[[], Awaitable[None]]:
        saver, pool = await checkpoints.open_checkpointer(settings)
        gateway.checkpointer = saver
        return pool.close

    async def thread_turn(
        self,
        gateway: Gateway,
        *,
        user_ref: str,
        conversation_id: str,
        messages: Sequence[Message],
        tools: Sequence[Callable[..., Any]] = (),
        stream: bool = False,
    ) -> str:
        agent = gateway.agent(list(tools))
        given = {"messages": convert_to_messages(list(messages))}
        config: Any = {"configurable": {"thread_id": thread_id(user_ref, conversation_id)}}
        if not stream:
            state = await agent.ainvoke(given, config=config)
            return str(state["messages"][-1].text)
        # Token by token, as a chat route streams ("messages"), with the state after each step.
        last: Any = None
        async for mode, part in agent.astream(
            given, config=config, stream_mode=["messages", "values"]
        ):
            if mode == "values":
                last = part
        return str(last["messages"][-1].text)

    async def context_report(
        self,
        gateway: Gateway,
        *,
        user_ref: str,
        conversation_id: str,
        tools: Sequence[Callable[..., Any]] = (),
    ) -> ContextReport:
        return await gateway.context_report(thread_id(user_ref, conversation_id), tools=tools)

    async def forget_threads(
        self, gateway: Gateway, *, user_ref: str, conversation_id: str
    ) -> None:
        assert gateway.checkpointer is not None
        await gateway.checkpointer.adelete_thread(thread_id(user_ref, conversation_id))

    def last_request_offered_tools(self, gateway: Gateway) -> bool:
        return offers_tools(self._scripted(gateway).seen_kwargs[-1])

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
        self,
        gateway: Gateway,
        text: str,
        *,
        feature: str | None = None,
        user_ref: str | None = None,
    ) -> list[float]:
        return await gateway.embeddings().aembed_query(text, feature=feature, user_ref=user_ref)

    def raised_by_embeddings(self, failure: Failure) -> type[BaseException]:
        # The same family as chat calls: the embeddings adapter turns the core's ProviderError
        # into LangChain's (app/llm/langchain/embeddings.py).
        return ERRORS[failure.kind]


HARNESS = LangChainHarness()
