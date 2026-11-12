"""The MAF harness: the real Gateway, with its chat client swapped for a scripted one composed from
MAF's own layers (function invocation, chat middleware, telemetry) over a scripted base client, and
carrying the gateway's own redaction and budget middleware. A call goes through exactly what a real
call goes through; only the provider is replaced, and its failures are the errors the real Anthropic
client raises (the SDK's error wrapped by MAF's own wrapper)."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from types import SimpleNamespace
from typing import Any, cast

import anthropic
import httpx2
from agent_framework import (
    BaseChatClient,
    ChatMiddlewareLayer,
    ChatResponse,
    ChatResponseUpdate,
    Content,
    FunctionInvocationLayer,
    FunctionTool,
    Message,
    ResponseStream,
    UsageDetails,
)
from agent_framework.observability import ChatTelemetryLayer
from agent_framework_anthropic._chat_client import _wrap_anthropic_error
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.llm.core.budget import CallTags, Meter
from app.llm.core.errors import ProviderError
from app.llm.core.redaction import Redactor
from app.llm.maf.gateway import Gateway, NoExtraChatMiddleware, check_chain
from app.llm.maf.middleware import tagged
from app.settings import Settings
from tests.harness import Failure, FakeEmbeddingsApi, Reply
from tests.harness import Message as Neutral

# What a failed chat call raises for each kind: the core's ProviderError, which the middleware makes
# from the error MAF raised (a plain RuntimeError for "unknown": of no known kind, it is not
# changed).
RAISES: dict[str, type[BaseException]] = {
    "rate_limit": ProviderError,
    "timeout": ProviderError,
    "connection": ProviderError,
    "server": ProviderError,
    "context_overflow": ProviderError,
    "authentication": ProviderError,
    "invalid_request": ProviderError,
    "unknown": RuntimeError,
}


def sdk_error(failure: Failure) -> Exception:
    """The Anthropic SDK's own exception for a failure kind, as the provider's SDK raises it."""
    request = httpx2.Request("POST", "http://x")
    headers = {"retry-after": failure.retry_after} if failure.retry_after is not None else {}

    def status(code: int) -> httpx2.Response:
        return httpx2.Response(code, request=request, headers=headers)

    message = failure.message
    match failure.kind:
        case "timeout":
            return anthropic.APITimeoutError(request=request)
        case "connection":
            return anthropic.APIConnectionError(message=message, request=request)
        case "rate_limit":
            return anthropic.RateLimitError(message, response=status(429), body=None)
        case "server":
            return anthropic.InternalServerError(message, response=status(500), body=None)
        case "context_overflow":
            return anthropic.BadRequestError(
                f"prompt is too long: {message}", response=status(400), body=None
            )
        case "invalid_request":
            return anthropic.BadRequestError(message, response=status(400), body=None)
        case "authentication":
            return anthropic.AuthenticationError(message, response=status(401), body=None)
    return RuntimeError(message)


def error_of(failure: Failure) -> Exception:
    """What MAF's Anthropic client raises: the SDK's error wrapped by MAF's own wrapper (a 429
    arrives as an invalid-request error), or an unknown error as it is."""
    error = sdk_error(failure)
    if failure.kind == "unknown":
        return error
    wrapped = _wrap_anthropic_error(error)
    wrapped.__cause__ = error
    return wrapped


class ScriptedBase(BaseChatClient[Any]):
    """The provider-level fake: answers (or fails) from a script, one item per call, and remembers
    the messages and options it was given."""

    def __init__(
        self, script: Sequence[Reply | Failure | ChatResponse[Any]], **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.script: list[Reply | Failure | ChatResponse[Any]] = list(script)
        self.seen: list[list[Message]] = []
        self.seen_options: list[Mapping[str, Any]] = []

    def _inner_get_response(
        self,
        *,
        messages: Sequence[Message],
        stream: bool,
        options: Mapping[str, Any],
        **kwargs: Any,
    ) -> Awaitable[ChatResponse[Any]] | ResponseStream[ChatResponseUpdate, ChatResponse[Any]]:
        assert not stream, "the gateway refuses streamed calls before they reach the provider"

        async def answer() -> ChatResponse[Any]:
            self.seen.append(list(messages))
            self.seen_options.append(dict(options))
            item = self.script.pop(0)
            if isinstance(item, ChatResponse):  # a test that needs tool calls or other contents
                if options.get("tool_choice") == "none":
                    # Like a provider told to offer no tools: it answers in text.
                    return ChatResponse(messages=[Message("assistant", ["done"])])
                return item
            if isinstance(item, Failure):
                raise error_of(item)
            response = ChatResponse(
                messages=[Message("assistant", [item.text])], model="scripted-model"
            )
            if item.usage is not None:
                given, produced = item.usage
                response.usage_details = UsageDetails(
                    input_token_count=given, output_token_count=produced
                )
            return response

        return answer()


class ScriptedClient(
    NoExtraChatMiddleware,
    FunctionInvocationLayer[Any],
    ChatMiddlewareLayer[Any],
    ChatTelemetryLayer[Any],
    ScriptedBase,
):
    """The layers of MAF's real clients, in their order, over the scripted base."""


class PlainScriptedClient(NoExtraChatMiddleware, ChatMiddlewareLayer[Any], ScriptedBase):
    """Without the function-invocation layer: a tool call the model makes comes back to the caller
    instead of being run, so a test can read what the middleware restored in it."""


def to_maf(message: Neutral) -> Message:
    """A neutral message (`{"role", "content"}`, content a string or blocks) as MAF's."""
    content = message["content"]
    if isinstance(content, str):
        return Message(message["role"], [content])
    parts: list[Content] = []
    for block in content:
        match block["type"]:
            case "text":
                parts.append(Content.from_text(block["text"]))
            case "image":
                parts.append(Content.from_uri(block["url"], media_type="image/png"))
            case "file":
                parts.append(Content.from_uri(block["url"], media_type=block["mime_type"]))
            case other:
                raise ValueError(f"the harness has no MAF form for a {other!r} block")
    return Message(message["role"], parts)


def to_neutral(message: Message) -> Neutral:
    """A MAF message as a neutral one: its text as a string, or a list of blocks when it holds
    anything else."""
    kinds = {content.type for content in message.contents}
    if kinds <= {"text"}:
        return {"role": message.role, "content": "".join(c.text or "" for c in message.contents)}
    blocks: list[dict[str, Any]] = []
    for content in message.contents:
        if content.type == "text":
            blocks.append({"type": "text", "text": content.text})
        elif content.type in {"uri", "data"}:
            blocks.append({"type": "image", "url": content.uri})
        else:
            blocks.append({"type": content.type})
    return {"role": message.role, "content": blocks}


class MafHarness:
    name = "maf"

    def gateway(
        self, settings: Settings, sessionmaker: Callable[[], async_sessionmaker[AsyncSession]]
    ) -> Gateway:
        return Gateway(settings, sessionmaker)

    def meter(self, gateway: Gateway) -> Meter:
        return gateway.meter

    def install_scripted(
        self, gateway: Gateway, script: Sequence[Reply | Failure | ChatResponse[Any]]
    ) -> None:
        client = ScriptedClient(
            script=script,
            middleware=gateway.middleware,
            function_invocation_configuration=gateway.tool_loop,
        )
        check_chain(client, gateway.middleware)
        gateway._client = client

    def install_plain(self, gateway: Gateway, script: Sequence[Any]) -> PlainScriptedClient:
        client = PlainScriptedClient(script=script, middleware=gateway.middleware)
        check_chain(client, gateway.middleware)
        gateway._client = client
        return client

    def _scripted(self, gateway: Gateway) -> ScriptedClient:
        client = gateway._client
        assert isinstance(client, ScriptedClient)
        return client

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
        messages: Sequence[Neutral],
        tags: dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
    ) -> str:
        with tagged(CallTags(**(tags or {}))):
            response = await gateway.client().get_response(
                [to_maf(message) for message in messages],
                options=options,
            )
        return response.text

    async def ask_with_tool(
        self, gateway: Gateway, messages: Sequence[Neutral], tool: Callable[..., Any]
    ) -> str:
        response = await gateway.client().get_response(
            [to_maf(message) for message in messages], options={"tools": [tool]}
        )
        return response.text

    def last_request_tools(self, gateway: Gateway) -> list[dict[str, Any]]:
        tools = self._scripted(gateway).seen_options[-1].get("tools") or []
        return [
            {"name": tool.name, "description": tool.description, "parameters": tool.parameters()}
            for tool in tools
            if isinstance(tool, FunctionTool)
        ]

    async def ask_structured(
        self, gateway: Gateway, messages: Sequence[Neutral], schema: type[BaseModel]
    ) -> Any:
        response = await gateway.client().get_response(
            [to_maf(message) for message in messages], options={"response_format": schema}
        )
        return response.value

    def last_request_schema(self, gateway: Gateway) -> dict[str, Any] | None:
        response_format = self._scripted(gateway).seen_options[-1].get("response_format")
        if isinstance(response_format, type) and issubclass(response_format, BaseModel):
            return response_format.model_json_schema()  # what the provider client would send
        return cast("dict[str, Any] | None", response_format)

    def last_request_messages(self, gateway: Gateway) -> list[Neutral] | None:
        seen = self._scripted(gateway).seen
        return [to_neutral(message) for message in seen[-1]] if seen else None

    def raised_by(self, failure: Failure) -> type[BaseException]:
        return RAISES[failure.kind]

    def install_embeddings(self, gateway: Gateway, api: FakeEmbeddingsApi) -> None:
        gateway.embeddings()._client = SimpleNamespace(embeddings=api)

    def set_redactor(self, gateway: Gateway, redactor: Redactor | None) -> None:
        gateway.redactor = redactor
        gateway.redaction.redactor = redactor
        gateway.embeddings()._redactor = redactor

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
        return await gateway.embeddings().embed_documents(texts, feature=feature, user_ref=user_ref)

    async def embed_query(
        self, gateway: Gateway, text: str, *, user_ref: str | None = None
    ) -> list[float]:
        return await gateway.embeddings().embed_query(text, user_ref=user_ref)

    def raised_by_embeddings(self, failure: Failure) -> type[BaseException]:
        # The core's ProviderError, which app.llm.core.errors answers like a chat call's.
        return ProviderError


HARNESS = MafHarness()
