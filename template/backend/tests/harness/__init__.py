"""The test harness the shared parity tests (tests/test_llm_parity_*.py) run through.

A parity test states what every model framework must do (settle a call at its real cost, release
the reservation of a call the provider rejected, refuse a request the budget cannot bound) and
imports no framework. Everything framework-specific is behind the `Harness` protocol below; one
module per framework (tests/harness/langchain.py) implements it, and the `framework` fixture
(tests/conftest.py) runs each parity test once per framework the project chose. This module and the
parity files import only the standard library, pytest, the app's own framework-neutral code and the
provider SDK the fake embeddings API raises from (tests/test_parity_guards.py in the template
repository forbids the rest).

The vocabulary is neutral on purpose:
- A message is a dict, `{"role": "user", "content": "text"}`, whose content is a string or a list
  of blocks (`{"type": "text", ...}`, `{"type": "image", "url": ...}`).
- What the provider answers is a `Reply`; how it fails is a `Failure` of one of FAILURE_KINDS.
- Anything a request asks for beyond its messages (a bigger output limit, a tool, a service tier)
  is built by `params_for_tools` and passed to `ask` as `options`, so a test never knows how a
  framework spells it.
"""

from __future__ import annotations

import importlib
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Protocol

from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.llm.core.budget import Meter
from app.llm.core.context import ContextReport
from app.llm.core.redaction import Redactor
from app.settings import Settings

__all__ = [
    "FAILURE_KINDS",
    "AsksTool",
    "Failure",
    "FakeEmbeddingsApi",
    "Harness",
    "Message",
    "Reply",
    "load",
    "user",
]

Message = dict[str, Any]

# Each kind is a way a provider call fails, named as the core's FailureKind is (app/llm/core/
# errors.py), plus "unknown" for an error of no known kind (a bug, a library error).
FAILURE_KINDS = (
    "rate_limit",
    "timeout",
    "connection",
    "server",
    "context_overflow",
    "authentication",
    "invalid_request",
    "unknown",
)


@dataclass(frozen=True, slots=True)
class Reply:
    """What the provider answers: the text, and the usage it reports (input tokens, output
    tokens), or None when it reports none. Streamed, the text arrives in `pieces` when given (they
    must add up to `text`), else in one piece, and the usage after the last piece."""

    text: str
    usage: tuple[int, int] | None = (1_000, 500)
    pieces: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        assert not self.pieces or "".join(self.pieces) == self.text, self.pieces

    def streamed(self) -> tuple[str, ...]:
        return self.pieces or (self.text,)


@dataclass(frozen=True, slots=True)
class AsksTool:
    """A model that asks for the tool `name` (with `args`) whenever it is offered tools, and
    answers `text` when it is offered none: the model that loops until the step limit stops it.
    Each call reports the same usage."""

    name: str
    args: dict[str, Any]
    text: str = "final answer"
    usage: tuple[int, int] = (100, 10)


@dataclass(frozen=True, slots=True)
class Failure:
    """A failed provider call. `message` stands for the provider's own text (which can echo the
    prompt), so a test can check it never reaches a client; `retry_after` is the provider's
    Retry-After value, when it sent one."""

    kind: str
    message: str = "the provider failed"
    retry_after: str | None = None

    def __post_init__(self) -> None:
        assert self.kind in FAILURE_KINDS, self.kind


def user(content: Any) -> Message:
    return {"role": "user", "content": content}


class Harness(Protocol):
    """What a framework must offer the parity tests."""

    name: str

    def gateway(
        self, settings: Settings, sessionmaker: Callable[[], async_sessionmaker[AsyncSession]]
    ) -> Any:
        """The project's gateway, built the way the app builds it, on `sessionmaker`."""
        ...

    def meter(self, gateway: Any) -> Meter:
        """The budget's Meter for chat calls (its cap, user_cap, price, today are plain attributes
        a test may set)."""
        ...

    def install_scripted(self, gateway: Any, script: Sequence[Reply | Failure | AsksTool]) -> None:
        """Replace the provider with one that answers `script` in order, one item per provider
        call, through the same path a real call takes (the budget and redaction included)."""
        ...

    def scripted_left(self, gateway: Any) -> int:
        """How many scripted items the provider has not been asked for yet."""
        ...

    def params_for_tools(
        self,
        *,
        tools: Sequence[dict[str, Any]] | None = None,
        service_tier: str | None = None,
        max_output_tokens: int | None = None,
    ) -> Any:
        """The `options` for `ask`: tools (provider-shaped dicts), a service tier, a limit on the
        answer's tokens."""
        ...

    async def ask(
        self,
        gateway: Any,
        messages: Sequence[Message],
        tags: dict[str, Any] | None = None,
        options: Any = None,
    ) -> str:
        """One call; returns the answer's text. `tags` are the call's metadata: prompt,
        prompt_sha, feature, user_ref."""
        ...

    async def ask_with_tool(
        self, gateway: Any, messages: Sequence[Message], tool: Callable[..., Any]
    ) -> str:
        """One call that offers `tool` (a plain function: its name, docstring and parameters are
        the definition) the way the framework offers a tool; returns the answer's text."""
        ...

    def last_request_tools(self, gateway: Any) -> list[dict[str, Any]]:
        """The tool definitions the provider was last given, after redaction, as
        `{"name", "description", "parameters"}` (the parameters a JSON schema)."""
        ...

    async def ask_structured(
        self, gateway: Any, messages: Sequence[Message], schema: type[BaseModel]
    ) -> Any:
        """One call that asks for an answer in `schema` the way the framework asks for structured
        output; returns the parsed answer."""
        ...

    def last_request_schema(self, gateway: Any) -> dict[str, Any] | None:
        """The output schema the provider was last given, after redaction, as a JSON schema, or
        None when the call asked for none."""
        ...

    async def stream(
        self,
        gateway: Any,
        messages: Sequence[Message],
        *,
        take: int | None = None,
        tags: dict[str, Any] | None = None,
    ) -> list[str]:
        """One streamed call, read the way the framework streams; returns the text of each update
        that carried text. With `take`, stops reading after that many and closes the stream."""
        ...

    async def run_agent(
        self,
        gateway: Any,
        question: str,
        tools: Sequence[Callable[..., Any]],
        *,
        tags: dict[str, Any] | None = None,
        instructions: str | None = None,
    ) -> tuple[str, bool]:
        """One run of the gateway's agent with `tools` (and its system prompt, `instructions`): its
        final answer's text, and whether the run ended on the forced final answer (the neutral
        step_limit_reached flag). Nothing of the run is kept."""
        ...

    async def open_threads(self, gateway: Any, settings: Settings) -> Callable[[], Awaitable[None]]:
        """Make the gateway keep conversations in the test database, the way the app does at
        start-up; returns what closes it. `settings` carries the database URL."""
        ...

    async def thread_turn(
        self,
        gateway: Any,
        *,
        user_ref: str,
        conversation_id: str,
        messages: Sequence[Message],
        tools: Sequence[Callable[..., Any]] = (),
        stream: bool = False,
    ) -> str:
        """One turn of a stored conversation, keyed by the user and the conversation id (derived on
        the server, as the app must), by an agent with `tools`; returns the answer's text. With
        `stream`, the agent's run is streamed and read to its end."""
        ...

    async def context_report(
        self,
        gateway: Any,
        *,
        user_ref: str,
        conversation_id: str,
        tools: Sequence[Callable[..., Any]] = (),
    ) -> ContextReport:
        """The gateway's context report for the stored conversation thread_turn keeps (#46), for
        an agent with these tools."""
        ...

    async def forget_threads(self, gateway: Any, *, user_ref: str, conversation_id: str) -> None:
        """Delete what the turns above stored (the test database outlives one test)."""
        ...

    def last_request_offered_tools(self, gateway: Any) -> bool:
        """Whether the provider's last request let the model call a tool."""
        ...

    def last_request_messages(self, gateway: Any) -> list[Message] | None:
        """What the provider was last given, after redaction, as neutral messages
        (`{"role": "user", "content": "text"}`, role one of system, user, assistant, tool), or None
        when it was not asked yet. No test uses it in PR 1; it is the first of what the protocol
        grows with (streaming, tools, structured output) as shared tests move here."""
        ...

    def raised_by(self, failure: Failure) -> type[BaseException]:
        """The exception type a failed chat call raises for `failure` (the framework's own)."""
        ...

    def install_embeddings(self, gateway: Any, api: FakeEmbeddingsApi) -> None: ...

    def set_redactor(self, gateway: Any, redactor: Redactor | None) -> None:
        """Replace the redactor chat and embedding calls use (None is redaction off)."""
        ...

    def embedding_meter(self, gateway: Any) -> Meter: ...

    async def embed_documents(
        self,
        gateway: Any,
        texts: list[str],
        *,
        feature: str | None = None,
        user_ref: str | None = None,
    ) -> list[list[float]]: ...

    async def embed_query(
        self,
        gateway: Any,
        text: str,
        *,
        feature: str | None = None,
        user_ref: str | None = None,
    ) -> list[float]: ...

    def raised_by_embeddings(self, failure: Failure) -> type[BaseException]:
        """The exception type a failed embeddings call raises for `failure`."""
        ...


class FakeEmbeddingsApi:
    """Answers like OpenAI's embeddings endpoint: one vector per input, usage in tokens. Core's
    OpenAI embeddings call it whichever framework the project chose."""

    def __init__(self, tokens_per_input: int = 7, error: Failure | None = None) -> None:
        self.requests: list[list[str]] = []
        self.tokens_per_input = tokens_per_input
        self.error = error

    async def create(self, *, model: str, input: list[str], **kwargs: Any) -> Any:
        self.requests.append(list(input))
        if self.error:
            raise openai_error(self.error)
        data = [SimpleNamespace(index=i, embedding=[float(i)] * 3) for i in range(len(input))]
        usage = SimpleNamespace(prompt_tokens=self.tokens_per_input * len(input))
        return SimpleNamespace(data=list(reversed(data)), usage=usage)


def openai_error(failure: Failure) -> Exception:
    """The OpenAI SDK's own exception for a failure kind (what the provider's SDK raises)."""
    import httpx2
    import openai

    request = httpx2.Request("POST", "http://x")

    def status(code: int) -> httpx2.Response:
        return httpx2.Response(code, request=request)

    message = failure.message
    match failure.kind:
        case "timeout":
            return openai.APITimeoutError(request=request)
        case "connection":
            return openai.APIConnectionError(message=message, request=request)
        case "rate_limit":
            return openai.RateLimitError(message, response=status(429), body=None)
        case "server":
            return openai.InternalServerError(message, response=status(500), body=None)
        case "context_overflow" | "invalid_request":
            return openai.BadRequestError(message, response=status(400), body=None)
        case "authentication":
            return openai.AuthenticationError(message, response=status(401), body=None)
    return RuntimeError(message)


def load(framework: str) -> Harness:
    """The harness of one framework, imported only when that framework is the one under test (its
    module imports the framework)."""
    module = importlib.import_module(f"tests.harness.{framework}")
    harness: Harness = module.HARNESS
    return harness
