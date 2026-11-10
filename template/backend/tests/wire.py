"""A fake provider wire for the redaction tests (#78): the real OpenAI and Anthropic SDK models
send their real requests, and the HTTP layer is replaced, so a test sees exactly the bytes that
would leave the machine and answers with a canned JSON or event stream. Nothing touches the network.

The provider SDKs send through httpx2, so `httpx2.AsyncClient.send` is what is replaced. The
in-process test client uses httpx and is not affected.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any
from unittest import mock

import httpx2
import requests
from langchain_anthropic import ChatAnthropic
from langchain_core.callbacks import AsyncCallbackHandler, BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from app.llm.core.redaction import Redactor
from app.llm.langchain.redaction import RedactingChatModel
from app.llm.redaction_rules import DETECTORS

__all__ = [
    "DETECTORS",
    "EMAIL",
    "HISTORY",
    "NAME",
    "NINO",
    "PHONE",
    "PROVIDERS",
    "SEEDS",
    "STREAMING_PROVIDERS",
    "TRICKY",
    "Address",
    "Contact",
    "FakeLangSmithSession",
    "Recorder",
    "Transport",
    "anthropic_stream",
    "anthropic_stream_text",
    "anthropic_text",
    "anthropic_tool",
    "captured",
    "echo",
    "openai_stream_text",
    "openai_stream_tool",
    "openai_text",
    "openai_tool",
    "raw_model",
    "responses_stream_text",
    "responses_stream_tool",
    "responses_text",
    "responses_tool",
    "text_reply",
    "tool_reply",
    "wrapped",
]

# Chat Completions, Anthropic Messages and the OpenAI Responses API: three wire formats.
PROVIDERS = ["openai", "openai-responses", "anthropic"]
STREAMING_PROVIDERS = ["openai", "openai-responses", "anthropic"]

Reply = Callable[[dict[str, Any]], Any]


class Transport:
    """Every request body that was sent, in order."""

    def __init__(self, reply: Reply) -> None:
        self.reply = reply
        self.bodies: list[bytes] = []

    def json_bodies(self) -> list[dict[str, Any]]:
        return [json.loads(body) for body in self.bodies]

    @property
    def text(self) -> str:
        """All request bodies as one string, to search for a value that must not be there."""
        return b"\n".join(self.bodies).decode()


def sse(events: list[tuple[str | None, dict[str, Any] | str]]) -> bytes:
    out = ""
    for name, data in events:
        if name:
            out += f"event: {name}\n"
        out += "data: " + (data if isinstance(data, str) else json.dumps(data)) + "\n\n"
    return out.encode()


@contextmanager
def captured(reply: Reply) -> Iterator[Transport]:
    """Capture what the provider SDKs send. `reply` gets each request body (parsed) and returns a
    dict (a JSON answer) or bytes (an event stream)."""
    transport = Transport(reply)

    async def send(self: Any, request: httpx2.Request, **kwargs: Any) -> httpx2.Response:
        body = request.content if hasattr(request, "_content") else await request.aread()
        transport.bodies.append(body)
        await asyncio.sleep(0)  # a real send waits on the network: let other tasks run
        result = transport.reply(json.loads(body))
        if isinstance(result, bytes):
            return httpx2.Response(
                200, content=result, headers={"content-type": "text/event-stream"}, request=request
            )
        return httpx2.Response(200, json=result, request=request)

    with mock.patch.object(httpx2.AsyncClient, "send", send):
        yield transport


def raw_model(
    provider: str, *, callbacks: Sequence[BaseCallbackHandler] | None = None
) -> BaseChatModel:
    """A real provider model with a throwaway key and no retries."""
    if provider == "openai":
        return ChatOpenAI(
            model="gpt-4o-mini",
            api_key="k",
            max_retries=0,
            stream_usage=True,
            callbacks=list(callbacks or []),
        )
    if provider == "openai-responses":
        return ChatOpenAI(
            model="gpt-6-luna",
            api_key="k",
            max_retries=0,
            stream_usage=True,
            use_responses_api=True,
            callbacks=list(callbacks or []),
        )
    return ChatAnthropic(
        model="claude-haiku-5-5",
        api_key="k",
        max_retries=0,
        max_tokens=256,
        callbacks=list(callbacks or []),
    )


# --- canned answers --------------------------------------------------------------------------

USAGE = {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}


def openai_text(text: str) -> dict[str, Any]:
    return {
        "id": "c1",
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": USAGE,
    }


def openai_tool(name: str, args: str) -> dict[str, Any]:
    return {
        "id": "c1",
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": name, "arguments": args},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": USAGE,
    }


def anthropic_text(text: str, *, thinking: str | None = None) -> dict[str, Any]:
    content: list[dict[str, Any]] = [{"type": "text", "text": text}]
    if thinking is not None:
        content.insert(0, {"type": "thinking", "thinking": thinking, "signature": "sig-abc"})
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-haiku-5-5",
        "content": content,
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 11, "output_tokens": 7},
    }


def anthropic_tool(name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-haiku-5-5",
        "content": [{"type": "tool_use", "id": "toolu_1", "name": name, "input": tool_input}],
        "stop_reason": "tool_use",
        "stop_sequence": None,
        "usage": {"input_tokens": 11, "output_tokens": 7},
    }


def _responses(output: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "id": "resp_1",
        "object": "response",
        "created_at": 1,
        "model": "gpt-6-luna",
        "status": "completed",
        "output": output,
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "usage": {
            "input_tokens": 11,
            "output_tokens": 7,
            "total_tokens": 18,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    }


def responses_text(text: str) -> dict[str, Any]:
    return _responses(
        [
            {
                "type": "message",
                "id": "msg_1",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ]
    )


def responses_tool(name: str, args: str) -> dict[str, Any]:
    return _responses(
        [
            {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": name,
                "arguments": args,
                "status": "completed",
            }
        ]
    )


def text_reply(provider: str, text: str) -> dict[str, Any]:
    if provider == "openai-responses":
        return responses_text(text)
    return openai_text(text) if provider == "openai" else anthropic_text(text)


def tool_reply(provider: str, name: str, args: dict[str, Any]) -> dict[str, Any]:
    """A tool call in the provider's own shape (OpenAI's arguments are a JSON string)."""
    if provider == "anthropic":
        return anthropic_tool(name, args)
    if provider == "openai-responses":
        return responses_tool(name, json.dumps(args))
    return openai_tool(name, json.dumps(args))


# --- canned streams --------------------------------------------------------------------------


def _chunk(delta: dict[str, Any], finish: str | None = None) -> dict[str, Any]:
    return {
        "id": "c1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gpt-4o-mini",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


def _usage_chunk() -> dict[str, Any]:
    return {
        "id": "c1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gpt-4o-mini",
        "choices": [],
        "usage": USAGE,
    }


def openai_stream_text(pieces: list[str]) -> bytes:
    events: list[tuple[str | None, Any]] = [(None, _chunk({"role": "assistant", "content": ""}))]
    events += [(None, _chunk({"content": piece})) for piece in pieces]
    events += [(None, _chunk({}, "stop")), (None, _usage_chunk()), (None, "[DONE]")]
    return sse(events)


def openai_stream_tool(name: str, pieces: list[str]) -> bytes:
    start = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "index": 0,
                "id": "call_1",
                "type": "function",
                "function": {"name": name, "arguments": ""},
            }
        ],
    }
    events: list[tuple[str | None, Any]] = [(None, _chunk(start))]
    events += [
        (None, _chunk({"tool_calls": [{"index": 0, "function": {"arguments": piece}}]}))
        for piece in pieces
    ]
    events += [(None, _chunk({}, "tool_calls")), (None, _usage_chunk()), (None, "[DONE]")]
    return sse(events)


def anthropic_stream(blocks: list[tuple[dict[str, Any], list[dict[str, Any]]]]) -> bytes:
    """Each block is (the content_block_start block, [its deltas])."""
    events: list[tuple[str | None, Any]] = [
        (
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-haiku-5-5",
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 11, "output_tokens": 1},
                },
            },
        )
    ]
    for index, (start, deltas) in enumerate(blocks):
        events.append(
            (
                "content_block_start",
                {"type": "content_block_start", "index": index, "content_block": start},
            )
        )
        for delta in deltas:
            events.append(
                (
                    "content_block_delta",
                    {"type": "content_block_delta", "index": index, "delta": delta},
                )
            )
        events.append(("content_block_stop", {"type": "content_block_stop", "index": index}))
    events.append(
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 7},
            },
        )
    )
    events.append(("message_stop", {"type": "message_stop"}))
    return sse(events)


def anthropic_stream_text(pieces: list[str]) -> bytes:
    deltas: list[dict[str, Any]] = [{"type": "text_delta", "text": piece} for piece in pieces]
    return anthropic_stream([({"type": "text", "text": ""}, deltas)])


def responses_stream_text(pieces: list[str]) -> bytes:
    """A Responses API stream of one text message, in the order OpenAI sends its events."""
    item = {
        "type": "message",
        "id": "msg_1",
        "role": "assistant",
        "status": "in_progress",
        "content": [],
    }
    part = {"type": "output_text", "text": "", "annotations": []}
    events: list[tuple[str | None, Any]] = [
        (
            "response.created",
            {
                "type": "response.created",
                "sequence_number": 0,
                "response": {**_responses([]), "status": "in_progress"},
            },
        ),
        (
            "response.output_item.added",
            {
                "type": "response.output_item.added",
                "sequence_number": 1,
                "output_index": 0,
                "item": item,
            },
        ),
        (
            "response.content_part.added",
            {
                "type": "response.content_part.added",
                "sequence_number": 2,
                "item_id": "msg_1",
                "output_index": 0,
                "content_index": 0,
                "part": part,
            },
        ),
    ]
    sequence = 3
    for piece in pieces:
        events.append(
            (
                "response.output_text.delta",
                {
                    "type": "response.output_text.delta",
                    "sequence_number": sequence,
                    "item_id": "msg_1",
                    "output_index": 0,
                    "content_index": 0,
                    "delta": piece,
                    "logprobs": [],
                },
            )
        )
        sequence += 1
    text = "".join(pieces)
    done_part = {"type": "output_text", "text": text, "annotations": []}
    done_item = {**item, "status": "completed", "content": [done_part]}
    events += [
        (
            "response.output_text.done",
            {
                "type": "response.output_text.done",
                "sequence_number": sequence,
                "item_id": "msg_1",
                "output_index": 0,
                "content_index": 0,
                "text": text,
                "logprobs": [],
            },
        ),
        (
            "response.content_part.done",
            {
                "type": "response.content_part.done",
                "sequence_number": sequence + 1,
                "item_id": "msg_1",
                "output_index": 0,
                "content_index": 0,
                "part": done_part,
            },
        ),
        (
            "response.output_item.done",
            {
                "type": "response.output_item.done",
                "sequence_number": sequence + 2,
                "output_index": 0,
                "item": done_item,
            },
        ),
        (
            "response.completed",
            {
                "type": "response.completed",
                "sequence_number": sequence + 3,
                "response": responses_text(text),
            },
        ),
    ]
    return sse(events)


def responses_stream_tool(name: str, pieces: list[str]) -> bytes:
    """A Responses API stream of one function call whose arguments arrive in `pieces`."""
    args = "".join(pieces)
    item = {
        "type": "function_call",
        "id": "fc_1",
        "call_id": "call_1",
        "name": name,
        "arguments": "",
        "status": "in_progress",
    }
    events: list[tuple[str | None, Any]] = [
        (
            "response.created",
            {
                "type": "response.created",
                "sequence_number": 0,
                "response": {**_responses([]), "status": "in_progress"},
            },
        ),
        (
            "response.output_item.added",
            {
                "type": "response.output_item.added",
                "sequence_number": 1,
                "output_index": 0,
                "item": item,
            },
        ),
    ]
    sequence = 2
    for piece in pieces:
        events.append(
            (
                "response.function_call_arguments.delta",
                {
                    "type": "response.function_call_arguments.delta",
                    "sequence_number": sequence,
                    "item_id": "fc_1",
                    "output_index": 0,
                    "delta": piece,
                },
            )
        )
        sequence += 1
    events += [
        (
            "response.function_call_arguments.done",
            {
                "type": "response.function_call_arguments.done",
                "sequence_number": sequence,
                "item_id": "fc_1",
                "output_index": 0,
                "name": name,
                "arguments": args,
            },
        ),
        (
            "response.output_item.done",
            {
                "type": "response.output_item.done",
                "sequence_number": sequence + 1,
                "output_index": 0,
                "item": {**item, "arguments": args, "status": "completed"},
            },
        ),
        (
            "response.completed",
            {
                "type": "response.completed",
                "sequence_number": sequence + 2,
                "response": responses_tool(name, args),
            },
        ),
    ]
    return sse(events)


# --- a callback handler that records what it is told -------------------------------------------


class Recorder(AsyncCallbackHandler):
    """Stands in for the budget guard and for stream consumers: it records each model run's start
    (with the messages and metadata it was given), its tokens, and how it ended."""

    raise_error = True
    run_inline = True

    def __init__(self) -> None:
        self.events: list[tuple[str, Any]] = []
        self.starts: list[tuple[Any, list[Any], dict[str, Any]]] = []
        self.tokens: list[tuple[Any, Any]] = []

    async def on_chat_model_start(
        self,
        serialized: Any,
        messages: Any,
        *,
        run_id: Any,
        metadata: Any = None,
        tags: Any = None,
        **kwargs: Any,
    ) -> None:
        self.events.append(("start", run_id))
        self.starts.append((run_id, messages, metadata or {}))

    async def on_llm_new_token(self, token: Any, *, run_id: Any, **kwargs: Any) -> None:
        self.tokens.append((run_id, token))

    async def on_llm_end(self, response: Any, *, run_id: Any, **kwargs: Any) -> None:
        self.events.append(("end", run_id))

    async def on_llm_error(self, error: BaseException, *, run_id: Any, **kwargs: Any) -> None:
        self.events.append(("error", run_id))

    def names(self) -> list[str]:
        return [name for name, _ in self.events]


# --- test data shared by the redaction tests ---------------------------------------------------
# Synthetic: example.com is reserved, 07700 900xxx is Ofcom's drama range, and the NINO has a prefix
# no one is given.

EMAIL = "jo.bloggs@example.com"
PHONE = "07700 900123"
NINO = "QQ123456C"
NAME = "Jo Bloggs"
SEEDS = [EMAIL, PHONE, NINO, NAME]
# A name with a quote and a backslash: putting it into JSON text unescaped would break the JSON.
TRICKY = 'Jo "JB" O\\Brien'

# Every place a personal value can sit in a conversation: system text, human text, an earlier
# answer, an earlier tool call's arguments and a tool result.
HISTORY = [
    SystemMessage(f"You help {NAME}. Reach them on {EMAIL}."),
    HumanMessage(f"My phone is {PHONE} and NI number {NINO}."),
    AIMessage(
        f"Thanks {NAME}.",
        tool_calls=[
            {
                "name": "lookup",
                "args": {"who": NAME, "mail": EMAIL},
                "id": "call_1",
                "type": "tool_call",
            }
        ],
    ),
    ToolMessage(f"found {NAME} at {PHONE}", tool_call_id="call_1"),
    HumanMessage("Summarise."),
]


def wrapped(provider: str) -> tuple[BaseChatModel, RedactingChatModel]:
    """(the provider model, a redacting wrapper around it) with the project's own detectors."""
    inner = raw_model(provider)
    return inner, RedactingChatModel(inner, Redactor(DETECTORS))


def echo(provider: str) -> Reply:
    """Answers with every placeholder it can find in the request, so restoring is exercised."""
    import re

    def reply(body: dict[str, Any]) -> Any:
        found = list(dict.fromkeys(re.findall(r"\[[A-Z]+_\d+\]", json.dumps(body))))
        return text_reply(provider, "Noted: " + " ".join(found))

    return reply


class Address(BaseModel):
    city: str
    line: str


class Contact(BaseModel):
    """A structured answer with a nested object, for the structured-output and agent tests."""

    name: str
    email: str
    address: Address


# The same schema as a plain dict (additionalProperties false, as strict mode needs).
DICT_SCHEMA: dict[str, Any] = {
    "title": "Contact",
    "description": "A contact.",
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "email": {"type": "string"},
        "address": {
            "type": "object",
            "properties": {"city": {"type": "string"}, "line": {"type": "string"}},
            "required": ["city", "line"],
            "additionalProperties": False,
        },
    },
    "required": ["name", "email", "address"],
    "additionalProperties": False,
}


class FakeLangSmithSession(requests.Session):
    """A requests session that records what a LangSmith client would send, as text, and answers
    200. Give it to `langsmith.Client(session=...)`: nothing leaves the machine."""

    def __init__(self) -> None:
        super().__init__()
        self.sent: list[str] = []

    def request(self, method: str, url: str, **kwargs: Any) -> requests.Response:  # type: ignore[override]
        data = kwargs.get("data")
        if data is None and kwargs.get("json") is not None:
            data = json.dumps(kwargs["json"])
        if isinstance(data, bytes):
            data = data.decode("utf-8", "replace")
        self.sent.append(f"{method} {url} {data or ''}")
        response = requests.Response()
        response.status_code = 200
        response._content = b"{}"
        return response

    @property
    def text(self) -> str:
        return "\n".join(self.sent)
