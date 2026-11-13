"""What a call may cost before it is sent: the most input tokens its request can hold, the most
output tokens it may generate, and the requests the budget refuses because it can't bound or
price them. Framework-neutral: an adapter passes its request as plain data (app/llm/langchain/
callbacks.py for LangChain).

Images are reserved at IMAGE_TOKENS each, however they are sent. Documents, audio and video have
no per-item bound (a PDF costs per page), so a request carrying one is refused before it is sent.

app/llm/prices.py prices tokens at the standard tier only, so a request that would be billed for
more is refused before it is sent: one binding a tool the provider runs itself (web search, file
search, code execution: billed per use on top of tokens), or asking for a dearer service tier
(priority).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from app.llm.core.prices import Price
from app.problems import Problem

__all__ = [
    "IMAGE_TOKENS",
    "INPUT_MARGIN_TOKENS",
    "PromptTooLong",
    "UnboundedMedia",
    "UnpricedRequest",
    "check_priced",
    "check_prompt_size",
    "count_images",
    "output_bound",
    "request_bound",
]

# Added to every reservation's input: message framing, and the system prompt a provider adds
# when tools are bound (Anthropic documents a few hundred tokens).
INPUT_MARGIN_TOKENS = 1024
# Per image, above Anthropic's documented maximum of 4,784 (high-resolution models, vision docs,
# checked 26 Oct 2026). Check the provider's image pricing before sending images to another one.
IMAGE_TOKENS = 6_000
_IMAGE_BLOCKS = {"image", "image_url", "input_image"}
_UNBOUNDED_BLOCKS = {"document", "file", "input_file", "audio", "input_audio", "video"}
# Tools the app runs itself, which cost only the tokens of the call: OpenAI's "function" and
# "custom" tools, and Anthropic's untyped ones. Any other type is a tool the provider runs.
_TOKEN_ONLY_TOOL_TYPES = frozenset({None, "function", "custom"})
# Service tiers billed at prices.py's standard rates or less (flex is cheaper, which only ever
# over-records).
_PRICED_SERVICE_TIERS = frozenset({None, "auto", "default", "flex", "standard_only"})
# The request parameters that can raise the output limit above the gateway's.
_OUTPUT_LIMITS = ("max_tokens", "max_completion_tokens")


class PromptTooLong(Problem):
    def __init__(self) -> None:
        super().__init__(422, detail="The input is too long for the language model.")


class UnboundedMedia(Problem):
    """A message carries content whose token cost the budget can't bound in advance."""

    def __init__(self, kind: str) -> None:
        super().__init__(422, detail="This kind of attachment can't be sent to the language model.")
        self.kind = kind


class UnpricedRequest(ValueError):
    """A call that would be billed for something prices.py doesn't price: a mistake in the code
    that made it, refused before anything is reserved or sent."""


def count_images(content: object) -> int:
    """Image blocks anywhere in message content, nested ones too (tool results can hold them).
    Content is a provider's wire shape: a string, a list of blocks, a block as a dict. Raises
    UnboundedMedia for a document, file, audio or video block."""
    if isinstance(content, list):
        return sum(count_images(item) for item in content)
    if not isinstance(content, dict):
        return 0
    kind = content.get("type")
    if kind in _IMAGE_BLOCKS:
        return 1
    if kind in _UNBOUNDED_BLOCKS:
        raise UnboundedMedia(str(kind))
    return sum(count_images(item) for item in content.values())


def request_bound(payload: object, content: object) -> int:
    """More input tokens than a request can hold: the UTF-8 bytes of `payload` as JSON (byte-level
    BPE tokenizers never use more than one token per byte), IMAGE_TOKENS per image in `content`
    (an image sent by URL or file id is only a few bytes in the payload), plus a margin. `payload`
    is everything sent (messages, tools, schemas); `content` the messages' content blocks."""
    size = len(json.dumps(payload, default=str, ensure_ascii=False).encode())
    return size + count_images(content) * IMAGE_TOKENS + INPUT_MARGIN_TOKENS


def output_bound(limit: int, params: Mapping[str, Any]) -> int:
    """The most output tokens to reserve for: the gateway's limit, or a larger one the caller
    bound (max_tokens or max_completion_tokens; a value that is not an integer is ignored)."""
    bound = [params.get(key) for key in _OUTPUT_LIMITS]
    return max([limit, *(b for b in bound if isinstance(b, int))])


def check_prompt_size(price: Price, input_tokens: int) -> None:
    """Refuse a prompt longer than its price covers (Price.max_prompt_tokens). `input_tokens` is
    an over-estimate (request_bound), so a prompt near the limit may be refused, never one past it
    sent and under-recorded."""
    # The context report (app/llm/core/context.py, #46) shows this same bound as a share of
    # max_prompt_tokens, so its 100% is where this refuses.
    if price.max_prompt_tokens and input_tokens > price.max_prompt_tokens:
        raise PromptTooLong


def check_priced(invocation_params: Mapping[str, Any]) -> None:
    """Refuse a request billed beyond tokens: a tool the provider runs, or a dearer service tier."""
    for tool in invocation_params.get("tools") or ():
        kind = tool.get("type") if isinstance(tool, dict) else None
        if kind not in _TOKEN_ONLY_TOOL_TYPES:
            raise UnpricedRequest(
                f"The provider runs the {kind!r} tool and bills it per use, which "
                "app/llm/prices.py doesn't price: use a tool the app runs instead."
            )
    tier = invocation_params.get("service_tier")
    if tier not in _PRICED_SERVICE_TIERS:
        raise UnpricedRequest(
            f"The {tier!r} service tier costs more than the standard rates app/llm/prices.py "
            "holds: add its prices before using it."
        )
