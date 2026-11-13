"""How full a stored conversation is, and which of its messages a trim would drop (#46). No model
call is made here: the count comes from what the provider reported for the conversation's last call
plus an over-estimate of what came after it, and the trim policy only chooses.

An adapter turns its framework's stored messages into `Entry`s (app/llm/langchain/context.py,
app/llm/maf/context.py) and asks `report` for the numbers:

- `tokens`: about how many tokens the next call would send. With usage (`measured`), it is the last
  call's input plus its output (the answer is now history), plus an estimate of anything stored
  after that answer: the call's input already held the system prompt and the tools. Without usage,
  every message is estimated, plus `fixed` (the system prompt and tool definitions the agent sends
  with every call, which the adapter sizes when it knows them) and INPUT_MARGIN_TOKENS (message
  framing). The estimate is an upper bound: one token per UTF-8 byte (byte-level tokenizers never
  use more) and IMAGE_TOKENS per image. Output includes reasoning tokens, which later calls may not
  resend. So `tokens` over-counts, as long as the adapter is given the agent's system prompt and
  tools.
- `bound`: the budget's own over-estimate of the same request, one token per byte (the stored
  messages, `fixed` and the margin): the figure the budget compares with the price's
  max_prompt_tokens before every call (app/llm/core/bounds.py check_prompt_size).
- `window`: the model's context window from its price entry (app/llm/prices.py), or None when the
  entry has none.
- `limit`: the most input a call can send: the window minus LLM_MAX_TOKENS (room for the answer).
- `prompt_limit`: the price's max_prompt_tokens, when it caps the prompt (OpenAI's entries).
- `percent`: the higher of `tokens` as a share of `limit` and `bound` as a share of
  `prompt_limit`, to one decimal place, so 100% is where calls start failing (founder decisions,
  13 Nov 2026): past the window the provider refuses, past the cap the budget does. It passes 100
  when a conversation has outgrown either. None when neither is known.
- `trimmed`: how many stored messages a trim has dropped (always 0 until trimming lands, #46 PR 2).

`plan_trim` is the trim policy, for #46 PR 2 to apply. Over its target, it drops in this order:
1. old tool calls, each with everything that belongs to it, oldest first;
2. if that's not enough, whole old turns, oldest first.
Messages belong together when they share a tool call id (`Entry.links`): the call, its results, and
an approval request and its answer. They are always kept or dropped together, so no result is left
without its call, however the framework orders them (MAF stores an approval's answer as a user
message between a call and its result). A turn starts at a user message that belongs to no tool
call. The policy never drops a system message, the user's first message (it often states the goal
of the whole conversation), the latest turn, or anything still waiting for a result or an approval.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from app.llm.core.bounds import IMAGE_TOKENS, INPUT_MARGIN_TOKENS, count_images
from app.llm.core.prices import Price
from app.llm.core.usage import Usage

__all__ = [
    "USAGE_KEY",
    "ContextReport",
    "Count",
    "Entry",
    "OutputTooLong",
    "Role",
    "TrimPlan",
    "check_output_limit",
    "count",
    "input_limit",
    "plan_trim",
    "report",
    "size_bound",
    "usage_from_record",
    "usage_record",
]

Role = Literal["system", "user", "assistant", "tool"]

# Where an adapter that has no usage field of its own (MAF) records a call's usage on the answer it
# stored: a key in the message's own properties, which are saved with it and never sent.
USAGE_KEY = "app.llm.usage"


@dataclass(frozen=True, slots=True, kw_only=True)
class Entry:
    """One stored message, as the policy and the counter see it.

    `size` is an upper bound of its tokens (size_bound). `links` are the tool call ids the message
    makes, answers, or asks or gives an approval for: messages that share one are a group. `waiting`
    marks a message whose tool calls have no result yet, or that asks for an approval not yet given.
    `usage` is what the provider reported for the call that wrote this message (an assistant
    message), when known."""

    role: Role
    size: int
    links: frozenset[str] = frozenset()
    waiting: bool = False
    usage: Usage | None = None


def size_bound(payload: object, *, media: object) -> int:
    """More tokens than `payload` can hold: its UTF-8 bytes as JSON, plus IMAGE_TOKENS for each
    image in `media` (an image sent by URL is only a few bytes in the payload). The budget's rule
    (app/llm/core/bounds.py request_bound), without its per-request margin. `media` is only the
    message content, as the budget looks for it: a tool call's arguments or a tool's schema may
    hold `"type": "document"` without being a document."""
    size = len(json.dumps(payload, default=str, ensure_ascii=False).encode())
    return size + count_images(media) * IMAGE_TOKENS


def usage_record(usage: Usage) -> dict[str, int]:
    """The part of a call's usage the counter needs, as plain data an adapter can store."""
    return {"input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens}


def _whole(value: object) -> int | None:
    """A whole, non-negative count, or None (a bool is not a count)."""
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def usage_from_record(record: object) -> Usage | None:
    """A Usage back from usage_record's data, or None when it isn't that (an older or edited
    document): the counter then estimates instead."""
    if not isinstance(record, Mapping):
        return None
    given, written = _whole(record.get("input_tokens")), _whole(record.get("output_tokens"))
    if given is None or written is None:
        return None
    return Usage(input_tokens=given, output_tokens=written)


@dataclass(frozen=True, slots=True)
class Count:
    tokens: int
    # The budget's over-estimate of the same request (one token per byte; module notes).
    bound: int
    # True when the count starts from a call's reported usage; False when it is all estimate.
    measured: bool


def count(entries: Sequence[Entry], *, fixed: int = 0) -> Count:
    """About how many tokens the next call would send for these stored messages, and the budget's
    bound of it (module notes). `fixed` sizes the system prompt and tools sent with every call."""
    bound = sum(entry.size for entry in entries) + fixed + INPUT_MARGIN_TOKENS
    for last in range(len(entries) - 1, -1, -1):
        usage = entries[last].usage
        if usage is not None:
            after = sum(entry.size for entry in entries[last + 1 :])
            tokens = usage.input_tokens + usage.output_tokens + after
            return Count(tokens, bound, measured=True)
    return Count(bound, bound, measured=False)


class OutputTooLong(ValueError):
    """LLM_MAX_TOKENS is more than the model may write: every call would be refused by the
    provider, so the gateway refuses to start instead."""


def check_output_limit(price: Price, max_tokens: int) -> None:
    """Refuse an output limit above the model's (its Window.max_output), at start-up."""
    if price.window is not None and max_tokens > price.window.max_output:
        raise OutputTooLong(
            f"LLM_MAX_TOKENS is {max_tokens}, but the model writes at most "
            f"{price.window.max_output} tokens (app/llm/prices.py)"
        )


def input_limit(price: Price, max_tokens: int) -> int | None:
    """The most input one call can send within the window, or None without a window: the window
    minus `max_tokens` (LLM_MAX_TOKENS, kept for the answer). check_output_limit keeps it above 0.
    The price's max_prompt_tokens is a separate line, which the budget compares with its byte bound
    (ContextReport.prompt_limit)."""
    if price.window is None:
        return None
    return price.window.tokens - max_tokens


@dataclass(frozen=True, slots=True)
class ContextReport:
    """How full a conversation is (the module notes say what each number means)."""

    tokens: int
    bound: int
    window: int | None
    limit: int | None
    prompt_limit: int | None
    percent: float | None
    trimmed: int
    measured: bool

    def as_dict(self) -> dict[str, Any]:
        """The report as plain data, for an endpoint's response."""
        return {
            "tokens": self.tokens,
            "bound": self.bound,
            "window": self.window,
            "limit": self.limit,
            "prompt_limit": self.prompt_limit,
            "percent": self.percent,
            "trimmed": self.trimmed,
            "measured": self.measured,
        }


def report(
    entries: Sequence[Entry],
    price: Price,
    max_tokens: int,
    *,
    fixed: int = 0,
    trimmed: int = 0,
) -> ContextReport:
    """The context report for a conversation's stored messages. `fixed` sizes the system prompt and
    the tools the agent sends with every call (size_bound of them), used when there is no usage."""
    counted = count(entries, fixed=fixed)
    limit = input_limit(price, max_tokens)
    cap = price.max_prompt_tokens
    shares = []
    if limit:
        shares.append(counted.tokens * 100 / limit)
    if cap:
        shares.append(counted.bound * 100 / cap)
    return ContextReport(
        tokens=counted.tokens,
        bound=counted.bound,
        window=price.window.tokens if price.window else None,
        limit=limit,
        prompt_limit=cap,
        percent=round(max(shares), 1) if shares else None,
        trimmed=trimmed,
        measured=counted.measured,
    )


@dataclass(frozen=True, slots=True)
class TrimPlan:
    """What a trim would drop: the entries' indices, the tokens left (the entries' sizes plus the
    fixed part), and whether that is at or under the target."""

    drop: frozenset[int]
    tokens: int
    fits: bool


def _groups(entries: Sequence[Entry]) -> list[list[int]]:
    """The entries in groups kept or dropped together, oldest first: entries that share a tool
    call id (directly or through another entry) are one group; every other entry is its own."""
    parent = list(range(len(entries)))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    first: dict[str, int] = {}
    for index, entry in enumerate(entries):
        for link in entry.links:
            if link in first:
                parent[root(index)] = root(first[link])
            else:
                first[link] = index
    groups: dict[int, list[int]] = {}
    for index in range(len(entries)):
        groups.setdefault(root(index), []).append(index)
    return sorted(groups.values())


def plan_trim(entries: Sequence[Entry], target: int, *, fixed: int = 0) -> TrimPlan:
    """Which entries to drop to bring `fixed` plus the entries' sizes down to `target` (the order
    and the rules are in the module notes). `fixed` is what can't be trimmed and isn't an entry
    (the system prompt and tools an agent sends with every call). Under the target nothing is
    dropped; if even the most the rules allow is not enough, `fits` is False."""
    total = fixed + sum(entry.size for entry in entries)
    if total <= target:
        return TrimPlan(frozenset(), total, fits=True)

    # A turn starts at a user message that belongs to no tool call (MAF stores an approval's
    # answer as a user message: it is part of its call's group, not a new turn).
    turns = [i for i, entry in enumerate(entries) if entry.role == "user" and not entry.links]
    first_user = turns[0] if turns else None
    latest_turn = turns[-1] if turns else 0

    def protected(group: list[int]) -> bool:
        return any(
            i >= latest_turn or i == first_user or entries[i].role == "system" or entries[i].waiting
            for i in group
        )

    droppable = [group for group in _groups(entries) if not protected(group)]
    dropped: set[int] = set()

    def drop(group: list[int]) -> None:
        nonlocal total
        dropped.update(group)
        total -= sum(entries[i].size for i in group)

    # 1. Old tool calls with everything that belongs to them, oldest first.
    for group in droppable:
        if total <= target:
            break
        if any(entries[i].links for i in group):
            drop(group)
    # 2. Whole old turns, oldest first (a turn runs from one turn start to the next; anything
    # stored before the first counts as one more turn, the oldest).
    starts = sorted({0, *turns, len(entries)})
    for start, end in itertools.pairwise(starts):
        if total <= target:
            break
        for group in droppable:
            if start <= group[0] < end and group[0] not in dropped:
                drop(group)
    return TrimPlan(frozenset(dropped), total, fits=total <= target)
