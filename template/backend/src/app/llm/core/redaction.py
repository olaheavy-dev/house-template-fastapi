"""Personal-data redaction, the same for every framework (#78, docs/decisions/0006).

Before text goes to a model provider, values that look like contact, bank or ID numbers, and the
exact values the app names, are replaced by placeholders such as [EMAIL_1]. After the answer comes
back, the placeholders are turned back into the originals, so the user sees their own details. This
is pseudonymisation, not anonymisation: the mapping is the key, it lives in memory for one call and
is never stored or logged, and the text that is sent can still be personal data. It is not
complete: it finds common formats and exact values, not every name, address or special-category
detail.

Pieces:

- Detector: finds spans of one kind in a text. The project's list is app/llm/redaction_rules.py.
  RegexDetector, luhn_valid, nhs_valid and context are building blocks for it.
- Scope: what the app tells us for this request. `known` are exact values to remove (the user's own
  name, address, ...), by kind; `allow` are values that must stay (the property's own address,
  say). A known value beats an allowed one. It is set with `async with redaction_scope(...)` and
  held in a context variable, never in call metadata (which a tracer would record).
- Redactor.redact(texts): replaces across the whole message list, so a number is fixed by the first
  appearance and stays the same as the history grows (a provider's prompt cache keeps matching).
  Any failure raises RedactionFailed, and nothing is sent: it always fails closed.
- restore / StreamRestorer: put the originals back into an answer, or into a stream of chunks.

Log events never carry a value: only a kind, a count or an error type name. An adapter calls the
Redactor on the outgoing messages before the budget reservation (app/llm/langchain/ for LangChain).
"""

from __future__ import annotations

import re
from bisect import bisect_left
from collections import Counter
from collections.abc import AsyncIterator, Callable, Iterable, Mapping, Sequence
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Protocol

import structlog

from app.problems import Problem

__all__ = [
    "Detector",
    "Redacted",
    "RedactionFailed",
    "Redactor",
    "RegexDetector",
    "Scope",
    "Span",
    "StreamRestorer",
    "context",
    "current_scope",
    "luhn_valid",
    "nhs_valid",
    "redaction_scope",
    "restore",
]

log = structlog.get_logger()

# [KIND_n]: capital letters, digits and underscores, ending in _digits.
_PLACEHOLDER = re.compile(r"\[[A-Z][A-Z0-9_]*_\d+\]")
_LITERAL = re.compile(r"\[([A-Z][A-Z0-9_]*)_(\d+)\]")
# What a placeholder is made of, for holding back the start of one in a stream.
_PLACEHOLDER_CHARS = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")


class RedactionFailed(Problem):
    """The text could not be prepared, so nothing was sent (503). It carries no value and no cause:
    the original error is not chained, because it may hold the text."""

    def __init__(self) -> None:
        super().__init__(
            503,
            detail="The request couldn't be prepared for the language model, so nothing was sent.",
        )


@dataclass(frozen=True, slots=True)
class Span:
    """The characters text[start:end] are a value of `kind`."""

    start: int
    end: int
    kind: str


class Detector(Protocol):
    kind: str

    def find(self, text: str) -> Iterable[Span]: ...


@dataclass(frozen=True, slots=True)
class Scope:
    known: Mapping[str, Sequence[str]] = field(default_factory=dict)
    allow: Sequence[str] = ()


_EMPTY = Scope()
_scope: ContextVar[Scope] = ContextVar("redaction_scope", default=_EMPTY)


def current_scope() -> Scope:
    return _scope.get()


@asynccontextmanager
async def redaction_scope(
    *, known: Mapping[str, Sequence[str]] | None = None, allow: Sequence[str] = ()
) -> AsyncIterator[Scope]:
    """Sets what the app knows for the calls made inside the block (not for other requests)."""
    scope = Scope(known=dict(known or {}), allow=list(allow))
    token = _scope.set(scope)
    try:
        yield scope
    finally:
        _scope.reset(token)


@dataclass(frozen=True, slots=True)
class Redacted:
    texts: tuple[str, ...]
    # placeholder -> original, in memory for one call. It is personal data: never store or log it.
    mapping: Mapping[str, str]
    # kind -> how many occurrences were replaced.
    counts: Mapping[str, int]


# --- building blocks for detectors ------------------------------------------------------------


def luhn_valid(number: str) -> bool:
    """The Luhn check of the digits in `number` (card numbers)."""
    digits = [int(c) for c in number if c.isdigit()]
    total = 0
    for index, digit in enumerate(reversed(digits)):
        if index % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return bool(digits) and total % 10 == 0


def nhs_valid(number: str) -> bool:
    """The modulus 11 check of a 10-digit NHS number: weights 10 down to 2 on the first nine
    digits; check digit = 11 - (sum mod 11), where 11 means 0 and 10 means invalid."""
    digits = [int(c) for c in number if c.isdigit()]
    if len(digits) != 10:
        return False
    total = sum(d * w for d, w in zip(digits[:9], range(10, 1, -1), strict=True))
    check = 11 - total % 11
    if check == 11:
        check = 0
    return check != 10 and check == digits[9]


def context(pattern: str) -> re.Pattern[str]:
    """A case-insensitive pattern for the words that must appear just before a value."""
    return re.compile(pattern, re.IGNORECASE)


class RegexDetector:
    """A pattern, optionally with a check on the matched text (a checksum) and a context pattern
    that must match within `window` characters before it. Write the pattern so it runs in linear
    time: no nested quantifiers, possessive quantifiers (Python 3.11 and later) and a lookbehind so
    a match can only start at the beginning of a run."""

    def __init__(
        self,
        kind: str,
        pattern: str | re.Pattern[str],
        *,
        check: Callable[[str], bool] | None = None,
        before: re.Pattern[str] | None = None,
        window: int = 30,
    ) -> None:
        self.kind = kind
        self._pattern = re.compile(pattern) if isinstance(pattern, str) else pattern
        self._check = check
        self._before = before
        self._window = window

    def find(self, text: str) -> Iterable[Span]:
        for match in self._pattern.finditer(text):
            if self._check is not None and not self._check(match.group()):
                continue
            if self._before is not None and not self._before.search(
                text[max(0, match.start() - self._window) : match.start()]
            ):
                continue
            yield Span(match.start(), match.end(), self.kind)


# --- the redactor -----------------------------------------------------------------------------


def _normal(value: str) -> str:
    return " ".join(value.split()).casefold()


def _value_pattern(value: str) -> re.Pattern[str] | None:
    """Matches `value` ignoring case and any run of whitespace, not inside a longer word."""
    words = value.split()
    if not words:
        return None
    body = r"\s+".join(re.escape(word) for word in words)
    return re.compile(rf"(?<!\w){body}(?!\w)", re.IGNORECASE)


def _union(spans: list[tuple[int, int]]) -> tuple[list[int], list[int]]:
    """Overlapping spans merged, as sorted starts and ends."""
    starts: list[int] = []
    ends: list[int] = []
    for start, end in sorted(spans):
        if ends and start < ends[-1]:
            ends[-1] = max(ends[-1], end)
        else:
            starts.append(start)
            ends.append(end)
    return starts, ends


def _touches(starts: list[int], ends: list[int], start: int, end: int) -> bool:
    at = bisect_left(starts, start)
    return (at > 0 and ends[at - 1] > start) or (at < len(starts) and starts[at] < end)


@dataclass(frozen=True, slots=True)
class _Found:
    span: Span
    known: bool  # a value the app named (wins over a detector)
    original: str  # what to put back
    key: str  # the same key, the same placeholder


class Redactor:
    def __init__(self, detectors: Sequence[Detector]) -> None:
        self._detectors = tuple(detectors)

    def redact(self, texts: Sequence[str], scope: Scope | None = None) -> Redacted:
        """Replaces the values in every text. The scope defaults to the current one. Raises
        RedactionFailed on any error, so the caller sends nothing."""
        failure: str | None = None
        try:
            return self._redact(texts, scope if scope is not None else current_scope())
        except Exception as exc:  # fail closed: a bug here must not let the text through
            failure = type(exc).__name__
        # Outside the except block, so the error (which may hold the text) is not chained.
        log.error("llm_redaction_failed", error_type=failure)
        raise RedactionFailed

    def _redact(self, texts: Sequence[str], scope: Scope) -> Redacted:
        allowed = [p for value in scope.allow if (p := _value_pattern(value)) is not None]
        named = [
            (kind, value, pattern)
            for kind, values in scope.known.items()
            for value in values
            if (pattern := _value_pattern(value)) is not None
        ]
        used = {(kind, int(number)) for text in texts for kind, number in _LITERAL.findall(text)}
        next_number: Counter[str] = Counter()
        placeholders: dict[tuple[str, str], str] = {}
        mapping: dict[str, str] = {}
        counts: Counter[str] = Counter()
        conflicts: set[str] = set()
        out: list[str] = []

        def placeholder(kind: str, key: str, original: str) -> str:
            existing = placeholders.get((kind, key))
            if existing is not None:
                return existing
            number = next_number[kind] + 1
            while (kind, number) in used:
                number += 1
            next_number[kind] = number
            name = f"[{kind}_{number}]"
            placeholders[(kind, key)] = name
            mapping[name] = original
            return name

        for text in texts:
            allow_starts, allow_ends = _union(
                [m.span() for pattern in allowed for m in pattern.finditer(text)]
            )
            found = [
                _Found(Span(m.start(), m.end(), kind), True, value, _normal(value))
                for kind, value, pattern in named
                for m in pattern.finditer(text)
            ]
            for detector in self._detectors:
                for span in detector.find(text):
                    found.append(
                        _Found(
                            span, False, text[span.start : span.end], text[span.start : span.end]
                        )
                    )
            # Known values first, then longer, then earlier; keep what does not overlap.
            found.sort(key=lambda f: (not f.known, f.span.start - f.span.end, f.span.start))
            kept: list[_Found] = []
            starts: list[int] = []  # of `kept`, which never overlap, so only neighbours can clash
            ends: list[int] = []
            for item in found:
                span = item.span
                at = bisect_left(starts, span.start)
                if (at > 0 and ends[at - 1] > span.start) or (
                    at < len(starts) and starts[at] < span.end
                ):
                    continue
                overlaps_allowed = _touches(allow_starts, allow_ends, span.start, span.end)
                if overlaps_allowed:
                    if not item.known:
                        continue  # the allow-list beats a detector
                    conflicts.add(span.kind)  # a known value beats the allow-list
                kept.insert(at, item)
                starts.insert(at, span.start)
                ends.insert(at, span.end)
            pieces: list[str] = []
            position = 0
            for item in kept:
                span = item.span
                pieces.append(text[position : span.start])
                pieces.append(placeholder(span.kind, item.key, item.original))
                counts[span.kind] += 1
                position = span.end
            pieces.append(text[position:])
            out.append("".join(pieces))
        for kind in sorted(conflicts):
            log.warning("redaction_allow_conflict", kind=kind)
        return Redacted(tuple(out), mapping, dict(counts))


def restore(text: str, mapping: Mapping[str, str]) -> tuple[str, int]:
    """Puts the originals back, in one pass. Returns the text and how many placeholder-shaped
    strings were not in the mapping (invented or mangled by the model, or literal in the input);
    those stay as they are."""
    unrestored = 0

    def put_back(match: re.Match[str]) -> str:
        nonlocal unrestored
        original = mapping.get(match.group())
        if original is None:
            unrestored += 1
            return match.group()
        return original

    return _PLACEHOLDER.sub(put_back, text), unrestored


class StreamRestorer:
    """Restores placeholders in streamed chunks. Text from an unclosed `[` is held back while it
    could still become a placeholder (only placeholder characters follow, and it is no longer than
    the longest placeholder), and released at the first character that rules that out or at
    flush()."""

    def __init__(self, mapping: Mapping[str, str]) -> None:
        self._mapping = mapping
        self._longest = max((len(name) for name in mapping), default=0)
        self._held = ""
        self.unrestored = 0

    def _out(self, text: str) -> str:
        restored, missed = restore(text, self._mapping)
        self.unrestored += missed
        return restored

    def feed(self, delta: str) -> str:
        buffer = self._held + delta
        start = buffer.rfind("[")
        if start != -1:
            tail = buffer[start + 1 :]
            if len(buffer) - start <= self._longest and all(c in _PLACEHOLDER_CHARS for c in tail):
                self._held = buffer[start:]
                return self._out(buffer[:start])
        self._held = ""
        return self._out(buffer)

    def flush(self) -> str:
        buffer, self._held = self._held, ""
        return self._out(buffer)
