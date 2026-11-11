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
- Redactor.redact_each(texts): one-way, for text whose answer is not text (embeddings): each text
  is numbered on its own, and no mapping is kept, so there is nothing to restore.
- restore / StreamRestorer: put the originals back into an answer, or into a stream of chunks.

Log events never carry a value: only a kind, a count or an error type name. An adapter calls the
Redactor on the outgoing messages before the budget reservation (app/llm/langchain/ for LangChain).
"""

from __future__ import annotations

import json
import re
import unicodedata
from bisect import bisect_left, bisect_right
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
    "RedactedEach",
    "RedactionFailed",
    "Redactor",
    "RegexDetector",
    "Scope",
    "Span",
    "StreamRestorer",
    "choose_detectors",
    "context",
    "current_scope",
    "luhn_valid",
    "nhs_valid",
    "redaction_scope",
    "restore",
]

log = structlog.get_logger()

# A placeholder is [KIND_n]: capital letters, digits and underscores, ending in _digits. KIND is
# validated against _KIND wherever it comes in, so what we write is always matched on the way back.
_KIND = re.compile(r"[A-Z][A-Z0-9_]{0,48}")
# What is counted as placeholder-shaped in an answer: ours, and the ways a model mangles one (any
# case, a space or hyphen for the underscore, a space inside the brackets). A shape that is not in
# the mapping is not restored and is counted. Bounded, so it runs in linear time.
_PLACEHOLDER = re.compile(r"\[ ?[A-Za-z][A-Za-z0-9_]{0,48}[ _-]\d{1,9} ?\]")
# Every prefix of _PLACEHOLDER (and a few more), to hold back the start of one in a stream.
_PLACEHOLDER_PREFIX = re.compile(r" ?(?:[A-Za-z][A-Za-z0-9_]{0,48}(?:[ _-]\d{0,9} ?)?)?")
# The longest text _PLACEHOLDER can match is 64 characters.
_HOLD_MAX = 70
# A placeholder already written in the input, in exactly our form.
_LITERAL = re.compile(r"\[([A-Z][A-Z0-9_]*)_(\d+)\]")


def _check_kind(kind: object) -> str:
    if not isinstance(kind, str) or _KIND.fullmatch(kind) is None:
        raise ValueError(
            "a kind must be capital letters, digits and underscores, starting with a letter"
        )
    return kind


def _check_strings(values: object, what: str) -> list[str]:
    """`values` as a list of str: a single str is refused, not read one character at a time."""
    if isinstance(values, str) or not isinstance(values, Iterable):
        raise TypeError(f"{what} must be a list of str, not {type(values).__name__}")
    items = list(values)
    if not all(isinstance(item, str) for item in items):
        raise TypeError(f"{what} must hold only str")
    return items


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
    """What the app tells us for one request. `known` maps a kind to exact values to remove; `allow`
    lists values that must stay. Both are checked when the scope is made: a kind must be capital
    letters, digits and underscores, and a value list must be a list of str (not one str)."""

    known: Mapping[str, Sequence[str]] = field(default_factory=dict)
    allow: Sequence[str] = ()

    def __post_init__(self) -> None:
        for kind, values in self.known.items():
            _check_kind(kind)
            _check_strings(values, f"known values for {kind}")
        _check_strings(self.allow, "allow")


_EMPTY = Scope()
_scope: ContextVar[Scope] = ContextVar("redaction_scope", default=_EMPTY)


def current_scope() -> Scope:
    return _scope.get()


@asynccontextmanager
async def redaction_scope(
    *, known: Mapping[str, Sequence[str]] | None = None, allow: Sequence[str] = ()
) -> AsyncIterator[Scope]:
    """Sets what the app knows for the calls made inside the block (not for other requests)."""
    copied: dict[str, list[str]] = {}
    for kind, values in (known or {}).items():
        # The kind first: its name goes into the message below, and a bad one may be a value.
        name = _check_kind(kind)
        copied[name] = _check_strings(values, f"known values for {name}")
    scope = Scope(known=copied, allow=_check_strings(allow, "allow"))
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


@dataclass(frozen=True, slots=True)
class RedactedEach:
    """The result of Redactor.redact_each: one-way, so there is no mapping."""

    texts: tuple[str, ...]
    # How many values were replaced in each text, in the same order.
    counts: tuple[int, ...]
    # kind -> how many occurrences were replaced over all the texts.
    kinds: Mapping[str, int]


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


def choose_detectors(detectors: Sequence[Detector], kinds: str) -> tuple[Detector, ...]:
    """The detectors whose kind is in `kinds`, a comma separated list such as LLM_REDACT_KINDS.
    Raises ValueError for a name that no detector has, so a typo stops the app starting instead of
    quietly redacting less. An empty list chooses none (only known values are replaced)."""
    wanted = [name.strip() for name in kinds.split(",") if name.strip()]
    available = {detector.kind for detector in detectors}
    unknown = sorted({name for name in wanted if name not in available})
    if unknown:
        raise ValueError(
            "LLM_REDACT_KINDS names " + ", ".join(unknown) + ", which no detector in "
            "app/llm/redaction_rules.py has. Available: " + ", ".join(sorted(available))
        )
    return tuple(detector for detector in detectors if detector.kind in wanted)


# --- the redactor -----------------------------------------------------------------------------


def _normal(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).split()).casefold()


def _value_patterns(value: str) -> list[re.Pattern[str]]:
    """Patterns matching `value` ignoring case and any run of whitespace, not inside a longer
    word, written in either Unicode normal form (NFC, "é" as one character, or NFD, "e" and a
    combining accent), and also as it is written inside JSON text (a quote, a backslash or a
    control character escaped, non-ASCII as \\uXXXX or as it is), so a known value is found in a
    JSON document that was not decoded. None for a value of only whitespace."""
    bodies: dict[str, None] = {}  # in order, without repeats (an ASCII value has one form)
    forms = [value]
    for ensure_ascii in (False, True):
        escaped = json.dumps(value, ensure_ascii=ensure_ascii)[1:-1]
        if escaped != value:
            forms.append(escaped)
    for text in forms:
        for form in ("NFC", "NFD"):
            words = unicodedata.normalize(form, text).split()
            if not words:
                return []
            bodies[r"\s+".join(re.escape(word) for word in words)] = None
    return [re.compile(rf"(?<!\w){body}(?!\w)", re.IGNORECASE) for body in bodies]


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


def _contains(starts: list[int], ends: list[int], start: int, end: int) -> bool:
    """True when one merged span covers all of start..end."""
    at = bisect_right(starts, start) - 1
    return at >= 0 and ends[at] >= end


def _touches(starts: list[int], ends: list[int], start: int, end: int) -> bool:
    at = bisect_left(starts, start)
    return (at > 0 and ends[at - 1] > start) or (at < len(starts) and starts[at] < end)


_Compiled = tuple[
    list[re.Pattern[str]], list[tuple[str, str, re.Pattern[str]]]
]  # the allow patterns, and the known values with theirs


def _compile(scope: Scope) -> _Compiled:
    allowed = [p for value in scope.allow for p in _value_patterns(value)]
    named = [
        (kind, value, pattern)
        for kind, values in scope.known.items()
        for value in values
        for pattern in _value_patterns(value)
    ]
    return allowed, named


@dataclass(frozen=True, slots=True)
class _Found:
    span: Span
    known: bool  # a value the app named
    original: str  # for a known value: what to put back, as the app gave it
    key: str  # known: "k" + its normal form; see _merged for the detected form


class Redactor:
    """Replaces personal data with placeholders.

    Spans that overlap are replaced as one: the union of all of them, so a known value that covers
    only part of a detected number never leaves the rest of it behind. The union takes the kind of
    its longest member (a known value on a tie), and when one known value covers all of it, that
    value's own placeholder and form. An allowed value only keeps a detected span that it contains
    whole; it never splits one. A known value beats an allowed one."""

    def __init__(self, detectors: Sequence[Detector]) -> None:
        self._detectors = tuple(detectors)
        for detector in self._detectors:
            _check_kind(detector.kind)

    def redact(
        self,
        texts: Sequence[str],
        scope: Scope | None = None,
        *,
        log_conflicts: bool = True,
        contexts: Sequence[str] | None = None,
    ) -> Redacted:
        """Replaces the values in every text. The scope defaults to the current one. Raises
        TypeError for a single str (pass a list), and RedactionFailed on any error while
        redacting, so the caller sends nothing. `log_conflicts=False` skips the
        redaction_allow_conflict warning, for a caller that redacts many payloads under one scope
        (trace masking) and would repeat it for each. `contexts`, one string per text, is text that
        stands in front of that text for detection only (a field name such as "date of birth: "),
        so a rule that needs a word before the value can see it; only a match inside the text
        itself is replaced, and the context is never part of the result."""
        items = _check_strings(texts, "texts")
        prefixes = _check_strings(contexts, "contexts") if contexts is not None else None
        failure: str | None = None
        try:
            if prefixes is not None and len(prefixes) != len(items):
                raise ValueError("one context for each text")
            return self._redact(
                items, scope if scope is not None else current_scope(), log_conflicts, prefixes
            )
        except Exception as exc:  # fail closed: a bug here must not let the text through
            failure = type(exc).__name__
        # Outside the except block, so the error (which may hold the text) is not chained.
        log.error("llm_redaction_failed", error_type=failure)
        raise RedactionFailed

    def redact_each(self, texts: Sequence[str], scope: Scope | None = None) -> RedactedEach:
        """Replaces the values in each text on its own, for text that is sent one way (embeddings):
        placeholders are numbered inside each text, so what a text becomes depends only on the text
        and the scope, not on the texts around it. The scope's patterns are built once for the
        list, and an allow-list conflict is logged once. No mapping is kept. Raises TypeError for a
        single str, and RedactionFailed if any text fails: the whole list fails closed."""
        items = _check_strings(texts, "texts")
        failure: str | None = None
        try:
            active = scope if scope is not None else current_scope()
            compiled = _compile(active)
            conflicts: set[str] = set()
            results = [
                self._redact([text], active, False, None, compiled=compiled, conflicts=conflicts)
                for text in items
            ]
            for kind in sorted(conflicts):
                log.warning("redaction_allow_conflict", kind=kind)
            kinds: Counter[str] = Counter()
            for result in results:
                kinds.update(result.counts)
            return RedactedEach(
                texts=tuple(result.texts[0] for result in results),
                counts=tuple(sum(result.counts.values()) for result in results),
                kinds=dict(kinds),
            )
        except Exception as exc:  # fail closed, as redact() does
            failure = type(exc).__name__
        log.error("llm_redaction_failed", error_type=failure)
        raise RedactionFailed

    def _redact(
        self,
        texts: list[str],
        scope: Scope,
        log_conflicts: bool = True,
        contexts: list[str] | None = None,
        *,
        compiled: _Compiled | None = None,
        conflicts: set[str] | None = None,
    ) -> Redacted:
        allowed, named = compiled if compiled is not None else _compile(scope)
        used = {(kind, int(number)) for text in texts for kind, number in _LITERAL.findall(text)}
        next_number: Counter[str] = Counter()
        placeholders: dict[tuple[str, str], str] = {}
        mapping: dict[str, str] = {}
        counts: Counter[str] = Counter()
        found_conflicts: set[str] = conflicts if conflicts is not None else set()
        checked_kinds: set[str] = set()
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

        for number, text in enumerate(texts):
            prefix = contexts[number] if contexts is not None else ""
            seen = prefix + text  # detectors see the field name too; only the value is replaced
            allow_starts, allow_ends = _union(
                [m.span() for pattern in allowed for m in pattern.finditer(text)]
            )
            found = [
                _Found(Span(m.start(), m.end(), kind), True, value, "k" + _normal(value))
                for kind, value, pattern in named
                for m in pattern.finditer(text)
            ]
            known_starts, known_ends = _union([(i.span.start, i.span.end) for i in found])
            for item in found:  # a known value beats the allow-list
                if _touches(allow_starts, allow_ends, item.span.start, item.span.end):
                    found_conflicts.add(item.span.kind)
            for detector in self._detectors:
                for found_span in detector.find(seen):
                    if (
                        not isinstance(found_span, Span)
                        or not isinstance(found_span.start, int)
                        or not isinstance(found_span.end, int)
                        or not 0 <= found_span.start < found_span.end <= len(seen)
                    ):
                        raise ValueError("a detector returned a span outside the text")
                    if found_span.start < len(prefix):
                        continue  # a match that begins in the field name is not the value
                    span = Span(
                        found_span.start - len(prefix),
                        found_span.end - len(prefix),
                        found_span.kind,
                    )
                    if span.kind not in checked_kinds:
                        checked_kinds.add(_check_kind(span.kind))
                    if _contains(allow_starts, allow_ends, span.start, span.end) and not _touches(
                        known_starts, known_ends, span.start, span.end
                    ):
                        # The allow-list keeps a value it contains whole, unless a known value
                        # overlaps it: then it stays in, and is replaced with the known value.
                        continue
                    found.append(_Found(span, False, "", ""))
            found.sort(key=lambda f: (f.span.start, -f.span.end))
            pieces: list[str] = []
            position = 0
            index = 0
            while index < len(found):
                group = [found[index]]
                end = found[index].span.end
                index += 1
                while index < len(found) and found[index].span.start < end:
                    group.append(found[index])
                    end = max(end, found[index].span.end)
                    index += 1
                start = group[0].span.start
                kind, key, original = self._merged(text, start, end, group)
                pieces.append(text[position:start])
                pieces.append(placeholder(kind, key, original))
                counts[kind] += 1
                position = end
            pieces.append(text[position:])
            out.append("".join(pieces))
        for kind in sorted(found_conflicts) if log_conflicts else ():
            log.warning("redaction_allow_conflict", kind=kind)
        return Redacted(tuple(out), mapping, dict(counts))

    @staticmethod
    def _merged(text: str, start: int, end: int, group: list[_Found]) -> tuple[str, str, str]:
        """(kind, key, original) for the union of the overlapping spans in `group`."""
        for item in group:
            if item.known and item.span.start == start and item.span.end == end:
                return item.span.kind, item.key, item.original  # one known value covers it all
        longest = max(group, key=lambda f: (f.span.end - f.span.start, f.known))
        covered = text[start:end]
        return longest.span.kind, "d" + covered, covered  # "d": detected, written exactly


def restore(text: str, mapping: Mapping[str, str]) -> tuple[str, int]:
    """Puts the originals back, in one pass. Returns the text and how many placeholder-shaped
    strings were not in the mapping (invented by the model, or mangled: another case, a space or
    hyphen for the underscore, spaces inside the brackets, or literal in the input); those stay as
    they are. Restore parsed values, not raw JSON: an original holding a quote or a backslash makes
    invalid JSON when put into a tool call's argument string, so an adapter restores each parsed
    argument value instead."""
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
    """Restores placeholders in streamed chunks, with the same result and the same `unrestored`
    count as restore() on the joined text, whatever the chunking. Text from an unclosed `[` is held
    back while it could still become a placeholder-shaped string, and released at the first
    character that rules that out, when it is too long to be one, or at flush()."""

    def __init__(self, mapping: Mapping[str, str]) -> None:
        self._mapping = mapping
        self._held = ""
        self.unrestored = 0

    def _out(self, text: str) -> str:
        restored, missed = restore(text, self._mapping)
        self.unrestored += missed
        return restored

    def feed(self, delta: str) -> str:
        buffer = self._held + delta
        start = buffer.rfind("[")
        if (
            start != -1
            and len(buffer) - start <= _HOLD_MAX
            and _PLACEHOLDER_PREFIX.fullmatch(buffer, start + 1) is not None
        ):
            self._held = buffer[start:]
            return self._out(buffer[:start])
        self._held = ""
        return self._out(buffer)

    def flush(self) -> str:
        buffer, self._held = self._held, ""
        return self._out(buffer)
