"""The framework-neutral half of personal-data redaction (#78, docs/decisions/0006): walking the
strings of a request and putting the originals back into an answer.

`walk_data` and `walk_text` call a function on every string of plain data or of JSON text (with the
field name in front of it as context); `Restorer` puts the originals back into an answer's text,
transport strings and data. Nothing here knows a model framework: LangChain's wrapper
(app/llm/langchain/redaction.py) adds its message and content-block shapes on top, and a later
adapter does the same for its own. Standard library and pydantic only (lint-imports and a test
enforce it). The engine itself, the detectors and the fail-closed rules are in redaction.py.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterator, Mapping
from typing import Any

from pydantic import BaseModel

from app.llm.core.redaction import restore

__all__ = [
    "MEDIA",
    "RAW_KEYS",
    "STRUCTURAL",
    "UNTOUCHED",
    "Leaf",
    "Restorer",
    "escaped",
    "field_context",
    "opens_json",
    "walk_data",
    "walk_text",
]

# Content blocks that are not text: counted, and sent as they are.
MEDIA = frozenset(
    {"image", "image_url", "input_image", "audio", "input_audio", "file", "input_file", "video"}
)
# Blocks the provider checks or the model produced from placeholders: never changed.
UNTOUCHED = frozenset({"thinking", "redacted_thinking", "reasoning"})
# String values of a content block that are structure, not text.
STRUCTURAL = frozenset(
    {
        "type",
        "id",
        "signature",
        "data",
        "encrypted_content",
        "call_id",
        "tool_use_id",
        "name",
        "media_type",
        "mime_type",
        "role",
        "status",
        "format",
        "detail",
    }
)
# Keys whose string value is raw JSON text (a provider's argument string, a streamed fragment).
RAW_KEYS = frozenset({"arguments", "partial_json"})

# What the walkers call for each string: (the text, the field name in front of it) -> the text. The
# field name ("date of birth: ") is context for detection only; see Redactor.redact(contexts=).
Leaf = Callable[[str, str], str]

# --- walking the strings of a message ----------------------------------------------------------

_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
# A dict that pairs a label with a `value` ("label": "Date of birth", "value": "12/03/1980").
_LABEL_KEYS = ("label", "name", "field", "key", "title")


def field_context(key: Any) -> str:
    """A field name as words in front of its value: `date_of_birth`, `dateOfBirth` and
    `NHSNumber` become "date of birth: " and "NHS Number: ", so a rule that needs a word before the
    value ("date of birth", "account", "sort code", "NHS") finds it when the value sits in a field
    of that name."""
    if not isinstance(key, str) or not key:
        return ""
    words = _CAMEL.sub(" ", key).replace("_", " ").replace("-", " ").replace(".", " ")
    return " ".join(words.split())[-40:] + ": "


def _label_of(pairs: Any) -> str | None:
    """The text that labels a `value` in a dict of (key, value) pairs, if there is one."""
    for key, value in pairs:
        key = key.value if isinstance(key, _Str) else key
        text = value.value if isinstance(value, _Str) else value
        if key in _LABEL_KEYS and isinstance(text, str) and text.strip():
            return text
    return None


def _value_context(key: Any, label: str | None) -> str:
    return field_context(label) if key == "value" and label else field_context(key)


def walk_data(obj: Any, leaf: Leaf, context: str = "") -> Any:
    """Every string value in plain data (tool arguments, a parsed answer), keys untouched. A value
    carries its field name as context (a list item, its list's; a `value` its label's), and a
    string that is JSON text is walked as JSON text."""
    if isinstance(obj, str):
        return walk_text(obj, leaf, context)
    if isinstance(obj, dict):
        label = _label_of(obj.items())
        out: dict[Any, Any] = {}
        for key, value in obj.items():
            # The key is text too (an email or a phone number can be a key): the key first, then
            # its value, the same order the scanner uses for JSON text.
            new_key = leaf(key, context) if isinstance(key, str) else key
            out[new_key] = walk_data(value, leaf, _value_context(key, label))
        return out
    if isinstance(obj, list | tuple):
        return [walk_data(value, leaf, context) for value in obj]
    return obj


class _Str:
    """A JSON string token: where it is in the text, and what it says."""

    __slots__ = ("end", "start", "value")

    def __init__(self, start: int, end: int, value: str) -> None:
        self.start, self.end, self.value = start, end, value


class _Obj:
    __slots__ = ("members",)

    def __init__(self, members: list[tuple[_Str, Any]]) -> None:
        self.members = members  # (key token, value), in order, duplicate keys kept


class _Arr:
    __slots__ = ("items",)

    def __init__(self, items: list[Any]) -> None:
        self.items = items


def _scan(text: str) -> Any:
    """The structure of JSON text that has already been parsed by json.loads (so it is valid),
    with the position of every string token. A number, true, false or null is not kept: it is never
    changed, and the text around a replaced string is never rewritten."""
    position = 0

    def skip() -> None:
        nonlocal position
        while text[position] in " \t\r\n":
            position += 1

    def string() -> _Str:
        nonlocal position
        start = position
        position += 1
        while text[position] != '"':
            position += 2 if text[position] == "\\" else 1
        position += 1
        return _Str(start, position, json.loads(text[start:position]))

    def value() -> Any:
        nonlocal position
        skip()
        char = text[position]
        if char == '"':
            return string()
        if char == "{":
            position += 1
            members: list[tuple[_Str, Any]] = []
            skip()
            if text[position] == "}":
                position += 1
                return _Obj(members)
            while True:
                skip()
                key = string()
                skip()
                position += 1  # the colon
                members.append((key, value()))
                skip()
                char = text[position]
                position += 1
                if char == "}":
                    return _Obj(members)
        if char == "[":
            position += 1
            items: list[Any] = []
            skip()
            if text[position] == "]":
                position += 1
                return _Arr(items)
            while True:
                items.append(value())
                skip()
                char = text[position]
                position += 1
                if char == "]":
                    return _Arr(items)
        while position < len(text) and text[position] not in ",]} \t\r\n":
            position += 1  # a number or a literal
        return None

    return value()


def _visit(node: Any, leaf: Leaf, context: str, edits: list[tuple[int, int, str]]) -> None:
    if isinstance(node, _Str):
        new = walk_text(node.value, leaf, context)
        if new != node.value:
            edits.append((node.start, node.end, json.dumps(new, ensure_ascii=False)))
    elif isinstance(node, _Obj):
        label = _label_of(node.members)
        for key, child in node.members:
            changed = leaf(key.value, context)  # the key is text too, with its parent's context
            if changed != key.value:
                edits.append((key.start, key.end, json.dumps(changed, ensure_ascii=False)))
            _visit(child, leaf, _value_context(key.value, label), edits)
    elif isinstance(node, _Arr):
        for child in node.items:
            _visit(child, leaf, context, edits)


def walk_text(text: str, leaf: Leaf, context: str = "") -> str:
    """One string. JSON text (an object or array: an argument string, a tool result) is read, its
    string values are walked with their field names, and only those string tokens that changed are
    written again, so numbers, whitespace, key order and duplicate keys stay exactly as they were.
    Reading matters: a value with a quote or a backslash is escaped in JSON text and would not match
    as it is. JSON nested too deeply to read raises RecursionError, which prepare turns into a
    refusal. Anything else is one string."""
    if text.lstrip()[:1] in ("{", "["):
        try:
            parsed = json.loads(text)
        except ValueError:
            return leaf(text, context)
        if isinstance(parsed, dict | list):
            edits: list[tuple[int, int, str]] = []
            _visit(_scan(text), leaf, context, edits)
            # Edits are in document order (keys and values as the scanner meets them): one pass of
            # slices, so many replacements stay linear.
            pieces: list[str] = []
            position = 0
            for start, end, replacement in edits:
                pieces.append(text[position:start])
                pieces.append(replacement)
                position = end
            pieces.append(text[position:])
            return "".join(pieces)
    return leaf(text, context)


def _string_tokens(node: Any) -> Iterator[_Str]:
    """Every string token of a scanned JSON text, keys and values, in document order."""
    if isinstance(node, _Str):
        yield node
    elif isinstance(node, _Obj):
        for key, child in node.members:
            yield key
            yield from _string_tokens(child)
    elif isinstance(node, _Arr):
        for child in node.items:
            yield from _string_tokens(child)


# --- putting the originals back ----------------------------------------------------------------


def escaped(mapping: Mapping[str, str]) -> dict[str, str]:
    """The mapping with each original JSON-escaped, for text that is itself JSON: a quote, a
    backslash or a newline in an original must not break the document."""
    return {key: json.dumps(value, ensure_ascii=False)[1:-1] for key, value in mapping.items()}


_JSON_OPENING = re.compile(r"\s*(?:\{|\[\s*[^A-Za-z\s])")


def opens_json(text: str) -> bool:
    """The rule for text that is JSON without the call having asked for a response format, the same
    for a whole answer and for the first text of a stream. It is JSON when it opens with `{`, or
    with `[` followed (after any spaces) by something that is not a letter. A placeholder is `[`
    and a letter, so an answer that starts with one is plain text. Missed on purpose: an array
    that starts with a letter (`[true, ...]`, `[null]`). A lone `[` is undecided (a stream waits
    for the next text). A misfire, rare and accepted: prose that opens with a footnote marker
    (`[1]`, `[2]`, `[*`) is taken as JSON, so a later original holding `"`, `\\` or a newline is
    shown escaped."""
    return _JSON_OPENING.match(text) is not None


class Restorer:
    """Puts originals back into an answer. Made for one call; counts the placeholder-shaped
    strings it could not restore (invented or mangled by the model)."""

    def __init__(self, mapping: Mapping[str, str], *, json_text: bool = False) -> None:
        self.plain = mapping
        self.escaped = escaped(mapping)
        self.unrestored = 0
        self._json_text = json_text

    def text(self, text: str) -> str:
        restored, missed = restore(text, self.plain)
        self.unrestored += missed
        return restored

    def raw(self, text: str) -> str:
        """A provider's transport string (an argument string, a fragment of one, JSON-mode text).
        Whole JSON is restored string by string, at each nesting level; anything else (a cut-off
        fragment) by substituting the JSON-escaped originals."""
        restored = self._restore_json(text)
        if restored is not None:
            return restored
        restored, missed = restore(text, self.escaped)
        self.unrestored += missed
        return restored

    def string(self, text: str) -> str:
        """The decoded text of a string value. JSON text inside it is restored at its own level, so
        an original with a quote or a backslash does not break the document; anything else is plain
        text."""
        if text.lstrip()[:1] in ("{", "["):
            restored = self._restore_json(text)
            if restored is not None:
                return restored
        return self.text(text)

    def _restore_json(self, text: str) -> str | None:
        """`text` as JSON with the placeholders in its strings and keys restored, each string at its
        own level (a string that is itself JSON text is restored inside); None if it is not a JSON
        object or array, or too deeply nested to read."""
        if text.lstrip()[:1] not in ("{", "["):
            return None
        try:
            parsed = json.loads(text)
            if not isinstance(parsed, dict | list):
                return None
            edits: list[tuple[int, int, str]] = []
            for token in _string_tokens(_scan(text)):
                new = self.string(token.value)
                if new != token.value:
                    edits.append((token.start, token.end, json.dumps(new, ensure_ascii=False)))
        except (ValueError, RecursionError):
            return None
        pieces: list[str] = []
        position = 0
        for start, end, replacement in edits:
            pieces.append(text[position:start])
            pieces.append(replacement)
            position = end
        pieces.append(text[position:])
        return "".join(pieces)

    def answer(self, text: str) -> str:
        """Answer text: JSON text gets escaped originals. It is JSON when `json_text` is set (the
        call asked for JSON) or the answer opens with { or [ (opens_json), as in a stream."""
        return self.raw(text) if self._json_text or opens_json(text) else self.text(text)

    def data(self, obj: Any) -> Any:
        """Plain data (tool arguments, a parsed answer): every string restored as text, JSON text
        inside a string at its own level. No field name changes how: a field called `arguments`
        in a tool's own schema is data like any other."""
        if isinstance(obj, BaseModel):
            # A parsed schema object (OpenAI's json_schema): copy it with restored fields. Not
            # validated again: the provider's SDK validated the placeholder form.
            fields = {name: self.data(getattr(obj, name)) for name in type(obj).model_fields}
            return obj.model_copy(update=fields)
        if isinstance(obj, str):
            return self.string(obj)
        if isinstance(obj, dict):
            return {
                # A key can hold a placeholder too (the model copied one into a field name).
                (self.string(key) if isinstance(key, str) else key): self.data(value)
                for key, value in obj.items()
            }
        if isinstance(obj, list | tuple):
            return [self.data(value) for value in obj]
        return obj
