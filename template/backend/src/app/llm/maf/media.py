"""Files are stored as pointers, never their bytes (#73, record 0007): the one check the session
store and the checkpoint store share.

MAF keeps a file's bytes in a content's `uri` as a data URI: `Content.from_data` and
`Content.from_uri("data:...")` become a data content, and "DATA:..." or " data:..." stay URI
contents with the same bytes (measured on agent-framework-core 1.25.0). So the test is any `uri`
that is a data URI, wherever it sits: in a session's serialized state, or in a checkpoint's live
objects (messages, agent responses, executor state), which MAF would pickle as they are.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from typing import Any

__all__ = ["InlineMedia", "is_data_uri", "refuse_inline_media"]

_PLAIN = (str, bytes, int, float, bool, type(None))


class InlineMedia(ValueError):
    """A file's bytes (a data URI) where only a pointer may be stored: store the file and send its
    URI."""


def is_data_uri(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower().startswith("data:")


def _refuse() -> InlineMedia:
    return InlineMedia(
        "a session or checkpoint must not store a file's bytes: store the file and send its URI"
        " (#73)"
    )


def refuse_inline_media(value: Any) -> None:
    """Raise InlineMedia if `value` holds a data URI anywhere: in plain data (dicts, lists) or in
    objects (a MAF Message or Content, a dataclass, anything with `to_dict` or attributes)."""
    seen: set[int] = set()
    stack: list[Any] = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, _PLAIN):
            continue
        if id(item) in seen:
            continue
        seen.add(id(item))
        if isinstance(item, Mapping):
            if is_data_uri(item.get("uri")):
                raise _refuse()
            stack.extend(item.values())
        elif isinstance(item, list | tuple | set | frozenset):
            stack.extend(item)
        else:
            if is_data_uri(getattr(item, "uri", None)):
                raise _refuse()
            to_dict = getattr(item, "to_dict", None)
            if callable(to_dict):
                try:
                    stack.append(to_dict())
                    continue
                except Exception:  # noqa: S110 - then its attributes are walked instead
                    pass
            if is_dataclass(item) and not isinstance(item, type):
                stack.extend(getattr(item, f.name) for f in fields(item))
            elif hasattr(item, "__dict__"):
                stack.extend(vars(item).values())
