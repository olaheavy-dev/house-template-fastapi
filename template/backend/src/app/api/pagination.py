"""Cursor pagination for list endpoints: `?limit=20&cursor=...`, answered with
{"items": [...], "next_cursor": "..." | null}.

Keyset, not offset: the cursor holds the last row's sort key, so the next page starts after it.
Pages stay stable while rows are added, and deep pages cost no more than the first. Cursors are
opaque to clients (URL-safe base64 of JSON); a tampered one is a 400.
"""

from __future__ import annotations

import base64
import binascii
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Annotated, Any

from fastapi import Depends, Query
from pydantic import BaseModel

from app.problems import Problem

MAX_LIMIT = 100


class Page[T](BaseModel):
    items: list[T]
    next_cursor: str | None


class BadCursor(Problem):
    def __init__(self) -> None:
        super().__init__(400, detail="That page cursor isn't valid.")


def encode_cursor(key: Any) -> str:
    raw = json.dumps(key, separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def decode_cursor(cursor: str) -> Any:
    try:
        raw = base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4))
        return json.loads(raw)
    except (binascii.Error, UnicodeDecodeError, ValueError):
        raise BadCursor from None


@dataclass(frozen=True, slots=True)
class PageParams:
    limit: int
    cursor: str | None

    def after[K](self, kind: type[K]) -> K | None:
        """The previous page's last key, checked to be of the expected type."""
        if self.cursor is None:
            return None
        key = decode_cursor(self.cursor)
        if not isinstance(key, kind) or isinstance(key, bool):
            raise BadCursor
        return key


def page_params(
    limit: Annotated[int, Query(ge=1, le=MAX_LIMIT)] = 20,
    cursor: Annotated[str | None, Query(max_length=512)] = None,
) -> PageParams:
    return PageParams(limit=limit, cursor=cursor)


PageParamsDep = Annotated[PageParams, Depends(page_params)]


def page[R, T](
    rows: Sequence[R], params: PageParams, *, key: Callable[[R], Any], item: Callable[[R], T]
) -> Page[T]:
    """A page from `limit + 1` fetched rows: the extra row only says there is a next page."""
    shown = rows[: params.limit]
    more = len(rows) > params.limit
    return Page[T](
        items=[item(row) for row in shown],
        next_cursor=encode_cursor(key(shown[-1])) if more and shown else None,
    )
