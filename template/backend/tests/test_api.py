from __future__ import annotations

import httpx
import pytest
from fastapi import FastAPI

from app.api.pagination import Page, PageParamsDep, decode_cursor, encode_cursor, page
from app.auth import DEV_USER_HEADER

pytestmark = pytest.mark.anyio

ITEMS = list(range(1, 46))  # 45 rows, ids 1..45


async def test_me_is_versioned_and_protected(client: httpx.AsyncClient) -> None:
    assert (await client.get("/v1/me")).status_code == 401
    response = await client.get("/v1/me", headers={DEV_USER_HEADER: "user-7"})
    assert response.json() == {"id": "user-7"}


def add_numbers(app: FastAPI) -> None:
    @app.get("/numbers")
    async def numbers(params: PageParamsDep) -> Page[int]:
        after = params.after(int)
        rows = [n for n in ITEMS if after is None or n > after][: params.limit + 1]
        return page(rows, params, key=lambda n: n, item=lambda n: n)


async def test_pages_walk_every_item_once(app: FastAPI, client: httpx.AsyncClient) -> None:
    add_numbers(app)
    seen: list[int] = []
    cursor: str | None = None
    for _ in range(10):
        query: dict[str, str | int] = {"limit": 20}
        if cursor:
            query["cursor"] = cursor
        body = (await client.get("/numbers", params=query)).json()
        seen += body["items"]
        cursor = body["next_cursor"]
        if cursor is None:
            break
    assert seen == ITEMS


@pytest.mark.parametrize(
    "cursor", ["not base64!", encode_cursor("a string, not an id"), encode_cursor(True)]
)
async def test_a_bad_cursor_is_a_400(app: FastAPI, client: httpx.AsyncClient, cursor: str) -> None:
    add_numbers(app)
    response = await client.get("/numbers", params={"cursor": cursor})
    assert response.status_code == 400
    assert response.headers["content-type"] == "application/problem+json"


@pytest.mark.parametrize("limit", [0, 101])
async def test_limits_outside_1_to_100_are_refused(
    app: FastAPI, client: httpx.AsyncClient, limit: int
) -> None:
    add_numbers(app)
    assert (await client.get("/numbers", params={"limit": limit})).status_code == 422


@pytest.mark.parametrize("key", [42, "2026-10-26T12:00:00Z", ["2026-10-26", 7]])
def test_cursors_round_trip_any_json_key(key: object) -> None:
    assert decode_cursor(encode_cursor(key)) == key
