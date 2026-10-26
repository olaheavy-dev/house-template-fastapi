from __future__ import annotations

from collections.abc import Iterator

import httpx
import pytest
from asgi_lifespan import LifespanManager
from fastapi import FastAPI
from pydantic import SecretStr

from app.main import create_app
from app.settings import Settings
from app.storage import InvalidStorageKey, ObjectNotFound, Storage, StorageDep, check_key

pytestmark = pytest.mark.anyio


@pytest.fixture
def store(storage_settings: Settings) -> Iterator[Storage]:
    storage = Storage(storage_settings)
    yield storage
    storage.close()


async def test_objects_round_trip(store: Storage) -> None:
    await store.put("notes/a.txt", b"hello", content_type="text/plain")
    assert await store.get("notes/a.txt") == b"hello"
    await store.delete("notes/a.txt")
    with pytest.raises(ObjectNotFound):
        await store.get("notes/a.txt")


async def test_presigned_urls_upload_and_download_without_the_api(store: Storage) -> None:
    put = store.presigned_put("uploads/report.csv", content_type="text/csv")
    async with httpx.AsyncClient() as http:
        uploaded = await http.put(put, content=b"a,b\n1,2\n", headers={"Content-Type": "text/csv"})
        assert uploaded.status_code == 200
        downloaded = await http.get(store.presigned_get("uploads/report.csv"))
    assert downloaded.status_code == 200
    assert downloaded.content == b"a,b\n1,2\n"


async def test_a_presigned_upload_only_takes_the_signed_content_type(store: Storage) -> None:
    put = store.presigned_put("uploads/photo.png", content_type="image/png")
    async with httpx.AsyncClient() as http:
        refused = await http.put(put, content=b"<html>", headers={"Content-Type": "text/html"})
    assert refused.status_code == 403


async def test_presigned_urls_are_signed_for_the_public_endpoint(
    storage_settings: Settings,
) -> None:
    # The API reaches the storage at one address and browsers at another (Compose: storage:9000
    # and localhost:9000). Here: localhost for the API, 127.0.0.1 for the "browser".
    internal = storage_settings.storage_endpoint_url
    assert internal is not None
    public = internal.replace("localhost", "127.0.0.1")
    assert public != internal, "the test storage should be addressed as localhost"
    store = Storage(storage_settings.model_copy(update={"storage_public_endpoint_url": public}))
    try:
        await store.put("a.txt", b"hi", content_type="text/plain")
        url = store.presigned_get("a.txt")
        assert url.startswith(f"{public}/")
        async with httpx.AsyncClient() as http:
            assert (await http.get(url)).content == b"hi"
            # The same URL sent to the other host fails: the host is part of the signature.
            assert (await http.get(url.replace(public, internal))).status_code == 403
    finally:
        store.close()


@pytest.mark.parametrize(
    "key", ["", "/abs", "a/../b", "..", "./a", "a/./b", "line\nbreak", "x" * 1025, "é" * 513]
)
def test_unsafe_keys_are_refused(key: str) -> None:
    with pytest.raises(InvalidStorageKey):
        check_key(key)


@pytest.mark.parametrize("key", ["a", "uploads/2026/report.csv", "a..b", ".hidden", "x" * 1024])
def test_ordinary_keys_are_accepted(key: str) -> None:
    assert check_key(key) == key


async def test_ready_checks_the_storage_when_it_is_configured(
    storage_settings: Settings, database_url: str
) -> None:
    app = create_app(storage_settings.model_copy(update={"database_url": SecretStr(database_url)}))
    async with LifespanManager(app) as manager:
        transport = httpx.ASGITransport(app=manager.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            assert (await client.get("/health/ready")).status_code == 200


async def test_ready_fails_when_the_storage_is_unreachable(storage_settings: Settings) -> None:
    unreachable = storage_settings.model_copy(update={"storage_endpoint_url": "http://127.0.0.1:1"})
    app = create_app(unreachable)
    async with LifespanManager(app) as manager:
        transport = httpx.ASGITransport(app=manager.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/health/ready")
    assert response.status_code == 503
    assert "storage" in response.json()["failing"]


async def test_without_a_bucket_there_is_no_storage_and_no_readiness_check(
    app: FastAPI, client: httpx.AsyncClient
) -> None:
    @app.get("/files")
    async def files(storage: StorageDep) -> str:
        return storage.bucket

    response = await client.get("/files")
    assert response.status_code == 503
    assert response.json()["detail"] == "File storage isn't configured."
    assert "storage" not in app.state.readiness_checks
