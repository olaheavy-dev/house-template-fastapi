from __future__ import annotations

import json

import httpx
import pytest

from app.main import create_app
from app.settings import Settings

pytestmark = pytest.mark.anyio

PRODUCTION_DB = "postgresql+psycopg://app:secret@db.internal:5432/app"


async def test_request_id_is_echoed_when_safe_and_replaced_when_not(
    client: httpx.AsyncClient,
) -> None:
    kept = await client.get("/health/live", headers={"X-Request-ID": "abc-123"})
    assert kept.headers["X-Request-ID"] == "abc-123"
    replaced = await client.get("/health/live", headers={"X-Request-ID": "bad\nid"})
    assert replaced.headers["X-Request-ID"] != "bad\nid"
    assert len(replaced.headers["X-Request-ID"]) == 32


@pytest.mark.parametrize(
    ("env", "docs_enabled", "expected"),
    [
        ("local", None, 200),
        ("production", None, 404),
        ("production", True, 200),
        ("local", False, 404),
    ],
)
async def test_docs_are_off_in_production_unless_turned_on(
    env: str, docs_enabled: bool | None, expected: int
) -> None:
    settings = Settings.model_validate(
        {"app_env": env, "docs_enabled": docs_enabled, "database_url": PRODUCTION_DB}
    )
    transport = httpx.ASGITransport(app=create_app(settings))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        assert (await client.get("/openapi.json")).status_code == expected


async def test_json_logs_carry_the_request_id_and_a_structured_traceback(
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = Settings.model_validate({"app_env": "test", "log_json": True})
    app = create_app(settings)

    @app.get("/boom")
    async def boom() -> None:
        api_key = "sk-must-not-be-logged"
        raise RuntimeError(f"kaboom {len(api_key)}")

    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        await client.get("/boom", headers={"X-Request-ID": "req-1"})
    (line,) = [ln for ln in capsys.readouterr().out.splitlines() if "unhandled_error" in ln]
    event = json.loads(line)
    assert event["request_id"] == "req-1"
    assert event["exception"][0]["exc_type"] == "RuntimeError"
    # Frame locals can hold request data or credentials, so tracebacks never include them.
    assert "sk-xxxxxxxx" not in line


@pytest.mark.parametrize("log_json", [True, False])
async def test_tracebacks_never_include_frame_locals(
    log_json: bool, capsys: pytest.CaptureFixture[str]
) -> None:
    app = create_app(Settings.model_validate({"app_env": "test", "log_json": log_json}))

    @app.get("/boom")
    async def boom() -> None:
        # Built at run time, so only a captured local (not a source line) could show it.
        password = "hunter" + "2" * 3
        raise RuntimeError(str(len(password)))

    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        await client.get("/boom")
    captured = capsys.readouterr()
    assert "RuntimeError" in captured.out + captured.err
    assert "hunter222" not in captured.out + captured.err
