"""OpenTelemetry request spans from FastAPI's native telemetry: requests are traced, health probes
are not."""

from __future__ import annotations

import httpx
import pytest
from asgi_lifespan import LifespanManager
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from app.main import create_app
from app.settings import Settings

pytestmark = pytest.mark.anyio


async def test_requests_are_traced_but_health_probes_are_not() -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    app = create_app(
        Settings(_env_file=None, app_env="test"), telemetry={"tracer_provider": provider}
    )

    @app.get("/ping")
    async def ping() -> str:
        return "pong"

    async with LifespanManager(app) as manager:
        transport = httpx.ASGITransport(app=manager.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            await client.get("/health/live")
            await client.get("/ping")
    names = {span.name for span in exporter.get_finished_spans()}
    assert "GET /ping" in names
    assert not any("/health" in name for name in names)
