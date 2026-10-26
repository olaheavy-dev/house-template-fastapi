"""The FastAPI app; `fastapi dev` and `fastapi run` find it through [tool.fastapi].

create_app builds a fresh app (tests build one per test); resources that live as long as the
process (the database engine, model clients) are opened and closed in lifespan.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, MutableMapping
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.telemetry import TelemetryConfig

from app import db, health
from app.llm import checkpoints
from app.llm.errors import install_model_error_handlers
from app.llm.gateway import Gateway
from app.log import configure_logging, request_context
from app.problems import install_problem_handlers
from app.settings import Settings, get_settings
from app.storage import Storage


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    engine = db.create_engine(app.state.settings)
    app.state.engine = engine
    app.state.sessionmaker = db.create_sessionmaker(engine)
    settings: Settings = app.state.settings
    app.state.storage = Storage(settings) if settings.storage_bucket else None
    # The budget looks the sessionmaker up per call, so tests can swap it after startup.
    app.state.gateway = Gateway(app.state.settings, lambda: app.state.sessionmaker)
    app.state.gateway.checkpointer, checkpoint_pool = await checkpoints.open_checkpointer(
        app.state.settings
    )
    try:
        yield
    finally:
        await app.state.gateway.aclose()
        await checkpoint_pool.close()
        if app.state.storage:
            app.state.storage.close()
        await engine.dispose()


async def _database_ready(app: FastAPI) -> None:
    await db.ping(app.state.engine)


def _not_health(scope: MutableMapping[str, Any]) -> bool:
    # Probes hit /health every few seconds; their spans and metrics would drown the real traffic.
    return str(scope.get("path", "")).startswith("/health/")


def telemetry_config(
    settings: Settings, overrides: TelemetryConfig | None = None
) -> TelemetryConfig:
    """FastAPI's native OpenTelemetry: request spans, HTTP metrics and exception logs. Nothing is
    exported until FASTAPI_OTEL_AUTO_CONFIGURE=true and OTEL_EXPORTER_OTLP_ENDPOINT are set (or a
    provider is passed in); tests never export."""
    config: TelemetryConfig = {"exclude": _not_health}
    if settings.app_env == "test":
        config["auto_configure"] = False
    return {**config, **(overrides or {})}


def create_app(
    settings: Settings | None = None, *, telemetry: TelemetryConfig | None = None
) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings)
    docs = settings.docs_on
    app = FastAPI(
        lifespan=lifespan,
        docs_url="/docs" if docs else None,
        redoc_url=None,
        openapi_url="/openapi.json" if docs else None,
        telemetry=telemetry_config(settings, telemetry),
    )
    app.state.settings = settings
    app.state.readiness_checks = {}
    install_problem_handlers(app)
    install_model_error_handlers(app)
    app.middleware("http")(request_context)
    app.include_router(health.router)
    health.add_readiness_check(app, "database", lambda: _database_ready(app))
    if settings.storage_bucket:
        health.add_readiness_check(app, "storage", lambda: app.state.storage.ping())
    return app


app = create_app()
