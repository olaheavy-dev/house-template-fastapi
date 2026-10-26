"""The FastAPI app; `fastapi dev` and `fastapi run` find it through [tool.fastapi].

create_app builds a fresh app (tests build one per test); resources that live as long as the
process (the database engine, model clients) are opened and closed in lifespan.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, MutableMapping
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.telemetry import TelemetryConfig

from app import api, db, health
from app.limits import BodyLimit, SecurityHeaders
from app.llm import checkpoints
from app.llm.errors import install_model_error_handlers
from app.llm.gateway import Gateway
from app.log import REQUEST_ID_HEADER, configure_logging, request_context
from app.problems import UnhandledErrors, install_problem_handlers
from app.settings import Settings, get_settings
from app.storage import Storage


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    engine = db.create_engine(app.state.settings)
    app.state.engine = engine
    app.state.sessionmaker = db.create_sessionmaker(engine)
    settings: Settings = app.state.settings
    app.state.storage = Storage(settings) if settings.storage_bucket else None
    # The budget's own small pool: a request holding a session from the main pool must never
    # wait on that same pool to reserve or settle a model call.
    budget_engine = db.create_engine(settings, pool_size=2, max_overflow=3)
    app.state.budget_sessionmaker = db.create_sessionmaker(budget_engine)
    # Looked up per call, so tests can swap it after startup.
    app.state.gateway = Gateway(settings, lambda: app.state.budget_sessionmaker)
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
        await budget_engine.dispose()
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
    # Registered before request_context, which therefore wraps it: a 413 still gets an access line.
    app.add_middleware(BodyLimit, max_bytes=settings.max_request_bytes)
    install_model_error_handlers(app)
    app.middleware("http")(request_context)
    # Outside request_context (so it knows the request id), inside CORS and the headers (so an
    # unexpected 500 still carries them).
    app.add_middleware(UnhandledErrors)
    # Outermost, in this order: hosts are checked first, then CORS answers preflights.
    app.add_middleware(SecurityHeaders)
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=[REQUEST_ID_HEADER, "Retry-After"],
        )
    if settings.allowed_hosts != ["*"]:
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.allowed_hosts)
    app.include_router(health.router)
    app.include_router(api.router)
    health.add_readiness_check(app, "database", lambda: _database_ready(app))
    if settings.storage_bucket:
        health.add_readiness_check(app, "storage", lambda: app.state.storage.ping())
    return app


app = create_app()
