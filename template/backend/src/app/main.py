"""The FastAPI app; `fastapi dev` and `fastapi run` find it through [tool.fastapi].

create_app builds a fresh app (tests build one per test); resources that live as long as the
process (the database engine, model clients) are opened and closed in lifespan.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import health
from app.log import configure_logging, request_context
from app.problems import install_problem_handlers
from app.settings import Settings, get_settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    yield


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings)
    docs = settings.docs_on
    app = FastAPI(
        lifespan=lifespan,
        docs_url="/docs" if docs else None,
        redoc_url=None,
        openapi_url="/openapi.json" if docs else None,
    )
    app.state.settings = settings
    app.state.readiness_checks = {}
    install_problem_handlers(app)
    app.middleware("http")(request_context)
    app.include_router(health.router)
    return app


app = create_app()
