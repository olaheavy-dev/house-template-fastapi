"""Liveness and readiness.

/health/live says the process is up and never touches a dependency, so a slow database can't
get the container restarted. /health/ready runs each registered readiness check (the database
layer adds one) and returns 503 with the failing names until all pass.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.responses import JSONResponse

ReadinessCheck = Callable[[], Awaitable[None]]
log = structlog.get_logger()
router = APIRouter(prefix="/health", tags=["health"])


def readiness_checks(request: Request) -> dict[str, ReadinessCheck]:
    checks: dict[str, ReadinessCheck] = request.app.state.readiness_checks
    return checks


def add_readiness_check(app: FastAPI, name: str, check: ReadinessCheck) -> None:
    app.state.readiness_checks[name] = check


@router.get("/live")
async def live() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ready", responses={503: {"description": "A dependency isn't ready"}})
async def ready(
    checks: Annotated[dict[str, ReadinessCheck], Depends(readiness_checks)],
) -> JSONResponse:
    failing: list[str] = []
    for name, check in checks.items():
        try:
            await check()
        except Exception:
            log.warning("readiness_check_failed", check=name, exc_info=True)
            failing.append(name)
    if failing:
        return JSONResponse({"status": "unavailable", "failing": failing}, status_code=503)
    return JSONResponse({"status": "ok"})
