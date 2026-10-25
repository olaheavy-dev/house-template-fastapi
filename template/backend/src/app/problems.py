"""Errors as RFC 9457 problem details (application/problem+json).

FastAPI has no built-in support, so these handlers replace its default error bodies. Raise
Problem for expected failures; unexpected exceptions become a 500 that hides the details
from the client and logs them with the request id.
"""

from __future__ import annotations

from http import HTTPStatus
from typing import Any

import structlog
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.log import REQUEST_ID_HEADER

MEDIA_TYPE = "application/problem+json"
log = structlog.get_logger()


class Problem(Exception):
    """An expected failure the client should see, e.g. Problem(404, detail="No such course")."""

    def __init__(
        self,
        status: int,
        *,
        detail: str | None = None,
        type_: str = "about:blank",
        title: str | None = None,
        **extensions: Any,
    ) -> None:
        super().__init__(detail or title or HTTPStatus(status).phrase)
        self.status = status
        self.detail = detail
        self.type = type_
        self.title = title
        self.extensions = extensions


def problem_response(
    status: int,
    *,
    detail: str | None = None,
    type_: str = "about:blank",
    title: str | None = None,
    instance: str | None = None,
    **extensions: Any,
) -> JSONResponse:
    body: dict[str, Any] = {
        "type": type_,
        # With the default type the title must be the status phrase (RFC 9457 section 4.2.1).
        "title": title if title and type_ != "about:blank" else HTTPStatus(status).phrase,
        "status": status,
    }
    if detail:
        body["detail"] = detail
    if instance:
        body["instance"] = instance
    body.update(extensions)
    return JSONResponse(body, status_code=status, media_type=MEDIA_TYPE)


async def _problem(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, Problem)  # noqa: S101 - registered for Problem only
    return problem_response(
        exc.status,
        detail=exc.detail,
        type_=exc.type,
        title=exc.title,
        instance=request.url.path,
        **exc.extensions,
    )


async def _http_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, StarletteHTTPException)  # noqa: S101
    detail = exc.detail if exc.status_code < 500 else None
    response = problem_response(exc.status_code, detail=detail, instance=request.url.path)
    for name, value in (exc.headers or {}).items():
        response.headers[name] = value
    return response


async def _validation_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, RequestValidationError)  # noqa: S101
    # Echo where and why, never the submitted values: they may hold personal data.
    errors = [{"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]} for e in exc.errors()]
    return problem_response(
        422, detail="The request is not valid.", instance=request.url.path, errors=errors
    )


async def _unexpected(request: Request, exc: Exception) -> JSONResponse:
    log.exception("unhandled_error", path=request.url.path)
    response = problem_response(500, instance=request.url.path)
    request_id = getattr(request.state, "request_id", None)
    if request_id:
        response.headers[REQUEST_ID_HEADER] = request_id
    return response


def install_problem_handlers(app: FastAPI) -> None:
    app.add_exception_handler(Problem, _problem)
    app.add_exception_handler(StarletteHTTPException, _http_error)
    app.add_exception_handler(RequestValidationError, _validation_error)
    app.add_exception_handler(Exception, _unexpected)
