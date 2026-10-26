"""Errors as RFC 9457 problem details (application/problem+json).

FastAPI has no built-in support, so these handlers replace its default error bodies. Raise
Problem for expected failures; unexpected exceptions become a 500 that hides the details
from the client and logs them with the request id.
"""

from __future__ import annotations

from collections.abc import Mapping
from http import HTTPStatus
from typing import Any

import structlog
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.log import REQUEST_ID_HEADER

MEDIA_TYPE = "application/problem+json"
log = structlog.get_logger()


# Members RFC 9457 defines; an extension may not reuse their names.
RESERVED = frozenset({"type", "title", "status", "detail", "instance"})


class Problem(Exception):
    """An expected failure the client should see, e.g. Problem(404, detail="No such course").
    `headers` go on the response (Retry-After on a 503 or 429, say)."""

    def __init__(
        self,
        status: int,
        *,
        detail: str | None = None,
        type_: str = "about:blank",
        title: str | None = None,
        headers: Mapping[str, str] | None = None,
        **extensions: Any,
    ) -> None:
        if clash := RESERVED & extensions.keys():
            raise ValueError(f"problem extensions can't be named {sorted(clash)}")
        super().__init__(detail or title or HTTPStatus(status).phrase)
        self.status = status
        self.detail = detail
        self.type = type_
        self.title = title
        self.headers = dict(headers or {})
        self.extensions = extensions


def problem_response(
    status: int,
    *,
    detail: str | None = None,
    type_: str = "about:blank",
    title: str | None = None,
    instance: str | None = None,
    headers: Mapping[str, str] | None = None,
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
    body.update({k: v for k, v in extensions.items() if k not in RESERVED})
    return JSONResponse(body, status_code=status, media_type=MEDIA_TYPE, headers=headers)


async def _problem(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, Problem)  # noqa: S101 - registered for Problem only
    return problem_response(
        exc.status,
        detail=exc.detail,
        type_=exc.type,
        title=exc.title,
        instance=request.url.path,
        headers=exc.headers,
        **exc.extensions,
    )


async def _http_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, StarletteHTTPException)  # noqa: S101
    # RFC 9457's detail is a string: a structured HTTPException detail travels as "errors".
    shown: Any = exc.detail if exc.status_code < 500 else None
    extra: dict[str, Any] = {}
    if shown is not None and not isinstance(shown, str):
        shown, extra = None, {"errors": shown}
    return problem_response(
        exc.status_code,
        detail=shown,
        instance=request.url.path,
        headers=exc.headers,
        **extra,
    )


async def _validation_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, RequestValidationError)  # noqa: S101
    # Echo where and why, never the submitted values: they may hold personal data.
    errors = [{"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]} for e in exc.errors()]
    return problem_response(
        422, detail="The request is not valid.", instance=request.url.path, errors=errors
    )


async def _unexpected(request: Request, exc: Exception) -> JSONResponse:
    # Called from inside an `except`, so the traceback logged is the original one.
    log.exception("unhandled_error", path=request.url.path)
    response = problem_response(500, instance=request.url.path)
    request_id = getattr(request.state, "request_id", None)
    if request_id:
        response.headers[REQUEST_ID_HEADER] = request_id
    return response


class UnhandledErrors:
    """Turns an unexpected exception into the bare 500 problem response inside the app's own
    middleware. Starlette runs its catch-all handler outermost, outside CORS and the security
    headers, so a 500 from there reaches a browser as a CORS error with no request id; caught
    here, it passes back out through them. The registered handler stays as a last resort."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        started = False

        async def watched(message: Message) -> None:
            nonlocal started
            started = started or message["type"] == "http.response.start"
            await send(message)

        try:
            await self.app(scope, receive, watched)
        except Exception:
            if started:  # too late to answer: let the server log it and drop the connection
                raise
            request = Request(scope)
            response = await _unexpected(request, Exception())
            await response(scope, receive, send)


def install_problem_handlers(app: FastAPI) -> None:
    app.add_exception_handler(Problem, _problem)
    app.add_exception_handler(StarletteHTTPException, _http_error)
    app.add_exception_handler(RequestValidationError, _validation_error)
    app.add_exception_handler(Exception, _unexpected)
