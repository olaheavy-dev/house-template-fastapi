"""Structured logging with structlog, one JSON line per event in production.

Every request gets a request id (from the X-Request-ID header when a proxy sets one, otherwise
new), bound to all log lines for that request and echoed in the response header, and one access
line when it ends. Logs from the standard library (uvicorn, SQLAlchemy, httpx, LangChain) go
through the same renderer, so production output is JSON throughout.
"""

from __future__ import annotations

import logging
import re
import sys
import time
import uuid
from collections.abc import Awaitable, Callable

import structlog
from fastapi import Request, Response
from structlog.tracebacks import ExceptionDictTransformer

from app.settings import Settings

REQUEST_ID_HEADER = "X-Request-ID"
log = structlog.get_logger()
# Accept a caller's id only if it looks like an id, so a header can't inject log content.
_SAFE_ID = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


def configure_logging(settings: Settings) -> None:
    # JSON lines carry tracebacks as structured data; the console renderer formats them itself.
    # Neither shows frame locals: they can hold request data or credentials (both default to on).
    rendering: list[structlog.types.Processor] = (
        [
            structlog.processors.ExceptionRenderer(ExceptionDictTransformer(show_locals=False)),
            structlog.processors.JSONRenderer(),
        ]
        if settings.log_json
        else [
            structlog.dev.ConsoleRenderer(
                exception_formatter=structlog.dev.RichTracebackFormatter(show_locals=False)
            )
        ]
    )
    shared: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
    ]
    level = logging.getLevelNamesMapping()[settings.log_level]
    structlog.configure(
        processors=[*shared, *rendering],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(),
    )
    _route_stdlib_logging(shared, rendering, level)


class _NotTheDuplicateTraceback(logging.Filter):
    # The app's 500 handler has already logged the error with its request id; uvicorn then logs
    # the same exception again, as plain text without one, when Starlette re-raises it.
    def filter(self, record: logging.LogRecord) -> bool:
        return not record.getMessage().startswith("Exception in ASGI application")


def _route_stdlib_logging(
    shared: list[structlog.types.Processor],
    rendering: list[structlog.types.Processor],
    level: int,
) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=shared,
            processors=[structlog.stdlib.ProcessorFormatter.remove_processors_meta, *rendering],
        )
    )
    handler.addFilter(_NotTheDuplicateTraceback())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)
    # uvicorn (`fastapi run`) installs its own plain-text handlers before the app is imported:
    # send its logs to the root handler instead. Its access log is replaced by request_context's.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.propagate = True
    logging.getLogger("uvicorn.access").disabled = True
    # A line per outgoing HTTP request is noise at INFO.
    for name in ("httpx", "httpx2", "httpcore", "openai", "anthropic", "botocore", "urllib3"):
        logging.getLogger(name).setLevel(max(level, logging.WARNING))


async def request_context(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    incoming = request.headers.get(REQUEST_ID_HEADER, "")
    request_id = incoming if _SAFE_ID.match(incoming) else uuid.uuid4().hex
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(request_id=request_id)
    # Kept on the request too: the 500 handler runs outside this middleware and adds it itself.
    request.state.request_id = request_id
    started = time.perf_counter()
    response = await call_next(request)
    response.headers[REQUEST_ID_HEADER] = request_id
    # The route's template (/courses/{course_id}), not the path: ids and other values in URLs
    # stay out of the logs. Unhandled errors are logged by the 500 handler instead.
    route = request.scope.get("route")
    log.info(
        "request",
        method=request.method,
        route=getattr(route, "path", None),
        status=response.status_code,
        duration_ms=round((time.perf_counter() - started) * 1000),
    )
    return response
