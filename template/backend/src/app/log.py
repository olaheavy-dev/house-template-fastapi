"""Structured logging with structlog, one JSON line per event in production.

Every request gets a request id (from the X-Request-ID header when a proxy sets one, otherwise
new), bound to all log lines for that request and echoed in the response header.
"""

from __future__ import annotations

import logging
import re
import uuid
from collections.abc import Awaitable, Callable

import structlog
from fastapi import Request, Response
from structlog.tracebacks import ExceptionDictTransformer

from app.settings import Settings

REQUEST_ID_HEADER = "X-Request-ID"
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
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            *rendering,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelNamesMapping()[settings.log_level]
        ),
    )


async def request_context(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    incoming = request.headers.get(REQUEST_ID_HEADER, "")
    request_id = incoming if _SAFE_ID.match(incoming) else uuid.uuid4().hex
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(request_id=request_id)
    # Kept on the request too: the 500 handler runs outside this middleware and adds it itself.
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers[REQUEST_ID_HEADER] = request_id
    return response
