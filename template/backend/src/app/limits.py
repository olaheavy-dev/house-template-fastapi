"""A cap on request bodies, enforced before the app reads them, and baseline security headers.

Neither uvicorn nor FastAPI limits body size, so one large POST could be read whole into memory.
This middleware answers 413 when the declared Content-Length is over MAX_REQUEST_BYTES, and stops
reading (also 413) once a body without one streams past it. Files don't come through the API:
browsers upload them straight to storage with presigned URLs (app/storage.py), so the cap can be
small.
"""

from __future__ import annotations

from starlette.exceptions import HTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.problems import problem_response


class _TooLarge(HTTPException):
    # An HTTPException, so FastAPI's own body parsing passes it on as a 413 (any other exception
    # raised while it reads the body becomes a 400) and the problem handler renders it.
    def __init__(self, max_bytes: int) -> None:
        super().__init__(413, detail=f"The request body is over {max_bytes} bytes.")


class BodyLimit:
    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        declared = dict(scope["headers"]).get(b"content-length")
        if declared is not None and declared.isdigit() and int(declared) > self.max_bytes:
            await self._refuse(scope, receive, send)
            return

        seen = 0
        started = False

        async def counted() -> Message:
            nonlocal seen
            message = await receive()
            if message["type"] == "http.request":
                seen += len(message.get("body", b""))
                if seen > self.max_bytes:
                    raise _TooLarge(self.max_bytes)
            return message

        async def watched(message: Message) -> None:
            nonlocal started
            started = started or message["type"] == "http.response.start"
            await send(message)

        try:
            await self.app(scope, counted, watched)
        except _TooLarge:
            if started:  # the app answered before reading it all; nothing more to say
                return
            await self._refuse(scope, receive, send)

    async def _refuse(self, scope: Scope, receive: Receive, send: Send) -> None:
        response = problem_response(
            413, detail=f"The request body is over {self.max_bytes} bytes.", instance=scope["path"]
        )
        await response(scope, receive, send)


# Baseline headers for an API: no MIME sniffing, no framing, no referrer, and responses kept out of
# shared caches unless a route says otherwise. The docs page (/docs) is left alone: its inline
# scripts would break under a strict Content-Security-Policy.
BASELINE = {
    b"x-content-type-options": b"nosniff",
    b"x-frame-options": b"DENY",
    b"referrer-policy": b"no-referrer",
}


class SecurityHeaders:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                present = {name.lower() for name, _ in headers}
                headers += [(k, v) for k, v in BASELINE.items() if k not in present]
                if b"cache-control" not in present:
                    headers.append((b"cache-control", b"no-store"))
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, with_headers)
