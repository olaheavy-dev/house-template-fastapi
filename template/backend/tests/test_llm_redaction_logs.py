"""Logs, errors and the 503 body never hold a personal value (#78, test 24): every event redaction
writes carries a kind, a count or an error type, and nothing else."""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from langchain_core.messages import HumanMessage
from structlog.testing import capture_logs

from app.llm import GatewayDep
from app.llm.core.redaction import RedactionFailed, Redactor, redaction_scope
from app.llm.langchain.gateway import Gateway
from app.llm.langchain.redaction import RedactingChatModel
from app.settings import Settings
from tests import wire
from tests.wire import EMAIL, NAME, NINO, PHONE, SEEDS, TRICKY

pytestmark = pytest.mark.anyio

IMAGE = {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}}
EVENTS = {
    "llm_redaction",
    "redaction_allow_conflict",
    "llm_redaction_failed",
    "llm_unrestored_placeholders",
    "llm_unscanned_images",
}


class Boom:
    kind = "EMAIL"

    def find(self, text: str) -> Any:
        raise RuntimeError(f"cannot read {text}")  # an error whose message holds the text


async def test_no_event_redaction_writes_holds_a_personal_value(settings: Settings) -> None:
    _, model = wire.wrapped("openai")
    failing = RedactingChatModel(wire.raw_model("openai"), Redactor([Boom()]))
    with capture_logs() as logs:
        Gateway(settings, lambda: None)  # type: ignore[arg-type,return-value]  # llm_redaction
        # A known value that overlaps an allowed one (a conflict), a made-up placeholder, media.
        async with redaction_scope(known={"NAME": [NAME]}, allow=[f"{NAME} Jr"]):
            with wire.captured(lambda _b: wire.openai_text("Hi [NAME_1], [EMAIL_9], [PHONE_5]")):
                text = f"{NAME} Jr, {EMAIL}, {PHONE}, {NINO}"
                content: list[str | dict[Any, Any]] = [{"type": "text", "text": text}, IMAGE]
                await model.ainvoke([HumanMessage(content=content)])
        # A failing detector.
        with (
            wire.captured(lambda _b: wire.openai_text("x")),
            pytest.raises(RedactionFailed) as caught,
        ):
            await failing.ainvoke([HumanMessage(f"{EMAIL} {TRICKY}")])
    assert {e["event"] for e in logs} >= EVENTS
    text = repr(logs)
    for seed in [*SEEDS, TRICKY, "iVBORw0KGgo"]:
        assert seed not in text
    for seed in [*SEEDS, TRICKY]:
        assert seed not in str(caught.value)
        assert seed not in repr(caught.value)
    # Every field is a plain count, kind, type name or the word "langsmith".
    allowed = {"event", "log_level", "count", "kind", "kinds", "state", "error_type", "where"}
    for event in logs:
        if event["event"] in EVENTS:
            assert set(event) <= allowed


async def test_the_503_body_and_the_logs_of_a_failed_request_hold_no_value(
    client: httpx.AsyncClient, app: FastAPI
) -> None:
    gateway: Gateway = app.state.gateway
    gateway.redactor = Redactor([Boom()])
    gateway._model = wire.raw_model("openai", callbacks=[gateway.guard])

    @app.post("/ask")
    async def ask(gateway: GatewayDep) -> dict[str, str]:
        answer = await gateway.model().ainvoke([HumanMessage(f"{EMAIL} {TRICKY}")])
        return {"text": answer.text}

    with capture_logs() as logs, wire.captured(lambda _b: wire.openai_text("x")) as sent:
        response = await client.post("/ask")
    assert response.status_code == 503
    assert response.headers["content-type"] == "application/problem+json"
    assert sent.bodies == []
    for seed in (EMAIL, TRICKY):
        assert seed not in response.text
        assert seed not in repr(logs)
