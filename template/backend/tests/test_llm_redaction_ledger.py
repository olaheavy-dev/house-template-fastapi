"""Redaction counts on the ledger (#78), against a real PostgreSQL: each call's llm_call row says
how many values were replaced in its request, and NULL when redaction was off."""

from __future__ import annotations

import httpx
import pytest
from fastapi import FastAPI
from langchain_core.messages import HumanMessage
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.llm.core.budget import LlmCall
from app.llm.langchain.gateway import Gateway
from app.llm.langchain.redaction import RedactingChatModel
from tests import wire
from tests.wire import EMAIL, NINO, PHONE

pytestmark = pytest.mark.anyio


async def counts(session: AsyncSession) -> list[int | None]:
    rows = await session.scalars(select(LlmCall.redacted_count).order_by(LlmCall.redacted_count))
    return list(rows)


async def test_each_call_row_counts_the_values_replaced_in_its_request(
    app: FastAPI, db_client: httpx.AsyncClient, db_session: AsyncSession
) -> None:
    gateway: Gateway = app.state.gateway
    assert gateway.redactor is not None
    model = RedactingChatModel(
        wire.raw_model("openai", callbacks=[gateway.guard]), gateway.redactor
    )
    with wire.captured(lambda _b: wire.openai_text("ok")):
        await model.ainvoke([HumanMessage(f"Mail {EMAIL}, call {PHONE}, NI {NINO}")])
        await model.ainvoke([HumanMessage("What is a freehold?")])
    rows = (await db_session.scalars(select(LlmCall))).all()
    # Email, phone and NINO are three values; the second request had none (zero is not NULL).
    assert sorted(row.redacted_count for row in rows if row.redacted_count is not None) == [0, 3]
    assert [row.status for row in rows] == ["settled", "settled"]
