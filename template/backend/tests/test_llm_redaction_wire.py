"""Nothing unredacted reaches the wire (#78, test 22), proved with the real provider SDK and a
recorded cassette. Replay matches on the request body, with the network blocked, so the test only
passes if the bytes the SDK sends are exactly the redacted bytes that were recorded; and the
cassette file itself must hold none of the seeded values.

The cassette is recorded once, by hand, with the pinned model of tests/models.py and a small spend
cap, never by `make record-llm`. From backend/, with the key loaded from a file outside the
repository (never echoed):

    (set -a; . <file holding OPENAI_API_KEY>; set +a; \\
     env -u LLM_EVAL LLM_DAILY_BUDGET_USD=0.10 \\
     uv run --extra db --extra langchain --extra storage \\
     pytest tests/test_llm_redaction_wire.py -m vcr --record-mode=once -p no:randomly)

Until the cassette exists the replay test is skipped (a missing cassette would otherwise fail it).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from langchain_core.messages import HumanMessage, SystemMessage
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.llm.core.budget import LlmCall
from app.llm.core.redaction import redaction_scope
from app.llm.langchain.gateway import Gateway
from tests.wire import EMAIL, NAME, NINO, PHONE, SEEDS

pytestmark = pytest.mark.anyio

CASSETTE = (
    Path(__file__).parent
    / "cassettes"
    / "test_llm_redaction_wire"
    / "test_nothing_unredacted_reaches_the_wire.yaml"
)
RECORD_COMMAND = "see the module notes of tests/test_llm_redaction_wire.py"


@pytest.fixture
def cassette_or_skip(request: pytest.FixtureRequest) -> None:
    """Skip only while there is nothing to replay and nobody is recording."""
    recording = request.config.getoption("--record-mode") not in (None, "none")
    if not CASSETTE.exists() and not recording:
        pytest.skip(f"the cassette is not recorded yet: {RECORD_COMMAND}")


@pytest.mark.vcr
@pytest.mark.block_network
@pytest.mark.usefixtures("cassette_or_skip")
async def test_nothing_unredacted_reaches_the_wire(
    app: FastAPI, db_client: httpx.AsyncClient, db_session: AsyncSession, vcr: Any
) -> None:
    gateway: Gateway = app.state.gateway
    messages = [
        SystemMessage(f"You help {NAME} with a tenancy."),
        HumanMessage(f"Mail {EMAIL}, phone {PHONE}, NI {NINO}. Reply with one word: ok."),
    ]
    async with redaction_scope(known={"NAME": [NAME]}):
        answer = await gateway.model().ainvoke(messages)
    assert answer.text  # the call was answered, from the recording on replay

    [request] = vcr.requests
    sent = request.body.decode() if isinstance(request.body, bytes) else str(request.body)
    for seed in SEEDS:
        assert seed not in sent, "the request body held an unredacted value"
    for placeholder in ("[NAME_1]", "[EMAIL_1]", "[PHONE_1]", "[NINO_1]"):
        assert placeholder in sent
    # Nor does the recorded answer: the model never saw the originals.
    for seed in SEEDS:
        assert seed not in str(vcr.responses)
    [call] = (await db_session.scalars(select(LlmCall))).all()
    assert call.redacted_count == 4
    assert call.status == "settled"


def test_the_recorded_cassette_holds_no_seeded_value() -> None:
    if not CASSETTE.exists():
        pytest.skip(f"the cassette is not recorded yet: {RECORD_COMMAND}")
    text = CASSETTE.read_text()
    for seed in SEEDS:
        assert seed not in text, seed
    assert "[EMAIL_1]" in text
