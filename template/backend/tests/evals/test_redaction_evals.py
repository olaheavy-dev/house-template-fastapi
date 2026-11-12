"""Live checks of personal-data redaction with the real provider (#78, decision 0006). The offline
tests prove the detectors on a synthetic corpus; these prove what a mock cannot: that the bytes the
real SDK puts on the wire hold none of the seeded values, and that a real model hands placeholders
back intact so a user still sees their own name and email. They run through the shared harness, so
they hold for whichever framework the project chose.

Spend: the whole file is meant to cost a few cents. Run it with a small cap, for example
`LLM_EVAL=1 LLM_DAILY_BUDGET_USD=0.10 pytest tests/evals/test_redaction_evals.py -m eval`, and a
fixture fails the run if the settled cost of every call so far reaches that cap."""

from __future__ import annotations

import re
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
import vcr  # type: ignore[import-untyped]
from langsmith import testing as t
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from structlog.testing import capture_logs

from app.llm.core.budget import BudgetExhausted, LlmCall
from app.llm.core.redaction import RedactionFailed, Redactor, Scope, redaction_scope
from app.llm.redaction_rules import DETECTORS
from tests.evals.redaction_corpus import KNOWN_MISSES, LEAK_CORPUS
from tests.harness import Harness, user

pytestmark = [pytest.mark.anyio, pytest.mark.eval]

CAP_MICRO_USD = 100_000  # $0.10 for the whole file
PLACEHOLDER = re.compile(r"\[[A-Z]+_\d+\]")
SECRET_HEADERS = ["authorization", "x-api-key", "openai-organization", "openai-project"]


@pytest.fixture(autouse=True)
async def spend_stays_under_the_cap(database_url: str) -> AsyncIterator[None]:
    """After each test: every call settled so far in this run costs less than the cap."""
    yield
    engine = create_async_engine(database_url)
    try:
        async with async_sessionmaker(engine)() as session:
            spent = await session.scalar(select(func.coalesce(func.sum(LlmCall.cost_micro_usd), 0)))
    finally:
        await engine.dispose()
    print(f"redaction evals: settled spend so far {spent or 0} micro-USD (cap {CAP_MICRO_USD})")
    assert (spent or 0) < CAP_MICRO_USD, f"the redaction evals spent {spent} micro-USD"


@pytest.fixture
async def short_gateway(make_gateway: Any) -> Any:
    """The gateway on the real provider with a tiny output limit: the leak eval reads request
    bodies, so the answers can be as short as the model likes."""
    return await make_gateway(llm_max_tokens=32)


def system(text: str) -> dict[str, str]:
    return {"role": "system", "content": text}


@pytest.mark.langsmith
async def test_redaction_keeps_drafts_useful(framework: Harness, gateway: Any) -> None:
    # A draft signed with the user's own name and email: the model sees placeholders, the user
    # must see their details, exactly, and no placeholder.
    name, mail = "Priya Shah", "priya.shah@example.com"
    prompt = (
        "Write a polite two-line closing for a message to a landlord. The second line must be "
        f"exactly: {name}, {mail}"
    )
    t.log_inputs({"prompt": prompt, "framework": framework.name})
    with capture_logs() as logs:
        async with redaction_scope(known={"NAME": [name]}):
            answer = await framework.ask(
                gateway, [system("You write short, polite messages."), user(prompt)]
            )
    unrestored = [e for e in logs if e["event"] == "llm_unrestored_placeholders"]
    t.log_outputs({"answer": answer, "unrestored_events": len(unrestored)})
    t.log_feedback(key="name_and_email_back", score=name in answer and mail in answer)
    t.log_feedback(key="no_placeholder_shown", score=PLACEHOLDER.search(answer) is None)
    assert name in answer
    assert mail in answer
    assert PLACEHOLDER.search(answer) is None
    assert unrestored == []


@pytest.mark.langsmith
async def test_redaction_leak_corpus(
    framework: Harness, short_gateway: Any, tmp_path: Path
) -> None:
    # Twenty synthetic documents through the real SDK. The calls are recorded (all of them, into a
    # throwaway cassette) only to read what was really sent.
    refused: list[str] = []
    recorder = vcr.VCR(record_mode="all", filter_headers=SECRET_HEADERS)
    with recorder.use_cassette(str(tmp_path / "leak.yaml")) as cassette:
        for document in LEAK_CORPUS:
            async with redaction_scope(known=document.known, allow=document.allow):
                try:
                    await framework.ask(
                        short_gateway,
                        [system("Reply with the single word: ok"), user(document.text)],
                    )
                except (RedactionFailed, BudgetExhausted):
                    raise
                except Exception as error:
                    # A reasoning model may use its whole 32-token limit before it answers, and
                    # the provider refuses that with a 400. The request was still sent, and the
                    # request is what is checked; count the refusals instead of failing.
                    refused.append(type(error).__name__)
        sent = [
            r.body.decode() if isinstance(r.body, bytes) else str(r.body) for r in cassette.requests
        ]
    wire_text = "\n".join(sent)
    leaked = [v for document in LEAK_CORPUS for v in document.seeded if v in wire_text]
    t.log_inputs({"documents": len(LEAK_CORPUS), "framework": framework.name})
    t.log_outputs({"requests": len(sent), "leaked": len(leaked), "refused": len(refused)})
    t.log_feedback(key="seeded_values_sent", score=len(leaked))
    for kind in sorted({k for document in LEAK_CORPUS for k in document.counts}):
        total = sum(document.counts.get(kind, 0) for document in LEAK_CORPUS)
        t.log_feedback(key=f"replaced_{kind.lower()}", score=total)
    # The known misses (#97) are reported, never asserted: 1 means the text goes out as it is.
    redactor = Redactor(DETECTORS)
    for label, text in KNOWN_MISSES:
        missed = redactor.redact([text], Scope()).texts == (text,)
        t.log_feedback(key="missed_" + re.sub(r"\W+", "_", label.lower()), score=int(missed))
    assert len(sent) >= len(LEAK_CORPUS)
    assert leaked == []
    assert PLACEHOLDER.search(wire_text) is not None  # placeholders really went instead


# Five synthetic chunks, two of them from people with the same kind of details: once cleaned, those
# two read the same apart from the problem they describe, so only the rest of the text can pick one.
CHUNKS = (
    "Tenant Priya Shah (priya.shah@example.com, 07700 900123) reports a dripping kitchen tap. The "
    "landlord has booked a plumber for Thursday.",
    "The tenancy deposit of five weeks' rent is protected in a government-approved scheme within "
    "30 days of payment.",
    "The boiler was serviced in October and the gas safety certificate is valid for twelve months.",
    "Tenant Sam Okafor (sam.okafor@example.com, 07700 900456) asks to end the contract early. The "
    "break clause allows this after six months with two months' notice.",
    "Council tax band D. Broadband up to 900 Mbps. One parking space is included with the flat.",
)
# (a question, the index of the chunk that answers it)
QUESTIONS = (
    ("What did Priya Shah (priya.shah@example.com) report about the kitchen tap?", 0),
    ("Sam Okafor wants to leave early: what does the break clause say?", 3),
    ("Is my deposit protected, and how soon?", 1),
    ("When does the gas safety certificate run out?", 2),
)


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norms: float = sum(x * x for x in a) ** 0.5 * sum(y * y for y in b) ** 0.5
    return float(dot / norms)


@pytest.mark.langsmith
async def test_redacted_embeddings_still_find_the_right_chunk(
    framework: Harness, gateway: Any
) -> None:
    # The wire test (tests/test_llm_redaction_embeddings.py) covers what the vectors are made from;
    # this eval checks that they still rank well.
    # Documents and questions are embedded under the same scope, as the docs say to. The vectors
    # are of the cleaned text (no name, email or phone), and the right chunk must still rank first
    # for every question.
    scope = {"NAME": ["Priya Shah", "Sam Okafor"]}
    with capture_logs():
        async with redaction_scope(known=scope):
            documents = await framework.embed_documents(gateway, list(CHUNKS), feature="eval")
            questions = [
                await framework.embed_query(gateway, question, feature="eval")
                for question, _ in QUESTIONS
            ]
    first = [
        max(range(len(CHUNKS)), key=lambda i: cosine(vector, documents[i])) for vector in questions
    ]
    t.log_inputs({"chunks": len(CHUNKS), "questions": len(QUESTIONS), "framework": framework.name})
    t.log_outputs({"first_ranked": first})
    right = sum(found == want for found, (_, want) in zip(first, QUESTIONS, strict=True))
    t.log_feedback(key="questions_answered_first", score=right)
    assert first == [want for _, want in QUESTIONS]
