"""Redaction of embeddings, the parts that belong to LangChain and to the source tree (#78, part 3):
the bytes OpenAI's real SDK puts on the wire, a retriever run in a LangSmith trace, and two guards
that fail the build if a second way to the embeddings API appears under app/ (it would skip both
the budget and the redaction). What every framework does is in tests/test_llm_parity_redaction.py.
The wire and the LangSmith session are replaced, so nothing leaves the machine."""

from __future__ import annotations

import ast
import base64
import json
import struct
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.callbacks import AsyncCallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.tracers.langchain import LangChainTracer, get_client, wait_for_all_tracers
from langsmith import tracing_context
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.llm.core.redaction import redaction_scope
from app.llm.langchain.gateway import Gateway
from app.settings import Settings
from tests import wire
from tests.harness import FakeEmbeddingsApi
from tests.wire import EMAIL, NAME, PHONE, TRICKY

pytestmark = pytest.mark.anyio

DAY = date(2026, 1, 3)
APP = Path(__file__).resolve().parents[1] / "src" / "app"


@pytest.fixture
def gateway(settings: Settings, db_session: AsyncSession) -> Gateway:
    sessions = async_sessionmaker(
        bind=db_session.bind, join_transaction_mode="create_savepoint", expire_on_commit=False
    )
    built = Gateway(settings, lambda: sessions)
    meter = built.embeddings().meter
    meter.cap, meter.today = 1_000_000, lambda: DAY
    return built


def embeddings_reply(body: dict[str, Any]) -> dict[str, Any]:
    """What OpenAI answers: one vector per input (base64 when the SDK asked for it)."""
    vector = (0.5, 0.25, 0.125)
    packed = base64.b64encode(struct.pack("<3f", *vector)).decode()
    wanted = packed if body.get("encoding_format") == "base64" else list(vector)
    count = len(body["input"])
    return {
        "object": "list",
        "model": body["model"],
        "data": [{"object": "embedding", "index": i, "embedding": wanted} for i in range(count)],
        "usage": {"prompt_tokens": 7 * count, "total_tokens": 7 * count},
    }


async def test_the_real_sdk_request_holds_no_seeded_value_and_the_cleaned_text_instead(
    gateway: Gateway,
) -> None:
    texts = [f"{TRICKY} wrote from {EMAIL}", f"call {PHONE}", "a freehold"]
    async with redaction_scope(known={"NAME": [TRICKY]}):
        with wire.captured(embeddings_reply) as sent:
            vectors = await gateway.embeddings().aembed_documents(texts)
    assert len(vectors) == 3
    [body] = sent.json_bodies()
    assert body["input"] == ["[NAME_1] wrote from [EMAIL_1]", "call [PHONE_1]", "a freehold"]
    for seed in (TRICKY, EMAIL, PHONE, NAME):
        assert seed not in sent.text
    assert json.dumps(TRICKY)[1:-1] not in sent.text  # nor JSON-escaped


class TracedRetriever(BaseRetriever):
    """A retriever that embeds the question and returns a document naming a person."""

    gateway: Gateway

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> list[Document]:
        await self.gateway.embeddings().aembed_query(query, feature="ask")
        return [Document(page_content=f"{TRICKY} can be reached on {EMAIL}")]

    def _get_relevant_documents(self, query: str, *, run_manager: Any) -> list[Document]:
        raise RuntimeError("await it")  # pragma: no cover


async def test_a_retriever_run_that_embeds_the_question_is_traced_masked(
    gateway: Gateway,
) -> None:
    # The gateway sets the process-wide LangSmith client with the masking anonymizer, so the
    # retriever's own run (its question and the documents it returns) is masked like a chat run.
    session = wire.FakeLangSmithSession()
    get_client().session = session
    fake = FakeEmbeddingsApi()
    gateway.embeddings()._client = SimpleNamespace(embeddings=fake)
    tracer = LangChainTracer(project_name="test")
    async with redaction_scope(known={"NAME": [TRICKY]}):
        with tracing_context(enabled=True, project_name="test"):
            documents = await TracedRetriever(gateway=gateway).ainvoke(
                f"who is {TRICKY}, {EMAIL}?", config={"callbacks": [tracer]}
            )
        wait_for_all_tracers()
        get_client().flush()
    assert EMAIL in documents[0].page_content  # the caller still gets its own documents
    assert fake.requests == [["who is [NAME_1], [EMAIL_1]?"]]
    assert '"retriever"' in session.text  # the run was traced
    assert "[EMAIL_1]" in session.text
    for seed in (TRICKY, EMAIL, PHONE):
        assert seed not in session.text


# --- no second way to the embeddings API ----------------------------------------------------------

BANNED = {"init_embeddings", "OpenAIEmbeddings", "AzureOpenAIEmbeddings"}


def embedding_bypasses(source: str) -> list[str]:
    """What in `source` builds an embeddings client of LangChain's own (it would send the original
    texts and skip the budget): a name, an attribute or an import of one of BANNED."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Name) and node.id in BANNED:
            found.append(f"{node.id} at line {node.lineno}")
        elif isinstance(node, ast.Attribute) and node.attr in BANNED:
            found.append(f"{node.attr} at line {node.lineno}")
        elif isinstance(node, ast.alias) and node.name.rsplit(".", 1)[-1] in BANNED:
            found.append(f"{node.name} imported")
    return found


def direct_creates(source: str) -> list[str]:
    """Where `source` calls OpenAI's embeddings request of core/openai_embeddings.py, as
    `openai_embeddings.create(...)` or after `from ... import create`."""
    found = []
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and node.attr == "create"
            and isinstance(node.value, ast.Name)
            and node.value.id == "openai_embeddings"
        ):
            found.append(f"openai_embeddings.create at line {node.lineno}")
        elif isinstance(node, ast.ImportFrom) and (node.module or "").endswith("openai_embeddings"):
            found += [
                f"create imported at line {node.lineno}" for a in node.names if a.name == "create"
            ]
    return found


def sources() -> dict[str, str]:
    return {str(p.relative_to(APP)): p.read_text() for p in sorted(APP.rglob("*.py"))}


def test_only_the_gateways_embeddings_call_the_provider_embeddings_request() -> None:
    callers = {path for path, source in sources().items() if direct_creates(source)}
    # core/embeddings.py is handed `create` (it never names the SDK module); the one caller is
    # the LangChain gateway's class, which passes the redactor with it.
    assert callers == {"llm/langchain/embeddings.py"}


def test_no_langchain_embeddings_client_is_built_anywhere_under_app() -> None:
    assert sources(), "no source files found"
    problems = {path: embedding_bypasses(source) for path, source in sources().items()}
    assert {path: found for path, found in problems.items() if found} == {}


@pytest.mark.parametrize(
    "source",
    [
        "from langchain.embeddings import init_embeddings\nx = init_embeddings('openai:m')",
        "import langchain.embeddings as e\ne.init_embeddings('openai:m')",
        "from langchain_openai import OpenAIEmbeddings",
        "from langchain_openai import AzureOpenAIEmbeddings as A",
        "import langchain_openai\nlangchain_openai.OpenAIEmbeddings()",
    ],
)
def test_the_bypass_guard_catches(source: str) -> None:
    assert embedding_bypasses(source), source


@pytest.mark.parametrize(
    "source",
    [
        "from app.llm.core import openai_embeddings\nopenai_embeddings.create(c, 'm', [])",
        "from app.llm.core.openai_embeddings import create",
    ],
)
def test_the_create_guard_catches(source: str) -> None:
    assert direct_creates(source), source


def test_the_create_guard_allows_the_other_names_of_the_module() -> None:
    assert not direct_creates("from app.llm.core.openai_embeddings import openai_client, kind_of")
    assert not direct_creates(
        "from app.llm.core import openai_embeddings\nopenai_embeddings.kind_of(e)"
    )
