"""MAF workflow checkpoints in Postgres (#35 PR 6, record 0007): a workflow resumes from a stored
checkpoint after a restart, from the newest or an earlier one; a stored row cannot run code; a user
sees only their own checkpoints. The values are synthetic.

The workflow counts down: the start executor sends Countdown(3), the worker records each number in
its executor state and sends the next one until 0, then yields the record. Each superstep ends with
a checkpoint, so the run leaves several, the first taken before the worker ran.
"""

from __future__ import annotations

import ast
import base64
import pickle
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from agent_framework import (
    Content,
    Executor,
    Message,
    WorkflowBuilder,
    WorkflowContext,
    handler,
)
from agent_framework._workflows._checkpoint import WorkflowCheckpoint
from agent_framework._workflows._checkpoint_encoding import decode_checkpoint_value
from agent_framework.exceptions import WorkflowCheckpointException
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.auth import User
from app.llm import workflow_checkpoint_types
from app.llm.core.checkpoints import WorkflowCheckpointRow
from app.llm.core.sessions import AgentSessionRow
from app.llm.maf.gateway import Gateway
from app.llm.maf.media import InlineMedia
from app.settings import Settings
from tests.harness import Reply
from tests.harness.maf import HARNESS

pytestmark = pytest.mark.anyio

ALICE, BOB = User(id="u_alice"), User(id="u_bob")
COUNTDOWN = f"{__name__}:Countdown"
FIRED: list[str] = []


def fire(note: str) -> None:
    """What a planted payload calls: a module-level function, which pickle names by reference, as an
    attacker's `os.system` would be."""
    FIRED.append(note)


@dataclass
class Countdown:
    remaining: int


@dataclass
class Unlisted:
    note: str = "not on the allow-list"


class Start(Executor):
    @handler
    async def start(self, n: int, ctx: WorkflowContext[Countdown]) -> None:
        await ctx.send_message(Countdown(n))


@dataclass
class _Seen:
    numbers: list[int] = field(default_factory=list[int])


class Worker(Executor):
    def __init__(self, id: str) -> None:
        super().__init__(id=id)
        self.seen: list[int] = []

    @handler
    async def step(self, task: Countdown, ctx: WorkflowContext[Countdown, list[int]]) -> None:
        self.seen.append(task.remaining)
        if task.remaining == 0:
            await ctx.yield_output(list(self.seen))
        else:
            await ctx.send_message(Countdown(task.remaining - 1))

    async def on_checkpoint_save(self) -> dict[str, Any]:
        return {"seen": list(self.seen)}

    async def on_checkpoint_restore(self, state: dict[str, Any]) -> None:
        self.seen = list(state.get("seen", []))


def workflow(storage: Any = None) -> Any:
    start, worker = Start(id="start"), Worker(id="worker")
    return (
        WorkflowBuilder(start_executor=start, checkpoint_storage=storage, name="countdown")
        .add_edge(start, worker)
        .add_edge(worker, worker)
        .build()
    )


async def outputs(run: Any) -> list[Any]:
    return [event.data async for event in run if event.type == "output"]


@pytest.fixture
def sessions(db_session: AsyncSession) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(
        bind=db_session.bind, join_transaction_mode="create_savepoint", expire_on_commit=False
    )


@pytest.fixture
def make(
    settings: Settings,
    sessions: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> Any:
    # The project lists the type its workflow passes between executors.
    monkeypatch.setattr(workflow_checkpoint_types, "ALLOWED", frozenset({COUNTDOWN}))

    def build() -> Gateway:
        return Gateway(
            settings.model_copy(update={"llm_model": "anthropic:claude-sonnet-5-5"}),
            lambda: sessions,
        )

    return build


async def test_a_workflow_resumes_after_a_restart_from_the_latest_or_an_earlier_checkpoint(
    make: Any,
) -> None:
    first = make().checkpoints(ALICE)
    assert await outputs(workflow(first).run(3, stream=True)) == [[3, 2, 1, 0]]
    saved = await first.list_checkpoints(workflow_name="countdown")
    assert len(saved) >= 3

    restarted = make().checkpoints(ALICE)  # a new gateway: everything comes back from Postgres
    latest = await restarted.get_latest(workflow_name="countdown")
    assert latest is not None
    earliest = (await restarted.list_checkpoints(workflow_name="countdown"))[0]
    for checkpoint in (latest, earliest):
        resumed = workflow().run(
            checkpoint_id=checkpoint.checkpoint_id, checkpoint_storage=restarted, stream=True
        )
        result = await outputs(resumed)
        # From the latest there is nothing left to do; from an earlier one the rest of the
        # countdown runs again, with the executor state restored.
        assert result in ([], [[3, 2, 1, 0]])
    from_earliest = workflow().run(
        checkpoint_id=earliest.checkpoint_id, checkpoint_storage=restarted, stream=True
    )
    assert await outputs(from_earliest) == [[3, 2, 1, 0]]


async def test_a_stored_row_that_would_run_code_is_refused_and_runs_nothing(
    make: Any, db_session: AsyncSession
) -> None:
    storage = make().checkpoints(ALICE)
    await outputs(workflow(storage).run(1, stream=True))
    [first, *_] = await storage.list_checkpoints(workflow_name="countdown")

    class Payload:
        def __reduce__(self) -> tuple[Any, tuple[str]]:
            return (fire, ("code ran",))

    planted = {
        "__pickled__": base64.b64encode(pickle.dumps(Payload())).decode("ascii"),
        "__type__": "builtins:NoneType",
    }
    # The payload is real: MAF's decoder without an allow-list (plain pickle.loads) runs it.
    decode_checkpoint_value(planted, allowed_types=None)
    assert FIRED == ["code ran"]
    FIRED.clear()

    row = await db_session.get(WorkflowCheckpointRow, (ALICE.id, first.checkpoint_id))
    assert row is not None
    document = dict(row.document)
    document["state"] = {**document["state"], "planted": planted}
    await db_session.execute(
        update(WorkflowCheckpointRow)
        .where(WorkflowCheckpointRow.checkpoint_id == first.checkpoint_id)
        .values(document=document)
    )
    with pytest.raises(WorkflowCheckpointException):
        await storage.load(first.checkpoint_id)
    assert FIRED == []


async def test_a_checkpoint_holding_an_unlisted_type_is_refused_when_saved(
    make: Any, db_session: AsyncSession
) -> None:
    storage = make().checkpoints(ALICE)
    checkpoint = WorkflowCheckpoint(
        workflow_name="countdown", graph_signature_hash="h", state={"x": Unlisted()}
    )
    with pytest.raises(WorkflowCheckpointException, match="blocked for type"):
        await storage.save(checkpoint)
    assert await db_session.scalar(select(func.count()).select_from(WorkflowCheckpointRow)) == 0
    listed = WorkflowCheckpoint(
        workflow_name="countdown", graph_signature_hash="h", state={"x": Countdown(1)}
    )
    await storage.save(listed)
    assert (await storage.load(listed.checkpoint_id)).state["x"] == Countdown(1)


async def test_a_user_sees_only_their_own_checkpoints(make: Any) -> None:
    gateway = make()
    alice = gateway.checkpoints(ALICE)
    await outputs(workflow(alice).run(1, stream=True))
    [checkpoint, *_] = await alice.list_checkpoints(workflow_name="countdown")
    bob = gateway.checkpoints(BOB)
    assert await bob.list_checkpoints(workflow_name="countdown") == []
    assert await bob.list_checkpoint_ids(workflow_name="countdown") == []
    assert await bob.get_latest(workflow_name="countdown") is None
    with pytest.raises(WorkflowCheckpointException, match="No checkpoint"):
        await bob.load(checkpoint.checkpoint_id)
    assert await bob.delete(checkpoint.checkpoint_id) is False
    assert (await alice.load(checkpoint.checkpoint_id)).checkpoint_id == checkpoint.checkpoint_id


async def test_the_latest_of_checkpoints_taken_at_once_follows_the_lineage(make: Any) -> None:
    # Two checkpoints can share a timestamp (a pending request and the response entry); MAF's rule
    # picks the one the other does not precede, whatever the save order.
    storage = make().checkpoints(ALICE)
    moment = datetime(2026, 1, 13, 12, tzinfo=UTC).isoformat()
    parent = WorkflowCheckpoint(
        workflow_name="countdown", graph_signature_hash="h", checkpoint_id="z", timestamp=moment
    )
    child = WorkflowCheckpoint(
        workflow_name="countdown",
        graph_signature_hash="h",
        checkpoint_id="a",
        previous_checkpoint_id="z",
        timestamp=moment,
    )
    await storage.save(child)
    await storage.save(parent)
    latest = await storage.get_latest(workflow_name="countdown")
    assert latest is not None
    assert latest.checkpoint_id == "a"


async def test_a_users_data_is_deleted_with_their_sessions_and_checkpoints(
    make: Any, db_session: AsyncSession
) -> None:
    gateway = make()
    HARNESS.install_scripted(gateway, [Reply("hi")])
    agent = gateway.agent([], name="helper")
    session = await gateway.sessions.load(ALICE, "c1", agent)
    await agent.run("hello", session=session)
    await gateway.sessions.save(ALICE, "c1", agent, session)
    await outputs(workflow(gateway.checkpoints(ALICE)).run(1, stream=True))
    await outputs(workflow(gateway.checkpoints(BOB)).run(1, stream=True))
    sessions_gone, checkpoints_gone = await gateway.delete_user_data(ALICE.id)
    assert sessions_gone == 1
    assert checkpoints_gone >= 2
    owners = set(await db_session.scalars(select(WorkflowCheckpointRow.user_ref)))
    assert owners == {BOB.id}
    assert await db_session.scalar(select(func.count()).select_from(AgentSessionRow)) == 0


async def test_a_users_own_data_that_looks_like_an_envelope_comes_back_as_data(make: Any) -> None:
    # MAF pickles a plain dict that uses its reserved keys, so such a dict in the state (say, JSON
    # a user sent) is stored as data and never read as an envelope.
    storage = make().checkpoints(ALICE)
    lookalike = {"__pickled__": "AAAA", "__type__": "os:system"}
    checkpoint = WorkflowCheckpoint(
        workflow_name="countdown", graph_signature_hash="h", state={"input": lookalike}
    )
    await storage.save(checkpoint)
    assert (await storage.load(checkpoint.checkpoint_id)).state["input"] == lookalike


def test_the_private_maf_names_the_storage_uses_exist_in_the_pinned_core() -> None:
    # app/llm/maf/checkpoints.py imports these from agent-framework-core 1.25.0's private modules
    # (founder's decision on #35); a version that moves them must fail here.
    from agent_framework._workflows import _checkpoint, _checkpoint_encoding

    for module, name in (
        (_checkpoint_encoding, "encode_checkpoint_value"),
        (_checkpoint_encoding, "decode_checkpoint_value"),
        (_checkpoint, "_select_latest_checkpoint"),
        (_checkpoint, "WorkflowCheckpoint"),
    ):
        assert hasattr(module, name), name


def test_every_decode_in_the_gateway_passes_an_allow_list() -> None:
    # None means plain pickle.loads: no call in app/llm may leave allowed_types out or pass None.
    root = Path(__file__).parents[1] / "src" / "app" / "llm"
    calls = []
    for path in root.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if (
                isinstance(node, ast.Call)
                and getattr(node.func, "id", getattr(node.func, "attr", None))
                == "decode_checkpoint_value"
            ):
                calls.append((path.name, node))
    assert calls, "the storage decodes checkpoints"
    for name, call in calls:
        allowed = [k for k in call.keywords if k.arg == "allowed_types"]
        assert allowed, name
        value = allowed[0].value
        assert not (isinstance(value, ast.Constant) and value.value is None), name


@pytest.mark.parametrize(
    "state",
    [
        {"history": [Message("user", [Content.from_data(b"\x89PNG", media_type="image/png")])]},
        {
            "_executor_state": {
                "worker": {
                    "last": Content.from_uri(
                        "DATA:image/png;base64,iVBORw0KGgo=", media_type="image/png"
                    )
                }
            }
        },
    ],
    ids=["message-with-bytes", "executor-state-with-a-data-uri"],
)
async def test_a_checkpoint_holding_a_files_bytes_is_refused_and_nothing_is_stored(
    make: Any, db_session: AsyncSession, state: dict[str, Any]
) -> None:
    # #73 and record 0007: a stored workflow keeps a pointer to a file, as a stored session does
    # (PR #121 review). MAF would pickle these objects as they are, bytes and all.
    storage = make().checkpoints(ALICE)
    checkpoint = WorkflowCheckpoint(
        workflow_name="countdown", graph_signature_hash="h", state=state
    )
    with pytest.raises(InlineMedia):
        await storage.save(checkpoint)
    assert await db_session.scalar(select(func.count()).select_from(WorkflowCheckpointRow)) == 0


async def test_a_checkpoint_holding_a_pointer_to_a_file_is_stored(make: Any) -> None:
    storage = make().checkpoints(ALICE)
    pointer = Content.from_uri("https://files.example/rooms/1.png", media_type="image/png")
    checkpoint = WorkflowCheckpoint(
        workflow_name="countdown",
        graph_signature_hash="h",
        state={"history": [Message("user", [pointer, "Which room?"])]},
    )
    await storage.save(checkpoint)
    [message] = (await storage.load(checkpoint.checkpoint_id)).state["history"]
    assert message.contents[0].uri == "https://files.example/rooms/1.png"
