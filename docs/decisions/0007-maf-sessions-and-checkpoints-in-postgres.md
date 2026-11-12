# 0007. MAF sessions and workflow checkpoints live in Postgres, per user; checkpoints load only through MAF's restricted unpickler

Status: Accepted. The founder decided how checkpoints load on [#35](https://github.com/olaheavy-dev/house-template-fastapi/issues/35) (11 Nov 2026: "import MAF's private restricted encode/decode functions, packages pinned exactly, with a guard test") and approved the plan in which this record supersedes [0004](0004-postgres-threads-and-checkpoints-without-pickle.md). This text was drafted by an agent and is merged by the founder. It answers 0004's three open questions with what the pinned packages allow.

## Context

Record 0004 decided that MAF's conversation history and workflow checkpoints live in our own database (not at the provider, not in Redis), that a thread id is derived on the server, that images in a thread are pointers to stored files ([#73](https://github.com/olaheavy-dev/house-template-fastapi/issues/73)), and that rows holding conversation text are deleted with the user's data. It left open the storage design: JSON or an allow-list for checkpoints, the table layout, and what the pinned MAF interfaces allow.

What the pinned packages (agent-framework-core 1.25.0, read in their source and checked against the official docs, 12 Nov 2026) provide:

- **Sessions.** `AgentSession.to_dict()` gives plain JSON: the session id and its `state`, which holds the history (`InMemoryHistoryProvider` keeps `Message` objects there) and pending tool approvals. `AgentSession.from_dict()` rebuilds only types registered with `register_state_type` (`Message` is) and leaves any other `type` as plain data; there is no pickle and no import by name. MAF has no Postgres history provider for Python (Redis and Cosmos DB only). Its docs: "Persist the full session object, not only message text" and "bind each stored session to the authenticated user or tenant before allowing it to resume".
- **Checkpoints.** MAF's `CheckpointStorage` protocol has six async methods (save, load, list_checkpoints, delete, get_latest, list_checkpoint_ids). Its own stores (memory, file, Cosmos DB) encode a checkpoint as JSON with every non-JSON value pickled and embedded as base64, and load it with a restricted unpickler that allows a built-in safe set, MAF's and openai's types, and the types the store lists. MAF has no Postgres checkpoint store. The encoder and decoder (`encode_checkpoint_value`, `decode_checkpoint_value`) are not exported. `decode_checkpoint_value(..., allowed_types=None)` means plain `pickle.loads`. MAF's docs: "Checkpoint storage is a trust boundary… Never load checkpoints from untrusted or potentially tampered sources", and the restricted unpickler "is **not** a security boundary".
- So 0004's title, "without pickle", cannot hold for checkpoints if MAF's own format is used: the values a workflow passes between executors are pickled by MAF itself. The alternative is a custom serializer per type, which MAF's docs name only as the route when "your threat model does not permit pickle-based serialization at all".

## Decision

- **Sessions:** one `agent_session` row per user, conversation and agent name, holding MAF's own `AgentSession.to_dict()` as JSONB, read back with `AgentSession.from_dict` (`app/llm/maf/sessions.py`, migration 0007, [#120](https://github.com/olaheavy-dev/house-template-fastapi/pull/120)). The key is derived from the current user, never taken from a caller. Saves are checked against a version, so a second concurrent save of one conversation is a 409. A file's bytes (a data URI) are refused: a thread keeps a pointer.
- **Checkpoints:** one `workflow_checkpoint` row per user and checkpoint, holding MAF's own encoding as JSONB (`app/llm/maf/checkpoints.py`, migration 0008). The storage is MAF's `CheckpointStorage` for one user (`gateway.checkpoints(user)`): a user lists, loads and deletes only their own checkpoints. It loads every checkpoint through MAF's `decode_checkpoint_value` with an explicit allow-list, never `None`: MAF's built-in safe set plus the "module:qualname" strings in the project's `app/llm/workflow_checkpoint_types.py` (empty by default). It decodes each checkpoint under the same list before saving it, as MAF's file store does, so a checkpoint that could not be resumed fails when it is saved. A checkpoint that holds a file's bytes (a data URI anywhere in its messages or state) is refused, as a session is (`app/llm/maf/media.py`, shared by both stores). "Latest" uses MAF's own rule (the newest timestamp, ties settled by the lineage chain). Resuming from any earlier checkpoint is supported.
- **MAF's private functions are imported** (`encode_checkpoint_value`, `decode_checkpoint_value`, `_select_latest_checkpoint`), with the MAF packages pinned exactly (decision 0002) and a test that fails if they move. A test proves a planted `__reduce__` payload runs under plain pickle and is refused, running nothing, through the storage; another proves every `decode_checkpoint_value` call in the gateway passes an allow-list.
- **The trust boundary is the database.** Rows are written only by these stores, in the app's own database, scoped to the user. Checkpoint rows are never loaded from anything a client sent. A dict in a workflow's state that uses MAF's reserved keys (`__pickled__`, `__type__`), such as JSON a user typed, is pickled by MAF as a whole and comes back as data; a test pins it.
- **Both tables are created with any model gateway**, not only MAF's, so a project switched from LangChain to MAF by `copier update` has them; a LangChain project keeps them empty. `gateway.delete_user_data(user_ref)` deletes a user's sessions and checkpoints.

## Consequences

- A project lists the types its workflows pass and keep in `app/llm/workflow_checkpoint_types.py`; an unlisted type is refused at save time with MAF's message naming it.
- Anyone who can write rows into the database can plant a payload that the restricted unpickler might still let through (MAF says the allow-list is not a security boundary). Database write access is already the keys to everything else; the deployment must keep it so.
- MAF upgrades: the packages move together, by hand (0002). A newer MAF may write sessions an older one rejects ("don't run mixed old and new workers against the same session or history store"); each row records the package version that wrote it (`written_by`). Checkpoints from before MAF 1.13.0 are documented as still restorable; no general policy is documented.
- Two small storage classes are ours until MAF ships Postgres stores of its own.

## Alternatives considered

| Option | In practice | Verdict |
| --- | --- | --- |
| MAF's encoding with its restricted unpickler and an explicit allow-list, per user, in Postgres | MAF's own format and loader, as its file and Cosmos DB stores use them; private functions, so pinned and guarded | **Chosen** (founder, 11 Nov 2026) |
| A serializer of our own with no pickle (0004's recommendation) | every type a workflow passes needs our own codec; MAF's executor state and pending requests would need one too; drifts from MAF's format on every upgrade | Not now: MAF's docs offer it only when pickle is ruled out entirely |
| MAF's `FileCheckpointStorage` | no per-user scoping, files outside the database, not deleted with the user's data | No |
| Plain pickle (`allowed_types=None`) | simplest | No: a stored row runs code on load (a test shows it) |
| Redis | left out by the founder (6 Nov 2026) | No |

## Sources

- [#35](https://github.com/olaheavy-dev/house-template-fastapi/issues/35) (the founder's answers of 11 Nov 2026), [#73](https://github.com/olaheavy-dev/house-template-fastapi/issues/73), [0004](0004-postgres-threads-and-checkpoints-without-pickle.md).
- MAF sessions: https://learn.microsoft.com/en-us/agent-framework/concepts/agents/conversations/session
- MAF storage: https://learn.microsoft.com/en-us/agent-framework/concepts/agents/conversations/storage
- MAF workflow checkpoints (Python): https://learn.microsoft.com/en-us/agent-framework/workflows/checkpoints?pivots=programming-language-python
- MAF human-in-the-loop and checkpoints: https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop?pivots=programming-language-python
- MAF 1.13.0 checkpoint upgrade guide: https://learn.microsoft.com/en-us/agent-framework/support/upgrade/python-1-13-0-workflow-checkpoint-upgrade-guide
- agent-framework-core 1.25.0 source: `agent_framework/_sessions.py`, `agent_framework/_workflows/_checkpoint.py`, `agent_framework/_workflows/_checkpoint_encoding.py`.
- Python's pickle warning: https://docs.python.org/3/library/pickle.html
