# 0004. MAF conversation threads and workflow checkpoints live in Postgres, without pickle

Status: Proposed. Part of it is decided; the storage design is not.

**Decided**
- History lives in our own database, with `store: False` sent on every OpenAI Responses call (founder, 6 Nov 2026, [#35](https://github.com/olaheavy-dev/house-template-fastapi/issues/35) "Decisions already made").
- No Redis sessions (founder, 6 Nov 2026, comment on #35).
- Image threads keep a pointer to a stored file, not bytes, and a shared image-in-thread parity test is required on both frameworks (founder, 8 Nov 2026, recorded in [#73](https://github.com/olaheavy-dev/house-template-fastapi/issues/73)).

**Open:** the questions below.

## Context

For OpenAI's Responses client, the spike found MAF sends no `store` by default and OpenAI's default is `true`; passing `options={"store": False}` keeps history with us ([#32](https://github.com/olaheavy-dev/house-template-fastapi/issues/32), item 5). As of the planner's review on 6 Nov 2026, MAF has no Postgres history provider for agent sessions and no Postgres workflow `CheckpointStorage` ([#35](https://github.com/olaheavy-dev/house-template-fastapi/issues/35)). On the LangChain side, threads already use LangGraph's Postgres checkpointer (README).

## Decision (proposed)

- **Threads:** a `PostgresSessionStore` keeps the session as JSON per `thread_id`, with a migration. The thread id is derived on the server from the current user, never taken from the caller ([#35](https://github.com/olaheavy-dev/house-template-fastapi/issues/35) 5b: "thread ids are server-derived"; the template's rule: "Derive or check every `thread_id` server-side", `template/.claude/rules/fastapi.md.jinja` line 37). 5b also persists the full `AgentSession` state (it holds pending approvals).
- **Workflow checkpoints:** a Postgres `CheckpointStorage` that uses no pickle. In practice: loading a checkpoint can only build data we expect. Python's own documentation warns that unpickling untrusted data can run arbitrary code ([pickle docs](https://docs.python.org/3/library/pickle.html)).
- Checkpoint rows contain conversation text, so they are deleted with the user's data; resuming from an earlier checkpoint, not only the latest, is in scope (5c).
- Images in a thread are a pointer to a stored file. That file lives in the template's storage option, so image threads need the `storage` answer on ([#73](https://github.com/olaheavy-dev/house-template-fastapi/issues/73)). The store must pass the shared image-in-thread parity test.

## Open questions, with a recommendation each

1. **JSON or a strict allow-list for checkpoints?** JSON means checkpoints are plain data that a reader parses, with no code path that can build arbitrary objects. A strict allow-list means a loader that rebuilds typed objects but only from a list of names we maintain. Recommendation: JSON. MAF's Session page talks about "serialized" sessions that must be stored as trusted application state ([Session docs](https://learn.microsoft.com/en-us/agent-framework/concepts/agents/conversations/session)); `AgentSession.to_dict()`/`from_dict()` are named only in a #35 comment on a course repository, which is "ideas only, re-verify". An allow-list is a pickle-like loader we would have to police. Unverified for checkpoints until the pinned docs are read (question 3).
2. **Table layout and migration for sessions and checkpoints.** No recommendation until question 3: the layout depends on what the pinned interfaces store. The shape for sessions is already set: one JSON document per `thread_id`.
3. **Does the pinned MAF version's checkpoint interface allow this?** Unverified. Recommendation: read the docs for the pinned version first (the project rule), then settle 1 and 2. Session history written by a newer MAF may not be readable by an older one ([MAF Storage docs](https://learn.microsoft.com/en-us/agent-framework/concepts/agents/conversations/storage)), so rollbacks need care.

## What exists today

Nothing of this is built: [#35](https://github.com/olaheavy-dev/house-template-fastapi/issues/35) sub-tasks 5b (sessions) and 5c (checkpoints) are open, and there is no MAF adapter yet.

Separately, the LangChain side already loads its checkpoints with LangGraph's strict allowlist ([#36](https://github.com/olaheavy-dev/house-template-fastapi/issues/36)), so a MAF store should be no less strict (question 1).

## Consequences

- Rows stay under our control for deletion, export and budget accounting.
- We maintain two small storage classes until MAF ships its own.

## Alternatives considered

- Let the provider store conversations (the OpenAI default): history outside our database, a privacy risk if the setting is forgotten. **Rejected** by the founder decision of 6 Nov 2026 ("server-side storage off").
- Pickle for checkpoints: simplest, but unsafe for data read back from a database.
- Redis sessions: left out by the founder (comment on #35, 6 Nov 2026).

## Sources

- [#35](https://github.com/olaheavy-dev/house-template-fastapi/issues/35) and its comments; [#73](https://github.com/olaheavy-dev/house-template-fastapi/issues/73); spike [#32](https://github.com/olaheavy-dev/house-template-fastapi/issues/32).
- pickle: https://docs.python.org/3/library/pickle.html
- MAF Session docs: https://learn.microsoft.com/en-us/agent-framework/concepts/agents/conversations/session
- MAF Storage docs: https://learn.microsoft.com/en-us/agent-framework/concepts/agents/conversations/storage
