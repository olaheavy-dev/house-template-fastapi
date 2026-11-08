# 0004. MAF conversation threads and workflow checkpoints live in Postgres, without pickle

Status: Proposed. [#35](https://github.com/olaheavy-dev/house-template-fastapi/issues/35) lists these as gaps to build by hand, but the issue does not record a founder decision on the design.

## Context

Conversation history must live in the project's own database, not on the provider's servers. For OpenAI's Responses client, the spike found MAF sends no `store` by default and OpenAI's default is `true`; passing `options={"store": False}` keeps history with us ([#32](https://github.com/olaheavy-dev/house-template-fastapi/issues/32), item 5). As of the planner's review on 6 Nov 2026, MAF has no Postgres history provider for agent sessions and no Postgres workflow `CheckpointStorage` ([#35](https://github.com/olaheavy-dev/house-template-fastapi/issues/35)). On the LangChain side, threads already use LangGraph's Postgres checkpointer (README).

## Decision (proposed)

- **Threads:** a `PostgresSessionStore` keeps the session as JSON per `thread_id`, with a migration. The thread id is derived on the server from the current user, never taken from the caller. 5b in #35 also persists the full `AgentSession` state (it holds pending approvals).
- **Workflow checkpoints:** a Postgres `CheckpointStorage` that uses no pickle: JSON or a strict allow-list of types. In practice: loading a checkpoint can only build types we named. Python's own documentation warns that unpickling untrusted data can run arbitrary code ([pickle docs](https://docs.python.org/3/library/pickle.html)).
- Checkpoint rows contain conversation text, so they are deleted with the user's data; resuming from an earlier checkpoint, not only the latest, is in scope (5c).
- Images in a thread are a pointer to a stored file, not bytes (founder answer, 8 Nov 2026, recorded in #73 and noted on #35). The store must pass the shared image-in-thread parity test on both frameworks.

## Open questions

1. JSON or a strict allow-list for checkpoints (the issue says "JSON or a strict allow-list").
2. The table layout and migration for sessions and checkpoints (not designed in the issues).
3. Whether the pinned MAF version's checkpoint interface allows this: **unverified**, to be read from the docs for the pinned version first. Session history written by a newer MAF may not be readable by an older one ([MAF Storage docs](https://learn.microsoft.com/en-us/agent-framework/concepts/agents/conversations/storage)), so rollbacks need care.

## What exists today

Nothing of this is built: [#35](https://github.com/olaheavy-dev/house-template-fastapi/issues/35) sub-tasks 5b (sessions) and 5c (checkpoints) are open, and there is no MAF adapter yet.

## Consequences

- Rows stay under our control for deletion, export and budget accounting.
- We maintain two small storage classes until MAF ships its own.

## Alternatives considered

- Let the provider store conversations (the OpenAI default): history outside our database, a privacy risk if the setting is forgotten.
- Pickle for checkpoints: simplest, but unsafe for data read back from a database.
- Redis sessions: left out by the founder (comment on #35, 6 Nov 2026).

## Sources

- [#35](https://github.com/olaheavy-dev/house-template-fastapi/issues/35) and its comments; spike [#32](https://github.com/olaheavy-dev/house-template-fastapi/issues/32).
- pickle: https://docs.python.org/3/library/pickle.html
- MAF Storage docs: https://learn.microsoft.com/en-us/agent-framework/concepts/agents/conversations/storage
