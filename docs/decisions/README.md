# Decisions

Hard-to-reverse choices this template makes for every project that uses it get a numbered record in this folder. The format is the base layer's: [0001 Record decisions](https://github.com/olaheavy-dev/house-template-base/blob/main/template/docs/decisions/0001-record-decisions.md) (Context, Decision, Consequences, Alternatives considered, Sources). Easily reversible choices get one line in the register below. Accepted records aren't rewritten: a new record supersedes an old one. A record whose status is Proposed lists its open questions and becomes Accepted only when the founder decides them.

These records describe the template repository itself. They are not copied into generated projects (the template's `_subdirectory` is `template/`). A test (`tests/test_decisions.py`) fails if a record file is missing from the table below, or a row has no file.

## Records

| No. | Decision | Status |
| --- | --- | --- |
| [0001](0001-framework-neutral-core-with-adapters.md) | A framework-neutral core with framework adapters, chosen by the `llm` answer | Accepted |
| [0002](0002-dependencies-as-extras-in-one-lockfile.md) | Dependencies are chosen by extras in one lockfile, with conflicts | Accepted |
| [0003](0003-step-limit-returns-a-final-answer.md) | Both frameworks return a final answer at the step limit | Accepted |
| [0004](0004-postgres-threads-and-checkpoints-without-pickle.md) | MAF conversation threads and workflow checkpoints live in Postgres, without pickle | Proposed |
| [0005](0005-llm-answer-and-its-migrations.md) | The `llm` answer: `none` or `langchain`, tied to the database; the gateway's migrations stay in the chain and do nothing without it | Accepted |

## Register

| Date | Choice | Why |
| --- | --- | --- |
| 2026-11-08 | Strict checkpoint loading is set in code (`serializer()` in `app/llm/langchain/checkpoints.py`), not with `LANGGRAPH_STRICT_MSGPACK`; a project lists its own classes in `app/llm/checkpoint_types.py`, nothing is allowed by default (#36) | LangGraph reads the variable once, at import, and `.env` never sets it under `make dev` or the tests, so dev and tests would load differently from production. Not equivalent: with the variable on, `StateGraph.compile` derives an allowlist from the graph's state types (pydantic models, dataclasses, enums); the explicit setting gets none, so a project must list every state class itself. Easy to reverse: drop `serde=` |
| 2026-11-09 | The agent limit counts model calls: `AGENT_MAX_STEPS` (LangGraph steps, default 25) is replaced by `AGENT_MAX_MODEL_CALLS` (default 13), and the app refuses to start while the old name is set (#34). A `GraphRecursionError` is a plain 500. The rule lives in `app/llm/core/steps.py`. 0003 is not rewritten; its "keep the number" consequence is overtaken by this rename | 25 steps allowed 13 model calls, so counting calls with a default of 13 keeps the worst-case spend of one looping run. Keeping the old name with a new meaning would silently double that for a project that set it. Easy to reverse: nothing is stored |
