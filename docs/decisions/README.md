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

## Register

| Date | Choice | Why |
| --- | --- | --- |
| 2026-11-08 | Strict checkpoint loading is set in code (`serializer()` in `app/llm/langchain/checkpoints.py`), not with `LANGGRAPH_STRICT_MSGPACK`; a project lists its own classes in `app/llm/checkpoint_types.py`, nothing is allowed by default (#36) | LangGraph reads the variable once, at import, and `.env` never sets it under `make dev` or the tests, so dev and tests would load differently from production. Not equivalent: with the variable on, `StateGraph.compile` derives an allowlist from the graph's state types (pydantic models, dataclasses, enums); the explicit setting gets none, so a project must list every state class itself. Easy to reverse: drop `serde=` |
