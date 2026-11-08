# 0001. A framework-neutral core with framework adapters, chosen by the `llm` answer

Status: Accepted (the split; the founder decisions are recorded on [#33](https://github.com/olaheavy-dev/house-template-fastapi/issues/33)). The `llm` answer itself is not built yet: see "What exists today".

## Context

The model gateway was written around LangChain. The founder's projects will also use Microsoft Agent Framework (MAF, [#35](https://github.com/olaheavy-dev/house-template-fastapi/issues/35)). The money and safety parts (price arithmetic, the daily budget, failure kinds, request bounds, metrics) do not depend on either framework, so copying them per framework would let two copies drift apart.

## Decision

- Everything framework-free lives in `app/llm/core/`: `usage`, `prices` (arithmetic), `registry`, `errors`, `budget` (with the `Meter`: reserve, settle, failed), `bounds`, `metrics`, `embeddings` and `openai_embeddings` (the one place that imports `openai`). Each framework gets an adapter beside it: `app/llm/langchain/` today, a MAF adapter later ([#33 plan](https://github.com/olaheavy-dev/house-template-fastapi/issues/33)).
- A project picks the framework with one Copier answer, `llm: none | langchain | maf`. In practice: `none` leaves out the whole gateway, `langchain` or `maf` brings that adapter and its packages ([#30](https://github.com/olaheavy-dev/house-template-fastapi/issues/30), amendment of 6 Nov 2026).
- An import-linter contract (`backend/llm-contracts.toml`) fails `make check` if the core imports a model framework.
- Price tables stay in the project-owned `app/llm/prices.py`, so a project's own entries survive updates (founder decision Q1, [#33](https://github.com/olaheavy-dev/house-template-fastapi/issues/33)).

## What exists today

- The core and the LangChain adapter are built: T1 [#65](https://github.com/olaheavy-dev/house-template-fastapi/pull/65), T2 [#66](https://github.com/olaheavy-dev/house-template-fastapi/pull/66), T3 [#67](https://github.com/olaheavy-dev/house-template-fastapi/pull/67), T4 [#70](https://github.com/olaheavy-dev/house-template-fastapi/pull/70). Prompts, gateway, checkpoints and report (T5) and docs (T6) are still open on #33.
- The `llm` answer does not exist yet. `copier.yml` has `database` and `storage`, and the gateway is switched off with `database: false`. Until [#30](https://github.com/olaheavy-dev/house-template-fastapi/issues/30) is done, this part of the record is the target design.
- There is no MAF adapter yet ([#35](https://github.com/olaheavy-dev/house-template-fastapi/issues/35)).

## Consequences

- A second framework reuses the same budget, prices and error rules, so cost behaviour is the same whichever one a project picks.
- Old import paths (`app.llm.budget`, `app.llm.errors` and others) are shims that warn with `DeprecationWarning`; the plan removes them in 0.8.0 ([#33](https://github.com/olaheavy-dev/house-template-fastapi/issues/33)). Several PRs are breaking (`refactor!`).
- Still open, decided by #30: the final extra name and how `llm` is tied to `database`. The #30 comments say `llm` must refuse `database: false`, because the budget tables and checkpoints are in PostgreSQL.

## Alternatives considered

- Keep everything in the LangChain gateway and copy it for MAF: quicker at first, but two budgets to keep equal. Not chosen.
- Make the choice later per project by hand: no tested update path. Not chosen.

## Sources

- Plan and founder decisions: [#33](https://github.com/olaheavy-dev/house-template-fastapi/issues/33), [#30](https://github.com/olaheavy-dev/house-template-fastapi/issues/30), [#35](https://github.com/olaheavy-dev/house-template-fastapi/issues/35).
- Copier, updating a project: https://copier.readthedocs.io/en/stable/updating/
