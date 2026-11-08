# 0001. A framework-neutral core with framework adapters, chosen by the `llm` answer

Status: Accepted, for the split into a core and adapters only (founder decisions on [#33](https://github.com/olaheavy-dev/house-template-fastapi/issues/33)). **Not covered by this status:** the `llm` answer's details, which are still open on [#30](https://github.com/olaheavy-dev/house-template-fastapi/issues/30): the final extra name, and how `llm` is tied to `database`.

## Context

The model gateway was written around LangChain. The founder's projects will also use Microsoft Agent Framework (MAF, [#35](https://github.com/olaheavy-dev/house-template-fastapi/issues/35)). The money and safety parts (price arithmetic, the daily budget, failure kinds, request bounds, metrics) do not depend on either framework, so copying them per framework would let two copies drift apart.

## Decision

- Everything framework-free lives in `app/llm/core/`: `usage`, `prices` (arithmetic), `registry`, `errors`, `budget` (with the `Meter`: reserve, settle, failed), `bounds`, `metrics`, `prompts`, `embeddings`, `openai_embeddings` (the one core module that imports `openai`) and `report`. Each framework gets an adapter beside it: `app/llm/langchain/` today, a MAF adapter later ([#33 plan](https://github.com/olaheavy-dev/house-template-fastapi/issues/33)).
- A project picks the framework with one Copier answer, `llm: none | langchain | maf`. In practice: `none` leaves out the whole gateway, `langchain` or `maf` brings that adapter and its packages ([#30](https://github.com/olaheavy-dev/house-template-fastapi/issues/30), amendment of 6 Nov 2026).
- An import-linter contract (`backend/llm-contracts.toml`) fails `make check` if the core imports a model framework.
- Price tables stay in the project-owned `app/llm/prices.py`, so a project's own entries survive updates (founder decision Q1, [#33](https://github.com/olaheavy-dev/house-template-fastapi/issues/33)).

## Where the work stands

Progress goes stale quickly, so it is not recorded here. Follow it on [#33](https://github.com/olaheavy-dev/house-template-fastapi/issues/33) (the core and the LangChain adapter), [#30](https://github.com/olaheavy-dev/house-template-fastapi/issues/30) (the `llm` answer, which does not exist yet: `copier.yml` has only `database` and `storage`, so the answer part of this record is the target design) and [#35](https://github.com/olaheavy-dev/house-template-fastapi/issues/35) (the MAF adapter, which does not exist yet).

## Consequences

- A second framework reuses the same budget, prices and error rules, so cost behaviour is the same whichever one a project picks.
- Old import paths (`app.llm.budget`, `app.llm.errors` and others) are shims that warn with `DeprecationWarning`; the plan removes them in 0.8.0 ([#33](https://github.com/olaheavy-dev/house-template-fastapi/issues/33)). Several of the PRs are breaking (`refactor!`).
- The #30 comments say `llm` must refuse `database: false`, because the budget tables and checkpoints are in PostgreSQL; the final coupling is still open there.

## Alternatives considered

- Keep everything in the LangChain gateway and copy it for MAF: quicker at first, but two budgets to keep equal. Not chosen.
- Make the choice later per project by hand: no tested update path. Not chosen.

## Sources

- Plan and founder decisions: [#33](https://github.com/olaheavy-dev/house-template-fastapi/issues/33), [#30](https://github.com/olaheavy-dev/house-template-fastapi/issues/30), [#35](https://github.com/olaheavy-dev/house-template-fastapi/issues/35).
- Copier, updating a project (version 9.19.1, as read in the spike): https://copier.readthedocs.io/en/v9.19.1/updating/
