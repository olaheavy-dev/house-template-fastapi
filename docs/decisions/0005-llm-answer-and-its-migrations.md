# 0005. The `llm` answer, and what it does to the database and its migrations

Status: Accepted (founder decisions, 9 Nov 2026, recorded on [#30](https://github.com/olaheavy-dev/house-template-fastapi/issues/30)). It settles the part of [0001](0001-framework-neutral-core-with-adapters.md) that was left open: the final extra names and how `llm` is tied to `database`. 0001 is not rewritten.

## Context

Not every project calls a model. The gateway brings LangChain, LangGraph, LangSmith, two provider packages, three budget tables, the agent checkpoint tables, prompts, cassettes and evals. A project that never calls a model should not carry them. The gateway keeps its budget and its agent checkpoints in PostgreSQL, so it cannot exist without the database. The database's migrations are one chain shared by both: 0001 enables pgvector, 0002 to 0005 are the gateway's tables, and projects that already ran them must keep a chain Alembic can follow.

## Decision

- **A choice answer, `llm: none | langchain`**, asked after `database`. Its default is `langchain` with the database and `none` without it, so every existing project updates to what it had. A choice validator refuses `langchain` without `database: true`; the question has no `when`, so `llm: none` is recorded in the answers file too. `maf` joins as a third choice in [#35](https://github.com/olaheavy-dev/house-template-fastapi/issues/35) with the same validator.
- **`database: false` always means `llm: none`.** There are three valid shapes (database with the gateway, database without it, neither), each with storage on or off. `-d database=true` no longer brings the gateway back by itself: add `-d llm=langchain`. `-d database=false` alone sets `llm: none` (Copier drops a last answer that fails validation and recomputes it from the default; this is not documented, so a test guards it).
- **With `llm: none` the database keeps its migrations, and the gateway's four become empty.** Revisions 0002 to 0005 stay in the chain with the same ids and parents and do nothing, so a fresh project has no dead tables, and a project that ran the gateway keeps a chain Alembic can follow. `models.py` maps only `Base`. `migrations/env.py` ignores the gateway's tables (`llm_call`, `llm_spend_day`, `llm_spend_user_day` and the four checkpoint tables), so a project that ran the gateway and turned it off never sees autogenerate propose dropping them. Each of these files is a `.jinja` twin of a plain all-on file.
- **Tables are never dropped automatically.** A project that turns the gateway off and wants them gone runs `alembic downgrade 0001` first. A database that ran without the tables and later turns the gateway on runs `alembic stamp 0001` and then `alembic upgrade heads` (tested in `backend/tests/test_llm_migrations.py`).
- **Extras stay `db`, `langchain` and `storage`, with no coupling in `pyproject.toml`.** The Makefile and the image add `db` whenever they add `langchain`. Dev tools (`langsmith[pytest]`, `vcrpy`, `pytest-recording`) stay in the `dev` group for every project.
- **Settings with `llm: none` drop every model, embedding, key, agent and LangSmith field, and the `AGENT_MAX_STEPS` refusal.** Leftover variables are ignored (`extra="ignore"`).

## Consequences

- With the default answers the output is the same as before #30 apart from the answers file, the rules file and one new test (checked against the commit before it, in `tests/test_golden_render.py`). Existing projects update without a file being removed.
- Turning the gateway off on update deletes its files even if edited, including `prices.py` and `checkpoint_types.py`, and changes `models.py`, `env.py` and four migrations (conflicts are likely where a project edited them). The README's Upgrading section says so.
- Every future gateway migration (including MAF's) must follow the pattern: a twin, the same revision id, an empty body without the gateway, and its tables added to `env.py`'s ignored set. Switching from LangChain to MAF leaves the other framework's tables behind.
- The `dev` group still installs LangSmith's pytest plugin, `vcrpy` and `pytest-recording` for a project with `llm: none`, which the pinned versions keep inert (the plugin does nothing without the conftest lines that enable it); an optional group is a follow-up.

## Alternatives considered

- **Keep the gateway's migrations whole under `llm: none`** (an unused budget table). Not chosen: dead tables in every project that never calls a model.
- **Drop 0002 to 0005 from the chain under `none`.** Not chosen: a project that already ran them would have a database stamped at a revision its files no longer contain.
- **Make `database: true` bring the gateway back by itself.** Not chosen: turning the database on would install LangChain and create budget tables nobody asked for.
- **Couple the extras in `pyproject.toml`** (`langchain` requiring `db`). Not chosen: uv offers no clean way to express this across extras, and the validator and the install commands already make sure they come together.
- **A separate yes/no question.** Not chosen: [0001](0001-framework-neutral-core-with-adapters.md) wants one answer that later adds `maf`.

## Sources

- Plan and founder decisions: [#30](https://github.com/olaheavy-dev/house-template-fastapi/issues/30).
- Copier, questions, `choices` with `validator`, `default`: https://copier.readthedocs.io/en/v9.19.1/configuring/ (Copier 9.19.1).
- Alembic, `stamp` and `downgrade`: https://alembic.sqlalchemy.org/en/latest/tutorial.html and https://alembic.sqlalchemy.org/en/latest/branches.html
- [0001](0001-framework-neutral-core-with-adapters.md), [0002](0002-dependencies-as-extras-in-one-lockfile.md).
