# house-template-fastapi

The FastAPI layer of my Copier project templates. It adds a Python backend under `backend/` to a project generated from [house-template-base](https://github.com/olaheavy-dev/house-template-base), and plugs the backend's checks into the base `make check`, CI, CodeQL and Dependabot.

## Use it

```sh
copier copy --trust gh:olaheavy-dev/house-template-base my-app      # choose the fastapi stack
cd my-app && git init -b main && git add -A && git commit -m "chore: start from house-template-base"
copier copy --trust -a .copier-answers.fastapi.yml gh:olaheavy-dev/house-template-fastapi .
make install && make check      # the backend tests need Docker
make db-up && make migrate && make dev
```

The layer reads the project name and stacks from `.copier-answers.base.yml` and refuses to apply without them. Update it later with `copier update -a .copier-answers.fastapi.yml`. The project owns `backend/uv.lock`, so after an update that changes dependencies, run `make lock` and commit the lockfile before `make install`.

## What it adds

| Path | What |
|---|---|
| `backend/src/app/` | FastAPI app factory with a lifespan, settings from the environment, RFC 9457 problem responses, structured JSON logs with request ids, and live and ready health endpoints |
| `backend/src/app/db.py`, `models.py` | SQLAlchemy 2.1 async with psycopg 3: a declarative `Base` with named constraints, the engine opened in the lifespan, a session per request (`SessionDep`), and a database readiness check |
| `backend/migrations/` | Alembic (async, configured in `pyproject.toml`, no `alembic.ini`); the first migration enables pgvector, and new revisions are lint-clean |
| `backend/compose.yaml` | Local PostgreSQL 18 with pgvector 0.8.7, pinned by digest; included by base's root `compose.yaml` |
| `backend/tests/` | Async tests that drive the app in-process with its lifespan running, and database tests against a real PostgreSQL (testcontainers, or `TEST_DATABASE_URL`), each rolled back afterwards; `alembic check` and a downgrade/upgrade round trip run with them |
| `backend/pyproject.toml`, `backend/uv.lock` | uv project; ruff, mypy (strict), import-linter and pytest configuration |
| `mk/fastapi.mk` | `install-fastapi`, `check-fastapi`, `check-fast-fastapi`, `lock`, `dev`, `test`, `db-up`, `migrate`, `migration`, joined to the base targets |
| `.claude/rules/fastapi.md` | Backend rules for coding agents, loaded only when they touch `backend/` |
| `.python-version` | The backend's Python (3.14 by default; 3.12 is the oldest supported) |

Still to come, through reviewed pull requests: the budget-capped LangChain model gateway with LangSmith tracing and recorded responses; S3-compatible storage; the container image with the API and a one-off migration service in Compose; releases.

## Develop the template

`uv run pytest` renders the layer and checks it. CI also generates a real project from house-template-base plus this layer and runs its `make check`. After changing `template/backend/pyproject.toml`, run `uv lock --directory template/backend`.
