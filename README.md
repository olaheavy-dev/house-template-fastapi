# house-template-fastapi

The FastAPI layer of my Copier project templates. It adds a Python backend under `backend/` to a project generated from [house-template-base](https://github.com/olaheavy-dev/house-template-base), and plugs the backend's checks into the base `make check`, CI, CodeQL and Dependabot.

## Use it

```sh
copier copy --trust gh:olaheavy-dev/house-template-base my-app      # choose the fastapi stack
cd my-app && git init -b main && git add -A && git commit -m "chore: start from house-template-base"
copier copy --trust -a .copier-answers.fastapi.yml gh:olaheavy-dev/house-template-fastapi .
make install && make check
```

The layer reads the project name and stacks from `.copier-answers.base.yml` and refuses to apply without them. Update it later with `copier update -a .copier-answers.fastapi.yml`.

## What it adds

| Path | What |
|---|---|
| `backend/src/app/` | FastAPI app factory with a lifespan, settings from the environment, RFC 9457 problem responses, structured JSON logs with request ids, and live and ready health endpoints |
| `backend/tests/` | Async tests that drive the app in-process with its lifespan running |
| `backend/pyproject.toml`, `backend/uv.lock` | uv project; ruff, mypy (strict), import-linter and pytest configuration |
| `mk/fastapi.mk` | `install-fastapi`, `check-fastapi`, `check-fast-fastapi`, `lock`, `dev`, `test`, joined to the base targets |
| `.claude/rules/fastapi.md` | Backend rules for coding agents, loaded only when they touch `backend/` |
| `.python-version` | The backend's Python (3.14 by default; 3.12 is the oldest supported) |

Still to come, through reviewed pull requests: PostgreSQL with pgvector, Alembic and Compose; the budget-capped LangChain model gateway with LangSmith tracing and recorded responses; S3-compatible storage; the container image; releases.

## Develop the template

`uv run pytest` renders the layer and checks it. CI also generates a real project from house-template-base plus this layer and runs its `make check`. After changing `template/backend/pyproject.toml`, run `uv lock --directory template/backend`.
