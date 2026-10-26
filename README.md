# house-template-fastapi

The FastAPI layer of my Copier project templates. It adds a Python backend under `backend/` to a project generated from [house-template-base](https://github.com/olaheavy-dev/house-template-base), and plugs the backend's checks into the base `make check`, CI, CodeQL and Dependabot.

## Use it

```sh
copier copy --trust gh:olaheavy-dev/house-template-base my-app      # choose the fastapi stack
cd my-app && git init -b main && git add -A && git commit -m "chore: start from house-template-base"
copier copy --trust -a .copier-answers.fastapi.yml gh:olaheavy-dev/house-template-fastapi .
make install && make check      # the backend tests need Docker
make db-up && make migrate && make dev
make storage-up                    # for file storage, with STORAGE_BUCKET set in backend/.env
```

The layer reads the project name and stacks from `.copier-answers.base.yml` and refuses to apply without them. Update it later with `copier update -a .copier-answers.fastapi.yml`. The project owns `backend/uv.lock`, so after an update that changes dependencies, run `make lock` and commit the lockfile before `make install`.

## What it adds

| Path | What |
|---|---|
| `backend/src/app/` | FastAPI app factory with a lifespan, settings from the environment, RFC 9457 problem responses, structured JSON logs with request ids, live and ready health endpoints, and FastAPI's native OpenTelemetry (request traces without health probes, HTTP metrics, exception logs; exported only once `OTEL_*` is set) |
| `backend/src/app/db.py`, `models.py` | SQLAlchemy 2.1 async with psycopg 3: a declarative `Base` with named constraints, the engine opened in the lifespan, a session per request (`SessionDep`), and a database readiness check |
| `backend/src/app/llm/`, `prompts/` | The model gateway: one LangChain chat model built from `LLM_MODEL` (OpenAI by default, `openai:gpt-6-luna`, or Anthropic; a pinned id), whose provider key is the only one recording and evals ask for with a token limit, a timeout and the SDK's retries; a daily budget that reserves each call's worst-case cost in Postgres before it is sent and settles the real cost from `usage_metadata`, one `llm_call` row per call; an integer price table; versioned prompt files whose hash is recorded with each call; optional LangSmith tracing with sampling and hidden inputs; every call's tokens (reasoning and cache included), cost, latency and feature/user tags recorded, a spend report (`make llm-report`), `llm.*` OpenTelemetry metrics and a warning at 80% of the budget; `gateway.agent()` for LangChain `create_agent` tool loops on the gateway's model, with threads kept by LangGraph's Postgres checkpointer (tables from an Alembic migration, pool opened only when used). `lint-imports` keeps `langchain` and the provider SDKs inside it, and a test rejects agents or models built from a model string |
| `backend/migrations/` | Alembic (async, configured in `pyproject.toml`, no `alembic.ini`) on two branches: the template's migrations (`template`, from pgvector onwards) and the project's own (`app`, where `make migration` writes), so template updates never collide with a project's migrations; `make migrate` runs `upgrade heads`. New revisions are lint-clean |
| `backend/src/app/storage.py` | Optional S3-compatible object storage (on when `STORAGE_BUCKET` is set): one boto3 client from the lifespan, presigned GET and PUT URLs so browsers move files directly, network calls in a worker thread, safe keys and a readiness check. R2 in production |
| `backend/Dockerfile`, `.dockerignore` | The API image, following uv's Docker guide: a build stage syncs the lockfile into a virtualenv (`--locked`, `--no-editable`, bytecode compiled), and the final stage on the same digest-pinned `python:<version>-slim-trixie` holds only that virtualenv, the migrations and `pyproject.toml`, running `fastapi run` as UID 10001 |
| `.github/workflows/publish-api.yml` | When base's `container_images` is on: every push to main publishes the image through base's reusable workflow (GHCR, by digest, with provenance and SBOM attestations) |
| `backend/compose.yaml` | Local PostgreSQL 18 with pgvector 0.8.7 and RustFS 1.0.4 storage, pinned by digest, plus the image's one-off `migrate` service and the `api` that waits for it; reachable only from this machine and included by base's root `compose.yaml` |
| `backend/tests/` | Async tests that drive the app in-process with its lifespan running, and database tests against a real PostgreSQL (testcontainers, or `TEST_DATABASE_URL`), each rolled back afterwards; storage tests against a real RustFS with a bucket per test; `alembic check` and a downgrade/upgrade round trip run with them. Model calls replay recorded responses (pytest-recording and vcrpy, network blocked, credentials filtered); `make record-llm` re-records them with a real key under a spending cap. Evals in `tests/evals/` use LangSmith's pytest integration and run only with `make eval`; the shipped ones check the gateway against the real provider (real cost settled, streamed usage, structured output, an agent's tool loop, the budget refusing). Both read keys from `backend/.env` |
| `backend/pyproject.toml`, `backend/uv.lock` | uv project; ruff, mypy (strict), import-linter and pytest configuration |
| `mk/fastapi.mk` | `install-fastapi`, `check-fastapi`, `check-fast-fastapi`, `lock`, `dev`, `test`, `db-up`, `storage-up`, `image`, `up`, `llm-report`, `migrate`, `migration`, `record-llm`, `eval`, joined to the base targets |
| `.claude/rules/fastapi.md` | Backend rules for coding agents, loaded only when they touch `backend/` |
| `.python-version` | The backend's Python (3.14 by default; 3.12 is the oldest supported) |

The layer now covers everything in its plan.

## Develop the template

To re-record the model cassette or run the evals here, put `OPENAI_API_KEY=...` (and optionally `LANGSMITH_API_KEY=...`) in `template/backend/.env`, which git ignores and Copier never copies, then from `template/backend`:

```sh
LLM_DAILY_BUDGET_USD=0.50 uv run --env-file .env pytest -m vcr --record-mode=rewrite   # make record-llm
LLM_EVAL=1 LLM_DAILY_BUDGET_USD=1 uv run --env-file .env pytest tests/evals -m eval   # make eval
```

`uv run pytest` renders the layer and checks it. CI also generates a real project from house-template-base plus this layer and runs its `make check`. After changing `template/backend/pyproject.toml`, run `uv lock --directory template/backend`.

`main` is protected by [`.github/rulesets/main.json`](.github/rulesets/main.json): squash merges through pull requests, with the tests, workflow lint (actionlint and zizmor), the PR title check and CodeQL required, and CodeQL and zizmor findings blocking a merge. A test checks that every required check is a job the workflows run.

## Releases

release-please turns Conventional Commit titles into a release PR and, on merge, a `vX.Y.Z` tag. Projects update from those tags.
