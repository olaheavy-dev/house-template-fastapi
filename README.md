# house-template-fastapi

The FastAPI layer of my Copier project templates. It adds a Python backend under `backend/` to a project generated from [house-template-base](https://github.com/olaheavy-dev/house-template-base), and plugs the backend's checks into the base `make check`, CI, CodeQL and Dependabot.

## Use it

```sh
copier copy --trust gh:olaheavy-dev/house-template-base my-app      # choose the fastapi stack
cd my-app && git init -b main && git add -A && git commit -m "chore: start from house-template-base"
copier copy --trust -a .copier-answers.fastapi.yml gh:olaheavy-dev/house-template-fastapi .
cp backend/.env.example backend/.env
# In backend/.env set LLM_MODEL and EMBEDDING_MODEL (<provider>:<model-id>); the template picks none.
# Add each model's prices to backend/src/app/llm/prices.py first, with the date you read them and
# the provider's pricing page (the app refuses to start otherwise).
make install && make check      # the backend tests need Docker
make db-up && make migrate && make dev
make storage-up                    # for file storage, with STORAGE_BUCKET set in backend/.env
```

The layer checks the base answers (the fastapi stack) and copies `container_images` from them into its own answer in `.copier-answers.fastapi.yml`, which is what lets `copier update` work; template files never read the base answers. `copy` and `recopy` refuse without `.copier-answers.base.yml`; `update` cannot check for it and carries on. The refusal comes after an interactive answer or `-d container_images=...` (without either, Copier stops earlier with "Question container_images is required"; the interactive flow is untested), and by then files may already be written. If this directory had no uncommitted work before the run (check `git status`), undo it with `git clean -fd` and `git checkout .`; otherwise remove only the files the run added. Then apply the base first and run it again.

## Upgrading

From v0.7.0 on, update the layer with `copier update --trust -a .copier-answers.fastapi.yml`. `--trust` is required: Copier refuses any template that has `_tasks` without it, even when the task does not run in an update. The project owns `backend/uv.lock`, so after an update that changes dependencies, run `make lock` and commit the lockfile before `make install`. The update prints a harmless `MissingFileWarning` for `.copier-answers.base.yml`.

If the base's `container_images` changes, the next update follows it: it deletes `.github/workflows/publish-api.yml` when it turns off and adds it when it turns on. If you edited that file, expect a conflict (untested).

Projects made from v0.6.1 or earlier cannot `copier update` (their old version refuses without the base file; #49). Move them once with `recopy`, then update normally:

1. Commit everything, so `git diff` shows exactly what the next step changes.
2. `copier recopy --trust --defaults --overwrite -a .copier-answers.fastapi.yml --vcs-ref <tag>`, with `<tag>` v0.7.0 or newer. It applies the layer again as if new and overwrites your edits to any file it renders; it keeps `backend/uv.lock` and your own files.
3. Review `git diff` and restore your own edits (especially to the template-owned files listed in `.github/template-owned/fastapi.txt`).
4. `make lock && make install && make check`, then commit.

Tested from v0.6.1 only; v0.3.0 to v0.5.x and the checks after the recopy are untested.

## What it adds

| Path | What |
|---|---|
| `backend/src/app/` | FastAPI app factory with a lifespan, settings from the environment, RFC 9457 problem responses (with headers such as `Retry-After`, and model and storage failures mapped to proper statuses), structured JSON logs with request ids for the app, uvicorn and libraries alike, one access line per request (route template, never the raw path), live and ready health endpoints, and FastAPI's native OpenTelemetry (request traces without health probes, HTTP metrics, exception logs; exported only once `OTEL_*` is set); production defaults: it refuses to start on local credentials, database statement and idle-transaction timeouts, a separate pool for the budget, opt-in CORS and allowed hosts, and baseline security headers |
| `backend/src/app/api/` | The product's API, versioned at `/v1`, with `GET /v1/me` as the example protected route and keyset (cursor) pagination: `PageParamsDep`, `Page[T]` and `page()`, with opaque cursors and limits from 1 to 100 |
| `backend/src/app/auth.py` | The authentication hook: `CurrentUser` and `OptionalUser` dependencies over one `authenticate()` the project replaces with its provider's verification. Fails closed (401) in production until then; an `X-Dev-User` header stands in locally and in tests |
| `backend/src/app/db.py`, `models.py` | SQLAlchemy 2.1 async with psycopg 3: a declarative `Base` with named constraints, the engine opened in the lifespan, a session per request (`SessionDep`), and a database readiness check |
| `backend/src/app/llm/`, `prompts/` | The model gateway: one LangChain chat model built from `LLM_MODEL` (`<provider>:<model-id>`, OpenAI or Anthropic; no default, so you choose one and add its prices first), whose provider key is the only one the evals ask for (recording asks for the key of the model pinned in `tests/models.py`) with a token limit, a timeout and the SDK's retries; a daily budget, globally and optionally per user, that reserves each call's worst-case cost in Postgres before it is sent and settles the real cost from `usage_metadata`, one `llm_call` row per call; an integer price table in which every entry records the date it was read and the provider's pricing page it came from, and calls billed beyond it (a provider-run tool such as web search, a priority service tier) refused before they are sent; `gateway.embeddings()` for OpenAI embeddings (`EMBEDDING_MODEL`, also with no default) under the same budget, settled from the usage OpenAI returns; versioned prompt files whose hash is recorded with each call; optional LangSmith tracing with sampling and hidden inputs; every call's tokens (reasoning and cache included), cost, latency and feature/user tags recorded, a spend report (`make llm-report`), `llm.*` OpenTelemetry metrics and a warning at 80% of the budget; `gateway.agent()` for LangChain `create_agent` tool loops on the gateway's model, capped at `AGENT_MAX_STEPS`, with threads kept by LangGraph's Postgres checkpointer (tables from an Alembic migration, pool opened only when used). `lint-imports` keeps `langchain` and the provider SDKs inside it, and a test rejects agents or models built from a model string |
| `backend/migrations/` | Alembic (async, configured in `pyproject.toml`, no `alembic.ini`) on two branches: the template's migrations (`template`, from pgvector onwards) and the project's own (`app`, where `make migration` writes), so template updates never collide with a project's migrations; `make migrate` runs `upgrade heads`. New revisions are lint-clean |
| `backend/src/app/storage.py`, `limits.py` | Optional S3-compatible object storage (on when `STORAGE_BUCKET` is set): one boto3 client from the lifespan, presigned GET and PUT URLs so browsers move files directly (uploads limited by signed size and an allowed-type list, downloads always attachments), a 1 MiB cap on request bodies, network calls in a worker thread, safe keys and a readiness check. R2 in production |
| `backend/Dockerfile`, `.dockerignore` | The API image, following uv's Docker guide: a build stage syncs the lockfile into a virtualenv (`--locked`, `--no-editable`, bytecode compiled), and the final stage on the same digest-pinned `python:<version>-slim-trixie` holds only that virtualenv, the migrations and `pyproject.toml`, running uvicorn as UID 10001 with graceful shutdown |
| `.github/workflows/publish-api.yml` | When base's `container_images` is on: every push to main publishes the image through base's reusable workflow (GHCR, by digest, with provenance and SBOM attestations) |
| `backend/compose.yaml` | Local PostgreSQL 18 with pgvector 0.8.7 and RustFS 1.0.4 storage, pinned by digest, plus the image's one-off `migrate` service and the `api` that waits for it; reachable only from this machine and included by base's root `compose.yaml` |
| `backend/tests/` | Async tests that drive the app in-process with its lifespan running, and database tests against a real PostgreSQL (testcontainers, or `TEST_DATABASE_URL`), each rolled back afterwards; storage tests against a real RustFS with a bucket per test; `alembic check` and a downgrade/upgrade round trip run with them. Model calls replay recorded responses (pytest-recording and vcrpy, network blocked, credentials filtered); `make record-llm` re-records them with the models pinned in `tests/models.py` (change them there to record with another model) and a real key under a spending cap; `make eval` uses your own `LLM_MODEL` and `EMBEDDING_MODEL`. Evals in `tests/evals/` use LangSmith's pytest integration and run only with `make eval`; the shipped ones check the gateway against the real provider (real cost settled, streamed usage, structured output, an agent's tool loop, the budget refusing). Both read keys from `backend/.env` |
| `backend/pyproject.toml`, `backend/uv.lock` | uv project; ruff, mypy (strict), import-linter and pytest configuration |
| `mk/fastapi.mk` | `install-fastapi`, `check-fastapi`, `check-fast-fastapi`, `lock`, `dev`, `test`, `db-up`, `storage-up`, `image`, `up`, `llm-report`, `migrate`, `migration`, `record-llm`, `eval`, joined to the base targets |
| `.claude/rules/fastapi.md` | Backend rules for coding agents, loaded only when they touch `backend/` |
| `.python-version` | The backend's Python (3.14 by default; 3.12 is the oldest supported) |

The layer now covers everything in its plan.

## Develop the template

To re-record the model cassette or run the evals here, put `OPENAI_API_KEY=...` (and optionally `LANGSMITH_API_KEY=...`) in `template/backend/.env`; the evals also need `LLM_MODEL` and `EMBEDDING_MODEL` there, while recording uses the models pinned in `tests/models.py`. The `.env` file is git-ignored and Copier never copies it. Then from `template/backend`:

```sh
env -u LLM_EVAL LLM_DAILY_BUDGET_USD=0.50 uv run --env-file .env pytest -m vcr --record-mode=rewrite   # make record-llm
LLM_EVAL=1 LLM_DAILY_BUDGET_USD=1 uv run --env-file .env pytest tests/evals -m eval   # make eval
```

`uv run pytest` renders the layer and checks it. CI also generates a real project from house-template-base plus this layer and runs its `make check`. After changing `template/backend/pyproject.toml`, run `uv lock --directory template/backend`.

`main` is protected by [`.github/rulesets/main.json`](.github/rulesets/main.json): squash merges through pull requests, with the tests, workflow lint (actionlint and zizmor), the PR title check and CodeQL required, and CodeQL and zizmor findings blocking a merge. A test checks that every required check is a job the workflows run.

`tests/test_golden_render.py` keeps the default output from changing by accident: it renders `BASELINE_REF` (a commit of this repository) and the working tree with the default answers and compares every file. A feature PR that changes the default output on purpose adds the path to `INTENDED` with a reason and an issue number, and the test fails if a listed path no longer differs. A path-level entry hides a second, accidental change to the same file until the next baseline move. After each release, a `chore:` PR moves `BASELINE_REF` to the new tag and empties `INTENDED`. It needs the full git history (`git fetch --unshallow` in a shallow clone).

## Releases

release-please turns Conventional Commit titles into a release PR and, on merge, a `vX.Y.Z` tag. Projects update from those tags.
