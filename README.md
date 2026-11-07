# house-template-fastapi

The FastAPI layer of my Copier project templates. It adds a Python backend under `backend/` to a project generated from [house-template-base](https://github.com/olaheavy-dev/house-template-base), and plugs the backend's checks into the base `make check`, CI, CodeQL and Dependabot.

## Use it

```sh
copier copy --trust gh:olaheavy-dev/house-template-base my-app      # choose the fastapi stack
cd my-app && git init -b main && git add -A && git commit -m "chore: start from house-template-base"
copier copy --trust -a .copier-answers.fastapi.yml gh:olaheavy-dev/house-template-fastapi .
cp backend/.env.example backend/.env
# With the database (the default): in backend/.env set LLM_MODEL and EMBEDDING_MODEL
# (<provider>:<model-id>); the template picks none. Add each model's prices to
# backend/src/app/llm/prices.py first, with the date you read them and the provider's pricing page
# (the app refuses to start otherwise).
make install && make check      # the tests need Docker, unless both database and storage are off
make db-up && make migrate && make dev   # with the database (the default; see Options)
make storage-up                    # for file storage (the default; see Options), with STORAGE_BUCKET set in backend/.env
make dev                           # with neither part: just the API
```

The layer checks the base answers (the fastapi stack) and copies `container_images` from them into its own answer in `.copier-answers.fastapi.yml`, which is what lets `copier update` work; template files never read the base answers. `copy` and `recopy` refuse without `.copier-answers.base.yml`; `update` cannot check for it and carries on. The refusal comes after an interactive answer or `-d container_images=...` (without either, Copier stops earlier with "Question container_images is required"; the interactive flow is untested), and by then files may already be written. If this directory had no uncommitted work before the run (check `git status`), undo it with `git clean -fd` and `git checkout .`; otherwise remove only the files the run added. Then apply the base first and run it again.

## Options

Answers that switch parts of the backend on or off. Each default keeps today's output.

| Answer | Default | When `false` |
|---|---|---|
| `database` | `true` | No PostgreSQL, SQLAlchemy, Alembic migrations or **model gateway** (the gateway keeps its budget and agent checkpoints in the database, so it goes too): no `app/db.py`, `app/models.py`, `app/llm/`, `app/prompts/`, `migrations/`, `compose/db/compose.yaml`, database and model tests, evals or cassettes, model and database settings, `make db-up`, `migrate`, `migration`, `llm-report`, `record-llm` and `eval`, the `db` and `langchain` extras in the image and make targets, and the database and model rules text. With both parts off, `make check` needs no Docker |
| `storage` | `true` | No `app/storage.py`, storage settings, `.env.example` entries, readiness check, storage tests, RustFS compose service, `make storage-up`, storage rules text or `boto3` in the image and make targets |

```sh
copier copy --trust -a .copier-answers.fastapi.yml -d storage=false gh:olaheavy-dev/house-template-fastapi .   # likewise -d database=false
copier update --trust --defaults -a .copier-answers.fastapi.yml -d storage=false   # an existing project; commit first (without --defaults it asks the other questions again)
```

How it works: files that exist only with an answer are listed in `copier.yml` `_exclude` as Jinja patterns, and `copier update` deletes them when the answer turns off (tested; your own files and `backend/uv.lock` stay). Files that change with an answer are templates (`x.jinja`) with an all-on plain twin next to them so the template's own tests run; a test keeps each twin equal to its template. Dependencies are optional extras in the one plain `backend/pyproject.toml` (`[project.optional-dependencies]`: `db` and `langchain` come with the `database` answer, `storage` with `storage`), so Dependabot still updates it. The `langchain` extra needs PostgreSQL at runtime; #30 decides how a separate model answer ties to the database. `make install`, the other make targets and the image install only the extras the project answered for (`EXTRAS` in `mk/fastapi.mk`, `--extra` in the Dockerfile); the lockfile holds all of them. **Commit first.** When you turn the database off, see "Turning the database off" under Upgrading. When you turn storage off, `copier update` deletes the template's storage files even if you edited them (`app/storage.py`, `tests/storage.py`, `tests/test_storage.py`, `compose/storage/compose.yaml`); your own files and `backend/uv.lock` stay, and you can recover edits from git. It does not drop what is in a bucket. Your own code that imports `app.storage` fails `mypy`, your own tests that use the `storage_settings` fixture fail in pytest, and a running RustFS container and the `storage-data` volume are left behind as orphans (`docker compose down --remove-orphans`, then remove the volume). Afterwards run `make lock && make install && make check`.

## Upgrading

**Breaking: the database and model packages moved into extras.** `alembic`, `sqlalchemy`, `psycopg`, `pgvector` and the rest of the database stack are now in the `db` extra, and `langchain`, `langgraph`, `langsmith` and the provider packages in the `langchain` extra; only `fastapi`, `pydantic-settings` and `structlog` stay in the main dependencies. `make install`, `make dev`, `make check` and the Docker image install them for you. A plain `uv sync`, `pip install .` or your own Dockerfile no longer gets them: add `--extra db --extra langchain` (and `--extra storage`). After `copier update --trust -a .copier-answers.fastapi.yml`, run `make lock` (your `uv.lock` is yours). Projects made before the `database` question existed take `database: true` on update, so nothing is removed. Packages moved; none were added.

**Turning the database off** (`database: false`, also removes the model gateway):

1. Commit everything first.
2. Projects from v0.6.1 or earlier must first `copier recopy` to v0.7.0 or newer (#49, below).
3. `copier update` never touches a database: tables are never dropped. To drop them, run `uv --directory backend run --locked --extra db --extra langchain alembic downgrade base` BEFORE updating, while Alembic and the migrations still exist (the migrations import the model packages too); it removes every template and project table.
4. Run `copier update --trust --defaults -a .copier-answers.fastapi.yml -d database=false` (without `--defaults` it asks every question again).

What the update deletes: the template's database and gateway files, even ones you edited (recover them from git): `app/db.py`, `app/models.py`, `app/llm/`, `app/prompts/`, `migrations/`, `compose/db/compose.yaml`, `tests/db.py` and the database and model tests, evals and cassettes. Your own files stay. Your own migrations in `backend/migrations/versions/` are left behind; `make check` still lints them with ruff but no longer type-checks them (Alembic is no longer installed): delete or move them. Your own tests that use `db_session` or `db_client` fail in pytest (the fixtures are gone), and code that imports `app.db`, `app.models` or `app.llm` fails `mypy`. Remove `DATABASE_URL` and the model settings from `backend/.env` (a leftover value is ignored). Run `docker compose down --remove-orphans`, then remove the `db-data` volume. Afterwards run `make lock && make install && make check`. Turning it back on restores the template's versions of those files, not your old edits. If you edited a twin file (`main.py`, `settings.py`, `log.py`, `tests/conftest.py`, `.env.example`, `app/api/__init__.py`), expect conflict markers in it after an update (untested).

**After the recopy below (v0.6.1 and earlier) or an update, from a release before this one (breaking):** `boto3` moved from the main dependencies to the `storage` extra. `make install`, `make dev`, `make check` and the Docker image install it for you; anything that installs the backend without them (a plain `uv sync`, `pip install .` or your own Dockerfile) no longer gets `boto3`, and importing `app.storage` fails. Use `uv sync --extra storage`, or add `--extra storage` to your own install commands. After `copier update --trust -a .copier-answers.fastapi.yml`, run `make lock`: your `uv.lock` is yours and is not rewritten. Projects made before the `storage` question existed take `storage: true` on update, so nothing is removed. The compose services moved to `backend/compose/db/compose.yaml` and `backend/compose/storage/compose.yaml`, included by `backend/compose.yaml`.

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
| `backend/src/app/` | FastAPI app factory with a lifespan, settings from the environment, RFC 9457 problem responses (with headers such as `Retry-After`, and model failures (with `database`) and storage failures (with `storage`) mapped to proper statuses), structured JSON logs with request ids for the app, uvicorn and libraries alike, one access line per request (route template, never the raw path), live and ready health endpoints, and FastAPI's native OpenTelemetry (request traces without health probes, HTTP metrics, exception logs; exported only once `OTEL_*` is set); production defaults: it refuses to start on local credentials, database statement and idle-transaction timeouts and a separate pool for the budget (with `database`), opt-in CORS and allowed hosts, and baseline security headers |
| `backend/src/app/api/` | The product's API, versioned at `/v1`, with `GET /v1/me` as the example protected route and keyset (cursor) pagination: `PageParamsDep`, `Page[T]` and `page()`, with opaque cursors and limits from 1 to 100 |
| `backend/src/app/auth.py` | The authentication hook: `CurrentUser` and `OptionalUser` dependencies over one `authenticate()` the project replaces with its provider's verification. Fails closed (401) in production until then; an `X-Dev-User` header stands in locally and in tests |
| `backend/src/app/db.py`, `models.py` (answer `database`) | SQLAlchemy 2.1 async with psycopg 3: a declarative `Base` with named constraints, the engine opened in the lifespan, a session per request (`SessionDep`), and a database readiness check |
| `backend/src/app/llm/`, `prompts/` (answer `database`) | The model gateway: one LangChain chat model built from `LLM_MODEL` (`<provider>:<model-id>`, OpenAI or Anthropic; no default, so you choose one and add its prices first), whose provider key is the only one the evals ask for (recording asks for the key of the model pinned in `tests/models.py`) with a token limit, a timeout and the SDK's retries; a daily budget, globally and optionally per user, that reserves each call's worst-case cost in Postgres before it is sent and settles the real cost from `usage_metadata`, one `llm_call` row per call; an integer price table in which every entry records the date it was read and the provider's pricing page it came from, and calls billed beyond it (a provider-run tool such as web search, a priority service tier) refused before they are sent; `gateway.embeddings()` for OpenAI embeddings (`EMBEDDING_MODEL`, also with no default) under the same budget, settled from the usage OpenAI returns; versioned prompt files whose hash is recorded with each call; optional LangSmith tracing with sampling and hidden inputs; every call's tokens (reasoning and cache included), cost, latency and feature/user tags recorded, a spend report (`make llm-report`), `llm.*` OpenTelemetry metrics and a warning at 80% of the budget; `gateway.agent()` for LangChain `create_agent` tool loops on the gateway's model, capped at `AGENT_MAX_STEPS`, with threads kept by LangGraph's Postgres checkpointer (tables from an Alembic migration, pool opened only when used). `lint-imports` keeps `langchain` and the provider SDKs inside it, and a test rejects agents or models built from a model string |
| `backend/migrations/` (answer `database`) | Alembic (async, configured in `pyproject.toml`, no `alembic.ini`) on two branches: the template's migrations (`template`, from pgvector onwards) and the project's own (`app`, where `make migration` writes), so template updates never collide with a project's migrations; `make migrate` runs `upgrade heads`. New revisions are lint-clean |
| `backend/src/app/storage.py` (answer `storage`), `limits.py` | Optional S3-compatible object storage, in the project unless `storage` is false (active when `STORAGE_BUCKET` is set): one boto3 client from the lifespan, presigned GET and PUT URLs so browsers move files directly (uploads limited by signed size and an allowed-type list, downloads always attachments), a 1 MiB cap on request bodies, network calls in a worker thread, safe keys and a readiness check. R2 in production |
| `backend/Dockerfile`, `.dockerignore` | The API image, following uv's Docker guide: a build stage syncs the lockfile into a virtualenv (`--locked`, `--no-editable`, bytecode compiled), and the final stage on the same digest-pinned `python:<version>-slim-trixie` holds only that virtualenv, the migrations (with `database`) and `pyproject.toml`, running uvicorn as UID 10001 with graceful shutdown |
| `.github/workflows/publish-api.yml` | When base's `container_images` is on: every push to main publishes the image through base's reusable workflow (GHCR, by digest, with provenance and SBOM attestations) |
| `backend/compose.yaml`, `backend/compose/` | The image's one-off `migrate` service and the `api` that waits for it (both only with `database`; without it, `api` starts on its own), including the local PostgreSQL 18 with pgvector 0.8.7 (`compose/db/`) and, with `storage`, RustFS 1.0.4 (`compose/storage/`), each pinned by digest in its own plain file; reachable only from this machine and included by base's root `compose.yaml` |
| `backend/tests/` | Async tests that drive the app in-process with its lifespan running. With `storage`, storage tests against a real RustFS with a bucket per test. With `database`: database tests against a real PostgreSQL (testcontainers, or `TEST_DATABASE_URL`), each rolled back afterwards, with `alembic check` and a downgrade/upgrade round trip; model calls replay recorded responses (pytest-recording and vcrpy, network blocked, credentials filtered); `make record-llm` re-records them with the models pinned in `tests/models.py` (change them there to record with another model) and a real key under a spending cap; `make eval` uses your own `LLM_MODEL` and `EMBEDDING_MODEL`; evals in `tests/evals/` use LangSmith's pytest integration and run only with `make eval`; the shipped ones check the gateway against the real provider (real cost settled, streamed usage, structured output, an agent's tool loop, the budget refusing). Both read keys from `backend/.env` |
| `backend/pyproject.toml`, `backend/uv.lock` | uv project with optional extras (`db`, `langchain`, `storage`); ruff, mypy (strict), import-linter and pytest configuration |
| `mk/fastapi.mk` | `install-fastapi`, `check-fastapi`, `check-fast-fastapi`, `lock`, `dev`, `test`, `image`, `up`, plus `db-up`, `llm-report`, `migrate`, `migration`, `record-llm` and `eval` (with `database`) and `storage-up` (with `storage`), joined to the base targets |
| `.claude/rules/fastapi.md` | Backend rules for coding agents, loaded only when they touch `backend/` |
| `.python-version`, `backend/.python-version` | The backend's Python (3.14 by default; 3.12 is the oldest supported). uv stops looking for the file at the project root (`backend/`), so `backend/.python-version` is the copy it reads; the root copy is for the editor and root-level tools |

The layer now covers everything in its plan.

## Develop the template

To re-record the model cassette or run the evals here, put `OPENAI_API_KEY=...` (and optionally `LANGSMITH_API_KEY=...`) in `template/backend/.env`; the evals also need `LLM_MODEL` and `EMBEDDING_MODEL` there, while recording uses the models pinned in `tests/models.py`. The `.env` file is git-ignored and Copier never copies it. Then from `template/backend`:

```sh
env -u LLM_EVAL LLM_DAILY_BUDGET_USD=0.50 uv run --extra db --extra langchain --extra storage --env-file .env pytest -m vcr --record-mode=rewrite   # make record-llm
LLM_EVAL=1 LLM_DAILY_BUDGET_USD=1 uv run --extra db --extra langchain --extra storage --env-file .env pytest tests/evals -m eval   # make eval
```

`uv run pytest` renders the layer and checks it. CI also generates a real project from house-template-base plus this layer and runs its `make check`, once with the default answers and once per variant (`storage=false`; `database=false` with storage; `python_version=3.12`, the oldest supported Python; and `database=false storage=false`, whose `make check` runs with an unreachable `DOCKER_HOST` to show it needs no Docker, and which then starts the image with no `backend/.env` and calls `/health/ready`; every job also checks the backend runs on the Python its `.python-version` names). The variant jobs are not required checks yet: that is a repository-ruleset change for the owner. In `template/backend`, run `uv run --extra db --extra langchain --extra storage pytest` (the plain files next to each `.jinja` are the all-on twins; the extras they need are not in the default sync, and without them the tests fail to import `sqlalchemy` and the rest). On Python 3.14 the backend's langsmith pytest plugin raises a deprecation warning when it loads; one scoped filter in `template/backend/pyproject.toml` ignores that message until langsmith fixes it (#59). After changing `template/backend/pyproject.toml`, run `uv lock --directory template/backend`.

`main` is protected by [`.github/rulesets/main.json`](.github/rulesets/main.json): squash merges through pull requests, with the tests, workflow lint (actionlint and zizmor), the PR title check and CodeQL required, and CodeQL and zizmor findings blocking a merge. A test checks that every required check is a job the workflows run.

`tests/test_golden_render.py` keeps the default output from changing by accident: it renders `BASELINE_REF` (a commit of this repository) and the working tree with the default answers and compares every file. A feature PR that changes the default output on purpose adds the path to `INTENDED` with a reason and an issue number, and the test fails if a listed path no longer differs. A path-level entry hides a second, accidental change to the same file until the next baseline move. After each release, a `chore:` PR moves `BASELINE_REF` to the new tag and empties `INTENDED`. It needs the full git history (`git fetch --unshallow` in a shallow clone).

## Releases

release-please turns Conventional Commit titles into a release PR and, on merge, a `vX.Y.Z` tag. Projects update from those tags.
