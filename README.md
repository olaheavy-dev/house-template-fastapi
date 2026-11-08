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
| `database` | `true` | No PostgreSQL, SQLAlchemy, Alembic migrations or **model gateway** (the gateway keeps its budget and agent checkpoints in the database, so it goes too): no `app/db.py`, `app/models.py`, `app/llm/`, `app/prompts/`, `migrations/`, `llm-contracts.toml`, `compose/db/compose.yaml`, database and model tests, evals or cassettes, model and database settings, `make db-up`, `migrate`, `migration`, `llm-report`, `record-llm` and `eval`, the `db` and `langchain` extras in the image and make targets, and the database and model rules text. With both parts off, `make check` needs no Docker |
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
2. Projects from v0.6.1 or earlier must first `copier recopy` to v0.7.0 (#49, "Upgrading to 0.7.0" below).
3. `copier update` never touches a database: tables are never dropped. To drop them, run `uv --directory backend run --locked --extra db --extra langchain alembic downgrade base` BEFORE updating, while Alembic and the migrations still exist (the migrations import the model packages too); it removes every template and project table.
4. Run `copier update --trust --defaults -a .copier-answers.fastapi.yml -d database=false` (without `--defaults` it asks every question again).

What the update deletes: the template's database and gateway files, even ones you edited (recover them from git): `app/db.py`, `app/models.py`, `app/llm/`, `app/prompts/`, `migrations/`, `llm-contracts.toml`, `compose/db/compose.yaml`, `tests/db.py` and the database and model tests, evals and cassettes. Your own files stay. Your own migrations in `backend/migrations/versions/` are left behind; `make check` still lints them with ruff but no longer type-checks them (Alembic is no longer installed): delete or move them. Your own tests that use `db_session` or `db_client` fail in pytest (the fixtures are gone), and code that imports `app.db`, `app.models` or `app.llm` fails `mypy`. Remove `DATABASE_URL` and the model settings from `backend/.env` (a leftover value is ignored). Run `docker compose down --remove-orphans`, then remove the `db-data` volume. Afterwards run `make lock && make install && make check`. Turning it back on restores the template's versions of those files, not your old edits. If you edited a twin file (`main.py`, `settings.py`, `log.py`, `tests/conftest.py`, `.env.example`, `app/api/__init__.py`), expect conflict markers in it after an update (untested).

**After the recopy below (v0.6.1 and earlier) or an update, from a release before this one (breaking):** `boto3` moved from the main dependencies to the `storage` extra. `make install`, `make dev`, `make check` and the Docker image install it for you; anything that installs the backend without them (a plain `uv sync`, `pip install .` or your own Dockerfile) no longer gets `boto3`, and importing `app.storage` fails. Use `uv sync --extra storage`, or add `--extra storage` to your own install commands. After `copier update --trust -a .copier-answers.fastapi.yml`, run `make lock`: your `uv.lock` is yours and is not rewritten. Projects made before the `storage` question existed take `storage: true` on update, so nothing is removed. The compose services moved to `backend/compose/db/compose.yaml` and `backend/compose/storage/compose.yaml`, included by `backend/compose.yaml`.

From v0.7.0 on, update the layer with `copier update --trust -a .copier-answers.fastapi.yml`. `--trust` is required: Copier refuses any template that has `_tasks` without it, even when the task does not run in an update. The project owns `backend/uv.lock`, so after an update that changes dependencies, run `make lock` and commit the lockfile before `make install`. The update prints a harmless `MissingFileWarning` for `.copier-answers.base.yml`.

If the base's `container_images` changes, the next update follows it: it deletes `.github/workflows/publish-api.yml` when it turns off and adds it when it turns on. If you edited that file, expect a conflict (untested).

Projects made from v0.6.1 or earlier cannot `copier update` (their old version refuses without the base file; #49). Move them once with `recopy`, then update normally. The steps, and what else changed in 0.7.0, are in "Upgrading to 0.7.0" below.

## Upgrading to 0.7.0

0.7.0 moves the model gateway into a framework-neutral `app/llm/core/` and a LangChain adapter `app/llm/langchain/` (#33), besides the move of the database and model packages into extras (above). Projects with `database: false` have no model gateway, so "The model gateway moved" does not apply to them.

**What you must do** (it works once v0.7.0 is released; details follow):

1. Commit everything.
2. Move the project to 0.7.0: recopy once, or `copier update`, depending on `.copier-answers.fastapi.yml` (next section).
3. `make lock && make install`.
4. Fix the old import paths (the finder and grep under "The model gateway moved").
5. Recopy only: restore your model imports, router registrations and price entries.
6. Set `LLM_MODEL` and `EMBEDDING_MODEL` in every environment the app runs in (see step 4 of the recopy).
7. `make check`, then commit.
8. Change scheduled jobs to `python -m app.llm.core.report --expire-stale`.

Pick the route from `.copier-answers.fastapi.yml`:

- No `container_images:` line (the project was made from v0.6.1 or earlier, or from a commit before #53): recopy once, below.
- A `container_images:` line: `copier update --trust -a .copier-answers.fastapi.yml`, then `make lock && make install` and "The model gateway moved". An update of a project from before the split merges `prices.py` (your entries stay, and it gains a `_MOVED` table) and conflicts exactly on an edited `gateway.py`, `checkpoints.py`, `errors.py` or `embeddings.py`, which are now shims: move the edit to the new module. A test in `tests/test_update.py` ([PR #83](https://github.com/olaheavy-dev/house-template-fastapi/pull/83)) pins this: it updates from a commit just before #33 started. An edited `budget.py`, `metrics.py`, `prompts.py` or `report.py` should conflict the same way (untested). Edits to `backend/pyproject.toml` (the gateway `ignore_imports` line changed) or `mk/fastapi.mk` may conflict if they touch the changed lines (untested).

### Recopy (v0.6.1 and earlier)

1. Before, note your own price entries, the models in `app/models.py`, the router registrations in `app/api/__init__.py`, and your edits to `main.py`, `settings.py`, `auth.py`, `tests/conftest.py`, `backend/pyproject.toml` and `.env.example`. The recopy overwrites every file the template renders, silently: the `conflict <path>` line Copier prints before `overwrite <path>` means "differs, so overwritten", not a merge conflict. It leaves `backend/uv.lock`, the files you added (routes, tests, your own migrations) and the database alone. It adds files a v0.6.1 project did not have (`tests/recording.py`, `tests/models.py`, `backend/.python-version`, `backend/compose/`, `backend/llm-contracts.toml`, `app/llm/core/`, `app/llm/langchain/` and many `test_llm_*` files).
2. `copier recopy --trust --defaults --overwrite -a .copier-answers.fastapi.yml --vcs-ref v0.7.0`. Use `v0.7.0`, or a later 0.7.x patch release, and never 0.8.0: in 0.7.x the old import paths still work and warn, and 0.8.0 removes them. This works once v0.7.0 is released. It needs `.copier-answers.base.yml`, as above.
3. Restore your edits from `git diff` (especially to the files listed in `.github/template-owned/fastapi.txt`). Two traps:
   - **Models.** The recopy replaces `app/models.py`, so your model imports vanish, and your router registrations in `app/api/__init__.py` vanish without a message. Restore both BEFORE any `make migration` or `alembic check`. Otherwise `alembic check` says `Detected removed table '<yours>'`, and `make migration` would propose dropping that table.
   - **Prices.** `app/llm/prices.py` is overwritten with the template's table. A v0.6.1 `Price(...)` entry pasted back fails at import with `TypeError: Price.__init__() missing 2 required keyword-only arguments: 'checked' and 'source'`. Read the price again on the provider's pricing page and add it back with `checked="YYYY-MM-DD"` and `source="https://..."`; never retype one from memory.
4. `make lock && make install` (the recopy changed `backend/pyproject.toml`, and your `uv.lock` is skipped). Then fix the old import paths as described in "The model gateway moved", run `make check`, and commit. `make check` does not need the model settings (`tests/conftest.py` sets them), but running the app does: v0.6.1 defaulted both models in `settings.py`, and 0.7.0 has no default, so the gateway refuses to start without them. Set `LLM_MODEL` and `EMBEDDING_MODEL` in EVERY environment the app runs in (your host's settings, the image's environment), not only `backend/.env`. A `.env` made from the v0.6.1 `.env.example` already holds both.

Recorded run ([PR #83](https://github.com/olaheavy-dev/house-template-fastapi/pull/83), manual, not in CI; it used the code of that pull request's branch, not a v0.7.0 tag): Copier 9.19.1, base v0.2.1, a v0.6.1 project on Python 3.12 with edits to a price entry, `gateway.py` and `checkpoints.py` (real modules at v0.6.1, shims after the upgrade), a route on old import paths, a `Course` model with a migration and a row, a project migration importing `checkpoints`, a test, `pyproject.toml` and `main.py`. The recopy exited 0 with no `.rej`, `.orig` or conflict markers, modified 49 files and deleted none. After the steps above `make check` passed (604 tests). `alembic current` was unchanged, `alembic upgrade heads` applied nothing, `alembic check` was clean (with the `models.py` imports restored), the saved row survived, and `checkpoint_migrations` held 10 rows, equal to `len(MIGRATIONS)`. Run Docker commands from the project root as `docker compose exec -T db ...`: `docker compose -f backend/compose.yaml exec ...` fails with "service db is not running". Only v0.6.1 was tried; v0.3.0 to v0.5.x are untested.

### The model gateway moved (#33)

The old import paths keep working until 0.8.0: each answers with a `DeprecationWarning` and returns the same object as the new one. **Update imports first.** The template runs pytest with `filterwarnings = error`, so one old import in app code (which conftest loads) stops the whole run at the first one; in test modules pytest lists every collection error.

Find every use with the template's own finder (it exists after the recopy or update, and lists uses in `src`, `tests` and `migrations`), then grep for what it cannot see, such as string patch targets like `monkeypatch.setattr("app.llm.gateway.openai_client", ...)`:

```sh
uv --directory backend run --locked --extra db --extra langchain --extra storage pytest -vv -W ignore::DeprecationWarning "tests/test_llm_shims.py::test_the_template_never_uses_an_old_path"
grep -rnE 'app\.llm\.(budget|errors|embeddings|gateway|checkpoints|metrics|prompts|prices|report)\b|from app\.llm import .*(budget|errors|embeddings|gateway|checkpoints|metrics|prompts|prices|report)' backend/src backend/tests backend/migrations --include='*.py'
```

In the grep's output, ignore every hit under `backend/src/app/llm/` (the old modules, and docstrings in the new ones) and in the template's own `test_llm_*` tests (`test_llm_shims.py`, `test_llm_surface.py` and `test_llm_report_cli.py` name the old paths on purpose); a `test_llm_*` file of your own is not exempt. Also ignore `from app.llm import prices` or `app.llm.prices` where it only reaches `PRICES`, `EMBEDDING_PRICES`, `Price`, `Rates` or `MICRO_USD`, which stay there without a warning.

Old name to new (the same as `MOVED` in `backend/tests/test_llm_surface.py`):

| Old module | Name | Now in |
|---|---|---|
| `app.llm.prices` | `cost`, `ceiling`, `token_details` | `app.llm.core.prices` |
| `app.llm.gateway` | `key_setting`, `KEY_SETTINGS` | `app.llm.core.registry` |
| | `Gateway`, `get_gateway` | `app.llm.langchain.gateway` |
| | `GatewayDep` | `app.llm` (`from app.llm import GatewayDep`) |
| `app.llm.errors` | `Outcome`, `DEFAULT_RETRY_AFTER` | `app.llm.core.errors` |
| | `OUTCOMES`, `outcome_for`, `install_model_error_handlers` | `app.llm.langchain.errors` |
| `app.llm.budget` | `LlmCall`, `LlmSpendDay`, `LlmSpendUserDay`, `BudgetExhausted`, `UserBudgetExhausted`, `reserve`, `reserve_for`, `settle`, `expire_stale`, `utc_today`, `WARN_SHARE` | `app.llm.core.budget` |
| | `PromptTooLong`, `UnboundedMedia`, `UnpricedRequest`, `check_priced`, `INPUT_MARGIN_TOKENS`, `IMAGE_TOKENS` | `app.llm.core.bounds` |
| | `BudgetGuard`, `input_bound` | `app.llm.langchain.callbacks` |
| `app.llm.metrics` | every name | `app.llm.core.metrics` |
| `app.llm.embeddings` | `GatewayEmbeddings` | `app.llm.langchain.embeddings` |
| | `openai_client` | `app.llm.core.openai_embeddings` |
| | `BudgetExhausted`, `UserBudgetExhausted`, `reserve_for`, `settle` | `app.llm.core.budget` |
| | `INPUT_MARGIN_TOKENS` | `app.llm.core.bounds` |
| `app.llm.prompts` | `Prompt`, `load_prompt` | `app.llm.langchain.prompts` |
| | `PROMPTS` | `app.llm.core.prompts` |
| `app.llm.report` | every name | `app.llm.core.report` |
| `app.llm.checkpoints` | every name | `app.llm.langchain.checkpoints` |

`install_model_error_handlers` still registers only LangChain's and LangGraph's exceptions. If your own app factory (not the template's `main.py`) calls it, also call `app.llm.core.errors.install_failure_handlers(app)`, or a `ProviderError` or `StepLimitReached` becomes a plain 500. If your own migrations call `checkpoints.apply_migrations` (the rules tell projects to write them), import `from app.llm.langchain import checkpoints`, or conftest's in-process upgrade fails every database test. Scheduled jobs: change `python -m app.llm.report --expire-stale` to `python -m app.llm.core.report --expire-stale` now. The old command still runs, with one `DeprecationWarning`, until 0.8.0, and `make llm-report` already uses the new one.

**Signatures.** `token_details`, `settle(usage=)` and `record_call` take `app.llm.core.usage.Usage`, or `None` for no usage. `cost` takes a `Usage` only (`cost(price, None)` raises `AttributeError`), so skip pricing when there is no usage. A dict, even `{}`, raises `AttributeError` in all of them. Convert LangChain's usage with `from_langchain(usage) if usage else None` (`app.llm.langchain.usage`) where `None` is allowed. Inside a settlement the error is logged as `llm_settle_failed` and the reservation stays, so spend is over-recorded, never hidden.

**Names that now raise `AttributeError` with no warning**, for example (found by comparing every top-level name of each old module at v0.6.1 and at the commit before #33 started with what the 0.7.0 module answers; the full list also holds every name a module merely imported, such as `Any` or `AsyncSession`): on `budget`, `_images`, `_usage`, `_REJECTED`, `metrics`, `prices`, `log`, `time`, `json` and `Problem`; on `embeddings`, `_REJECTED`, `_AS_MODEL_ERROR`, `_as_model_error`, `metrics`, `openai`, `log` and `utc_today`; on `errors`, `problem_response`; on `gateway`, `_tracer`, `openai_client`, `init_chat_model`, `GatewayEmbeddings`, `BudgetGuard`, `Prompt`, `Problem`, `Settings` and `prices`; on `checkpoints`, `Settings`; on `report`, `expire_stale`, `LlmCall`, `utc_today`, `MICRO_USD`, `get_settings`, `create_engine` and `asyncio`; on `prompts`, `hashlib`, `cache` and `RunnableConfig`; on `metrics`, `_meter` and `token_details`. They are private helpers and names a module only imported: import them from where they live. The descriptions of pull requests [#65](https://github.com/olaheavy-dev/house-template-fastapi/pull/65), [#66](https://github.com/olaheavy-dev/house-template-fastapi/pull/66), [#67](https://github.com/olaheavy-dev/house-template-fastapi/pull/67), [#70](https://github.com/olaheavy-dev/house-template-fastapi/pull/70), [#79](https://github.com/olaheavy-dev/house-template-fastapi/pull/79) and [#81](https://github.com/olaheavy-dev/house-template-fastapi/pull/81) list more.

**Patch targets.** Patching an old path no longer reaches the gateway, and under `filterwarnings = error` the patch's own lookup raises. Patch `app.llm.langchain.gateway.openai_client` and `.init_chat_model`; `app.llm.core.budget.*`; `app.llm.core.metrics.*`; `app.llm.core.report.main` and `.get_settings`; `app.llm.core.errors.DEFAULT_RETRY_AFTER`. A test that reloads the metrics module must reload `app.llm.core.metrics`, never `app.llm.core.budget` (it would map the tables a second time).

**Overrides that change.** `BudgetGuard._settle` is no longer called (use `guard.meter`). On `GatewayEmbeddings`, `_settle` and `_batches` overrides are no longer called; `_embed` is still called but with a new signature `(texts, tags)` and does the whole metered call, so a v0.6.1 override `_embed(self, batch, *, feature, user_ref)` raises `TypeError` on every embeddings call: move it to `_create` (`MAX_INPUTS` and `MAX_BYTES` overrides still apply). A `Prompt.config()` override inside `gateway.call_config` is no longer called. A core `Prompt` does not equal a LangChain `Prompt` with the same fields: compare `.sha`.

**Behaviour changes** ([#66](https://github.com/olaheavy-dev/house-template-fastapi/pull/66), [#70](https://github.com/olaheavy-dev/house-template-fastapi/pull/70)). An embedding price's `max_prompt_tokens` now applies to every request: one text over the limit refuses the whole call with a 422 before anything is reserved (no shipped entry sets a limit). An unknown embeddings error is raised as itself, no longer as its own cause. An embeddings reply with no usage now closes its row `failed` at its reservation, where it used to stay `reserved`. If a provider error fits two failure kinds, the first matching kind decides whether the reservation is refunded, so spend can be over-recorded but never under-recorded.

**Unchanged:** the database schema and migrations (`migrations/env.py` and migration 0003 changed only import lines and comments), metric, log and problem names, HTTP bodies, rounding (up, once, on the total) and routes.

## What it adds

| Path | What |
|---|---|
| `backend/src/app/` | FastAPI app factory with a lifespan, settings from the environment, RFC 9457 problem responses (with headers such as `Retry-After`, and model failures (with `database`) and storage failures (with `storage`) mapped to proper statuses), structured JSON logs with request ids for the app, uvicorn and libraries alike, one access line per request (route template, never the raw path), live and ready health endpoints, and FastAPI's native OpenTelemetry (request traces without health probes, HTTP metrics, exception logs; exported only once `OTEL_*` is set); production defaults: it refuses to start on local credentials, database statement and idle-transaction timeouts and a separate pool for the budget (with `database`), opt-in CORS and allowed hosts, and baseline security headers |
| `backend/src/app/api/` | The product's API, versioned at `/v1`, with `GET /v1/me` as the example protected route and keyset (cursor) pagination: `PageParamsDep`, `Page[T]` and `page()`, with opaque cursors and limits from 1 to 100 |
| `backend/src/app/auth.py` | The authentication hook: `CurrentUser` and `OptionalUser` dependencies over one `authenticate()` the project replaces with its provider's verification. Fails closed (401) in production until then; an `X-Dev-User` header stands in locally and in tests |
| `backend/src/app/db.py`, `models.py` (answer `database`) | SQLAlchemy 2.1 async with psycopg 3: a declarative `Base` with named constraints, the engine opened in the lifespan, a session per request (`SessionDep`), and a database readiness check |
| `backend/src/app/llm/`, `prompts/` (answer `database`) | The model gateway: one LangChain chat model built from `LLM_MODEL` (`<provider>:<model-id>`, OpenAI or Anthropic; no default, so you choose one and add its prices first), whose provider key is the only one the evals ask for (recording asks for the key of the model pinned in `tests/models.py`) with a token limit, a timeout and the SDK's retries; a daily budget, globally and optionally per user, that reserves each call's worst-case cost in Postgres before it is sent and settles the real cost from `usage_metadata`, one `llm_call` row per call; an integer price table in which every entry records the date it was read and the provider's pricing page it came from, and calls billed beyond it (a provider-run tool such as web search, a priority service tier) refused before they are sent; `gateway.embeddings()` for OpenAI embeddings (`EMBEDDING_MODEL`, also with no default) under the same budget, settled from the usage OpenAI returns; versioned prompt files whose hash is recorded with each call; optional LangSmith tracing with sampling and hidden inputs; every call's tokens (reasoning and cache included), cost, latency and feature/user tags recorded, a spend report (`make llm-report`), `llm.*` OpenTelemetry metrics and a warning at 80% of the budget; `gateway.agent()` for LangChain `create_agent` tool loops on the gateway's model, capped at `AGENT_MAX_STEPS`, with threads kept by LangGraph's Postgres checkpointer (tables from an Alembic migration, pool opened only when used). `lint-imports` keeps `langchain` and the provider SDKs inside it, a second contract (`backend/llm-contracts.toml`, run as `lint-imports --config llm-contracts.toml`) keeps `app/llm/core/` (price arithmetic, failure kinds, the budget ledger and its `Meter`, request bounds, metrics, and metered, batched embeddings, whose one `openai` import is `core/openai_embeddings.py`, the prompt and its version hash, and the spend report) free of any model framework, while the LangChain parts (gateway, callbacks, embeddings, errors, usage, prompts, checkpointer) are in `app/llm/langchain/`, and a test rejects agents or models built from a model string |
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

`tests/test_golden_render.py` keeps the default output from changing by accident: it renders `BASELINE_REF` (the latest release tag of this repository, v0.7.0 now) and the working tree with the default answers and compares every file. A feature PR that changes the default output on purpose adds the path to `INTENDED` with a reason and an issue number, and the test fails if a listed path no longer differs. A path-level entry hides a second, accidental change to the same file until the next baseline move. After each release, a `chore:` PR moves `BASELINE_REF` to the new tag and empties `INTENDED`. It needs the full git history (`git fetch --unshallow --tags` in a shallow clone).

Design records for choices every project inherits are in [`docs/decisions/`](docs/decisions/README.md).

## Releases

release-please turns Conventional Commit titles into a release PR and, on merge, a `vX.Y.Z` tag. Projects update from those tags.
