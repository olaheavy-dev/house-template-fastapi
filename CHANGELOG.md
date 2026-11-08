# Changelog

## [0.7.0](https://github.com/olaheavy-dev/house-template-fastapi/compare/v0.6.1...v0.7.0) (2026-11-08)


### ⚠ BREAKING CHANGES

* `app.llm.checkpoints.*` moved to `app.llm.langchain.checkpoints`; `app.llm.gateway.Gateway` and `get_gateway` moved to `app.llm.langchain.gateway`; `app.llm.gateway.GatewayDep` moved to `app.llm.GatewayDep`. The old paths warn until 0.8.0 and return the same objects. A project's own migration that calls `apply_migrations` must import the new module, and patch targets such as `openai_client` and `init_chat_model` are now on `app.llm.langchain.gateway`.
* `Prompt`, `load_prompt` and `PROMPTS` moved: `app.llm.prompts.Prompt` and `load_prompt` are now `app.llm.langchain.prompts.Prompt` and `load_prompt`, and `app.llm.prompts.PROMPTS` is now `app.llm.core.prompts.PROMPTS`. The spend report moved: `app.llm.report.*` is now `app.llm.core.report.*`, and `python -m app.llm.report` (and `make llm-report`) is now `python -m app.llm.core.report`. The old paths and the old command warn until 0.8.0 and return the same objects.
* `app.llm.embeddings.GatewayEmbeddings` and `openai_client` moved to `app.llm.langchain.embeddings.GatewayEmbeddings` and `app.llm.core.openai_embeddings.openai_client` (the old paths warn until 0.8.0). The private `_KINDS`, `_kind_of`, `_as_model_error`, `_batches`, `_embed` and `_settle` are gone, as are the non-budget names the module used to import. An embedding price's `max_prompt_tokens` is now enforced per request.
* `settle(session, ..., usage=)` and `record_call(attributes, status, cost, usage, latency_ms)` now take `app.llm.core.usage.Usage`, or `None` for no usage; `{}` is no longer accepted. Convert LangChain usage with `app.llm.langchain.usage.from_langchain`. The budget moved and its old paths warn until 0.8.0 and return the same objects: `app.llm.budget.{LlmCall, LlmSpendDay, LlmSpendUserDay, BudgetExhausted, UserBudgetExhausted, reserve, reserve_for, settle, expire_stale, utc_today, WARN_SHARE}` are now `app.llm.core.budget.*`; `app.llm.budget.{PromptTooLong, UnboundedMedia, UnpricedRequest, check_priced, INPUT_MARGIN_TOKENS, IMAGE_TOKENS}` are now `app.llm.core.bounds.*`; `app.llm.budget.{BudgetGuard, input_bound}` are now `app.llm.langchain.callbacks.*`; `app.llm.metrics.*` is now `app.llm.core.metrics.*`. New: `app.llm.core.budget.Meter` and `CallTags`.
* model-failure handling moved. `app.llm.errors.Outcome` and `DEFAULT_RETRY_AFTER` are now `app.llm.core.errors.Outcome` and `DEFAULT_RETRY_AFTER`; `app.llm.errors.OUTCOMES`, `outcome_for` and `install_model_error_handlers` are now `app.llm.langchain.errors.OUTCOMES`, `outcome_for` and `install_model_error_handlers`. The old paths warn until 0.8.0 and return the same objects. `install_model_error_handlers` still registers only LangChain's and LangGraph's exceptions; call `app.llm.core.errors.install_failure_handlers(app)` beside it so the new `ProviderError` and `StepLimitReached` are answered.
* `cost(price, usage)` and `token_details(usage)` now take `app.llm.core.usage.Usage` (or `None` for no usage; `{}` is no longer accepted by `token_details`). Convert LangChain usage with `app.llm.langchain.usage.from_langchain`.
* alembic, sqlalchemy, psycopg, pgvector and the LangChain/LangGraph/LangSmith packages moved from the main dependencies into the `db` and `langchain` extras; a plain `uv sync` or `pip install .` now needs `--extra db --extra langchain` (the make targets and the image already install them). Projects made from v0.6.1 or earlier cannot `copier update`: move them once with `copier recopy --trust --defaults --overwrite -a .copier-answers.fastapi.yml --vcs-ref <tag>` (v0.7.0 or newer; commit first; see the README Upgrading section), then run `make lock`. After that, update with `copier update --trust --defaults`.
* boto3 is now an optional extra named `storage`. Projects that install dependencies without the Makefile or the Dockerfile (for example a plain `uv sync`) lose boto3: run `uv sync --locked --extra storage`, or `make install` / `make image`, which pass the extra. After `copier update`, run `make lock` then `make install`. Turning storage off with `copier update --trust --defaults -d storage=false` deletes the template's storage files even if you edited them, so commit first.
* projects made from v0.6.1 or earlier cannot `copier update`. Follow the one-time `copier recopy` steps in the README Upgrading section (commit first, review `git diff`, then update normally with `copier update --trust -a .copier-answers.fastapi.yml`).
* the template no longer ships default models or prices. Set `LLM_MODEL` and `EMBEDDING_MODEL` in the environment or `backend/.env`: a deployment that relied on the old defaults will not start. Every `Price(...)` entry in `app/llm/prices.py` now requires the keyword-only arguments `checked=` (a YYYY-MM-DD date) and `source=` (an https URL); an entry without them, such as one written for 0.6.1, raises a `TypeError` when `prices.py` is imported.

### Features

* make object storage optional ([#54](https://github.com/olaheavy-dev/house-template-fastapi/issues/54)) ([a8837db](https://github.com/olaheavy-dev/house-template-fastapi/commit/a8837db1475957c0a64248be2100419f5e4da9eb))
* optional database and migrations ([#29](https://github.com/olaheavy-dev/house-template-fastapi/issues/29)) ([#60](https://github.com/olaheavy-dev/house-template-fastapi/issues/60)) ([6f346ba](https://github.com/olaheavy-dev/house-template-fastapi/commit/6f346ba7e90accf9a31fb45a459aaf1e626c0c3a))


### Bug Fixes

* copier update works for projects made from this layer ([#53](https://github.com/olaheavy-dev/house-template-fastapi/issues/53)) ([91943a4](https://github.com/olaheavy-dev/house-template-fastapi/commit/91943a4f696d414abb721066926bf1b926cbaad8))
* **deps:** bump the backend-minor-and-patch group across 1 directory with 10 updates ([#57](https://github.com/olaheavy-dev/house-template-fastapi/issues/57)) ([4a684fd](https://github.com/olaheavy-dev/house-template-fastapi/commit/4a684fdc823d8a9d43a2a01da8634fb16ce5d08f))
* list the gateway shims and the template's migrations as template-owned ([#33](https://github.com/olaheavy-dev/house-template-fastapi/issues/33)) (T6a) ([#83](https://github.com/olaheavy-dev/house-template-fastapi/issues/83)) ([cb46fd2](https://github.com/olaheavy-dev/house-template-fastapi/commit/cb46fd29539ca3c11a39b7f2aa05c77c5737ab05))
* no default models or prices; every price records its date and source ([#42](https://github.com/olaheavy-dev/house-template-fastapi/issues/42)) ([d67ca38](https://github.com/olaheavy-dev/house-template-fastapi/commit/d67ca3807ae74a78cc9728bf20da829be035117c))
* recording checks the keys of both pinned models; follow-ups from the no-default-models review ([#52](https://github.com/olaheavy-dev/house-template-fastapi/issues/52)) ([b34c5c6](https://github.com/olaheavy-dev/house-template-fastapi/commit/b34c5c6a09b45e91a0c6ff835f1edc1359b912f1))
* run the generated backend on its own Python, and keep pytest starting on 3.14 ([#62](https://github.com/olaheavy-dev/house-template-fastapi/issues/62)) ([160b845](https://github.com/olaheavy-dev/house-template-fastapi/commit/160b8453a96e996355f1d7e5ff46e0c14f3bbcda))


### Code Refactoring

* move embeddings onto the metered core and OpenAI's call into core/openai_embeddings ([#33](https://github.com/olaheavy-dev/house-template-fastapi/issues/33)) (T4) ([#70](https://github.com/olaheavy-dev/house-template-fastapi/issues/70)) ([ae6dfad](https://github.com/olaheavy-dev/house-template-fastapi/commit/ae6dfad22969e4e61981dc5490cc48ddb1a2716b))
* move model-failure classification and its problem answers into the framework-neutral core ([#33](https://github.com/olaheavy-dev/house-template-fastapi/issues/33)) ([#66](https://github.com/olaheavy-dev/house-template-fastapi/issues/66)) ([e948247](https://github.com/olaheavy-dev/house-template-fastapi/commit/e948247393644749769d2db84273e255e8d9b77b))
* move prompts and the spend report into the framework-neutral core ([#33](https://github.com/olaheavy-dev/house-template-fastapi/issues/33)) (T5a) ([#79](https://github.com/olaheavy-dev/house-template-fastapi/issues/79)) ([5f4e2dd](https://github.com/olaheavy-dev/house-template-fastapi/commit/5f4e2ddf42b5a99d981251707bd3915f9c734a28))
* move the daily budget, its request bounds and the llm metrics into the framework-neutral core ([#33](https://github.com/olaheavy-dev/house-template-fastapi/issues/33)) ([#67](https://github.com/olaheavy-dev/house-template-fastapi/issues/67)) ([c7765eb](https://github.com/olaheavy-dev/house-template-fastapi/commit/c7765ebde5595fd9159e0495ec2600c9d2640562))
* move the gateway and the checkpointer into app/llm/langchain ([#33](https://github.com/olaheavy-dev/house-template-fastapi/issues/33)) (T5b) ([#81](https://github.com/olaheavy-dev/house-template-fastapi/issues/81)) ([4e6a630](https://github.com/olaheavy-dev/house-template-fastapi/commit/4e6a630ec5978dfa6498215ca16c7d6acccb8f82))
* move usage, price arithmetic and provider lookup into a framework-neutral core ([#33](https://github.com/olaheavy-dev/house-template-fastapi/issues/33)) ([#65](https://github.com/olaheavy-dev/house-template-fastapi/issues/65)) ([4f010e4](https://github.com/olaheavy-dev/house-template-fastapi/commit/4f010e4328dd97a07030f77545e092043be1b528))

## [0.6.1](https://github.com/olaheavy-dev/house-template-fastapi/compare/v0.6.0...v0.6.1) (2026-10-27)


### Bug Fixes

* meter embeddings and refuse calls billed beyond the price table ([#26](https://github.com/olaheavy-dev/house-template-fastapi/issues/26)) ([6fda88f](https://github.com/olaheavy-dev/house-template-fastapi/commit/6fda88f960a0817655bcae44c34bb556a9302a58))

## [0.6.0](https://github.com/olaheavy-dev/house-template-fastapi/compare/v0.5.0...v0.6.0) (2026-10-26)


### Features

* a request body limit, and presigned uploads limited by signed size and allowed types ([#22](https://github.com/olaheavy-dev/house-template-fastapi/issues/22)) ([3a6c2b3](https://github.com/olaheavy-dev/house-template-fastapi/commit/3a6c2b3e755ec3325c3ba07e70b6edd3bcf91d8e))
* an authentication hook, a per-user model budget and an agent step limit ([#21](https://github.com/olaheavy-dev/house-template-fastapi/issues/21)) ([da31eb8](https://github.com/olaheavy-dev/house-template-fastapi/commit/da31eb80eded65d516d530159bd9d615a08c02f2))
* observability with OpenTelemetry request traces and llm metrics, call details, a spend report and a budget warning ([#17](https://github.com/olaheavy-dev/house-template-fastapi/issues/17)) ([f2aecda](https://github.com/olaheavy-dev/house-template-fastapi/commit/f2aecda93460895479d38f0a34353636099775f6))
* problem responses for model and storage failures, and JSON logs for the app, uvicorn and libraries ([#20](https://github.com/olaheavy-dev/house-template-fastapi/issues/20)) ([f6fdc9f](https://github.com/olaheavy-dev/house-template-fastapi/commit/f6fdc9f41e3b3c878966d43a89f52a74da3d53a4))
* production hardening with timeouts, a budget pool, graceful shutdown, CORS, allowed hosts and security headers ([#23](https://github.com/olaheavy-dev/house-template-fastapi/issues/23)) ([ab54d64](https://github.com/olaheavy-dev/house-template-fastapi/commit/ab54d64ee3d68433895d837324e9586e62ddc134))
* versioned API conventions with a /v1 router, an example protected route and cursor pagination ([#25](https://github.com/olaheavy-dev/house-template-fastapi/issues/25)) ([d632913](https://github.com/olaheavy-dev/house-template-fastapi/commit/d63291385a078fb38ea0d7eb93381c3e25903449))


### Bug Fixes

* close stuck reservations, refuse unpriced long prompts, and keep the compose images updated ([#24](https://github.com/olaheavy-dev/house-template-fastapi/issues/24)) ([6b5587e](https://github.com/olaheavy-dev/house-template-fastapi/commit/6b5587e2ef98f85bf80b62f2ab11abfd56ee06b5))
* keep template and project migrations on separate Alembic branches ([#19](https://github.com/olaheavy-dev/house-template-fastapi/issues/19)) ([a9e452f](https://github.com/olaheavy-dev/house-template-fastapi/commit/a9e452fc01564f4c9aac7272967c034d16dbeaad))

## [0.5.0](https://github.com/olaheavy-dev/house-template-fastapi/compare/v0.4.0...v0.5.0) (2026-10-26)


### Features

* live evals of the gateway against the real provider, with keys read from backend/.env ([#13](https://github.com/olaheavy-dev/house-template-fastapi/issues/13)) ([801ad41](https://github.com/olaheavy-dev/house-template-fastapi/commit/801ad41652e30594558b0df918526cd00487d470))

## [0.4.0](https://github.com/olaheavy-dev/house-template-fastapi/compare/v0.3.0...v0.4.0) (2026-10-26)


### Features

* default to OpenAI, and let recording and evals follow LLM_MODEL's provider ([#12](https://github.com/olaheavy-dev/house-template-fastapi/issues/12)) ([e62c3da](https://github.com/olaheavy-dev/house-template-fastapi/commit/e62c3dae76049ae2d0faf40060bd56de84999711))
* the API container image, a one-off migrate service and an attested publish workflow ([#10](https://github.com/olaheavy-dev/house-template-fastapi/issues/10)) ([55bf8cb](https://github.com/olaheavy-dev/house-template-fastapi/commit/55bf8cbc623487f93586ad8ff1c39e2f6706a720))

## [0.3.0](https://github.com/olaheavy-dev/house-template-fastapi/compare/v0.2.0...v0.3.0) (2026-10-26)


### Features

* agents on the gateway's model with LangGraph's Postgres checkpointer, and LangSmith eval suites ([#7](https://github.com/olaheavy-dev/house-template-fastapi/issues/7)) ([2c7e215](https://github.com/olaheavy-dev/house-template-fastapi/commit/2c7e215257be1ef7e1ca5644aefa53a3aa6c789c))
* object storage over the S3 API with presigned URLs, R2 in production and RustFS locally ([#9](https://github.com/olaheavy-dev/house-template-fastapi/issues/9)) ([22f0b8b](https://github.com/olaheavy-dev/house-template-fastapi/commit/22f0b8bea907d8320379f09115a69db7092dc67c))

## [0.2.0](https://github.com/olaheavy-dev/house-template-fastapi/compare/v0.1.0...v0.2.0) (2026-10-26)


### Features

* model gateway with a daily budget, cost rows, versioned prompts, LangSmith tracing and recorded responses ([#5](https://github.com/olaheavy-dev/house-template-fastapi/issues/5)) ([18f90be](https://github.com/olaheavy-dev/house-template-fastapi/commit/18f90be7e27ed0698d0a65df5c51b5062128150c))

## 0.1.0 (2026-10-26)


### Features

* app skeleton and Copier template for the FastAPI layer ([#1](https://github.com/olaheavy-dev/house-template-fastapi/issues/1)) ([5ebab65](https://github.com/olaheavy-dev/house-template-fastapi/commit/5ebab65cc3c1b65bced11434bbeef2fdaa476a6d))
* database layer with PostgreSQL, pgvector, async SQLAlchemy and Alembic ([#2](https://github.com/olaheavy-dev/house-template-fastapi/issues/2)) ([3ff3038](https://github.com/olaheavy-dev/house-template-fastapi/commit/3ff3038453e64b36bc1986e6069821e7b76ac268))
