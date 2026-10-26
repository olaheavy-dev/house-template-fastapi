# Changelog

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
