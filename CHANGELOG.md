# Changelog

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
