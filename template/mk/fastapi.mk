# Owned by house-template-fastapi. Included by the base Makefile; adds the backend's install and
# checks to `make install`, `make check` and `make check-fast`.

BACKEND := uv --directory backend run --locked
comma := ,
# Recording and evals call the real model, so they also read keys from backend/.env when it exists.
# Variables already set (the budget caps below) win over the file. The tests ignore it.
LIVE := uv --directory backend run --locked $(if $(wildcard backend/.env),--env-file .env)

INSTALLS += install-fastapi
CHECKS += check-fastapi
FAST_CHECKS += check-fast-fastapi

.PHONY: install-fastapi check-fastapi check-fast-fastapi lock dev test db-up storage-up migrate migration record-llm eval image up llm-report

install-fastapi: ## Install the backend's Python and dependencies (from backend/uv.lock)
	uv --directory backend sync --locked

check-fastapi: ## Backend: lint, format, types, import contracts and tests (tests need Docker)
	$(BACKEND) ruff check
	$(BACKEND) ruff format --check
	$(BACKEND) mypy
	$(BACKEND) lint-imports
	$(BACKEND) pytest

check-fast-fastapi: ## Backend: lint, format and types (the pre-push and stop-hook checks)
	$(BACKEND) ruff check
	$(BACKEND) ruff format --check
	$(BACKEND) mypy

lock: ## Update backend/uv.lock after changing backend/pyproject.toml
	uv --directory backend lock

dev: ## Run the API with reload on http://127.0.0.1:8000
	$(BACKEND) fastapi dev

test: ## Run the backend tests
	$(BACKEND) pytest

db-up: ## Start the local PostgreSQL with pgvector (backend/compose.yaml) and wait until it's healthy
	docker compose up -d --wait db

storage-up: ## Start the local S3-compatible storage (RustFS, backend/compose.yaml) and create the bucket
	docker compose up -d --wait storage
	$(BACKEND) python -m app.storage

llm-report: ## Model spend from llm_call: make llm-report [days=7] [by=day,model] (by: day, model, prompt, feature)
	$(BACKEND) python -m app.llm.report --days $(or $(days),7) --by $(or $(by),day$(comma)model)

image: ## Build the API image from backend/Dockerfile
	docker compose build api

up: ## Run the backend from its image: database, migrations once, then the API on http://127.0.0.1:8000
	docker compose up -d --build --wait api

migrate: ## Apply migrations to DATABASE_URL (the local db by default)
	$(BACKEND) alembic upgrade heads

migration: ## Write a migration from model changes: make migration name="add courses"
	@test -n "$(name)" || { echo 'usage: make migration name="what changed"'; exit 2; }
	$(BACKEND) alembic revision --autogenerate --head app@head -m "$(name)"

record-llm: ## Re-record the model cassettes with real calls, with LLM_MODEL's provider key set: make record-llm [budget=0.50]
	LLM_DAILY_BUDGET_USD=$(or $(budget),0.50) $(LIVE) pytest -m vcr --record-mode=rewrite

eval: ## Run the evals against the real model, with LLM_MODEL's provider key set: make eval [budget=1] (uploads to LangSmith when LANGSMITH_API_KEY is set)
	LLM_EVAL=1 LLM_DAILY_BUDGET_USD=$(or $(budget),1) $(LIVE) pytest tests/evals -m eval
