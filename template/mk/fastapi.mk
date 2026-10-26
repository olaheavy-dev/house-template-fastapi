# Owned by house-template-fastapi. Included by the base Makefile; adds the backend's install and
# checks to `make install`, `make check` and `make check-fast`.

BACKEND := uv --directory backend run --locked

INSTALLS += install-fastapi
CHECKS += check-fastapi
FAST_CHECKS += check-fast-fastapi

.PHONY: install-fastapi check-fastapi check-fast-fastapi lock dev test db-up storage-up migrate migration record-llm eval

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

migrate: ## Apply migrations to DATABASE_URL (the local db by default)
	$(BACKEND) alembic upgrade head

migration: ## Write a migration from model changes: make migration name="add courses"
	@test -n "$(name)" || { echo 'usage: make migration name="what changed"'; exit 2; }
	$(BACKEND) alembic revision --autogenerate -m "$(name)"

record-llm: ## Re-record the model cassettes with real calls: ANTHROPIC_API_KEY=... make record-llm [budget=0.50]
	@test -n "$$ANTHROPIC_API_KEY" || { echo 'set ANTHROPIC_API_KEY to record'; exit 2; }
	LLM_DAILY_BUDGET_USD=$(or $(budget),0.50) $(BACKEND) pytest -m vcr --record-mode=rewrite

eval: ## Run the evals against the real model: ANTHROPIC_API_KEY=... make eval [budget=1] (uploads to LangSmith when LANGSMITH_API_KEY is set)
	@test -n "$$ANTHROPIC_API_KEY" || { echo 'evals call the real model: set ANTHROPIC_API_KEY'; exit 2; }
	LLM_EVAL=1 LLM_DAILY_BUDGET_USD=$(or $(budget),1) \
	LANGSMITH_TEST_TRACKING=$$([ -n "$$LANGSMITH_API_KEY" ] && echo true || echo false) \
	LANGSMITH_TRACING=$$([ -n "$$LANGSMITH_API_KEY" ] && echo true || echo false) \
	$(BACKEND) pytest tests/evals -m eval
