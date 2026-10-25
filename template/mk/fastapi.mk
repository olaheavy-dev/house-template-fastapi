# Owned by house-template-fastapi. Included by the base Makefile; adds the backend's install and
# checks to `make install`, `make check` and `make check-fast`.

BACKEND := uv --directory backend run --locked

INSTALLS += install-fastapi
CHECKS += check-fastapi
FAST_CHECKS += check-fast-fastapi

.PHONY: install-fastapi check-fastapi check-fast-fastapi lock dev test

install-fastapi: ## Install the backend's Python and dependencies (from backend/uv.lock)
	uv --directory backend sync --locked

check-fastapi: ## Backend: lint, format, types, import contracts and tests
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
