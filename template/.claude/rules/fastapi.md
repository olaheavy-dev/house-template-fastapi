---
paths:
  - "backend/**"
---

# Backend (FastAPI)

Owned by house-template-fastapi. Sources: the FastAPI docs (https://fastapi.tiangolo.com/), RFC 9457 (https://www.rfc-editor.org/rfc/rfc9457), the pydantic-settings docs (https://docs.pydantic.dev/latest/concepts/pydantic_settings/), the LangChain and LangSmith docs (https://docs.langchain.com/) and pytest-recording (https://github.com/kiwicom/pytest-recording).

- The import package is `app` under `backend/src/`; the app is built by `app.main.create_app`. Process-long resources (database engine, model clients) open and close in `lifespan`, never at import time.
- Dependencies use `Annotated[T, Depends(...)]` aliases. Settings come only from `app.settings.get_settings()` (tests pass a `Settings` to `create_app`); secrets are `SecretStr`, and every new setting goes in `backend/.env.example` with a safe value.
- Errors are RFC 9457 problem details: raise `app.problems.Problem(status, detail=...)` for expected failures. Never put submitted values, secrets or stack traces in an error body; unexpected errors are logged with the request id and return a bare 500.
- Log with `structlog.get_logger()` and keyword fields (`log.info("course_imported", course_id=...)`), never f-strings, and never log secrets or personal data.
- Readiness checks are registered with `app.health.add_readiness_check`; `/health/live` never touches a dependency.
- Database: SQLAlchemy 2.1 async with psycopg 3. Inject `app.db.SessionDep`, commit explicitly, load relationships with `selectinload()` and declare them `lazy="raise"`. Models subclass `app.db.Base` and are imported in `app/models.py`.
- Every schema change is an Alembic migration (`make migration name="..."`), reviewed by hand: autogenerate misses renames, extensions and some constraints, and a migration says what happens to existing rows. `alembic check` runs in the tests, so a model without a migration fails `make check`.
- Database tests use the `db_session` or `db_client` fixtures: a real PostgreSQL with pgvector (testcontainers, or `TEST_DATABASE_URL`), rolled back after each test. Never mock the database.
- Language models only through the gateway: inject `app.llm.gateway.GatewayDep` and call `gateway.model()`. Only `app/llm/` may import `langchain` or a provider SDK (`lint-imports` enforces it); never build a model elsewhere or pass a `"provider:model"` string to LangChain, because that skips the budget. `langchain_core` (messages, runnables) is fine anywhere.
- Always await model calls (`ainvoke`, `astream`, `abatch`): the budget can't stop a synchronous `invoke()`. A call the daily budget can't cover is refused with a 503 before it is sent; every call gets an `llm_call` row with its real cost.
- A new model needs its prices in `app/llm/prices.py` (integer micro-USD per million tokens, from the provider's pricing page, with the date checked) before `LLM_MODEL` names it; the gateway refuses to start otherwise.
- Prompts are Markdown files in `app/prompts/`, loaded with `load_prompt(name)` and sent with `config=prompt.config()` so each call records the prompt's version. Never fetch prompts at runtime.
- Tests never call a model: mark a test that does `@pytest.mark.vcr` and `@pytest.mark.block_network`, and it replays `tests/cassettes/`. Record with `make record-llm` (real key, capped spend), check the cassette holds no secrets or personal data, and commit it. Test budget logic with LangChain's fake chat models.
- Tests are async (`pytestmark = pytest.mark.anyio`) and drive the app in-process through the `client` fixture; a bug fix comes with a test that fails without it.
- Dependencies: add to `backend/pyproject.toml`, run `make lock`, commit both files. FastAPI stays pinned to one minor (`<0.N+1`) because 0.x minors can break.
