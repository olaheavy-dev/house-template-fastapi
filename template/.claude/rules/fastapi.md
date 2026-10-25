---
paths:
  - "backend/**"
---

# Backend (FastAPI)

Owned by house-template-fastapi. Sources: the FastAPI docs (https://fastapi.tiangolo.com/), RFC 9457 (https://www.rfc-editor.org/rfc/rfc9457) and the pydantic-settings docs (https://docs.pydantic.dev/latest/concepts/pydantic_settings/).

- The import package is `app` under `backend/src/`; the app is built by `app.main.create_app`. Process-long resources (database engine, model clients) open and close in `lifespan`, never at import time.
- Dependencies use `Annotated[T, Depends(...)]` aliases. Settings come only from `app.settings.get_settings()` (tests pass a `Settings` to `create_app`); secrets are `SecretStr`, and every new setting goes in `backend/.env.example` with a safe value.
- Errors are RFC 9457 problem details: raise `app.problems.Problem(status, detail=...)` for expected failures. Never put submitted values, secrets or stack traces in an error body; unexpected errors are logged with the request id and return a bare 500.
- Log with `structlog.get_logger()` and keyword fields (`log.info("course_imported", course_id=...)`), never f-strings, and never log secrets or personal data.
- Readiness checks are registered with `app.health.add_readiness_check`; `/health/live` never touches a dependency.
- Database: SQLAlchemy 2.1 async with psycopg 3. Inject `app.db.SessionDep`, commit explicitly, load relationships with `selectinload()` and declare them `lazy="raise"`. Models subclass `app.db.Base` and are imported in `app/models.py`.
- Every schema change is an Alembic migration (`make migration name="..."`), reviewed by hand: autogenerate misses renames, extensions and some constraints, and a migration says what happens to existing rows. `alembic check` runs in the tests, so a model without a migration fails `make check`.
- Database tests use the `db_session` or `db_client` fixtures: a real PostgreSQL with pgvector (testcontainers, or `TEST_DATABASE_URL`), rolled back after each test. Never mock the database.
- Only the model gateway may import a model provider SDK; `lint-imports` (in `make check`) enforces it.
- Tests are async (`pytestmark = pytest.mark.anyio`) and drive the app in-process through the `client` fixture; a bug fix comes with a test that fails without it.
- Dependencies: add to `backend/pyproject.toml`, run `make lock`, commit both files. FastAPI stays pinned to one minor (`<0.N+1`) because 0.x minors can break.
