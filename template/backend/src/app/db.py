"""Database access: the declarative Base, the async engine and a session per request.

SQLAlchemy 2.1 with psycopg 3 (`postgresql+psycopg://`), which serves the async app and sync
tools such as Alembic with one URL. The engine lives as long as the process (opened in the app's
lifespan); each request gets its own AsyncSession and commits explicitly, for example:

    @router.post("/courses")
    async def create_course(body: CourseIn, session: SessionDep) -> CourseOut:
        session.add(course := Course(title=body.title))
        await session.commit()
        return CourseOut.model_validate(course, from_attributes=True)

Lazy loading can't run under asyncio: load relationships with selectinload() and declare them
with lazy="raise" so a forgotten load fails loudly instead of at an awkward moment.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy import MetaData, text
from sqlalchemy.ext.asyncio import (
    AsyncAttrs,
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.settings import Settings

# Named constraints, so migrations can drop and alter them by name on every database
# (https://alembic.sqlalchemy.org/en/latest/naming.html).
NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(AsyncAttrs, DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)


def create_engine(settings: Settings) -> AsyncEngine:
    return create_async_engine(
        settings.database_url.get_secret_value(),
        pool_pre_ping=True,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
    )


def create_sessionmaker(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    # expire_on_commit=False: attributes stay readable after commit without another query.
    return async_sessionmaker(engine, expire_on_commit=False)


async def get_session(request: Request) -> AsyncIterator[AsyncSession]:
    sessionmaker: async_sessionmaker[AsyncSession] = request.app.state.sessionmaker
    async with sessionmaker() as session:
        yield session


SessionDep = Annotated[AsyncSession, Depends(get_session)]


async def ping(engine: AsyncEngine, seconds: float = 3.0) -> None:
    # Bounded: a host that drops packets would otherwise hang readiness for the TCP timeout.
    async with asyncio.timeout(seconds), engine.connect() as connection:
        await connection.execute(text("SELECT 1"))
