"""A real PostgreSQL with pgvector for tests, migrated to head once per test session.

TEST_DATABASE_URL points the tests at a database you provide (a CI service container, say).
The tests migrate it down to nothing and back, so never point it at data you want to keep.
Otherwise testcontainers starts the same image backend/compose.yaml uses. Each test gets a
session inside a transaction that is rolled back afterwards, so tests never see each other's rows.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import yaml
from alembic.config import Config

BACKEND = Path(__file__).resolve().parents[1]


def db_image() -> str:
    compose = yaml.safe_load((BACKEND / "compose.yaml").read_text())
    image: str = compose["services"]["db"]["image"]
    return image


def alembic_config(url: str) -> Config:
    config = Config(toml_file=str(BACKEND / "pyproject.toml"))
    config.set_main_option("script_location", str(BACKEND / "migrations"))
    config.attributes["database_url"] = url
    return config


def database_url() -> Iterator[str]:
    if url := os.environ.get("TEST_DATABASE_URL"):
        yield url
        return
    from testcontainers.community.postgres import PostgresContainer

    with PostgresContainer(db_image(), driver="psycopg") as postgres:
        yield postgres.get_connection_url()
