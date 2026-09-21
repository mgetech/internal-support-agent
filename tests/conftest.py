"""Fixtures for tests that need a real Postgres database. DB-backed tests skip
cleanly when no database is reachable, so `make test` stays green without
`make db-up`; run `make db-up` first to actually exercise them.
"""

from __future__ import annotations

from pathlib import Path

import psycopg
import pytest
from pydantic_settings import BaseSettings, SettingsConfigDict

INIT_SQL = (Path(__file__).resolve().parent.parent / "db" / "init.sql").read_text()


def _statements(sql: str) -> list[str]:
    """Split a SQL script on statement-terminating semicolons, ignoring any
    that appear inside a `--` line comment.
    """
    without_comments = "\n".join(line.split("--", 1)[0] for line in sql.splitlines())
    return [s.strip() for s in without_comments.split(";") if s.strip()]


class _DatabaseSettings(BaseSettings):
    """Reads only DATABASE_URL, from the same .env file as the real Settings.
    DB tests need nothing else, and must not fail just because unrelated
    settings (Azure OpenAI keys, absent in CI) aren't configured.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str | None = None


@pytest.fixture(scope="session")
def db_conn():
    """A connection to a database freshly (re)built from db/init.sql."""
    database_url = _DatabaseSettings().database_url
    if not database_url:
        pytest.skip("DATABASE_URL not set; run `make db-up` first")
    try:
        conn = psycopg.connect(database_url, autocommit=True, connect_timeout=2)
    except psycopg.OperationalError:
        pytest.skip("no database reachable; run `make db-up` first")
    conn.execute("DROP SCHEMA public CASCADE")
    conn.execute("CREATE SCHEMA public")
    for statement in _statements(INIT_SQL):
        conn.execute(statement)
    yield conn
    conn.close()


@pytest.fixture
def clean_db(db_conn):
    """The same connection, with every table emptied first so tests don't
    see rows left behind by another test.
    """
    tables = [
        row[0]
        for row in db_conn.execute(
            "select tablename from pg_tables where schemaname = 'public'"
        ).fetchall()
    ]
    if tables:
        db_conn.execute(f"TRUNCATE {', '.join(tables)} RESTART IDENTITY CASCADE")
    return db_conn
