"""Keeps the tests away from a database that holds real data.

The database tests drop and rebuild the schema of the database they connect to. So the
suite refuses to start unless that database is a test database, which is one whose name
ends with `_test` or `_scratch`. The check runs before any table is touched.

`make test` runs this file first (`python -m tests.database_guard`). It takes the database
from DATABASE_URL, makes sure that a test database exists on the same server (the name plus
`_test`), and prints its URL. A URL that is a test database already is printed as it is.
"""

from __future__ import annotations

import sys

import psycopg
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo
from pydantic_settings import BaseSettings, SettingsConfigDict

TEST_DATABASE_SUFFIXES = ("_test", "_scratch")


class DatabaseSettings(BaseSettings):
    """Reads only DATABASE_URL, from the same .env file as the real Settings.
    DB tests need nothing else, and must not fail just because unrelated
    settings (Azure OpenAI keys, absent in CI) aren't configured.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str | None = None


def get_database_name(url: str) -> str:
    """The database name in a URL or a key=value connection string. Empty when there is
    none, or when the text cannot be read.
    """
    try:
        return conninfo_to_dict(url).get("dbname") or ""
    except psycopg.ProgrammingError:
        return ""


def check_test_database(url: str | None) -> str | None:
    """A message that says why the tests must not run against this database, or None when
    they may. No database configured is fine: the database tests skip then.
    """
    if not url:
        return None
    name = get_database_name(url)
    if name.endswith(TEST_DATABASE_SUFFIXES):
        return None
    return (
        f"The tests drop and rebuild the schema of the database they connect to, and "
        f"{name or 'this database'!r} is not a test database. The name of a test database "
        f"ends with {' or '.join(TEST_DATABASE_SUFFIXES)}. Point DATABASE_URL at one, or "
        f"run `make test`, which creates and uses support_agent_test."
    )


def get_test_database_url(url: str) -> str:
    """The URL of the test database that belongs to this database: the same server, with
    `_test` added to the name. A URL that is a test database already stays as it is.
    """
    name = get_database_name(url)
    if name.endswith(TEST_DATABASE_SUFFIXES):
        return url
    return make_conninfo(url, dbname=f"{name or 'support_agent'}_test")


def create_test_database(url: str) -> None:
    """Create the database of this URL on its server, if it is missing."""
    name = get_database_name(url)
    with psycopg.connect(make_conninfo(url, dbname="postgres"), autocommit=True) as conn:
        exists = conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (name,)).fetchone()
        if not exists:
            conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))


def main() -> int:
    """Print the URL that the tests should use. Prints nothing when no database is
    configured. If the server cannot be reached, the URL is still printed, and the
    database tests skip.
    """
    url = DatabaseSettings().database_url
    if not url:
        return 0
    test_url = get_test_database_url(url)
    if test_url != url:
        try:
            create_test_database(test_url)
        except psycopg.OperationalError as error:
            print(f"could not create the test database: {type(error).__name__}", file=sys.stderr)
    print(test_url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
