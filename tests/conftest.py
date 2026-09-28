"""Fixtures for tests that need a real Postgres database. DB-backed tests skip
cleanly when no database is reachable, so `make test` stays green without
`make db-up`; run `make db-up` first to actually exercise them.
"""

from __future__ import annotations

from pathlib import Path

import psycopg
import pytest
from data.generate import (
    generate_employees,
    generate_known_outages,
    generate_leave_balances,
    generate_policy_chunks,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

from support_agent.tools import policy_search

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


def fake_vector(position: int) -> list[float]:
    """A vector that points at one position only. Two fake vectors are the same only
    when their positions are the same, so a query with chunk N's vector finds chunk N
    first. Tests use these instead of real embeddings, so they need no Azure keys.
    """
    vector = [0.0] * 1536
    vector[position] = 1.0
    return vector


@pytest.fixture
def seeded_db(clean_db):
    """The generated employees, balances, outages and policy chunks. Chunk number i
    (in generate_policy_chunks order) gets fake_vector(i).
    """
    for e in generate_employees():
        clean_db.execute(
            "INSERT INTO employees (id, name, role, employment_type, weekly_hours, country,"
            " hired_at) VALUES (%(id)s, %(name)s, %(role)s, %(employment_type)s,"
            " %(weekly_hours)s, %(country)s, %(hired_at)s)",
            e,
        )
    for b in generate_leave_balances():
        clean_db.execute(
            "INSERT INTO leave_balances (employee_id, year, entitlement_days, taken_days,"
            " pending_days) VALUES (%(employee_id)s, %(year)s, %(entitlement_days)s,"
            " %(taken_days)s, %(pending_days)s)",
            b,
        )
    for o in generate_known_outages():
        clean_db.execute(
            "INSERT INTO known_outages (system, status, started_at, note)"
            " VALUES (%(system)s, %(status)s, %(started_at)s, %(note)s)",
            o,
        )
    for i, c in enumerate(generate_policy_chunks()):
        clean_db.execute(
            "INSERT INTO policy_chunks (id, policy, heading, content, domain, chunker, embedding)"
            " VALUES (%s, %s, %s, %s, %s, %s, %s::vector)",
            (
                c["id"],
                c["policy"],
                c["heading"],
                c["content"],
                c["domain"],
                c["chunker"],
                str(fake_vector(i)),
            ),
        )
    return clean_db


@pytest.fixture
def search_embeds_like(monkeypatch):
    """Returns a function. Calling it with a chunk id makes search_policies embed any
    question as that chunk's fake vector, with no call to Azure.
    """
    positions = {c["id"]: i for i, c in enumerate(generate_policy_chunks())}

    def point_at(chunk_id: str) -> None:
        vector = fake_vector(positions[chunk_id])
        monkeypatch.setattr(policy_search, "embed_texts", lambda texts: [vector for _ in texts])

    return point_at
