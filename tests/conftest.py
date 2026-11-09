"""Fixtures for tests that need a real Postgres database. DB-backed tests skip
cleanly when no database is reachable, so `make test` stays green without
`make db-up`; run `make db-up` first to actually exercise them.
"""

from __future__ import annotations

import uuid
from datetime import date
from pathlib import Path

import psycopg
import pytest
from data.generate import (
    calculate_index_build_id,
    generate_employees,
    generate_known_outages,
    generate_leave_balances,
    generate_policy_chunks,
)
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from tests.database_guard import DatabaseSettings, check_test_database

from support_agent import tracing, util
from support_agent.config import Settings
from support_agent.tools import policy_search

# the date every test runs on, a Tuesday. The seed is generated as of this date too.
TEST_TODAY = date(2026, 9, 22)

INIT_SQL = (Path(__file__).resolve().parent.parent / "db" / "init.sql").read_text()


def _statements(sql: str) -> list[str]:
    """Split a SQL script on statement-terminating semicolons, ignoring any
    that appear inside a `--` line comment.
    """
    without_comments = "\n".join(line.split("--", 1)[0] for line in sql.splitlines())
    return [s.strip() for s in without_comments.split(";") if s.strip()]


def pytest_configure(config):
    """Stop before any test runs if the database is not a test database. The database tests
    drop and rebuild the schema of the database they connect to. See database_guard.py.
    """
    problem = check_test_database(DatabaseSettings().database_url)
    if problem:
        pytest.exit(problem, returncode=3)


@pytest.fixture(autouse=True)
def fixed_today(monkeypatch):
    """Every test runs on the date the data was generated for, so leave dates in the
    tests never turn into dates in the past. A test can patch `util.get_today` again.
    """
    monkeypatch.setattr(util, "get_today", lambda: TEST_TODAY)


@pytest.fixture(autouse=True)
def tracing_off(monkeypatch):
    """Tracing is off in every test, even when .env has Langfuse keys, so no test sends a
    trace. Inputs and outputs are off too. A test turns tracing on with the `spans` fixture,
    and the content on with `monkeypatch.setattr(tracing, "is_content_traced", lambda: True)`.
    """
    monkeypatch.setattr(tracing, "get_langfuse", lambda: None)
    monkeypatch.setattr(tracing, "is_content_traced", lambda: False)


@pytest.fixture
def spans(monkeypatch):
    """Turns tracing on, with a client that keeps the spans in memory. Returns a function
    that gives the spans finished so far.
    """
    exporter = InMemorySpanExporter()
    settings = Settings(
        _env_file=None,
        database_url="postgresql://user:pass@localhost:5432/support_agent",
        azure_openai_endpoint="https://example.invalid/openai/v1",
        azure_openai_api_key="test-key",
        azure_openai_deployment_agent="main",
        azure_openai_deployment_classifier="small",
        azure_openai_deployment_embedding="embedding",
        # each client needs its own public key, because Langfuse keeps one per key
        langfuse_public_key=f"pk-lf-{uuid.uuid4().hex}",
        langfuse_secret_key="sk-lf-test",
        langfuse_base_url="https://langfuse.invalid",
    )
    client = tracing.create_langfuse(settings, span_exporter=exporter)
    monkeypatch.setattr(tracing, "get_langfuse", lambda: client)

    def finished():
        client.flush()
        return exporter.get_finished_spans()

    yield finished
    client.shutdown()


@pytest.fixture(scope="session")
def db_conn():
    """A connection to a database freshly (re)built from db/init.sql."""
    database_url = DatabaseSettings().database_url
    if not database_url:
        pytest.skip("DATABASE_URL not set; run `make db-up` first")
    # the same check as at the start, right before the schema is dropped
    problem = check_test_database(database_url)
    if problem:
        pytest.exit(problem, returncode=3)
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
    clean_db.execute(
        "INSERT INTO seed_meta (as_of, index_build_id) VALUES (%s, %s)",
        (TEST_TODAY, calculate_index_build_id(generate_policy_chunks(), "test-embedding")),
    )
    for e in generate_employees(TEST_TODAY):
        clean_db.execute(
            "INSERT INTO employees (id, name, role, employment_type, weekly_hours, country,"
            " hired_at) VALUES (%(id)s, %(name)s, %(role)s, %(employment_type)s,"
            " %(weekly_hours)s, %(country)s, %(hired_at)s)",
            e,
        )
    for b in generate_leave_balances(TEST_TODAY):
        clean_db.execute(
            "INSERT INTO leave_balances (employee_id, year, entitlement_days, taken_days,"
            " approved_days) VALUES (%(employee_id)s, %(year)s, %(entitlement_days)s,"
            " %(taken_days)s, %(approved_days)s)",
            b,
        )
    for o in generate_known_outages(TEST_TODAY):
        clean_db.execute(
            "INSERT INTO known_outages (system, status, started_at, note)"
            " VALUES (%(system)s, %(status)s, %(started_at)s, %(note)s)",
            o,
        )
    for i, c in enumerate(generate_policy_chunks()):
        clean_db.execute(
            "INSERT INTO policy_chunks (id, policy, heading, content, domain, chunker,"
            " policy_version, embedding)"
            " VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector)",
            (
                c["id"],
                c["policy"],
                c["heading"],
                c["content"],
                c["domain"],
                c["chunker"],
                c["policy_version"],
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
