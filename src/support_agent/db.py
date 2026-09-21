"""Postgres access: a connection pool and parameterized-query helpers. No ORM —
every query is a plain string with placeholders, never string interpolation.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from support_agent.config import get_settings

_pool: ConnectionPool | None = None


def get_pool() -> ConnectionPool:
    """The process-wide connection pool, opened lazily on first use."""
    global _pool
    if _pool is None:
        _pool = ConnectionPool(get_settings().database_url, open=True)
    return _pool


@contextmanager
def transaction() -> Iterator[psycopg.Cursor[dict[str, Any]]]:
    """A cursor scoped to one transaction: commits on success, rolls back on
    exception, and always returns the connection to the pool.
    """
    with get_pool().connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        yield cur


def fetch_all(query: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    with transaction() as cur:
        cur.execute(query, params)
        return cur.fetchall()


def fetch_one(query: str, params: tuple[Any, ...] = ()) -> dict[str, Any] | None:
    with transaction() as cur:
        cur.execute(query, params)
        return cur.fetchone()


def execute(query: str, params: tuple[Any, ...] = ()) -> None:
    with transaction() as cur:
        cur.execute(query, params)
