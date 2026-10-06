"""Gated writes propose; they never write. Called directly, with no graph and no
approval, a write tool may add one `pending_actions` row and touch nothing it would
change on approval.
"""

from __future__ import annotations

import pytest
from tests.tool_args import sample_args

from support_agent.request_context import bind_request_context
from support_agent.tools import TOOLS, Tool, is_gated_write

# Tools that change state on approval. Adding one is a contract change.
EXPECTED_GATED_WRITES = {"submit_leave_request", "create_ticket"}

GATED_WRITES = [t for t in TOOLS if is_gated_write(t)]


def _count(conn, table: str) -> int:
    return conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]


def test_gated_writes_are_marked():
    assert sorted(t.__name__ for t in GATED_WRITES) == sorted(EXPECTED_GATED_WRITES)


# INTEGRATION TEST: this test uses the real Postgres (make db-up).
@pytest.mark.parametrize("fn", GATED_WRITES, ids=lambda t: t.__name__)
def test_gated_write_only_proposes(fn: Tool, seeded_db):
    # emp_001 has 18 days left, so a one-day leave request passes the balance check
    with bind_request_context("emp_001", "req-write", "rest"):
        fn(**sample_args(fn))

    assert _count(seeded_db, "leave_requests") == 0
    assert _count(seeded_db, "tickets") == 0
    rows = seeded_db.execute(
        "SELECT request_id, employee_id, tool, status FROM pending_actions"
    ).fetchall()
    assert rows == [("req-write", "emp_001", fn.__name__, "pending_approval")]
