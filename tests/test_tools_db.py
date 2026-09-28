"""The tools against the seeded database: the numbers match the planted fixtures, the
checks happen in the tool, duplicates return the same action, and every call is
audited and recorded as evidence.
"""

from __future__ import annotations

import json
from datetime import date

import pytest
from tests.tool_args import sample_args

from support_agent.request_context import bind_request_context
from support_agent.tools import (
    TOOLS,
    Tool,
    create_ticket,
    get_known_outages,
    get_leave_balance,
    search_policies,
    submit_leave_request,
)
from support_agent.tools.policy_search import fuse


def _as(employee_id: str, fn: Tool, *args, **kwargs) -> dict:
    with bind_request_context(employee_id, f"req-{employee_id}", "rest"):
        return json.loads(fn(*args, **kwargs))


def _count(conn, table: str) -> int:
    return conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]


def test_fuse_ranks_ids_found_by_both_rankings_first():
    # "b" is second in both lists, "a" and "c" are first in only one
    assert fuse([["a", "b"], ["c", "b"]])[0] == "b"


@pytest.mark.parametrize(
    ("employee_id", "remaining"),
    [("emp_001", 18.0), ("emp_004", 3.5), ("emp_012", 0.0)],
)
def test_leave_balance_matches_the_planted_fixtures(seeded_db, employee_id, remaining):
    assert _as(employee_id, get_leave_balance)["remaining_days"] == remaining


def test_leave_balance_for_a_year_without_data(seeded_db):
    assert "error" in _as("emp_001", get_leave_balance, 2025)


def test_known_outages_lists_only_unresolved_ones(seeded_db):
    outages = _as("emp_002", get_known_outages)["outages"]
    assert [(o["system"], o["status"]) for o in outages] == [("vpn", "identified")]


def test_over_balance_is_refused_by_the_tool(seeded_db):
    # emp_004 has 3.5 days left
    result = _as("emp_004", submit_leave_request, date(2026, 11, 2), date(2026, 11, 11), 8)

    assert result["proposed"] is False
    assert result["remaining_days"] == 3.5
    assert _count(seeded_db, "pending_actions") == 0


def test_waiting_requests_are_subtracted_from_the_balance(seeded_db):
    first = _as("emp_004", submit_leave_request, date(2026, 11, 2), date(2026, 11, 5), 3.5)
    second = _as("emp_004", submit_leave_request, date(2026, 12, 1), date(2026, 12, 1), 1)

    assert first["proposed"] is True
    assert second["proposed"] is False
    assert second["remaining_days"] == 0.0


def test_duplicate_leave_request_returns_the_same_id(seeded_db):
    args = (date(2026, 11, 2), date(2026, 11, 6), 5)
    first = _as("emp_001", submit_leave_request, *args)
    second = _as("emp_001", submit_leave_request, *args)

    assert second["action_id"] == first["action_id"]
    assert second["already_sent"] is True
    assert _count(seeded_db, "pending_actions") == 1


def test_duplicate_ticket_returns_the_same_id(seeded_db):
    first = _as("emp_002", create_ticket, "hardware", "Screen flickers", "Since Monday.")
    second = _as("emp_002", create_ticket, "hardware", " Screen flickers ", "Since Monday.")

    assert second["action_id"] == first["action_id"]
    assert _count(seeded_db, "pending_actions") == 1


def test_search_returns_ids_that_exist(seeded_db, search_embeds_like):
    search_embeds_like("vacation-policy#entitlement#0")
    results = _as("emp_001", search_policies, "How many vacation days do I get?")["results"]
    ids = [r["chunk_id"] for r in results]

    assert ids[0] == "vacation-policy#entitlement#0"
    assert 0 < len(ids) <= 5
    known = {row[0] for row in seeded_db.execute("SELECT id FROM policy_chunks").fetchall()}
    assert set(ids) <= known


def test_search_finds_chunks_by_their_words(seeded_db, search_embeds_like):
    # the vector points at an IT chunk, so only the word search can find the carryover one
    search_embeds_like("it-access-policy#vpn-access#0")
    results = _as("emp_001", search_policies, "carry over unused vacation")["results"]

    assert "vacation-policy#carryover#0" in [r["chunk_id"] for r in results]


@pytest.mark.parametrize("fn", TOOLS, ids=lambda t: t.__name__)
def test_every_tool_call_is_audited_and_recorded(fn: Tool, seeded_db, search_embeds_like):
    search_embeds_like("vacation-policy#entitlement#0")
    with bind_request_context("emp_001", "req-audit", "rest") as ctx:
        fn(**sample_args(fn))

    rows = seeded_db.execute(
        "SELECT request_id, payload->>'tool' FROM audit_log WHERE event = 'tool_call'"
    ).fetchall()
    assert rows == [("req-audit", fn.__name__)]
    assert [(e["type"], e["tool"]) for e in ctx.evidence] == [("tool_result", fn.__name__)]
