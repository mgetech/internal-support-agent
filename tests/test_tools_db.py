"""The tools against the seeded database: the numbers match the planted fixtures, the
checks happen in the tool, duplicates return the same action, and every call is
audited and recorded as evidence.
"""

from __future__ import annotations

import json
from datetime import date

import pytest
from data.generate import GENERATED_AS_OF
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
from support_agent.tools.hr_it import calculate_working_days
from support_agent.tools.policy_search import fuse


def _as(employee_id: str, fn: Tool, *args, **kwargs) -> dict:
    with bind_request_context(employee_id, f"req-{employee_id}", "rest"):
        return json.loads(fn(*args, **kwargs))


def _count(conn, table: str) -> int:
    return conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]


def test_fuse_ranks_ids_found_by_both_rankings_first():
    # "b" is second in both lists, "a" and "c" are first in only one
    assert fuse([["a", "b"], ["c", "b"]])[0] == "b"


# INTEGRATION TESTS: the tests below use the real Postgres (make db-up).
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


@pytest.mark.parametrize(
    ("start", "end", "expected"),
    [
        (date(2026, 11, 2), date(2026, 11, 13), 10),  # two full weeks
        (date(2026, 11, 6), date(2026, 11, 9), 2),  # Friday to Monday: the weekend is out
        (date(2026, 11, 2), date(2026, 11, 2), 1),
        (date(2026, 11, 7), date(2026, 11, 8), 0),  # Saturday and Sunday only
    ],
)
def test_working_days_are_monday_to_friday(start, end, expected):
    assert calculate_working_days(start, end) == expected


def test_the_tool_counts_the_working_days_itself(seeded_db):
    result = _as("emp_001", submit_leave_request, date(2026, 11, 2), date(2026, 11, 13))

    assert result["proposed"] is True
    assert result["working_days"] == 10
    payload = seeded_db.execute("SELECT payload FROM pending_actions").fetchone()[0]
    assert payload["working_days"] == 10


def test_a_range_over_a_weekend_does_not_count_it(seeded_db):
    result = _as("emp_001", submit_leave_request, date(2026, 11, 6), date(2026, 11, 9))

    assert result["working_days"] == 2


def test_a_request_can_start_today(seeded_db):
    # the fixed test date is Tuesday 2026-09-22
    result = _as("emp_001", submit_leave_request, GENERATED_AS_OF, GENERATED_AS_OF)

    assert result["proposed"] is True
    assert result["working_days"] == 1


@pytest.mark.parametrize(
    ("start", "end", "reason"),
    [
        (date(2026, 11, 10), date(2026, 11, 9), "the end date is before the start date"),
        (date(2026, 9, 21), date(2026, 9, 22), "the start date is in the past"),
        (
            date(2026, 11, 7),
            date(2026, 11, 8),
            "the range has no working days; it covers only a weekend",
        ),
    ],
    ids=["end-before-start", "start-in-the-past", "no-working-days"],
)
def test_a_bad_range_is_refused_before_any_balance_check(seeded_db, start, end, reason):
    result = _as("emp_001", submit_leave_request, start, end)

    assert result == {"proposed": False, "reason": reason}
    assert _count(seeded_db, "pending_actions") == 0


def test_over_balance_is_refused_by_the_tool(seeded_db):
    # emp_004 has 3.5 days left and asks for 10 working days
    result = _as("emp_004", submit_leave_request, date(2026, 11, 2), date(2026, 11, 13))

    assert result["proposed"] is False
    assert result["requested_days"] == 10
    assert result["remaining_days"] == 3.5
    assert _count(seeded_db, "pending_actions") == 0


def test_waiting_requests_are_subtracted_from_the_balance(seeded_db):
    first = _as("emp_004", submit_leave_request, date(2026, 11, 2), date(2026, 11, 3))
    second = _as("emp_004", submit_leave_request, date(2026, 11, 9), date(2026, 11, 10))

    assert first["proposed"] is True
    assert second["proposed"] is False
    assert second["remaining_days"] == 1.5


def test_duplicate_leave_request_returns_the_same_id(seeded_db):
    args = (date(2026, 11, 2), date(2026, 11, 6))
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
