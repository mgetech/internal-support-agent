"""The tools against the seeded database: the numbers match the planted fixtures, the
checks happen in the tool, duplicates return the same action, and every call is
audited and recorded as evidence.
"""

from __future__ import annotations

import json
from datetime import date

import pytest
from tests.tool_args import sample_args

from support_agent import audit, util
from support_agent.request_context import bind_request_context
from support_agent.tools import (
    TOOLS,
    Tool,
    base,
    create_ticket,
    get_known_outages,
    get_leave_balance,
    search_policies,
    submit_leave_request,
)
from support_agent.tools.base import record_tool_call
from support_agent.tools.hr_it import calculate_working_days
from support_agent.tools.policy_search import fuse


def _as(employee_id: str, fn: Tool, *args, **kwargs) -> dict:
    with bind_request_context(employee_id, f"req-{employee_id}", "rest"):
        return json.loads(fn(*args, **kwargs))


def _count(conn, table: str) -> int:
    return conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]


def test_a_recorded_tool_call_has_the_time_since_it_started(monkeypatch):
    events = []
    monkeypatch.setattr(audit, "record", lambda event, payload, **kwargs: events.append(payload))
    monkeypatch.setattr(base.time, "perf_counter", lambda: 12.5)

    with bind_request_context("emp_001", "req-latency", "rest") as ctx:
        record_tool_call("some_tool", {}, "done", None, started_at=12.0)

    assert ctx.evidence[0]["latency_ms"] == 500
    assert events[0]["latency_ms"] == 500


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


def test_working_days_leave_out_both_christmas_days():
    # in 2025 Dec 25 is a Thursday and Dec 26 a Friday: five days Monday to Friday, two holidays
    assert calculate_working_days(date(2025, 12, 22), date(2025, 12, 26)) == 3
    assert calculate_working_days(date(2025, 12, 25), date(2025, 12, 26)) == 0


def test_a_public_holiday_on_a_weekend_does_not_remove_a_weekday():
    # Dec 25 2026 is a Friday and Dec 26 a Saturday: Monday to Thursday, then Monday again
    assert calculate_working_days(date(2026, 12, 21), date(2026, 12, 28)) == 5


def test_a_state_holiday_still_counts_as_a_working_day():
    # Epiphany (Jan 6) is a holiday in some states only
    assert calculate_working_days(date(2027, 1, 6), date(2027, 1, 6)) == 1


def test_the_tool_counts_the_working_days_itself(seeded_db):
    result = _as("emp_001", submit_leave_request, date(2026, 11, 2), date(2026, 11, 13))

    assert result["proposed"] is True
    assert result["working_days"] == 10
    payload = seeded_db.execute("SELECT payload FROM pending_actions").fetchone()[0]
    assert payload["working_days"] == 10


def test_the_tool_leaves_out_the_public_holidays(seeded_db):
    # Dec 25 and Dec 26 2028 are a Monday and a Tuesday: Friday to Wednesday has 4 weekdays
    seeded_db.execute(
        "INSERT INTO leave_balances (employee_id, year, entitlement_days)"
        " VALUES ('emp_001', 2028, 30)"
    )
    result = _as("emp_001", submit_leave_request, date(2028, 12, 22), date(2028, 12, 27))

    assert result["proposed"] is True
    assert result["working_days"] == 2
    payload = seeded_db.execute("SELECT payload FROM pending_actions").fetchone()[0]
    assert payload["working_days"] == 2


def test_a_range_with_only_a_public_holiday_is_refused(seeded_db):
    result = _as("emp_001", submit_leave_request, date(2028, 12, 25), date(2028, 12, 26))

    assert result == {
        "proposed": False,
        "reason": "the range has no working days; it covers only weekends and public holidays",
    }
    assert _count(seeded_db, "pending_actions") == 0


def test_a_range_over_a_weekend_does_not_count_it(seeded_db):
    result = _as("emp_001", submit_leave_request, date(2026, 11, 6), date(2026, 11, 9))

    assert result["working_days"] == 2


def test_a_request_can_start_today(seeded_db):
    # the fixed test date is Tuesday 2026-09-22
    result = _as("emp_001", submit_leave_request, util.get_today(), util.get_today())

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
            "the range has no working days; it covers only weekends and public holidays",
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
    # emp_004 has 3.5 days left: each request fits alone, but the two together do not
    first = _as("emp_004", submit_leave_request, date(2026, 11, 2), date(2026, 11, 3))
    second = _as("emp_004", submit_leave_request, date(2026, 11, 9), date(2026, 11, 10))

    assert first["proposed"] is True
    assert second["proposed"] is False
    assert second["requested_days"] == 2
    assert second["awaiting_approval_days"] == 2.0
    assert second["remaining_days"] == 1.5
    assert _count(seeded_db, "pending_actions") == 1


def test_a_request_that_fits_next_to_the_waiting_ones_is_proposed(seeded_db):
    _as("emp_004", submit_leave_request, date(2026, 11, 2), date(2026, 11, 3))

    second = _as("emp_004", submit_leave_request, date(2026, 11, 9), date(2026, 11, 9))

    assert second["proposed"] is True
    assert second["remaining_after_approval"] == 0.5


def test_leave_balance_without_waiting_requests(seeded_db):
    balance = _as("emp_004", get_leave_balance)

    assert balance["awaiting_approval_days"] == 0.0
    assert balance["remaining_days"] == 3.5
    assert balance["remaining_if_all_approved"] == 3.5


def test_leave_balance_shows_the_days_that_wait_for_approval(seeded_db):
    _as("emp_004", submit_leave_request, date(2026, 11, 2), date(2026, 11, 3))

    balance = _as("emp_004", get_leave_balance)

    assert balance["awaiting_approval_days"] == 2.0
    # the stored numbers do not change until an approver approves
    assert "approved_days" not in balance
    assert balance["remaining_days"] == 3.5
    assert balance["remaining_if_all_approved"] == 1.5


def test_leave_balance_has_no_approved_days_when_there_are_none(seeded_db):
    balance = _as("emp_004", get_leave_balance)

    assert "approved_days" not in balance


def test_leave_balance_shows_the_approved_days_when_there_are_some(seeded_db):
    seeded_db.execute(
        "UPDATE leave_balances SET approved_days = 2 WHERE employee_id = 'emp_004' AND year = 2026"
    )

    balance = _as("emp_004", get_leave_balance)

    assert balance["approved_days"] == 2.0
    # the approved days are already out of the remaining days
    assert balance["remaining_days"] == 1.5


def test_leave_balance_counts_only_the_own_requests_that_still_wait(seeded_db):
    _as("emp_004", submit_leave_request, date(2026, 11, 2), date(2026, 11, 3))
    _as("emp_001", submit_leave_request, date(2026, 11, 2), date(2026, 11, 6))

    # the five days of emp_001 are not counted for emp_004
    assert _as("emp_004", get_leave_balance)["awaiting_approval_days"] == 2.0
    assert _as("emp_001", get_leave_balance)["awaiting_approval_days"] == 5.0

    seeded_db.execute(
        "UPDATE pending_actions SET status = 'rejected' WHERE employee_id = 'emp_001'"
    )

    # a rejected request does not wait any more
    assert _as("emp_001", get_leave_balance)["awaiting_approval_days"] == 0.0


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


def test_search_evidence_has_the_id_and_policy_version_of_each_chunk(seeded_db, search_embeds_like):
    search_embeds_like("vacation-policy#entitlement#0")
    with bind_request_context("emp_001", "req-search", "rest") as ctx:
        search_policies("vacation days and the VPN")

    chunks = ctx.evidence[0]["ref"]
    stored = dict(seeded_db.execute("SELECT id, policy_version FROM policy_chunks").fetchall())
    assert [chunk["id"] for chunk in chunks][0] == "vacation-policy#entitlement#0"
    assert all(set(chunk) == {"id", "policy_version"} for chunk in chunks)
    assert {chunk["id"]: chunk["policy_version"] for chunk in chunks} == {
        chunk["id"]: stored[chunk["id"]] for chunk in chunks
    }
    # chunks of two policies with different versions
    assert {chunk["policy_version"] for chunk in chunks} == {"2026-09.1", "2026-11.1"}


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
        "SELECT request_id, payload->>'tool', (payload->>'latency_ms')::int"
        " FROM audit_log WHERE event = 'tool_call'"
    ).fetchall()
    assert [(r[0], r[1]) for r in rows] == [("req-audit", fn.__name__)]
    assert [(e["type"], e["tool"]) for e in ctx.evidence] == [("tool_result", fn.__name__)]
    # the evidence and the audit event have the same time, and it is not negative
    assert ctx.evidence[0]["latency_ms"] == rows[0][2] >= 0


def test_leave_balance_year_defaults_to_the_current_year(seeded_db, monkeypatch):
    # the seed has the year of the test date (2026) and the year after
    assert _as("emp_001", get_leave_balance)["year"] == 2026

    monkeypatch.setattr(util, "get_today", lambda: date(2027, 1, 4))
    assert _as("emp_001", get_leave_balance)["year"] == 2027
    assert _as("emp_001", get_leave_balance)["remaining_days"] == 18.0

    monkeypatch.setattr(util, "get_today", lambda: date(2025, 3, 1))
    assert "error" in _as("emp_001", get_leave_balance)
