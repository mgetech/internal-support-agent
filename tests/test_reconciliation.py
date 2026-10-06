"""The check for proposals left without a Decision Record. The tests write to Postgres, so
they need `make db-up`. They are skipped when no database is reachable.

Each test makes a pending action by hand, with the age it needs, and runs the check. The
seed data gives the employee emp_004.
"""

import logging

from psycopg.types.json import Jsonb

from support_agent.reconciliation import flag_orphaned_actions

GRACE_MINUTES = 10


def add_action(db, request_id="req-1", age_minutes=30):
    """Add a pending action made `age_minutes` ago. Returns its id."""
    row = db.execute(
        "INSERT INTO pending_actions (request_id, employee_id, tool, payload, created_at)"
        " VALUES (%s, 'emp_004', 'create_ticket', %s, now() - make_interval(mins => %s))"
        " RETURNING id",
        (request_id, Jsonb({"title": "VPN"}), age_minutes),
    ).fetchone()
    return row[0]


def add_decision_record(db, request_id):
    db.execute(
        "INSERT INTO decision_records (request_id, employee_id, channel, outcome, summary,"
        " evidence) VALUES (%s, 'emp_004', 'rest', 'propose_action', 'proposed', '[]')",
        (request_id,),
    )


def get_flags(db):
    """The `orphan_flagged` events as (request_id, actor, payload), oldest first."""
    return db.execute(
        "SELECT request_id, actor, payload FROM audit_log WHERE event = 'orphan_flagged'"
        " ORDER BY id"
    ).fetchall()


# INTEGRATION TESTS: the tests below use the real Postgres (make db-up).
def test_an_old_action_without_a_record_is_flagged_once(seeded_db):
    action_id = add_action(seeded_db, "req-1", age_minutes=11)

    flagged = flag_orphaned_actions(GRACE_MINUTES)

    assert flagged == 1
    assert get_flags(seeded_db) == [
        ("req-1", "system", {"action_id": action_id, "tool": "create_ticket"})
    ]


def test_the_flagged_action_keeps_its_status(seeded_db):
    add_action(seeded_db)

    flag_orphaned_actions(GRACE_MINUTES)

    (status,) = seeded_db.execute("SELECT status FROM pending_actions").fetchone()
    assert status == "pending_approval"


def test_an_action_inside_the_grace_period_is_not_flagged(seeded_db):
    add_action(seeded_db, age_minutes=5)

    flagged = flag_orphaned_actions(GRACE_MINUTES)

    assert flagged == 0
    assert get_flags(seeded_db) == []


def test_an_action_whose_request_has_a_record_is_not_flagged(seeded_db):
    add_action(seeded_db, "req-1")
    add_decision_record(seeded_db, "req-1")

    flagged = flag_orphaned_actions(GRACE_MINUTES)

    assert flagged == 0
    assert get_flags(seeded_db) == []


def test_a_second_check_adds_no_new_event(seeded_db):
    add_action(seeded_db)
    flag_orphaned_actions(GRACE_MINUTES)

    flagged_again = flag_orphaned_actions(GRACE_MINUTES)

    assert flagged_again == 0
    assert len(get_flags(seeded_db)) == 1


def test_each_action_of_one_request_is_flagged(seeded_db):
    first = add_action(seeded_db, "req-1")
    second = add_action(seeded_db, "req-1")

    flagged = flag_orphaned_actions(GRACE_MINUTES)

    assert flagged == 2
    assert [flag[2]["action_id"] for flag in get_flags(seeded_db)] == [first, second]


def test_an_action_that_comes_later_is_flagged_by_the_next_check(seeded_db):
    add_action(seeded_db, "req-1")
    flag_orphaned_actions(GRACE_MINUTES)
    add_action(seeded_db, "req-2")

    flagged = flag_orphaned_actions(GRACE_MINUTES)

    assert flagged == 1
    assert [flag[0] for flag in get_flags(seeded_db)] == ["req-1", "req-2"]


def test_the_number_flagged_is_logged(seeded_db, caplog):
    add_action(seeded_db)

    with caplog.at_level(logging.INFO, logger="support_agent.reconciliation"):
        flag_orphaned_actions(GRACE_MINUTES)

    assert "flagged 1 pending actions" in caplog.text
