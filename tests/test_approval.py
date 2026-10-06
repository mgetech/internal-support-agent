"""Who may decide on a pending action, and what a decision does. The role check comes
first, then the check that the approver is not the requester. An approval makes the
write; a rejection writes nothing; a second decision is refused.
"""

from __future__ import annotations

import threading

import pytest

from support_agent import db
from support_agent.guardrails.approval import AlreadyDecidedError, decide, validate_approver


def _propose(conn, employee_id: str) -> int:
    """Insert a pending leave request for the employee and return its id."""
    return conn.execute(
        """
        INSERT INTO pending_actions (request_id, employee_id, tool, payload)
        VALUES ('req-approval', %s, 'submit_leave_request',
                '{"start_date": "2026-11-02", "end_date": "2026-11-02", "working_days": 1}')
        RETURNING id
        """,
        (employee_id,),
    ).fetchone()[0]


def _propose_ticket(conn, employee_id: str) -> int:
    """Insert a pending ticket for the employee and return its id."""
    return conn.execute(
        """
        INSERT INTO pending_actions (request_id, employee_id, tool, payload)
        VALUES ('req-approval', %s, 'create_ticket',
                '{"category": "access", "title": "VPN access", "body": "I need the VPN."}')
        RETURNING id
        """,
        (employee_id,),
    ).fetchone()[0]


def _pending_days(conn, employee_id: str) -> float:
    return float(
        conn.execute(
            "SELECT pending_days FROM leave_balances WHERE employee_id = %s AND year = 2026",
            (employee_id,),
        ).fetchone()[0]
    )


def _count(conn, table: str) -> int:
    return conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]


def _status(conn, action_id: int) -> tuple[str, str | None]:
    return conn.execute(
        "SELECT status, decided_by FROM pending_actions WHERE id = %s", (action_id,)
    ).fetchone()


# INTEGRATION TEST: this test uses the real Postgres (make db-up).
@pytest.mark.parametrize("approver_id", ["emp_005", "emp_010"], ids=["hr_partner", "manager"])
def test_approver_roles_are_allowed(approver_id, seeded_db):
    action_id = _propose(seeded_db, "emp_001")

    validate_approver(action_id, approver_id)


def test_employee_role_is_rejected(seeded_db):
    action_id = _propose(seeded_db, "emp_001")

    with pytest.raises(PermissionError, match="not allowed to approve"):
        decide(action_id, "emp_002", approve=True)

    assert _status(seeded_db, action_id) == ("pending_approval", None)


def test_unknown_approver_is_rejected(seeded_db):
    action_id = _propose(seeded_db, "emp_001")

    with pytest.raises(PermissionError, match="not allowed to approve"):
        decide(action_id, "emp_999", approve=True)


@pytest.mark.parametrize("approver_id", ["emp_005", "emp_010"], ids=["hr_partner", "manager"])
def test_self_approval_is_rejected(approver_id, seeded_db):
    action_id = _propose(seeded_db, approver_id)

    with pytest.raises(PermissionError, match="own request"):
        decide(action_id, approver_id, approve=True)

    assert _status(seeded_db, action_id) == ("pending_approval", None)


def test_role_is_checked_before_self_approval(seeded_db):
    # emp_001 is the requester and also has the wrong role: the role error comes first
    action_id = _propose(seeded_db, "emp_001")

    with pytest.raises(PermissionError, match="not allowed to approve"):
        decide(action_id, "emp_001", approve=True)


def test_unknown_action_is_rejected(seeded_db):
    with pytest.raises(LookupError, match="no pending action 999"):
        validate_approver(999, "emp_005")


def test_unauthorized_approver_learns_nothing_about_the_action(seeded_db):
    # the role check runs before the action is read, so a missing action looks the same
    # as an existing one to someone who may not approve
    with pytest.raises(PermissionError):
        validate_approver(999, "emp_002")


def test_approved_leave_request_is_written(seeded_db):
    action_id = _propose(seeded_db, "emp_001")
    pending_before = _pending_days(seeded_db, "emp_001")

    assert decide(action_id, "emp_005", approve=True) == "executed"

    assert _status(seeded_db, action_id) == ("executed", "emp_005")
    rows = seeded_db.execute(
        "SELECT employee_id, start_date, end_date, days, status FROM leave_requests"
    ).fetchall()
    assert [(r[0], str(r[1]), str(r[2]), float(r[3]), r[4]) for r in rows] == [
        ("emp_001", "2026-11-02", "2026-11-02", 1.0, "approved")
    ]
    assert _pending_days(seeded_db, "emp_001") == pending_before + 1


def test_approved_ticket_is_written(seeded_db):
    action_id = _propose_ticket(seeded_db, "emp_001")

    assert decide(action_id, "emp_010", approve=True) == "executed"

    assert _status(seeded_db, action_id) == ("executed", "emp_010")
    rows = seeded_db.execute(
        "SELECT employee_id, category, title, body, status, created_by FROM tickets"
    ).fetchall()
    assert rows == [("emp_001", "access", "VPN access", "I need the VPN.", "open", "emp_001")]


def test_rejected_action_writes_nothing(seeded_db):
    action_id = _propose(seeded_db, "emp_001")
    pending_before = _pending_days(seeded_db, "emp_001")

    assert decide(action_id, "emp_005", approve=False) == "rejected"

    assert _status(seeded_db, action_id) == ("rejected", "emp_005")
    assert _count(seeded_db, "leave_requests") == 0
    assert _count(seeded_db, "tickets") == 0
    assert _pending_days(seeded_db, "emp_001") == pending_before


@pytest.mark.parametrize(
    ("first", "second"),
    [(True, True), (True, False), (False, True), (False, False)],
    ids=["approve-approve", "approve-reject", "reject-approve", "reject-reject"],
)
def test_second_decision_is_rejected(first, second, seeded_db):
    action_id = _propose(seeded_db, "emp_001")
    decide(action_id, "emp_005", approve=first)
    leave_requests = _count(seeded_db, "leave_requests")
    pending_days = _pending_days(seeded_db, "emp_001")
    status = _status(seeded_db, action_id)

    with pytest.raises(AlreadyDecidedError, match=f"already {status[0]}"):
        decide(action_id, "emp_010", approve=second)

    assert _count(seeded_db, "leave_requests") == leave_requests
    assert _pending_days(seeded_db, "emp_001") == pending_days
    assert _status(seeded_db, action_id) == status


def test_two_decisions_at_the_same_time_run_one_after_the_other(seeded_db):
    action_id = _propose(seeded_db, "emp_001")
    pending_before = _pending_days(seeded_db, "emp_001")
    barrier = threading.Barrier(2)
    results: list[str] = []

    def run(approver_id: str) -> None:
        barrier.wait()
        try:
            results.append(decide(action_id, approver_id, approve=True))
        except AlreadyDecidedError:
            results.append("already_decided")

    threads = [threading.Thread(target=run, args=(a,)) for a in ("emp_005", "emp_010")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert sorted(results) == ["already_decided", "executed"]
    assert _count(seeded_db, "leave_requests") == 1
    assert _pending_days(seeded_db, "emp_001") == pending_before + 1


def test_checks_run_before_the_row_is_locked(seeded_db):
    action_id = _propose(seeded_db, "emp_001")
    errors: list[PermissionError] = []

    def run() -> None:
        try:
            decide(action_id, "emp_002", approve=True)
        except PermissionError as error:
            errors.append(error)

    # another connection holds the row lock; a decision that tried to lock would wait
    with db.get_pool().connection() as holder:
        holder.execute("SELECT 1 FROM pending_actions WHERE id = %s FOR UPDATE", (action_id,))
        thread = threading.Thread(target=run)
        thread.start()
        thread.join(timeout=5)
        finished = not thread.is_alive()

    thread.join()
    assert finished
    assert len(errors) == 1


def test_failed_write_leaves_the_action_pending(seeded_db):
    action_id = _propose(seeded_db, "emp_001")
    seeded_db.execute("DELETE FROM leave_balances WHERE employee_id = 'emp_001'")

    with pytest.raises(LookupError, match="no leave balance"):
        decide(action_id, "emp_005", approve=True)

    assert _status(seeded_db, action_id) == ("pending_approval", None)
    assert _count(seeded_db, "leave_requests") == 0
    assert _count(seeded_db, "audit_log") == 0


def test_every_transition_is_audited_under_the_proposal_request(seeded_db):
    action_id = _propose(seeded_db, "emp_001")

    decide(action_id, "emp_005", approve=True)

    events = seeded_db.execute(
        "SELECT request_id, actor, event, payload FROM audit_log ORDER BY id"
    ).fetchall()
    assert [(r[0], r[1], r[2]) for r in events] == [
        ("req-approval", "approver:emp_005", "action_approved"),
        ("req-approval", "approver:emp_005", "action_executed"),
    ]
    assert events[0][3] == {"action_id": action_id, "tool": "submit_leave_request"}
    assert events[1][3]["written"]["table"] == "leave_requests"


def test_rejection_is_audited(seeded_db):
    action_id = _propose(seeded_db, "emp_001")

    decide(action_id, "emp_010", approve=False)

    events = seeded_db.execute("SELECT actor, event FROM audit_log").fetchall()
    assert events == [("approver:emp_010", "action_rejected")]
