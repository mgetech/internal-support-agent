"""Who may decide on a pending action. The role check comes first, then the check that
the approver is not the requester.
"""

from __future__ import annotations

import pytest

from support_agent.guardrails.approval import decide, validate_approver


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

    with pytest.raises(PermissionError, match="hr_partner or a manager"):
        decide(action_id, "emp_002", approve=True)

    assert _status(seeded_db, action_id) == ("pending_approval", None)


def test_unknown_approver_is_rejected(seeded_db):
    action_id = _propose(seeded_db, "emp_001")

    with pytest.raises(PermissionError, match="hr_partner or a manager"):
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

    with pytest.raises(PermissionError, match="hr_partner or a manager"):
        decide(action_id, "emp_001", approve=True)


def test_unknown_action_is_rejected(seeded_db):
    with pytest.raises(LookupError, match="no pending action 999"):
        validate_approver(999, "emp_005")


def test_unauthorized_approver_learns_nothing_about_the_action(seeded_db):
    # the role check runs before the action is read, so a missing action looks the same
    # as an existing one to someone who may not approve
    with pytest.raises(PermissionError):
        validate_approver(999, "emp_002")
