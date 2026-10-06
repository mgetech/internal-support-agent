"""Approval of gated writes. Only the service layer calls this, never the agent. A
proposal in `pending_actions` changes nothing until an approver with the right role
decides on it.
"""

from __future__ import annotations

from support_agent import db

APPROVER_ROLES = ("hr_partner", "manager")


def validate_approver(action_id: int, approver_id: str) -> None:
    """Raise if the approver may not decide on this action. The role is checked first.
    The checks only read: no row is locked.

    Raises PermissionError if the approver does not have an approver role, or is the
    employee who made the request. Raises LookupError if the action does not exist.
    """
    approver = db.fetch_one("SELECT role FROM employees WHERE id = %s", (approver_id,))
    if approver is None or approver["role"] not in APPROVER_ROLES:
        raise PermissionError("you are not allowed to approve an action")

    action = db.fetch_one("SELECT employee_id FROM pending_actions WHERE id = %s", (action_id,))
    if action is None:
        raise LookupError(f"no pending action {action_id}")
    if action["employee_id"] == approver_id:
        raise PermissionError("an employee cannot approve their own request")


def decide(action_id: int, approver_id: str, approve: bool) -> None:
    """Approve or reject a pending action. The approver is checked before anything else."""
    validate_approver(action_id, approver_id)
    raise NotImplementedError("deciding on an action is not built yet")
