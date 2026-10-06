"""Approval of gated writes. Only the service layer calls this, never the agent. A
proposal in `pending_actions` changes nothing until an approver with the right role
decides on it.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from typing import Any

import psycopg

from support_agent import audit, db
from support_agent.request_context import bind_request_context

APPROVER_ROLES = ("hr_partner", "manager")


class AlreadyDecidedError(Exception):
    """Raised when an action was approved, rejected or expired before this decision."""


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


def _insert_leave_request(cur: psycopg.Cursor, action: dict[str, Any]) -> dict[str, Any]:
    """Add the approved leave request and count its days as pending in the balance."""
    payload = action["payload"]
    start_date = date.fromisoformat(payload["start_date"])
    days = payload["working_days"]
    row = cur.execute(
        """
        INSERT INTO leave_requests (employee_id, start_date, end_date, days, status)
        VALUES (%s, %s, %s, %s, 'approved')
        RETURNING id
        """,
        (action["employee_id"], start_date, payload["end_date"], days),
    ).fetchone()
    updated = cur.execute(
        """
        UPDATE leave_balances
        SET pending_days = pending_days + %s
        WHERE employee_id = %s AND year = %s
        """,
        (days, action["employee_id"], start_date.year),
    )
    if updated.rowcount != 1:
        # raising rolls back the insert and the approval
        raise LookupError(f"no leave balance on record for {start_date.year}")
    return {"table": "leave_requests", "id": row["id"]}


def _insert_ticket(cur: psycopg.Cursor, action: dict[str, Any]) -> dict[str, Any]:
    """Open the approved ticket for the employee who asked for it."""
    payload = action["payload"]
    row = cur.execute(
        """
        INSERT INTO tickets (employee_id, category, title, body, created_by)
        VALUES (%s, %s, %s, %s, %s)
        RETURNING id
        """,
        (
            action["employee_id"],
            payload["category"],
            payload["title"],
            payload["body"],
            action["employee_id"],
        ),
    ).fetchone()
    return {"table": "tickets", "id": row["id"]}


_EXECUTORS: dict[str, Callable[[psycopg.Cursor, dict[str, Any]], dict[str, Any]]] = {
    "submit_leave_request": _insert_leave_request,
    "create_ticket": _insert_ticket,
}


def decide(action_id: int, approver_id: str, approve: bool) -> str:
    """Approve or reject a pending action. Returns the new status: "executed" or
    "rejected".

    The approver is checked before anything is locked. Then the action row is locked, so
    two decisions on the same action run one after the other, and the second raises
    AlreadyDecidedError. An approval makes the write and marks the action executed. A
    rejection writes nothing. Every transition is audited under the request that made the
    proposal, with the approver as the actor. If the write fails, nothing is saved and
    the action stays pending.
    """
    validate_approver(action_id, approver_id)

    with db.transaction() as cur:
        action = cur.execute(
            """
            SELECT request_id, employee_id, tool, payload, status
            FROM pending_actions
            WHERE id = %s
            FOR UPDATE
            """,
            (action_id,),
        ).fetchone()
        if action["status"] != "pending_approval":
            raise AlreadyDecidedError(f"action {action_id} is already {action['status']}")

        # the audit log takes the request id and the approver from the context. The
        # channel is not stored with the proposal, and the audit log does not use it.
        with bind_request_context(approver_id, action["request_id"], "rest"):
            event = {"action_id": action_id, "tool": action["tool"]}
            if not approve:
                cur.execute(
                    """
                    UPDATE pending_actions
                    SET status = 'rejected', decided_by = %s, decided_at = now()
                    WHERE id = %s
                    """,
                    (approver_id, action_id),
                )
                audit.record("action_rejected", event, actor="approver", conn=cur)
                return "rejected"

            cur.execute(
                """
                UPDATE pending_actions
                SET status = 'approved', decided_by = %s, decided_at = now()
                WHERE id = %s
                """,
                (approver_id, action_id),
            )
            audit.record("action_approved", event, actor="approver", conn=cur)

            written = _EXECUTORS[action["tool"]](cur, action)
            cur.execute(
                "UPDATE pending_actions SET status = 'executed' WHERE id = %s", (action_id,)
            )
            audit.record(
                "action_executed", {**event, "written": written}, actor="approver", conn=cur
            )
            return "executed"
