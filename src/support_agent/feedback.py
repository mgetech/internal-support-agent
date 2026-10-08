"""Feedback on an answer: a rating and an optional comment, linked to the Decision
Record of the request. Only the employee who made the request can rate it.
"""

from __future__ import annotations

from typing import Literal

from support_agent import audit, db
from support_agent.request_context import bind_request_context

Rating = Literal["up", "down"]


def create_feedback(request_id: str, employee_id: str, rating: Rating, comment: str | None) -> int:
    """Save the feedback and its `feedback_received` audit event in one transaction.
    Returns the feedback id. Raises LookupError if the employee has no Decision Record
    with this request id.
    """
    with db.transaction() as cur:
        record = cur.execute(
            "SELECT 1 FROM decision_records WHERE request_id = %s AND employee_id = %s",
            (request_id, employee_id),
        ).fetchone()
        if record is None:
            raise LookupError(f"no decision record {request_id}")
        row = cur.execute(
            """
            INSERT INTO feedback (request_id, employee_id, rating, comment)
            VALUES (%s, %s, %s, %s)
            RETURNING id
            """,
            (request_id, employee_id, rating, comment),
        ).fetchone()
        # the audit log takes the request id and the actor from the context. The channel
        # is not used by the audit log.
        with bind_request_context(employee_id, request_id, "rest"):
            audit.record(
                "feedback_received",
                {"feedback_id": row["id"], "rating": rating},
                actor="employee",
                conn=cur,
            )
    return row["id"]
