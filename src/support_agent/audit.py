"""Append-only audit trail. Every write takes its request_id, and any employee or
approver id in the actor, from the bound request context — never from an argument.
"""

from __future__ import annotations

from typing import Any, Literal

import psycopg
from psycopg.types.json import Jsonb

from support_agent import db
from support_agent.request_context import get_request_context

Event = Literal[
    "tool_call",
    "action_proposed",
    "action_approved",
    "action_rejected",
    "action_executed",
    "decision_recorded",
    "feedback_received",
    "model_call",
]
Actor = Literal["agent", "employee", "approver", "system"]

_INSERT = "INSERT INTO audit_log (request_id, actor, event, payload) VALUES (%s, %s, %s, %s)"


def record(
    event: Event,
    payload: dict[str, Any],
    actor: Actor | None = None,
    conn: psycopg.Connection | psycopg.Cursor | None = None,
) -> None:
    """Write one audit event. `actor` defaults to the agent; `employee` and
    `approver` resolve to `employee:<id>` / `approver:<id>` for the bound employee.
    Pass `conn` to write inside the caller's transaction.
    """
    ctx = get_request_context()
    role = actor or "agent"
    actor_label = f"{role}:{ctx.employee_id}" if role in ("employee", "approver") else role
    params = (ctx.request_id, actor_label, event, Jsonb(payload))
    if conn is None:
        db.execute(_INSERT, params)
    else:
        conn.execute(_INSERT, params)
