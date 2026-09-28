"""HR and IT tools. Each one reads the employee from the bound request context, so the
model can only ever see or act on the requester's own records.
"""

from __future__ import annotations

import json
from datetime import date

from psycopg.types.json import Jsonb

from support_agent import db
from support_agent.request_context import get_request_context
from support_agent.tools.base import gated_write


def get_leave_balance(year: int = 2026) -> str:
    """Get the requester's vacation balance for a year: entitlement, days taken, days
    pending approval and days remaining. Use it for any question about how much leave
    they have left.
    """
    ctx = get_request_context()
    row = db.fetch_one(
        """
        SELECT entitlement_days, taken_days, pending_days
        FROM leave_balances
        WHERE employee_id = %s AND year = %s
        """,
        (ctx.employee_id, year),
    )
    if row is None:
        return json.dumps({"year": year, "error": "no leave balance on record for this year"})

    entitlement = float(row["entitlement_days"])
    taken = float(row["taken_days"])
    pending = float(row["pending_days"])
    return json.dumps(
        {
            "year": year,
            "entitlement_days": entitlement,
            "taken_days": taken,
            "pending_days": pending,
            "remaining_days": entitlement - taken - pending,
        }
    )


def get_known_outages() -> str:
    """List IT outages that are not resolved yet, with the affected system, status and a
    short note. Check this before suggesting a ticket for something that is not working.
    """
    # no per-employee data here, but every tool still refuses to run outside a request
    get_request_context()
    rows = db.fetch_all(
        """
        SELECT system, status, started_at, note
        FROM known_outages
        WHERE status <> 'resolved'
        ORDER BY started_at DESC
        """
    )
    return json.dumps(
        {
            "outages": [
                {
                    "system": r["system"],
                    "status": r["status"],
                    "started_at": r["started_at"].isoformat(),
                    "note": r["note"],
                }
                for r in rows
            ]
        }
    )


def _not_proposed(reason: str, **details: float) -> str:
    return json.dumps({"proposed": False, "reason": reason, **details})


@gated_write
def submit_leave_request(start_date: date, end_date: date, working_days: float) -> str:
    """Send the requester's vacation request to an approver. This does not book the
    vacation. An approver must approve it first.

    working_days: the number of days off, without weekends. Example: Monday to Friday
    is 5 working days.

    The result has "proposed": true if the request was sent. If it has "proposed":
    false, the request was not sent. Tell the requester the "reason" from the result.
    """
    ctx = get_request_context()

    if end_date < start_date:
        return _not_proposed("the end date is before the start date")
    if end_date.year != start_date.year:
        return _not_proposed("a request cannot span two years; split it at the new year")
    calendar_days = (end_date - start_date).days + 1
    if not 0 < working_days <= calendar_days:
        return _not_proposed(
            "working days must be more than 0 and no more than the days in the range",
            calendar_days=calendar_days,
        )

    year = start_date.year
    with db.transaction() as cur:
        # lock this employee's balance row, so two requests sent at the same time
        # can't both pass the check
        balance = cur.execute(
            """
            SELECT entitlement_days, taken_days, pending_days
            FROM leave_balances
            WHERE employee_id = %s AND year = %s
            FOR UPDATE
            """,
            (ctx.employee_id, year),
        ).fetchone()
        if balance is None:
            return _not_proposed(f"no leave balance on record for {year}")

        # also subtract the days of requests that are still waiting for approval,
        # because pending_days only includes approved requests
        awaiting = cur.execute(
            """
            SELECT coalesce(sum((payload->>'working_days')::numeric), 0) AS days
            FROM pending_actions
            WHERE employee_id = %s AND tool = 'submit_leave_request'
              AND status = 'pending_approval'
              AND extract(year FROM (payload->>'start_date')::date) = %s
            """,
            (ctx.employee_id, year),
        ).fetchone()

        remaining = float(
            balance["entitlement_days"]
            - balance["taken_days"]
            - balance["pending_days"]
            - awaiting["days"]
        )
        if working_days > remaining:
            return _not_proposed(
                "the request is more than the remaining balance",
                requested_days=working_days,
                remaining_days=remaining,
            )

        payload = {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "working_days": working_days,
        }
        action = cur.execute(
            """
            INSERT INTO pending_actions (request_id, employee_id, tool, payload)
            VALUES (%s, %s, 'submit_leave_request', %s)
            RETURNING id
            """,
            (ctx.request_id, ctx.employee_id, Jsonb(payload)),
        ).fetchone()

    return json.dumps(
        {
            "proposed": True,
            "action_id": action["id"],
            "status": "pending_approval",
            "working_days": working_days,
            "remaining_after_approval": remaining - working_days,
        }
    )
