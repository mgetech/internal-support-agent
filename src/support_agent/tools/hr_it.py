"""HR and IT tools. Each one reads the employee from the bound request context, so the
model can only ever see or act on the requester's own records.
"""

from __future__ import annotations

import json

from support_agent import db
from support_agent.request_context import get_request_context


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
