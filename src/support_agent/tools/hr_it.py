"""HR and IT tools. Each one reads the employee from the bound request context, so the
model can only ever see or act on the requester's own records.
"""

from __future__ import annotations

import json
import time
from datetime import date, timedelta
from typing import Any, Literal

import psycopg
from psycopg.types.json import Jsonb

from support_agent import audit, db, util
from support_agent.request_context import get_request_context
from support_agent.tools.base import gated_write, record_tool_call


def _read_leave_balance(cur: psycopg.Cursor, employee_id: str, year: int) -> dict[str, Any] | None:
    """The employee's leave balance for a year, or None if there is none. Besides the
    stored numbers it has `awaiting_approval_days`: the days of the employee's leave
    requests that still wait for approval. The stored `approved_days` only has the days
    that were already approved.
    """
    return cur.execute(
        """
        SELECT b.entitlement_days, b.taken_days, b.approved_days,
               (
                   SELECT coalesce(sum((a.payload->>'working_days')::numeric), 0)
                   FROM pending_actions AS a
                   WHERE a.employee_id = b.employee_id AND a.tool = 'submit_leave_request'
                     AND a.status = 'pending_approval'
                     AND extract(year FROM (a.payload->>'start_date')::date) = b.year
               ) AS awaiting_approval_days
        FROM leave_balances AS b
        WHERE b.employee_id = %s AND b.year = %s
        """,
        (employee_id, year),
    ).fetchone()


def get_leave_balance(year: int = 2026) -> str:
    """Get the requester's vacation balance for a year: entitlement, days taken, days
    approved but not taken yet ("approved_days"), days that still wait for approval
    ("awaiting_approval_days") and days remaining.

    "approved_days" is left out when it is 0. "remaining_days" already has the approved
    days taken out, but not the days that still wait for approval.
    "remaining_if_all_approved" takes those out too. When "awaiting_approval_days" is more
    than 0, tell the requester about both numbers. Use this for any question about how
    much leave they have left.
    """
    started_at = time.perf_counter()
    ctx = get_request_context()
    with db.transaction() as cur:
        row = _read_leave_balance(cur, ctx.employee_id, year)
    ref = f"leave_balances:{ctx.employee_id}:{year}"
    if row is None:
        record_tool_call(
            "get_leave_balance", {"year": year}, f"no balance for {year}", None, started_at
        )
        return json.dumps({"year": year, "error": "no leave balance on record for this year"})

    entitlement = float(row["entitlement_days"])
    taken = float(row["taken_days"])
    approved = float(row["approved_days"])
    awaiting = float(row["awaiting_approval_days"])
    remaining = entitlement - taken - approved
    remaining_if_all_approved = remaining - awaiting
    record_tool_call(
        "get_leave_balance",
        {"year": year},
        f"{remaining} of {entitlement} days left in {year}, "
        f"{awaiting} awaiting approval, {remaining_if_all_approved} left if all are approved",
        ref,
        started_at,
    )
    result = {
        "year": year,
        "entitlement_days": entitlement,
        "taken_days": taken,
        "awaiting_approval_days": awaiting,
        "remaining_days": remaining,
        "remaining_if_all_approved": remaining_if_all_approved,
    }
    # left out when it is 0, so the model has one number less to explain
    if approved:
        result["approved_days"] = approved
    return json.dumps(result)


def get_known_outages() -> str:
    """List IT outages that are not resolved yet, with the affected system, status and a
    short note. Check this before suggesting a ticket for something that is not working.
    """
    started_at = time.perf_counter()
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
    systems = [r["system"] for r in rows]
    summary = f"open outages: {', '.join(systems)}" if systems else "no open outages"
    record_tool_call("get_known_outages", {}, summary, "known_outages", started_at)
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


def _lock_requester(cur: psycopg.Cursor) -> None:
    """Make the gated writes of one employee run one at a time. Postgres releases the
    lock when the transaction ends.
    """
    ctx = get_request_context()
    cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (ctx.employee_id,))


def _find_pending(cur: psycopg.Cursor, tool: str, payload: dict[str, Any]) -> int | None:
    """The id of the requester's request with the same tool and payload that is still
    waiting for approval, or None if there is no such request.
    """
    ctx = get_request_context()
    row = cur.execute(
        """
        SELECT id
        FROM pending_actions
        WHERE employee_id = %s AND tool = %s AND payload = %s
          AND status = 'pending_approval'
        ORDER BY id
        LIMIT 1
        """,
        (ctx.employee_id, tool, Jsonb(payload)),
    ).fetchone()
    return row["id"] if row else None


def _propose(cur: psycopg.Cursor, tool: str, payload: dict[str, Any]) -> int:
    """Add one row to pending_actions for the requester, write an `action_proposed`
    audit event, and return the new id. Every gated write saves its request through
    this function.
    """
    ctx = get_request_context()
    row = cur.execute(
        """
        INSERT INTO pending_actions (request_id, employee_id, tool, payload)
        VALUES (%s, %s, %s, %s)
        RETURNING id
        """,
        (ctx.request_id, ctx.employee_id, tool, Jsonb(payload)),
    ).fetchone()
    audit.record(
        "action_proposed", {"action_id": row["id"], "tool": tool, "payload": payload}, conn=cur
    )
    return row["id"]


def calculate_working_days(start_date: date, end_date: date) -> int:
    """The number of Monday to Friday days from start_date to end_date, both included.
    Public holidays are not taken out.
    """
    days = (end_date - start_date).days + 1
    return sum(1 for n in range(days) if (start_date + timedelta(days=n)).weekday() < 5)


@gated_write
def submit_leave_request(start_date: date, end_date: date) -> str:
    """Send the requester's vacation request to an approver. This does not book the
    vacation. An approver must approve it first.

    The tool counts the working days itself: Monday to Friday, without weekends. Public
    holidays are not taken out yet, so a public holiday in the range counts as a
    working day.

    The result has "proposed": true if the request was sent. It then has "working_days",
    the number of days that were requested. If it has "proposed": false, the request was
    not sent. Tell the requester the "reason" from the result. If "already_sent" is
    true, the same request was sent before and is still waiting for approval. No new
    request was made.
    """
    started_at = time.perf_counter()
    get_request_context()
    args = {"start_date": start_date.isoformat(), "end_date": end_date.isoformat()}
    with db.transaction() as cur:
        result = _submit_leave_request(cur, start_date, end_date)
        if result["proposed"]:
            action = "already sent" if result["already_sent"] else "proposed"
            summary = f"{action} {args['start_date']} to {args['end_date']}"
            ref = f"pending_actions:{result['action_id']}"
        else:
            summary = f"not proposed: {result['reason']}"
            ref = None
        record_tool_call("submit_leave_request", args, summary, ref, started_at, conn=cur)
    return json.dumps(result)


def _submit_leave_request(cur: psycopg.Cursor, start_date: date, end_date: date) -> dict[str, Any]:
    """The checks and the insert for submit_leave_request. Returns the tool's result."""
    ctx = get_request_context()

    if end_date < start_date:
        return {"proposed": False, "reason": "the end date is before the start date"}
    if start_date < util.get_today():
        return {"proposed": False, "reason": "the start date is in the past"}
    if end_date.year != start_date.year:
        return {
            "proposed": False,
            "reason": "a request cannot span two years; split it at the new year",
        }
    working_days = calculate_working_days(start_date, end_date)
    if working_days == 0:
        return {
            "proposed": False,
            "reason": "the range has no working days; it covers only a weekend",
        }

    year = start_date.year
    payload = {
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "working_days": working_days,
    }

    # the lock makes two requests sent at the same time run one after the other,
    # so they can't both pass the checks below
    _lock_requester(cur)

    # check for a duplicate before the balance: the first request's days are
    # already subtracted, so the balance check would refuse the duplicate
    existing_id = _find_pending(cur, "submit_leave_request", payload)
    if existing_id is not None:
        return {
            "proposed": True,
            "action_id": existing_id,
            "status": "pending_approval",
            "already_sent": True,
        }

    balance = _read_leave_balance(cur, ctx.employee_id, year)
    if balance is None:
        return {"proposed": False, "reason": f"no leave balance on record for {year}"}

    # the days of requests that still wait for approval are subtracted too, because
    # approved_days only has the approved ones
    remaining = float(
        balance["entitlement_days"]
        - balance["taken_days"]
        - balance["approved_days"]
        - balance["awaiting_approval_days"]
    )
    if working_days > remaining:
        return {
            "proposed": False,
            "reason": "the request is more than the remaining balance",
            "requested_days": working_days,
            "awaiting_approval_days": float(balance["awaiting_approval_days"]),
            "remaining_days": remaining,
        }

    action_id = _propose(cur, "submit_leave_request", payload)
    return {
        "proposed": True,
        "action_id": action_id,
        "status": "pending_approval",
        "already_sent": False,
        "working_days": working_days,
        "remaining_after_approval": remaining - working_days,
    }


TicketCategory = Literal["access", "hardware", "software", "security"]


@gated_write
def create_ticket(category: TicketCategory, title: str, body: str) -> str:
    """Send the requester's IT ticket to an approver. This does not open the ticket.
    An approver must approve it first.

    category: what the problem is about. Pick one:
    - "access": VPN, accounts, passwords, permissions.
    - "hardware": laptops, monitors, phones, peripherals.
    - "software": installing, licensing or fixing applications.
    - "security": phishing, lost devices, exposed credentials.
    title: one short line that says what the problem is.
    body: the details from the requester.

    Before you call this, check get_known_outages. If the problem is a known outage,
    tell the requester about the outage instead of creating a ticket.

    The result has "proposed": true if the ticket was sent. If it has "proposed":
    false, the ticket was not sent. Tell the requester the "reason" from the result.
    If "already_sent" is true, the same ticket was sent before and is still waiting
    for approval. No new ticket was made.
    """
    started_at = time.perf_counter()
    get_request_context()
    args = {"category": category, "title": title, "body": body}
    with db.transaction() as cur:
        result = _create_ticket(cur, category, title, body)
        if result["proposed"]:
            action = "already sent" if result["already_sent"] else "proposed"
            summary = f"{action} {category} ticket: {title.strip()}"
            ref = f"pending_actions:{result['action_id']}"
        else:
            summary = f"not proposed: {result['reason']}"
            ref = None
        record_tool_call("create_ticket", args, summary, ref, started_at, conn=cur)
    return json.dumps(result)


def _create_ticket(
    cur: psycopg.Cursor, category: TicketCategory, title: str, body: str
) -> dict[str, Any]:
    """The checks and the insert for create_ticket. Returns the tool's result."""
    title, body = title.strip(), body.strip()
    if not title or not body:
        return {"proposed": False, "reason": "the ticket needs a title and a body"}

    payload = {"category": category, "title": title, "body": body}
    _lock_requester(cur)
    existing_id = _find_pending(cur, "create_ticket", payload)
    if existing_id is not None:
        return {
            "proposed": True,
            "action_id": existing_id,
            "status": "pending_approval",
            "already_sent": True,
        }

    action_id = _propose(cur, "create_ticket", payload)
    return {
        "proposed": True,
        "action_id": action_id,
        "status": "pending_approval",
        "already_sent": False,
    }
