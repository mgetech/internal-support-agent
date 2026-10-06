"""The check for proposals left without a Decision Record. A request that proposed an
action and then crashed before `finalize` leaves a `pending_actions` row that nothing
explains. This check finds those rows and flags them in the audit log. It changes no row.
"""

from __future__ import annotations

import logging

from support_agent import audit, db
from support_agent.request_context import bind_request_context

logger = logging.getLogger(__name__)


def flag_orphaned_actions(grace_minutes: int) -> int:
    """Write an `orphan_flagged` audit event for every pending action whose request has
    no Decision Record. Returns the number of actions flagged.

    Only actions created more than `grace_minutes` ago are checked, so a request that is
    still running is never flagged. An action that has an event already is skipped, so
    running this at every start adds no duplicates.
    """
    with db.transaction() as cur:
        # two processes that start together must not flag the same action twice
        cur.execute("SELECT pg_advisory_xact_lock(hashtext('flag_orphaned_actions'))")
        orphans = cur.execute(
            """
            SELECT id, request_id, employee_id, tool
            FROM pending_actions AS action
            WHERE created_at < now() - make_interval(mins => %s)
              AND NOT EXISTS (
                  SELECT 1 FROM decision_records AS record
                  WHERE record.request_id = action.request_id
              )
              AND NOT EXISTS (
                  SELECT 1 FROM audit_log AS flag
                  WHERE flag.request_id = action.request_id
                    AND flag.event = 'orphan_flagged'
                    AND flag.payload->>'action_id' = action.id::text
              )
            ORDER BY id
            """,
            (grace_minutes,),
        ).fetchall()
        for orphan in orphans:
            # the audit log takes the request id from the context. The channel is not
            # stored with the proposal, and the audit log does not use it.
            with bind_request_context(orphan["employee_id"], orphan["request_id"], "rest"):
                audit.record(
                    "orphan_flagged",
                    {"action_id": orphan["id"], "tool": orphan["tool"]},
                    actor="system",
                    conn=cur,
                )
    logger.info("flagged %d pending actions without a decision record", len(orphans))
    return len(orphans)
