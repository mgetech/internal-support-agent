"""Shared pieces every tool uses: what a tool is, the marker for tools that propose a
change instead of making it, and the recording of each tool call. Kept apart from the
registry so tool modules can import it without a cycle.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import psycopg

from support_agent import audit
from support_agent.request_context import get_request_context

Tool = Callable[..., str]
Ref = str | list[str] | None


def gated_write(fn: Tool) -> Tool:
    """Mark a tool that proposes a change into `pending_actions` instead of making it."""
    fn.gated_write = True  # type: ignore[attr-defined]
    return fn


def is_gated_write(fn: Tool) -> bool:
    return getattr(fn, "gated_write", False)


def record_tool_call(
    tool: str,
    args: dict[str, Any],
    summary: str,
    ref: Ref,
    conn: psycopg.Cursor | None = None,
) -> None:
    """Write a `tool_call` audit event and add an evidence item to the request.

    summary: one short sentence about the result, e.g. "18.0 of 30.0 days left in 2026".
    ref: where the result came from, e.g. a pending action id or the returned chunk ids.
    conn: pass the tool's cursor to save the audit event in the tool's own transaction.
    """
    audit.record(
        "tool_call", {"tool": tool, "args": args, "summary": summary, "ref": ref}, conn=conn
    )
    get_request_context().evidence.append(
        {"type": "tool_result", "tool": tool, "summary": summary, "ref": ref}
    )
