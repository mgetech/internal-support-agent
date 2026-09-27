"""The tool belt. `TOOLS` is the only list the graph, the MCP server and the tests read;
a tool that isn't registered here isn't reachable from any transport.

Tools are plain functions. LangGraph's ToolNode and the MCP server both take callables
and build the model-visible schema from the signature, so the signature is the contract.
Identity is never part of it: tools read it from the bound request context.
"""

from __future__ import annotations

from collections.abc import Callable

from support_agent.tools.hr_it import get_leave_balance

Tool = Callable[..., str]

TOOLS: list[Tool] = [get_leave_balance]


def gated_write(fn: Tool) -> Tool:
    """Mark a tool that proposes a change into `pending_actions` instead of making it."""
    fn.gated_write = True  # type: ignore[attr-defined]
    return fn


def is_gated_write(fn: Tool) -> bool:
    return getattr(fn, "gated_write", False)


__all__ = ["TOOLS", "Tool", "gated_write", "is_gated_write"]
