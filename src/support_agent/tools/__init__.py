"""The tool belt. `TOOLS` is the only list the graph, the MCP server and the tests read;
a tool that isn't registered here isn't reachable from any transport.

Tools are plain functions. LangGraph's ToolNode and the MCP server both take callables
and build the model-visible schema from the signature, so the signature is the contract.
Identity is never part of it: tools read it from the bound request context.
"""

from __future__ import annotations

from support_agent.tools.base import Tool, gated_write, is_gated_write
from support_agent.tools.hr_it import get_known_outages, get_leave_balance, submit_leave_request
from support_agent.tools.policy_search import search_policies

TOOLS: list[Tool] = [get_leave_balance, get_known_outages, search_policies, submit_leave_request]

__all__ = ["TOOLS", "Tool", "gated_write", "is_gated_write"]
