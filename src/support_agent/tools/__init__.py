"""The tool belt. `TOOLS` is the only list the graph, the MCP server and the tests read;
a tool that isn't registered here isn't reachable from any transport.

Tools are plain functions. LangGraph's ToolNode and the MCP server both take callables
and build the model-visible schema from the signature, so the signature is the contract.
Identity is never part of it: tools read it from the bound request context.
"""

from __future__ import annotations

from collections.abc import Callable

Tool = Callable[..., str]

TOOLS: list[Tool] = []

__all__ = ["TOOLS", "Tool"]
