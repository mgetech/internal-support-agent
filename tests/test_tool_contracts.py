"""Structural checks on the tool belt. Identity comes from the request context, so no
model-visible tool schema may carry a parameter that could name an employee.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Iterator
from datetime import date
from typing import Any, Literal

import pytest
from langchain_core.tools import tool as as_langchain_tool
from mcp.server.mcpserver import MCPServer
from tests.tool_args import sample_args

from support_agent.request_context import NoRequestContextError
from support_agent.tools import TOOLS, Tool

# The tool belt as published to the model. Adding or removing a tool is a contract change.
EXPECTED_TOOLS = {
    "get_leave_balance",
    "search_policies",
    "get_known_outages",
    "submit_leave_request",
    "create_ticket",
}

# Parameter names that would let the caller say whose data to use. Compared after
# lowercasing and stripping separators, so `employeeId`, `employee-id` and
# `EMPLOYEE_ID` all match `employeeid`.
FORBIDDEN_NAMES = {
    # direct identifiers
    "employee",
    "employeeid",
    "empid",
    "user",
    "userid",
    "username",
    "person",
    "personid",
    "staffid",
    "email",
    # acting on someone else's behalf
    "onbehalfof",
    "asuser",
    "impersonate",
    "requester",
    "requesterid",
    "actor",
    "owner",
    "ownerid",
    # approval identity belongs to the service layer, never to a tool
    "approver",
    "approverid",
}

# Catches variants the explicit list misses, e.g. `target_employee` or `for_user`.
FORBIDDEN_FRAGMENTS = re.compile(r"employee|user|behalf|requester|approver|impersonat")


def _normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def is_identity_field(name: str) -> bool:
    norm = _normalize(name)
    return norm in FORBIDDEN_NAMES or bool(FORBIDDEN_FRAGMENTS.search(norm))


def property_names(schema: Any) -> Iterator[str]:
    """Every property name in a JSON schema, including nested objects and $defs."""
    if isinstance(schema, dict):
        props = schema.get("properties")
        if isinstance(props, dict):
            yield from props
        for value in schema.values():
            yield from property_names(value)
    elif isinstance(schema, list):
        for item in schema:
            yield from property_names(item)


def langgraph_schema(fn: Tool) -> dict[str, Any]:
    """The schema ToolNode hands the model for a plain-function tool."""
    return as_langchain_tool(fn).tool_call_schema.model_json_schema()


def mcp_schema(fn: Tool) -> dict[str, Any]:
    """The input schema an MCP client sees for the tool."""
    server = MCPServer("contract-check")
    server.add_tool(fn)
    (published,) = asyncio.run(server.list_tools())
    return published.input_schema


def identity_fields(fn: Tool) -> set[str]:
    names = set(property_names(langgraph_schema(fn))) | set(property_names(mcp_schema(fn)))
    return {n for n in names if is_identity_field(n)}


def test_registry_publishes_the_tool_contract():
    assert sorted(t.__name__ for t in TOOLS) == sorted(EXPECTED_TOOLS)


@pytest.mark.parametrize("fn", TOOLS, ids=lambda t: t.__name__)
def test_tool_schema_has_no_identity_field(fn: Tool):
    assert identity_fields(fn) == set(), (
        f"{fn.__name__} exposes an identity parameter; read it from the request context"
    )


@pytest.mark.parametrize("fn", TOOLS, ids=lambda t: t.__name__)
def test_unbound_tool_raises(fn: Tool):
    with pytest.raises(NoRequestContextError):
        fn(**sample_args(fn))


# INTEGRATION TEST: this test uses the real Postgres (make db-up).
@pytest.mark.parametrize("fn", TOOLS, ids=lambda t: t.__name__)
def test_unbound_tool_writes_nothing(fn: Tool, clean_db):
    with pytest.raises(NoRequestContextError):
        fn(**sample_args(fn))

    tables = [
        row[0]
        for row in clean_db.execute(
            "select tablename from pg_tables where schemaname = 'public'"
        ).fetchall()
    ]
    written = {
        table: count
        for table in tables
        if (count := clean_db.execute(f"SELECT count(*) FROM {table}").fetchone()[0])
    }
    assert written == {}


def test_sample_args_cover_the_contract_types():
    def propose(
        start_date: date, working_days: float, category: Literal["access", "hardware"]
    ) -> str:
        """Fake gated write."""
        return ""

    def search(query: str, limit: int | None = 5) -> str:
        """Fake read."""
        return ""

    assert sample_args(propose) == {
        "start_date": date(2026, 11, 2),
        "working_days": 1.0,
        "category": "access",
    }
    assert sample_args(search) == {"query": "x"}


def test_checker_catches_an_identity_parameter():
    def leaky(year: int, on_behalf_of: str, filters: dict[str, str] | None = None) -> str:
        """Returns data for whoever the caller names."""
        return ""

    def clean(start_date: str, end_date: str, working_days: float) -> str:
        """Takes no identity."""
        return ""

    assert identity_fields(leaky) == {"on_behalf_of"}
    assert identity_fields(clean) == set()
    assert is_identity_field("employeeId")
    assert is_identity_field("target_employee")
    assert not is_identity_field("query")
