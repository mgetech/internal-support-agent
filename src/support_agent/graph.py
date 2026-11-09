"""The agent graph: how the nodes are wired together. The nodes are in `nodes.py`."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import partial
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from support_agent.config import DEFAULT_MAX_TOOL_CALLS_PER_REQUEST
from support_agent.decision_record import DecisionRecord, save_decision_record
from support_agent.llm import LLMClient
from support_agent.nodes import (
    agent_node,
    classify_node,
    decline_node,
    finalize_node,
    guard_node,
    refuse_node,
    route_after_agent,
    route_after_classify,
    route_after_tools,
    route_after_verify,
    tool_limit_node,
    tools_node,
    verify_node,
)
from support_agent.state import AgentState
from support_agent.tools import TOOLS, Tool
from support_agent.tools.policy_search import get_policy_version
from support_agent.tracing import trace_node


def build_graph(
    llm: LLMClient,
    classifier_model: str,
    agent_model: str,
    tools: Sequence[Tool] = TOOLS,
    max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS_PER_REQUEST,
    save_record: Callable[[DecisionRecord], None] = save_decision_record,
    get_policy_version: Callable[[], str | None] = get_policy_version,
    classifier_effort: str = "",
    agent_effort: str = "",
) -> CompiledStateGraph:
    """classify -> agent <-> tools -> verify -> finalize. A refusal goes from tools through
    decline to verify. Out-of-scope requests, failures, a reached tool limit
    (`max_tool_calls`) and a failed check go through refuse. An error that escapes a node
    ends the request in escalate: the graph goes straight to finalize (see `guard_node`).
    Every path ends in finalize, which saves the Decision Record with `save_record`. The record
    names the policy version that `get_policy_version` returns. The efforts are the
    reasoning efforts of the classify and agent calls; empty means the model's default.
    """
    graph = StateGraph(AgentState)

    def add_node(name: str, node: Callable[[AgentState], dict[str, Any]]) -> None:
        # the guard is outside the span, so the span sees the error before the guard ends it
        graph.add_node(name, guard_node(name, trace_node(name, node)))

    def or_finalize(route: Callable[[AgentState], str]) -> Callable[[AgentState], str]:
        """After a node failed, every route goes to finalize."""

        def routed(state: AgentState) -> str:
            return "finalize" if state.get("failed_node") else route(state)

        return routed

    add_node("classify", classify_node(llm, classifier_model, classifier_effort))
    add_node("refuse", refuse_node)
    add_node("agent", agent_node(llm, agent_model, tools, agent_effort))
    add_node("tools", tools_node(tools))
    add_node("decline", decline_node)
    add_node("tool_limit_reached", partial(tool_limit_node, max_tool_calls=max_tool_calls))
    add_node("verify", verify_node)
    # finalize is traced but not guarded: if saving the record fails, the request fails
    graph.add_node(
        "finalize",
        trace_node(
            "finalize",
            partial(finalize_node, save_record=save_record, get_policy_version=get_policy_version),
        ),
    )

    graph.add_edge(START, "classify")
    graph.add_conditional_edges(
        "classify", or_finalize(route_after_classify), ["agent", "refuse", "finalize"]
    )
    graph.add_conditional_edges(
        "agent",
        or_finalize(partial(route_after_agent, max_tool_calls=max_tool_calls)),
        {
            "tools": "tools",
            "verify": "verify",
            "tool_limit_reached": "tool_limit_reached",
            "refuse": "refuse",
            "finalize": "finalize",
        },
    )
    graph.add_conditional_edges(
        "verify",
        or_finalize(route_after_verify),
        {"agent": "agent", "end": "finalize", "refuse": "refuse", "finalize": "finalize"},
    )
    graph.add_conditional_edges(
        "tools", or_finalize(route_after_tools), ["agent", "decline", "finalize"]
    )
    graph.add_conditional_edges(
        "decline", or_finalize(lambda state: "verify"), ["verify", "finalize"]
    )
    graph.add_conditional_edges(
        "tool_limit_reached", or_finalize(lambda state: "refuse"), ["refuse", "finalize"]
    )
    graph.add_edge("refuse", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile()
