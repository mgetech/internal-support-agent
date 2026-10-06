"""The agent graph: how the nodes are wired together. The nodes are in `nodes.py`."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import partial

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from support_agent.config import DEFAULT_MAX_TOOL_CALLS_PER_REQUEST
from support_agent.decision_record import DecisionRecord, save_decision_record
from support_agent.llm import LLMClient
from support_agent.nodes import (
    agent_node,
    classify_node,
    finalize_node,
    refuse_node,
    route_after_agent,
    route_after_classify,
    route_after_verify,
    tool_limit_node,
    tools_node,
    verify_node,
)
from support_agent.state import AgentState
from support_agent.tools import TOOLS, Tool
from support_agent.tools.policy_search import get_policy_version


def build_graph(
    llm: LLMClient,
    classifier_model: str,
    agent_model: str,
    tools: Sequence[Tool] = TOOLS,
    max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS_PER_REQUEST,
    save_record: Callable[[DecisionRecord], None] = save_decision_record,
    get_policy_version: Callable[[], str | None] = get_policy_version,
) -> CompiledStateGraph:
    """classify -> agent <-> tools -> verify -> finalize. Out-of-scope requests, failures,
    a reached tool limit (`max_tool_calls`) and a failed check go through refuse. Every
    path ends in finalize, which saves the Decision Record with `save_record`. The record
    names the policy version that `get_policy_version` returns.
    """
    graph = StateGraph(AgentState)
    graph.add_node("classify", classify_node(llm, classifier_model))
    graph.add_node("refuse", refuse_node)
    graph.add_node("agent", agent_node(llm, agent_model, tools))
    graph.add_node("tools", tools_node(tools))
    graph.add_node("tool_limit_reached", partial(tool_limit_node, max_tool_calls=max_tool_calls))
    graph.add_node("verify", verify_node)
    graph.add_node(
        "finalize",
        partial(finalize_node, save_record=save_record, get_policy_version=get_policy_version),
    )

    graph.add_edge(START, "classify")
    graph.add_conditional_edges("classify", route_after_classify, ["agent", "refuse"])
    graph.add_conditional_edges(
        "agent",
        partial(route_after_agent, max_tool_calls=max_tool_calls),
        {
            "tools": "tools",
            "verify": "verify",
            "tool_limit_reached": "tool_limit_reached",
            "refuse": "refuse",
        },
    )
    graph.add_conditional_edges(
        "verify", route_after_verify, {"agent": "agent", "end": "finalize", "refuse": "refuse"}
    )
    graph.add_edge("tools", "agent")
    graph.add_edge("tool_limit_reached", "refuse")
    graph.add_edge("refuse", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile()
