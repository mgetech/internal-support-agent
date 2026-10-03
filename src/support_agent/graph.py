"""The agent graph: how the nodes are wired together. The nodes are in `nodes.py`."""

from __future__ import annotations

from collections.abc import Sequence

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from support_agent.llm import LLMClient
from support_agent.nodes import (
    agent_node,
    classify_node,
    refuse,
    route_after_agent,
    route_after_classify,
    tools_node,
)
from support_agent.state import AgentState
from support_agent.tools import TOOLS, Tool


def build_graph(
    llm: LLMClient,
    classifier_model: str,
    agent_model: str,
    tools: Sequence[Tool] = TOOLS,
) -> CompiledStateGraph:
    """classify -> agent <-> tools. Out-of-scope requests and failures go to refuse."""
    graph = StateGraph(AgentState)
    graph.add_node("classify", classify_node(llm, classifier_model))
    graph.add_node("refuse", refuse)
    graph.add_node("agent", agent_node(llm, agent_model, tools))
    graph.add_node("tools", tools_node(tools))

    graph.add_edge(START, "classify")
    graph.add_conditional_edges("classify", route_after_classify, ["agent", "refuse"])
    # a reply without tool calls ends the graph until the verify node exists
    graph.add_conditional_edges(
        "agent", route_after_agent, {"tools": "tools", "verify": END, "refuse": "refuse"}
    )
    graph.add_edge("tools", "agent")
    graph.add_edge("refuse", END)
    return graph.compile()
