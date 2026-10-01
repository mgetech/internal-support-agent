from typing import get_args, get_type_hints

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph

from support_agent.state import AgentState, Outcome


def _echo_graph():
    """A graph with one node that returns the same update twice in a row."""
    graph = StateGraph(AgentState)
    graph.add_node(
        "step",
        lambda state: {
            "messages": [AIMessage("hi")],
            "retrieved_chunk_ids": ["vacation-policy#entitlement#0"],
            "proposed_action_ids": [7],
            "decision_evidence": [{"type": "tool_result"}],
            "tool_calls_used": state["tool_calls_used"] + 1,
        },
    )
    graph.add_edge(START, "step")
    graph.add_edge("step", END)
    return graph.compile()


def _start_state():
    return {
        "messages": [HumanMessage("hello")],
        "retrieved_chunk_ids": ["it-policy#access#0"],
        "proposed_action_ids": [],
        "tool_calls_used": 0,
        "outcome": None,
        "verifier_objection": None,
        "decision_evidence": [{"type": "classification"}],
    }


def test_state_has_the_fields_the_graph_needs():
    assert set(get_type_hints(AgentState)) == {
        "messages",
        "family",
        "risk",
        "retrieved_chunk_ids",
        "proposed_action_ids",
        "tool_calls_used",
        "outcome",
        "verifier_objection",
        "decision_evidence",
    }


def test_outcome_is_the_stable_enum():
    assert set(get_args(Outcome)) == {
        "resolve",
        "propose_action",
        "escalate",
        "refuse_with_citation",
    }


def test_lists_accumulate_across_nodes():
    result = _echo_graph().invoke(_start_state())

    assert result["retrieved_chunk_ids"] == ["it-policy#access#0", "vacation-policy#entitlement#0"]
    assert result["proposed_action_ids"] == [7]
    assert result["decision_evidence"] == [{"type": "classification"}, {"type": "tool_result"}]
    assert [m.content for m in result["messages"]] == ["hello", "hi"]


def test_counter_and_outcome_are_replaced_not_added():
    result = _echo_graph().invoke(_start_state())

    assert result["tool_calls_used"] == 1
    assert result["outcome"] is None
