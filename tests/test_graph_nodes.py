import json
from types import SimpleNamespace
from typing import get_args, get_type_hints

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph

from support_agent.llm import LLMUnavailableError
from support_agent.nodes import (
    CLASSIFICATION_FORMAT,
    REFUSAL_TEXT,
    make_classify_node,
    refuse,
    route_after_classify,
)
from support_agent.prompts import get_prompt
from support_agent.request_context import bind_request_context, get_request_context
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
        "language",
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


class StubLLM:
    """Stands in for LLMClient. `reply` is the JSON text the model returns, or an
    exception to raise. The arguments of each call are kept.
    """

    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def create(self, model, input, prompt=None, **options):
        self.calls.append({"model": model, "input": input, "prompt": prompt, **options})
        if isinstance(self.reply, LLMUnavailableError):
            # the real client adds this item to the request before it raises
            get_request_context().evidence.append(
                {"type": "model_degraded", "outcome": "escalate", "models_tried": [model]}
            )
        if isinstance(self.reply, Exception):
            raise self.reply
        return SimpleNamespace(output_text=self.reply)


def answer(family="hr", risk="read", language="en"):
    return json.dumps({"family": family, "risk": risk, "language": language})


def run_classify(reply, messages=None):
    """Run the classify node in a bound request. Returns the stub, the update the node
    returned and the request's evidence.
    """
    llm = StubLLM(reply)
    state = {"messages": messages or [HumanMessage("How many vacation days do I have?")]}
    with bind_request_context("emp_001", "req-1", "rest") as ctx:
        update = make_classify_node(llm, "small")(state)
    return llm, update, ctx.evidence


def test_classify_returns_family_risk_and_language():
    _, update, _ = run_classify(answer("it", "write", "de"))

    assert (update["family"], update["risk"], update["language"]) == ("it", "write", "de")
    assert "outcome" not in update


def test_classify_makes_one_constrained_call_to_the_small_deployment():
    llm, _, _ = run_classify(answer(), [HumanMessage("old question"), HumanMessage("my laptop")])

    (call,) = llm.calls
    assert call["model"] == "small"
    assert call["input"] == [{"role": "user", "content": "my laptop"}]
    assert call["instructions"] == get_prompt("classifier").text
    assert call["prompt"] == get_prompt("classifier")
    assert call["text"] == {"format": CLASSIFICATION_FORMAT}
    assert CLASSIFICATION_FORMAT["strict"] is True


def test_classify_adds_the_verdict_as_an_evidence_item():
    _, update, evidence = run_classify(answer("hr", "read", "en"))

    verdict = {"type": "classification", "status": "ok", "family": "hr", "risk": "read"}
    assert update["decision_evidence"] == [{**verdict, "language": "en"}]
    assert update["decision_evidence"] == evidence


@pytest.mark.parametrize("risk", ["read", "write"])
def test_in_scope_requests_go_to_the_agent(risk):
    _, update, _ = run_classify(answer(risk=risk))

    assert route_after_classify(update) == "agent"


def test_out_of_scope_goes_to_refusal_and_ends_with_escalate():
    _, update, _ = run_classify(answer("other", "out_of_scope", "en"))
    assert route_after_classify(update) == "refuse"

    with bind_request_context("emp_001", "req-1", "rest") as ctx:
        final = refuse(update)

    assert final["outcome"] == "escalate"
    assert final["messages"][0].content == REFUSAL_TEXT["en"]
    assert final["decision_evidence"] == [{"type": "escalation", "reason": "out_of_scope"}]
    assert final["decision_evidence"] == ctx.evidence


def test_the_refusal_is_in_the_language_of_the_request():
    with bind_request_context("emp_001", "req-1", "rest"):
        final = refuse({"risk": "out_of_scope", "language": "de"})

    assert final["messages"][0].content == REFUSAL_TEXT["de"]


@pytest.mark.parametrize(
    "reply",
    [
        "not json",
        json.dumps({"family": "legal", "risk": "read", "language": "en"}),
        json.dumps({"family": "hr", "risk": "read"}),
        json.dumps({"family": "hr", "risk": "read", "language": "en", "extra": 1}),
    ],
)
def test_an_answer_that_does_not_fit_the_schema_escalates(reply):
    _, update, evidence = run_classify(reply)

    assert update["outcome"] == "escalate"
    assert "family" not in update
    assert evidence[0]["type"] == "classification"
    assert evidence[0]["status"] == "invalid"
    assert route_after_classify(update) == "refuse"


def test_a_model_outage_escalates_and_refusal_defaults_to_english():
    outage = LLMUnavailableError(["small"], RuntimeError("down"))
    _, update, _ = run_classify(outage)
    degraded = {"type": "model_degraded", "outcome": "escalate", "models_tried": ["small"]}
    assert update == {"outcome": "escalate", "decision_evidence": [degraded]}
    assert route_after_classify(update) == "refuse"

    with bind_request_context("emp_001", "req-1", "rest"):
        final = refuse(update)
    assert final["messages"][0].content == REFUSAL_TEXT["en"]
    assert final["decision_evidence"] == [{"type": "escalation", "reason": "classification_failed"}]


def _classify_graph(llm):
    """classify -> refuse or a stand-in agent node, wired the way the real graph will be."""
    graph = StateGraph(AgentState)
    graph.add_node("classify", make_classify_node(llm, "small"))
    graph.add_node("refuse", refuse)
    graph.add_node("agent", lambda state: {"messages": [AIMessage("agent ran")]})
    graph.add_edge(START, "classify")
    graph.add_conditional_edges("classify", route_after_classify, ["agent", "refuse"])
    graph.add_edge("refuse", END)
    graph.add_edge("agent", END)
    return graph.compile()


def _invoke(llm):
    with bind_request_context("emp_001", "req-1", "rest"):
        return _classify_graph(llm).invoke({"messages": [HumanMessage("hi")]})


def test_the_graph_stops_at_refusal_when_out_of_scope():
    result = _invoke(StubLLM(answer("other", "out_of_scope", "de")))

    assert result["outcome"] == "escalate"
    assert result["messages"][-1].content == REFUSAL_TEXT["de"]
    assert [item["type"] for item in result["decision_evidence"]] == [
        "classification",
        "escalation",
    ]


def test_the_graph_reaches_the_agent_when_in_scope():
    result = _invoke(StubLLM(answer("hr", "read", "en")))

    assert result.get("outcome") is None
    assert result["messages"][-1].content == "agent ran"
