"""The classify node and the refuse node. StubLLM stands in for the model client and
returns the JSON text the classifier would answer with.
"""

import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from tests.graph_helpers import (
    ScriptedLLM,
    answer,
    run_graph,
    text_reply,
)

from support_agent.llm import LLMUnavailableError
from support_agent.nodes import (
    CLASSIFICATION_FORMAT,
    REFUSAL_TEXT,
    classify_node,
    refuse_node,
    route_after_classify,
)
from support_agent.prompts import get_prompt
from support_agent.request_context import bind_request_context, get_request_context
from support_agent.state import AgentState


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


def run_classify(reply, messages=None):
    """Run the classify node in a bound request. Returns the stub, the update the node
    returned and the request's evidence.
    """
    llm = StubLLM(reply)
    state = {"messages": messages or [HumanMessage("How many vacation days do I have?")]}
    with bind_request_context("emp_001", "req-1", "rest") as ctx:
        state_update = classify_node(llm, "small")(state)
    return llm, state_update, ctx.evidence


def test_classify_returns_family_risk_and_language():
    _, state_update, _ = run_classify(answer("it", "write", "de"))

    assert (state_update["family"], state_update["risk"], state_update["language"]) == (
        "it",
        "write",
        "de",
    )
    assert "outcome" not in state_update


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
    _, state_update, evidence = run_classify(answer("hr", "read", "en"))

    verdict = {"type": "classification", "status": "ok", "family": "hr", "risk": "read"}
    assert state_update["decision_evidence"] == [{**verdict, "language": "en"}]
    assert state_update["decision_evidence"] == evidence


@pytest.mark.parametrize("risk", ["read", "write"])
def test_in_scope_requests_go_to_the_agent(risk):
    _, state_update, _ = run_classify(answer(risk=risk))

    assert route_after_classify(state_update) == "agent"


def test_out_of_scope_goes_to_refusal_and_ends_with_escalate():
    _, state_update, evidence = run_classify(answer("other", "out_of_scope", "en"))

    assert state_update["outcome"] == "escalate"
    assert evidence[-1] == {"type": "escalation", "reason": "out_of_scope"}
    assert route_after_classify(state_update) == "refuse"

    final = refuse_node(state_update)

    assert final["outcome"] == "escalate"
    assert final["messages"][0].content == REFUSAL_TEXT["en"]


def test_the_refusal_is_in_the_language_of_the_request():
    final = refuse_node({"risk": "out_of_scope", "language": "de"})

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
    _, state_update, evidence = run_classify(reply)

    assert state_update["outcome"] == "escalate"
    assert "family" not in state_update
    assert evidence[0]["type"] == "classification"
    assert evidence[0]["status"] == "invalid"
    assert evidence[-1] == {"type": "escalation", "reason": "classification_failed"}
    assert route_after_classify(state_update) == "refuse"


def test_a_model_outage_escalates_and_refusal_defaults_to_english():
    outage = LLMUnavailableError(["small"], RuntimeError("down"))
    _, state_update, _ = run_classify(outage)
    degraded = {"type": "model_degraded", "outcome": "escalate", "models_tried": ["small"]}
    escalation = {"type": "escalation", "reason": "classification_failed"}
    assert state_update == {"outcome": "escalate", "decision_evidence": [degraded, escalation]}
    assert route_after_classify(state_update) == "refuse"

    assert refuse_node(state_update)["messages"][0].content == REFUSAL_TEXT["en"]


def _classify_graph(llm):
    """classify -> refuse or a stand-in agent node, wired the way the real graph will be."""
    graph = StateGraph(AgentState)
    graph.add_node("classify", classify_node(llm, "small"))
    graph.add_node("refuse", refuse_node)
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


def test_out_of_scope_never_reaches_the_agent():
    llm = ScriptedLLM(text_reply(answer("other", "out_of_scope", "en")))

    result, _, _ = run_graph(llm)

    assert len(llm.calls) == 1
    assert result["outcome"] == "escalate"
