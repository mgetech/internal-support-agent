"""The agent node: what it sends to the model and how it handles the reply. The stand-ins
(ScriptedLLM, text_reply, call_reply, run_graph) are in tests/graph_helpers.py.
"""

import json

from tests.graph_helpers import (
    CLASSIFIED,
    ScriptedLLM,
    call_reply,
    run_graph,
    text_reply,
)

from support_agent.graph import build_graph
from support_agent.llm import LLMUnavailableError
from support_agent.nodes import (
    REFUSAL_TEXT,
    route_after_agent,
)
from support_agent.prompts import get_prompt
from support_agent.request_context import bind_request_context
from support_agent.state import new_state


def test_the_agent_sends_the_system_prompt_and_the_tool_belt():
    llm = ScriptedLLM(CLASSIFIED, text_reply("done"))

    run_graph(llm)

    call = llm.calls[1]
    assert call["model"] == "main"
    assert call["instructions"] == get_prompt("agent_system").text
    assert call["prompt"] == get_prompt("agent_system")
    assert [tool["name"] for tool in call["tools"]] == [
        "search_policies",
        "get_leave_balance",
        "create_ticket",
    ]
    assert all(tool["type"] == "function" and tool["strict"] is False for tool in call["tools"])


def test_the_real_tool_belt_has_no_identity_parameter_in_its_schemas():
    llm = ScriptedLLM(CLASSIFIED, text_reply("done"))
    graph = build_graph(llm, "small", "main")

    with bind_request_context("emp_001", "req-1", "rest"):
        graph.invoke(new_state("hi"))

    for tool in llm.calls[1]["tools"]:
        assert not any("employee" in name for name in tool["parameters"].get("properties", {}))


def test_a_reply_without_tool_calls_ends_the_loop():
    result, _, _ = run_graph(ScriptedLLM(CLASSIFIED, text_reply("You have 18 days.")))

    assert result["messages"][-1].content == "You have 18 days."
    assert route_after_agent(result) == "verify"
    assert result["tool_calls_used"] == 0


def test_tool_calls_run_and_the_result_goes_back_to_the_model():
    llm = ScriptedLLM(
        CLASSIFIED,
        call_reply("call_1", "get_leave_balance", json.dumps({"year": 2026})),
        text_reply("You have 18 days left."),
    )

    result, tool_calls, _ = run_graph(llm)

    assert tool_calls == [("get_leave_balance", {"year": 2026})]
    assert result["messages"][-1].content == "You have 18 days left."
    assert llm.calls[2]["input"] == [
        {"role": "user", "content": "hi"},
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "get_leave_balance",
            "arguments": json.dumps({"year": 2026}),
        },
        {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": json.dumps({"remaining_days": 18.0}),
        },
    ]


def test_a_model_outage_in_the_agent_escalates():
    outage = LLMUnavailableError(["main"], RuntimeError("down"))

    result, _, _ = run_graph(ScriptedLLM(CLASSIFIED, outage))

    assert result["outcome"] == "escalate"
    assert result["messages"][-1].content == REFUSAL_TEXT["en"]
    assert [item["type"] for item in result["decision_evidence"]] == [
        "classification",
        "model_degraded",
        "escalation",
    ]
    assert result["decision_evidence"][-1]["reason"] == "model_unavailable"


def test_tool_arguments_that_are_not_json_escalate():
    llm = ScriptedLLM(CLASSIFIED, call_reply("c", "get_leave_balance", "{not json"))

    result, tool_calls, _ = run_graph(llm)

    assert tool_calls == []
    assert result["outcome"] == "escalate"
    assert result["decision_evidence"][-1]["reason"] == "invalid_tool_arguments"
