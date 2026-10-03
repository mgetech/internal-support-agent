import json
from types import SimpleNamespace
from typing import get_args, get_type_hints

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph

from support_agent.graph import build_graph
from support_agent.llm import LLMUnavailableError
from support_agent.nodes import (
    CLASSIFICATION_FORMAT,
    REFUSAL_TEXT,
    classify_node,
    refuse_node,
    route_after_agent,
    route_after_classify,
    tool_limit_node,
)
from support_agent.prompts import get_prompt
from support_agent.request_context import bind_request_context, get_request_context
from support_agent.state import AgentState, Outcome, new_state
from support_agent.tools import gated_write


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


# --- agent node, tools node and the wired graph ---


class ScriptedLLM:
    """Stands in for LLMClient. Each call returns, or raises, the next scripted reply.
    The arguments of every call are kept.
    """

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def create(self, model, input, prompt=None, **options):
        self.calls.append({"model": model, "input": input, "prompt": prompt, **options})
        reply = self.replies[len(self.calls) - 1]
        if isinstance(reply, LLMUnavailableError):
            get_request_context().evidence.append(
                {"type": "model_degraded", "outcome": "escalate", "models_tried": [model]}
            )
        if isinstance(reply, Exception):
            raise reply
        return reply


def text_reply(text):
    return SimpleNamespace(output_text=text, output=[SimpleNamespace(type="message")])


def call_reply(call_id, name, arguments):
    """A model reply that asks for one tool call. `arguments` is the JSON text."""
    item = SimpleNamespace(type="function_call", call_id=call_id, name=name, arguments=arguments)
    return SimpleNamespace(output_text="", output=[item])


CHUNK = "vacation-policy#entitlement#0"
CLASSIFIED = text_reply(answer("hr", "read", "en"))


def _add_tool_result(tool, summary, ref):
    get_request_context().evidence.append(
        {"type": "tool_result", "tool": tool, "summary": summary, "ref": ref}
    )


def make_tools(tool_calls):
    """Tools named like the real ones, with no database. Each call they receive is added
    to `tool_calls` as (name, arguments). The fake `create_ticket` acts like the real one:
    a title it has seen before gets the same action id, a new title gets the next id.
    """
    action_ids: dict[str, int] = {}

    def search_policies(query: str) -> str:
        """Search the policies."""
        tool_calls.append(("search_policies", {"query": query}))
        _add_tool_result("search_policies", "1 chunks found", [CHUNK])
        return json.dumps({"results": [{"chunk_id": CHUNK}]})

    def get_leave_balance(year: int = 2026) -> str:
        """Get the balance."""
        tool_calls.append(("get_leave_balance", {"year": year}))
        _add_tool_result("get_leave_balance", "18.0 of 30.0 days left", "leave_balances:x")
        return json.dumps({"remaining_days": 18.0})

    @gated_write
    def create_ticket(title: str) -> str:
        """Propose a ticket."""
        tool_calls.append(("create_ticket", {"title": title}))
        action_id = action_ids.setdefault(title, 5 + len(action_ids))
        _add_tool_result("create_ticket", "proposed a ticket", f"pending_actions:{action_id}")
        return json.dumps({"proposed": True, "action_id": action_id})

    return [search_policies, get_leave_balance, create_ticket]


def run_graph(llm, text="hi", **graph_options):
    """Run the wired graph in a bound request. Returns the final state, the tools' log
    and the request's evidence. `graph_options` go to build_graph, for example
    `max_tool_calls=2`.
    """
    tool_calls = []
    graph = build_graph(llm, "small", "main", make_tools(tool_calls), **graph_options)
    with bind_request_context("emp_001", "req-1", "rest") as ctx:
        result = graph.invoke(new_state(text))
    return result, tool_calls, ctx.evidence


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


def test_tool_results_become_evidence_and_the_calls_are_counted():
    llm = ScriptedLLM(
        CLASSIFIED,
        call_reply("call_1", "get_leave_balance", "{}"),
        call_reply("call_2", "get_leave_balance", json.dumps({"year": 2025})),
        text_reply("done"),
    )

    result, _, evidence = run_graph(llm)

    assert result["tool_calls_used"] == 2
    assert [item["type"] for item in result["decision_evidence"]] == [
        "classification",
        "tool_result",
        "tool_result",
    ]
    assert result["decision_evidence"] == evidence


def test_search_results_fill_the_citation_whitelist():
    llm = ScriptedLLM(
        CLASSIFIED,
        call_reply("call_1", "search_policies", json.dumps({"query": "vacation"})),
        text_reply(f"30 days [chunk:{CHUNK}]"),
    )

    result, _, _ = run_graph(llm)

    assert result["retrieved_chunk_ids"] == [CHUNK]


def test_other_tools_do_not_touch_the_whitelist():
    llm = ScriptedLLM(CLASSIFIED, call_reply("c", "get_leave_balance", "{}"), text_reply("ok"))

    result, _, _ = run_graph(llm)

    assert result["retrieved_chunk_ids"] == []


def test_same_proposals_are_collected_once():
    llm = ScriptedLLM(
        CLASSIFIED,
        call_reply("c1", "create_ticket", json.dumps({"title": "laptop"})),
        call_reply("c2", "create_ticket", json.dumps({"title": "laptop"})),
        text_reply("sent for approval"),
    )

    result, _, _ = run_graph(llm)

    assert result["proposed_action_ids"] == [5]


def test_each_new_proposal_adds_its_own_id():
    llm = ScriptedLLM(
        CLASSIFIED,
        call_reply("c1", "create_ticket", json.dumps({"title": "laptop"})),
        call_reply("c2", "create_ticket", json.dumps({"title": "monitor"})),
        text_reply("sent for approval"),
    )

    result, _, _ = run_graph(llm)

    assert result["proposed_action_ids"] == [5, 6]


def test_an_employee_id_argument_from_the_model_never_reaches_the_tool():
    llm = ScriptedLLM(
        CLASSIFIED,
        call_reply("c", "get_leave_balance", json.dumps({"year": 2025, "employee_id": "emp_999"})),
        text_reply("ok"),
    )

    _, tool_calls, _ = run_graph(llm)

    assert tool_calls == [("get_leave_balance", {"year": 2025})]


def test_an_unknown_tool_is_answered_with_an_error_and_not_run():
    llm = ScriptedLLM(CLASSIFIED, call_reply("c", "run_sql", "{}"), text_reply("sorry"))

    result, tool_calls, _ = run_graph(llm)

    assert tool_calls == []
    assert "run_sql" in llm.calls[2]["input"][-1]["output"]
    assert result["messages"][-1].content == "sorry"


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


def test_out_of_scope_never_reaches_the_agent():
    llm = ScriptedLLM(text_reply(answer("other", "out_of_scope", "en")))

    result, _, _ = run_graph(llm)

    assert len(llm.calls) == 1
    assert result["outcome"] == "escalate"


# --- the tool limit ---
#
# The tool limit is the most tool calls one request may use (`max_tool_calls`, 8 by default).
# Before the tools run, the router adds the calls the model just asked for to the calls
# already used. If the total is over the limit, none of the new calls run and the request
# ends with `escalate`. A group of calls is never cut to fit, because the model would
# then answer from a partial result.
#
# The router tests below call `route_after_agent` directly with a small hand-made state.
# The graph tests run the whole graph with the same stand-ins as the tests above
# (ScriptedLLM, text_reply, call_reply, make_tools).


def state_after_agent_reply(tool_calls_used, requested_calls):
    """The state as `route_after_agent` sees it, right after the agent node replied.
    `tool_calls_used` is the count so far. The last message asks for `requested_calls`
    new tool calls.
    """
    calls = []
    for number in range(requested_calls):
        calls.append(
            {"type": "tool_call", "id": f"call_{number}", "name": "get_leave_balance", "args": {}}
        )
    return {
        "messages": [AIMessage("", tool_calls=calls)],
        "tool_calls_used": tool_calls_used,
    }


def several_calls_reply(*calls):
    """A model reply that asks for several tool calls at once. Each call is a
    (call_id, tool name) pair, and every call has no arguments.
    """
    items = []
    for call_id, name in calls:
        items.append(
            SimpleNamespace(type="function_call", call_id=call_id, name=name, arguments="{}")
        )
    return SimpleNamespace(output_text="", output=items)


def test_calls_that_fit_in_the_limit_go_to_the_tools():
    state = state_after_agent_reply(tool_calls_used=3, requested_calls=2)

    assert route_after_agent(state, max_tool_calls=8) == "tools"


def test_using_exactly_the_limit_is_allowed():
    state = state_after_agent_reply(tool_calls_used=7, requested_calls=1)

    assert route_after_agent(state, max_tool_calls=8) == "tools"


def test_one_call_over_the_limit_goes_to_tool_limit_reached():
    state = state_after_agent_reply(tool_calls_used=8, requested_calls=1)

    assert route_after_agent(state, max_tool_calls=8) == "tool_limit_reached"


def test_a_group_of_calls_that_does_not_fit_is_rejected_as_a_whole():
    state = state_after_agent_reply(tool_calls_used=6, requested_calls=3)

    assert route_after_agent(state, max_tool_calls=8) == "tool_limit_reached"


def test_a_reply_without_tool_calls_is_never_over_the_limit():
    state = state_after_agent_reply(tool_calls_used=8, requested_calls=0)

    assert route_after_agent(state, max_tool_calls=8) == "verify"


def test_the_tool_limit_node_escalates_and_records_the_numbers():
    state = state_after_agent_reply(tool_calls_used=8, requested_calls=1)

    with bind_request_context("emp_001", "req-1", "rest"):
        state_update = tool_limit_node(state, max_tool_calls=8)

    assert state_update["outcome"] == "escalate"
    assert state_update["decision_evidence"] == [
        {"type": "tool_limit", "tool_calls_used": 8, "tool_calls_requested": 1, "limit": 8},
        {"type": "escalation", "reason": "tool_limit_reached"},
    ]


def test_a_model_that_keeps_asking_for_tools_is_stopped_and_escalated():
    # The limit is 2. The model asks for a tool 3 times in a row.
    llm = ScriptedLLM(
        CLASSIFIED,
        call_reply("call_1", "get_leave_balance", "{}"),
        call_reply("call_2", "get_leave_balance", "{}"),
        call_reply("call_3", "get_leave_balance", "{}"),
        # There is no fifth reply. If the graph called the model again after
        # the stop, ScriptedLLM would fail with an IndexError.
    )

    result, tool_calls, _ = run_graph(llm, max_tool_calls=2)

    assert len(tool_calls) == 2
    assert result["tool_calls_used"] == 2
    # the model was called 4 times: classify, then the agent 3 times
    assert len(llm.calls) == 4
    assert result["outcome"] == "escalate"
    assert result["messages"][-1].content == REFUSAL_TEXT["en"]


def test_the_limit_stop_is_visible_in_the_decision_evidence():
    llm = ScriptedLLM(
        CLASSIFIED,
        call_reply("call_1", "get_leave_balance", "{}"),
        call_reply("call_2", "get_leave_balance", "{}"),
        call_reply("call_3", "get_leave_balance", "{}"),
    )

    result, _, _ = run_graph(llm, max_tool_calls=2)

    # in order: the classification, the two tool results, then the stop
    assert [item["type"] for item in result["decision_evidence"]] == [
        "classification",
        "tool_result",
        "tool_result",
        "tool_limit",
        "escalation",
    ]
    assert result["decision_evidence"][-2] == {
        "type": "tool_limit",
        "tool_calls_used": 2,
        "tool_calls_requested": 1,
        "limit": 2,
    }
    assert result["decision_evidence"][-1]["reason"] == "tool_limit_reached"


def test_a_group_of_calls_over_the_limit_runs_none_of_them():
    # The limit is 2. In one reply the model asks for 3 tools.
    llm = ScriptedLLM(
        CLASSIFIED,
        several_calls_reply(
            ("call_1", "get_leave_balance"),
            ("call_2", "get_leave_balance"),
            ("call_3", "get_leave_balance"),
        ),
    )

    result, tool_calls, _ = run_graph(llm, max_tool_calls=2)

    # not even the first two ran: the group is not cut to fit
    assert tool_calls == []
    assert result["tool_calls_used"] == 0
    assert result["outcome"] == "escalate"


def test_a_request_that_uses_exactly_the_limit_still_finishes():
    # The limit is 2. The model uses 2 tools and then answers.
    llm = ScriptedLLM(
        CLASSIFIED,
        call_reply("call_1", "get_leave_balance", "{}"),
        call_reply("call_2", "get_leave_balance", "{}"),
        text_reply("You have 18 days left."),
    )

    result, tool_calls, _ = run_graph(llm, max_tool_calls=2)

    assert len(tool_calls) == 2
    assert result["messages"][-1].content == "You have 18 days left."
    # no outcome is set yet. The verify node, which sets it, comes in a later commit
    assert result["outcome"] is None
