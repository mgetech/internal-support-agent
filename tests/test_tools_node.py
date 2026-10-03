"""The tools node and the tool limit. The stand-ins (ScriptedLLM, make_tools, run_graph) are
in tests/graph_helpers.py.
"""

import json

from langchain_core.messages import AIMessage
from tests.graph_helpers import (
    CHUNK,
    CLASSIFIED,
    ScriptedLLM,
    call_reply,
    run_graph,
    several_calls_reply,
    text_reply,
)

from support_agent.nodes import (
    REFUSAL_TEXT,
    route_after_agent,
    tool_limit_node,
)
from support_agent.request_context import bind_request_context


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
