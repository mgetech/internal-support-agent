"""An error that escapes a graph node ends the request in escalate: the Decision Record is
saved with the error as evidence, and the employee gets a short reply with the request id.
The error message is kept in the record only, never in a log. The tests use the real
database (make db-up) to save and read the record.
"""

import json
import logging
from types import SimpleNamespace

import psycopg
from fastapi.testclient import TestClient
from tests.graph_helpers import (
    CLASSIFIED,
    ScriptedLLM,
    answer,
    call_reply,
    make_tools,
    text_reply,
)

from support_agent import db
from support_agent.api import app, get_current_employee_id, get_graph
from support_agent.decision_record import save_decision_record
from support_agent.graph import build_graph
from support_agent.nodes import FAILURE_TEXT
from support_agent.reconciliation import flag_orphaned_actions
from support_agent.request_context import bind_request_context
from support_agent.state import new_state
from support_agent.tools.hr_it import create_ticket

MARKER = "marker-91c4d7e2"
EMPLOYEE = "emp_001"
TICKET = {"category": "access", "title": "VPN is down", "body": "It does not connect."}


def explode() -> str:
    """Fails with a message that has the marker."""
    raise RuntimeError(f"cannot read the record of {EMPLOYEE}: {MARKER}")


def two_calls_reply(*calls):
    """A model reply that asks for several tool calls at once: (call id, name, arguments)."""
    items = [
        SimpleNamespace(type="function_call", call_id=call_id, name=name, arguments=arguments)
        for call_id, name, arguments in calls
    ]
    return SimpleNamespace(output_text="", output=items)


def make_graph(llm, tools=None, **options):
    tools = [*make_tools([]), explode] if tools is None else tools
    options = {"save_record": save_decision_record, "get_policy_version": lambda: None, **options}
    return build_graph(llm, "small", "main", tools, **options)


def post_chat(graph):
    app.dependency_overrides[get_graph] = lambda: graph
    try:
        return TestClient(app).post(
            "/chat", json={"message": "hello"}, headers={"X-Employee-Id": EMPLOYEE}
        )
    finally:
        app.dependency_overrides.clear()


def get_record(conn, request_id):
    return conn.execute(
        "SELECT outcome, summary, evidence FROM decision_records WHERE request_id = %s",
        (request_id,),
    ).fetchone()


def failing_llm():
    return ScriptedLLM(CLASSIFIED, call_reply("c1", "explode", "{}"), text_reply("never reached"))


# --- the reply and the record ---


def test_an_error_in_the_tools_node_gives_the_failure_reply_and_an_escalate_record(
    seeded_db, caplog
):
    caplog.set_level(logging.DEBUG)
    llm = failing_llm()

    response = post_chat(make_graph(llm))

    body = response.json()
    assert response.status_code == 200
    assert body["outcome"] == "escalate"
    assert body["answer"] == FAILURE_TEXT["en"].format(request_id=body["request_id"])
    assert "HR or IT support can help" in body["answer"]
    outcome, summary, evidence = get_record(seeded_db, body["request_id"])
    assert outcome == "escalate"
    assert summary == "Request failed at tools; handed to a human."
    error = next(item for item in evidence if item["type"] == "node_error")
    assert error == {
        "type": "node_error",
        "node": "tools",
        "error_type": "RuntimeError",
        "error_message": f"cannot read the record of {EMPLOYEE}: {MARKER}",
    }
    # the agent was not called again after the error
    assert len(llm.calls) == 2


def test_the_error_message_is_in_the_record_only_and_never_in_a_log(seeded_db, caplog):
    caplog.set_level(logging.DEBUG)

    body = post_chat(make_graph(failing_llm())).json()

    assert MARKER not in caplog.text
    assert EMPLOYEE not in caplog.text
    assert "Traceback" not in caplog.text
    request_id = body["request_id"]
    assert f"node_failed node=tools error_type=RuntimeError request_id={request_id}" in caplog.text


def test_the_reply_is_in_the_language_of_the_request_when_it_is_known(seeded_db):
    german = text_reply(answer("hr", "read", "de"))
    llm = ScriptedLLM(german, call_reply("c1", "explode", "{}"))

    body = post_chat(make_graph(llm)).json()

    assert body["answer"] == FAILURE_TEXT["de"].format(request_id=body["request_id"])


def test_an_error_in_the_classify_node_is_handled_the_same_way(seeded_db):
    llm = ScriptedLLM(RuntimeError(f"model failed: {MARKER}"))

    body = post_chat(make_graph(llm)).json()

    assert body["outcome"] == "escalate"
    assert body["answer"] == FAILURE_TEXT["en"].format(request_id=body["request_id"])
    _, summary, evidence = get_record(seeded_db, body["request_id"])
    assert summary == "Request failed at classify; handed to a human."
    assert [item["node"] for item in evidence if item["type"] == "node_error"] == ["classify"]


def test_secrets_in_the_error_message_are_removed_before_the_record_is_saved(seeded_db):
    def connect() -> str:
        """Fails with a connection string, a password and a key in the message."""
        raise RuntimeError(
            "cannot connect to postgresql://app:s3cret-pw@db:5432/app, password=hunter2, "
            "key sk-proj-AbCdEfGhIjKlMnOpQrStUvWxYz0123456789 "
            f"for {EMPLOYEE}"
        )

    llm = ScriptedLLM(CLASSIFIED, call_reply("c1", "connect", "{}"))

    body = post_chat(make_graph(llm, tools=[connect])).json()

    _, _, evidence = get_record(seeded_db, body["request_id"])
    message = next(i for i in evidence if i["type"] == "node_error")["error_message"]
    for secret in ("s3cret-pw", "hunter2", "sk-proj-AbCd"):
        assert secret not in message
    assert message == (
        "cannot connect to postgresql://[redacted]@db:5432/app, password=[redacted], "
        f"key [redacted] for {EMPLOYEE}"
    )


# --- a proposal made before the error ---


def propose_then_explode():
    """The ticket is proposed in one tools node, and the next tools node fails."""
    return ScriptedLLM(
        CLASSIFIED,
        call_reply("c1", "create_ticket", json.dumps(TICKET)),
        call_reply("c2", "explode", "{}"),
    )


def test_a_proposal_made_before_the_error_has_the_record_of_the_request(seeded_db):
    graph = make_graph(propose_then_explode(), tools=[create_ticket, explode])

    body = post_chat(graph).json()

    action = seeded_db.execute(
        "SELECT id, request_id, status FROM pending_actions WHERE request_id = %s",
        (body["request_id"],),
    ).fetchone()
    assert action is not None
    assert action[2] == "pending_approval"
    outcome, _, evidence = get_record(seeded_db, body["request_id"])
    assert outcome == "escalate"
    proposal = next(
        i for i in evidence if i["type"] == "tool_result" and i["tool"] == "create_ticket"
    )
    assert proposal["ref"] == f"pending_actions:{action[0]}"
    # the orphan check finds nothing: the action has a record
    assert flag_orphaned_actions(grace_minutes=0) == 0


def run_graph_directly(graph):
    with bind_request_context(EMPLOYEE, "req-direct", "rest"):
        return graph.invoke(new_state("hello"))


def test_the_state_keeps_the_proposal_of_an_earlier_node(seeded_db):
    graph = make_graph(propose_then_explode(), tools=[create_ticket, explode])

    state = run_graph_directly(graph)

    assert state["failed_node"] == "tools"
    assert len(state["proposed_action_ids"]) == 1


def test_the_state_keeps_the_proposal_made_in_the_same_node_before_the_error(seeded_db):
    llm = ScriptedLLM(
        CLASSIFIED,
        two_calls_reply(("c1", "create_ticket", json.dumps(TICKET)), ("c2", "explode", "{}")),
    )
    graph = make_graph(llm, tools=[create_ticket, explode])

    state = run_graph_directly(graph)

    assert state["failed_node"] == "tools"
    assert len(state["proposed_action_ids"]) == 1
    types = [item["type"] for item in state["decision_evidence"]]
    assert types == ["classification", "tool_result", "node_error"]


# --- when saving the record fails too ---


def test_with_the_database_unavailable_the_request_is_the_generic_500(monkeypatch, caplog):
    caplog.set_level(logging.DEBUG)

    def unavailable():
        raise psycopg.OperationalError(f"connection refused for {EMPLOYEE}: {MARKER}")

    monkeypatch.setattr(db, "transaction", unavailable)
    app.dependency_overrides[get_current_employee_id] = lambda: EMPLOYEE

    # fake tools need no database, so the only database use is saving the record
    response = post_chat(make_graph(failing_llm()))

    assert response.status_code == 500
    assert response.json()["detail"] == "the request failed"
    request_id = response.json()["request_id"]
    assert MARKER not in response.text
    assert MARKER not in caplog.text
    assert EMPLOYEE not in caplog.text
    assert f"request_failed error_type=OperationalError request_id={request_id}" in caplog.text
