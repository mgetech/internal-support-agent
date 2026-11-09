"""A trace must never change what a request does, and the logs must never hold what an
employee wrote. The chat tests use the real database (make db-up) to check the Decision
Record. MARKER stands for the text of an employee's message and for anything else that
must not leak. The employee in the header is emp_001.
"""

import logging
import re

import pytest
from fastapi.testclient import TestClient
from tests.graph_helpers import CLASSIFIED, ScriptedLLM, call_reply, make_tools, text_reply
from tests.test_llm_fallback import PRICES, FakeOpenAI, answer

from support_agent import audit, tracing
from support_agent.api import app, get_graph
from support_agent.decision_record import save_decision_record
from support_agent.graph import build_graph
from support_agent.llm import LLMClient
from support_agent.request_context import bind_request_context
from support_agent.tracing import get_langfuse, get_trace_id, start_observation

MARKER = "marker-7f3a9c41"
EMPLOYEE = "emp_001"
ANSWER = "You have 18 days left."


class BrokenClient:
    """A Langfuse client where every call fails with this error."""

    def __init__(self, error):
        self.error = error

    def __getattr__(self, name):
        raise self.error


class BrokenAtTheEnd:
    """A Langfuse client that opens spans, but fails when a span is updated or closed and
    when the trace id is asked for.
    """

    def __init__(self, error):
        self.error = error

    def start_as_current_observation(self, **kwargs):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *args):
        raise self.error

    def update(self, **fields):
        raise self.error

    def get_current_trace_id(self):
        raise self.error


FAILURES = {
    "the client raises": BrokenClient(RuntimeError(f"cannot send {MARKER}")),
    "the client times out": BrokenClient(TimeoutError(f"timed out for {MARKER}")),
    "a span fails when it is updated and closed": BrokenAtTheEnd(RuntimeError(f"bad {MARKER}")),
}


@pytest.fixture(params=FAILURES)
def broken_client(request, monkeypatch):
    monkeypatch.setattr(tracing, "get_langfuse", lambda: FAILURES[request.param])


def post_chat(message=MARKER):
    """A chat request through the API with the wired graph, a scripted model and fake tools.
    The Decision Record is saved in the database.
    """
    llm = ScriptedLLM(CLASSIFIED, call_reply("c1", "get_leave_balance", "{}"), text_reply(ANSWER))
    graph = build_graph(
        llm,
        "small",
        "main",
        make_tools([]),
        save_record=save_decision_record,
        get_policy_version=lambda: None,
    )
    app.dependency_overrides[get_graph] = lambda: graph
    try:
        return TestClient(app).post(
            "/chat", json={"message": message}, headers={"X-Employee-Id": EMPLOYEE}
        )
    finally:
        app.dependency_overrides.clear()


# --- a failing trace does not change the request ---


def test_a_chat_request_still_answers_and_saves_its_record_when_tracing_fails(
    broken_client, seeded_db
):
    response = post_chat()

    assert response.status_code == 200
    body = response.json()
    assert (body["answer"], body["outcome"]) == (ANSWER, "resolve")
    row = seeded_db.execute(
        "SELECT outcome, trace_id FROM decision_records WHERE request_id = %s",
        (body["request_id"],),
    ).fetchone()
    assert row == ("resolve", None)


def test_a_failing_trace_does_not_stop_the_body_of_a_span_or_hide_its_error(broken_client):
    with start_observation("request") as span:
        span.update(output="x")
        result = get_trace_id()
    assert result is None

    with pytest.raises(ValueError, match="from the body"), start_observation("request"):
        raise ValueError("from the body")


def test_a_model_call_still_returns_its_reply_when_tracing_fails(broken_client, monkeypatch):
    monkeypatch.setattr(audit, "record", lambda event, payload, **kwargs: None)
    client = LLMClient(FakeOpenAI([answer("hello")]), PRICES)

    with bind_request_context(EMPLOYEE, "req-1", "rest"):
        response = client.create("main", [{"role": "user", "content": "hi"}])

    assert response.output_text == "hello"


def test_a_client_that_cannot_be_made_means_tracing_is_off(monkeypatch, caplog):
    def fail():
        raise RuntimeError(f"cannot read the settings for {MARKER}")

    monkeypatch.setattr(tracing, "get_settings", fail)
    get_langfuse.cache_clear()
    try:
        assert get_langfuse() is None
    finally:
        get_langfuse.cache_clear()

    assert "tracing_failed event=create_client error_type=RuntimeError" in caplog.text
    assert MARKER not in caplog.text


# --- the logs ---


def assert_clean(text):
    assert MARKER not in text
    assert EMPLOYEE not in text


def test_a_failing_trace_is_logged_with_the_error_type_and_the_request_id_only(
    broken_client, seeded_db, caplog
):
    caplog.set_level(logging.DEBUG)

    request_id = post_chat().json()["request_id"]

    warnings = [
        r.getMessage() for r in caplog.records if r.getMessage().startswith("tracing_failed")
    ]
    assert warnings
    assert all(f"request_id={request_id}" in line for line in warnings)
    assert all(re.search(r"error_type=(RuntimeError|TimeoutError)", line) for line in warnings)
    assert_clean(caplog.text)


def test_the_logs_of_a_traced_request_have_neither_the_message_nor_the_employee(
    spans, seeded_db, caplog, monkeypatch
):
    monkeypatch.setattr(tracing, "is_content_traced", lambda: True)
    caplog.set_level(logging.DEBUG)

    assert post_chat().status_code == 200

    assert_clean(caplog.text)


class FailingGraph:
    def invoke(self, state):
        raise RuntimeError(f"cannot read the record of {EMPLOYEE}: {MARKER}")


def test_an_unexpected_error_is_logged_by_type_and_request_id_without_message_or_trace(
    seeded_db, caplog
):
    caplog.set_level(logging.DEBUG)
    app.dependency_overrides[get_graph] = FailingGraph
    try:
        response = TestClient(app).post(
            "/chat", json={"message": MARKER}, headers={"X-Employee-Id": EMPLOYEE}
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 500
    assert_clean(response.text)
    request_id = response.json()["request_id"]
    assert f"request_failed error_type=RuntimeError request_id={request_id}" in caplog.text
    assert "Traceback" not in caplog.text
    assert_clean(caplog.text)
