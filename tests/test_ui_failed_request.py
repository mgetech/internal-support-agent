"""The "Why this answer?" view of a request that failed at a node: it shows the node and the
type of the error, and never the error message. The view runs on a saved record, so no
server and no database is needed.
"""

from __future__ import annotations

import json

from streamlit.testing.v1 import AppTest
from ui.app import format_failure, without_error_messages

MARKER = "marker-3c8e5a90"

FAILURE = {
    "type": "node_error",
    "node": "tools",
    "error_type": "RuntimeError",
    "error_message": f"cannot read the record of emp_001: {MARKER}",
}


def record(evidence):
    return {
        "request_id": "req-1",
        "outcome": "escalate",
        "summary": "Request failed at tools; handed to a human.",
        "total_tokens": 0,
        "total_cost_eur": 0.0,
        "latency_ms": 5,
        "citations": [],
        "policy_version": None,
        "prompt_versions": {},
        "model_calls": [],
        "evidence": evidence,
        "trace_id": None,
    }


def show(record_to_show) -> AppTest:
    def script():
        import streamlit as st
        from ui.app import show_decision_record

        show_decision_record(st.session_state["record"])

    app = AppTest.from_function(script, default_timeout=10)
    app.session_state["record"] = record_to_show
    return app.run()


def rendered_text(app: AppTest) -> str:
    """Everything the view shows, as one text."""
    parts = [element.value for element in [*app.markdown, *app.error, *app.caption, *app.json]]
    parts += [str(element.value) for element in app.metric]
    return "\n".join(str(part) for part in parts)


def test_a_failure_is_one_line_with_the_node_and_the_error_type():
    assert format_failure(FAILURE) == "Request failed at tools (RuntimeError)"


def test_the_error_message_is_taken_out_of_the_evidence_and_nothing_else():
    evidence = [{"type": "classification", "status": "ok"}, FAILURE]

    shown = without_error_messages(evidence)

    assert shown == [
        {"type": "classification", "status": "ok"},
        {"type": "node_error", "node": "tools", "error_type": "RuntimeError"},
    ]
    # the saved record is not changed
    assert evidence[1]["error_message"].endswith(MARKER)


def test_a_failed_request_shows_the_node_and_the_error_type():
    app = show(record([{"type": "classification", "status": "ok"}, FAILURE]))

    assert not app.exception
    assert [error.value for error in app.error] == ["Request failed at tools (RuntimeError)"]


def test_the_error_message_is_not_shown_anywhere_in_the_view():
    app = show(record([FAILURE]))

    assert MARKER not in rendered_text(app)
    assert "emp_001" not in rendered_text(app)
    assert "error_message" not in rendered_text(app)
    # the evidence is still there, without the message
    shown = [json.loads(element.value) for element in app.json]
    assert {"type": "node_error", "node": "tools", "error_type": "RuntimeError"} in shown[-1]


def test_a_request_that_did_not_fail_shows_no_failure_line():
    app = show(record([{"type": "classification", "status": "ok"}]))

    assert not app.exception
    assert len(app.error) == 0
