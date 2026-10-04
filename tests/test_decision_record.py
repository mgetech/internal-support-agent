"""Decision Records: how a record is built from the final state, how it is saved, and
that every way a request can end writes one.

Three groups of tests:
1. `summarize` and `create_decision_record` are plain code. The tests give them a
   hand-made final state and check the record.
2. `save_decision_record` writes to Postgres, so those tests need `make db-up`. They are
   skipped when no database is reachable.
3. The graph tests run the whole graph with the stand-ins from tests/graph_helpers.py.
   The record is not written to the database there. `run_graph` collects it in a list.
"""

import json

import pytest
from tests.graph_helpers import (
    CHUNK,
    CLASSIFIED,
    ScriptedLLM,
    answer,
    call_reply,
    run_graph,
    several_calls_reply,
    text_reply,
)

from support_agent import audit
from support_agent.decision_record import (
    create_decision_record,
    save_decision_record,
    summarize,
)
from support_agent.llm import LLMUnavailableError
from support_agent.request_context import bind_request_context

AGENT_PROMPT = {"prompt_id": "agent_system", "prompt_version": "1.0.0"}
CLASSIFIER_PROMPT = {"prompt_id": "classifier", "prompt_version": "1.0.0"}


# --- summarize: the one-sentence "why" ---


def test_a_resolved_request_names_the_tools_it_used():
    evidence = [
        {"type": "tool_result", "tool": "get_leave_balance"},
        {"type": "tool_result", "tool": "get_known_outages"},
        {"type": "tool_result", "tool": "get_leave_balance"},
    ]

    summary = summarize("resolve", evidence, citations=[], proposed_action_ids=[])

    # a tool that was used twice is named once
    assert summary == "Resolved using get_leave_balance, get_known_outages."


def test_a_resolved_request_without_tools_says_so():
    summary = summarize("resolve", [], citations=[], proposed_action_ids=[])

    assert summary == "Resolved using no tools."


def test_a_resolved_request_lists_the_chunks_it_cited():
    evidence = [{"type": "tool_result", "tool": "search_policies"}]

    summary = summarize("resolve", evidence, citations=[CHUNK], proposed_action_ids=[])

    assert summary == f"Resolved using search_policies, citing {CHUNK}."


def test_a_proposal_names_the_pending_actions():
    summary = summarize("propose_action", [], citations=[], proposed_action_ids=[5, 6])

    assert summary == "Proposed for approval (pending action 5, 6); nothing was changed yet."


@pytest.mark.parametrize(
    ("reason", "words"),
    [
        ("out_of_scope", "the request is outside HR and IT support"),
        ("classification_failed", "the request could not be classified"),
        ("model_unavailable", "no model answered"),
        ("invalid_tool_arguments", "the model sent tool arguments that were not valid JSON"),
        ("tool_limit_reached", "the tool limit was reached"),
        ("citation_check_failed", "the answer cited a source this request did not retrieve"),
    ],
)
def test_an_escalation_gives_its_reason(reason, words):
    evidence = [{"type": "escalation", "reason": reason}]

    summary = summarize("escalate", evidence, citations=[], proposed_action_ids=[])

    assert summary == f"Escalated to a person: {words}."


# --- create_decision_record: building the record from the final state ---


def final_state(outcome="resolve", evidence=(), proposed_action_ids=()):
    """A state as the finalize node sees it, at the end of a request."""
    return {
        "messages": [],
        "outcome": outcome,
        "decision_evidence": list(evidence),
        "proposed_action_ids": list(proposed_action_ids),
        "retrieved_chunk_ids": [],
    }


def model_call(model, prompt, input_tokens, output_tokens, cost_eur):
    """An evidence item for a model call that worked, as the model client writes it."""
    return {
        "type": "model_call",
        "model": model,
        "attempt": 1,
        "status": "ok",
        "prompt": prompt,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_eur": cost_eur,
    }


def passed_verdict(citations):
    return {
        "type": "verifier_verdict",
        "check": "citations_in_retrieval_set",
        "passed": True,
        "citations": citations,
        "unknown_citations": [],
        "objection": None,
    }


def failed_verdict(citations):
    return {
        "type": "verifier_verdict",
        "check": "citations_in_retrieval_set",
        "passed": False,
        "citations": citations,
        "unknown_citations": citations,
        "objection": "an objection",
    }


def build_record(state, policy_version=None, channel="rest"):
    """Build the record inside a bound request for emp_004."""
    with bind_request_context("emp_004", "req-77", channel):
        return create_decision_record(state, policy_version)


def test_the_record_takes_the_request_identity_from_the_context():
    record = build_record(final_state(), channel="mcp")

    assert (record.request_id, record.employee_id, record.channel) == ("req-77", "emp_004", "mcp")


def test_the_record_keeps_the_evidence_in_order():
    evidence = [{"type": "classification"}, {"type": "tool_result", "tool": "t"}]

    record = build_record(final_state(evidence=evidence))

    assert record.evidence == evidence


def test_the_record_adds_up_tokens_and_cost_over_all_model_calls():
    evidence = [
        model_call("small", CLASSIFIER_PROMPT, 200, 50, 0.0005),
        model_call("main", AGENT_PROMPT, 1000, 500, 0.002),
    ]

    record = build_record(final_state(evidence=evidence))

    assert record.total_tokens == 1750
    assert record.total_cost_eur == pytest.approx(0.0025)
    assert [call["model"] for call in record.model_calls] == ["small", "main"]


def test_a_failed_model_try_is_in_the_evidence_but_not_in_the_model_calls():
    failed_try = {
        "type": "model_call",
        "model": "main",
        "attempt": 1,
        "status": "retryable_error",
        "prompt": AGENT_PROMPT,
    }
    evidence = [failed_try, model_call("main", AGENT_PROMPT, 100, 10, 0.001)]

    record = build_record(final_state(evidence=evidence))

    assert len(record.evidence) == 2
    assert len(record.model_calls) == 1


def test_the_record_lists_each_prompt_version_that_was_used():
    evidence = [
        model_call("small", CLASSIFIER_PROMPT, 1, 1, 0.0),
        model_call("main", AGENT_PROMPT, 1, 1, 0.0),
        model_call("main", AGENT_PROMPT, 1, 1, 0.0),
    ]

    record = build_record(final_state(evidence=evidence))

    assert record.prompt_versions == {"classifier": "1.0.0", "agent_system": "1.0.0"}


def test_the_record_keeps_every_verdict_and_cites_what_the_final_answer_cited():
    # the first draft failed, the retried draft passed and cited CHUNK
    evidence = [failed_verdict(["other#x#0"]), passed_verdict([CHUNK])]

    record = build_record(final_state(evidence=evidence))

    assert [verdict["passed"] for verdict in record.verifier] == [False, True]
    assert record.citations == [CHUNK]


def test_an_escalated_request_has_no_citations():
    # both drafts failed, so no answer was sent
    evidence = [failed_verdict(["a#b#0"]), failed_verdict(["a#b#0"])]

    record = build_record(final_state(outcome="escalate", evidence=evidence))

    assert record.citations == []


def test_the_policy_version_is_recorded_when_given():
    assert build_record(final_state(), policy_version="2026-09.1").policy_version == "2026-09.1"
    assert build_record(final_state()).policy_version is None


def test_the_record_has_a_latency_and_no_trace_id_yet():
    record = build_record(final_state())

    assert record.latency_ms >= 0
    assert record.trace_id is None


def test_a_request_without_an_outcome_cannot_be_recorded():
    with pytest.raises(ValueError, match="without an outcome"):
        build_record(final_state(outcome=None))


# --- save_decision_record: writing to Postgres ---
#
# These tests use the `seeded_db` fixture (tests/conftest.py), which rebuilds the schema
# and loads the seed data, so employee emp_004 exists. They are skipped without a database.


def save_a_record(evidence=(), outcome="resolve", request_id="req-77"):
    """Build a record for emp_004 and save it, all inside one bound request."""
    state = final_state(outcome=outcome, evidence=evidence)
    with bind_request_context("emp_004", request_id, "rest"):
        record = create_decision_record(state, policy_version="2026-09.1")
        save_decision_record(record)
    return record


def test_a_saved_record_can_be_read_back(seeded_db):
    evidence = [
        {"type": "tool_result", "tool": "search_policies", "summary": "1 chunks found"},
        model_call("main", AGENT_PROMPT, 1000, 500, 0.002),
        passed_verdict([CHUNK]),
    ]

    record = save_a_record(evidence)

    row = seeded_db.execute(
        "SELECT employee_id, channel, outcome, summary, evidence, citations, policy_version,"
        " prompt_versions, verifier, model_calls, total_tokens, total_cost_eur, trace_id"
        " FROM decision_records WHERE request_id = 'req-77'"
    ).fetchone()
    assert row[:4] == ("emp_004", "rest", "resolve", record.summary)
    assert row[4] == evidence
    assert row[5] == [CHUNK]
    assert row[6] == "2026-09.1"
    assert row[7] == {"agent_system": "1.0.0"}
    assert row[8] == [passed_verdict([CHUNK])]
    assert row[9] == record.model_calls
    assert row[10] == 1500
    assert float(row[11]) == pytest.approx(0.002)
    assert row[12] is None


def test_a_record_without_citations_is_saved_with_an_empty_list(seeded_db):
    save_a_record(evidence=[{"type": "tool_result", "tool": "get_leave_balance"}])

    (citations,) = seeded_db.execute("SELECT citations FROM decision_records").fetchone()
    assert citations == []


def test_saving_a_record_writes_the_final_audit_event(seeded_db):
    record = save_a_record()

    event = seeded_db.execute(
        "SELECT request_id, actor, event, payload FROM audit_log WHERE event = 'decision_recorded'"
    ).fetchone()
    assert event == (
        "req-77",
        "system",
        "decision_recorded",
        {"outcome": "resolve", "summary": record.summary},
    )


def test_the_record_and_its_audit_event_are_saved_together(seeded_db, monkeypatch):
    # If the audit write fails, the record must not stay behind.
    def failing_audit(event, payload, actor=None, conn=None):
        raise RuntimeError("the audit write failed")

    monkeypatch.setattr(audit, "record", failing_audit)

    with pytest.raises(RuntimeError, match="the audit write failed"):
        save_a_record()

    (count,) = seeded_db.execute("SELECT count(*) FROM decision_records").fetchone()
    assert count == 0


def test_two_records_for_the_same_request_are_refused(seeded_db):
    import psycopg

    save_a_record()

    with pytest.raises(psycopg.errors.UniqueViolation):
        save_a_record()


# --- every way a request can end writes a record with evidence ---
#
# `run_graph` collects the saved records in the list `records`. Each test below ends the
# request in a different way and checks that exactly one record was saved, with the
# right outcome and with evidence that is not empty.


def run_and_collect(llm, **graph_options):
    """Run the graph. Returns the final state and the list of saved records."""
    records = []
    result, _, _ = run_graph(llm, saved_records=records, **graph_options)
    return result, records


def assert_one_record(records, outcome):
    assert len(records) == 1
    assert records[0].outcome == outcome
    assert records[0].evidence != []
    assert records[0].summary != ""


def test_a_resolved_request_is_recorded():
    llm = ScriptedLLM(CLASSIFIED, text_reply("You have 18 days left."))

    _, records = run_and_collect(llm)

    assert_one_record(records, "resolve")


def test_a_request_with_a_proposal_is_recorded():
    llm = ScriptedLLM(
        CLASSIFIED,
        call_reply("call_1", "create_ticket", '{"title": "laptop"}'),
        text_reply("I sent your ticket for approval."),
    )

    _, records = run_and_collect(llm)

    assert_one_record(records, "propose_action")
    assert (
        records[0].summary == "Proposed for approval (pending action 5); nothing was changed yet."
    )


def test_an_out_of_scope_request_is_recorded():
    llm = ScriptedLLM(text_reply(answer("other", "out_of_scope", "en")))

    _, records = run_and_collect(llm)

    assert_one_record(records, "escalate")
    assert records[0].summary.startswith("Escalated to a person: the request is outside")


def test_a_classifier_outage_is_recorded():
    outage = LLMUnavailableError(["small"], RuntimeError("down"))

    _, records = run_and_collect(ScriptedLLM(outage))

    assert_one_record(records, "escalate")


def test_an_agent_outage_is_recorded():
    outage = LLMUnavailableError(["main"], RuntimeError("down"))

    _, records = run_and_collect(ScriptedLLM(CLASSIFIED, outage))

    assert_one_record(records, "escalate")
    assert "no model answered" in records[0].summary


def test_invalid_tool_arguments_are_recorded():
    llm = ScriptedLLM(CLASSIFIED, call_reply("c", "get_leave_balance", "{not json"))

    _, records = run_and_collect(llm)

    assert_one_record(records, "escalate")


def test_a_reached_tool_limit_is_recorded():
    llm = ScriptedLLM(
        CLASSIFIED,
        several_calls_reply(("c1", "get_leave_balance"), ("c2", "get_leave_balance")),
    )

    _, records = run_and_collect(llm, max_tool_calls=1)

    assert_one_record(records, "escalate")
    assert "tool limit" in records[0].summary


def test_two_failed_drafts_are_recorded():
    bad_draft = text_reply(f"30 days [chunk:{CHUNK}].")
    llm = ScriptedLLM(CLASSIFIED, bad_draft, bad_draft)

    _, records = run_and_collect(llm)

    assert_one_record(records, "escalate")
    # the record shows both failed verdicts and the retry between them
    assert [verdict["passed"] for verdict in records[0].verifier] == [False, False]
    assert "retry" in [item["type"] for item in records[0].evidence]


def test_the_record_matches_the_final_state_and_names_the_policy_version():
    llm = ScriptedLLM(CLASSIFIED, text_reply("You have 18 days left."))

    result, records = run_and_collect(llm, get_policy_version=lambda: "2026-09.1")

    assert records[0].evidence == result["decision_evidence"]
    assert records[0].policy_version == "2026-09.1"
    assert records[0].request_id == "req-1"


def test_the_graph_result_is_still_json_serializable_evidence():
    # the record's evidence is stored as JSON, so every item must survive json.dumps
    llm = ScriptedLLM(
        CLASSIFIED,
        call_reply("call_1", "search_policies", '{"query": "vacation"}'),
        text_reply(f"30 days [chunk:{CHUNK}]."),
    )

    _, records = run_and_collect(llm)

    assert json.loads(json.dumps(records[0].evidence)) == records[0].evidence
