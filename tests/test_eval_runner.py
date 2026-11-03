"""The scenario runner. The checks are plain comparisons, so most tests build a `Run` by
hand. The last tests send real scenario files through the API, with a scripted model and
the real tools on the seeded database.
"""

from __future__ import annotations

import json

import pytest
from eval import run_eval
from eval.run_eval import (
    Run,
    Scenario,
    check_run,
    get_identity_parameters,
    load_scenarios,
    main,
    run_scenario,
)
from fastapi.testclient import TestClient
from tests.graph_helpers import CLASSIFIED, ScriptedLLM, call_reply, text_reply

from support_agent.api import app, get_graph
from support_agent.decision_record import DecisionRecord
from support_agent.graph import build_graph

CHUNK = "vacation-policy#entitlement#0"


def _evidence(tools: list[str], retrieved: list[str]) -> list[dict]:
    """One `tool_result` item per tool. A search returns the `retrieved` chunks."""
    return [
        {
            "type": "tool_result",
            "tool": tool,
            "summary": "done",
            "ref": [{"id": c, "policy_version": "2026-09.1"} for c in retrieved]
            if tool == "search_policies"
            else None,
        }
        for tool in tools
    ]


def _record(**overrides: object) -> DecisionRecord:
    fields = {
        "request_id": "req-1",
        "employee_id": "emp_001",
        "channel": "rest",
        "outcome": "resolve",
        "summary": "Resolved using get_leave_balance.",
        "evidence": _evidence(["get_leave_balance"], []),
        "citations": [],
        "policy_version": None,
        "prompt_versions": {},
        "verifier": [],
        "model_calls": [],
        "total_tokens": 0,
        "total_cost_eur": 0.0,
        "latency_ms": 1,
        "trace_id": None,
    }
    return DecisionRecord(**{**fields, **overrides})


def _scenario(feedback: dict | None = None, **expected: object) -> Scenario:
    return Scenario(
        id="S900",
        title="A test scenario",
        persona="emp_001",
        request="How many days do I have left?",
        expected={"outcome": "resolve", **expected},
        feedback=feedback,
        notes="why",
    )


def _run(retrieved: list[str] | None = None, **overrides: object) -> Run:
    """A run whose record shows the same tools as the audit log, unless a test says
    otherwise. `retrieved` is what a search in the run returned.
    """
    fields = {
        "request_id": "req-1",
        "answer": "You have 18 days left.",
        "outcome": "resolve",
        "tools_called": ["get_leave_balance"],
        "proposed_actions": [],
        **overrides,
    }
    if "record" not in fields:
        evidence = _evidence(fields["tools_called"], retrieved or [])
        fields["record"] = _record(outcome=fields["outcome"], evidence=evidence)
    return Run(**fields)


def _failed(scenario: Scenario, run: Run) -> dict[str, str]:
    return {c.name: c.detail for c in check_run(scenario, run) if not c.passed}


# --- the checks ---


def test_a_run_that_matches_the_expectation_passes_every_check():
    scenario = _scenario(
        must_call=["get_leave_balance"],
        must_not_call=["search_policies"],
        answer_must_contain=["18"],
    )

    checks = check_run(scenario, _run())

    assert [c.name for c in checks] == [
        "outcome",
        "proposed_actions",
        "citations_in_retrieval_set",
        "decision_record",
        "must_call",
        "must_not_call",
        "answer_must_contain",
    ]
    assert all(c.passed for c in checks)


def test_a_scenario_that_asks_for_no_more_gets_the_checks_every_run_gets():
    assert [c.name for c in check_run(_scenario(), _run())] == [
        "outcome",
        "proposed_actions",
        "citations_in_retrieval_set",
        "decision_record",
    ]


def test_a_wrong_outcome_fails():
    failed = _failed(_scenario(), _run(outcome="escalate"))

    assert failed == {"outcome": "expected resolve, got escalate"}


def test_a_missing_required_tool_fails():
    failed = _failed(
        _scenario(must_call=["get_leave_balance"]), _run(tools_called=["search_policies"])
    )

    assert failed == {"must_call": "get_leave_balance was not called"}


def test_a_forbidden_tool_fails():
    scenario = _scenario(must_not_call=["search_policies"])

    failed = _failed(scenario, _run(tools_called=["search_policies", "get_leave_balance"]))

    assert failed == {"must_not_call": "search_policies was called"}


def test_a_proposal_the_scenario_does_not_allow_fails():
    scenario = _scenario(outcome="propose_action", proposed_actions=["submit_leave_request"])

    failed = _failed(scenario, _run(outcome="propose_action", proposed_actions=["create_ticket"]))

    assert failed == {"proposed_actions": "proposed create_ticket, which is not allowed"}


def test_a_proposal_the_scenario_allows_passes():
    scenario = _scenario(outcome="propose_action", proposed_actions=["submit_leave_request"])

    run = _run(outcome="propose_action", proposed_actions=["submit_leave_request"])

    assert _failed(scenario, run) == {}


def test_any_proposal_fails_when_the_scenario_allows_none():
    failed = _failed(_scenario(), _run(proposed_actions=["create_ticket"]))

    assert "proposed_actions" in failed


def test_answer_text_is_compared_without_regard_to_case():
    scenario = _scenario(answer_must_contain=["AWAITS"], answer_must_not_contain=["SECRET"])

    assert _failed(scenario, _run(answer="It awaits approval.")) == {}
    failed = _failed(scenario, _run(answer="It is a Secret."))
    assert failed == {
        "answer_must_contain": "the answer lacks 'AWAITS'",
        "answer_must_not_contain": "the answer has 'SECRET'",
    }


# --- citations ---


def test_a_citation_from_this_requests_search_passes():
    run = _run(
        retrieved=[CHUNK],
        tools_called=["search_policies"],
        answer=f"Ten days is too many [chunk:{CHUNK}].",
        record=_record(evidence=_evidence(["search_policies"], [CHUNK]), citations=[CHUNK]),
    )

    assert _failed(_scenario(must_cite=[CHUNK]), run) == {}


def test_a_citation_the_request_did_not_retrieve_fails():
    run = _run(answer=f"Ten days is too many [chunk:{CHUNK}].")

    problem = f"the answer cites {CHUNK}, which the request did not retrieve"
    assert _failed(_scenario(), run) == {"citations_in_retrieval_set": problem}


def test_a_record_citation_the_request_did_not_retrieve_fails():
    run = _run(record=_record(citations=[CHUNK]))

    problem = f"the record cites {CHUNK}, which the request did not retrieve"
    assert _failed(_scenario(), run) == {"citations_in_retrieval_set": problem}


def test_an_answer_that_lacks_a_required_citation_fails():
    scenario = _scenario(must_cite=[CHUNK])

    assert _failed(scenario, _run()) == {"must_cite": f"the answer does not cite {CHUNK}"}


# --- the Decision Record ---


def test_a_missing_record_fails():
    failed = _failed(_scenario(), _run(record=None))

    assert failed == {"decision_record": "no Decision Record was saved"}


def test_a_record_with_a_blank_summary_or_no_evidence_fails():
    run = _run(record=_record(summary="  ", evidence=[]), tools_called=[])

    assert _failed(_scenario(), run) == {
        "decision_record": "the summary is empty; the evidence is empty"
    }


def test_a_record_that_disagrees_with_the_answer_on_the_outcome_fails():
    run = _run(record=_record(outcome="escalate"))

    assert _failed(_scenario(), run) == {
        "decision_record": "the record has outcome escalate, the answer resolve"
    }


def test_a_tool_call_the_evidence_does_not_show_fails():
    run = _run(tools_called=["get_leave_balance", "search_policies"], record=_record())

    assert _failed(_scenario(), run) == {
        "decision_record": "search_policies was called, but the evidence does not show it"
    }


def test_the_same_tool_called_twice_needs_two_evidence_items():
    run = _run(tools_called=["get_leave_balance", "get_leave_balance"], record=_record())

    assert "get_leave_balance was called" in _failed(_scenario(), run)["decision_record"]


def test_evidence_of_a_tool_the_audit_log_does_not_have_fails():
    run = _run(tools_called=[], record=_record())

    assert _failed(_scenario(), run) == {
        "decision_record": "the evidence shows get_leave_balance, but the audit log does not"
    }


def test_the_feedback_check_needs_a_saved_feedback_id():
    scenario = _scenario(feedback={"rating": "down"})

    assert _failed(scenario, _run(feedback_id=3)) == {}
    assert _failed(scenario, _run()) == {"feedback_saved": "the API did not save the feedback"}


# --- the mechanism check ---


def test_no_tool_the_model_sees_has_an_identity_parameter():
    assert get_identity_parameters() == []


def test_the_mechanism_check_finds_a_tool_that_takes_an_employee(monkeypatch):
    def leaky(year: int, target_employee: str) -> str:
        """Returns data for whoever the caller names."""
        return ""

    monkeypatch.setattr(run_eval, "TOOLS", [leaky])
    scenario = _scenario(no_identity_parameter=True)

    assert get_identity_parameters() == ["leaky.target_employee"]
    assert _failed(scenario, _run()) == {
        "no_identity_parameter": "leaky.target_employee could name an employee"
    }


# --- the command line ---


def test_an_unknown_scenario_id_is_refused(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["S999"])

    assert exit_info.value.code == 2
    assert "unknown scenario: S999" in capsys.readouterr().err


# --- a run through the API ---


@pytest.fixture
def api(seeded_db):
    """The real graph and tools, with a scripted model. Each test sets `llm.replies`
    through `script()`. The client is the one the runner uses for a live stack.
    """
    llm = ScriptedLLM()
    graph = build_graph(llm, "small", "main")
    app.dependency_overrides[get_graph] = lambda: graph
    with TestClient(app) as client:
        client.llm = llm
        yield client
    app.dependency_overrides.clear()


def _script(client: TestClient, *replies: object) -> None:
    client.llm.replies = list(replies)
    client.llm.calls.clear()


def test_s001_passes_through_the_api(api):
    _script(api, CLASSIFIED, call_reply("c1", "get_leave_balance", "{}"), text_reply("18.0 left."))
    scenario = load_scenarios()["S001"]

    run = run_scenario(api, scenario)

    assert run.outcome == "resolve"
    assert run.tools_called == ["get_leave_balance"]
    assert run.record.request_id == run.request_id
    assert [c.name for c in check_run(scenario, run) if not c.passed] == []


def test_s001_fails_when_the_agent_searches_the_policies(api):
    _script(
        api,
        CLASSIFIED,
        call_reply("c1", "search_policies", '{"query": "vacation days"}'),
        call_reply("c2", "get_leave_balance", "{}"),
        text_reply("18.0 left."),
    )
    scenario = load_scenarios()["S001"]

    run = run_scenario(api, scenario)

    assert run.tools_called == ["search_policies", "get_leave_balance"]
    assert _failed(scenario, run) == {"must_not_call": "search_policies was called"}


def test_s003_refuses_with_a_citation_and_a_complete_record(api, search_embeds_like):
    search_embeds_like(CHUNK)
    arguments = json.dumps({"reason": "You have 3.5 days left.", "citations": [CHUNK]})
    _script(
        api,
        CLASSIFIED,
        call_reply("c1", "search_policies", '{"query": "vacation entitlement"}'),
        call_reply("c2", "get_leave_balance", "{}"),
        call_reply("c3", "decline_request", arguments),
    )
    scenario = load_scenarios()["S003"]

    run = run_scenario(api, scenario)

    assert run.tools_called == ["search_policies", "get_leave_balance", "decline_request"]
    assert run.record.citations == [CHUNK]
    assert _failed(scenario, run) == {}


def test_a_proposal_is_read_back_from_the_callers_actions(api):
    arguments = json.dumps({"category": "access", "title": "VPN", "body": "No VPN access."})
    _script(api, CLASSIFIED, call_reply("c1", "create_ticket", arguments), text_reply("Sent."))
    scenario = _scenario(outcome="propose_action", proposed_actions=["create_ticket"])

    run = run_scenario(api, scenario)

    assert run.outcome == "propose_action"
    assert run.proposed_actions == ["create_ticket"]
    assert _failed(scenario, run) == {}


def test_s006_sends_the_feedback_and_checks_that_it_was_saved(api, seeded_db):
    _script(api, CLASSIFIED, call_reply("c1", "get_leave_balance", "{}"), text_reply("18.0 left."))
    scenario = load_scenarios()["S006"]

    run = run_scenario(api, scenario)

    assert _failed(scenario, run) == {}
    row = seeded_db.execute(
        "SELECT request_id, rating, comment FROM feedback WHERE id = %s", (run.feedback_id,)
    ).fetchone()
    assert tuple(row) == (run.request_id, "down", "smoke test")
