"""The scenario runner. The checks are plain comparisons, so most tests build a `Run` by
hand. The last tests send real scenario files through the API, with a scripted model and
the real tools on the seeded database.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, date, datetime

import pytest
from data.generate import calculate_index_build_id, generate_policy_chunks
from eval import run_eval
from eval.run_eval import (
    Check,
    Header,
    Result,
    Run,
    Scenario,
    check_run,
    format_results,
    format_summary,
    get_error_result,
    get_header,
    get_identity_parameters,
    get_result,
    get_run_path,
    load_scenarios,
    main,
    run_scenario,
)
from fastapi.testclient import TestClient
from tests.conftest import TEST_TODAY
from tests.graph_helpers import CLASSIFIED, ScriptedLLM, call_reply, text_reply

from support_agent.api import app, get_graph
from support_agent.decision_record import DecisionRecord
from support_agent.graph import build_graph
from support_agent.prompts import prompt_versions
from support_agent.tools.policy_search import get_policy_version

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


def test_s001_fails_when_the_agent_searches_the_policies(api, search_embeds_like):
    search_embeds_like(CHUNK)
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


# --- the results table ---


def _result(**overrides: object) -> Result:
    fields = {
        "scenario_id": "S001",
        "title": "Leave balance query",
        "expected_outcome": "resolve",
        "actual_outcome": "resolve",
        "passed": True,
        "total_tokens": 1200,
        "total_cost_eur": 0.0123,
        "latency_ms": 4500,
    }
    return Result(**{**fields, **overrides})


def _header(**overrides: object) -> Header:
    fields = {
        "seed_as_of": date(2026, 11, 3),
        "index_build_id": "a1b2c3d4e5f6",
        "policy_version": "expense-policy@2026-09.1; it-access-policy@2026-11.1",
        "prompt_versions": {"agent_system": "1.3.0", "classifier": "1.0.0"},
    }
    return Header(**{**fields, **overrides})


def test_the_table_has_a_header_a_row_per_scenario_and_a_total():
    results = [
        _result(),
        _result(
            scenario_id="S003",
            title="Over balance",
            expected_outcome="refuse_with_citation",
            actual_outcome="resolve",
            passed=False,
            total_tokens=800,
            total_cost_eur=0.01,
            latency_ms=3000,
            failed_checks=["outcome", "must_cite"],
        ),
    ]

    table = format_results(_header(), results).splitlines()

    assert table[0] == "# Eval results"
    assert "- Seed as of: 2026-11-03" in table
    assert "- Index build id: a1b2c3d4e5f6" in table
    assert "- Policy versions: expense-policy@2026-09.1; it-access-policy@2026-11.1" in table
    assert "- Prompt versions: agent_system 1.3.0, classifier 1.0.0" in table
    assert "- Runs per scenario: 1" in table
    assert (
        "| S001 Leave balance query | resolve | resolve | yes | 1200 | 0.0123 | 4500 | - |" in table
    )
    assert (
        "| S003 Over balance | refuse_with_citation | resolve | no | 800 | 0.0100 | 3000 "
        "| outcome, must_cite |" in table
    )
    assert "1 of 2 passed. Tokens 2000, EUR 0.0223." in table


def test_the_summary_is_one_line_with_the_totals():
    results = [_result(), _result(scenario_id="S003", passed=False, total_tokens=800)]

    assert format_summary(results) == "1 of 2 passed. Tokens 2000, EUR 0.0246."


def test_a_run_file_is_named_after_the_time_of_the_run():
    path = get_run_path(datetime(2026, 11, 5, 9, 15, 0, tzinfo=UTC))

    assert path == run_eval.RUNS_DIR / "20261105T091500Z.md"


def test_the_table_says_unknown_for_a_header_value_it_does_not_have():
    header = _header(seed_as_of=None, index_build_id=None, policy_version=None, prompt_versions={})

    table = format_results(header, [_result()])

    assert "- Seed as of: unknown" in table
    assert "- Index build id: unknown" in table
    assert "- Policy versions: unknown" in table
    assert "- Prompt versions: unknown" in table


def test_a_result_takes_tokens_cost_latency_and_prompt_versions_from_the_record():
    record = _record(
        total_tokens=900,
        total_cost_eur=0.02,
        latency_ms=2500,
        prompt_versions={"agent_system": "1.3.0"},
    )
    checks = [Check(name="outcome", passed=True), Check(name="must_call", passed=False)]

    result = get_result(_scenario(), _run(record=record), checks)

    assert (result.total_tokens, result.total_cost_eur, result.latency_ms) == (900, 0.02, 2500)
    assert result.prompt_versions == {"agent_system": "1.3.0"}
    assert result.failed_checks == ["must_call"]
    assert not result.passed


def test_a_result_without_a_record_has_zero_cost():
    result = get_result(
        _scenario(), _run(record=None), [Check(name="decision_record", passed=False)]
    )

    assert (result.total_tokens, result.total_cost_eur, result.latency_ms) == (0, 0.0, 0)


def test_an_error_result_fails_and_names_the_request():
    result = get_error_result(_scenario())

    assert (result.actual_outcome, result.passed, result.failed_checks) == (
        "error",
        False,
        ["request"],
    )


def test_the_header_names_the_seed_the_index_and_the_policies(seeded_db):
    header = get_header([_result(prompt_versions={"agent_system": "1.3.0"})])

    assert header.seed_as_of == TEST_TODAY
    assert header.index_build_id == calculate_index_build_id(
        generate_policy_chunks(), "test-embedding"
    )
    assert header.policy_version == get_policy_version()
    assert header.policy_version.startswith("expense-policy@")
    assert header.prompt_versions == {"agent_system": "1.3.0"}


def test_the_header_falls_back_to_the_prompt_versions_on_disk(seeded_db):
    header = get_header([_result()])

    assert header.prompt_versions == prompt_versions()


def test_main_writes_the_table_and_returns_zero_when_every_scenario_passes(
    api, monkeypatch, tmp_path, capsys
):
    _script(api, CLASSIFIED, call_reply("c1", "get_leave_balance", "{}"), text_reply("18.0 left."))
    monkeypatch.setattr(run_eval, "get_client", lambda api_url: api)
    output = tmp_path / "results" / "latest.md"

    exit_code = main(["S001", "--output", str(output)])

    assert exit_code == 0
    table = output.read_text(encoding="utf-8")
    assert "| S001 Leave balance query | resolve | resolve | yes |" in table
    assert f"- Seed as of: {TEST_TODAY}" in table
    assert "1 of 1 passed" in capsys.readouterr().out


def test_main_writes_a_new_file_in_the_runs_folder_by_default(api, monkeypatch, tmp_path, capsys):
    _script(api, CLASSIFIED, call_reply("c1", "get_leave_balance", "{}"), text_reply("18.0 left."))
    monkeypatch.setattr(run_eval, "get_client", lambda api_url: api)
    monkeypatch.setattr(run_eval, "RUNS_DIR", tmp_path / ".runs")

    exit_code = main(["S001"])

    assert exit_code == 0
    [run] = (tmp_path / ".runs").glob("*.md")
    assert re.fullmatch(r"\d{8}T\d{6}Z", run.stem)
    assert "| S001 Leave balance query | resolve | resolve | yes |" in run.read_text("utf-8")
    assert f"1 of 1 passed. Tokens 0, EUR 0.0000. Results are in {run}" in capsys.readouterr().out


def test_main_returns_one_and_names_the_failed_checks_when_a_scenario_fails(
    api, search_embeds_like, monkeypatch, tmp_path, capsys
):
    search_embeds_like(CHUNK)
    _script(
        api,
        CLASSIFIED,
        call_reply("c1", "search_policies", '{"query": "vacation days"}'),
        call_reply("c2", "get_leave_balance", "{}"),
        text_reply("18.0 left."),
    )
    monkeypatch.setattr(run_eval, "get_client", lambda api_url: api)
    output = tmp_path / "latest.md"

    exit_code = main(["S001", "--output", str(output)])

    assert exit_code == 1
    assert "| no |" in output.read_text(encoding="utf-8")
    assert "FAIL S001" in capsys.readouterr().out
