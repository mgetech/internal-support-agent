import re
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest
from data.generate import generate_employees, read_policies
from eval.run_eval import SCENARIOS_DIR, Scenario, load_scenarios, render_request
from pydantic import ValidationError

from support_agent import util
from support_agent.chunking import StructuralChunker
from support_agent.tools.hr_it import calculate_working_days


def _scenario(**expected: object) -> dict:
    return {
        "id": "S900",
        "title": "A test scenario",
        "persona": "emp_001",
        "request": "How many days do I have left?",
        "expected": {"outcome": "resolve", **expected},
        "notes": "why the expectation is what it is",
    }


def _write(directory: Path, name: str, body: str) -> None:
    (directory / name).write_text(body, encoding="utf-8")


VALID_FILE = """\
id: S900
title: A test scenario
persona: emp_001
request: How many days do I have left?
expected:
  outcome: resolve
notes: why
"""


def test_every_scenario_file_is_valid():
    # also checks the file name, the id and that no id repeats
    load_scenarios(SCENARIOS_DIR)


def test_expected_lists_default_to_empty():
    expected = Scenario(**_scenario()).expected

    assert expected.must_cite == []
    assert expected.must_call == []
    assert expected.must_not_call == []
    assert expected.answer_must_contain == []
    assert expected.answer_must_not_contain == []
    assert expected.proposed_actions == []
    assert expected.no_identity_parameter is False


def test_a_full_scenario_is_accepted():
    scenario = Scenario(
        **_scenario(
            outcome="refuse_with_citation",
            must_cite=["vacation-policy#entitlement#0"],
            must_call=["get_leave_balance"],
            must_not_call=["submit_leave_request"],
            answer_must_contain=["18"],
        )
    )

    assert scenario.expected.outcome == "refuse_with_citation"


def test_unknown_field_is_rejected():
    with pytest.raises(ValidationError):
        Scenario(**_scenario(), extra_field=1)
    with pytest.raises(ValidationError):
        Scenario(**_scenario(must_calls=["get_leave_balance"]))


def test_outcome_must_be_one_of_the_four():
    with pytest.raises(ValidationError):
        Scenario(**_scenario(outcome="approve"))


@pytest.mark.parametrize("field", ["must_call", "must_not_call", "proposed_actions"])
def test_unknown_tool_name_is_rejected(field):
    with pytest.raises(ValidationError, match="unknown tool"):
        Scenario(**_scenario(**{field: ["get_leave_balanse"]}))


def test_proposed_actions_must_be_write_tools():
    with pytest.raises(ValidationError, match="not a write tool"):
        Scenario(**_scenario(outcome="propose_action", proposed_actions=["get_leave_balance"]))


def test_chunk_id_must_have_the_standard_format():
    with pytest.raises(ValidationError, match="chunk id"):
        Scenario(**_scenario(must_cite=["vacation-policy"]))


def test_a_tool_cannot_be_both_required_and_forbidden():
    with pytest.raises(ValidationError, match="both must_call and must_not_call"):
        Scenario(
            **_scenario(must_call=["create_ticket"], must_not_call=["create_ticket"]),
        )


def test_a_value_cannot_be_listed_twice():
    with pytest.raises(ValidationError, match="twice"):
        Scenario(**_scenario(must_call=["get_leave_balance", "get_leave_balance"]))


def test_propose_action_needs_proposed_actions():
    with pytest.raises(ValidationError, match="needs proposed_actions"):
        Scenario(**_scenario(outcome="propose_action"))


def test_refuse_with_citation_needs_a_citation():
    with pytest.raises(ValidationError, match="needs must_cite"):
        Scenario(**_scenario(outcome="refuse_with_citation"))


def test_feedback_is_optional_and_checks_the_rating():
    assert Scenario(**_scenario()).feedback is None

    scenario = Scenario(**{**_scenario(), "feedback": {"rating": "down", "comment": "wrong"}})
    assert scenario.feedback.rating == "down"

    with pytest.raises(ValidationError):
        Scenario(**{**_scenario(), "feedback": {"rating": "meh"}})


@pytest.mark.parametrize("persona", ["emp_1", "alice", "EMP_001"])
def test_persona_must_look_like_an_employee_id(persona):
    with pytest.raises(ValidationError, match="persona"):
        Scenario(**{**_scenario(), "persona": persona})


@pytest.mark.parametrize("scenario_id", ["S1", "s001", "S0001", "001"])
def test_id_must_look_like_s001(scenario_id):
    with pytest.raises(ValidationError, match="id must look like"):
        Scenario(**{**_scenario(), "id": scenario_id})


@pytest.mark.parametrize("field", ["title", "request", "notes"])
def test_text_fields_must_not_be_blank(field):
    with pytest.raises(ValidationError, match="blank"):
        Scenario(**{**_scenario(), field: "  "})


def test_the_m0_roster_is_complete():
    assert set(load_scenarios()) == {f"S00{n}" for n in range(1, 7)}


def test_every_persona_is_in_the_seed_data():
    employee_ids = {employee["id"] for employee in generate_employees()}

    for scenario in load_scenarios().values():
        assert scenario.persona in employee_ids, scenario.id


def test_every_cited_chunk_exists_in_the_policy_corpus():
    chunker = StructuralChunker()
    chunk_ids = {
        chunk.id
        for policy_id, policy in read_policies().items()
        for chunk in chunker.chunk(policy_id, policy.text)
    }

    for scenario in load_scenarios().values():
        missing = set(scenario.expected.must_cite) - chunk_ids
        assert not missing, f"{scenario.id}: {sorted(missing)}"


def test_date_tokens_become_a_working_week_at_least_n_weeks_ahead():
    # 2026-09-22 is a Tuesday. Three weeks later is Tuesday October 13, so the first
    # Monday that is at least 3 weeks ahead is October 19.
    today = date(2026, 9, 22)

    assert (
        render_request("{monday_in_3_weeks} to {friday_in_3_weeks}", today)
        == "October 19, 2026 to October 23, 2026"
    )


def test_date_tokens_skip_a_week_with_a_public_holiday():
    # Good Friday is March 26 2027 and Easter Monday is March 29, so the weeks of
    # March 22 and March 29 are skipped
    today = date(2027, 3, 1)

    assert (
        render_request("{monday_in_3_weeks} to {friday_in_3_weeks}", today)
        == "April 5, 2027 to April 9, 2027"
    )


def test_date_tokens_follow_the_clock(monkeypatch):
    monkeypatch.setattr(util, "get_today", lambda: date(2026, 12, 30))

    assert render_request("{monday_in_1_weeks}") == "January 11, 2027"


def test_a_request_without_tokens_is_unchanged():
    assert render_request("How many days do I have left?", date(2026, 9, 22)) == (
        "How many days do I have left?"
    )


def test_an_unknown_date_token_is_rejected():
    with pytest.raises(ValidationError, match="unknown date token"):
        Scenario(**{**_scenario(), "request": "Book {tuesday_in_3_weeks}"})


def test_the_leave_scenarios_have_five_working_days_on_any_day_of_the_year():
    scenarios = load_scenarios()
    pattern = re.compile(r"(\w+ \d+, \d{4}) to (\w+ \d+, \d{4})")

    for scenario_id in ("S002", "S003"):
        day = date(2026, 1, 1)
        while day <= date(2030, 12, 31):
            request = render_request(scenarios[scenario_id].request, day)
            first, last = (
                datetime.strptime(text, "%B %d, %Y").date()
                for text in pattern.search(request).groups()
            )
            # a week with 5 working days, in one year, at least 3 weeks away
            assert calculate_working_days(first, last) == 5, (scenario_id, day)
            assert first.year == last.year, (scenario_id, day)
            assert (first - day).days >= 21, (scenario_id, day)
            day += timedelta(days=1)


def test_loader_reads_files_by_id(tmp_path):
    _write(tmp_path, "S900_balance.yaml", VALID_FILE)
    _write(tmp_path, "S901_other.yaml", VALID_FILE.replace("S900", "S901"))

    assert set(load_scenarios(tmp_path)) == {"S900", "S901"}


def test_loader_rejects_a_file_name_that_does_not_match_the_id(tmp_path):
    _write(tmp_path, "S901_balance.yaml", VALID_FILE)

    with pytest.raises(ValueError, match="file name must start with the id"):
        load_scenarios(tmp_path)


def test_loader_rejects_a_repeated_id(tmp_path):
    _write(tmp_path, "S900_a.yaml", VALID_FILE)
    _write(tmp_path, "S900_b.yaml", VALID_FILE)

    with pytest.raises(ValueError, match="used twice"):
        load_scenarios(tmp_path)


def test_loader_rejects_a_file_that_is_not_a_mapping(tmp_path):
    _write(tmp_path, "S900_a.yaml", "- just\n- a list\n")

    with pytest.raises(ValueError, match="mapping"):
        load_scenarios(tmp_path)
