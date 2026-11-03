"""Scenario evals. Each scenario is a YAML file in `eval/scenarios/` with the request, the
persona who sends it, and the outcome and behavior we expect. This module holds the
scenario schema, the loader and the runner. The runner sends each request to a running API,
reads back the Decision Record, and checks the run against the expectation. No model
judges anything: every check is a plain comparison.

    python -m eval.run_eval [--api-url http://localhost:8000] [S001 S002 ...]
"""

from __future__ import annotations

import argparse
import calendar
import re
from collections.abc import Iterator
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Self

import holidays
import httpx
import yaml
from langchain_core.utils.function_calling import convert_to_openai_function
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from support_agent import util
from support_agent.decision_record import DecisionRecord
from support_agent.feedback import Rating
from support_agent.state import Outcome
from support_agent.tools import TOOLS, is_gated_write

SCENARIOS_DIR = Path(__file__).resolve().parent / "scenarios"

# The employee who reads the Decision Records. Only an HR partner can.
RECORD_READER = "emp_005"

# Parameter names that could name an employee. The structural test in
# `tests/test_tool_contracts.py` has the full list; this catches the same kind of name.
IDENTITY_NAME_RE = re.compile(
    r"employee|empid|user|person|staff|email|behalf|requester|approver|impersonat|actor|owner"
)

# `<policy>#<heading-slug>#<n>`, for example `vacation-policy#entitlement#0`
CHUNK_ID_RE = re.compile(r"^[a-z0-9-]+#[a-z0-9-]+#\d+$")
SCENARIO_ID_RE = re.compile(r"^S\d{3}$")
PERSONA_RE = re.compile(r"^emp_\d{3}$")

# A request can name a date relative to today: `{monday_in_3_weeks}` is the Monday of the
# first full working week that starts at least 3 weeks after today and has no German public
# holiday. `{friday_in_3_weeks}` is the Friday of that same week. This keeps a scenario from
# going out of date, and a week with no holiday has the same number of working days in any
# year. Such a week never spans two years, because it would contain January 1.
DATE_TOKEN_RE = re.compile(r"\{(monday|friday)_in_(\d+)_weeks\}")
ANY_TOKEN_RE = re.compile(r"\{[^{}]*\}")

TOOL_NAMES = {tool.__name__ for tool in TOOLS}
WRITE_TOOL_NAMES = {tool.__name__ for tool in TOOLS if is_gated_write(tool)}


class Expected(BaseModel):
    """What a run must do for the scenario to pass."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    outcome: Outcome
    # chunk ids the answer must cite
    must_cite: list[str] = []
    # tools the run must call, and tools it must not call
    must_call: list[str] = []
    must_not_call: list[str] = []
    # text the final answer must contain, and text it must not contain, compared without
    # regard to case
    answer_must_contain: list[str] = []
    answer_must_not_contain: list[str] = []
    # the only write tools the run may propose
    proposed_actions: list[str] = []
    # when true, no tool may have a parameter that names an employee
    no_identity_parameter: bool = False

    @field_validator("must_cite")
    @classmethod
    def _chunk_ids(cls, values: list[str]) -> list[str]:
        for value in values:
            if not CHUNK_ID_RE.match(value):
                raise ValueError(f"{value!r} is not a chunk id like vacation-policy#entitlement#0")
        return values

    @field_validator("must_call", "must_not_call", "proposed_actions")
    @classmethod
    def _tool_names(cls, values: list[str]) -> list[str]:
        unknown = sorted(set(values) - TOOL_NAMES)
        if unknown:
            raise ValueError(f"unknown tool: {', '.join(unknown)}")
        return values

    @field_validator("proposed_actions")
    @classmethod
    def _write_tools(cls, values: list[str]) -> list[str]:
        not_writes = sorted(set(values) - WRITE_TOOL_NAMES)
        if not_writes:
            raise ValueError(f"not a write tool: {', '.join(not_writes)}")
        return values

    @field_validator("must_cite", "must_call", "must_not_call", "proposed_actions")
    @classmethod
    def _no_duplicates(cls, values: list[str]) -> list[str]:
        if len(set(values)) != len(values):
            raise ValueError("must not list the same value twice")
        return values

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        both = sorted(set(self.must_call) & set(self.must_not_call))
        if both:
            raise ValueError(f"in both must_call and must_not_call: {', '.join(both)}")
        if self.outcome == "propose_action" and not self.proposed_actions:
            raise ValueError("outcome propose_action needs proposed_actions")
        if self.outcome == "refuse_with_citation" and not self.must_cite:
            raise ValueError("outcome refuse_with_citation needs must_cite")
        return self


class Feedback(BaseModel):
    """Feedback to send for the answer after the request has run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    rating: Rating
    comment: str | None = None


class Scenario(BaseModel):
    """One scenario file."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    title: str
    persona: str
    request: str
    expected: Expected
    # set only by a scenario that tests the feedback loop
    feedback: Feedback | None = None
    notes: str

    @field_validator("id")
    @classmethod
    def _id_format(cls, value: str) -> str:
        if not SCENARIO_ID_RE.match(value):
            raise ValueError(f"id must look like S001, got {value!r}")
        return value

    @field_validator("persona")
    @classmethod
    def _persona_format(cls, value: str) -> str:
        if not PERSONA_RE.match(value):
            raise ValueError(f"persona must look like emp_001, got {value!r}")
        return value

    @field_validator("request")
    @classmethod
    def _known_date_tokens(cls, value: str) -> str:
        unknown = [t for t in ANY_TOKEN_RE.findall(value) if not DATE_TOKEN_RE.fullmatch(t)]
        if unknown:
            raise ValueError(f"unknown date token: {', '.join(unknown)}")
        return value

    @field_validator("title", "request", "notes")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value


def get_working_week(weeks_ahead: int, today: date) -> tuple[date, date]:
    """The Monday and Friday of the first week that starts at least `weeks_ahead` weeks
    after `today` and has no German public holiday from Monday to Friday.
    """
    earliest = today + timedelta(weeks=weeks_ahead)
    monday = earliest + timedelta(days=(7 - earliest.weekday()) % 7)
    # national holidays only, the same as the leave tool counts them
    public_holidays = holidays.country_holidays("DE", years=range(today.year, today.year + 3))
    while any(monday + timedelta(days=n) in public_holidays for n in range(5)):
        monday += timedelta(weeks=1)
    return monday, monday + timedelta(days=4)


def render_request(request: str, today: date | None = None) -> str:
    """The request text with each date token replaced by a date, like "October 12, 2026".
    `today` is `util.get_today()` unless given.
    """
    today = today or util.get_today()

    def replace(match: re.Match[str]) -> str:
        weekday, weeks_ahead = match.group(1), int(match.group(2))
        monday, friday = get_working_week(weeks_ahead, today)
        day = monday if weekday == "monday" else friday
        return f"{calendar.month_name[day.month]} {day.day}, {day.year}"

    return DATE_TOKEN_RE.sub(replace, request)


def load_scenarios(directory: Path = SCENARIOS_DIR) -> dict[str, Scenario]:
    """Read every `S*.yaml` file in `directory`. The file name must start with the
    scenario id, so an id can only appear once.
    """
    scenarios: dict[str, Scenario] = {}
    for path in sorted(directory.glob("S*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"{path.name}: expected a mapping at the top level")
        scenario = Scenario(**data)
        if path.stem != scenario.id and not path.stem.startswith(f"{scenario.id}_"):
            raise ValueError(f"{path.name}: file name must start with the id {scenario.id!r}")
        if scenario.id in scenarios:
            raise ValueError(f"{path.name}: id {scenario.id!r} is used twice")
        scenarios[scenario.id] = scenario
    return scenarios


class Run(BaseModel):
    """What one run of a scenario produced."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    request_id: str
    answer: str
    outcome: Outcome
    record: DecisionRecord
    # the tools called, in order, from the evidence of the Decision Record
    tools_called: list[str]
    # the write tools the request proposed, from the caller's own actions
    proposed_actions: list[str]
    # set when the scenario sent feedback and the API saved it
    feedback_id: int | None = None


class Check(BaseModel):
    """One comparison between the expectation and the run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    passed: bool
    detail: str = ""


def run_scenario(client: httpx.Client, scenario: Scenario, today: date | None = None) -> Run:
    """Send the request as the persona, then read back the Decision Record and the actions
    it proposed. Raises httpx.HTTPStatusError if the API answers with an error.
    """
    as_persona = {"X-Employee-Id": scenario.persona}
    chat = client.post(
        "/chat", headers=as_persona, json={"message": render_request(scenario.request, today)}
    )
    chat.raise_for_status()
    answer = chat.json()
    request_id = answer["request_id"]

    response = client.get(f"/decisions/{request_id}", headers={"X-Employee-Id": RECORD_READER})
    response.raise_for_status()
    record = DecisionRecord.model_validate(response.json())

    response = client.get("/me/actions", headers=as_persona, params={"limit": 200})
    response.raise_for_status()
    proposed = [a["tool"] for a in response.json() if a["request_id"] == request_id]

    feedback_id = None
    if scenario.feedback:
        response = client.post(
            f"/feedback/{request_id}",
            headers=as_persona,
            json=scenario.feedback.model_dump(exclude_none=True),
        )
        response.raise_for_status()
        saved = response.json()
        if saved["request_id"] == request_id and saved["rating"] == scenario.feedback.rating:
            feedback_id = saved["id"]

    return Run(
        request_id=request_id,
        answer=answer["answer"],
        outcome=answer["outcome"],
        record=record,
        tools_called=[i["tool"] for i in record.evidence if i["type"] == "tool_result"],
        proposed_actions=proposed,
        feedback_id=feedback_id,
    )


def _property_names(schema: Any) -> Iterator[str]:
    """Every property name in a JSON schema, including nested objects."""
    if isinstance(schema, dict):
        properties = schema.get("properties")
        if isinstance(properties, dict):
            yield from properties
        for value in schema.values():
            yield from _property_names(value)
    elif isinstance(schema, list):
        for item in schema:
            yield from _property_names(item)


def get_identity_parameters() -> list[str]:
    """The tool parameters, as `tool.parameter`, that could name an employee. These are
    read from the same schemas the agent node sends to the model. The list is empty when
    the model has no way to say whose data to use.
    """
    found = []
    for tool in TOOLS:
        schema = convert_to_openai_function(tool)["parameters"]
        for name in _property_names(schema):
            if IDENTITY_NAME_RE.search(re.sub(r"[^a-z0-9]", "", name.lower())):
                found.append(f"{tool.__name__}.{name}")
    return found


def _check(name: str, problems: list[str]) -> Check:
    return Check(name=name, passed=not problems, detail="; ".join(problems))


def check_run(scenario: Scenario, run: Run) -> list[Check]:
    """Compare the run with the expectation. A check is added only when the scenario asks
    for it, except `outcome` and `proposed_actions`, which every run is checked on.
    """
    expected = scenario.expected
    answer = run.answer.lower()

    outcome = (
        []
        if run.outcome == expected.outcome
        else [f"expected {expected.outcome}, got {run.outcome}"]
    )
    # Action safety: the run may propose only the write tools the scenario lists
    unexpected = [t for t in run.proposed_actions if t not in expected.proposed_actions]
    checks = [
        _check("outcome", outcome),
        _check("proposed_actions", [f"proposed {t}, which is not allowed" for t in unexpected]),
    ]
    if expected.must_call:
        checks.append(
            _check(
                "must_call",
                [f"{t} was not called" for t in expected.must_call if t not in run.tools_called],
            )
        )
    if expected.must_not_call:
        checks.append(
            _check(
                "must_not_call",
                [f"{t} was called" for t in expected.must_not_call if t in run.tools_called],
            )
        )
    if expected.answer_must_contain:
        checks.append(
            _check(
                "answer_must_contain",
                [
                    f"the answer lacks {t!r}"
                    for t in expected.answer_must_contain
                    if t.lower() not in answer
                ],
            )
        )
    if expected.answer_must_not_contain:
        checks.append(
            _check(
                "answer_must_not_contain",
                [
                    f"the answer has {t!r}"
                    for t in expected.answer_must_not_contain
                    if t.lower() in answer
                ],
            )
        )
    if expected.no_identity_parameter:
        checks.append(
            _check(
                "no_identity_parameter",
                [f"{name} could name an employee" for name in get_identity_parameters()],
            )
        )
    if scenario.feedback:
        checks.append(
            _check(
                "feedback_saved",
                [] if run.feedback_id is not None else ["the API did not save the feedback"],
            )
        )
    return checks


def main(argv: list[str] | None = None) -> int:
    """Run the scenarios against the API and print one result per scenario. Returns 1 if
    any scenario failed.
    """
    parser = argparse.ArgumentParser(description="Run the scenarios against a running API.")
    parser.add_argument("scenario_ids", nargs="*", help="run only these, for example S001")
    parser.add_argument("--api-url", default="http://localhost:8000")
    args = parser.parse_args(argv)

    scenarios = load_scenarios()
    unknown = [s for s in args.scenario_ids if s not in scenarios]
    if unknown:
        parser.error(f"unknown scenario: {', '.join(unknown)}")

    failed = 0
    # a model run can take a while
    with httpx.Client(base_url=args.api_url, timeout=180) as client:
        for scenario_id in args.scenario_ids or scenarios:
            scenario = scenarios[scenario_id]
            try:
                checks = check_run(scenario, run_scenario(client, scenario))
            except httpx.HTTPError as error:
                print(f"ERROR {scenario_id} {scenario.title}\n  - {error}")
                failed += 1
                continue
            problems = [c for c in checks if not c.passed]
            print(f"{'FAIL' if problems else 'PASS'} {scenario_id} {scenario.title}")
            for check in problems:
                print(f"  - {check.name}: {check.detail}")
            failed += bool(problems)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
