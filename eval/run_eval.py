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
from collections import Counter
from collections.abc import Iterator
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Self

import holidays
import httpx
import yaml
from data.generate import get_seed_as_of
from langchain_core.utils.function_calling import convert_to_openai_function
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from support_agent import db, util
from support_agent.decision_record import DecisionRecord
from support_agent.feedback import Rating
from support_agent.guardrails.verifier import CITATIONS_CHECK, parse_citations
from support_agent.prompts import prompt_versions
from support_agent.state import Outcome
from support_agent.tools import TOOLS, is_gated_write
from support_agent.tools.policy_search import get_index_build_id, get_policy_version

SCENARIOS_DIR = Path(__file__).resolve().parent / "scenarios"
# Every run writes its table here, in a file named after the time of the run. This folder
# is not committed. `eval/publish.py` copies one run to RESULTS_DIR, which is committed.
RUNS_DIR = Path(__file__).resolve().parent / ".runs"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

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
    # None when the API has no Decision Record for the request
    record: DecisionRecord | None
    # the tools called, in order, from the audit log. The record is checked against it.
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


def get_audited_tool_calls(request_id: str) -> list[str]:
    """The tools the request called, in order, from the `tool_call` events of the audit
    log. This reads the database that the API writes to, so it needs `DATABASE_URL`.
    """
    rows = db.fetch_all(
        """
        SELECT payload->>'tool' AS tool
        FROM audit_log
        WHERE request_id = %s AND event = 'tool_call'
        ORDER BY id
        """,
        (request_id,),
    )
    return [row["tool"] for row in rows]


def run_scenario(client: httpx.Client, scenario: Scenario, today: date | None = None) -> Run:
    """Send the request as the persona, then read back the Decision Record, the audit log
    and the actions it proposed. Raises httpx.HTTPStatusError if the API answers with an
    error. A missing Decision Record is not an error here: the run has no record, and the
    `decision_record` check fails.
    """
    as_persona = {"X-Employee-Id": scenario.persona}
    chat = client.post(
        "/chat", headers=as_persona, json={"message": render_request(scenario.request, today)}
    )
    chat.raise_for_status()
    answer = chat.json()
    request_id = answer["request_id"]

    response = client.get(f"/decisions/{request_id}", headers={"X-Employee-Id": RECORD_READER})
    record = None
    if response.status_code != 404:
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
        tools_called=get_audited_tool_calls(request_id),
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


def get_retrieved_chunk_ids(record: DecisionRecord) -> set[str]:
    """The chunk ids that the `search_policies` calls of the request returned, from the
    evidence of the record.
    """
    ids: set[str] = set()
    for item in record.evidence:
        is_search = item["type"] == "tool_result" and item["tool"] == "search_policies"
        if is_search and isinstance(item["ref"], list):
            ids.update(chunk["id"] for chunk in item["ref"])
    return ids


def _citation_problems(run: Run) -> list[str]:
    """Every citation in the answer, and in the record, must be a chunk that the request
    retrieved. The runner checks this again, apart from the verifier in the graph.
    """
    retrieved = get_retrieved_chunk_ids(run.record) if run.record else set()
    problems = [
        f"the answer cites {chunk_id}, which the request did not retrieve"
        for chunk_id in parse_citations(run.answer)
        if chunk_id not in retrieved
    ]
    if run.record:
        problems += [
            f"the record cites {chunk_id}, which the request did not retrieve"
            for chunk_id in run.record.citations
            if chunk_id not in retrieved
        ]
    return problems


def _record_problems(run: Run) -> list[str]:
    """The record must exist, say why in a sentence, have evidence, agree with the answer
    on the outcome, and show every tool call that the audit log has.
    """
    record = run.record
    if record is None:
        return ["no Decision Record was saved"]
    problems = []
    if not record.summary.strip():
        problems.append("the summary is empty")
    if not record.evidence:
        problems.append("the evidence is empty")
    if record.outcome != run.outcome:
        problems.append(f"the record has outcome {record.outcome}, the answer {run.outcome}")
    in_evidence = Counter(i["tool"] for i in record.evidence if i["type"] == "tool_result")
    audited = Counter(run.tools_called)
    problems += [
        f"{tool} was called, but the evidence does not show it"
        for tool in sorted(audited - in_evidence)
    ]
    problems += [
        f"the evidence shows {tool}, but the audit log does not"
        for tool in sorted(in_evidence - audited)
    ]
    return problems


def _check(name: str, problems: list[str]) -> Check:
    return Check(name=name, passed=not problems, detail="; ".join(problems))


def check_run(scenario: Scenario, run: Run) -> list[Check]:
    """Compare the run with the expectation. A check is added only when the scenario asks
    for it, except `outcome`, `proposed_actions`, the citations and the Decision Record,
    which every run is checked on.
    """
    expected = scenario.expected
    answer = run.answer.lower()
    cited = parse_citations(run.answer)

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
        _check(CITATIONS_CHECK, _citation_problems(run)),
        _check("decision_record", _record_problems(run)),
    ]
    if expected.must_cite:
        checks.append(
            _check(
                "must_cite",
                [f"the answer does not cite {c}" for c in expected.must_cite if c not in cited],
            )
        )
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


class Result(BaseModel):
    """One row of the results table."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    scenario_id: str
    title: str
    expected_outcome: str
    # "error" when the API could not run the request
    actual_outcome: str
    passed: bool
    # tokens, cost and latency come from the Decision Record
    total_tokens: int = 0
    total_cost_eur: float = 0.0
    latency_ms: int = 0
    failed_checks: list[str] = []
    prompt_versions: dict[str, str] = {}


def get_result(scenario: Scenario, run: Run, checks: list[Check]) -> Result:
    failed = [c.name for c in checks if not c.passed]
    record = run.record
    return Result(
        scenario_id=scenario.id,
        title=scenario.title,
        expected_outcome=scenario.expected.outcome,
        actual_outcome=run.outcome,
        passed=not failed,
        total_tokens=record.total_tokens if record else 0,
        total_cost_eur=record.total_cost_eur if record else 0.0,
        latency_ms=record.latency_ms if record else 0,
        failed_checks=failed,
        prompt_versions=record.prompt_versions if record else {},
    )


def get_error_result(scenario: Scenario) -> Result:
    return Result(
        scenario_id=scenario.id,
        title=scenario.title,
        expected_outcome=scenario.expected.outcome,
        actual_outcome="error",
        passed=False,
        failed_checks=["request"],
    )


class Header(BaseModel):
    """What the results were measured on. Two result files are comparable only when these
    match.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # None when the database was not seeded
    seed_as_of: date | None
    index_build_id: str | None
    policy_version: str | None
    prompt_versions: dict[str, str]


def get_header(results: list[Result]) -> Header:
    """The header for these results. The seed and the policies are read from the
    database. The prompt versions are the ones the records name, so they are the versions
    that ran. Without a record that names one, they are the versions on disk.
    """
    versions: dict[str, str] = {}
    for result in results:
        versions.update(result.prompt_versions)
    return Header(
        seed_as_of=get_seed_as_of(),
        index_build_id=get_index_build_id(),
        policy_version=get_policy_version(),
        prompt_versions=versions or prompt_versions(),
    )


def format_results(header: Header, results: list[Result]) -> str:
    """The results as markdown: the header, one table row per scenario, and a total."""
    unknown = "unknown"
    versions = ", ".join(f"{name} {version}" for name, version in header.prompt_versions.items())
    lines = [
        "# Eval results",
        "",
        f"- Seed as of: {header.seed_as_of or unknown}",
        f"- Index build id: {header.index_build_id or unknown}",
        f"- Policy versions: {header.policy_version or unknown}",
        f"- Prompt versions: {versions or unknown}",
        "- Runs per scenario: 1",
        "",
        "| Scenario | Expected outcome | Actual outcome | Pass | Tokens | EUR | Latency ms |"
        " Failed checks |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r.scenario_id} {r.title} | {r.expected_outcome} | {r.actual_outcome} "
            f"| {'yes' if r.passed else 'no'} | {r.total_tokens} | {r.total_cost_eur:.4f} "
            f"| {r.latency_ms} | {', '.join(r.failed_checks) or '-'} |"
        )
    lines += ["", format_summary(results), ""]
    return "\n".join(lines)


def format_summary(results: list[Result]) -> str:
    """One line: how many scenarios passed, and the total tokens and cost."""
    passed = sum(r.passed for r in results)
    tokens = sum(r.total_tokens for r in results)
    cost = sum(r.total_cost_eur for r in results)
    return f"{passed} of {len(results)} passed. Tokens {tokens}, EUR {cost:.4f}."


def get_run_path(now: datetime | None = None) -> Path:
    """The file for a run that starts at `now` (UTC), for example `20261105T091500Z.md`.
    The name sorts by time, and it can be given to `make eval-publish`.
    """
    now = now or datetime.now(UTC)
    return RUNS_DIR / f"{now:%Y%m%dT%H%M%SZ}.md"


def get_client(api_url: str) -> httpx.Client:
    """The client for the API. A model run can take a while, so the timeout is long."""
    return httpx.Client(base_url=api_url, timeout=180)


def main(argv: list[str] | None = None) -> int:
    """Run the scenarios against the API, print one result per scenario and write the
    results table to a new file in eval/.runs/. Returns 1 if any scenario failed.
    """
    parser = argparse.ArgumentParser(description="Run the scenarios against a running API.")
    parser.add_argument("scenario_ids", nargs="*", help="run only these, for example S001")
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument(
        "--output", type=Path, help="the results file, by default a new file in eval/.runs/"
    )
    args = parser.parse_args(argv)

    scenarios = load_scenarios()
    unknown = [s for s in args.scenario_ids if s not in scenarios]
    if unknown:
        parser.error(f"unknown scenario: {', '.join(unknown)}")

    results: list[Result] = []
    with get_client(args.api_url) as client:
        for scenario_id in args.scenario_ids or scenarios:
            scenario = scenarios[scenario_id]
            try:
                run = run_scenario(client, scenario)
            except httpx.HTTPError as error:
                print(f"ERROR {scenario_id} {scenario.title}\n  - {error}")
                results.append(get_error_result(scenario))
                continue
            checks = check_run(scenario, run)
            results.append(get_result(scenario, run, checks))
            print(f"{'PASS' if results[-1].passed else 'FAIL'} {scenario_id} {scenario.title}")
            for check in checks:
                if not check.passed:
                    print(f"  - {check.name}: {check.detail}")

    output = args.output or get_run_path()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(format_results(get_header(results), results), encoding="utf-8")
    print(f"{format_summary(results)} Results are in {output}")
    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
