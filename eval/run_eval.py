"""Scenario evals. Each scenario is a YAML file in `eval/scenarios/` with the request, the
persona who sends it, and the outcome and behavior we expect. This module holds the
scenario schema and the loader.
"""

from __future__ import annotations

import calendar
import re
from datetime import date, timedelta
from pathlib import Path
from typing import Self

import holidays
import yaml
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from support_agent import util
from support_agent.feedback import Rating
from support_agent.state import Outcome
from support_agent.tools import TOOLS, is_gated_write

SCENARIOS_DIR = Path(__file__).resolve().parent / "scenarios"

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
