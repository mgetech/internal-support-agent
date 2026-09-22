"""Deterministic synthetic data: employees, leave balances, IT outages and the
HR/IT policy corpus.

No randomness — every row is planted data, so two runs are byte-identical
without needing an RNG seed. Entitlement is computed from the vacation policy
rule (pre-/post-2024 contracts, part-time proration, first-year probation
accrual) against a fixed reference date, never `date.today()`, so the output
never drifts with wall-clock time.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

YEAR = 2026
POLICY_VERSION = "2026-09.1"
POLICIES_DIR = Path(__file__).resolve().parent / "policies"

# the date this dataset is generated "as of" — fixed, not date.today(), so
# entitlement accrual (which depends on tenure) is stable across runs
GENERATED_AS_OF = date(2026, 9, 22)

FULL_TIME_ENTITLEMENT_PRE_2024 = 30.0
FULL_TIME_ENTITLEMENT_POST_2024 = 28.0
REFERENCE_WEEKLY_HOURS = 40.0
PROBATION_MONTHLY_ACCRUAL = 2.5
PROBATION_MONTHS = 12


@dataclass(frozen=True)
class _EmployeeProfile:
    id: str
    name: str
    role: str
    employment_type: str
    weekly_hours: float
    country: str
    hired_at: date
    taken_days: float
    pending_days: float = 0.0


# the planted edge cases (§7 of the spec): emp_001 happy path, emp_003
# part-time proration, emp_004 over-balance, emp_005/emp_010 authorized
# approver roles, emp_006 first-year probation accrual, emp_012 zero
# remaining. The rest fill out the roster of 12 with plausible history.
# (id, name, role, employment_type, weekly_hours, country, hired_at, taken_days)
_PROFILE_ROWS: list[tuple[str, str, str, str, float, str, date, float]] = [
    ("emp_001", "Anna Keller", "employee", "full_time", 40.0, "DE", date(2021, 3, 1), 12.0),
    ("emp_002", "Jonas Vogel", "employee", "full_time", 40.0, "DE", date(2022, 6, 15), 5.0),
    ("emp_003", "Mia Schuster", "employee", "part_time", 20.0, "DE", date(2023, 9, 1), 2.0),
    ("emp_004", "Paul Brandt", "employee", "full_time", 40.0, "DE", date(2020, 1, 10), 26.5),
    ("emp_005", "Laura Fink", "hr_partner", "full_time", 40.0, "DE", date(2019, 11, 1), 8.0),
    ("emp_006", "Elif Yildiz", "employee", "full_time", 40.0, "DE", date(2026, 7, 1), 0.0),
    ("emp_007", "Tobias Wagner", "employee", "full_time", 40.0, "DE", date(2022, 2, 1), 10.0),
    ("emp_008", "Sara Nowak", "employee", "full_time", 40.0, "DE", date(2021, 8, 15), 20.0),
    ("emp_009", "David Hoffmann", "employee", "full_time", 40.0, "DE", date(2024, 3, 1), 4.0),
    ("emp_010", "Nora Bergmann", "manager", "full_time", 40.0, "DE", date(2018, 5, 20), 15.0),
    ("emp_011", "Felix Lorenz", "employee", "full_time", 40.0, "DE", date(2023, 1, 5), 7.0),
    ("emp_012", "Katrin Sommer", "employee", "full_time", 40.0, "DE", date(2017, 4, 1), 30.0),
]
_PROFILES = [_EmployeeProfile(*row) for row in _PROFILE_ROWS]


def _months_employed(hired_at: date, as_of: date) -> int:
    months = (as_of.year - hired_at.year) * 12 + (as_of.month - hired_at.month)
    if as_of.day < hired_at.day:
        months -= 1
    return max(months, 0)


def _entitlement_days(hired_at: date, employment_type: str, weekly_hours: float) -> float:
    """Vacation-policy entitlement: pro-rata during the first year, else the
    flat full-time rate (post-2024 contracts get 28, earlier ones 30),
    prorated by weekly hours for part-time, rounded to the nearest half day.
    """
    months_employed = _months_employed(hired_at, GENERATED_AS_OF)
    if months_employed < PROBATION_MONTHS:
        base = months_employed * PROBATION_MONTHLY_ACCRUAL
    elif hired_at.year >= 2024:
        base = FULL_TIME_ENTITLEMENT_POST_2024
    else:
        base = FULL_TIME_ENTITLEMENT_PRE_2024

    if employment_type == "part_time":
        base = round(base * weekly_hours / REFERENCE_WEEKLY_HOURS * 2) / 2
    return base


def generate_employees() -> list[dict]:
    return [
        {
            "id": p.id,
            "name": p.name,
            "role": p.role,
            "employment_type": p.employment_type,
            "weekly_hours": p.weekly_hours,
            "country": p.country,
            "hired_at": p.hired_at,
        }
        for p in _PROFILES
    ]


def generate_leave_balances(year: int = YEAR) -> list[dict]:
    return [
        {
            "employee_id": p.id,
            "year": year,
            "entitlement_days": _entitlement_days(p.hired_at, p.employment_type, p.weekly_hours),
            "taken_days": p.taken_days,
            "pending_days": p.pending_days,
        }
        for p in _PROFILES
    ]


def generate_known_outages() -> list[dict]:
    return [
        {
            "system": "vpn",
            "status": "identified",
            "started_at": datetime(2026, 9, 20, 9, 15, tzinfo=UTC),
            "note": "certificate expiry on the VPN gateway; replacement certificate in progress",
        },
        {
            "system": "sso",
            "status": "resolved",
            "started_at": datetime(2026, 9, 10, 14, 0, tzinfo=UTC),
            "note": "identity provider outage; resolved after failover to the secondary node",
        },
    ]


@dataclass(frozen=True)
class _PolicyDoc:
    id: str
    domain: str
    title: str
    body: str


# escalation-only areas (marked in prose below) are what refusals and
# escalations cite: mid-year hours changes, sick leave beyond six weeks,
# parental-leave eligibility, and security incidents.
_POLICIES = [
    _PolicyDoc(
        id="vacation-policy",
        domain="hr",
        title="Vacation Policy",
        body="""\
## Entitlement

Full-time employees accrue 30 vacation days per calendar year. Employees whose
employment contract began in 2024 or later accrue 28 days per year instead. During
an employee's probationary period and for the remainder of their first year of
employment, vacation entitlement accrues pro-rata at 2.5 days per completed month
of service, rather than the full annual amount.

## Part-Time and Mid-Year Changes

Part-time entitlement is prorated against a 40-hour full-time reference week and
rounded to the nearest half day. An employee working 20 hours per week at the
standard 30-day full-time rate is entitled to 15 days per year.

Changes to an employee's contracted weekly hours partway through the year are not
handled through self-service. HR reviews each mid-year hours change individually
and recalculates entitlement case by case.

## Booking and Approval

Vacation requests require the requesting employee's manager to approve before the
days are booked. Requests covering ten or more consecutive working days require at
least two weeks' notice before the first day of leave.

## Carryover

Up to 5 unused vacation days may be carried over into the following calendar year.
Carried-over days must be used by March 31 of that year; any carryover remaining
after March 31 is forfeited.
""",
    ),
    _PolicyDoc(
        id="sick-leave-policy",
        domain="hr",
        title="Sick Leave Policy",
        body="""\
## Notification and Certification

Employees must notify their manager before the start of the working day on the
first day of a sickness absence. A medical certificate from a doctor is required
starting on the third consecutive day of absence.

## Pay Continuation

Statutory continued pay covers up to six consecutive weeks of sickness absence per
illness. Absences that extend beyond six weeks are not handled by this policy
directly — they must be referred to HR, which determines continued-pay eligibility
and any transition to statutory sick pay from the health insurer.

## Sickness During Vacation

An employee who falls sick during an approved vacation period and provides a
medical certificate covering the affected days has those days re-credited to their
vacation balance rather than counted as vacation taken.
""",
    ),
    _PolicyDoc(
        id="parental-leave-policy",
        domain="hr",
        title="Parental Leave Policy",
        body="""\
## Entitlement and Notice

Employees are entitled to take parental leave for up to three years per child.
Written notice of the intended start date must be given to HR at least seven weeks
in advance.

## Part-Time During Leave

Working part-time during parental leave is possible within statutory limits,
subject to agreement with the employer on the reduced hours and schedule.

## Eligibility and Benefit Interactions

Individual eligibility for parental leave, and how it interacts with Elterngeld
(state parental allowance) payments, depends on circumstances specific to each
employee. These determinations are not made by self-service tools and always
require review by HR.
""",
    ),
    _PolicyDoc(
        id="expense-policy",
        domain="hr",
        title="Expense Policy",
        body="""\
## Receipts and Submission

A receipt is required for any expense over €10. Expense claims must be submitted
within 60 days of the date the expense was incurred.

## Late Submissions

Claims submitted after the 60-day window are not reimbursed automatically. A late
submission requires approval from both the employee's manager and Finance before
it can be processed.
""",
    ),
    _PolicyDoc(
        id="it-access-policy",
        domain="it",
        title="IT Access Policy",
        body="""\
## VPN Access

VPN access is enabled by default for all employees. Before opening a ticket for a
VPN connectivity problem, check current known outages — if the issue matches an
active outage, no new ticket is needed; the employee should be informed of the
outage and its status instead.

## Password Self-Service

Password resets are handled through self-service via the identity portal. Support
tickets may be opened to verify an employee's identity when self-service fails,
but passwords are never handled, reset, or accepted in conversation by anyone
other than the employee themselves through the portal.

## Software Licenses

Software licenses costing more than €100 per year require the employee's manager
to approve the cost before purchase.

## Security Incidents

Suspected phishing, a lost or stolen device, or exposed credentials are security
incidents. These are always escalated immediately as a priority ticket; there is
no self-remediation step for a security incident, regardless of how minor it may
appear.
""",
    ),
]


def _render_policy_markdown(policy: _PolicyDoc) -> str:
    return f"# {policy.title}\n\n_Policy version: {POLICY_VERSION}_\n\n{policy.body}"


def generate_policies() -> dict[str, str]:
    """Policy id -> full markdown text, in corpus order."""
    return {p.id: _render_policy_markdown(p) for p in _POLICIES}


def write_policies(output_dir: Path = POLICIES_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for policy_id, text in generate_policies().items():
        (output_dir / f"{policy_id}.md").write_text(text, encoding="utf-8", newline="\n")
