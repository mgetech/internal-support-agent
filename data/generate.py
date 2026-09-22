"""Deterministic synthetic data: employees, leave balances and IT outages.

No randomness — every row is planted data, so two runs are byte-identical
without needing an RNG seed. Entitlement is computed from the vacation policy
rule (pre-/post-2024 contracts, part-time proration, first-year probation
accrual) against a fixed reference date, never `date.today()`, so the output
never drifts with wall-clock time.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime

YEAR = 2026

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
