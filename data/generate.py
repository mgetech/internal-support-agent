"""Deterministic synthetic data: employees, leave balances, IT outages, plus
reading the hand-authored HR/IT policy corpus and turning it into a seed.

No randomness for the employee data — every row is planted, so two runs are
byte-identical without needing an RNG seed. Entitlement is computed from the
vacation policy rule (pre-/post-2024 contracts, part-time proration,
first-year probation accrual) against a fixed reference date, never
`date.today()`, so the output never drifts with wall-clock time.

The policy corpus itself (`data/policies/*.md`) is the source of truth, not
generated — adding or editing a policy means editing its markdown file, the
same way any document-ingestion pipeline picks up whatever's on disk.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import yaml

from support_agent.chunking import StructuralChunker
from support_agent.embeddings import embed_texts

YEAR = 2026
POLICIES_DIR = Path(__file__).resolve().parent / "policies"
SEED_SQL_PATH = Path(__file__).resolve().parent / "seed.sql"
CACHE_DIR = Path(__file__).resolve().parent.parent / ".embeddings-cache"
EMBEDDING_MODEL = "text-embedding-3-small"

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


# the planted edge cases: emp_001 happy path, emp_003 part-time proration,
# emp_004 over-balance, emp_005/emp_010 authorized approver roles, emp_006
# first-year probation accrual, emp_012 zero remaining. The rest fill out
# the roster of 12 with plausible history.
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
class PolicyFile:
    id: str
    domain: str
    policy_version: str
    text: str  # markdown body, frontmatter stripped


_FRONTMATTER_RE = re.compile(r"\A---\n(.*?\n)---\n", re.DOTALL)


def _split_frontmatter(raw: str) -> tuple[dict, str]:
    match = _FRONTMATTER_RE.match(raw)
    if not match:
        raise ValueError("policy file is missing its frontmatter block")
    meta = yaml.safe_load(match.group(1))
    return meta, raw[match.end() :]


def read_policies(policies_dir: Path = POLICIES_DIR) -> dict[str, PolicyFile]:
    """Read every `*.md` file in `policies_dir` — the hand-authored source of
    truth for the corpus. Adding a policy means adding a file here, not
    editing this module.
    """
    policies = {}
    for path in sorted(policies_dir.glob("*.md")):
        meta, body = _split_frontmatter(path.read_text(encoding="utf-8"))
        policies[path.stem] = PolicyFile(
            id=path.stem,
            domain=meta["domain"],
            policy_version=meta["policy_version"],
            text=body,
        )
    return policies


def generate_policy_chunks() -> list[dict]:
    """One row per chunk, ready for the `policy_chunks` table — everything
    except the embedding, which is attached separately (see
    `embed_policy_chunks`) rather than computed as part of chunking.
    """
    chunker = StructuralChunker()
    rows = []
    for policy in read_policies().values():
        for chunk in chunker.chunk(policy.id, policy.text):
            rows.append(
                {
                    "id": chunk.id,
                    "policy": policy.id,
                    "heading": chunk.heading,
                    "content": chunk.content,
                    "domain": policy.domain,
                    "chunker": chunk.meta["chunker"],
                }
            )
    return rows


EmbedFn = Callable[[list[str]], list[list[float]]]


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _cache_path(cache_dir: Path, model: str) -> Path:
    return cache_dir / f"{model}.json"


def _load_cache(cache_dir: Path, model: str) -> dict[str, list[float]]:
    path = _cache_path(cache_dir, model)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_cache(cache_dir: Path, model: str, cache: dict[str, list[float]]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    _cache_path(cache_dir, model).write_text(json.dumps(cache), encoding="utf-8")


def embed_policy_chunks(
    rows: list[dict] | None = None,
    embed_fn: EmbedFn = embed_texts,
    model: str = EMBEDDING_MODEL,
    cache_dir: Path = CACHE_DIR,
) -> dict[str, list[float]]:
    """Chunk id -> embedding vector. Only chunks whose content hash is
    missing from the on-disk cache are sent to `embed_fn`, so regenerating
    is free once every current chunk has been embedded once.
    """
    rows = generate_policy_chunks() if rows is None else rows
    cache = _load_cache(cache_dir, model)

    missing = {_content_hash(row["content"]): row["content"] for row in rows}
    missing = {h: t for h, t in missing.items() if h not in cache}

    if missing:
        hashes = list(missing)
        vectors = embed_fn([missing[h] for h in hashes])
        cache.update(zip(hashes, vectors, strict=True))
        _save_cache(cache_dir, model, cache)

    return {row["id"]: cache[_content_hash(row["content"])] for row in rows}


_EMPLOYEE_COLUMNS = ["id", "name", "role", "employment_type", "weekly_hours", "country", "hired_at"]
_LEAVE_BALANCE_COLUMNS = ["employee_id", "year", "entitlement_days", "taken_days", "pending_days"]
_OUTAGE_COLUMNS = ["system", "status", "started_at", "note"]
_POLICY_CHUNK_COLUMNS = ["id", "policy", "heading", "content", "domain", "chunker"]


def _sql_literal(value: object) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, int | float):
        return str(value)
    if isinstance(value, date):
        return "'" + value.isoformat() + "'"
    return "'" + str(value).replace("'", "''") + "'"


def _insert(table: str, columns: list[str], rows: list[dict]) -> str:
    if not rows:
        return ""
    values = ",\n".join(
        "    (" + ", ".join(_sql_literal(row[c]) for c in columns) + ")" for row in rows
    )
    return f"INSERT INTO {table} ({', '.join(columns)}) VALUES\n{values};\n"


def render_seed_sql() -> str:
    """The full seed as INSERT statements. `policy_chunks` rows carry every
    column except `embedding` — vectors never enter this file; they're
    attached from the on-disk cache when the seed is loaded into Postgres.
    """
    parts = [
        "-- generated by data/generate.py — do not edit by hand\n",
        _insert("employees", _EMPLOYEE_COLUMNS, generate_employees()),
        _insert("leave_balances", _LEAVE_BALANCE_COLUMNS, generate_leave_balances()),
        _insert("known_outages", _OUTAGE_COLUMNS, generate_known_outages()),
        _insert("policy_chunks", _POLICY_CHUNK_COLUMNS, generate_policy_chunks()),
    ]
    return "\n".join(part for part in parts if part)


def write_seed_sql(output_path: Path = SEED_SQL_PATH) -> None:
    output_path.write_text(render_seed_sql(), encoding="utf-8", newline="\n")
