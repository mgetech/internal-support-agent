"""Deterministic synthetic data: employees, leave balances, IT outages, plus
reading the hand-authored HR/IT policy corpus and turning it into a seed.

No randomness for the employee data — every row is planted, so two runs with the
same `as_of` date are byte-identical without needing an RNG seed. The date defaults
to today (`util.get_today`). Leave balances are written for the year of `as_of` and
the year after, and outage dates are set relative to `as_of`, so the data is current
in any year. Entitlement is computed from the vacation policy rule (pre-/post-2024
contracts, part-time proration, first-year probation accrual) as of that date.

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
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path

import psycopg
import yaml
from pgvector.psycopg import register_vector

from support_agent import util
from support_agent.chunking import StructuralChunker
from support_agent.config import get_settings
from support_agent.db import fetch_one, get_pool
from support_agent.embeddings import embed_texts

POLICIES_DIR = Path(__file__).resolve().parent / "policies"
SEED_SQL_PATH = Path(__file__).resolve().parent / "seed.sql"
CACHE_DIR = Path(__file__).resolve().parent.parent / ".embeddings-cache"

# emp_006 is hired this many days before `as_of`, but never before January 1 of its
# year, so the employee is in their first calendar year on any date
EMP_006_DAYS_EMPLOYED = 83

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
    approved_days: float = 0.0


def _resolve_as_of(as_of: date | None) -> date:
    return as_of or util.get_today()


# the planted edge cases: emp_001 happy path, emp_003 part-time proration,
# emp_004 over-balance, emp_005/emp_010 authorized approver roles, emp_006
# first-year probation accrual, emp_012 zero remaining. The rest fill out
# the roster of 12 with plausible history.
# (id, name, role, employment_type, weekly_hours, country, hired_at, taken_days)
# emp_006's hire date is None here: `_get_profiles` sets it relative to `as_of`.
_PROFILE_ROWS: list[tuple[str, str, str, str, float, str, date | None, float]] = [
    ("emp_001", "Anna Keller", "employee", "full_time", 40.0, "DE", date(2021, 3, 1), 12.0),
    ("emp_002", "Jonas Vogel", "employee", "full_time", 40.0, "DE", date(2022, 6, 15), 5.0),
    ("emp_003", "Mia Schuster", "employee", "part_time", 20.0, "DE", date(2023, 9, 1), 2.0),
    ("emp_004", "Paul Brandt", "employee", "full_time", 40.0, "DE", date(2020, 1, 10), 26.5),
    ("emp_005", "Laura Fink", "hr_partner", "full_time", 40.0, "DE", date(2019, 11, 1), 8.0),
    ("emp_006", "Elif Yildiz", "employee", "full_time", 40.0, "DE", None, 0.0),
    ("emp_007", "Tobias Wagner", "employee", "full_time", 40.0, "DE", date(2022, 2, 1), 10.0),
    ("emp_008", "Sara Nowak", "employee", "full_time", 40.0, "DE", date(2021, 8, 15), 20.0),
    ("emp_009", "David Hoffmann", "employee", "full_time", 40.0, "DE", date(2024, 3, 1), 4.0),
    ("emp_010", "Nora Bergmann", "manager", "full_time", 40.0, "DE", date(2018, 5, 20), 15.0),
    ("emp_011", "Felix Lorenz", "employee", "full_time", 40.0, "DE", date(2023, 1, 5), 7.0),
    ("emp_012", "Katrin Sommer", "employee", "full_time", 40.0, "DE", date(2017, 4, 1), 30.0),
]


def _get_profiles(as_of: date) -> list[_EmployeeProfile]:
    first_day_of_year = date(as_of.year, 1, 1)
    emp_006_hired_at = max(as_of - timedelta(days=EMP_006_DAYS_EMPLOYED), first_day_of_year)
    return [
        _EmployeeProfile(*row[:6], row[6] or emp_006_hired_at, *row[7:]) for row in _PROFILE_ROWS
    ]


def _months_employed(hired_at: date, as_of: date) -> int:
    months = (as_of.year - hired_at.year) * 12 + (as_of.month - hired_at.month)
    if as_of.day < hired_at.day:
        months -= 1
    return max(months, 0)


def _entitlement_days(
    hired_at: date, employment_type: str, weekly_hours: float, as_of: date
) -> float:
    """Vacation-policy entitlement: pro-rata during the first year, else the
    flat full-time rate (post-2024 contracts get 28, earlier ones 30),
    prorated by weekly hours for part-time, rounded to the nearest half day.
    """
    months_employed = _months_employed(hired_at, as_of)
    if months_employed < PROBATION_MONTHS:
        base = months_employed * PROBATION_MONTHLY_ACCRUAL
    elif hired_at.year >= 2024:
        base = FULL_TIME_ENTITLEMENT_POST_2024
    else:
        base = FULL_TIME_ENTITLEMENT_PRE_2024

    if employment_type == "part_time":
        base = round(base * weekly_hours / REFERENCE_WEEKLY_HOURS * 2) / 2
    return base


def generate_employees(as_of: date | None = None) -> list[dict]:
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
        for p in _get_profiles(_resolve_as_of(as_of))
    ]


def generate_leave_balances(as_of: date | None = None) -> list[dict]:
    """Balances for the year of `as_of` and the year after, with the same numbers."""
    as_of = _resolve_as_of(as_of)
    return [
        {
            "employee_id": p.id,
            "year": year,
            "entitlement_days": _entitlement_days(
                p.hired_at, p.employment_type, p.weekly_hours, as_of
            ),
            "taken_days": p.taken_days,
            "approved_days": p.approved_days,
        }
        for year in (as_of.year, as_of.year + 1)
        for p in _get_profiles(as_of)
    ]


def _days_before(as_of: date, days: int, at: time) -> datetime:
    return datetime.combine(as_of - timedelta(days=days), at, tzinfo=UTC)


def generate_known_outages(as_of: date | None = None) -> list[dict]:
    """The VPN outage is still open and started 2 days before `as_of`. The SSO outage
    was resolved and started 12 days before.
    """
    as_of = _resolve_as_of(as_of)
    return [
        {
            "system": "vpn",
            "status": "identified",
            "started_at": _days_before(as_of, 2, time(9, 15)),
            "note": "certificate expiry on the VPN gateway; replacement certificate in progress",
        },
        {
            "system": "sso",
            "status": "resolved",
            "started_at": _days_before(as_of, 12, time(14, 0)),
            "note": "identity provider outage; resolved after failover to the secondary node",
        },
    ]


def calculate_index_build_id(chunks: list[dict], embedding_model: str) -> str:
    """A short hash that names one build of the policy index: the sorted list of
    `policy@version`, the chunker name and the embedding model. The same inputs give the
    same id, and a change in any of them gives another.
    """
    inputs = {
        "policies": sorted({f"{chunk['policy']}@{chunk['policy_version']}" for chunk in chunks}),
        "chunkers": sorted({chunk["chunker"] for chunk in chunks}),
        "embedding_model": embedding_model,
    }
    digest = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()[:12]


def get_seed_as_of() -> date | None:
    """The `as_of` date of the data in the database, or None if it was not seeded."""
    row = fetch_one("SELECT as_of FROM seed_meta")
    return row["as_of"] if row else None


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
                    "policy_version": policy.policy_version,
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


def get_embedding_model() -> str:
    """The name the embedding cache is keyed on: the deployment that `embed_texts` calls."""
    return get_settings().azure_openai_deployment_embedding


def embed_policy_chunks(
    rows: list[dict] | None = None,
    embed_fn: EmbedFn = embed_texts,
    model: str | None = None,
    cache_dir: Path | None = None,
) -> dict[str, list[float]]:
    """Chunk id -> embedding vector. The cache is keyed on the model and on a hash of the
    chunk text, not on the chunk id: each model has its own cache file, and an entry is
    the hash of the text. Only chunks whose text hash is missing from the cache are sent
    to `embed_fn`, so regenerating is free once every current text has been embedded
    once, and an edited chunk is embedded again. `model` is the embedding deployment from
    the settings unless given.
    """
    rows = generate_policy_chunks() if rows is None else rows
    model = model or get_embedding_model()
    cache_dir = cache_dir or CACHE_DIR
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
_LEAVE_BALANCE_COLUMNS = ["employee_id", "year", "entitlement_days", "taken_days", "approved_days"]
_OUTAGE_COLUMNS = ["system", "status", "started_at", "note"]
_SEED_META_COLUMNS = ["as_of", "index_build_id"]
_POLICY_CHUNK_COLUMNS = [
    "id",
    "policy",
    "heading",
    "content",
    "domain",
    "chunker",
    "policy_version",
]


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


def render_seed_sql(as_of: date | None = None) -> str:
    """The full seed as INSERT statements. The same `as_of` date, policy files and
    embedding model give the same text. `policy_chunks` rows carry every column except
    `embedding` — vectors never enter this file; they're attached from the on-disk cache
    when the seed is loaded into Postgres.
    """
    as_of = _resolve_as_of(as_of)
    chunks = generate_policy_chunks()
    index_build_id = calculate_index_build_id(chunks, get_embedding_model())
    seed_meta = {"as_of": as_of, "index_build_id": index_build_id}
    parts = [
        "-- generated by data/generate.py — do not edit by hand\n",
        _insert("seed_meta", _SEED_META_COLUMNS, [seed_meta]),
        _insert("employees", _EMPLOYEE_COLUMNS, generate_employees(as_of)),
        _insert("leave_balances", _LEAVE_BALANCE_COLUMNS, generate_leave_balances(as_of)),
        _insert("known_outages", _OUTAGE_COLUMNS, generate_known_outages(as_of)),
        _insert("policy_chunks", _POLICY_CHUNK_COLUMNS, chunks),
    ]
    return "\n".join(part for part in parts if part)


def write_seed_sql(output_path: Path = SEED_SQL_PATH, as_of: date | None = None) -> None:
    output_path.write_text(render_seed_sql(as_of), encoding="utf-8", newline="\n")


def _load_into(conn: psycopg.Connection, sql: str, embeddings: dict[str, list[float]]) -> None:
    register_vector(conn)
    conn.execute(
        "TRUNCATE seed_meta, employees, leave_balances, known_outages, policy_chunks CASCADE"
    )
    # no params here: psycopg sends a script with no bound parameters as a
    # single simple-query message, so Postgres itself parses statement
    # boundaries — unlike a naive split on `;`, that's safe even though some
    # content (e.g. an outage note) contains a literal semicolon.
    conn.execute(sql)
    for chunk_id, vector in embeddings.items():
        conn.execute("UPDATE policy_chunks SET embedding = %s WHERE id = %s", (vector, chunk_id))


def load_seed_data(
    embed_fn: EmbedFn = embed_texts,
    conn: psycopg.Connection | None = None,
    as_of: date | None = None,
) -> None:
    """Truncate the seeded tables and reload them from a freshly rendered
    seed, then attach embeddings from the on-disk cache. Truncating first is
    what makes this safe to re-run — every run starts from the same clean
    slate rather than accumulating duplicates.
    """
    write_seed_sql(as_of=as_of)
    sql = SEED_SQL_PATH.read_text(encoding="utf-8")
    embeddings = embed_policy_chunks(embed_fn=embed_fn)

    if conn is not None:
        _load_into(conn, sql, embeddings)
        return

    with get_pool().connection() as pooled_conn:
        _load_into(pooled_conn, sql, embeddings)


if __name__ == "__main__":
    load_seed_data()
