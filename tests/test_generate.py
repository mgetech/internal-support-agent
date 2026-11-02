"""The planted edge cases must show up exactly as promised, and the
generator must be deterministic — no wall-clock or RNG dependence.
"""

from datetime import date, timedelta
from types import SimpleNamespace

import pytest
from data import generate
from data.generate import (
    embed_policy_chunks,
    generate_employees,
    generate_known_outages,
    generate_leave_balances,
    generate_policy_chunks,
    get_seed_as_of,
    load_seed_data,
    read_policies,
    render_seed_sql,
    write_seed_sql,
)


@pytest.fixture(autouse=True)
def isolated_embedding_cache(tmp_path, monkeypatch):
    """No test here may write fake vectors into the real cache in .embeddings-cache, or
    need the settings to name the model.
    """
    monkeypatch.setattr(generate, "CACHE_DIR", tmp_path / "embedding-cache")
    monkeypatch.setattr(generate, "get_embedding_model", lambda: "fake")


def _balance(employee_id: str) -> dict:
    return next(b for b in generate_leave_balances() if b["employee_id"] == employee_id)


def _remaining(balance: dict) -> float:
    return balance["entitlement_days"] - balance["taken_days"] - balance["approved_days"]


def test_twelve_employees_with_unique_sequential_ids():
    employees = generate_employees()

    assert [e["id"] for e in employees] == [f"emp_{n:03d}" for n in range(1, 13)]


def test_emp_001_happy_path_eighteen_remaining():
    balance = _balance("emp_001")

    assert balance["entitlement_days"] == 30
    assert balance["taken_days"] == 12
    assert _remaining(balance) == 18


def test_emp_004_over_balance_three_point_five_remaining():
    assert _remaining(_balance("emp_004")) == 3.5


def test_emp_003_part_time_prorated_to_fifteen_days():
    employees = {e["id"]: e for e in generate_employees()}

    assert employees["emp_003"]["employment_type"] == "part_time"
    assert employees["emp_003"]["weekly_hours"] == 20
    assert _balance("emp_003")["entitlement_days"] == 15


def test_emp_006_accrues_pro_rata_on_the_test_date():
    # the test date is 2026-09-22, 83 days after the hire date
    employees = {e["id"]: e for e in generate_employees()}

    assert employees["emp_006"]["hired_at"] == date(2026, 7, 1)
    assert _balance("emp_006")["entitlement_days"] == 5.0


def test_emp_012_zero_remaining():
    assert _remaining(_balance("emp_012")) == 0


def test_authorized_approver_roles():
    employees = {e["id"]: e for e in generate_employees()}

    assert employees["emp_005"]["role"] == "hr_partner"
    assert employees["emp_010"]["role"] == "manager"


def test_no_balance_ever_goes_negative():
    for balance in generate_leave_balances():
        assert _remaining(balance) >= 0


def test_one_active_vpn_outage_and_one_resolved_sso_incident():
    outages = generate_known_outages()

    vpn = next(o for o in outages if o["system"] == "vpn")
    sso = next(o for o in outages if o["system"] == "sso")

    assert vpn["status"] == "identified"
    assert "certificate" in vpn["note"]
    assert sso["status"] == "resolved"


def test_generation_is_the_same_for_the_same_date():
    as_of = date(2027, 5, 17)

    assert generate_employees(as_of) == generate_employees(as_of)
    assert generate_leave_balances(as_of) == generate_leave_balances(as_of)
    assert generate_known_outages(as_of) == generate_known_outages(as_of)


def test_generation_follows_the_clock_when_no_date_is_given():
    # the tests fixture sets util.get_today to 2026-09-22
    assert generate_known_outages() == generate_known_outages(date(2026, 9, 22))


# --- the data is current on any date ---

# early, middle and late in one year, and a year other than the default one
YEAR_2027 = [date(2027, 1, 2), date(2027, 6, 30), date(2027, 12, 28)]


def _balance_at(employee_id: str, as_of: date, year: int | None = None) -> dict:
    year = year or as_of.year
    return next(
        b
        for b in generate_leave_balances(as_of)
        if b["employee_id"] == employee_id and b["year"] == year
    )


@pytest.mark.parametrize("as_of", YEAR_2027)
def test_the_planted_balances_are_the_same_on_any_date(as_of):
    assert _remaining(_balance_at("emp_001", as_of)) == 18
    assert _balance_at("emp_001", as_of)["entitlement_days"] == 30
    assert _remaining(_balance_at("emp_004", as_of)) == 3.5
    assert _remaining(_balance_at("emp_012", as_of)) == 0
    assert _balance_at("emp_003", as_of)["entitlement_days"] == 15


@pytest.mark.parametrize("as_of", YEAR_2027)
def test_balances_are_written_for_this_year_and_the_next(as_of):
    balances = generate_leave_balances(as_of)

    assert {b["year"] for b in balances} == {as_of.year, as_of.year + 1}
    assert len(balances) == 24
    for employee_id in ("emp_001", "emp_004", "emp_006"):
        this_year = _balance_at(employee_id, as_of)
        next_year = _balance_at(employee_id, as_of, as_of.year + 1)
        assert {**this_year, "year": 0} == {**next_year, "year": 0}


@pytest.mark.parametrize("as_of", YEAR_2027)
def test_emp_006_is_in_their_first_calendar_year(as_of):
    employee = next(e for e in generate_employees(as_of) if e["id"] == "emp_006")
    hired_at = employee["hired_at"]

    assert hired_at.year == as_of.year
    assert hired_at <= as_of
    # entitlement accrues 2.5 days for each completed month and is never negative
    months = (as_of.year - hired_at.year) * 12 + as_of.month - hired_at.month
    months -= as_of.day < hired_at.day
    assert _balance_at("emp_006", as_of)["entitlement_days"] == months * 2.5


@pytest.mark.parametrize("as_of", YEAR_2027)
def test_the_vpn_outage_is_active_and_the_sso_outage_is_resolved(as_of):
    outages = {o["system"]: o for o in generate_known_outages(as_of)}

    assert outages["vpn"]["status"] != "resolved"
    assert outages["vpn"]["started_at"].date() == as_of - timedelta(days=2)
    assert outages["sso"]["status"] == "resolved"
    assert outages["sso"]["started_at"].date() == as_of - timedelta(days=12)


def test_the_same_date_gives_byte_identical_seed_sql():
    assert render_seed_sql(date(2027, 6, 30)) == render_seed_sql(date(2027, 6, 30))


def test_another_date_gives_another_seed_sql():
    assert render_seed_sql(date(2027, 6, 30)) != render_seed_sql(date(2028, 6, 30))


def test_the_seed_sql_records_its_date():
    sql = render_seed_sql(date(2027, 6, 30))

    assert "INSERT INTO seed_meta (as_of) VALUES\n    ('2027-06-30');" in sql


def test_five_policies_with_expected_ids_and_domains():
    policies = read_policies()

    assert {p.id for p in policies.values()} == {
        "vacation-policy",
        "sick-leave-policy",
        "parental-leave-policy",
        "expense-policy",
        "it-access-policy",
    }
    assert policies["it-access-policy"].domain == "it"
    assert policies["vacation-policy"].domain == "hr"


# a policy file gets a new version when its text changes
POLICY_VERSIONS = {
    "vacation-policy": "2026-09.1",
    "sick-leave-policy": "2026-09.1",
    "parental-leave-policy": "2026-09.1",
    "expense-policy": "2026-09.1",
    "it-access-policy": "2026-11.1",
}


def test_every_policy_carries_its_policy_version_in_frontmatter():
    policies = read_policies()

    assert {id: policy.policy_version for id, policy in policies.items()} == POLICY_VERSIONS


def test_every_policy_chunk_row_carries_the_version_of_its_policy_file():
    rows = generate_policy_chunks()

    assert {row["policy"]: row["policy_version"] for row in rows} == POLICY_VERSIONS


def test_the_seed_sql_writes_the_policy_version_of_every_chunk():
    sql = render_seed_sql()

    # one quoted version in each chunk row: the 5 chunks of the IT access policy have the
    # new version and the other 12 have the old one
    chunks = generate_policy_chunks()
    new_version = [row for row in chunks if row["policy_version"] == "2026-11.1"]
    assert sql.count("'2026-11.1'") == len(new_version) == 5
    assert sql.count("'2026-09.1'") == len(chunks) - len(new_version)


def test_vacation_policy_has_the_entitlement_heading():
    # ground truth citations (e.g. vacation-policy#entitlement#0) depend on
    # this exact heading text once the structural chunker slugifies it
    assert "## Entitlement" in read_policies()["vacation-policy"].text


def test_escalation_only_areas_are_written_into_the_policy_text():
    # hard-wrapped prose, so compare against whitespace-normalized text
    policies = {k: " ".join(p.text.split()) for k, p in read_policies().items()}

    assert "not handled through self-service" in policies["vacation-policy"]
    assert "referred to HR" in policies["sick-leave-policy"]
    assert "always require review by HR" in policies["parental-leave-policy"]
    assert "always escalated immediately" in policies["it-access-policy"]


def test_read_policies_rejects_a_file_missing_frontmatter(tmp_path):
    (tmp_path / "broken-policy.md").write_text("# Broken Policy\n\nNo frontmatter here.\n")

    with pytest.raises(ValueError, match="frontmatter"):
        read_policies(tmp_path)


def test_generate_policy_chunks_covers_the_whole_corpus():
    assert len(generate_policy_chunks()) == 17


def test_generate_policy_chunks_is_stable_across_regeneration():
    # each call re-reads every policy file from disk; two independent runs
    # (as two separate `make seed` invocations would be) must agree exactly
    assert generate_policy_chunks() == generate_policy_chunks()


def test_policy_chunk_rows_carry_domain_from_their_policy():
    rows = {r["id"]: r for r in generate_policy_chunks()}

    assert rows["it-access-policy#vpn-access#0"]["domain"] == "it"
    assert rows["vacation-policy#entitlement#0"]["domain"] == "hr"
    assert rows["vacation-policy#entitlement#0"]["chunker"] == "structural"


def test_embed_policy_chunks_returns_a_vector_per_chunk_id(tmp_path):
    rows = generate_policy_chunks()[:3]

    def fake_embed(texts):
        return [[float(len(t))] for t in texts]

    vectors = embed_policy_chunks(rows, embed_fn=fake_embed, model="fake", cache_dir=tmp_path)

    assert set(vectors) == {row["id"] for row in rows}
    for row in rows:
        assert vectors[row["id"]] == [float(len(row["content"]))]


def test_embed_policy_chunks_caches_and_skips_already_embedded_content(tmp_path):
    rows = generate_policy_chunks()[:2]
    calls = []

    def counting_embed(texts):
        calls.append(list(texts))
        return [[0.0] for _ in texts]

    embed_policy_chunks(rows, embed_fn=counting_embed, model="fake", cache_dir=tmp_path)
    embed_policy_chunks(rows, embed_fn=counting_embed, model="fake", cache_dir=tmp_path)

    assert len(calls) == 1  # the second call found everything already cached
    assert (tmp_path / "fake.json").exists()


def test_an_edited_chunk_with_the_same_id_is_embedded_again(tmp_path):
    chunk = {"id": "vacation-policy#entitlement#0", "content": "30 days a year"}
    other = {"id": "vacation-policy#carryover#0", "content": "5 days carry over"}
    calls = []

    def counting_embed(texts):
        calls.append(list(texts))
        return [[float(len(t))] for t in texts]

    def embed(rows):
        return embed_policy_chunks(rows, embed_fn=counting_embed, model="m", cache_dir=tmp_path)

    first = embed([chunk, other])
    again = embed([chunk, other])
    edited = embed([{**chunk, "content": "28 days a year, or more"}, other])

    assert calls == [["30 days a year", "5 days carry over"], ["28 days a year, or more"]]
    assert again == first
    # same id, new text: a new vector. The unchanged chunk comes from the cache.
    assert edited[chunk["id"]] != first[chunk["id"]]
    assert edited[other["id"]] == first[other["id"]]


def test_the_cache_is_not_shared_between_embedding_models(tmp_path):
    rows = [{"id": "vacation-policy#carryover#0", "content": "5 days carry over"}]
    calls = []

    def counting_embed(texts):
        calls.append(list(texts))
        return [[0.0] for _ in texts]

    embed_policy_chunks(rows, embed_fn=counting_embed, model="model-a", cache_dir=tmp_path)
    embed_policy_chunks(rows, embed_fn=counting_embed, model="model-b", cache_dir=tmp_path)
    embed_policy_chunks(rows, embed_fn=counting_embed, model="model-a", cache_dir=tmp_path)

    # model-b is embedded on its own, and model-a is still cached
    assert len(calls) == 2
    assert {path.name for path in tmp_path.iterdir()} == {"model-a.json", "model-b.json"}


def test_the_cache_is_keyed_on_the_embedding_deployment_from_the_settings(monkeypatch, tmp_path):
    monkeypatch.undo()
    rows = [{"id": "vacation-policy#carryover#0", "content": "5 days carry over"}]
    settings = SimpleNamespace(azure_openai_deployment_embedding="my-embedding-deployment")
    monkeypatch.setattr(generate, "get_settings", lambda: settings)

    embed_policy_chunks(rows, embed_fn=lambda texts: [[0.0] for _ in texts], cache_dir=tmp_path)

    assert [path.name for path in tmp_path.iterdir()] == ["my-embedding-deployment.json"]


def test_render_seed_sql_contains_every_seeded_table():
    sql = render_seed_sql()

    assert "INSERT INTO seed_meta" in sql
    assert "INSERT INTO employees" in sql
    assert "INSERT INTO leave_balances" in sql
    assert "INSERT INTO known_outages" in sql
    assert "INSERT INTO policy_chunks" in sql


def test_render_seed_sql_excludes_the_embedding_column():
    sql = render_seed_sql()

    header = sql[sql.index("INSERT INTO policy_chunks") :].splitlines()[0]
    assert "embedding" not in header


def test_render_seed_sql_escapes_single_quotes():
    # "the requesting employee's manager" from the vacation policy text
    assert "employee''s manager" in render_seed_sql()


def test_render_seed_sql_is_deterministic():
    assert render_seed_sql(date(2026, 9, 22)) == render_seed_sql(date(2026, 9, 22))


def test_write_seed_sql_matches_render(tmp_path):
    output = tmp_path / "seed.sql"
    write_seed_sql(output, date(2026, 9, 22))

    assert output.read_text(encoding="utf-8") == render_seed_sql(date(2026, 9, 22))


def _fake_embed(texts):
    return [[0.0] * 1536 for _ in texts]


# INTEGRATION TESTS: the tests below use the real Postgres (make db-up).
def test_load_seed_data_populates_every_seeded_table(clean_db):
    load_seed_data(embed_fn=_fake_embed, conn=clean_db)

    assert clean_db.execute("SELECT count(*) FROM employees").fetchone()[0] == 12
    assert clean_db.execute("SELECT count(*) FROM known_outages").fetchone()[0] == 2
    assert clean_db.execute("SELECT count(*) FROM policy_chunks").fetchone()[0] == 17
    missing = clean_db.execute(
        "SELECT count(*) FROM policy_chunks WHERE embedding IS NULL"
    ).fetchone()[0]
    assert missing == 0


def test_load_seed_data_writes_its_vectors_to_the_cache_it_is_given(clean_db, tmp_path):
    load_seed_data(embed_fn=_fake_embed, conn=clean_db)

    assert (tmp_path / "embedding-cache" / "fake.json").exists()


def test_load_seed_data_is_safe_to_run_twice(clean_db):
    load_seed_data(embed_fn=_fake_embed, conn=clean_db)
    load_seed_data(embed_fn=_fake_embed, conn=clean_db)

    assert clean_db.execute("SELECT count(*) FROM employees").fetchone()[0] == 12
    assert clean_db.execute("SELECT count(*) FROM policy_chunks").fetchone()[0] == 17


def test_the_seed_date_can_be_read_back_from_the_database(clean_db):
    assert get_seed_as_of() is None

    load_seed_data(embed_fn=_fake_embed, conn=clean_db, as_of=date(2027, 6, 30))

    assert get_seed_as_of() == date(2027, 6, 30)
