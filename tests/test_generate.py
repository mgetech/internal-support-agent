"""The planted edge cases must show up exactly as promised, and the
generator must be deterministic — no wall-clock or RNG dependence.
"""

from datetime import date

import pytest
from data.generate import (
    embed_policy_chunks,
    generate_employees,
    generate_known_outages,
    generate_leave_balances,
    generate_policy_chunks,
    read_policies,
    render_seed_sql,
    write_seed_sql,
)


def _balance(employee_id: str) -> dict:
    return next(b for b in generate_leave_balances() if b["employee_id"] == employee_id)


def _remaining(balance: dict) -> float:
    return balance["entitlement_days"] - balance["taken_days"] - balance["pending_days"]


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


def test_emp_006_hired_2026_07_accrues_pro_rata():
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


def test_generation_is_deterministic():
    assert generate_employees() == generate_employees()
    assert generate_leave_balances() == generate_leave_balances()
    assert generate_known_outages() == generate_known_outages()


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


def test_every_policy_carries_the_policy_version_in_frontmatter():
    for policy in read_policies().values():
        assert policy.policy_version == "2026-09.1"


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
    assert len(generate_policy_chunks()) == 16


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


def test_render_seed_sql_contains_all_four_tables():
    sql = render_seed_sql()

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
    assert render_seed_sql() == render_seed_sql()


def test_write_seed_sql_matches_render(tmp_path):
    output = tmp_path / "seed.sql"
    write_seed_sql(output)

    assert output.read_text(encoding="utf-8") == render_seed_sql()
