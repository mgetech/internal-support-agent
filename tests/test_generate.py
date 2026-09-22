"""The planted edge cases must show up exactly as promised, and the
generator must be deterministic — no wall-clock or RNG dependence.
"""

from datetime import date

from data.generate import (
    POLICIES_DIR,
    POLICY_VERSION,
    generate_employees,
    generate_known_outages,
    generate_leave_balances,
    generate_policies,
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
    assert generate_policies() == generate_policies()


def test_five_policies_with_expected_ids():
    policies = generate_policies()

    assert set(policies) == {
        "vacation-policy",
        "sick-leave-policy",
        "parental-leave-policy",
        "expense-policy",
        "it-access-policy",
    }


def test_every_policy_carries_the_policy_version():
    for text in generate_policies().values():
        assert f"Policy version: {POLICY_VERSION}" in text


def test_vacation_policy_has_the_entitlement_heading():
    # ground truth citations (e.g. vacation-policy#entitlement#0) depend on
    # this exact heading text once the structural chunker slugifies it
    assert "## Entitlement" in generate_policies()["vacation-policy"]


def test_escalation_only_areas_are_written_into_the_policy_text():
    # hard-wrapped prose, so compare against whitespace-normalized text
    policies = {k: " ".join(v.split()) for k, v in generate_policies().items()}

    assert "not handled through self-service" in policies["vacation-policy"]
    assert "referred to HR" in policies["sick-leave-policy"]
    assert "always require review by HR" in policies["parental-leave-policy"]
    assert "always escalated immediately" in policies["it-access-policy"]


def test_committed_policy_files_match_the_generator():
    for policy_id, text in generate_policies().items():
        committed = (POLICIES_DIR / f"{policy_id}.md").read_text(encoding="utf-8")
        assert committed == text
