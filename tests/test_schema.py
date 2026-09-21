"""Smoke test: db/init.sql produces every table and CHECK constraint §6 of the
spec requires. Catches an accidentally dropped table or constraint, not
business logic.
"""

EXPECTED_TABLES = {
    "employees",
    "leave_balances",
    "leave_requests",
    "tickets",
    "known_outages",
    "policy_chunks",
    "pending_actions",
    "audit_log",
    "decision_records",
    "feedback",
    "usage_daily",
}

EXPECTED_CHECK_CONSTRAINTS = {
    "employees_employment_type_check",
    "known_outages_status_check",
    "policy_chunks_domain_check",
    "pending_actions_status_check",
    "decision_records_channel_check",
    "feedback_rating_check",
    "feedback_triage_status_check",
}


def test_every_expected_table_exists(clean_db):
    rows = clean_db.execute(
        "select tablename from pg_tables where schemaname = 'public'"
    ).fetchall()
    tables = {row[0] for row in rows}

    assert tables >= EXPECTED_TABLES


def test_every_expected_check_constraint_exists(clean_db):
    rows = clean_db.execute(
        "select conname from pg_constraint where contype = 'c' "
        "and connamespace = 'public'::regnamespace"
    ).fetchall()
    constraints = {row[0] for row in rows}

    assert constraints >= EXPECTED_CHECK_CONSTRAINTS
