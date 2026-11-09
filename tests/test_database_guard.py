"""The suite refuses to run on a database that is not a test database. The database tests
drop the schema of the database they connect to, so a dev database must never be one.
"""

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from tests import database_guard
from tests.database_guard import (
    check_test_database,
    get_database_name,
    get_test_database_url,
)

ROOT = Path(__file__).resolve().parent.parent
# nothing listens on port 1, so a run that tried to connect would skip its database tests
UNREACHABLE = "postgresql://support_agent:secret@localhost:1/{name}"


@pytest.mark.parametrize(
    "name",
    ["support_agent_test", "x_scratch", "t11_scratch", "anything_test"],
)
def test_a_test_database_is_allowed(name):
    assert check_test_database(UNREACHABLE.format(name=name)) is None


@pytest.mark.parametrize(
    "name", ["support_agent", "postgres", "support_agent_test_old", "testing", "test", "prod"]
)
def test_any_other_database_is_refused_with_a_clear_message(name):
    message = check_test_database(UNREACHABLE.format(name=name))

    assert message is not None
    assert "not a test database" in message
    assert f"'{name}'" in message
    assert "make test" in message
    # the password is never in the message
    assert "secret" not in message


def test_no_database_configured_is_allowed_because_the_database_tests_skip():
    assert check_test_database(None) is None
    assert check_test_database("") is None


def test_a_key_value_connection_string_is_read_too():
    assert get_database_name("host=localhost dbname=support_agent") == "support_agent"
    assert check_test_database("host=localhost dbname=support_agent") is not None
    assert check_test_database("host=localhost dbname=support_agent_test") is None


def test_a_connection_string_without_a_database_name_is_refused():
    assert check_test_database("host=localhost user=support_agent") is not None


def test_the_test_database_of_a_dev_database_is_the_same_server_with_test_added():
    url = get_test_database_url("postgresql://u:p@localhost:5433/support_agent")

    assert get_database_name(url) == "support_agent_test"
    assert "localhost" in url and "5433" in url and "u" in url


def test_a_test_database_url_stays_as_it_is():
    url = "postgresql://u:p@localhost:5433/support_agent_test"

    assert get_test_database_url(url) == url


def test_the_command_prints_nothing_without_a_database(monkeypatch, capsys):
    monkeypatch.setattr(
        database_guard, "DatabaseSettings", lambda: SimpleNamespace(database_url=None)
    )

    assert database_guard.main() == 0
    assert capsys.readouterr().out == ""


def test_the_command_prints_a_test_url_as_it_is_without_touching_the_server(monkeypatch, capsys):
    url = UNREACHABLE.format(name="support_agent_test")
    monkeypatch.setattr(
        database_guard, "DatabaseSettings", lambda: SimpleNamespace(database_url=url)
    )

    assert database_guard.main() == 0
    assert capsys.readouterr().out.strip() == url


def run_suite(database_url):
    """Start a second pytest on one small test file, with this DATABASE_URL."""
    env = {**os.environ, "DATABASE_URL": database_url}
    return subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_costing.py", "-q", "-p", "no:cacheprovider"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_the_suite_aborts_before_any_test_when_it_points_at_a_non_test_database():
    result = run_suite(UNREACHABLE.format(name="support_agent"))

    output = result.stdout + result.stderr
    assert result.returncode == 3
    assert "not a test database" in output
    assert "passed" not in output
    assert "secret" not in output


def test_the_suite_runs_when_it_points_at_a_test_database():
    result = run_suite(UNREACHABLE.format(name="support_agent_test"))

    assert result.returncode == 0, result.stdout + result.stderr
    assert "passed" in result.stdout
