import pytest

from support_agent.audit import record
from support_agent.request_context import NoRequestContextError, bind_request_context


class _CapturingConn:
    """Stands in for a connection; keeps the parameters of each execute call."""

    def __init__(self):
        self.calls = []

    def execute(self, query, params):
        self.calls.append(params)


def _written(conn):
    request_id, actor, event, payload = conn.calls[-1]
    return request_id, actor, event, payload.obj


def test_actor_defaults_to_agent():
    conn = _CapturingConn()
    with bind_request_context("emp_001", "req-1", "rest"):
        record("tool_call", {"tool": "get_leave_balance"}, conn=conn)

    assert _written(conn) == ("req-1", "agent", "tool_call", {"tool": "get_leave_balance"})


@pytest.mark.parametrize(
    ("actor", "expected"),
    [
        ("agent", "agent"),
        ("system", "system"),
        ("employee", "employee:emp_005"),
        ("approver", "approver:emp_005"),
    ],
)
def test_actor_resolves_from_the_bound_employee(actor, expected):
    conn = _CapturingConn()
    with bind_request_context("emp_005", "req-2", "rest"):
        record("action_approved", {"action_id": 1}, actor=actor, conn=conn)

    assert _written(conn)[1] == expected


def test_every_write_carries_the_bound_request_id():
    conn = _CapturingConn()
    with bind_request_context("emp_001", "req-a", "rest"):
        record("tool_call", {}, conn=conn)
    with bind_request_context("emp_001", "req-b", "mcp"):
        record("tool_call", {}, conn=conn)

    assert [call[0] for call in conn.calls] == ["req-a", "req-b"]


def test_unbound_record_raises_and_writes_nothing():
    conn = _CapturingConn()
    with pytest.raises(NoRequestContextError):
        record("tool_call", {}, conn=conn)

    assert conn.calls == []


def test_record_persists_a_row(clean_db):
    with bind_request_context("emp_004", "req-3", "mcp"):
        record("tool_call", {"tool": "get_known_outages"}, actor="employee", conn=clean_db)

    row = clean_db.execute("SELECT request_id, actor, event, payload FROM audit_log").fetchone()
    assert row == ("req-3", "employee:emp_004", "tool_call", {"tool": "get_known_outages"})
