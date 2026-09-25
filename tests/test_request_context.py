import dataclasses

import pytest

from support_agent.request_context import RequestContext, _current, bind_request_context


def test_bind_sets_context_for_the_block():
    with bind_request_context("emp_001", "req-1", "rest") as ctx:
        assert _current.get() == ctx
        assert ctx == RequestContext(employee_id="emp_001", request_id="req-1", channel="rest")

    assert _current.get() is None


def test_nested_bind_restores_outer_context():
    with bind_request_context("emp_001", "req-1", "rest") as outer:
        with bind_request_context("emp_004", "req-2", "mcp"):
            assert _current.get().employee_id == "emp_004"
        assert _current.get() == outer


def test_context_is_restored_after_exception():
    with pytest.raises(RuntimeError), bind_request_context("emp_001", "req-1", "rest"):
        raise RuntimeError

    assert _current.get() is None


def test_context_is_immutable():
    with (
        bind_request_context("emp_001", "req-1", "rest") as ctx,
        pytest.raises(dataclasses.FrozenInstanceError),
    ):
        ctx.employee_id = "emp_002"
