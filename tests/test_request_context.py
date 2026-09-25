import dataclasses
import threading

import pytest

from support_agent.request_context import (
    NoRequestContextError,
    RequestContext,
    bind_request_context,
    get_request_context,
)


def test_bind_sets_context_for_the_block():
    with bind_request_context("emp_001", "req-1", "rest") as ctx:
        assert get_request_context() == ctx
        assert ctx == RequestContext(employee_id="emp_001", request_id="req-1", channel="rest")

    with pytest.raises(NoRequestContextError):
        get_request_context()


def test_nested_bind_restores_outer_context():
    with bind_request_context("emp_001", "req-1", "rest") as outer:
        with bind_request_context("emp_004", "req-2", "mcp"):
            assert get_request_context().employee_id == "emp_004"
        assert get_request_context() == outer


def test_context_is_restored_after_exception():
    with pytest.raises(RuntimeError), bind_request_context("emp_001", "req-1", "rest"):
        raise RuntimeError

    with pytest.raises(NoRequestContextError):
        get_request_context()


def test_context_is_immutable():
    with (
        bind_request_context("emp_001", "req-1", "rest") as ctx,
        pytest.raises(dataclasses.FrozenInstanceError),
    ):
        ctx.employee_id = "emp_002"


def test_unbound_read_raises_with_a_clear_message():
    with pytest.raises(NoRequestContextError, match="no request context is bound"):
        get_request_context()


def test_new_thread_does_not_inherit_the_binding():
    errors = []

    def read():
        try:
            get_request_context()
        except NoRequestContextError as exc:
            errors.append(exc)

    with bind_request_context("emp_001", "req-1", "rest"):
        thread = threading.Thread(target=read)
        thread.start()
        thread.join()

    assert len(errors) == 1
