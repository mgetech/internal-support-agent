import asyncio
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


async def test_interleaved_requests_never_see_each_others_identity():
    a_bound = asyncio.Event()
    b_bound = asyncio.Event()

    async def request_a():
        with bind_request_context("emp_001", "req-a", "rest"):
            a_bound.set()
            await b_bound.wait()  # B is now bound in its own task
            return get_request_context()

    async def request_b():
        await a_bound.wait()  # A is bound and suspended mid-request
        with bind_request_context("emp_004", "req-b", "mcp"):
            b_bound.set()
            await asyncio.sleep(0)
            return get_request_context()

    seen_a, seen_b = await asyncio.gather(request_a(), request_b())

    assert (seen_a.employee_id, seen_a.request_id) == ("emp_001", "req-a")
    assert (seen_b.employee_id, seen_b.request_id) == ("emp_004", "req-b")


async def test_many_concurrent_requests_keep_their_own_identity():
    async def request(n):
        with bind_request_context(f"emp_{n:03}", f"req-{n}", "rest"):
            seen = []
            for _ in range(5):
                await asyncio.sleep(0)  # yield so the other requests run in between
                seen.append(get_request_context().employee_id)
            return seen

    results = await asyncio.gather(*(request(n) for n in range(1, 13)))

    for n, seen in enumerate(results, start=1):
        assert seen == [f"emp_{n:03}"] * 5


async def test_binding_inside_a_task_does_not_leak_to_the_caller():
    async def request():
        with bind_request_context("emp_001", "req-1", "rest"):
            await asyncio.sleep(0)

    await asyncio.create_task(request())

    with pytest.raises(NoRequestContextError):
        get_request_context()
