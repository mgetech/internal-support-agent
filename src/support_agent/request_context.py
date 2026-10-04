"""The single place identity enters the system. Transports bind the authenticated
employee here; tools read it from here. No tool takes an employee id as an argument.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Literal

Channel = Literal["rest", "mcp"]


@dataclass(frozen=True, slots=True)
class RequestContext:
    employee_id: str
    request_id: str
    channel: Channel
    # evidence items for this request's Decision Record, in the order they happened.
    # Tools add to this list; the graph reads it.
    evidence: list[dict[str, Any]] = field(default_factory=list, compare=False, repr=False)
    # when the request started, for the latency in the Decision Record (time.perf_counter)
    started_at: float = field(default_factory=time.perf_counter, compare=False, repr=False)


class NoRequestContextError(RuntimeError):
    """Raised when identity is read outside a bound request."""


_current: ContextVar[RequestContext | None] = ContextVar("request_context", default=None)


def get_request_context() -> RequestContext:
    """The context bound for this request. Raises instead of defaulting when none is bound."""
    ctx = _current.get()
    if ctx is None:
        raise NoRequestContextError(
            "no request context is bound; the transport must call bind_request_context() "
            "before any tool runs"
        )
    return ctx


@contextmanager
def bind_request_context(
    employee_id: str, request_id: str, channel: Channel
) -> Iterator[RequestContext]:
    """Bind a request context for the duration of the block, then restore the previous one."""
    ctx = RequestContext(employee_id=employee_id, request_id=request_id, channel=channel)
    token = _current.set(ctx)
    try:
        yield ctx
    finally:
        _current.reset(token)
