"""The single place identity enters the system. Transports bind the authenticated
employee here; tools read it from here. No tool takes an employee id as an argument.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Literal

Channel = Literal["rest", "mcp"]


@dataclass(frozen=True, slots=True)
class RequestContext:
    employee_id: str
    request_id: str
    channel: Channel


_current: ContextVar[RequestContext | None] = ContextVar("request_context", default=None)


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
