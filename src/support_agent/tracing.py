"""Tracing with Langfuse. A request is one trace, with a span for every graph node and
tool call, and a generation for every model call. Tracing is off unless both Langfuse
keys are set: then no span is made and nothing is sent.

The Decision Record is the system of record. The trace is only for debugging, and nothing
reads it back. A trace must never change what a request does: every call to Langfuse is
guarded, and a failure is logged as a warning and then ignored.

What a span carries depends on `TRACE_CONTENT`. When it is false, a span has names, times,
token counts and statuses only. When it is true, it also has inputs and outputs: the
employee's message, tool arguments and results, prompts and answers. Only use true with
synthetic data, or with a Langfuse that you host yourself.

Spans never get a user id or a session id. Warnings carry the error type and the request
id, never the error message, because that may contain user input.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import lru_cache
from typing import Any

from langchain_core.messages import ToolMessage
from langfuse import Langfuse

from support_agent.config import Settings, get_settings
from support_agent.request_context import NoRequestContextError, get_request_context
from support_agent.state import AgentState

logger = logging.getLogger(__name__)

CONTENT_FIELDS = ("input", "output")


def log_failure(event: str, error: BaseException) -> None:
    """Log a warning for a failed call to Langfuse: the event, the error type and the
    request id. Not the error message.
    """
    try:
        request_id = get_request_context().request_id
    except NoRequestContextError:
        request_id = "-"
    logger.warning(
        "tracing_failed event=%s error_type=%s request_id=%s",
        event,
        type(error).__name__,
        request_id,
    )


def create_langfuse(settings: Settings, **options: Any) -> Langfuse | None:
    """The Langfuse client for these settings, or None when a key is missing. `options`
    go to the client, for example `span_exporter`.
    """
    if not (settings.langfuse_public_key and settings.langfuse_secret_key):
        return None
    return Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        base_url=settings.langfuse_base_url,
        **options,
    )


@lru_cache
def get_langfuse() -> Langfuse | None:
    """The client of the process, made on the first use. None when tracing is off, or when
    the client could not be made.
    """
    try:
        return create_langfuse(get_settings())
    except Exception as error:
        log_failure("create_client", error)
        return None


@lru_cache
def is_content_traced() -> bool:
    """Whether spans carry inputs and outputs (`TRACE_CONTENT`)."""
    return get_settings().trace_content


class Span:
    """A span that can be updated. With tracing off, or after a failure, every call does
    nothing.
    """

    def __init__(self, span: Any = None) -> None:
        self._span = span

    def update(self, **fields: Any) -> None:
        """Set fields of the span, for example `level`, `usage_details` or `metadata`.
        `input` and `output` are dropped unless `TRACE_CONTENT` is true.
        """
        if self._span is None:
            return
        try:
            if not is_content_traced():
                for name in CONTENT_FIELDS:
                    fields.pop(name, None)
            self._span.update(**fields)
        except Exception as error:
            log_failure("update", error)


@contextmanager
def start_observation(
    name: str,
    as_type: str = "span",
    metadata: dict[str, Any] | None = None,
    input: Any = None,
    **options: Any,
) -> Iterator[Span]:
    """A span that is the current one inside the `with` block. An exception in the block
    is marked on the span and raised again. `input` is dropped unless `TRACE_CONTENT` is
    true. `options` go to Langfuse, for example `model` for a generation.
    """
    client = get_langfuse()
    manager = None
    span = Span()
    if client is not None:
        try:
            if input is not None and is_content_traced():
                options["input"] = input
            manager = client.start_as_current_observation(
                name=name, as_type=as_type, metadata=metadata, **options
            )
            span = Span(manager.__enter__())
        except Exception as error:
            log_failure("start", error)
            manager = None
    try:
        yield span
    except BaseException as error:
        # The span gets the error type only. If the error went to the span itself, it would
        # add the message and the stack trace to the span as an event.
        span.update(level="ERROR", status_message=type(error).__name__)
        _finish(manager)
        raise
    _finish(manager)


def _finish(manager: Any) -> None:
    if manager is None:
        return
    try:
        manager.__exit__(None, None, None)
    except Exception as exit_error:
        log_failure("finish", exit_error)


def get_trace_id() -> str | None:
    """The id of the trace that the current span belongs to. None when tracing is off or
    when no span is open.
    """
    client = get_langfuse()
    if client is None:
        return None
    try:
        return client.get_current_trace_id()
    except Exception as error:
        log_failure("get_trace_id", error)
        return None


def trace_node(
    name: str, node: Callable[[AgentState], dict[str, Any]]
) -> Callable[[AgentState], dict[str, Any]]:
    """The node, run inside a span with the node's name."""

    def traced(state: AgentState) -> dict[str, Any]:
        with start_observation(name):
            return node(state)

    return traced


def trace_tool_call(request: Any, execute: Callable[[Any], Any]) -> Any:
    """A `wrap_tool_call` hook for LangGraph's ToolNode: runs each tool call inside a span
    named like the tool. The span gets the ERROR level when the call came back as a failed
    ToolMessage.
    """
    with start_observation(
        request.tool_call["name"], as_type="tool", input=request.tool_call["args"]
    ) as span:
        result = execute(request)
        if isinstance(result, ToolMessage):
            if result.status == "error":
                span.update(level="ERROR")
            span.update(output=result.text)
        return result
