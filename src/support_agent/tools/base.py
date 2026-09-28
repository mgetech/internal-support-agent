"""What a tool is, and the marker for tools that propose a change instead of making it.
Kept apart from the registry so tool modules can import it without a cycle.
"""

from __future__ import annotations

from collections.abc import Callable

Tool = Callable[..., str]


def gated_write(fn: Tool) -> Tool:
    """Mark a tool that proposes a change into `pending_actions` instead of making it."""
    fn.gated_write = True  # type: ignore[attr-defined]
    return fn


def is_gated_write(fn: Tool) -> bool:
    return getattr(fn, "gated_write", False)
