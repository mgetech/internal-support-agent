"""Arguments for calling any registered tool, derived from its signature, so tests stay
generic over the registry instead of carrying per-tool fixtures.
"""

from __future__ import annotations

import inspect
import types
from datetime import date
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

from support_agent.tools import Tool

_SAMPLES: dict[Any, Any] = {str: "x", int: 1, float: 1.0, bool: True, date: date(2026, 11, 2)}


def _sample(annotation: Any) -> Any:
    if annotation in _SAMPLES:
        return _SAMPLES[annotation]
    origin = get_origin(annotation)
    if origin is Literal:
        return get_args(annotation)[0]
    if origin in (Union, types.UnionType):
        return _sample(next(a for a in get_args(annotation) if a is not type(None)))
    raise TypeError(f"no sample value for parameter type {annotation!r}")


def sample_args(fn: Tool) -> dict[str, Any]:
    """Plausible values for every required parameter; defaulted ones are left out."""
    hints = get_type_hints(fn)
    return {
        name: _sample(hints[name])
        for name, param in inspect.signature(fn).parameters.items()
        if param.default is inspect.Parameter.empty
    }
