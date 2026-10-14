"""Small helpers shared across modules."""

from __future__ import annotations

from datetime import date


def get_today() -> date:
    """Today's date. Code reads it here so tests can replace it with a fixed date."""
    return date.today()
