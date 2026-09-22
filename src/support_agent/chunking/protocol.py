"""The chunker contract every strategy implements — a heading-aware
structural default now, with room for other strategies (fixed-size,
recursive, semantic) later. Any object with a matching `chunk` method
satisfies this protocol; no shared base class is required.
"""

from __future__ import annotations

from typing import NamedTuple, Protocol


class Chunk(NamedTuple):
    id: str
    heading: str
    content: str
    meta: dict[str, str]


class Chunker(Protocol):
    def chunk(self, policy_name: str, text: str) -> list[Chunk]: ...
