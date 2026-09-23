"""Heading-aware chunker — the production default. Splits a policy's markdown
on `##` headings, prefixes each chunk with the policy title for context, and
sub-splits any section over the token budget with overlap so a long section
never gets silently truncated.
"""

from __future__ import annotations

import re

from support_agent.chunking.protocol import Chunk

# target 300-500 tokens per chunk; sections at or under this ship as one
# chunk, larger ones sub-split with overlap so no boundary loses context
MAX_CHUNK_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 50

_TITLE_RE = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)
_HEADING_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)


def _slugify(heading: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", heading.lower()).strip("-")


def _title(text: str) -> str:
    match = _TITLE_RE.search(text)
    if not match:
        raise ValueError("policy text has no H1 title")
    return match.group(1)


def _sections(text: str) -> list[tuple[str, str]]:
    """[(heading, body)] for every `##` section, in document order."""
    headings = list(_HEADING_RE.finditer(text))
    sections = []
    for i, match in enumerate(headings):
        end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
        sections.append((match.group(1), text[match.end() : end].strip()))
    return sections


def _windows(words: list[str], max_tokens: int, overlap: int) -> list[list[str]]:
    """Split words into overlapping chunks of at most `max_tokens`. A single
    window (no split) when the section already fits.
    """
    if len(words) <= max_tokens:
        return [words]
    stride = max_tokens - overlap
    windows = []
    start = 0
    while start < len(words):
        end = start + max_tokens
        windows.append(words[start:end])
        if end >= len(words):
            break
        start += stride
    return windows


class StructuralChunker:
    """Heading-aware chunker: split on `##`, one chunk per section (sub-split
    above `MAX_CHUNK_TOKENS`), ids `<policy>#<heading-slug>#<n>`.
    """

    name = "structural"

    def chunk(self, policy_name: str, text: str) -> list[Chunk]:
        title = _title(text)
        chunks: list[Chunk] = []
        for heading, body in _sections(text):
            slug = _slugify(heading)
            context = f"{title} — {heading}"
            windows = _windows(body.split(), MAX_CHUNK_TOKENS, CHUNK_OVERLAP_TOKENS)
            for n, window in enumerate(windows):
                chunks.append(
                    Chunk(
                        id=f"{policy_name}#{slug}#{n}",
                        heading=heading,
                        content=f"{context}\n\n{' '.join(window)}",
                        meta={"chunker": self.name},
                    )
                )
        return chunks
