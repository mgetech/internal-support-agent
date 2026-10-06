"""The verifier: deterministic checks on the draft answer. A model never judges whether
an answer is safe to send.

Check 1, `citations_in_retrieval_set`: every `[chunk:<id>]` in the draft must be an id
that a `search_policies` call returned in this request. An id the model made up, or
remembered from somewhere else, fails the check.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict

CITATION_PATTERN = re.compile(r"\[chunk:([^\]\s]+)\]")

CITATIONS_CHECK = "citations_in_retrieval_set"


class Verdict(BaseModel):
    """The result of one check. It is stored as an evidence item."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    check: str
    passed: bool
    citations: list[str]
    unknown_citations: list[str]
    # what to tell the model when the check fails, so it can fix the draft
    objection: str | None


def parse_citations(text: str) -> list[str]:
    """The chunk ids cited in `text` as `[chunk:<id>]`, in order, each one once."""
    citations: list[str] = []
    for chunk_id in CITATION_PATTERN.findall(text):
        if chunk_id not in citations:
            citations.append(chunk_id)
    return citations


def verify_citations(draft: str, retrieved_chunk_ids: list[str]) -> Verdict:
    """Check that every citation in `draft` is in `retrieved_chunk_ids`. A draft with no
    citations passes this check. Whether a policy claim needs a citation is a different
    check.
    """
    citations = parse_citations(draft)
    unknown = [chunk_id for chunk_id in citations if chunk_id not in retrieved_chunk_ids]
    objection = None
    if unknown:
        cited = ", ".join(f"[chunk:{chunk_id}]" for chunk_id in unknown)
        objection = (
            f"Your answer cites {cited}, which search_policies did not return in this "
            "conversation. Cite only ids from a search result. Remove the claim, or "
            "search again."
        )
    return Verdict(
        check=CITATIONS_CHECK,
        passed=not unknown,
        citations=citations,
        unknown_citations=unknown,
        objection=objection,
    )
