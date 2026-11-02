"""Hybrid policy search: a vector ranking and a keyword ranking, run separately and
merged with reciprocal rank fusion.
"""

from __future__ import annotations

import json
import time

from support_agent import db
from support_agent.embeddings import embed_texts
from support_agent.request_context import get_request_context
from support_agent.tools.base import record_tool_call

RRF_K = 60
CANDIDATES_PER_RANKING = 20
TOP_K = 5


def fuse(rankings: list[list[str]], k: int = RRF_K) -> list[str]:
    """Reciprocal rank fusion: each id scores sum(1 / (k + rank)) over the rankings it
    appears in. Ties keep the order in which ids were first seen.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda cid: scores[cid], reverse=True)


def get_policy_version() -> str | None:
    """The versions of the policies that search runs against, read from the chunks that
    `search_policies` reads: every policy with its version, sorted by policy name, like
    "expense-policy@2026-09.1; it-access-policy@2026-11.1". It is None when there are no
    chunks. A policy whose chunks come from more than one version, for example after a
    partial re-seed, is listed once for each version, so a Decision Record never names a
    version that was not used.
    """
    rows = db.fetch_all(
        """
        SELECT DISTINCT policy, policy_version
        FROM policy_chunks
        WHERE chunker = 'structural'
        ORDER BY policy, policy_version
        """
    )
    return "; ".join(f"{row['policy']}@{row['policy_version']}" for row in rows) or None


def get_policy_chunk(chunk_id: str) -> dict[str, str]:
    """One policy section by its chunk id, as the answer cites it. Raises LookupError if
    there is no such chunk.
    """
    row = db.fetch_one(
        """
        SELECT id, policy, heading, content, policy_version
        FROM policy_chunks
        WHERE id = %s AND chunker = 'structural'
        """,
        (chunk_id,),
    )
    if row is None:
        raise LookupError(f"no policy chunk {chunk_id}")
    return row


def search_policies(query: str) -> str:
    """Search the HR and IT policies and return the 5 most relevant passages, each with
    its chunk id. Any claim you make from a passage must cite it as [chunk:<id>], using
    only ids returned here. If nothing returned answers the question, say so instead
    of guessing.
    """
    started_at = time.perf_counter()
    get_request_context()
    (embedding,) = embed_texts([query])

    by_vector = db.fetch_all(
        """
        SELECT id
        FROM policy_chunks
        WHERE chunker = 'structural' AND embedding IS NOT NULL
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """,
        (str(embedding), CANDIDATES_PER_RANKING),
    )
    # plainto_tsquery ANDs every word, which almost never matches a natural question;
    # swapping to OR lets any shared term count and leaves the ranking to ts_rank_cd
    by_keyword = db.fetch_all(
        """
        WITH q AS (
            SELECT replace(plainto_tsquery('english', %s)::text, '&', '|')::tsquery AS q
        )
        SELECT id
        FROM policy_chunks, q
        WHERE chunker = 'structural' AND q.q <> ''::tsquery AND ts @@ q.q
        ORDER BY ts_rank_cd(ts, q.q) DESC, id
        LIMIT %s
        """,
        (query, CANDIDATES_PER_RANKING),
    )

    top = fuse([[r["id"] for r in by_vector], [r["id"] for r in by_keyword]])[:TOP_K]
    rows = db.fetch_all(
        "SELECT id, policy, heading, content, policy_version FROM policy_chunks WHERE id = ANY(%s)",
        (top,),
    )
    by_id = {r["id"]: r for r in rows}
    summary = f"{len(top)} chunks found" if top else "no chunks found"
    chunks = [{"id": cid, "policy_version": by_id[cid]["policy_version"]} for cid in top]
    record_tool_call("search_policies", {"query": query}, summary, chunks, started_at)
    if not top:
        return json.dumps({"results": []})

    return json.dumps(
        {
            "results": [
                {
                    "chunk_id": cid,
                    "policy": by_id[cid]["policy"],
                    "heading": by_id[cid]["heading"],
                    "content": by_id[cid]["content"],
                }
                for cid in top
            ]
        }
    )
