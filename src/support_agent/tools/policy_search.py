"""Hybrid policy search: a vector ranking and a keyword ranking, run separately and
merged with reciprocal rank fusion.
"""

from __future__ import annotations

import json

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
    """The version of the policy corpus that search runs against, read from the chunks
    that `search_policies` reads. It is None when there are no chunks. If the chunks come
    from more than one version, for example after a partial re-seed, every version is
    returned, joined with ", ", so a Decision Record never names a version that was not
    used.
    """
    rows = db.fetch_all(
        """
        SELECT DISTINCT policy_version
        FROM policy_chunks
        WHERE chunker = 'structural'
        ORDER BY policy_version
        """
    )
    return ", ".join(row["policy_version"] for row in rows) or None


def search_policies(query: str) -> str:
    """Search the HR and IT policies and return the 5 most relevant passages, each with
    its chunk id. Any claim you make from a passage must cite it as [chunk:<id>], using
    only ids returned here. If nothing returned answers the question, say so instead
    of guessing.
    """
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
    summary = f"{len(top)} chunks found" if top else "no chunks found"
    record_tool_call("search_policies", {"query": query}, summary, top)
    if not top:
        return json.dumps({"results": []})

    rows = db.fetch_all(
        "SELECT id, policy, heading, content FROM policy_chunks WHERE id = ANY(%s)", (top,)
    )
    by_id = {r["id"]: r for r in rows}
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
