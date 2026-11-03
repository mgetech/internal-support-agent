"""The tool that ends a request with a refusal. It changes no data and reads none. The
graph does the rest: it turns the call into the final answer and the verifier checks the
citations.
"""

from __future__ import annotations

import json
import time

from support_agent.request_context import get_request_context
from support_agent.tools.base import record_tool_call


def decline_request(reason: str, citations: list[str]) -> str:
    """End the request with a refusal. Use it when the request goes against a policy or
    against the requester's permissions. The requester sees your reason as the answer,
    followed by the citations.

    reason: why you refuse, written as the answer to the requester, in their language. Do
    not write the citations in it.
    citations: the chunk ids of the policy sections that the refusal rests on, copied from
    a search_policies result in this conversation. At least one is needed.

    The result has "declined": true if the request ended. If it has "declined": false, the
    request did not end. Read the "error".
    """
    started_at = time.perf_counter()
    get_request_context()
    reason = reason.strip()
    if reason:
        result = {"declined": True}
        summary = f"declined, {len(citations)} citations"
    else:
        result = {"declined": False, "error": "the reason is empty"}
        summary = "not declined: the reason is empty"
    record_tool_call(
        "decline_request", {"reason": reason, "citations": citations}, summary, None, started_at
    )
    return json.dumps(result)
