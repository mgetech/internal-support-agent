"""The graph state. Fields with a reducer are added to by each node. The others are
replaced by the last node that sets them.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.graph.message import add_messages

Family = Literal["hr", "it", "other"]
Risk = Literal["read", "write", "out_of_scope"]
Language = Literal["de", "en"]
Outcome = Literal["resolve", "propose_action", "escalate", "refuse_with_citation"]


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    # family, risk and language are set once, by the classify node
    family: Family
    risk: Risk
    language: Language
    # the citation whitelist: every chunk id a search_policies call returned in this request
    retrieved_chunk_ids: Annotated[list[str], operator.add]
    proposed_action_ids: Annotated[list[int], operator.add]
    tool_calls_used: int
    outcome: Outcome | None
    # set by the verifier when a draft fails; the agent node injects it, then it is cleared
    verifier_objection: str | None
    # ordered evidence items for the Decision Record
    decision_evidence: Annotated[list[dict[str, Any]], operator.add]


def new_state(text: str) -> AgentState:
    """The state a request starts with: the employee's message and empty accumulators."""
    return {
        "messages": [HumanMessage(text)],
        "retrieved_chunk_ids": [],
        "proposed_action_ids": [],
        "tool_calls_used": 0,
        "outcome": None,
        "verifier_objection": None,
        "decision_evidence": [],
    }
