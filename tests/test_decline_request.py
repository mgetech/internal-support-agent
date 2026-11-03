"""Refusal with a citation. The agent ends a request by calling `decline_request`. The
graph turns the call into the answer and sets `refuse_with_citation`, and the verifier
needs a citation from this request's search. A refusal without one gets one retry, then
escalates. The tests run the whole graph with a scripted model and fake tools, so they
need no database.
"""

import json

import pytest
from tests.graph_helpers import (
    CHUNK,
    CLASSIFIED,
    ScriptedLLM,
    call_reply,
    run_graph,
    text_reply,
)

from support_agent.guardrails.verifier import verify_citations
from support_agent.nodes import REFUSAL_TEXT

REASON = "Your request is for 10 days and you have 3.5 left."


def search():
    return call_reply("search-1", "search_policies", '{"query": "vacation entitlement"}')


def decline(call_id="decline-1", reason=REASON, citations=(CHUNK,)):
    arguments = json.dumps({"reason": reason, "citations": list(citations)})
    return call_reply(call_id, "decline_request", arguments)


def run(llm):
    records = []
    result, tool_calls, evidence = run_graph(llm, saved_records=records)
    return result, tool_calls, evidence, records


# --- the refusal passes the check ---


def test_a_refusal_with_a_valid_citation_ends_in_refuse_with_citation():
    llm = ScriptedLLM(CLASSIFIED, search(), decline())

    result, tool_calls, _, records = run(llm)

    assert result["outcome"] == "refuse_with_citation"
    assert result["messages"][-1].text == f"{REASON} [chunk:{CHUNK}]"
    assert [name for name, _ in tool_calls] == ["search_policies", "decline_request"]
    # the classifier and two agent calls: nothing is asked of the model after the refusal
    assert len(llm.calls) == 3
    assert len(records) == 1
    assert records[0].outcome == "refuse_with_citation"
    assert records[0].citations == [CHUNK]
    assert records[0].summary == (
        f"Refused because the request goes against policy or permissions, citing {CHUNK}."
    )
    assert [verdict["passed"] for verdict in records[0].verifier] == [True]


def test_a_citation_is_written_once_when_the_agent_lists_it_twice():
    llm = ScriptedLLM(CLASSIFIED, search(), decline(citations=[CHUNK, CHUNK]))

    result, _, _, _ = run(llm)

    assert result["messages"][-1].text.count(f"[chunk:{CHUNK}]") == 1


# --- the refusal fails the check: one retry, then escalate ---


def test_a_refusal_without_a_citation_gets_one_retry_and_then_escalates():
    llm = ScriptedLLM(
        CLASSIFIED,
        search(),
        decline("decline-1", citations=[]),
        decline("decline-2", citations=[]),
    )

    result, _, _, records = run(llm)

    assert result["outcome"] == "escalate"
    assert result["messages"][-1].text == REFUSAL_TEXT["en"]
    assert len(llm.calls) == 4
    # the agent was told what was wrong before its second try
    objection = llm.calls[3]["input"][-1]
    assert objection["role"] == "system"
    assert "no citation" in objection["content"]
    assert len(records) == 1
    assert records[0].outcome == "escalate"
    assert [verdict["passed"] for verdict in records[0].verifier] == [False, False]
    assert "retry" in [item["type"] for item in records[0].evidence]
    assert "citations failed the check" in records[0].summary


def test_a_refusal_with_a_citation_that_was_not_retrieved_is_not_accepted():
    llm = ScriptedLLM(
        CLASSIFIED,
        search(),
        decline("decline-1", citations=["it-access-policy#vpn-access#0"]),
        decline("decline-2", citations=["it-access-policy#vpn-access#0"]),
    )

    result, _, _, records = run(llm)

    assert result["outcome"] == "escalate"
    assert records[0].verifier[0]["unknown_citations"] == ["it-access-policy#vpn-access#0"]


def test_a_refusal_before_any_search_cannot_cite_anything():
    llm = ScriptedLLM(CLASSIFIED, decline("decline-1"), decline("decline-2"))

    result, _, _, _ = run(llm)

    assert result["outcome"] == "escalate"


def test_the_retry_can_fix_the_refusal():
    llm = ScriptedLLM(
        CLASSIFIED,
        search(),
        decline("decline-1", citations=[]),
        decline("decline-2", citations=[CHUNK]),
    )

    result, _, _, records = run(llm)

    assert result["outcome"] == "refuse_with_citation"
    assert records[0].citations == [CHUNK]
    assert [verdict["passed"] for verdict in records[0].verifier] == [False, True]


# --- a refusal the tool did not accept does not end the request ---


def test_a_blank_reason_does_not_end_the_request():
    llm = ScriptedLLM(
        CLASSIFIED,
        search(),
        decline("decline-1", reason="  "),
        text_reply("I cannot give you that."),
    )

    result, _, _, _ = run(llm)

    # the tool said "declined": false, so the graph went back to the agent
    assert result["outcome"] == "resolve"
    assert len(llm.calls) == 4


# --- the verifier rule on its own ---


def test_the_verifier_requires_a_citation_only_when_asked():
    assert verify_citations("No.", [CHUNK]).passed
    verdict = verify_citations("No.", [CHUNK], require_citation=True)

    assert not verdict.passed
    assert "decline_request" in verdict.objection
    assert verify_citations(f"No [chunk:{CHUNK}]", [CHUNK], require_citation=True).passed


@pytest.mark.parametrize("retrieved", [[], ["other#heading#0"]])
def test_the_verifier_still_rejects_an_unknown_citation_in_a_refusal(retrieved):
    verdict = verify_citations(f"No [chunk:{CHUNK}]", retrieved, require_citation=True)

    assert not verdict.passed
    assert verdict.unknown_citations == [CHUNK]
