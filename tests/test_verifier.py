"""The verifier and the verify node.

The verifier is plain code with no model and no database, so the first tests call it
directly with a text and a list of chunk ids.

The verify node runs it on the agent's draft answer (the last message) and sets the
outcome. The graph tests at the end run the whole graph with the stand-ins from
tests/graph_helpers.py (ScriptedLLM, text_reply, call_reply, run_graph).
"""

from langchain_core.messages import AIMessage
from tests.graph_helpers import (
    CHUNK,
    CLASSIFIED,
    ScriptedLLM,
    call_reply,
    run_graph,
    text_reply,
)

from support_agent.guardrails.verifier import parse_citations, verify_citations
from support_agent.nodes import REFUSAL_TEXT, route_after_verify, verify_node
from support_agent.request_context import bind_request_context

OTHER_CHUNK = "it-policy#access#0"


# --- parse_citations: finding the ids in a text ---


def test_a_citation_is_found_in_the_text():
    text = f"You get 30 days [chunk:{CHUNK}]."

    assert parse_citations(text) == [CHUNK]


def test_several_citations_are_returned_in_the_order_they_appear():
    text = f"First [chunk:{OTHER_CHUNK}], then [chunk:{CHUNK}]."

    assert parse_citations(text) == [OTHER_CHUNK, CHUNK]


def test_a_citation_that_appears_twice_is_returned_once():
    text = f"[chunk:{CHUNK}] and again [chunk:{CHUNK}]"

    assert parse_citations(text) == [CHUNK]


def test_text_without_citations_gives_an_empty_list():
    assert parse_citations("You have 18 days left.") == []


def test_only_the_exact_chunk_format_counts_as_a_citation():
    text = "[chunk:] [source:abc] chunk:abc (chunk:abc) [chunk: abc]"

    assert parse_citations(text) == []


# --- verify_citations: are the cited ids in this request's retrieval set? ---


def test_a_citation_from_the_retrieval_set_passes():
    verdict = verify_citations(f"30 days [chunk:{CHUNK}]", retrieved_chunk_ids=[CHUNK])

    assert verdict.passed is True
    assert verdict.citations == [CHUNK]
    assert verdict.unknown_citations == []
    assert verdict.objection is None


def test_a_citation_that_was_not_retrieved_fails():
    # the search returned CHUNK, but the draft cites another id
    verdict = verify_citations(f"[chunk:{OTHER_CHUNK}]", retrieved_chunk_ids=[CHUNK])

    assert verdict.passed is False
    assert verdict.unknown_citations == [OTHER_CHUNK]


def test_one_bad_citation_fails_the_draft_even_when_the_others_are_fine():
    draft = f"[chunk:{CHUNK}] and [chunk:{OTHER_CHUNK}]"

    verdict = verify_citations(draft, retrieved_chunk_ids=[CHUNK])

    assert verdict.passed is False
    assert verdict.citations == [CHUNK, OTHER_CHUNK]
    assert verdict.unknown_citations == [OTHER_CHUNK]


def test_any_citation_fails_when_nothing_was_retrieved():
    # the model never searched, so every id it cites was made up or remembered
    verdict = verify_citations(f"[chunk:{CHUNK}]", retrieved_chunk_ids=[])

    assert verdict.passed is False


def test_a_draft_without_citations_passes_this_check():
    # whether a policy claim needs a citation is a different check, added later
    verdict = verify_citations("You have 18 days left.", retrieved_chunk_ids=[])

    assert verdict.passed is True
    assert verdict.citations == []


def test_the_objection_names_the_bad_id_and_tells_the_model_what_to_do():
    verdict = verify_citations(f"[chunk:{OTHER_CHUNK}]", retrieved_chunk_ids=[CHUNK])

    assert f"[chunk:{OTHER_CHUNK}]" in verdict.objection
    assert "search_policies" in verdict.objection


# --- the verify node, called directly ---
#
# A failed draft gets one retry. The first failure saves the objection in the state's
# `verifier_objection` and leaves `outcome` unset, so the router sends the request back
# to the agent. A failure while `verifier_objection` is already set means the retry was
# used, so the request escalates. The tests below set `verifier_objection` to say which
# of the two drafts they are checking.

EARLIER_OBJECTION = "the objection from the first failed draft"


def draft_state(draft, retrieved_chunk_ids=(), proposed_action_ids=(), verifier_objection=None):
    """The state the verify node sees: the draft answer is the last message.
    `verifier_objection` is None for a first draft, and set for the retried draft.
    """
    return {
        "messages": [AIMessage(draft)],
        "retrieved_chunk_ids": list(retrieved_chunk_ids),
        "proposed_action_ids": list(proposed_action_ids),
        "verifier_objection": verifier_objection,
    }


def run_verify(state):
    """Run the verify node in a bound request. Returns the state update."""
    with bind_request_context("emp_001", "req-1", "rest"):
        return verify_node(state)


def test_a_passing_draft_without_proposals_resolves():
    state_update = run_verify(draft_state("You have 18 days left."))

    assert state_update["outcome"] == "resolve"


def test_a_passing_draft_with_proposals_ends_in_propose_action():
    state = draft_state("I sent your request for approval.", proposed_action_ids=[5])

    assert run_verify(state)["outcome"] == "propose_action"


def test_a_failing_retried_draft_escalates_even_when_there_are_proposals():
    # propose_action needs the verification to pass
    state = draft_state(
        f"[chunk:{CHUNK}]",
        retrieved_chunk_ids=[],
        proposed_action_ids=[5],
        verifier_objection=EARLIER_OBJECTION,
    )

    assert run_verify(state)["outcome"] == "escalate"


def test_a_failing_first_draft_saves_the_objection_and_sets_no_outcome():
    state = draft_state(f"[chunk:{CHUNK}]", retrieved_chunk_ids=[])

    state_update = run_verify(state)

    assert "outcome" not in state_update
    # the saved objection is the one the verifier wrote for this draft
    assert state_update["verifier_objection"] == verify_citations(f"[chunk:{CHUNK}]", []).objection
    assert f"[chunk:{CHUNK}]" in state_update["verifier_objection"]


def test_the_verdict_is_an_evidence_item():
    state = draft_state(f"[chunk:{CHUNK}]", retrieved_chunk_ids=[CHUNK])

    (verdict,) = run_verify(state)["decision_evidence"]

    assert verdict == {
        "type": "verifier_verdict",
        "check": "citations_in_retrieval_set",
        "passed": True,
        "citations": [CHUNK],
        "unknown_citations": [],
        "objection": None,
    }


def test_a_failing_first_draft_is_followed_by_a_retry_item():
    state = draft_state(f"[chunk:{CHUNK}]", retrieved_chunk_ids=[])

    verdict, retry = run_verify(state)["decision_evidence"]

    assert verdict["passed"] is False
    assert verdict["unknown_citations"] == [CHUNK]
    assert retry == {"type": "retry", "reason": "citation_check_failed"}


def test_a_failing_retried_draft_is_followed_by_an_escalation_item():
    state = draft_state(
        f"[chunk:{CHUNK}]", retrieved_chunk_ids=[], verifier_objection=EARLIER_OBJECTION
    )

    verdict, escalation = run_verify(state)["decision_evidence"]

    assert verdict["passed"] is False
    assert escalation == {"type": "escalation", "reason": "citation_check_failed"}


def test_a_retried_draft_that_passes_resolves():
    # the objection stays in the state after the retry. A pass still ends the request.
    state = draft_state("You have 18 days left.", verifier_objection=EARLIER_OBJECTION)

    assert run_verify(state)["outcome"] == "resolve"


def test_where_the_request_goes_after_the_verify_node():
    # passed: the request ends
    assert route_after_verify({"outcome": "resolve"}) == "end"
    assert route_after_verify({"outcome": "propose_action"}) == "end"
    # failed twice: refuse
    assert route_after_verify({"outcome": "escalate"}) == "refuse"
    # failed once: no outcome yet, so back to the agent for the retry
    assert route_after_verify({"outcome": None}) == "agent"


# --- the whole graph ---


def test_an_answer_that_cites_a_chunk_from_this_requests_search_is_accepted():
    llm = ScriptedLLM(
        CLASSIFIED,
        call_reply("call_1", "search_policies", '{"query": "vacation"}'),
        text_reply(f"You get 30 days [chunk:{CHUNK}]."),
    )

    result, _, _ = run_graph(llm)

    assert result["outcome"] == "resolve"
    assert result["messages"][-1].content == f"You get 30 days [chunk:{CHUNK}]."
    assert result["decision_evidence"][-1]["type"] == "verifier_verdict"
    assert result["decision_evidence"][-1]["passed"] is True


def test_an_answer_that_cites_an_id_nobody_searched_for_is_escalated_after_one_retry():
    # the model never calls search_policies, but it cites a chunk. It does it twice.
    bad_draft = text_reply(f"You get 30 days [chunk:{CHUNK}].")
    llm = ScriptedLLM(CLASSIFIED, bad_draft, bad_draft)

    result, _, _ = run_graph(llm)

    assert result["outcome"] == "escalate"
    # the employee does not get the unverified answer. They get the refusal text.
    assert result["messages"][-1].content == REFUSAL_TEXT["en"]
    assert [item["type"] for item in result["decision_evidence"]] == [
        "classification",
        "verifier_verdict",  # the first draft fails
        "retry",
        "verifier_verdict",  # the retried draft fails
        "escalation",
    ]
    assert result["decision_evidence"][1]["unknown_citations"] == [CHUNK]


def test_an_id_that_the_search_did_not_return_is_not_accepted():
    # the search returns CHUNK, but both drafts cite a different (real-looking) id
    bad_draft = text_reply(f"You can use the laptop [chunk:{OTHER_CHUNK}].")
    llm = ScriptedLLM(
        CLASSIFIED,
        call_reply("call_1", "search_policies", '{"query": "vacation"}'),
        bad_draft,
        bad_draft,
    )

    result, _, _ = run_graph(llm)

    assert result["retrieved_chunk_ids"] == [CHUNK]
    assert result["outcome"] == "escalate"


def test_a_request_with_a_proposal_and_a_passing_answer_ends_in_propose_action():
    llm = ScriptedLLM(
        CLASSIFIED,
        call_reply("call_1", "create_ticket", '{"title": "laptop"}'),
        text_reply("I sent your ticket for approval."),
    )

    result, _, _ = run_graph(llm)

    assert result["proposed_action_ids"] == [5]
    assert result["outcome"] == "propose_action"


def test_a_proposal_does_not_save_an_answer_that_fails_the_check_twice():
    bad_draft = text_reply(f"Sent for approval, see [chunk:{CHUNK}].")
    llm = ScriptedLLM(
        CLASSIFIED,
        call_reply("call_1", "create_ticket", '{"title": "laptop"}'),
        bad_draft,
        bad_draft,
    )

    result, _, _ = run_graph(llm)

    # the proposal is still in the database, waiting for a person. The answer is not sent.
    assert result["proposed_action_ids"] == [5]
    assert result["outcome"] == "escalate"


# --- the retry, in the whole graph ---
#
# In these tests the first draft cites an id that no search returned, so it fails. The
# agent then gets one more try, with the objection added to its input as a system message.

BAD_DRAFT = text_reply(f"You get 30 days [chunk:{CHUNK}].")


def test_a_failed_first_draft_is_retried_and_a_good_second_draft_is_accepted():
    llm = ScriptedLLM(CLASSIFIED, BAD_DRAFT, text_reply("You have 18 days left."))

    result, _, _ = run_graph(llm)

    assert result["outcome"] == "resolve"
    # the employee gets the second draft
    assert result["messages"][-1].content == "You have 18 days left."
    assert [item["type"] for item in result["decision_evidence"]] == [
        "classification",
        "verifier_verdict",  # the first draft fails
        "retry",
        "verifier_verdict",  # the second draft passes
    ]
    assert [item["passed"] for item in result["decision_evidence"] if "passed" in item] == [
        False,
        True,
    ]


def test_the_first_model_call_of_the_agent_has_no_objection():
    llm = ScriptedLLM(CLASSIFIED, BAD_DRAFT, text_reply("You have 18 days left."))

    run_graph(llm)

    first_agent_input = llm.calls[1]["input"]
    assert all(item.get("role") != "system" for item in first_agent_input)


def test_the_retry_sends_the_objection_as_the_last_system_message():
    llm = ScriptedLLM(CLASSIFIED, BAD_DRAFT, text_reply("You have 18 days left."))

    result, _, _ = run_graph(llm)

    objection = verify_citations(f"You get 30 days [chunk:{CHUNK}].", []).objection
    retry_input = llm.calls[2]["input"]
    # the conversation so far, including the failed draft, then the objection
    assert retry_input == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": f"You get 30 days [chunk:{CHUNK}]."},
        {"role": "system", "content": objection},
    ]
    assert result["verifier_objection"] == objection


def test_the_retried_agent_can_search_and_then_cite_what_it_found():
    llm = ScriptedLLM(
        CLASSIFIED,
        BAD_DRAFT,
        call_reply("call_1", "search_policies", '{"query": "vacation"}'),
        text_reply(f"You get 30 days [chunk:{CHUNK}]."),
    )

    result, _, _ = run_graph(llm)

    assert result["retrieved_chunk_ids"] == [CHUNK]
    assert result["outcome"] == "resolve"
    # the objection is added to every model call after the failure, including this one
    last_input = llm.calls[3]["input"]
    assert last_input[-1]["role"] == "system"


def test_a_draft_is_retried_exactly_once():
    # The script has no reply after the two drafts. A third agent call would fail with an
    # IndexError, and a classify call would be a fourth call too.
    llm = ScriptedLLM(CLASSIFIED, BAD_DRAFT, BAD_DRAFT)

    result, _, _ = run_graph(llm)

    assert len(llm.calls) == 3  # classify, first draft, retried draft
    assert result["outcome"] == "escalate"
    assert result["messages"][-1].content == REFUSAL_TEXT["en"]


def test_a_request_that_passes_the_first_time_is_never_retried():
    llm = ScriptedLLM(CLASSIFIED, text_reply("You have 18 days left."))

    result, _, _ = run_graph(llm)

    assert len(llm.calls) == 2  # classify, agent
    assert result["verifier_objection"] is None
    assert "retry" not in [item["type"] for item in result["decision_evidence"]]


def test_a_proposal_made_before_a_retry_is_kept():
    llm = ScriptedLLM(
        CLASSIFIED,
        call_reply("call_1", "create_ticket", '{"title": "laptop"}'),
        BAD_DRAFT,
        text_reply("I sent your ticket for approval."),
    )

    result, _, _ = run_graph(llm)

    assert result["proposed_action_ids"] == [5]
    assert result["outcome"] == "propose_action"
