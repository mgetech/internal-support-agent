"""Decision Records: the stored answer to "why did the agent do that?". One record per
request, built from the final graph state and written with the final audit event in one
transaction. The record does not depend on the trace; it is the system of record.
"""

from __future__ import annotations

import time
from typing import Any

from psycopg.types.json import Jsonb
from pydantic import BaseModel, ConfigDict

from support_agent import audit, db
from support_agent.costing import model_calls, request_total_cost
from support_agent.guardrails.approval import APPROVER_ROLES
from support_agent.request_context import Channel, get_request_context
from support_agent.state import AgentState, Outcome

# the words for each `escalation` evidence reason, used in the summary sentence
ESCALATION_REASONS = {
    "out_of_scope": "the request is outside HR and IT support",
    "classification_failed": "the request could not be classified",
    "model_unavailable": "no model answered",
    "invalid_tool_arguments": "the model sent tool arguments that were not valid JSON",
    "tool_limit_reached": "the tool limit was reached",
    "citation_check_failed": "the answer cited a source this request did not retrieve",
}


class DecisionRecord(BaseModel):
    """One row of the `decision_records` table."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    request_id: str
    employee_id: str
    channel: Channel
    outcome: Outcome
    summary: str
    evidence: list[dict[str, Any]]
    citations: list[str]
    policy_version: str | None
    prompt_versions: dict[str, str]
    verifier: list[dict[str, Any]]
    model_calls: list[dict[str, Any]]
    total_tokens: int
    total_cost_eur: float
    latency_ms: int
    trace_id: str | None


def summarize(
    outcome: Outcome,
    evidence: list[dict[str, Any]],
    citations: list[str],
    proposed_action_ids: list[int],
) -> str:
    """One sentence that says why the request ended the way it did. It is built from the
    evidence with fixed wording; no model writes it.
    """
    if outcome == "escalate":
        reasons = [item["reason"] for item in evidence if item["type"] == "escalation"]
        why = ESCALATION_REASONS.get(reasons[-1] if reasons else "", "a person needs to decide")
        return f"Escalated to a person: {why}."

    if outcome == "propose_action":
        ids = ", ".join(str(action_id) for action_id in proposed_action_ids)
        return f"Proposed for approval (pending action {ids}); nothing was changed yet."

    tools = []
    for item in evidence:
        if item["type"] == "tool_result" and item["tool"] not in tools:
            tools.append(item["tool"])
    sentence = f"Resolved using {', '.join(tools) if tools else 'no tools'}"
    if citations:
        sentence += f", citing {', '.join(citations)}"
    return sentence + "."


def create_decision_record(state: AgentState, policy_version: str | None = None) -> DecisionRecord:
    """Build the record for the bound request from the final state. Needs a bound request
    context. A request that reaches this point must have an outcome.
    """
    ctx = get_request_context()
    outcome = state["outcome"]
    if outcome is None:
        raise ValueError("a request cannot end without an outcome")

    evidence = state["decision_evidence"]
    verdicts = [item for item in evidence if item["type"] == "verifier_verdict"]

    # the citations of the answer the employee got: those of the last verdict, if it passed
    citations = verdicts[-1]["citations"] if verdicts and verdicts[-1]["passed"] else []

    prompt_versions: dict[str, str] = {}
    for item in evidence:
        if item["type"] == "model_call" and item["prompt"]:
            prompt_versions[item["prompt"]["prompt_id"]] = item["prompt"]["prompt_version"]

    totals = request_total_cost(evidence)
    return DecisionRecord(
        request_id=ctx.request_id,
        employee_id=ctx.employee_id,
        channel=ctx.channel,
        outcome=outcome,
        summary=summarize(outcome, evidence, citations, state["proposed_action_ids"]),
        evidence=evidence,
        citations=citations,
        policy_version=policy_version,
        prompt_versions=prompt_versions,
        verifier=verdicts,
        model_calls=model_calls(evidence),
        total_tokens=totals.tokens,
        total_cost_eur=totals.cost_eur,
        latency_ms=round((time.perf_counter() - ctx.started_at) * 1000),
        # there is no tracing yet
        trace_id=None,
    )


def save_decision_record(record: DecisionRecord) -> None:
    """Insert the record and write the final `decision_recorded` audit event in one
    transaction: both are saved, or neither is.
    """
    with db.transaction() as cur:
        cur.execute(
            """
            INSERT INTO decision_records (
                request_id, employee_id, channel, outcome, summary, evidence, citations,
                policy_version, prompt_versions, verifier, model_calls, total_tokens,
                total_cost_eur, latency_ms, trace_id
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                record.request_id,
                record.employee_id,
                record.channel,
                record.outcome,
                record.summary,
                Jsonb(record.evidence),
                record.citations,
                record.policy_version,
                Jsonb(record.prompt_versions),
                Jsonb(record.verifier),
                Jsonb(record.model_calls),
                record.total_tokens,
                record.total_cost_eur,
                record.latency_ms,
                record.trace_id,
            ),
        )
        audit.record(
            "decision_recorded",
            {"outcome": record.outcome, "summary": record.summary},
            actor="system",
            conn=cur,
        )


def get_decision_record(request_id: str, reader_id: str) -> DecisionRecord:
    """Read one record back. The employee who made the request may read it, and so may
    an approver. Raises LookupError if there is no such record, or if the reader may not
    read it: the two cases look the same, so a request id cannot be used to find out
    what exists.
    """
    row = db.fetch_one(
        """
        SELECT request_id, employee_id, channel, outcome, summary, evidence, citations,
               policy_version, prompt_versions, verifier, model_calls, total_tokens,
               total_cost_eur, latency_ms, trace_id
        FROM decision_records
        WHERE request_id = %s
          AND (
              employee_id = %s
              OR EXISTS (SELECT 1 FROM employees WHERE id = %s AND role = ANY(%s))
          )
        """,
        (request_id, reader_id, reader_id, list(APPROVER_ROLES)),
    )
    if row is None:
        raise LookupError(f"no decision record {request_id}")
    return DecisionRecord(**row)
