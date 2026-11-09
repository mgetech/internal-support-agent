"""The graph nodes. Each node returns only the fields it changes. Evidence items are
added to the request context, where the tools and the model client also add theirs, and
the node returns the ones it caused.

A node that decides to escalate adds an `escalation` evidence item with the reason, and
sets `outcome`. The `refuse` node then writes the answer.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence
from typing import Any, Literal

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_function
from langgraph.errors import GraphBubbleUp
from langgraph.prebuilt import ToolNode
from openai.types.responses import Response
from pydantic import BaseModel, ConfigDict, ValidationError

from support_agent.config import DEFAULT_MAX_TOOL_CALLS_PER_REQUEST
from support_agent.decision_record import DecisionRecord, create_decision_record
from support_agent.guardrails.verifier import verify_citations
from support_agent.llm import LLMClient, LLMUnavailableError
from support_agent.prompts import get_prompt
from support_agent.redact import redact_secrets
from support_agent.request_context import get_request_context
from support_agent.state import AgentState, Family, Language, Risk
from support_agent.tools import Tool, is_gated_write
from support_agent.tracing import trace_tool_call

logger = logging.getLogger(__name__)

DEFAULT_LANGUAGE: Language = "en"

# the longest error message kept in the Decision Record
MAX_ERROR_MESSAGE = 500

REFUSAL_TEXT: dict[Language, str] = {
    "en": (
        "I can't help with this request. A person from HR or IT support can take it over. "
        "Please contact them directly."
    ),
    "de": (
        "Dabei kann ich nicht helfen. Eine Person aus dem HR- oder IT-Support kann das "
        "übernehmen. Bitte wenden Sie sich direkt an sie."
    ),
}


FAILURE_TEXT: dict[Language, str] = {
    "en": (
        "Something went wrong while I handled your request. HR or IT support can help you. "
        "Your request id is {request_id}."
    ),
    "de": (
        "Bei der Bearbeitung Ihrer Anfrage ist etwas schiefgelaufen. Der HR- oder IT-Support "
        "kann Ihnen helfen. Ihre Anfrage-ID lautet {request_id}."
    ),
}


class Classification(BaseModel):
    """The answer the classifier must give."""

    model_config = ConfigDict(extra="forbid")

    family: Family
    risk: Risk
    language: Language


CLASSIFICATION_FORMAT = {
    "type": "json_schema",
    "name": "classification",
    "strict": True,
    "schema": Classification.model_json_schema(),
}


def _evidence_since(start: int) -> list[dict[str, Any]]:
    """The evidence items added to the request after the list had `start` items."""
    return list(get_request_context().evidence[start:])


def guard_node(
    name: str, node: Callable[[AgentState], dict[str, Any]]
) -> Callable[[AgentState], dict[str, Any]]:
    """Run the node. If an error escapes it, the request ends in escalate instead of failing.

    The node's evidence so far, and a `node_error` item with the error type, the node name
    and the error message, are added to the state, and `failed_node` is set. The graph then
    goes to finalize, which saves the Decision Record. The employee gets a short reply with
    the request id. A proposal that the node made before the error is kept, so it has a
    Decision Record.

    The message is saved in the Decision Record only, with secrets removed (see redact.py).
    The log gets the type, the node and the request id, because the message may contain what
    the employee wrote.
    """

    def guarded(state: AgentState) -> dict[str, Any]:
        ctx = get_request_context()
        start = len(ctx.evidence)
        try:
            return node(state)
        except GraphBubbleUp:
            raise
        except Exception as error:
            logger.error(
                "node_failed node=%s error_type=%s request_id=%s",
                name,
                type(error).__name__,
                ctx.request_id,
            )
            ctx.evidence.append(
                {
                    "type": "node_error",
                    "node": name,
                    "error_type": type(error).__name__,
                    "error_message": redact_secrets(str(error))[:MAX_ERROR_MESSAGE],
                }
            )
            evidence = _evidence_since(start)
            known = state["proposed_action_ids"]
            proposed = [
                int(item["ref"].split(":", 1)[1])
                for item in evidence
                if item["type"] == "tool_result" and str(item["ref"]).startswith("pending_actions:")
            ]
            text = FAILURE_TEXT[state.get("language", DEFAULT_LANGUAGE)]
            return {
                "messages": [AIMessage(text.format(request_id=ctx.request_id))],
                "outcome": "escalate",
                "failed_node": name,
                "decision_evidence": evidence,
                "proposed_action_ids": [i for i in dict.fromkeys(proposed) if i not in known],
            }

    return guarded


def _get_last_user_text(state: AgentState) -> str:
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            return message.text
    raise ValueError("the state has no user message")


def classify_node(
    llm: LLMClient, model: str, reasoning_effort: str = ""
) -> Callable[[AgentState], dict[str, Any]]:
    """The classify node. One constrained call to `model`, the small deployment, with the
    given reasoning effort. A model outage or an answer that does not fit the schema ends
    in escalate: no guessing.
    """
    prompt = get_prompt("classifier")

    def classify(state: AgentState) -> dict[str, Any]:
        ctx = get_request_context()
        start = len(ctx.evidence)
        try:
            response = llm.create(
                model,
                [{"role": "user", "content": _get_last_user_text(state)}],
                prompt,
                instructions=prompt.text,
                text={"format": CLASSIFICATION_FORMAT},
                reasoning_effort=reasoning_effort,
            )
            result = Classification.model_validate_json(response.output_text)
        except LLMUnavailableError:
            # the model client has already recorded the failed tries
            ctx.evidence.append({"type": "escalation", "reason": "classification_failed"})
            return {"outcome": "escalate", "decision_evidence": _evidence_since(start)}
        except ValidationError as error:
            ctx.evidence.append(
                {
                    "type": "classification",
                    "status": "invalid",
                    "error": redact_secrets(str(error))[:200],
                }
            )
            ctx.evidence.append({"type": "escalation", "reason": "classification_failed"})
            return {"outcome": "escalate", "decision_evidence": _evidence_since(start)}

        ctx.evidence.append({"type": "classification", "status": "ok", **result.model_dump()})
        state_update: dict[str, Any] = result.model_dump()
        if result.risk == "out_of_scope":
            ctx.evidence.append({"type": "escalation", "reason": "out_of_scope"})
            state_update["outcome"] = "escalate"
        return {**state_update, "decision_evidence": _evidence_since(start)}

    return classify


def route_after_classify(state: AgentState) -> Literal["agent", "refuse"]:
    return "refuse" if state.get("outcome") == "escalate" else "agent"


def refuse_node(state: AgentState) -> dict[str, Any]:
    """Tell the employee a person can take over, in the language of the request."""
    text = REFUSAL_TEXT[state.get("language", DEFAULT_LANGUAGE)]
    return {"messages": [AIMessage(text)], "outcome": "escalate"}


def _to_response_input(messages: Sequence[AnyMessage]) -> list[dict[str, Any]]:
    """Turn the conversation into Responses API input items."""
    items: list[dict[str, Any]] = []
    for message in messages:
        if isinstance(message, HumanMessage):
            items.append({"role": "user", "content": message.text})
        elif isinstance(message, AIMessage):
            if message.text:
                items.append({"role": "assistant", "content": message.text})
            for call in message.tool_calls:
                items.append(
                    {
                        "type": "function_call",
                        "call_id": call["id"],
                        "name": call["name"],
                        "arguments": json.dumps(call["args"]),
                    }
                )
        elif isinstance(message, ToolMessage):
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": message.tool_call_id,
                    "output": message.text,
                }
            )
        else:
            raise ValueError(f"cannot send a {type(message).__name__} to the model")
    return items


def _to_ai_message(response: Response) -> AIMessage:
    """The model's reply as one AIMessage: its text and its tool calls."""
    tool_calls = []
    for item in response.output:
        if item.type == "function_call":
            tool_calls.append(
                {
                    "type": "tool_call",
                    "id": item.call_id,
                    "name": item.name,
                    "args": json.loads(item.arguments),
                }
            )
    return AIMessage(content=response.output_text, tool_calls=tool_calls)


def agent_node(
    llm: LLMClient, model: str, tools: Sequence[Tool], reasoning_effort: str = ""
) -> Callable[[AgentState], dict[str, Any]]:
    """The agent node: one model call with the tool belt and the given reasoning effort.
    The reply may ask for tool calls. A model outage, or tool arguments that are not valid
    JSON, ends in escalate.
    """
    prompt = get_prompt("agent_system")
    # the Responses API treats a tool as strict unless told otherwise
    tool_schemas = [
        {"type": "function", **convert_to_openai_function(tool), "strict": False} for tool in tools
    ]

    def agent(state: AgentState) -> dict[str, Any]:
        ctx = get_request_context()
        start = len(ctx.evidence)
        model_input = _to_response_input(state["messages"])
        if state["verifier_objection"]:
            # the retry: tell the model what was wrong with its last draft
            model_input.append({"role": "system", "content": state["verifier_objection"]})
        try:
            response = llm.create(
                model,
                model_input,
                prompt,
                instructions=prompt.text,
                tools=tool_schemas,
                reasoning_effort=reasoning_effort,
            )
            message = _to_ai_message(response)
        except LLMUnavailableError:
            ctx.evidence.append({"type": "escalation", "reason": "model_unavailable"})
            return {"outcome": "escalate", "decision_evidence": _evidence_since(start)}
        except json.JSONDecodeError:
            ctx.evidence.append({"type": "escalation", "reason": "invalid_tool_arguments"})
            return {"outcome": "escalate", "decision_evidence": _evidence_since(start)}
        return {"messages": [message], "decision_evidence": _evidence_since(start)}

    return agent


def route_after_agent(
    state: AgentState, max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS_PER_REQUEST
) -> Literal["tools", "verify", "tool_limit_reached", "refuse"]:
    """Decide what runs after the agent node.

    A reply with tool calls goes to `tools`, unless running them would use more than
    `max_tool_calls` in total. The calls of one reply are never cut to fit: either all
    of them run or none of them does.
    """
    if state.get("outcome") == "escalate":
        return "refuse"
    requested_calls = state["messages"][-1].tool_calls
    if not requested_calls:
        return "verify"
    if state["tool_calls_used"] + len(requested_calls) > max_tool_calls:
        return "tool_limit_reached"
    return "tools"


def tool_limit_node(
    state: AgentState, max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS_PER_REQUEST
) -> dict[str, Any]:
    """End the request with escalate because the tool limit is reached. The requested
    calls are not run. The `refuse` node then writes the answer.
    """
    ctx = get_request_context()
    start = len(ctx.evidence)
    ctx.evidence.append(
        {
            "type": "tool_limit",
            "tool_calls_used": state["tool_calls_used"],
            "tool_calls_requested": len(state["messages"][-1].tool_calls),
            "limit": max_tool_calls,
        }
    )
    ctx.evidence.append({"type": "escalation", "reason": "tool_limit_reached"})
    return {"outcome": "escalate", "decision_evidence": _evidence_since(start)}


def tools_node(tools: Sequence[Tool]) -> Callable[[AgentState], dict[str, Any]]:
    """The tools node: runs the requested tool calls with LangGraph's ToolNode. The
    tools have already added their evidence items. This node returns them, adds the
    chunk ids of every search to the citation whitelist and the ids of new proposals to
    `proposed_action_ids`, and counts the calls.
    """
    tool_node = ToolNode(tools, wrap_tool_call=trace_tool_call)
    gated_writes = {tool.__name__ for tool in tools if is_gated_write(tool)}

    def run_tools(state: AgentState) -> dict[str, Any]:
        start = len(get_request_context().evidence)
        calls = state["messages"][-1].tool_calls
        result = tool_node.invoke({"messages": state["messages"]})
        evidence = _evidence_since(start)

        chunk_ids: list[str] = []
        action_ids: list[int] = []
        for item in evidence:
            if item.get("type") != "tool_result":
                continue
            if item["tool"] == "search_policies" and isinstance(item["ref"], list):
                chunk_ids.extend(chunk["id"] for chunk in item["ref"])
            if item["tool"] in gated_writes and str(item["ref"]).startswith("pending_actions:"):
                action_id = int(item["ref"].split(":", 1)[1])
                if action_id not in state["proposed_action_ids"] + action_ids:
                    action_ids.append(action_id)

        return {
            "messages": result["messages"],
            "retrieved_chunk_ids": chunk_ids,
            "proposed_action_ids": action_ids,
            "tool_calls_used": state["tool_calls_used"] + len(calls),
            "decision_evidence": evidence,
        }

    return run_tools


def _get_declined_args(state: AgentState) -> dict[str, Any] | None:
    """The arguments of the `decline_request` call in the latest reply of the agent, if
    the tool accepted it. The tool answers "declined": false when it did not.
    """
    messages = state["messages"]
    last_reply = next((m for m in reversed(messages) if isinstance(m, AIMessage)), None)
    if last_reply is None:
        return None
    results = {m.tool_call_id: m.text for m in messages if isinstance(m, ToolMessage)}
    for call in last_reply.tool_calls:
        if call["name"] != "decline_request" or call["id"] not in results:
            continue
        try:
            declined = json.loads(results[call["id"]]).get("declined") is True
        except (json.JSONDecodeError, AttributeError):
            declined = False
        if declined:
            return call["args"]
    return None


def route_after_tools(state: AgentState) -> Literal["agent", "decline"]:
    """A refusal that the tool accepted goes to `decline`. Anything else goes back to the
    agent.
    """
    return "decline" if _get_declined_args(state) is not None else "agent"


def decline_node(state: AgentState) -> dict[str, Any]:
    """Turn the accepted `decline_request` call into the final answer: the reason, then
    the citations as `[chunk:<id>]`. It sets the outcome and the verify node checks the
    citations.
    """
    args = _get_declined_args(state)
    if args is None:
        raise ValueError("the decline node needs an accepted decline_request call")
    citations = list(dict.fromkeys(args["citations"]))
    cited = " ".join(f"[chunk:{chunk_id}]" for chunk_id in citations)
    text = f"{args['reason'].strip()} {cited}".strip()
    return {"messages": [AIMessage(text)], "outcome": "refuse_with_citation"}


def verify_node(state: AgentState) -> dict[str, Any]:
    """Run the deterministic checks on the draft answer, which is the last message.

    If the draft passes, the request ends: `refuse_with_citation` for a refusal (the
    `decline` node has set it), `propose_action` when this request proposed actions,
    `resolve` otherwise. A refusal must have at least one citation.

    If it fails, the agent gets one more try. The objection is saved in
    `verifier_objection` and `outcome` is unset again. If a draft fails while an
    objection is already saved, the retry has been used, and the request ends in
    `escalate`.
    """
    ctx = get_request_context()
    start = len(ctx.evidence)
    draft = state["messages"][-1].text
    is_refusal = state.get("outcome") == "refuse_with_citation"
    verdict = verify_citations(draft, state["retrieved_chunk_ids"], require_citation=is_refusal)
    ctx.evidence.append({"type": "verifier_verdict", **verdict.model_dump()})

    if verdict.passed:
        if is_refusal:
            outcome = "refuse_with_citation"
        else:
            outcome = "propose_action" if state["proposed_action_ids"] else "resolve"
        return {"outcome": outcome, "decision_evidence": _evidence_since(start)}

    if state["verifier_objection"] is None:
        ctx.evidence.append({"type": "retry", "reason": "citation_check_failed"})
        update: dict[str, Any] = {
            "verifier_objection": verdict.objection,
            "decision_evidence": _evidence_since(start),
        }
        if is_refusal:
            # the `decline` node set the outcome. The retry starts without one.
            update["outcome"] = None
        return update

    ctx.evidence.append({"type": "escalation", "reason": "citation_check_failed"})
    return {"outcome": "escalate", "decision_evidence": _evidence_since(start)}


def finalize_node(
    state: AgentState,
    save_record: Callable[[DecisionRecord], None],
    get_policy_version: Callable[[], str | None],
) -> dict[str, Any]:
    """The last node, on every path. It reads the version of the policy corpus, builds
    the Decision Record from the final state and saves it with `save_record`. It changes
    nothing in the state.
    """
    save_record(create_decision_record(state, get_policy_version()))
    return {}


def route_after_verify(state: AgentState) -> Literal["agent", "refuse", "end"]:
    outcome = state.get("outcome")
    if outcome == "escalate":
        return "refuse"
    if outcome is None:
        # no outcome after a verify means the first draft failed: retry
        return "agent"
    return "end"
