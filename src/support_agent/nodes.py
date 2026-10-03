"""The graph nodes. Each node returns only the fields it changes. Evidence items are
added to the request context, where the tools and the model client also add theirs, and
the node returns the ones it caused.

A node that decides to escalate adds an `escalation` evidence item with the reason, and
sets `outcome`. The `refuse` node then writes the answer.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import Any, Literal

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_function
from langgraph.prebuilt import ToolNode
from openai.types.responses import Response
from pydantic import BaseModel, ConfigDict, ValidationError

from support_agent.config import DEFAULT_MAX_TOOL_CALLS_PER_REQUEST
from support_agent.llm import LLMClient, LLMUnavailableError
from support_agent.prompts import get_prompt
from support_agent.request_context import get_request_context
from support_agent.state import AgentState, Family, Language, Risk
from support_agent.tools import Tool, is_gated_write

DEFAULT_LANGUAGE: Language = "en"

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


def _get_last_user_text(state: AgentState) -> str:
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            return message.text
    raise ValueError("the state has no user message")


def classify_node(llm: LLMClient, model: str) -> Callable[[AgentState], dict[str, Any]]:
    """The classify node. One constrained call to `model`, the small deployment. A model
    outage or an answer that does not fit the schema ends in escalate: no guessing.
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
            )
            result = Classification.model_validate_json(response.output_text)
        except LLMUnavailableError:
            # the model client has already recorded the failed tries
            ctx.evidence.append({"type": "escalation", "reason": "classification_failed"})
            return {"outcome": "escalate", "decision_evidence": _evidence_since(start)}
        except ValidationError as error:
            ctx.evidence.append(
                {"type": "classification", "status": "invalid", "error": str(error)[:200]}
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
    llm: LLMClient, model: str, tools: Sequence[Tool]
) -> Callable[[AgentState], dict[str, Any]]:
    """The agent node: one model call with the tool belt. The reply may ask for tool
    calls. A model outage, or tool arguments that are not valid JSON, ends in escalate.
    """
    prompt = get_prompt("agent_system")
    # the Responses API treats a tool as strict unless told otherwise
    tool_schemas = [
        {"type": "function", **convert_to_openai_function(tool), "strict": False} for tool in tools
    ]

    def agent(state: AgentState) -> dict[str, Any]:
        ctx = get_request_context()
        start = len(ctx.evidence)
        try:
            response = llm.create(
                model,
                _to_response_input(state["messages"]),
                prompt,
                instructions=prompt.text,
                tools=tool_schemas,
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
    tool_node = ToolNode(tools)
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
                chunk_ids.extend(item["ref"])
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
