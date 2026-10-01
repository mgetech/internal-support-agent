"""The graph nodes. Each node returns only the fields it changes. Evidence items are
added to the request context, where the tools and the model client also add theirs, and
the node returns the ones it caused.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, ConfigDict, ValidationError

from support_agent.llm import LLMClient, LLMUnavailableError
from support_agent.prompts import get_prompt
from support_agent.request_context import get_request_context
from support_agent.state import AgentState, Family, Language, Risk

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


def make_classify_node(llm: LLMClient, model: str) -> Callable[[AgentState], dict[str, Any]]:
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
            return {"outcome": "escalate", "decision_evidence": _evidence_since(start)}
        except ValidationError as error:
            ctx.evidence.append(
                {"type": "classification", "status": "invalid", "error": str(error)[:200]}
            )
            return {"outcome": "escalate", "decision_evidence": _evidence_since(start)}

        ctx.evidence.append({"type": "classification", "status": "ok", **result.model_dump()})
        return {**result.model_dump(), "decision_evidence": _evidence_since(start)}

    return classify


def route_after_classify(state: AgentState) -> Literal["agent", "refuse"]:
    if state.get("outcome") == "escalate" or state.get("risk") == "out_of_scope":
        return "refuse"
    return "agent"


def refuse(state: AgentState) -> dict[str, Any]:
    """End the request with escalate and tell the employee a person can take over."""
    ctx = get_request_context()
    start = len(ctx.evidence)
    reason = "out_of_scope" if state.get("risk") == "out_of_scope" else "classification_failed"
    ctx.evidence.append({"type": "escalation", "reason": reason})
    text = REFUSAL_TEXT[state.get("language", DEFAULT_LANGUAGE)]
    return {
        "messages": [AIMessage(text)],
        "outcome": "escalate",
        "decision_evidence": _evidence_since(start),
    }
