"""The REST surface. Identity enters here and nowhere else: the `X-Employee-Id` header
names the employee, and the request is bound to that employee before the graph runs.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from functools import lru_cache
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, ConfigDict, Field

from support_agent import db
from support_agent.config import get_settings
from support_agent.graph import build_graph
from support_agent.guardrails.approval import AlreadyDecidedError, decide, get_pending_actions
from support_agent.llm import get_llm_client
from support_agent.request_context import bind_request_context
from support_agent.state import Outcome, new_state

app = FastAPI(title="Internal support agent")


class ChatRequest(BaseModel):
    # extra fields are refused, so a body can never carry an employee id
    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1)


class ChatResponse(BaseModel):
    answer: str
    outcome: Outcome
    request_id: str


class PendingAction(BaseModel):
    id: int
    request_id: str
    employee_id: str
    tool: str
    payload: dict[str, Any]
    created_at: datetime
    # false for the caller's own requests: the decision would be refused with a 403
    can_decide: bool


class ApprovalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: Literal["approve", "reject"]


class ApprovalResponse(BaseModel):
    action_id: int
    status: Literal["executed", "rejected"]


def get_current_employee_id(x_employee_id: Annotated[str | None, Header()] = None) -> str:
    """The employee who makes the request. This is the auth stub: the header is trusted
    as it is, and the only check is that the employee exists. This is the place where
    OIDC would sit. Raises a 401 for a missing header or an unknown employee.
    """
    if not x_employee_id:
        raise HTTPException(status_code=401, detail="the X-Employee-Id header is missing")
    row = db.fetch_one("SELECT id FROM employees WHERE id = %s", (x_employee_id,))
    if row is None:
        raise HTTPException(status_code=401, detail="unknown employee")
    return x_employee_id


@lru_cache
def get_graph() -> CompiledStateGraph:
    """The agent graph, built on the first request. Tests replace it through
    `app.dependency_overrides`, so they need no model keys.
    """
    settings = get_settings()
    return build_graph(
        get_llm_client(),
        settings.azure_openai_deployment_classifier,
        settings.azure_openai_deployment_agent,
        max_tool_calls=settings.max_tool_calls_per_request,
    )


@app.post("/chat")
def create_chat(
    body: ChatRequest,
    employee_id: Annotated[str, Depends(get_current_employee_id)],
    graph: Annotated[CompiledStateGraph, Depends(get_graph)],
) -> ChatResponse:
    """Run the agent for one message from the employee named in the header."""
    request_id = uuid.uuid4().hex
    with bind_request_context(employee_id, request_id, "rest"):
        result = graph.invoke(new_state(body.message))
    return ChatResponse(
        answer=result["messages"][-1].text, outcome=result["outcome"], request_id=request_id
    )


@app.get("/approvals")
def get_approvals(
    approver_id: Annotated[str, Depends(get_current_employee_id)],
) -> list[PendingAction]:
    """The actions waiting for approval. The employee in the header must be an approver."""
    try:
        return get_pending_actions(approver_id)
    except PermissionError as error:
        raise HTTPException(status_code=403, detail=str(error)) from error


@app.post("/approvals/{action_id}")
def create_approval(
    action_id: int,
    body: ApprovalRequest,
    approver_id: Annotated[str, Depends(get_current_employee_id)],
) -> ApprovalResponse:
    """Approve or reject an action. The approver is the employee in the header."""
    try:
        status = decide(action_id, approver_id, approve=body.decision == "approve")
    except PermissionError as error:
        raise HTTPException(status_code=403, detail=str(error)) from error
    except LookupError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except AlreadyDecidedError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    return ApprovalResponse(action_id=action_id, status=status)
