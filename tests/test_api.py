"""The REST surface. The graph is replaced by a stub, so no model keys are needed. The
requests that name an employee read the real employees table (make db-up).
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from tests.graph_helpers import CLASSIFIED, ScriptedLLM, call_reply, run_graph, text_reply
from tests.test_tool_contracts import is_identity_field, property_names

from support_agent.api import app, get_graph
from support_agent.decision_record import save_decision_record
from support_agent.request_context import bind_request_context, get_request_context

CHAT_HEADERS = {"X-Employee-Id": "emp_001"}


class StubGraph:
    """Stands in for the compiled graph. It keeps the request context bound while it ran
    and the state it got, and answers with a fixed reply.
    """

    def __init__(self, outcome: str = "resolve", answer: str = "You have 18 days left."):
        self.outcome = outcome
        self.answer = answer
        self.contexts = []
        self.states = []

    def invoke(self, state: dict[str, Any]) -> dict[str, Any]:
        self.contexts.append(get_request_context())
        self.states.append(state)
        return {"messages": [*state["messages"], AIMessage(self.answer)], "outcome": self.outcome}


@pytest.fixture
def graph():
    stub = StubGraph()
    app.dependency_overrides[get_graph] = lambda: stub
    yield stub
    app.dependency_overrides.clear()


@pytest.fixture
def client(graph):
    return TestClient(app)


def test_chat_without_the_header_is_unauthorized(client, graph):
    response = client.post("/chat", json={"message": "hi"})

    assert response.status_code == 401
    assert graph.contexts == []


def _body_schemas(spec: dict[str, Any]) -> list[Any]:
    """The request body schema of every operation, with the references followed."""
    schemas = spec["components"]["schemas"]

    def resolve(schema: Any) -> Any:
        if isinstance(schema, dict):
            if "$ref" in schema:
                return resolve(schemas[schema["$ref"].rsplit("/", 1)[1]])
            return {key: resolve(value) for key, value in schema.items()}
        if isinstance(schema, list):
            return [resolve(item) for item in schema]
        return schema

    return [
        resolve(operation["requestBody"]["content"]["application/json"]["schema"])
        for path in spec["paths"].values()
        for operation in path.values()
        if "requestBody" in operation
    ]


def test_no_route_takes_an_employee_id_except_the_header():
    spec = app.openapi()
    parameters = [
        p
        for path in spec["paths"].values()
        for op in path.values()
        for p in op.get("parameters", [])
    ]

    header_names = {p["name"].lower() for p in parameters if p["in"] == "header"}
    other_names = [p["name"] for p in parameters if p["in"] != "header"]
    body_names = [name for schema in _body_schemas(spec) for name in property_names(schema)]

    assert {n for n in header_names if is_identity_field(n)} <= {"x-employee-id"}
    assert [n for n in other_names + body_names if is_identity_field(n)] == []


# INTEGRATION TESTS: the tests below use the real Postgres (make db-up).
def test_chat_returns_answer_outcome_and_request_id(client, graph, seeded_db):
    response = client.post(
        "/chat", json={"message": "How many days do I have?"}, headers=CHAT_HEADERS
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "You have 18 days left."
    assert body["outcome"] == "resolve"
    assert body["request_id"] == graph.contexts[0].request_id


def test_chat_binds_the_employee_from_the_header(client, graph, seeded_db):
    client.post("/chat", json={"message": "hi"}, headers={"X-Employee-Id": "emp_003"})

    ctx = graph.contexts[0]
    assert (ctx.employee_id, ctx.channel) == ("emp_003", "rest")
    assert graph.states[0]["messages"][0].content == "hi"


def test_every_chat_gets_its_own_request_id(client, graph, seeded_db):
    first = client.post("/chat", json={"message": "hi"}, headers=CHAT_HEADERS).json()
    second = client.post("/chat", json={"message": "hi"}, headers=CHAT_HEADERS).json()

    assert first["request_id"] != second["request_id"]


def test_chat_with_an_unknown_employee_is_unauthorized(client, graph, seeded_db):
    response = client.post("/chat", json={"message": "hi"}, headers={"X-Employee-Id": "emp_999"})

    assert response.status_code == 401
    assert graph.contexts == []


def test_chat_ignores_an_employee_id_in_the_query(client, graph, seeded_db):
    client.post("/chat?employee_id=emp_002", json={"message": "hi"}, headers=CHAT_HEADERS)

    assert graph.contexts[0].employee_id == "emp_001"


def test_chat_refuses_an_employee_id_in_the_body(client, graph, seeded_db):
    response = client.post(
        "/chat", json={"message": "hi", "employee_id": "emp_002"}, headers=CHAT_HEADERS
    )

    assert response.status_code == 422
    assert graph.contexts == []


@pytest.mark.parametrize("body", [{}, {"message": ""}], ids=["missing", "empty"])
def test_chat_needs_a_message(body, client, graph, seeded_db):
    response = client.post("/chat", json=body, headers=CHAT_HEADERS)

    assert response.status_code == 422
    assert graph.contexts == []


def _propose(conn, employee_id: str) -> int:
    """Insert a pending one-day leave request for the employee and return its id."""
    return conn.execute(
        """
        INSERT INTO pending_actions (request_id, employee_id, tool, payload)
        VALUES ('req-api', %s, 'submit_leave_request',
                '{"start_date": "2026-11-02", "end_date": "2026-11-02", "working_days": 1}')
        RETURNING id
        """,
        (employee_id,),
    ).fetchone()[0]


def _approver(employee_id: str) -> dict[str, str]:
    return {"X-Employee-Id": employee_id}


def test_approvals_without_the_header_are_unauthorized(client):
    assert client.get("/approvals").status_code == 401
    assert client.post("/approvals/1", json={"decision": "approve"}).status_code == 401


def test_approval_queue_lists_pending_actions_with_payload(client, seeded_db):
    first = _propose(seeded_db, "emp_001")
    second = _propose(seeded_db, "emp_002")

    response = client.get("/approvals", headers=_approver("emp_005"))

    assert response.status_code == 200
    queue = response.json()
    assert [item["id"] for item in queue] == [first, second]
    assert queue[0]["employee_id"] == "emp_001"
    assert queue[0]["tool"] == "submit_leave_request"
    assert [item["can_decide"] for item in queue] == [True, True]
    assert queue[0]["payload"] == {
        "start_date": "2026-11-02",
        "end_date": "2026-11-02",
        "working_days": 1,
    }


def test_approval_queue_leaves_out_decided_actions(client, seeded_db):
    decided = _propose(seeded_db, "emp_001")
    open_action = _propose(seeded_db, "emp_002")
    client.post(f"/approvals/{decided}", json={"decision": "reject"}, headers=_approver("emp_010"))

    queue = client.get("/approvals", headers=_approver("emp_005")).json()

    assert [item["id"] for item in queue] == [open_action]


def test_approval_queue_marks_own_requests_as_not_decidable(client, seeded_db):
    other = _propose(seeded_db, "emp_001")
    own = _propose(seeded_db, "emp_005")

    queue = client.get("/approvals", headers=_approver("emp_005")).json()

    assert [(item["id"], item["can_decide"]) for item in queue] == [(other, True), (own, False)]
    # the flag is only a hint: the decision itself is still refused
    response = client.post(
        f"/approvals/{own}", json={"decision": "approve"}, headers=_approver("emp_005")
    )
    assert response.status_code == 403


def test_approval_queue_is_closed_to_employees(client, seeded_db):
    _propose(seeded_db, "emp_001")

    response = client.get("/approvals", headers=_approver("emp_002"))

    assert response.status_code == 403


def test_approving_executes_the_action(client, seeded_db):
    action_id = _propose(seeded_db, "emp_001")

    response = client.post(
        f"/approvals/{action_id}", json={"decision": "approve"}, headers=_approver("emp_005")
    )

    assert response.status_code == 200
    assert response.json() == {"action_id": action_id, "status": "executed"}
    row = seeded_db.execute(
        "SELECT status, decided_by FROM pending_actions WHERE id = %s", (action_id,)
    ).fetchone()
    assert row == ("executed", "emp_005")
    assert seeded_db.execute("SELECT count(*) FROM leave_requests").fetchone()[0] == 1


def test_rejecting_writes_nothing(client, seeded_db):
    action_id = _propose(seeded_db, "emp_001")

    response = client.post(
        f"/approvals/{action_id}", json={"decision": "reject"}, headers=_approver("emp_010")
    )

    assert response.json() == {"action_id": action_id, "status": "rejected"}
    assert seeded_db.execute("SELECT count(*) FROM leave_requests").fetchone()[0] == 0


def test_the_approver_comes_from_the_header(client, seeded_db):
    action_id = _propose(seeded_db, "emp_001")

    client.post(
        f"/approvals/{action_id}", json={"decision": "approve"}, headers=_approver("emp_010")
    )

    decided_by = seeded_db.execute(
        "SELECT decided_by FROM pending_actions WHERE id = %s", (action_id,)
    ).fetchone()[0]
    assert decided_by == "emp_010"


def test_an_employee_cannot_decide(client, seeded_db):
    action_id = _propose(seeded_db, "emp_001")

    response = client.post(
        f"/approvals/{action_id}", json={"decision": "approve"}, headers=_approver("emp_002")
    )

    assert response.status_code == 403
    assert seeded_db.execute("SELECT count(*) FROM leave_requests").fetchone()[0] == 0


def test_an_approver_cannot_decide_on_their_own_request(client, seeded_db):
    action_id = _propose(seeded_db, "emp_005")

    response = client.post(
        f"/approvals/{action_id}", json={"decision": "approve"}, headers=_approver("emp_005")
    )

    assert response.status_code == 403


def test_deciding_on_an_unknown_action_is_not_found(client, seeded_db):
    response = client.post(
        "/approvals/999", json={"decision": "approve"}, headers=_approver("emp_005")
    )

    assert response.status_code == 404


def test_a_second_decision_is_a_conflict(client, seeded_db):
    action_id = _propose(seeded_db, "emp_001")
    client.post(
        f"/approvals/{action_id}", json={"decision": "approve"}, headers=_approver("emp_005")
    )

    response = client.post(
        f"/approvals/{action_id}", json={"decision": "approve"}, headers=_approver("emp_010")
    )

    assert response.status_code == 409
    assert seeded_db.execute("SELECT count(*) FROM leave_requests").fetchone()[0] == 1


@pytest.mark.parametrize(
    "body",
    [{}, {"decision": "maybe"}, {"decision": "approve", "approver_id": "emp_010"}],
    ids=["missing", "unknown-value", "extra-field"],
)
def test_a_bad_decision_is_refused(body, client, seeded_db):
    action_id = _propose(seeded_db, "emp_001")

    response = client.post(f"/approvals/{action_id}", json=body, headers=_approver("emp_005"))

    assert response.status_code == 422
    assert seeded_db.execute(
        "SELECT status FROM pending_actions WHERE id = %s", (action_id,)
    ).fetchone() == ("pending_approval",)


def _save_record() -> dict[str, Any]:
    """Run the wired graph for emp_001 (request "req-1", fake tools, scripted model) and
    save its Decision Record. Returns the record as the API should show it.
    """
    records = []
    llm = ScriptedLLM(
        CLASSIFIED,
        call_reply("call_1", "get_leave_balance", "{}"),
        text_reply("You have 18 days left."),
    )
    run_graph(llm, saved_records=records)
    with bind_request_context("emp_001", "req-1", "rest"):
        save_decision_record(records[0])
    return records[0].model_dump()


def test_decision_endpoints_without_the_header_are_unauthorized(client):
    assert client.get("/decisions/req-1").status_code == 401
    assert client.post("/feedback/req-1", json={"rating": "up"}).status_code == 401


def test_decision_record_explains_the_answer_without_running_anything(client, seeded_db):
    saved = _save_record()

    response = client.get("/decisions/req-1", headers=CHAT_HEADERS)

    assert response.status_code == 200
    body = response.json()
    assert body == saved
    assert body["outcome"] == "resolve"
    assert body["summary"] == "Resolved using get_leave_balance."
    assert [item["type"] for item in body["evidence"]] == [
        "classification",
        "tool_result",
        "verifier_verdict",
    ]


@pytest.mark.parametrize("reader", ["emp_005", "emp_010"], ids=["hr_partner", "manager"])
def test_an_approver_can_read_another_employees_record(reader, client, seeded_db):
    _save_record()

    response = client.get("/decisions/req-1", headers=_approver(reader))

    assert response.status_code == 200


def test_another_employee_cannot_read_the_record(client, seeded_db):
    _save_record()

    response = client.get("/decisions/req-1", headers=_approver("emp_002"))

    assert response.status_code == 404


def test_an_unknown_request_is_not_found(client, seeded_db):
    assert client.get("/decisions/req-999", headers=CHAT_HEADERS).status_code == 404


def test_feedback_is_saved_with_a_rating_and_a_comment(client, seeded_db):
    _save_record()

    response = client.post(
        "/feedback/req-1", json={"rating": "down", "comment": "Wrong year."}, headers=CHAT_HEADERS
    )

    assert response.status_code == 201
    body = response.json()
    assert body["request_id"] == "req-1"
    assert body["rating"] == "down"
    rows = seeded_db.execute(
        "SELECT id, request_id, employee_id, rating, comment, triage_status FROM feedback"
    ).fetchall()
    assert rows == [(body["id"], "req-1", "emp_001", "down", "Wrong year.", "new")]


def test_feedback_comment_is_optional(client, seeded_db):
    _save_record()

    response = client.post("/feedback/req-1", json={"rating": "up"}, headers=CHAT_HEADERS)

    assert response.status_code == 201
    assert seeded_db.execute("SELECT comment FROM feedback").fetchone() == (None,)


def test_feedback_is_audited_under_the_request(client, seeded_db):
    _save_record()

    response = client.post("/feedback/req-1", json={"rating": "up"}, headers=CHAT_HEADERS)

    events = seeded_db.execute(
        "SELECT request_id, actor, payload FROM audit_log WHERE event = 'feedback_received'"
    ).fetchall()
    assert events == [
        ("req-1", "employee:emp_001", {"feedback_id": response.json()["id"], "rating": "up"})
    ]


def test_only_the_requester_can_rate_the_answer(client, seeded_db):
    _save_record()

    for reader in ("emp_002", "emp_005"):
        response = client.post("/feedback/req-1", json={"rating": "up"}, headers=_approver(reader))
        assert response.status_code == 404

    assert seeded_db.execute("SELECT count(*) FROM feedback").fetchone()[0] == 0
    assert seeded_db.execute("SELECT count(*) FROM audit_log").fetchone()[0] == 1


def test_feedback_on_an_unknown_request_is_not_found(client, seeded_db):
    response = client.post("/feedback/req-999", json={"rating": "up"}, headers=CHAT_HEADERS)

    assert response.status_code == 404


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"rating": "great"},
        {"rating": "up", "comment": "x" * 1001},
        {"rating": "up", "employee_id": "emp_002"},
    ],
    ids=["missing", "unknown-rating", "long-comment", "extra-field"],
)
def test_bad_feedback_is_refused(body, client, seeded_db):
    _save_record()

    response = client.post("/feedback/req-1", json=body, headers=CHAT_HEADERS)

    assert response.status_code == 422
    assert seeded_db.execute("SELECT count(*) FROM feedback").fetchone()[0] == 0
