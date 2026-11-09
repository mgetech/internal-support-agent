"""Tracing: a span for every graph node and tool call, a generation for every model call, one
trace for the request, the trace id in the Decision Record. Tracing is off in every test
(tests/conftest.py) unless the test asks for the `spans` fixture, which keeps the spans in
memory. The inputs and outputs are off too, unless the test turns TRACE_CONTENT on.
"""

import json

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langfuse import LangfuseOtelSpanAttributes as Attribute
from tests.graph_helpers import CLASSIFIED, ScriptedLLM, call_reply, run_graph, text_reply
from tests.test_config import REQUIRED
from tests.test_llm_fallback import PRICES, PROMPT, FakeOpenAI, answer, rate_limited

from support_agent import audit, tracing
from support_agent.api import app, get_current_employee_id, get_graph
from support_agent.config import Settings
from support_agent.llm import LLMClient
from support_agent.request_context import bind_request_context
from support_agent.tracing import (
    create_langfuse,
    get_trace_id,
    start_observation,
    trace_node,
)


def names(finished):
    return [span.name for span in finished]


def by_name(finished, name):
    return next(span for span in finished if span.name == name)


def attributes_text(finished):
    """All attributes of all spans as one text, to look for a string in it."""
    return json.dumps([dict(span.attributes) for span in finished], default=str)


@pytest.fixture
def content_traced(monkeypatch):
    monkeypatch.setattr(tracing, "is_content_traced", lambda: True)


# --- off without keys ---


def test_tracing_is_off_when_a_key_is_missing():
    assert create_langfuse(Settings(_env_file=None, **REQUIRED)) is None
    only_public = Settings(_env_file=None, **REQUIRED, langfuse_public_key="pk-lf-x")
    assert create_langfuse(only_public) is None


def test_the_content_of_spans_is_off_by_default():
    assert Settings(_env_file=None, **REQUIRED).trace_content is False


def test_with_tracing_off_nothing_is_made_and_the_node_still_runs():
    with start_observation("request", input="a message") as span:
        span.update(output="an answer", level="ERROR")
        assert get_trace_id() is None
        assert trace_node("classify", lambda state: {"outcome": "resolve"})({}) == {
            "outcome": "resolve"
        }


def test_a_request_runs_the_same_with_tracing_off():
    llm = ScriptedLLM(CLASSIFIED, call_reply("c1", "get_leave_balance", "{}"), text_reply("ok"))
    records = []

    result, _, _ = run_graph(llm, saved_records=records)

    assert result["outcome"] == "resolve"
    assert records[0].trace_id is None


# --- spans for nodes and tools ---


def test_every_node_and_tool_call_has_a_span_in_one_trace(spans):
    llm = ScriptedLLM(
        CLASSIFIED,
        call_reply("c1", "get_leave_balance", "{}"),
        text_reply("You have 18 days."),
    )

    with start_observation("request"):
        run_graph(llm)

    finished = spans()
    assert sorted(names(finished)) == sorted(
        [
            "request",
            "classify",
            "agent",
            "tools",
            "get_leave_balance",
            "agent",
            "verify",
            "finalize",
        ]
    )
    assert len({span.context.trace_id for span in finished}) == 1


def test_a_tool_span_is_inside_the_tools_node_span(spans):
    llm = ScriptedLLM(CLASSIFIED, call_reply("c1", "get_leave_balance", "{}"), text_reply("ok"))

    with start_observation("request"):
        run_graph(llm)

    finished = spans()
    tool = by_name(finished, "get_leave_balance")
    assert tool.parent.span_id == by_name(finished, "tools").context.span_id
    assert tool.attributes[Attribute.OBSERVATION_TYPE] == "tool"
    assert by_name(finished, "tools").parent.span_id == by_name(finished, "request").context.span_id


def test_a_path_that_skips_nodes_has_spans_only_for_the_nodes_it_ran(spans):
    llm = ScriptedLLM(
        text_reply(json.dumps({"family": "other", "risk": "out_of_scope", "language": "en"}))
    )

    with start_observation("request"):
        run_graph(llm)

    assert sorted(names(spans())) == ["classify", "finalize", "refuse", "request"]


def test_a_tool_call_that_fails_gets_the_error_level(spans):
    # create_ticket without its title: ToolNode answers with a failed ToolMessage
    llm = ScriptedLLM(CLASSIFIED, call_reply("c1", "create_ticket", "{}"), text_reply("ok"))

    with start_observation("request"):
        run_graph(llm)

    tool = by_name(spans(), "create_ticket")
    assert tool.attributes[Attribute.OBSERVATION_LEVEL] == "ERROR"


def test_a_node_that_raises_marks_its_span_with_the_error_type_and_raises_again(spans):
    def broken(state):
        raise RuntimeError("boom needle-error")

    with pytest.raises(RuntimeError, match="boom"):
        trace_node("verify", broken)({})

    span = by_name(spans(), "verify")
    assert span.attributes[Attribute.OBSERVATION_LEVEL] == "ERROR"
    assert span.attributes[Attribute.OBSERVATION_STATUS_MESSAGE] == "RuntimeError"


def test_the_error_message_and_stack_trace_are_not_on_the_span(spans, content_traced):
    def broken(state):
        raise RuntimeError("boom needle-error")

    with pytest.raises(RuntimeError):
        trace_node("verify", broken)({})

    span = by_name(spans(), "verify")
    assert span.events == ()
    assert "needle" not in attributes_text([span])


def test_spans_have_no_user_id_and_no_session(spans):
    llm = ScriptedLLM(CLASSIFIED, call_reply("c1", "get_leave_balance", "{}"), text_reply("ok"))

    with start_observation("request"):
        run_graph(llm)

    for span in spans():
        assert Attribute.TRACE_USER_ID not in span.attributes
        assert Attribute.TRACE_SESSION_ID not in span.attributes


# --- inputs and outputs follow TRACE_CONTENT ---


def run_search_request(spans_after):
    llm = ScriptedLLM(
        CLASSIFIED,
        call_reply("c1", "search_policies", json.dumps({"query": "needle-query"})),
        text_reply("needle-answer"),
    )
    with start_observation("request"):
        run_graph(llm, text="needle-message")
    return spans_after()


def test_with_trace_content_off_spans_carry_no_inputs_and_no_outputs(spans):
    assert "needle" not in attributes_text(run_search_request(spans))


def test_with_trace_content_on_tool_spans_carry_arguments_and_results(spans, content_traced):
    tool = by_name(run_search_request(spans), "search_policies")

    assert json.loads(tool.attributes[Attribute.OBSERVATION_INPUT]) == {"query": "needle-query"}
    assert "vacation-policy#entitlement#0" in tool.attributes[Attribute.OBSERVATION_OUTPUT]


class StubGraph:
    def invoke(self, state):
        return {"messages": [*state["messages"], AIMessage("needle-answer")], "outcome": "resolve"}


def post_chat():
    app.dependency_overrides[get_graph] = StubGraph
    app.dependency_overrides[get_current_employee_id] = lambda: "emp_001"
    try:
        return TestClient(app).post("/chat", json={"message": "needle-message"}).json()
    finally:
        app.dependency_overrides.clear()


def test_a_chat_request_is_one_request_span_with_its_id_and_outcome(spans):
    body = post_chat()

    request = by_name(spans(), "request")
    metadata = f"{Attribute.OBSERVATION_METADATA}."
    assert request.attributes[metadata + "request_id"] == body["request_id"]
    assert request.attributes[metadata + "outcome"] == "resolve"
    assert "needle" not in attributes_text([request])


def test_with_trace_content_on_the_request_span_carries_the_message_and_the_answer(
    spans, content_traced
):
    post_chat()

    request = by_name(spans(), "request")
    assert request.attributes[Attribute.OBSERVATION_INPUT] == "needle-message"
    assert request.attributes[Attribute.OBSERVATION_OUTPUT] == "needle-answer"


# --- one generation for each model call ---


@pytest.fixture
def model_client(monkeypatch):
    """Builds an LLMClient over a fake OpenAI client with these replies. The clock moves
    by 0.25 seconds each time it is read, so a try takes 250 ms.
    """
    monkeypatch.setattr(audit, "record", lambda event, payload, **kwargs: None)
    ticks = iter(i * 0.25 for i in range(100))

    def build(*replies, **options):
        fake = FakeOpenAI(replies)
        client = LLMClient(
            fake, PRICES, sleep=lambda seconds: None, clock=lambda: next(ticks), **options
        )
        return client, fake

    return build


def call_model(client, **options):
    """One call in a request, inside a node span."""
    with bind_request_context("emp_001", "req-1", "rest"), start_observation("classify"):
        return client.create(
            "main",
            [{"role": "user", "content": "needle-message"}],
            PROMPT,
            instructions="needle-instructions",
            reasoning_effort="low",
            **options,
        )


def test_a_model_call_is_a_generation_with_model_token_counts_and_latency(spans, model_client):
    client, _ = model_client(answer("needle-answer", 1000, 500, reasoning_tokens=200))

    call_model(client)

    finished = spans()
    generation = by_name(finished, "agent_system")
    assert generation.parent.span_id == by_name(finished, "classify").context.span_id
    attributes = generation.attributes
    assert attributes[Attribute.OBSERVATION_TYPE] == "generation"
    assert attributes[Attribute.OBSERVATION_MODEL] == "main"
    # the reasoning tokens are inside the output tokens, so the total does not add them
    assert json.loads(attributes[Attribute.OBSERVATION_USAGE_DETAILS]) == {
        "input": 1000,
        "output": 500,
        "output_reasoning_tokens": 200,
        "total": 1500,
    }
    metadata = f"{Attribute.OBSERVATION_METADATA}."
    assert attributes[metadata + "latency_ms"] == 250
    assert attributes[metadata + "prompt_id"] == "agent_system"
    assert attributes[metadata + "prompt_version"] == "1.2.0"
    assert json.loads(attributes[Attribute.OBSERVATION_MODEL_PARAMETERS]) == {
        "reasoning_effort": "low"
    }


def test_no_cost_is_sent_so_langfuse_works_out_its_own(spans, model_client):
    client, _ = model_client(answer("hello", 1000, 500))

    call_model(client)

    attributes = by_name(spans(), "agent_system").attributes
    assert Attribute.OBSERVATION_COST_DETAILS not in attributes
    assert "cost" not in attributes_text(spans()).lower()


def test_with_trace_content_off_a_generation_has_no_prompt_and_no_answer(spans, model_client):
    client, _ = model_client(answer("needle-answer"))

    call_model(client)

    assert "needle" not in attributes_text(spans())


def test_with_trace_content_on_a_generation_has_the_prompt_and_the_answer(
    spans, model_client, content_traced
):
    client, _ = model_client(answer("needle-answer"))

    call_model(client)

    attributes = by_name(spans(), "agent_system").attributes
    sent = json.loads(attributes[Attribute.OBSERVATION_INPUT])
    assert sent["instructions"] == "needle-instructions"
    assert sent["input"] == [{"role": "user", "content": "needle-message"}]
    assert json.loads(attributes[Attribute.OBSERVATION_OUTPUT])["text"] == "needle-answer"


def test_every_try_is_a_generation_and_a_failed_try_has_the_error_type(spans, model_client):
    client, fake = model_client(rate_limited(), answer("ok"))

    call_model(client)

    generations = [span for span in spans() if span.name == "agent_system"]
    assert len(generations) == 2
    failed, succeeded = generations
    assert failed.attributes[Attribute.OBSERVATION_LEVEL] == "ERROR"
    assert failed.attributes[Attribute.OBSERVATION_STATUS_MESSAGE] == "RateLimitError"
    assert Attribute.OBSERVATION_LEVEL not in succeeded.attributes
    assert fake.models == ["main", "main"]


def test_a_model_call_without_a_prompt_is_named_after_the_api_call(spans, model_client):
    client, _ = model_client(answer())

    with bind_request_context("emp_001", "req-1", "rest"):
        client.create("main", [{"role": "user", "content": "hi"}])

    assert names(spans()) == ["responses.create"]


# --- the trace id in the Decision Record ---


def test_the_record_has_the_id_of_the_trace_of_the_request(spans):
    llm = ScriptedLLM(CLASSIFIED, text_reply("ok"))
    records = []

    with start_observation("request"):
        run_graph(llm, saved_records=records)

    trace_ids = {format(span.context.trace_id, "032x") for span in spans()}
    assert trace_ids == {records[0].trace_id}
