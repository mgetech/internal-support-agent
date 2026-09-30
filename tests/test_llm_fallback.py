from types import SimpleNamespace

import httpx
import openai
import pytest

from support_agent import audit, llm
from support_agent.config import Settings
from support_agent.costing import request_total_cost
from support_agent.llm import LLMClient, LLMUnavailableError, is_retryable
from support_agent.prompts import Prompt
from support_agent.request_context import bind_request_context

PRICES = {
    "main": {"input_per_1k": 0.001, "output_per_1k": 0.002},
    "backup": {"input_per_1k": 0.01, "output_per_1k": 0.02},
}
PROMPT = Prompt(id="agent_system", version="1.2.0", text="text", changelog="change")
_REQUEST = httpx.Request("POST", "https://example.test/openai/v1/responses")


def _status_error(error_class, status_code):
    response = httpx.Response(status_code, request=_REQUEST)
    return error_class("error", response=response, body=None)


def rate_limited():
    return _status_error(openai.RateLimitError, 429)


def unauthorized():
    return _status_error(openai.AuthenticationError, 401)


def server_error():
    return _status_error(openai.InternalServerError, 503)


def connection_error():
    return openai.APIConnectionError(request=_REQUEST)


def answer(text="hello", input_tokens=5, output_tokens=2, status="completed"):
    return SimpleNamespace(
        status=status,
        error=None,
        output_text=text,
        output=[],
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
    )


class FakeOpenAI:
    """Stands in for the OpenAI client. Each call to `responses.create` returns or
    raises the next item from `responses`, and the keyword arguments are kept.
    """

    def __init__(self, responses):
        self._scripted_responses = list(responses)
        self.calls = []
        self.responses = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        response = self._scripted_responses[len(self.calls) - 1]
        if isinstance(response, Exception):
            raise response
        return response

    @property
    def models(self):
        return [call["model"] for call in self.calls]


@pytest.fixture
def audit_events(monkeypatch):
    events = []
    monkeypatch.setattr(audit, "record", lambda event, payload, **kwargs: events.append(payload))
    return events


@pytest.fixture
def sleeps():
    return []


def run(responses, sleeps, fallback_model="", **options):
    """Run one call in a bound request. `responses` is what the fake client returns
    or raises on each call. Returns the fake client, the result (a response or the
    LLMUnavailableError) and the request's evidence.
    """
    fake = FakeOpenAI(responses)
    client = LLMClient(fake, PRICES, fallback_model=fallback_model, sleep=sleeps.append)
    with bind_request_context("emp_001", "req-1", "rest") as ctx:
        try:
            result = client.create("main", [{"role": "user", "content": "hi"}], PROMPT, **options)
        except LLMUnavailableError as error:
            result = error
    return fake, result, ctx.evidence


def chain(evidence):
    """The evidence reduced to what happened, in order."""
    return [
        (item["type"], item.get("model") or item.get("to"), item.get("status")) for item in evidence
    ]


def test_first_try_succeeds_without_waiting(audit_events, sleeps):
    fake, result, evidence = run([answer("hello")], sleeps)

    assert result.output_text == "hello"
    assert fake.models == ["main"]
    assert sleeps == []
    assert chain(evidence) == [("model_call", "main", "ok")]


def test_successful_call_carries_tokens_and_cost(audit_events, sleeps):
    _, _, evidence = run([answer(input_tokens=1000, output_tokens=500)], sleeps)

    # 1 * 0.001 + 0.5 * 0.002
    assert evidence[0]["input_tokens"] == 1000
    assert evidence[0]["output_tokens"] == 500
    assert evidence[0]["cost_eur"] == pytest.approx(0.002)


@pytest.mark.parametrize("make_error", [rate_limited, server_error, connection_error])
def test_retryable_error_is_retried_on_the_same_model(make_error, audit_events, sleeps):
    fake, result, evidence = run([make_error(), answer()], sleeps)

    assert result.output_text == "hello"
    assert fake.models == ["main", "main"]
    assert chain(evidence) == [
        ("model_call", "main", "retryable_error"),
        ("model_call", "main", "ok"),
    ]


def test_wait_doubles_between_retries(audit_events, sleeps):
    run([rate_limited(), rate_limited(), answer()], sleeps)

    assert sleeps == [0.5, 1.0]


def test_hard_error_skips_the_retries_and_goes_to_the_fallback(audit_events, sleeps):
    fake, result, evidence = run([unauthorized(), answer("from backup")], sleeps, "backup")

    assert result.output_text == "from backup"
    assert fake.models == ["main", "backup"]
    assert sleeps == []
    assert chain(evidence) == [
        ("model_call", "main", "error"),
        ("model_fallback", "backup", None),
        ("model_call", "backup", "ok"),
    ]
    assert evidence[0]["status_code"] == 401


def test_exhausted_retries_go_to_the_fallback(audit_events, sleeps):
    responses = [rate_limited(), rate_limited(), rate_limited(), answer("from backup")]

    fake, result, evidence = run(responses, sleeps, "backup")

    assert result.output_text == "from backup"
    assert fake.models == ["main", "main", "main", "backup"]
    assert [item["attempt"] for item in evidence if item["type"] == "model_call"] == [1, 2, 3, 1]
    assert chain(evidence)[3] == ("model_fallback", "backup", None)


def test_fallback_is_priced_with_its_own_prices(audit_events, sleeps):
    responses = [unauthorized(), answer(input_tokens=1000, output_tokens=1000)]

    _, _, evidence = run(responses, sleeps, "backup")

    # 1 * 0.01 + 1 * 0.02
    assert request_total_cost(evidence).cost_eur == pytest.approx(0.03)


def test_all_models_failing_raises_and_records_the_degrade_to_escalate(audit_events, sleeps):
    fake, result, evidence = run([rate_limited()] * 6, sleeps, "backup")

    assert isinstance(result, LLMUnavailableError)
    assert result.models_tried == ["main", "backup"]
    assert isinstance(result.last_error, openai.RateLimitError)
    assert fake.models == ["main"] * 3 + ["backup"] * 3
    assert evidence[-1] == {
        "type": "model_degraded",
        "outcome": "escalate",
        "models_tried": ["main", "backup"],
    }


def test_without_a_fallback_the_request_degrades_after_the_retries(audit_events, sleeps):
    fake, result, evidence = run([server_error()] * 3, sleeps)

    assert isinstance(result, LLMUnavailableError)
    assert result.models_tried == ["main"]
    assert fake.models == ["main"] * 3
    assert sleeps == [0.5, 1.0]
    assert [item["type"] for item in evidence] == ["model_call"] * 3 + ["model_degraded"]


def test_a_fallback_equal_to_the_main_model_is_ignored(audit_events, sleeps):
    fake, result, _ = run([rate_limited()] * 3, sleeps, "main")

    assert result.models_tried == ["main"]
    assert fake.models == ["main"] * 3


def test_a_failed_response_counts_as_a_hard_error(audit_events, sleeps):
    failed = answer(status="failed")
    failed.error = SimpleNamespace(message="the model stopped")

    fake, result, evidence = run([failed, answer("from backup")], sleeps, "backup")

    assert result.output_text == "from backup"
    assert fake.models == ["main", "backup"]
    assert evidence[0]["status"] == "error"
    assert "the model stopped" in evidence[0]["error"]


def test_the_chain_can_be_read_from_the_evidence_alone(audit_events, sleeps):
    """Nothing but the evidence list is used to rebuild what happened."""
    responses = [rate_limited(), unauthorized(), answer("from backup")]

    _, _, evidence = run(responses, sleeps, "backup")

    assert [
        (item["type"], item.get("model"), item.get("attempt"), item.get("status"))
        for item in evidence
    ] == [
        ("model_call", "main", 1, "retryable_error"),
        ("model_call", "main", 2, "error"),
        ("model_fallback", None, None, None),
        ("model_call", "backup", 1, "ok"),
    ]
    assert (evidence[2]["from"], evidence[2]["to"]) == ("main", "backup")
    assert evidence[0]["status_code"] == 429
    assert evidence[1]["status_code"] == 401


def test_every_evidence_item_has_a_matching_audit_event(audit_events, sleeps):
    _, _, evidence = run([rate_limited()] * 6, sleeps, "backup")

    assert audit_events == evidence


def test_every_try_records_the_prompt_id_and_version(audit_events, sleeps):
    _, _, evidence = run([rate_limited(), answer()], sleeps)

    tries = [item for item in evidence if item["type"] == "model_call"]
    assert [item["prompt"] for item in tries] == [
        {"prompt_id": "agent_system", "prompt_version": "1.2.0"}
    ] * 2


def test_options_go_to_the_sdk_and_nothing_is_stored(audit_events, sleeps):
    fake, _, _ = run([answer()], sleeps, instructions="be brief")

    call = fake.calls[0]
    assert call["model"] == "main"
    assert call["input"] == [{"role": "user", "content": "hi"}]
    assert call["instructions"] == "be brief"
    assert call["store"] is False


@pytest.mark.parametrize(
    ("make_error", "expected"),
    [
        (connection_error, True),
        (rate_limited, True),
        (server_error, True),
        (lambda: _status_error(openai.APIStatusError, 408), True),
        (lambda: _status_error(openai.APIStatusError, 409), True),
        (lambda: _status_error(openai.BadRequestError, 400), False),
        (unauthorized, False),
        (lambda: _status_error(openai.NotFoundError, 404), False),
        (lambda: RuntimeError("failed"), False),
    ],
)
def test_which_errors_are_retryable(make_error, expected):
    assert is_retryable(make_error()) is expected


SETTINGS = {
    "database_url": "postgresql://user:pass@localhost:5432/support_agent",
    "azure_openai_endpoint": "https://example.services.ai.azure.com/openai/v1",
    "azure_openai_api_key": "test-key",
    "azure_openai_deployment_agent": "main",
    "azure_openai_deployment_classifier": "small",
    "azure_openai_deployment_embedding": "embed",
    "model_fallback_deployment": "backup",
    "price_table_eur": {
        "main": {"input_per_1k": 0.001, "output_per_1k": 0.002},
        "small": {"input_per_1k": 0.0001, "output_per_1k": 0.0002},
        "backup": {"input_per_1k": 0.01, "output_per_1k": 0.02},
    },
}


def test_get_llm_client_turns_off_the_sdk_retries(monkeypatch):
    monkeypatch.setattr(llm, "get_settings", lambda: Settings(_env_file=None, **SETTINGS))

    client = llm.get_llm_client()

    assert client._client.max_retries == 0


def test_get_llm_client_fails_at_startup_when_a_price_is_missing(monkeypatch):
    prices = {k: v for k, v in SETTINGS["price_table_eur"].items() if k != "backup"}
    settings = Settings(_env_file=None, **{**SETTINGS, "price_table_eur": prices})
    monkeypatch.setattr(llm, "get_settings", lambda: settings)

    with pytest.raises(ValueError, match="no price for: backup"):
        llm.get_llm_client()
