import pytest

from support_agent.costing import (
    Totals,
    call_cost_eur,
    model_calls,
    request_total_cost,
    validate_prices,
)

PRICES = {
    "gpt-5": {"input_per_1k": 0.001075, "output_per_1k": 0.0086},
    "gpt-5-mini": {"input_per_1k": 0.000215, "output_per_1k": 0.00172},
    "text-embedding-3-small": {"input_per_1k": 0.0000172},
}


@pytest.mark.parametrize(
    ("model", "input_tokens", "output_tokens", "expected"),
    [
        # 1 * 0.001075 + 0.5 * 0.0086
        ("gpt-5", 1000, 500, 0.005375),
        # 2 * 0.000215 + 1 * 0.00172
        ("gpt-5-mini", 2000, 1000, 0.00215),
        ("gpt-5", 0, 0, 0.0),
        # 0.1 * 0.001075
        ("gpt-5", 100, 0, 0.0001075),
    ],
)
def test_call_cost_from_known_token_counts(model, input_tokens, output_tokens, expected):
    assert call_cost_eur(PRICES, model, input_tokens, output_tokens) == pytest.approx(expected)


def test_embedding_model_costs_input_tokens_only():
    assert call_cost_eur(PRICES, "text-embedding-3-small", 1000) == pytest.approx(0.0000172)
    # output tokens have no price for an embedding model, so they add nothing
    assert call_cost_eur(PRICES, "text-embedding-3-small", 1000, 500) == pytest.approx(0.0000172)


def test_call_cost_for_an_unknown_model_raises():
    with pytest.raises(ValueError, match="no price for: gpt-9"):
        call_cost_eur(PRICES, "gpt-9", 100, 100)


def test_validate_prices_passes_when_every_model_has_a_price():
    validate_prices(PRICES, ["gpt-5", "gpt-5-mini"])


def test_validate_prices_names_every_missing_model():
    with pytest.raises(ValueError, match="gpt-4.1, gpt-9"):
        validate_prices(PRICES, ["gpt-5", "gpt-4.1", "gpt-9"])


def _ok(model, input_tokens, output_tokens, cost_eur):
    return {
        "type": "model_call",
        "model": model,
        "attempt": 1,
        "status": "ok",
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_eur": cost_eur,
    }


EVIDENCE = [
    {"type": "tool_result", "tool": "get_leave_balance", "summary": "x", "ref": None},
    {"type": "model_call", "model": "gpt-5", "attempt": 1, "status": "retryable_error"},
    _ok("gpt-5", 1000, 500, 0.005375),
    {"type": "model_fallback", "from": "gpt-5", "to": "gpt-5-mini"},
    {"type": "model_call", "model": "gpt-5-mini", "attempt": 1, "status": "error"},
    _ok("gpt-5-mini", 2000, 1000, 0.00215),
]


def test_model_calls_lists_only_successful_model_calls():
    assert model_calls(EVIDENCE) == [
        {"model": "gpt-5", "input_tokens": 1000, "output_tokens": 500, "cost_eur": 0.005375},
        {"model": "gpt-5-mini", "input_tokens": 2000, "output_tokens": 1000, "cost_eur": 0.00215},
    ]


def test_request_total_cost_add_up_the_successful_calls():
    totals = request_total_cost(EVIDENCE)

    assert totals == Totals(tokens=4500, cost_eur=pytest.approx(0.007525))


def test_request_total_cost_of_a_request_without_model_calls_are_zero():
    assert request_total_cost([]) == Totals(tokens=0, cost_eur=0)
    assert request_total_cost(EVIDENCE[:1]) == Totals(tokens=0, cost_eur=0)
