"""Token usage to euro cost. Prices come from the `PRICE_TABLE_EUR` setting:
`{"<model>": {"input_per_1k": eur, "output_per_1k": eur}}`. Embedding models have
`input_per_1k` only. The functions take the table as an argument, so they need no
settings and no model keys.
"""

from __future__ import annotations

from typing import Any, NamedTuple

PriceTable = dict[str, dict[str, float]]


class Totals(NamedTuple):
    tokens: int
    cost_eur: float


def validate_prices(price_table: PriceTable, models: list[str]) -> None:
    """Raise if a model has no price, so a missing price is found at startup."""
    missing = [model for model in models if model not in price_table]
    if missing:
        raise ValueError(f"PRICE_TABLE_EUR has no price for: {', '.join(missing)}")


def call_cost_eur(
    price_table: PriceTable, model: str, input_tokens: int, output_tokens: int = 0
) -> float:
    """The cost of one call in EUR."""
    validate_prices(price_table, [model])
    prices = price_table[model]
    cost = input_tokens / 1000 * prices["input_per_1k"]
    cost += output_tokens / 1000 * prices.get("output_per_1k", 0.0)
    return round(cost, 8)


def model_calls(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The successful model calls in a request's evidence, with model, tokens and cost.
    Failed tries are left out because they have no tokens and no cost.
    """
    return [
        {
            "model": item["model"],
            "input_tokens": item["input_tokens"],
            "output_tokens": item["output_tokens"],
            "cost_eur": item["cost_eur"],
        }
        for item in evidence
        if item["type"] == "model_call" and item["status"] == "ok"
    ]


def request_total_cost(evidence: list[dict[str, Any]]) -> Totals:
    """The total tokens (input plus output) and total cost of a request."""
    calls = model_calls(evidence)
    tokens = sum(call["input_tokens"] + call["output_tokens"] for call in calls)
    cost_eur = round(sum(call["cost_eur"] for call in calls), 8)
    return Totals(tokens, cost_eur)
