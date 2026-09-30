"""LLM client. It wraps the OpenAI SDK client and its `responses.create` call with
retries and a fallback model. A call is tried a few times on the main model. If all
tries fail, it is tried on the fallback model, when one is set. If that fails too, it
raises `LLMUnavailableError`, and the graph ends the request with `escalate`.

On Azure, `model` is the name of the deployment.

Every try, every switch to the fallback model and the final failure adds an evidence
item to the request and a `model_call` audit event. The item's `type` tells them apart.
The whole chain can be read back from the Decision Record. A successful try also
carries its token counts and its cost in EUR.

The retry rules follow the OpenAI SDK: retry on connection errors and on status 408,
409, 429 and 5xx. The SDK's own retries are turned off, so the limit is only in this file.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import openai
from openai import OpenAI
from openai.types.responses import Response

from support_agent import audit
from support_agent.config import get_settings
from support_agent.costing import PriceTable, call_cost_eur, validate_prices
from support_agent.prompts import Prompt
from support_agent.request_context import get_request_context

# same defaults as the OpenAI SDK
MAX_RETRIES = 2
INITIAL_RETRY_DELAY = 0.5
TIMEOUT_SECONDS = 30.0


class LLMUnavailableError(Exception):
    """Every try on every model failed."""

    def __init__(self, models_tried: list[str], last_error: Exception) -> None:
        super().__init__(f"no model answered, tried: {', '.join(models_tried)}")
        self.models_tried = models_tried
        self.last_error = last_error


def is_retryable(error: Exception) -> bool:
    if isinstance(error, openai.APIConnectionError):
        return True
    return isinstance(error, openai.APIStatusError) and (
        error.status_code in (408, 409, 429) or error.status_code >= 500
    )


class LLMClient:
    def __init__(
        self,
        client: OpenAI,
        price_table: PriceTable,
        fallback_model: str = "",
        max_retries: int = MAX_RETRIES,
        retry_delay: float = INITIAL_RETRY_DELAY,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._client = client
        self._price_table = price_table
        self._fallback = fallback_model
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._sleep = sleep

    def create(
        self,
        model: str,
        input: list[dict[str, Any]],
        prompt: Prompt | None = None,
        **options: Any,
    ) -> Response:
        """Call `responses.create` with retry and fallback. Pass `prompt` to record its
        id and version with every try. Other options, such as `instructions` or
        `tools`, go to the SDK unchanged. Needs a bound request context.
        """
        options.setdefault("store", False)
        models = [model]
        if self._fallback and self._fallback != model:
            models.append(self._fallback)
        prompt_tag = prompt.tag if prompt else None
        tries = self._max_retries + 1
        last_error: Exception | None = None

        for position, current in enumerate(models):
            for attempt in range(1, tries + 1):
                try:
                    response = self._client.responses.create(model=current, input=input, **options)
                    if response.status == "failed":
                        raise RuntimeError(response.error.message if response.error else "failed")
                except Exception as exc:
                    last_error = exc
                    retryable = is_retryable(exc)
                    status = "retryable_error" if retryable else "error"
                    self._record(current, attempt, status, prompt_tag, error=exc)
                    if retryable and attempt < tries:
                        # wait 1x, 2x, 4x ... the base delay before the next try
                        self._sleep(self._retry_delay * 2 ** (attempt - 1))
                        continue
                    break
                self._record(current, attempt, "ok", prompt_tag, response=response)
                return response

            if position + 1 < len(models):
                self._add_audit_and_evidence(
                    {"type": "model_fallback", "from": current, "to": models[position + 1]}
                )

        self._add_audit_and_evidence(
            {"type": "model_degraded", "outcome": "escalate", "models_tried": models}
        )
        raise LLMUnavailableError(models, last_error)

    def _record(
        self,
        model: str,
        attempt: int,
        status: str,
        prompt_tag: dict[str, str] | None,
        error: Exception | None = None,
        response: Response | None = None,
    ) -> None:
        item: dict[str, Any] = {
            "type": "model_call",
            "model": model,
            "attempt": attempt,
            "status": status,
            "prompt": prompt_tag,
        }
        if error is not None:
            item["error"] = f"{type(error).__name__}: {error}"[:200]
            item["status_code"] = getattr(error, "status_code", None)
        if response is not None:
            usage = response.usage
            item["input_tokens"] = usage.input_tokens if usage else 0
            item["output_tokens"] = usage.output_tokens if usage else 0
            item["cost_eur"] = call_cost_eur(
                self._price_table, model, item["input_tokens"], item["output_tokens"]
            )
        self._add_audit_and_evidence(item)

    def _add_audit_and_evidence(self, item: dict[str, Any]) -> None:
        """Write the item to the audit log and add it to the request's evidence."""
        audit.record("model_call", item)
        get_request_context().evidence.append(item)


def get_llm_client() -> LLMClient:
    """The client the app uses: Azure OpenAI through the OpenAI SDK, with the fallback
    model from settings.
    """
    settings = get_settings()
    models = [settings.azure_openai_deployment_agent, settings.azure_openai_deployment_classifier]
    if settings.model_fallback_deployment:
        models.append(settings.model_fallback_deployment)
    validate_prices(settings.price_table_eur, models)
    client = OpenAI(
        base_url=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
        max_retries=0,
        timeout=TIMEOUT_SECONDS,
    )
    return LLMClient(
        client, settings.price_table_eur, fallback_model=settings.model_fallback_deployment
    )
