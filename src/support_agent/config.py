"""Settings loaded from the environment. The single source of runtime configuration."""

from __future__ import annotations

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_MAX_TOOL_CALLS_PER_REQUEST = 8

# how long a reasoning model thinks before it answers. An empty string means the setting
# is not sent, which is needed for a deployment that is not a reasoning model.
ReasoningEffort = Literal["", "minimal", "low", "medium", "high"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str

    # the Azure v1 API base, e.g. https://<resource>.services.ai.azure.com/openai/v1
    azure_openai_endpoint: str
    azure_openai_api_key: str
    azure_openai_deployment_agent: str
    azure_openai_deployment_classifier: str
    azure_openai_deployment_embedding: str
    # empty string means no fallback is configured
    model_fallback_deployment: str = ""

    # one setting for each deployment, because the classifier only picks a label and
    # needs far less thinking than the agent loop. The fallback deployment has its own
    # setting, because it may not be a reasoning model and would refuse the option.
    reasoning_effort_classifier: ReasoningEffort = "minimal"
    reasoning_effort_agent: ReasoningEffort = "low"
    reasoning_effort_fallback: ReasoningEffort = ""

    max_tool_calls_per_request: int = DEFAULT_MAX_TOOL_CALLS_PER_REQUEST

    # the level of the app's own loggers. Every other logger stays at WARNING.
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # a pending action without a Decision Record is flagged only after this long, so a
    # request that is still running is never flagged. Read by reconciliation.py.
    orphan_grace_minutes: int = 10

    # Langfuse tracing is off unless both keys are set. Read by tracing.py.
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_base_url: str = "https://cloud.langfuse.com"
    # when true, spans and model calls carry their inputs and outputs: the employee's
    # message, the tool arguments and results, the prompts and the answers. When false,
    # they carry names, times, token counts and statuses only.
    trace_content: bool = False

    # {"<deployment>": {"input_per_1k": eur, "output_per_1k": eur}} — embedding
    # deployments carry input_per_1k only. Read by costing.py.
    price_table_eur: dict[str, dict[str, float]] = Field(default_factory=dict)

    # DE general civil-claims limitation period (§195 BGB): both tables are
    # evidence an employee could need to dispute a decision, so both survive
    # the window a claim could still be raised
    retention_days_decisions: int = 1095
    retention_days_audit: int = 1095


def get_settings() -> Settings:
    return Settings()
