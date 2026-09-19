"""Settings loaded from the environment. The single source of runtime configuration."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str

    azure_openai_endpoint: str
    azure_openai_api_key: str
    azure_openai_api_version: str
    azure_openai_deployment_agent: str
    azure_openai_deployment_classifier: str
    azure_openai_deployment_embedding: str
    # empty string means no fallback is configured
    model_fallback_deployment: str = ""

    max_tool_calls_per_request: int = 8

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
