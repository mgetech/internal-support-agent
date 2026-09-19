import pytest
from pydantic import ValidationError

from support_agent.config import Settings

REQUIRED = {
    "database_url": "postgresql://user:pass@localhost:5432/support_agent",
    "azure_openai_endpoint": "https://example.openai.azure.com",
    "azure_openai_api_key": "test-key",
    "azure_openai_api_version": "2024-10-21",
    "azure_openai_deployment_agent": "gpt-4.1",
    "azure_openai_deployment_classifier": "gpt-4.1-mini",
    "azure_openai_deployment_embedding": "text-embedding-3-small",
}


def test_defaults_when_only_required_fields_given():
    settings = Settings(_env_file=None, **REQUIRED)

    assert settings.max_tool_calls_per_request == 8
    assert settings.model_fallback_deployment == ""
    assert settings.retention_days_decisions == 1095
    assert settings.retention_days_audit == 1095
    assert settings.price_table_eur == {}


def test_missing_required_field_raises():
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_loads_from_environment(monkeypatch):
    for key, value in REQUIRED.items():
        monkeypatch.setenv(key.upper(), value)
    monkeypatch.setenv("MAX_TOOL_CALLS_PER_REQUEST", "3")
    monkeypatch.setenv("MODEL_FALLBACK_DEPLOYMENT", "gpt-4.1-mini")
    monkeypatch.setenv(
        "PRICE_TABLE_EUR",
        '{"gpt-4.1": {"input_per_1k": 0.005, "output_per_1k": 0.015}}',
    )
    monkeypatch.setenv("RETENTION_DAYS_DECISIONS", "30")
    monkeypatch.setenv("RETENTION_DAYS_AUDIT", "180")

    settings = Settings(_env_file=None)

    assert settings.max_tool_calls_per_request == 3
    assert settings.model_fallback_deployment == "gpt-4.1-mini"
    assert settings.price_table_eur == {"gpt-4.1": {"input_per_1k": 0.005, "output_per_1k": 0.015}}
    assert settings.retention_days_decisions == 30
    assert settings.retention_days_audit == 180
