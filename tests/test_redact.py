"""Secrets are removed from an exception message before it is stored in a Decision Record.
The samples are made up. None is a real key.
"""

import pytest

from support_agent.redact import REDACTED, redact_secrets

AZURE_KEY = "9f8e7d6c5b4a39281706f5e4d3c2b1a0"
OPENAI_KEY = "sk-proj-AbCdEfGhIjKlMnOpQrStUvWxYz0123456789"
LANGFUSE_SECRET = "sk-lf-3b9c2a10-5d4e-4f6a-8b7c-1d2e3f4a5b6c"
LANGFUSE_PUBLIC = "pk-lf-7a6b5c4d-3e2f-4a1b-9c8d-0e1f2a3b4c5d"
JWT = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJlbXBfMDAxIn0.c2lnbmF0dXJl"


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        # connection strings with credentials: the user and the password go, the host stays
        (
            'connection to "postgresql://support_agent:s3cret-pw@db:5432/support_agent" failed',
            'connection to "postgresql://[redacted]@db:5432/support_agent" failed',
        ),
        (
            "cannot reach redis://:hunter2@cache.internal:6379/0",
            "cannot reach redis://[redacted]@cache.internal:6379/0",
        ),
        ("GET https://user:pass@example.com/path", "GET https://[redacted]@example.com/path"),
        # key=value connection strings
        (
            "connection failed: host=db user=support_agent password=s3cret dbname=app",
            "connection failed: host=db user=support_agent password=[redacted] dbname=app",
        ),
        # the value after password, key and token
        ("password=hunter2", "password=[redacted]"),
        ("Password: hunter2, retry later", "Password: [redacted], retry later"),
        ('password: "two words here" rejected', "password: [redacted] rejected"),
        ("PASSWORD => 'abc def'", "PASSWORD => [redacted]"),
        ("the password is hunter2", "the password is [redacted]"),
        ("api_key=abc123", "api_key=[redacted]"),
        ("api-key: abc123", "api-key: [redacted]"),
        ("X-Api-Key: abc123", "X-Api-Key: [redacted]"),
        ("apikey=abc123", "apikey=[redacted]"),
        ("access_token=abc123&next=1", "access_token=[redacted]&next=1"),
        ("token: abc123", "token: [redacted]"),
        ("client_secret=abc123", "client_secret=[redacted]"),
        ("a key: abc123 was sent", "a key: [redacted] was sent"),
        ('{"password": "hunter2", "user": "anna"}', '{"password": [redacted], "user": "anna"}'),
        # bearer tokens
        ("Authorization: Bearer abcdef0123456789xyz", "Authorization: Bearer [redacted]"),
        # anything shaped like a key
        (f"Incorrect API key provided: {OPENAI_KEY}.", f"Incorrect API key provided: {REDACTED}."),
        (f"auth failed for {AZURE_KEY}", f"auth failed for {REDACTED}"),
        (
            f"public {LANGFUSE_PUBLIC} secret {LANGFUSE_SECRET}",
            f"public {REDACTED} secret {REDACTED}",
        ),
        ("id AKIAIOSFODNN7EXAMPLE rejected", f"id {REDACTED} rejected"),
        ("token ghp_" + "a1B2c3D4e5" * 4, f"token {REDACTED}"),
        (f"bad jwt {JWT}", f"bad jwt {REDACTED}"),
        # several in one message
        (
            f"postgresql://u:p@h/db failed, password=x, key {OPENAI_KEY}",
            f"postgresql://[redacted]@h/db failed, password=[redacted], key {REDACTED}",
        ),
    ],
)
def test_secrets_are_removed(message, expected):
    assert redact_secrets(message) == expected


@pytest.mark.parametrize(
    "message",
    [
        "cannot read the record of emp_001",
        "KeyError: 'employee_id'",
        "password authentication failed for user",
        "the key was not found in the table",
        "monkeys and keyboards",
        "TimeoutError: timed out after 30 s",
        "vacation-policy#entitlement#0 was not retrieved",
        "connection to server at db (172.18.0.2), port 5432 failed",
        "invalid literal for int() with base 10: '4,5'",
        "https://example.com/path?page=2",
        "",
    ],
)
def test_text_without_a_secret_is_not_changed(message):
    assert redact_secrets(message) == message


def test_redacting_twice_gives_the_same_text():
    once = redact_secrets(f"postgresql://u:p@h/db password=x {OPENAI_KEY}")

    assert redact_secrets(once) == once


def test_no_secret_is_left_in_a_realistic_database_error():
    message = (
        'connection failed: connection to server at "localhost" (::1), port 5432 failed: '
        'FATAL: password authentication failed for user "support_agent" '
        "(DSN: postgresql://support_agent:s3cret-pw@localhost:5432/support_agent)"
    )

    redacted = redact_secrets(message)

    assert "s3cret-pw" not in redacted
    assert "support_agent:" not in redacted
    assert "localhost" in redacted
