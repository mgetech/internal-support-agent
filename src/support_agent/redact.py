"""Removes secrets from text before it is stored. An exception message can have a
connection string, a key or a token in it. The Decision Record keeps the message for
engineers, so the secrets must not be in it.

This is a best effort on the shapes below. It cuts too much rather than too little, so a
long mix of letters and digits is removed even when it was an id.
"""

from __future__ import annotations

import re

REDACTED = "[redacted]"

# scheme://user:password@host -> scheme://[redacted]@host
_URL_CREDENTIALS = re.compile(r"(?P<scheme>\b[A-Za-z][A-Za-z0-9+.-]*://)[^\s/@]+@")

# password=..., api_key: ..., X-Api-Key: ..., access_token=..., password is ...
# The name can have a prefix such as "access_" or "x-". "KeyError" is not a name, because
# a letter follows "Key".
_SECRET_VALUE = re.compile(
    r"""
    (?P<name>
        (?<![A-Za-z])
        (?:[A-Za-z0-9]+[_-])*
        (?:api[_-]?key|apikey|password|passwd|pwd|secret|token|key)
        (?![A-Za-z])
    )
    (?P<separator>["']?\s*(?:=>|[=:])\s*|\s+is\s+)
    (?P<value>\[redacted\]|"[^"]*"|'[^']*'|[^\s,;&)}\]]+)
    """,
    re.IGNORECASE | re.VERBOSE,
)

# "Authorization: Bearer abc123" and "Bearer abc123"
_BEARER = re.compile(r"(?P<name>\bBearer\s+)[A-Za-z0-9._~+/=-]{8,}", re.IGNORECASE)

# values that look like a key, wherever they appear
_KEY_SHAPES = [
    # OpenAI style, and Langfuse: sk-..., pk-lf-..., sk-lf-...
    re.compile(r"\b(?:sk|pk)-[A-Za-z0-9_-]{16,}"),
    # AWS access key id
    re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    # GitHub tokens
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}"),
    # JSON web token
    re.compile(r"\beyJ[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]*"),
    # a long run of letters and digits, such as the 32 hex digits of an Azure key
    re.compile(
        r"(?<![A-Za-z0-9_-])(?=[A-Za-z0-9_-]*\d)(?=[A-Za-z0-9_-]*[A-Za-z])[A-Za-z0-9_-]{32,}"
    ),
]


def redact_secrets(text: str) -> str:
    """The text with connection string credentials, the values of password, key, token and
    secret settings, bearer tokens and anything shaped like an API key replaced by
    "[redacted]".
    """
    text = _URL_CREDENTIALS.sub(rf"\g<scheme>{REDACTED}@", text)
    text = _SECRET_VALUE.sub(rf"\g<name>\g<separator>{REDACTED}", text)
    text = _BEARER.sub(rf"\g<name>{REDACTED}", text)
    for shape in _KEY_SHAPES:
        text = shape.sub(REDACTED, text)
    return text
