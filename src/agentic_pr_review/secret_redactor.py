"""Secret redaction for prompts, logs, artifacts, and published comments."""

from __future__ import annotations

import base64
from collections.abc import Mapping, Sequence
import hashlib
import os
import re
from typing import Any


REDACTED_SECRET = "[REDACTED_SECRET]"
REDACTED_TOKEN = "[REDACTED_TOKEN]"
REDACTED_AUTH = "[REDACTED_AUTH]"
REDACTED_PRIVATE_KEY = "[REDACTED_PRIVATE_KEY]"

MIN_KNOWN_SECRET_LENGTH = 8
SECRET_ENV_NAME_RE = re.compile(
    r"(SECRET|TOKEN|PRIVATE[_-]?KEY|APP[_-]?KEY|API[_-]?KEY|PASSWORD|CREDENTIAL|AUTH|CLIENT[_-]?ID)",
    re.I,
)
LOW_VALUE_SECRET_STRINGS = {
    "true",
    "false",
    "none",
    "null",
    "nil",
    "0",
    "1",
    "yes",
    "no",
    "on",
    "off",
    "github",
    "gateway",
    "circuit",
}

PRIVATE_KEY_BLOCK_RE = re.compile(
    r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?-----END [A-Z0-9 ]*PRIVATE KEY-----",
    re.S,
)
AUTH_HEADER_RE = re.compile(
    r"(?i)\b(authorization\s*[:=]\s*)(?:bearer|basic)\s+[A-Za-z0-9._~+/=-]{8,}"
)
API_KEY_HEADER_RE = re.compile(
    r"(?i)\b((?:x-)?api[-_]?key\s*[:=]\s*)[A-Za-z0-9._~+/=-]{8,}"
)
QUOTED_SECRET_VALUE_RE = re.compile(
    r"(?i)\b(api[-_]?key|access[-_]?token|refresh[-_]?token|id[-_]?token|client[-_]?secret|client[-_]?id|password|private[-_]?key)\b"
    r"(\s*[:=]\s*)(['\"])([^'\"\n]{8,})(\3)"
)
GITHUB_CLASSIC_TOKEN_RE = re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b")
GITHUB_FINE_GRAINED_TOKEN_RE = re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b")
OPENAI_TOKEN_RE = re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b")
AWS_ACCESS_KEY_RE = re.compile(r"\bA(?:KIA|SIA)[0-9A-Z]{16}\b")
JWT_RE = re.compile(
    r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"
)


def redact_text(text: Any, *, extra_values: Sequence[str] | None = None) -> str:
    """Redact known runtime secrets and obvious token literals from text."""

    output = str(text)
    for value in known_secret_values(extra_values=extra_values):
        output = output.replace(value, REDACTED_SECRET)

    output = PRIVATE_KEY_BLOCK_RE.sub(REDACTED_PRIVATE_KEY, output)
    output = AUTH_HEADER_RE.sub(lambda match: f"{match.group(1)}{REDACTED_AUTH}", output)
    output = API_KEY_HEADER_RE.sub(lambda match: f"{match.group(1)}{REDACTED_SECRET}", output)
    output = QUOTED_SECRET_VALUE_RE.sub(
        lambda match: f"{match.group(1)}{match.group(2)}{match.group(3)}{REDACTED_SECRET}{match.group(5)}",
        output,
    )
    output = GITHUB_FINE_GRAINED_TOKEN_RE.sub(REDACTED_TOKEN, output)
    output = GITHUB_CLASSIC_TOKEN_RE.sub(REDACTED_TOKEN, output)
    output = OPENAI_TOKEN_RE.sub(REDACTED_TOKEN, output)
    output = AWS_ACCESS_KEY_RE.sub(REDACTED_TOKEN, output)
    output = JWT_RE.sub(REDACTED_TOKEN, output)
    return output


def redact_obj(value: Any, *, extra_values: Sequence[str] | None = None) -> Any:
    """Recursively redact string values from JSON-like objects."""

    if isinstance(value, str):
        return redact_text(value, extra_values=extra_values)
    if isinstance(value, Mapping):
        return {key: redact_obj(item, extra_values=extra_values) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(redact_obj(item, extra_values=extra_values) for item in value)
    if isinstance(value, list):
        return [redact_obj(item, extra_values=extra_values) for item in value]
    return value


def known_secret_values(
    *,
    extra_values: Sequence[str] | None = None,
    env: Mapping[str, str] | None = None,
) -> tuple[str, ...]:
    """Return runtime secret values and derived auth encodings worth redacting."""

    source = env if env is not None else os.environ
    values: set[str] = set()

    for key, value in source.items():
        if not SECRET_ENV_NAME_RE.search(str(key)):
            continue
        add_secret_variants(values, str(value))

    for prefix in ("GATEWAY", "CIRCUIT"):
        client_id = source.get(f"{prefix}_CLIENT_ID")
        client_secret = source.get(f"{prefix}_CLIENT_SECRET")
        if client_id and client_secret:
            basic = base64.b64encode(f"{client_id}:{client_secret}".encode("utf-8")).decode("ascii")
            add_secret_variants(values, basic)

    for value in extra_values or ():
        add_secret_variants(values, str(value))

    return tuple(sorted(values, key=len, reverse=True))


def add_secret_variants(values: set[str], raw: str) -> None:
    value = raw.strip()
    if not should_redact_known_value(value):
        return
    values.add(value)
    if "\\n" in value:
        expanded = value.replace("\\n", "\n")
        if should_redact_known_value(expanded):
            values.add(expanded)
    if "\n" in value:
        escaped = value.replace("\n", "\\n")
        if should_redact_known_value(escaped):
            values.add(escaped)


def should_redact_known_value(value: str) -> bool:
    if len(value) < MIN_KNOWN_SECRET_LENGTH:
        return False
    if value.lower() in LOW_VALUE_SECRET_STRINGS:
        return False
    if re.fullmatch(r"\d+", value):
        return False
    return True


def secret_fingerprint(value: str | None) -> str:
    """Return a non-reversible stable identifier for secret-backed metadata."""

    if not value:
        return ""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
