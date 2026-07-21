"""JSON and text helpers."""

from __future__ import annotations

import json
import re
from typing import Any


def truncate_text(value: str | None, limit: int) -> str:
    if not value:
        return ""
    if len(value) <= limit:
        return value
    omitted = len(value) - limit
    return f"{value[:limit]}\n\n...[truncated {omitted} characters]..."


def dump_json(data: Any) -> str:
    return json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False)


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract the first JSON object from a model response.

    Model gateways occasionally wrap otherwise-valid JSON in Markdown fences or
    short prose, or return JSON with harmless trailing commas. Recover those
    cases locally before spending another model request on repair.
    """
    text = text.strip()
    candidates = [text]

    fenced = re.findall(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    candidates.extend(item.strip() for item in fenced if item.strip())

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1])

    last_error: Exception | None = None
    for candidate in candidates:
        for variant in (candidate, remove_json_trailing_commas(candidate)):
            try:
                value = json.loads(variant)
            except json.JSONDecodeError as exc:
                last_error = exc
                continue
            if not isinstance(value, dict):
                raise ValueError("Expected model response to be a JSON object")
            return value
    if last_error is not None:
        raise last_error
    raise ValueError("Expected model response to include a JSON object")


def remove_json_trailing_commas(text: str) -> str:
    """Remove trailing commas before object/array closers outside strings."""

    output = []
    in_string = False
    escaped = False
    index = 0
    while index < len(text):
        char = text[index]
        if in_string:
            output.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == "\"":
                in_string = False
            index += 1
            continue
        if char == "\"":
            in_string = True
            output.append(char)
            index += 1
            continue
        if char == ",":
            lookahead = index + 1
            while lookahead < len(text) and text[lookahead].isspace():
                lookahead += 1
            if lookahead < len(text) and text[lookahead] in "}]":
                index += 1
                continue
        output.append(char)
        index += 1
    return "".join(output)
