"""Chat-completions gateway reviewer with OAuth client-credentials auth."""

from __future__ import annotations

import base64
from copy import deepcopy
from io import BytesIO
import json
import os
import threading
import time
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .config import RuntimeConfig
from .deep_review import build_chunk_review_input, dedupe_findings, filter_findings_for_chunk
from .json_utils import extract_json_object, truncate_text
from .models import normalize_review_output
from .progress import AdaptiveETA, format_duration
from .prompt import SYSTEM_PROMPT, build_synthesis_prompt, build_user_prompt
from .secret_redactor import redact_obj, redact_text


GATEWAY_HEARTBEAT_SECONDS = 60
TOKEN_EXPIRY_SKEW_SECONDS = 60
GATEWAY_AUTH_RETRY_STATUS_CODES = {401, 403}
GATEWAY_RESPONSE_FORMAT_RETRY_STATUS_CODES = {400, 422}

REVIEW_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "summary": {"type": "string"},
        "overall_status": {
            "type": "string",
            "enum": ["needs_review", "looks_good", "blocked_by_ci", "error"],
        },
        "safe_to_publish": {"type": "boolean"},
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "id": {"type": "string"},
                    "title": {"type": "string"},
                    "category": {
                        "type": "string",
                        "enum": [
                            "api_auth_correctness",
                            "polling_checkpoint",
                            "output_schema_mismatch",
                            "unsafe_logging",
                            "soar_metadata",
                            "docs_pr_accuracy",
                            "pagination",
                            "validation",
                            "missing_tests",
                            "precommit",
                            "merge_conflict",
                            "ci_synthesis",
                            "general",
                        ],
                    },
                    "severity": {"type": "string", "enum": ["critical", "high", "medium", "low", "info"]},
                    "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                    "file": {"type": ["string", "null"]},
                    "line": {"type": ["integer", "null"]},
                    "code_reference": {"type": ["string", "null"]},
                    "evidence": {"type": "string"},
                    "why_it_matters": {"type": "string"},
                    "suggested_fix": {"type": "string"},
                    "suggested_code": {"type": ["string", "null"]},
                    "source": {"type": "string"},
                },
                "required": [
                    "id",
                    "title",
                    "category",
                    "severity",
                    "confidence",
                    "file",
                    "line",
                    "code_reference",
                    "evidence",
                    "why_it_matters",
                    "suggested_fix",
                    "suggested_code",
                    "source",
                ],
            },
        },
        "model_notes": {"type": "string"},
    },
    "required": ["summary", "overall_status", "safe_to_publish", "findings", "model_notes"],
}


class GatewayReviewError(RuntimeError):
    pass


class GatewayModelFormatError(GatewayReviewError):
    pass


class GatewayClaudeReviewer:
    def __init__(
        self,
        config: RuntimeConfig,
        *,
        max_tokens: int = 4500,
        temperature: float = 0.0,
        progress: Callable[[str], None] | None = None,
    ) -> None:
        config.require_model()
        self.config = config
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.progress_enabled = progress is not None
        self.progress = progress or (lambda _message: None)
        self.eta: AdaptiveETA | None = None
        # Some gateways require the OAuth access token in both auth headers.
        # Generate it once per bot run and refresh it in memory before expiry.
        self._access_token: str | None = None
        self._access_token_expires_at = 0.0
        self._response_format_mode: str | None = "json_schema"

    def review(self, review_input: dict[str, Any], deterministic_findings: list[dict[str, Any]]) -> dict[str, Any]:
        if self.eta is None:
            self.eta = AdaptiveETA(total_requests=1)
            self.progress(f"Single model request started; estimated whole-review time remaining: {self._eta_text()}.")
        user_prompt = build_user_prompt(
            review_input,
            deterministic_findings,
            max_chars=self.config.max_model_input_chars,
        )
        return self._invoke_review(user_prompt, deterministic_findings)

    def review_deep(
        self,
        review_input: dict[str, Any],
        deterministic_findings: list[dict[str, Any]],
        *,
        max_chunks: int = 0,
    ) -> dict[str, Any]:
        chunks = list(((review_input.get("deep_review") or {}).get("chunks")) or [])
        if max_chunks > 0:
            chunks = chunks[:max_chunks]
        if not chunks:
            self.progress("No deep-review chunks found; starting a single model review.")
            return self.review(review_input, deterministic_findings)

        chunk_outputs: list[dict[str, Any]] = []
        total_chunks = len(chunks)
        self.eta = AdaptiveETA(total_requests=total_chunks + 1)
        self.progress(
            f"Deep model review started: {total_chunks} chunk(s) plus synthesis; "
            f"estimated whole-review time remaining: {self._eta_text()}."
        )
        for position, chunk in enumerate(chunks, start=1):
            path = str(chunk.get("path") or "unknown file")
            file_chunk = chunk.get("chunk_index")
            file_total = chunk.get("chunk_total")
            chunk_started = time.monotonic()
            self.progress(
                f"Deep-review chunk {position}/{total_chunks} started: "
                f"{path} (file chunk {file_chunk}/{file_total}); "
                f"estimated whole-review time remaining: {self._eta_text()}."
            )
            chunk_input = build_chunk_review_input(review_input, chunk)
            chunk_deterministic = filter_findings_for_chunk(deterministic_findings, chunk)
            chunk_prompt = build_user_prompt(
                chunk_input,
                chunk_deterministic,
                max_chars=self.config.max_model_input_chars,
            )
            output = self._invoke_review(chunk_prompt, chunk_deterministic)
            output["chunk_id"] = chunk.get("id")
            output["chunk_path"] = chunk.get("path")
            output["chunk_index"] = chunk.get("chunk_index")
            output["chunk_total"] = chunk.get("chunk_total")
            chunk_outputs.append(output)
            chunk_elapsed = format_duration(time.monotonic() - chunk_started)
            finding_count = len(output.get("findings") or [])
            self.progress(
                f"Deep-review chunk {position}/{total_chunks} completed in {chunk_elapsed} "
                f"with {finding_count} finding(s); estimated whole-review time remaining: {self._eta_text()}."
            )

        synthesis_started = time.monotonic()
        self.progress(
            f"Synthesis started for {total_chunks} reviewed chunk(s); "
            f"estimated whole-review time remaining: {self._eta_text()}."
        )
        synthesis_prompt = build_synthesis_prompt(
            review_input,
            deterministic_findings,
            chunk_outputs,
            max_chars=self.config.max_model_input_chars,
        )
        try:
            final_output = self._invoke_review(synthesis_prompt, deterministic_findings)
        except GatewayReviewError as exc:
            safe_error = redact_text(str(exc))
            self.progress(f"Synthesis failed; using deduplicated chunk findings: {safe_error}")
            final_output = fallback_deep_output(deterministic_findings, chunk_outputs, model_notes=safe_error)
        synthesis_elapsed = format_duration(time.monotonic() - synthesis_started)
        self.progress(f"Synthesis completed in {synthesis_elapsed}.")
        final_output["deep_review"] = {
            "enabled": True,
            "chunk_count": len(chunks),
            "reviewed_chunk_count": len(chunk_outputs),
            "max_chunks": max_chunks,
        }
        final_output["chunk_review_outputs"] = chunk_outputs
        return final_output

    def _invoke_review(self, user_prompt: str, deterministic_findings: list[dict[str, Any]]) -> dict[str, Any]:
        body = self._model_body(user_prompt)
        heartbeat_stop = threading.Event()
        heartbeat_thread: threading.Thread | None = None
        request_started = time.monotonic()
        if self.progress_enabled:
            heartbeat_thread = threading.Thread(
                target=self._report_gateway_heartbeat,
                args=(heartbeat_stop, request_started),
                name="gateway-progress-heartbeat",
                daemon=True,
            )
            heartbeat_thread.start()
        request_succeeded = False
        try:
            try:
                response_payload = self._post_model_json(body)
                request_succeeded = True
            except (HTTPError, URLError, TimeoutError, OSError) as exc:
                raise GatewayReviewError(f"Gateway model invocation failed: {redact_text(format_http_error(exc))}") from exc
        finally:
            heartbeat_stop.set()
            if heartbeat_thread is not None:
                heartbeat_thread.join(timeout=1)
            if request_succeeded and self.eta is not None:
                self.eta.complete_request(time.monotonic() - request_started)

        text = extract_chat_completion_text(response_payload)
        recovery_note = ""
        try:
            raw_output = extract_json_object(text)
        except Exception as exc:  # noqa: BLE001 - keep raw model text for prototype debugging
            raw_output, response_payload, recovery_note = self._recover_non_json_response(text, user_prompt)

        normalized = normalize_review_output(raw_output, deterministic_findings=deterministic_findings)
        if recovery_note:
            existing_notes = str(normalized.get("model_notes") or "").strip()
            normalized["model_notes"] = f"{existing_notes}\n{recovery_note}".strip() if existing_notes else recovery_note
        normalized["model"] = response_payload.get("model", self.config.gateway_model)
        normalized["usage"] = response_payload.get("usage")
        return normalized

    def _recover_non_json_response(
        self,
        previous_text: str,
        original_user_prompt: str,
    ) -> tuple[dict[str, Any], dict[str, Any], str]:
        self.progress("Gateway model returned non-JSON; attempting one JSON repair request.")
        try:
            repair_payload = self._post_model_json(self._model_body(build_json_repair_prompt(previous_text)))
            repair_text = extract_chat_completion_text(repair_payload)
            raw_output = extract_json_object(repair_text)
            return raw_output, repair_payload, "Recovered from a non-JSON model response with one JSON repair retry."
        except Exception:
            self.progress("Gateway JSON repair failed; retrying the original review with strict JSON-only instructions.")

        try:
            retry_payload = self._post_model_json(self._model_body(build_strict_json_retry_prompt(original_user_prompt)))
            retry_text = extract_chat_completion_text(retry_payload)
            raw_output = extract_json_object(retry_text)
            return raw_output, retry_payload, "Recovered from a non-JSON model response with a strict JSON retry."
        except Exception as retry_exc:  # noqa: BLE001 - retry can fail in several parser/provider ways
            raise GatewayModelFormatError(
                "Gateway model did not return valid JSON after repair and strict retry: "
                f"{redact_text(previous_text[:2000])}"
            ) from retry_exc

    def _model_body(self, user_prompt: str) -> dict[str, Any]:
        safe_user_prompt = redact_text(user_prompt)
        body = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": safe_user_prompt},
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            # CIRCUIT validates this exact JSON string metadata field.
            # It is auth metadata, not part of the chat prompt.
            "user": json.dumps({"appkey": self.config.gateway_app_key}),
        }
        if should_include_model_in_body(str(self.config.gateway_base_url or "")):
            body["model"] = self.config.gateway_model
        response_format = self._response_format()
        if response_format is not None:
            body["response_format"] = response_format
        return body

    def _response_format(self) -> dict[str, Any] | None:
        if self._response_format_mode == "json_schema":
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": "agentic_pr_review_output",
                    "strict": True,
                    "schema": REVIEW_RESPONSE_SCHEMA,
                },
            }
        if self._response_format_mode == "json_object":
            return {"type": "json_object"}
        return None

    def _post_model_json(self, body: dict[str, Any]) -> dict[str, Any]:
        max_attempts = max(1, int(self.config.gateway_request_max_attempts))
        attempt = 1
        auth_refreshed = False
        while True:
            try:
                return self._post_model_json_once(body)
            except HTTPError as exc:
                downgraded_body = self._downgrade_response_format_after_rejection(exc, body)
                if downgraded_body is not None:
                    body = downgraded_body
                    continue
                if exc.code not in GATEWAY_AUTH_RETRY_STATUS_CODES or auth_refreshed:
                    raise
                self.progress("Gateway model auth failed; refreshing token and retrying once.")
                self._access_token = None
                self._access_token_expires_at = 0.0
                auth_refreshed = True
                continue
            except (TimeoutError, URLError, OSError) as exc:
                if attempt >= max_attempts:
                    raise
                delay = self.config.gateway_request_retry_backoff_seconds * attempt
                self.progress(
                    f"Gateway model request failed transiently on attempt {attempt}/{max_attempts}: "
                    f"{redact_text(format_http_error(exc))}. Retrying in {format_duration(delay)}."
                )
                if delay > 0:
                    time.sleep(delay)
                attempt += 1

    def _downgrade_response_format_after_rejection(
        self,
        exc: HTTPError,
        body: dict[str, Any],
    ) -> dict[str, Any] | None:
        if exc.code not in GATEWAY_RESPONSE_FORMAT_RETRY_STATUS_CODES:
            return None
        if "response_format" not in body:
            return None
        error_body = read_error_body_preserving_stream(exc).lower()
        error_text = f"{exc.reason} {error_body}".lower()
        if not any(term in error_text for term in ("response_format", "json_schema", "json_object")):
            return None

        next_body = deepcopy(body)
        current_mode = self._response_format_mode
        if current_mode == "json_schema":
            self._response_format_mode = "json_object"
            next_body["response_format"] = self._response_format()
            self.progress("Gateway rejected JSON schema response_format; retrying with JSON object mode.")
            return next_body
        if current_mode == "json_object":
            self._response_format_mode = None
            next_body.pop("response_format", None)
            self.progress("Gateway rejected JSON object response_format; retrying without provider response_format.")
            return next_body
        return None

    def _post_model_json_once(self, body: dict[str, Any]) -> dict[str, Any]:
        request = Request(
            str(self.config.gateway_base_url),
            data=json.dumps(body).encode("utf-8"),
            headers=self._model_headers(),
            method="POST",
        )
        with urlopen(request, timeout=self.config.gateway_request_timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))

    def _model_headers(self) -> dict[str, str]:
        token = self._get_access_token()
        return {
            "Authorization": f"Bearer {token}",
            "api-key": token,
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "agentic-pr-review-gateway",
        }

    def _get_access_token(self) -> str:
        now = time.time()
        if self._access_token and now < self._access_token_expires_at - TOKEN_EXPIRY_SKEW_SECONDS:
            return self._access_token
        if not self.config.gateway_token_url:
            raise GatewayReviewError("GATEWAY_TOKEN_URL is required for gateway token exchange.")
        payload = urlencode({"grant_type": "client_credentials"}).encode("utf-8")
        basic = base64.b64encode(
            f"{self.config.gateway_client_id}:{self.config.gateway_client_secret}".encode("utf-8")
        ).decode("ascii")
        request = Request(
            self.config.gateway_token_url,
            data=payload,
            headers={
                "Authorization": f"Basic {basic}",
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "*/*",
                "User-Agent": "agentic-pr-review-gateway",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=60) as response:
                token_payload = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            raise GatewayReviewError(f"Gateway token exchange failed: {redact_text(format_http_error(exc))}") from exc

        token = token_payload.get("access_token") or token_payload.get("token") or token_payload.get("id_token")
        if not token:
            keys = ", ".join(sorted(str(key) for key in token_payload.keys()))
            raise GatewayReviewError(f"Gateway token response did not include access_token/token/id_token. Keys: {keys}")
        expires_in = token_payload.get("expires_in")
        try:
            ttl = int(expires_in)
        except (TypeError, ValueError):
            ttl = 3600
        self._access_token = str(token)
        self._access_token_expires_at = now + max(ttl, TOKEN_EXPIRY_SKEW_SECONDS)
        mask_secret_for_github_actions(self._access_token)
        return self._access_token

    def _report_gateway_heartbeat(self, stop: threading.Event, request_started: float) -> None:
        while not stop.wait(GATEWAY_HEARTBEAT_SECONDS):
            current_request_elapsed = time.monotonic() - request_started
            request_elapsed = format_duration(current_request_elapsed)
            self.progress(
                f"Gateway request still running ({request_elapsed} elapsed for this request); "
                f"estimated whole-review time remaining: "
                f"{self._eta_text(current_request_elapsed=current_request_elapsed)}."
            )

    def _eta_text(self, *, current_request_elapsed: float | None = None) -> str:
        if self.eta is None:
            return "calculating"
        remaining = self.eta.remaining_seconds(current_request_elapsed=current_request_elapsed)
        return f"~{format_duration(remaining)}"


def should_include_model_in_body(endpoint: str) -> bool:
    return "/deployments/" not in endpoint


def mask_secret_for_github_actions(value: str) -> None:
    if value and os.getenv("GITHUB_ACTIONS") == "true":
        print(f"::add-mask::{value}", flush=True)


def build_json_repair_prompt(previous_response: str) -> str:
    return (
        "Your previous response was not valid JSON. Convert it into exactly one "
        "valid JSON object using the schema below. Do not add Markdown, prose, "
        "or code fences. If the previous response does not prove any concrete "
        "actionable finding, return an empty findings array and overall_status "
        "`looks_good`.\n\n"
        "Required JSON shape:\n"
        "{\n"
        '  "summary": "",\n'
        '  "overall_status": "needs_review|looks_good|blocked_by_ci|error",\n'
        '  "safe_to_publish": true,\n'
        '  "findings": [\n'
        "    {\n"
        '      "id": "short-stable-id",\n'
        '      "title": "clear finding title",\n'
        '      "category": "api_auth_correctness|polling_checkpoint|output_schema_mismatch|unsafe_logging|soar_metadata|docs_pr_accuracy|pagination|validation|missing_tests|precommit|merge_conflict|ci_synthesis|general",\n'
        '      "severity": "critical|high|medium|low|info",\n'
        '      "confidence": "high|medium|low",\n'
        '      "file": "path or null",\n'
        '      "line": 123,\n'
        '      "code_reference": "reference or null",\n'
        '      "evidence": "specific evidence",\n'
        '      "why_it_matters": "why this matters",\n'
        '      "suggested_fix": "actionable fix",\n'
        '      "suggested_code": null,\n'
        '      "source": "claude"\n'
        "    }\n"
        "  ],\n"
        '  "model_notes": "brief notes"\n'
        "}\n\n"
        "Previous response:\n"
        f"{truncate_text(redact_text(previous_response), 8000)}"
    )


def build_strict_json_retry_prompt(original_user_prompt: str) -> str:
    return (
        "Retry the original review request below. The prior response was rejected because it was not valid JSON.\n"
        "You must return exactly one valid JSON object and nothing else. Do not include Markdown, prose, "
        "headings, explanations outside JSON, or code fences. Preserve review quality: inspect the same evidence "
        "and include only concrete, high-confidence actionable findings.\n\n"
        "Original review request:\n"
        f"{original_user_prompt}"
    )


def extract_chat_completion_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message")
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                return message["content"]
            if isinstance(first.get("text"), str):
                return first["text"]

    content = payload.get("content")
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                text_parts.append(item["text"])
            elif isinstance(item, str):
                text_parts.append(item)
        if text_parts:
            return "\n".join(text_parts)
    if isinstance(payload.get("output_text"), str):
        return payload["output_text"]
    raise GatewayReviewError(f"Gateway response did not include text content: {redact_obj(payload)}")


def format_http_error(exc: BaseException) -> str:
    if isinstance(exc, HTTPError):
        body = redact_text(read_error_body(exc))
        return f"HTTP {exc.code} {redact_text(exc.reason)}: {body}".strip()
    return redact_text(str(exc))


def read_error_body(error: HTTPError) -> str:
    file_obj = getattr(error, "fp", None)
    if file_obj is None:
        return ""
    try:
        raw = file_obj.read()
    except Exception:  # noqa: BLE001 - diagnostic best effort
        return ""
    if isinstance(raw, bytes):
        return redact_text(raw.decode("utf-8", errors="replace")[:2000])
    return redact_text(str(raw)[:2000])


def read_error_body_preserving_stream(error: HTTPError) -> str:
    file_obj = getattr(error, "fp", None)
    if file_obj is None:
        return ""
    try:
        raw = file_obj.read()
    except Exception:  # noqa: BLE001 - diagnostic best effort
        return ""
    if isinstance(raw, bytes):
        error.fp = BytesIO(raw)
        return raw.decode("utf-8", errors="replace")[:2000]
    text = str(raw)
    error.fp = BytesIO(text.encode("utf-8"))
    return text[:2000]


def fallback_deep_output(
    deterministic_findings: list[dict[str, Any]],
    chunk_outputs: list[dict[str, Any]],
    *,
    model_notes: str,
) -> dict[str, Any]:
    findings = []
    for output in chunk_outputs:
        findings.extend(output.get("findings") or [])
    findings = dedupe_findings(findings)
    return normalize_review_output(
        {
            "summary": "Deep chunk review completed, but synthesis failed; using deduplicated chunk findings.",
            "overall_status": "needs_review" if findings else "looks_good",
            "safe_to_publish": True,
            "findings": findings,
            "model_notes": f"Synthesis failed: {redact_text(model_notes)}",
        },
        deterministic_findings=deterministic_findings,
    )
