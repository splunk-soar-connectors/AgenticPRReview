"""Chat-completions gateway reviewer with OAuth client-credentials auth."""

from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from io import BytesIO
import json
import os
import re
import threading
import time
from typing import Any, Callable
import uuid
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .config import RuntimeConfig
from .deep_review import (
    build_chunk_review_input,
    dedupe_findings,
    filter_findings_for_chunk,
    plan_circuit_review_chunks,
    split_chunk_for_adaptive_retry,
)
from .json_utils import extract_json_object, truncate_text
from .models import deterministic_only_output, normalize_review_output
from .progress import AdaptiveETA, format_duration
from .prompt import SYSTEM_PROMPT, build_synthesis_prompt, build_user_prompt
from .secret_redactor import redact_obj, redact_text


GATEWAY_HEARTBEAT_SECONDS = 60
TOKEN_EXPIRY_SKEW_SECONDS = 60
GATEWAY_AUTH_RETRY_STATUS_CODES = {401, 403}
GATEWAY_RESPONSE_FORMAT_RETRY_STATUS_CODES = {400, 422}
CHUNK_MODEL_INPUT_CHARS = 70_000
ADAPTIVE_SUBCHUNK_MODEL_INPUT_CHARS = 45_000
CIRCUIT_CHAT_TRANSACTION_LIMIT = 8

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


class GatewayTransientModelError(GatewayReviewError):
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
        self._token_lock = threading.Lock()
        self._response_format_lock = threading.Lock()
        self._eta_lock = threading.Lock()
        self._run_chat_prefix = f"agentic-pr-review-{uuid.uuid4().hex}"
        self._chat_lock = threading.Lock()
        self._chat_session_index = 0
        self._chat_lanes: dict[str, dict[str, Any]] = {}

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
        deep_concurrency: int = 1,
    ) -> dict[str, Any]:
        chunks = list(((review_input.get("deep_review") or {}).get("chunks")) or [])
        if max_chunks > 0:
            chunks = chunks[:max_chunks]
        if not chunks:
            self.progress("No deep-review chunks found; starting a single model review.")
            return self.review(review_input, deterministic_findings)

        original_chunk_count = len(chunks)
        chunks, skipped_model_chunks = plan_circuit_review_chunks(chunks, deterministic_findings)
        deep_review_meta = review_input.setdefault("deep_review", {})
        deep_review_meta["model_packet_strategy"] = "circuit_focused_packets_skip_low_signal"
        deep_review_meta["model_skipped_chunk_count"] = len(skipped_model_chunks)
        deep_review_meta["model_skipped_chunks"] = skipped_model_chunks[:100]
        notes = review_input.setdefault("collector_notes", {})
        notes["deep_model_original_chunk_count"] = original_chunk_count
        notes["deep_model_review_chunk_count"] = len(chunks)
        notes["deep_model_skipped_chunk_count"] = len(skipped_model_chunks)
        if skipped_model_chunks:
            self.progress(
                f"Circuit packet planner skipped {len(skipped_model_chunks)} low-signal chunk(s); "
                f"{len(chunks)} chunk(s) remain for model review."
            )
        if not chunks:
            self.progress("No high-signal deep-review chunks remain; completing with deterministic checks only.")
            output = deterministic_only_output(deterministic_findings)
            output["summary"] = "No high-signal chunks required model review; deterministic checks completed."
            output["model_notes"] = "Circuit model review skipped because every deep-review chunk was low-signal."
            output["deep_review"] = {
                "enabled": True,
                "chunk_count": original_chunk_count,
                "reviewed_chunk_count": 0,
                "model_skipped_chunk_count": len(skipped_model_chunks),
                "max_chunks": max_chunks,
                "model_packet_strategy": "circuit_focused_packets_skip_low_signal",
            }
            output["chunk_review_outputs"] = []
            return output

        chunk_outputs: list[dict[str, Any]] = []
        total_chunks = len(chunks)
        self.eta = AdaptiveETA(total_requests=total_chunks + 1)
        self.progress(
            f"Deep model review started: {total_chunks} chunk(s) plus synthesis; "
            f"concurrency {normalize_deep_concurrency(deep_concurrency, total_chunks)}; "
            f"estimated whole-review time remaining: {self._eta_text()}."
        )
        chunk_outputs = self._review_deep_chunks(
            review_input,
            deterministic_findings,
            chunks,
            deep_concurrency=deep_concurrency,
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
            "chunk_count": original_chunk_count,
            "reviewed_chunk_count": len(chunk_outputs),
            "model_skipped_chunk_count": len(skipped_model_chunks),
            "max_chunks": max_chunks,
            "model_packet_strategy": "circuit_focused_packets_skip_low_signal",
        }
        final_output["chunk_review_outputs"] = chunk_outputs
        return final_output

    def _review_deep_chunks(
        self,
        review_input: dict[str, Any],
        deterministic_findings: list[dict[str, Any]],
        chunks: list[dict[str, Any]],
        *,
        deep_concurrency: int,
    ) -> list[dict[str, Any]]:
        total_chunks = len(chunks)
        concurrency = normalize_deep_concurrency(deep_concurrency, total_chunks)
        if concurrency <= 1:
            return [
                self._review_single_deep_chunk(review_input, deterministic_findings, chunk, position, total_chunks)
                for position, chunk in enumerate(chunks, start=1)
            ]

        # Warm the OAuth token once so concurrent chunk workers do not all start
        # by attempting the same token exchange.
        self._get_access_token()
        ordered_outputs: list[dict[str, Any] | None] = [None] * total_chunks
        with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="deep-review") as executor:
            futures = {
                executor.submit(
                    self._review_single_deep_chunk,
                    review_input,
                    deterministic_findings,
                    chunk,
                    position,
                    total_chunks,
                ): position
                for position, chunk in enumerate(chunks, start=1)
            }
            try:
                for future in as_completed(futures):
                    position = futures[future]
                    ordered_outputs[position - 1] = future.result()
            except Exception:
                for future in futures:
                    future.cancel()
                raise

        return [output for output in ordered_outputs if output is not None]

    def _review_single_deep_chunk(
        self,
        review_input: dict[str, Any],
        deterministic_findings: list[dict[str, Any]],
        chunk: dict[str, Any],
        position: int,
        total_chunks: int,
    ) -> dict[str, Any]:
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
            max_chars=ADAPTIVE_SUBCHUNK_MODEL_INPUT_CHARS if chunk.get("adaptive_retry") else CHUNK_MODEL_INPUT_CHARS,
        )
        try:
            output = self._invoke_review(chunk_prompt, chunk_deterministic, request_max_attempts=1)
        except GatewayTransientModelError as exc:
            output = self._review_chunk_as_adaptive_subchunks(
                review_input,
                deterministic_findings,
                chunk,
                parent_position=position,
                parent_total=total_chunks,
                original_error=exc,
            )
        output["chunk_id"] = chunk.get("id")
        output["chunk_path"] = chunk.get("path")
        output["chunk_index"] = chunk.get("chunk_index")
        output["chunk_total"] = chunk.get("chunk_total")
        chunk_elapsed = format_duration(time.monotonic() - chunk_started)
        finding_count = len(output.get("findings") or [])
        self.progress(
            f"Deep-review chunk {position}/{total_chunks} completed in {chunk_elapsed} "
            f"with {finding_count} finding(s); estimated whole-review time remaining: {self._eta_text()}."
        )
        return output

    def _review_chunk_as_adaptive_subchunks(
        self,
        review_input: dict[str, Any],
        deterministic_findings: list[dict[str, Any]],
        chunk: dict[str, Any],
        *,
        parent_position: int,
        parent_total: int,
        original_error: GatewayTransientModelError,
    ) -> dict[str, Any]:
        subchunks = split_chunk_for_adaptive_retry(chunk)
        if not subchunks:
            raise original_error
        path = str(chunk.get("path") or "unknown file")
        self.progress(
            f"Deep-review chunk {parent_position}/{parent_total} hit a transient gateway failure; "
            f"retrying {path} as {len(subchunks)} smaller focused subchunk(s)."
        )
        sub_outputs: list[dict[str, Any]] = []
        for sub_position, subchunk in enumerate(subchunks, start=1):
            self.progress(
                f"Adaptive subchunk {sub_position}/{len(subchunks)} started for {path}; "
                f"estimated whole-review time remaining: {self._eta_text()}."
            )
            sub_input = build_chunk_review_input(review_input, subchunk)
            sub_deterministic = filter_findings_for_chunk(deterministic_findings, subchunk)
            sub_prompt = build_user_prompt(
                sub_input,
                sub_deterministic,
                max_chars=ADAPTIVE_SUBCHUNK_MODEL_INPUT_CHARS,
            )
            sub_output = self._invoke_review(sub_prompt, sub_deterministic, request_max_attempts=1)
            sub_output["chunk_id"] = subchunk.get("id")
            sub_output["parent_chunk_id"] = subchunk.get("parent_chunk_id")
            sub_output["chunk_path"] = subchunk.get("path")
            sub_output["chunk_index"] = subchunk.get("chunk_index")
            sub_output["chunk_total"] = subchunk.get("chunk_total")
            sub_outputs.append(sub_output)
            self.progress(
                f"Adaptive subchunk {sub_position}/{len(subchunks)} completed for {path} "
                f"with {len(sub_output.get('findings') or [])} finding(s)."
            )
        return merge_adaptive_subchunk_outputs(
            chunk,
            sub_outputs,
            filter_findings_for_chunk(deterministic_findings, chunk),
            original_error=original_error,
        )

    def _invoke_review(
        self,
        user_prompt: str,
        deterministic_findings: list[dict[str, Any]],
        *,
        request_max_attempts: int | None = None,
    ) -> dict[str, Any]:
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
                response_payload = self._post_model_json(body, max_attempts=request_max_attempts)
                request_succeeded = True
            except (HTTPError, URLError, TimeoutError, OSError) as exc:
                raise GatewayTransientModelError(
                    f"Gateway model invocation failed: {redact_text(format_http_error(exc))}"
                ) from exc
        finally:
            heartbeat_stop.set()
            if heartbeat_thread is not None:
                heartbeat_thread.join(timeout=1)
            if request_succeeded:
                self._complete_eta_request(time.monotonic() - request_started)

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
        chat_id = self._next_chat_id()
        body = {
            "messages": [
                {"role": "user", "content": build_gateway_user_message(safe_user_prompt)},
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            # CIRCUIT validates this exact JSON string metadata field.
            # It is auth metadata, not part of the chat prompt.
            "user": json.dumps(
                {
                    "appkey": self.config.gateway_app_key,
                    "chat_id": chat_id,
                    "session_id": chat_id,
                    "conversation_id": chat_id,
                },
                separators=(",", ":"),
            ),
        }
        if should_include_model_in_body(str(self.config.gateway_base_url or "")):
            body["model"] = self.config.gateway_model
        response_format = self._response_format()
        if response_format is not None:
            body["response_format"] = response_format
        return body

    def _next_chat_id(self) -> str:
        lane = current_chat_lane()
        with self._chat_lock:
            state = self._chat_lanes.get(lane)
            if state is None or int(state.get("transaction_count") or 0) >= CIRCUIT_CHAT_TRANSACTION_LIMIT:
                self._chat_session_index += 1
                state = {
                    "chat_id": f"{self._run_chat_prefix}-{lane}-chat-{self._chat_session_index}",
                    "transaction_count": 0,
                }
                self._chat_lanes[lane] = state
            state["transaction_count"] = int(state.get("transaction_count") or 0) + 1
            return str(state["chat_id"])

    def _response_format(self) -> dict[str, Any] | None:
        with self._response_format_lock:
            mode = self._response_format_mode
        if mode == "json_schema":
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": "agentic_pr_review_output",
                    "strict": True,
                    "schema": REVIEW_RESPONSE_SCHEMA,
                },
            }
        if mode == "json_object":
            return {"type": "json_object"}
        return None

    def _post_model_json(self, body: dict[str, Any], *, max_attempts: int | None = None) -> dict[str, Any]:
        max_attempts = max(1, int(max_attempts or self.config.gateway_request_max_attempts))
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
                with self._token_lock:
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

        rejected_type = (body.get("response_format") or {}).get("type")
        next_body = deepcopy(body)
        if rejected_type == "json_schema":
            with self._response_format_lock:
                if self._response_format_mode is None:
                    next_body.pop("response_format", None)
                    self.progress(
                        "Gateway rejected JSON schema response_format; retrying without provider response_format."
                    )
                    return next_body
                if self._response_format_mode != "json_object":
                    self._response_format_mode = "json_object"
            next_body["response_format"] = self._response_format()
            self.progress("Gateway rejected JSON schema response_format; retrying with JSON object mode.")
            return next_body
        if rejected_type == "json_object":
            with self._response_format_lock:
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
        with self._token_lock:
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
                raise GatewayReviewError(
                    f"Gateway token response did not include access_token/token/id_token. Keys: {keys}"
                )
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
        eta_lock = getattr(self, "_eta_lock", None)
        if eta_lock is None:
            if self.eta is None:
                return "calculating"
            remaining = self.eta.remaining_seconds(current_request_elapsed=current_request_elapsed)
            return f"~{format_duration(remaining)}"
        with eta_lock:
            if self.eta is None:
                return "calculating"
            remaining = self.eta.remaining_seconds(current_request_elapsed=current_request_elapsed)
        return f"~{format_duration(remaining)}"

    def _complete_eta_request(self, duration_seconds: float) -> None:
        eta_lock = getattr(self, "_eta_lock", None)
        if eta_lock is None:
            if self.eta is not None:
                self.eta.complete_request(duration_seconds)
            return
        with eta_lock:
            if self.eta is not None:
                self.eta.complete_request(duration_seconds)


def should_include_model_in_body(endpoint: str) -> bool:
    return "/deployments/" not in endpoint


def build_gateway_user_message(user_prompt: str) -> str:
    return (
        "System review instructions:\n"
        f"{SYSTEM_PROMPT}\n\n"
        "PR review request:\n"
        f"{user_prompt}"
    )


def current_chat_lane() -> str:
    name = threading.current_thread().name or "main"
    if name == "MainThread":
        name = "main"
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip("-")
    return sanitized or "main"


def normalize_deep_concurrency(value: int, total_chunks: int) -> int:
    if total_chunks <= 1:
        return 1
    return min(max(1, int(value or 1)), total_chunks)


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


def merge_adaptive_subchunk_outputs(
    parent_chunk: dict[str, Any],
    sub_outputs: list[dict[str, Any]],
    deterministic_findings: list[dict[str, Any]],
    *,
    original_error: GatewayTransientModelError,
) -> dict[str, Any]:
    findings = []
    notes = []
    for output in sub_outputs:
        findings.extend(output.get("findings") or [])
        note = str(output.get("model_notes") or "").strip()
        if note:
            notes.append(note)
    findings = dedupe_findings(findings)
    path = str(parent_chunk.get("path") or "chunk")
    status = "needs_review" if findings else "looks_good"
    model_notes = (
        f"Reviewed `{path}` as {len(sub_outputs)} smaller focused subchunks after a transient gateway failure "
        f"on the original chunk: {redact_text(str(original_error))}"
    )
    if notes:
        model_notes = f"{model_notes}\n" + "\n".join(notes[:10])
    return normalize_review_output(
        {
            "summary": f"Reviewed {path} through adaptive subchunks after a transient gateway failure.",
            "overall_status": status,
            "safe_to_publish": True,
            "findings": findings,
            "model_notes": model_notes,
        },
        deterministic_findings=deterministic_findings,
    )
