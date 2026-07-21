"""Chat-completions gateway reviewer with OAuth client-credentials auth."""

from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import random
import threading
import time
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .config import RuntimeConfig
from .deep_review import (
    build_chunk_review_input,
    CONTEXT_AWARE_REVIEW_STRATEGY,
    dedupe_findings,
    filter_findings_for_chunk,
    plan_circuit_review_chunks,
    REVIEW_POLICY_VERSION,
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
GATEWAY_TRANSIENT_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}
CHUNK_MODEL_INPUT_CHARS = 70_000
ADAPTIVE_SUBCHUNK_MODEL_INPUT_CHARS = 45_000
PROACTIVE_SUBCHUNK_PROMPT_CHARS = 52_000
PROACTIVE_SUBCHUNK_DIFF_CHARS = 6_000
PROACTIVE_SUBCHUNK_CONTEXT_CHARS = 6_000
PROACTIVE_SUBCHUNK_DETERMINISTIC_FINDINGS = 8
PROACTIVE_SUBCHUNK_ESTIMATED_SECONDS = 90

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
                            "introduced_bug",
                            "introduced_regression",
                            "exposed_existing_bug",
                            "security_issue",
                            "pre_existing_issue",
                            "design_observation",
                            "maintainability_suggestion",
                            "repository_policy_suggestion",
                            "release_management_suggestion",
                            "insufficient_evidence",
                        ],
                    },
                    "review_area": {
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
                    "causality": {
                        "type": "string",
                        "enum": [
                            "introduced_by_pr",
                            "worsened_by_pr",
                            "exposed_by_pr",
                            "pre_existing_unrelated",
                            "unknown",
                        ],
                    },
                    "severity": {"type": "string", "enum": ["critical", "high", "medium", "low", "info"]},
                    "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                    "confidence_score": {"type": "number", "minimum": 0, "maximum": 1},
                    "merge_blocking": {"type": "boolean"},
                    "publication_destination": {
                        "type": "string",
                        "enum": [
                            "inline_blocking",
                            "inline_non_blocking",
                            "summary_high_priority",
                            "summary_observation",
                            "artifact_only",
                            "suppress",
                        ],
                    },
                    "file": {"type": ["string", "null"]},
                    "line": {"type": ["integer", "null"]},
                    "line_start": {"type": ["integer", "null"]},
                    "line_end": {"type": ["integer", "null"]},
                    "code_reference": {"type": ["string", "null"]},
                    "evidence": {"type": "string"},
                    "changed_line_evidence": {"type": ["string", "null"]},
                    "execution_path": {"type": ["string", "null"]},
                    "trigger": {"type": ["string", "null"]},
                    "observable_failure": {"type": ["string", "null"]},
                    "root_cause": {"type": ["string", "null"]},
                    "repository_rule": {"type": ["string", "null"]},
                    "why_it_matters": {"type": "string"},
                    "suggested_fix": {"type": "string"},
                    "suggested_code": {"type": ["string", "null"]},
                    "source": {"type": "string"},
                },
                "required": [
                    "id",
                    "title",
                    "category",
                    "review_area",
                    "causality",
                    "severity",
                    "confidence",
                    "confidence_score",
                    "merge_blocking",
                    "publication_destination",
                    "file",
                    "line",
                    "line_start",
                    "line_end",
                    "code_reference",
                    "evidence",
                    "changed_line_evidence",
                    "execution_path",
                    "trigger",
                    "observable_failure",
                    "root_cause",
                    "repository_rule",
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
        self._metrics_lock = threading.Lock()
        self._completed_model_prompt_chars: list[int] = []
        self._model_metrics: dict[str, Any] = {
            "model_http_attempts": 0,
            "successful_model_calls": 0,
            "transient_retries": 0,
            "non_json_responses": 0,
            "json_repair_model_calls": 0,
            "strict_json_retry_calls": 0,
            "new_chat_retries": 0,
        }

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
        checkpoint_path: str | Path | None = None,
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
        deep_review_meta["model_packet_strategy"] = CONTEXT_AWARE_REVIEW_STRATEGY
        deep_review_meta["model_skipped_chunk_count"] = len(skipped_model_chunks)
        deep_review_meta["model_skipped_chunks"] = skipped_model_chunks[:100]
        notes = review_input.setdefault("collector_notes", {})
        notes["deep_model_original_chunk_count"] = original_chunk_count
        notes["deep_model_review_chunk_count"] = len(chunks)
        notes["deep_model_skipped_chunk_count"] = len(skipped_model_chunks)
        planning = (review_input.get("deep_review") or {}).get("packet_planning") or {}
        excluded = (review_input.get("deep_review") or {}).get("model_excluded_paths") or []
        if excluded:
            self.progress(
                f"AI packet planner excluded {len(excluded)} path(s) from model review: "
                + ", ".join(str(item.get("path") or "") for item in excluded[:5])
            )
        if planning:
            self.progress(
                "AI packet planner dedup/merge summary: "
                f"{planning.get('pre_dedupe_chunk_count', len(chunks))} initial, "
                f"{planning.get('post_dedupe_chunk_count', len(chunks))} after dedupe, "
                f"{planning.get('post_merge_chunk_count', len(chunks))} after merge; "
                f"{planning.get('merged_packet_count', 0)} merged packet(s)."
            )
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
                "model_packet_strategy": CONTEXT_AWARE_REVIEW_STRATEGY,
            }
            output["chunk_review_outputs"] = []
            return output

        total_chunks = len(chunks)
        checkpoint = build_deep_review_checkpoint(
            checkpoint_path,
            review_input=review_input,
            chunks=chunks,
            model=self.config.gateway_model,
            progress=self.progress,
        )
        restored_outputs = checkpoint.load_outputs() if checkpoint is not None else {}
        if restored_outputs:
            planned_ids = {str(chunk.get("id") or "") for chunk in chunks}
            restored_count = sum(1 for chunk_id in restored_outputs if chunk_id in planned_ids)
            restored_subchunk_count = sum(1 for chunk_id in restored_outputs if ":retry-" in chunk_id)
            remaining_count = max(0, total_chunks - restored_count)
            self.progress(
                f"Restored deep review checkpoint with {restored_count}/{total_chunks} completed packet(s); "
                f"{restored_subchunk_count} adaptive subchunk(s) also cached; {remaining_count} packet(s) remain."
            )
        remaining_chunks = [chunk for chunk in chunks if str(chunk.get("id") or "") not in restored_outputs]
        self.eta = AdaptiveETA(total_requests=len(remaining_chunks) + 1)
        selected_concurrency = normalize_deep_concurrency(deep_concurrency, total_chunks, chunks)
        self.progress(
            f"Deep model review started: {total_chunks} chunk(s) plus synthesis "
            f"({len(restored_outputs)} restored, {len(remaining_chunks)} pending); "
            f"concurrency {selected_concurrency}; "
            f"estimated whole-review time remaining: {self._eta_text()}."
        )
        output_by_chunk_id: dict[str, dict[str, Any]] = dict(restored_outputs)
        checkpoint_lock = threading.Lock()

        def save_checkpoint_for_output(output: dict[str, Any]) -> None:
            chunk_id = str(output.get("chunk_id") or "")
            if not chunk_id:
                return
            output_by_chunk_id[chunk_id] = output
            if checkpoint is None:
                return
            with checkpoint_lock:
                checkpoint.save(output_by_chunk_id, chunks, status="in_progress")

        restored_subchunk_outputs = {
            chunk_id: output
            for chunk_id, output in output_by_chunk_id.items()
            if ":retry-" in chunk_id
        }
        if remaining_chunks:
            self._review_deep_chunks(
                review_input,
                deterministic_findings,
                remaining_chunks,
                deep_concurrency=selected_concurrency,
                total_planned_chunks=total_chunks,
                on_chunk_output=save_checkpoint_for_output,
                restored_subchunk_outputs=restored_subchunk_outputs,
                on_subchunk_output=save_checkpoint_for_output,
            )
        elif checkpoint is not None:
            checkpoint.save(output_by_chunk_id, chunks, status="chunks_complete")

        chunk_outputs = [
            output_by_chunk_id[str(chunk.get("id") or "")]
            for chunk in chunks
            if str(chunk.get("id") or "") in output_by_chunk_id
        ]
        if checkpoint is not None:
            checkpoint.save(output_by_chunk_id, chunks, status="chunks_complete")

        synthesis_started = time.monotonic()
        self.progress(
            f"Synthesis started for {len(chunk_outputs)} reviewed chunk(s); "
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
            "model_packet_strategy": CONTEXT_AWARE_REVIEW_STRATEGY,
            "model_metrics": self._model_metrics_snapshot(),
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
        total_planned_chunks: int | None = None,
        on_chunk_output: Callable[[dict[str, Any]], None] | None = None,
        restored_subchunk_outputs: dict[str, dict[str, Any]] | None = None,
        on_subchunk_output: Callable[[dict[str, Any]], None] | None = None,
    ) -> list[dict[str, Any]]:
        total_chunks = len(chunks)
        concurrency = normalize_deep_concurrency(deep_concurrency, total_chunks, chunks)
        if concurrency <= 1:
            outputs = []
            for position, chunk in enumerate(chunks, start=1):
                output = self._review_single_deep_chunk(
                    review_input,
                    deterministic_findings,
                    chunk,
                    position,
                    total_planned_chunks or total_chunks,
                    restored_subchunk_outputs=restored_subchunk_outputs,
                    on_subchunk_output=on_subchunk_output,
                )
                outputs.append(output)
                if on_chunk_output is not None:
                    on_chunk_output(output)
            return outputs

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
                    total_planned_chunks or total_chunks,
                    restored_subchunk_outputs=restored_subchunk_outputs,
                    on_subchunk_output=on_subchunk_output,
                ): position
                for position, chunk in enumerate(chunks, start=1)
            }
            try:
                for future in as_completed(futures):
                    position = futures[future]
                    output = future.result()
                    ordered_outputs[position - 1] = output
                    if on_chunk_output is not None:
                        on_chunk_output(output)
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
        *,
        restored_subchunk_outputs: dict[str, dict[str, Any]] | None = None,
        on_subchunk_output: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        path = str(chunk.get("path") or "unknown file")
        file_chunk = chunk.get("chunk_index")
        file_total = chunk.get("chunk_total")
        chunk_started = time.monotonic()
        creation_reason = str(chunk.get("packet_creation_reason") or chunk.get("model_review_reason") or "changed code")
        merged_count = int(chunk.get("merged_chunk_count") or 1)
        self.progress(
            f"Deep-review chunk {position}/{total_chunks} started: "
            f"{path} (file chunk {file_chunk}/{file_total}; strategy {chunk.get('chunk_strategy') or 'unknown'}; "
            f"merged logical chunks {merged_count}; reason: {creation_reason}); "
            f"estimated whole-review time remaining: {self._eta_text()}."
        )
        chunk_input = build_chunk_review_input(review_input, chunk)
        chunk_deterministic = filter_findings_for_chunk(deterministic_findings, chunk)
        chunk_prompt = build_user_prompt(
            chunk_input,
            chunk_deterministic,
            max_chars=ADAPTIVE_SUBCHUNK_MODEL_INPUT_CHARS if chunk.get("adaptive_retry") else CHUNK_MODEL_INPUT_CHARS,
        )
        estimated_request_seconds = self._estimated_request_seconds_for_prompt(len(chunk_prompt))
        can_split_for_adaptive_retry = can_adaptively_split_chunk(
            chunk,
            max_chars=PROACTIVE_SUBCHUNK_DIFF_CHARS,
        )
        if should_proactively_subchunk(
            chunk,
            chunk_prompt,
            chunk_deterministic,
            estimated_request_seconds=estimated_request_seconds,
        ) and can_split_for_adaptive_retry:
            output = self._review_chunk_as_adaptive_subchunks(
                review_input,
                deterministic_findings,
                chunk,
                parent_position=position,
                parent_total=total_chunks,
                original_error=GatewayTransientModelError(
                    "Proactively split timeout-risk CIRCUIT packet before first model request."
                ),
                proactive=True,
                restored_subchunk_outputs=restored_subchunk_outputs,
                on_subchunk_output=on_subchunk_output,
            )
            output = dict(output)
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
        try:
            output = self._invoke_review(
                chunk_prompt,
                chunk_deterministic,
                request_max_attempts=1 if can_split_for_adaptive_retry else None,
            )
        except GatewayTransientModelError as exc:
            if can_adaptively_split_chunk(chunk, max_chars=18_000):
                output = self._review_chunk_as_adaptive_subchunks(
                    review_input,
                    deterministic_findings,
                    chunk,
                    parent_position=position,
                    parent_total=total_chunks,
                    original_error=exc,
                    restored_subchunk_outputs=restored_subchunk_outputs,
                    on_subchunk_output=on_subchunk_output,
                )
            else:
                self.progress(
                    f"Deep-review chunk {position}/{total_chunks} for {path} could not be split into multiple "
                    "meaningful adaptive subchunks; retaining deterministic findings for this packet."
                )
                output = fallback_chunk_transient_output(chunk, chunk_deterministic, exc)
        except GatewayModelFormatError as exc:
            if can_adaptively_split_chunk(chunk, max_chars=18_000):
                output = self._review_chunk_as_adaptive_subchunks(
                    review_input,
                    deterministic_findings,
                    chunk,
                    parent_position=position,
                    parent_total=total_chunks,
                    original_error=GatewayTransientModelError(str(exc)),
                    restored_subchunk_outputs=restored_subchunk_outputs,
                    on_subchunk_output=on_subchunk_output,
                )
            else:
                self.progress(
                    f"Deep-review chunk {position}/{total_chunks} for {path} returned malformed JSON after "
                    "recovery and could not be split; retaining deterministic findings for this packet."
                )
                output = fallback_chunk_format_output(chunk, chunk_deterministic, exc)
        output = dict(output)
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
        proactive: bool = False,
        restored_subchunk_outputs: dict[str, dict[str, Any]] | None = None,
        on_subchunk_output: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        subchunks = split_chunk_for_adaptive_retry(
            chunk,
            max_chars=PROACTIVE_SUBCHUNK_DIFF_CHARS if proactive else 18_000,
            context_char_limit=PROACTIVE_SUBCHUNK_CONTEXT_CHARS if proactive else 30_000,
        )
        if len(subchunks) <= 1:
            raise original_error
        self._add_eta_requests(max(0, len(subchunks) - 1))
        if proactive:
            for subchunk in subchunks:
                subchunk["proactive_review"] = True
                reason = str(subchunk.get("model_review_reason") or chunk.get("model_review_reason") or "")
                if "proactive" not in reason.lower():
                    subchunk["model_review_reason"] = (
                        f"{reason}; proactive smaller packet for timeout-risk review"
                        if reason
                        else "proactive smaller packet for timeout-risk review"
                    )
        path = str(chunk.get("path") or "unknown file")
        if proactive:
            self.progress(
                f"Deep-review chunk {parent_position}/{parent_total} is timeout-risk; proactively reviewing "
                f"{path} as {len(subchunks)} smaller focused subchunk(s)."
            )
        else:
            self.progress(
                f"Deep-review chunk {parent_position}/{parent_total} hit a transient gateway failure; "
                f"retrying {path} as {len(subchunks)} smaller focused subchunk(s)."
            )
        sub_outputs: list[dict[str, Any]] = []
        for sub_position, subchunk in enumerate(subchunks, start=1):
            subchunk_id = str(subchunk.get("id") or "")
            if restored_subchunk_outputs and subchunk_id in restored_subchunk_outputs:
                sub_output = dict(restored_subchunk_outputs[subchunk_id])
                sub_outputs.append(sub_output)
                self.progress(
                    f"Adaptive subchunk {sub_position}/{len(subchunks)} restored for {path} "
                    f"with {len(sub_output.get('findings') or [])} finding(s)."
                )
                continue
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
            try:
                sub_output = self._invoke_review(sub_prompt, sub_deterministic, request_max_attempts=1)
            except GatewayTransientModelError as exc:
                self.progress(
                    f"Adaptive subchunk {sub_position}/{len(subchunks)} for {path} still hit a transient "
                    "gateway failure; retaining deterministic findings for this slice."
                )
                sub_output = fallback_chunk_transient_output(subchunk, sub_deterministic, exc)
            sub_output = dict(sub_output)
            sub_output["chunk_id"] = subchunk.get("id")
            sub_output["parent_chunk_id"] = subchunk.get("parent_chunk_id")
            sub_output["chunk_path"] = subchunk.get("path")
            sub_output["chunk_index"] = subchunk.get("chunk_index")
            sub_output["chunk_total"] = subchunk.get("chunk_total")
            sub_outputs.append(sub_output)
            if on_subchunk_output is not None:
                on_subchunk_output(sub_output)
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
            response_payload = self._post_model_json(body, max_attempts=request_max_attempts)
            request_succeeded = True
            self._increment_model_metric("successful_model_calls")
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            raise GatewayTransientModelError(
                f"Gateway model invocation failed: {redact_text(format_http_error(exc))}"
            ) from exc
        finally:
            heartbeat_stop.set()
            if heartbeat_thread is not None:
                heartbeat_thread.join(timeout=1)
            if request_succeeded:
                self._complete_eta_request(time.monotonic() - request_started, prompt_chars=len(user_prompt))

        text = extract_chat_completion_text(response_payload)
        recovery_note = ""
        try:
            raw_output = extract_json_object(text)
        except Exception as exc:  # noqa: BLE001 - keep raw model text for prototype debugging
            self._increment_model_metric("non_json_responses")
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
        if circuit_requests_new_chat(previous_text):
            self.progress("Gateway requested a new chat; retrying the original review as a fresh request.")
            try:
                self._increment_model_metric("new_chat_retries")
                fresh_payload = self._post_model_json(
                    self._model_body(build_fresh_chat_retry_prompt(original_user_prompt))
                )
                fresh_text = extract_chat_completion_text(fresh_payload)
                raw_output = extract_json_object(fresh_text)
                return raw_output, fresh_payload, "Recovered from a CIRCUIT new-chat prompt with a fresh retry."
            except Exception:
                self.progress("Gateway new-chat retry failed; attempting normal JSON recovery.")

        self.progress("Gateway model returned non-JSON; attempting one JSON repair request.")
        try:
            self._increment_model_metric("json_repair_model_calls")
            repair_payload = self._post_model_json(self._model_body(build_json_repair_prompt(previous_text)))
            repair_text = extract_chat_completion_text(repair_payload)
            raw_output = extract_json_object(repair_text)
            return raw_output, repair_payload, "Recovered from a non-JSON model response with one JSON repair retry."
        except Exception:
            self.progress("Gateway JSON repair failed; retrying the original review with strict JSON-only instructions.")

        try:
            self._increment_model_metric("strict_json_retry_calls")
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
            "stop": ["<|im_end|>"],
            # CIRCUIT validates this exact JSON string metadata field.
            # It is auth metadata, not part of the chat prompt.
            "user": json.dumps({"appkey": self.config.gateway_app_key}, separators=(",", ":")),
        }
        if should_include_model_in_body(str(self.config.gateway_base_url or "")):
            body["model"] = self.config.gateway_model
        response_format = self._response_format()
        if response_format is not None:
            body["response_format"] = response_format
        return body

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
                    if exc.code in GATEWAY_TRANSIENT_STATUS_CODES and attempt < max_attempts:
                        delay = retry_after_delay_seconds(exc)
                        if delay is None:
                            delay = transient_retry_delay_seconds(
                                base=self.config.gateway_request_retry_backoff_seconds,
                                attempt=attempt,
                            )
                        self.progress(
                            f"Gateway model request failed transiently on attempt {attempt}/{max_attempts}: "
                            f"{redact_text(format_http_error(exc))}. Retrying in {format_duration(delay)}."
                        )
                        self._increment_model_metric("transient_retries")
                        if delay > 0:
                            time.sleep(delay)
                        attempt += 1
                        continue
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
                delay = transient_retry_delay_seconds(
                    base=self.config.gateway_request_retry_backoff_seconds,
                    attempt=attempt,
                )
                self.progress(
                    f"Gateway model request failed transiently on attempt {attempt}/{max_attempts}: "
                    f"{redact_text(format_http_error(exc))}. Retrying in {format_duration(delay)}."
                )
                self._increment_model_metric("transient_retries")
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
        self._increment_model_metric("model_http_attempts")
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

    def _complete_eta_request(self, duration_seconds: float, *, prompt_chars: int | None = None) -> None:
        eta_lock = getattr(self, "_eta_lock", None)
        if eta_lock is None:
            if self.eta is not None:
                self.eta.complete_request(duration_seconds)
            if prompt_chars is not None:
                self._completed_model_prompt_chars.append(max(1, int(prompt_chars)))
            return
        with eta_lock:
            if self.eta is not None:
                self.eta.complete_request(duration_seconds)
            if prompt_chars is not None:
                self._completed_model_prompt_chars.append(max(1, int(prompt_chars)))

    def _add_eta_requests(self, count: int) -> None:
        eta_lock = getattr(self, "_eta_lock", None)
        if eta_lock is None:
            if self.eta is not None:
                self.eta.add_requests(count)
            return
        with eta_lock:
            if self.eta is not None:
                self.eta.add_requests(count)

    def _estimated_request_seconds_for_prompt(self, prompt_chars: int) -> float | None:
        eta_lock = getattr(self, "_eta_lock", None)
        if eta_lock is None:
            if self.eta is None or not self.eta.completed_request_seconds or not self._completed_model_prompt_chars:
                return None
            avg_seconds = self.eta.estimated_request_seconds
            avg_chars = sum(self._completed_model_prompt_chars) / len(self._completed_model_prompt_chars)
        else:
            with eta_lock:
                if self.eta is None or not self.eta.completed_request_seconds or not self._completed_model_prompt_chars:
                    return None
                avg_seconds = self.eta.estimated_request_seconds
                avg_chars = sum(self._completed_model_prompt_chars) / len(self._completed_model_prompt_chars)
        if avg_chars <= 0:
            return None
        scale = max(0.5, min(4.0, max(1, int(prompt_chars)) / avg_chars))
        return avg_seconds * scale

    def _increment_model_metric(self, key: str, count: int = 1) -> None:
        with self._metrics_lock:
            self._model_metrics[key] = int(self._model_metrics.get(key) or 0) + count

    def _model_metrics_snapshot(self) -> dict[str, Any]:
        with self._metrics_lock:
            metrics = dict(self._model_metrics)
        eta_lock = getattr(self, "_eta_lock", None)
        if eta_lock is None:
            latencies = list(self.eta.completed_request_seconds if self.eta is not None else [])
            prompt_chars = list(self._completed_model_prompt_chars)
        else:
            with eta_lock:
                latencies = list(self.eta.completed_request_seconds if self.eta is not None else [])
                prompt_chars = list(self._completed_model_prompt_chars)
        metrics["latency_seconds"] = summarize_numeric_samples(latencies)
        metrics["prompt_chars"] = summarize_numeric_samples(prompt_chars)
        return metrics


class DeepReviewCheckpoint:
    def __init__(
        self,
        path: Path,
        *,
        checkpoint_id: str,
        progress: Callable[[str], None],
    ) -> None:
        self.path = path
        self.checkpoint_id = checkpoint_id
        self.progress = progress

    def load_outputs(self) -> dict[str, dict[str, Any]]:
        if not self.path.exists():
            self.progress(f"No deep review checkpoint found at {self.path}; starting fresh.")
            return {}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            self.progress(f"Could not read deep review checkpoint at {self.path}: {redact_text(str(exc))}. Starting fresh.")
            return {}
        if payload.get("checkpoint_id") != self.checkpoint_id:
            self.progress("Deep review checkpoint exists but does not match this PR/head/model; ignoring stale checkpoint.")
            return {}
        outputs: dict[str, dict[str, Any]] = {}
        for item in payload.get("chunk_outputs") or []:
            if not isinstance(item, dict):
                continue
            chunk_id = str(item.get("chunk_id") or "")
            if chunk_id:
                outputs[chunk_id] = item
        self.progress(f"Deep review checkpoint restored from {self.path} with {len(outputs)} completed packet(s).")
        return outputs

    def save(
        self,
        output_by_chunk_id: dict[str, dict[str, Any]],
        chunks: list[dict[str, Any]],
        *,
        status: str,
    ) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        now = int(time.time())
        payload = {
            "schema_version": "0.1",
            "checkpoint_id": self.checkpoint_id,
            "status": status,
            "updated_at_epoch": now,
            "total_chunk_count": len(chunks),
            "completed_chunk_count": len(output_by_chunk_id),
            "planned_chunk_ids": [chunk.get("id") for chunk in chunks],
            "completed_chunk_ids": list(output_by_chunk_id.keys()),
            "chunk_outputs": [
                output_by_chunk_id[str(chunk.get("id") or "")]
                for chunk in chunks
                if str(chunk.get("id") or "") in output_by_chunk_id
            ],
        }
        temp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        temp_path.write_text(json.dumps(redact_obj(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temp_path.replace(self.path)
        self.progress(
            f"Deep review checkpoint saved at {self.path}: "
            f"{len(output_by_chunk_id)}/{len(chunks)} packet(s) complete."
        )


def build_deep_review_checkpoint(
    checkpoint_path: str | Path | None,
    *,
    review_input: dict[str, Any],
    chunks: list[dict[str, Any]],
    model: str,
    progress: Callable[[str], None],
) -> DeepReviewCheckpoint | None:
    if checkpoint_path is None:
        return None
    return DeepReviewCheckpoint(
        Path(checkpoint_path),
        checkpoint_id=deep_review_checkpoint_id(review_input, chunks, model=model),
        progress=progress,
    )


def deep_review_checkpoint_id(review_input: dict[str, Any], chunks: list[dict[str, Any]], *, model: str) -> str:
    pr = review_input.get("pr") or review_input.get("pull_request") or {}
    base = pr.get("base") if isinstance(pr.get("base"), dict) else {}
    head = pr.get("head") if isinstance(pr.get("head"), dict) else {}
    identity = {
        "schema_version": "0.1",
        "review_policy_version": REVIEW_POLICY_VERSION,
        "repo": review_input.get("repo") or (review_input.get("repository") or {}).get("full_name"),
        "pr_number": pr.get("number"),
        "base_sha": base.get("sha"),
        "head_sha": head.get("sha"),
        "model": model,
        "chunks": [
            {
                "id": chunk.get("id"),
                "path": chunk.get("path"),
                "strategy": chunk.get("chunk_strategy"),
                "diff_sha256": hashlib.sha256(str(chunk.get("diff") or "").encode("utf-8")).hexdigest(),
            }
            for chunk in chunks
        ],
    }
    blob = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def should_include_model_in_body(endpoint: str) -> bool:
    return "/deployments/" not in endpoint


def should_proactively_subchunk(
    chunk: dict[str, Any],
    chunk_prompt: str,
    chunk_deterministic: list[dict[str, Any]],
    *,
    estimated_request_seconds: float | None = None,
) -> bool:
    if chunk.get("adaptive_retry"):
        return False
    path = str(chunk.get("path") or "")
    suffix = path.rsplit(".", 1)[-1].lower() if "." in path else ""
    if suffix not in {"py", "json", "toml", "yaml", "yml", "xml", "html", "jinja", "j2"}:
        return False
    if estimated_request_seconds is not None and estimated_request_seconds >= PROACTIVE_SUBCHUNK_ESTIMATED_SECONDS:
        return True
    if len(chunk_prompt) >= PROACTIVE_SUBCHUNK_PROMPT_CHARS:
        return True
    if int(chunk.get("diff_chars") or 0) >= PROACTIVE_SUBCHUNK_DIFF_CHARS * 2:
        return True
    if suffix == "py" and len(chunk_deterministic) >= PROACTIVE_SUBCHUNK_DETERMINISTIC_FINDINGS:
        return True
    changes = int(chunk.get("changes") or 0)
    return suffix == "py" and changes >= 250


def can_adaptively_split_chunk(chunk: dict[str, Any], *, max_chars: int) -> bool:
    return len(split_chunk_for_adaptive_retry(chunk, max_chars=max_chars, context_char_limit=1_000)) > 1


def build_single_focused_retry_chunk(chunk: dict[str, Any], *, proactive: bool = False) -> dict[str, Any]:
    parent_id = str(chunk.get("id") or chunk.get("path") or "chunk")
    retry = dict(chunk)
    reason_suffix = (
        "proactive focused packet for timeout-risk review"
        if proactive
        else "focused retry after transient gateway failure"
    )
    retry.update(
        {
            "id": f"{parent_id}:focused-retry",
            "parent_chunk_id": parent_id,
            "adaptive_retry": True,
            "proactive_review": proactive,
            "chunk_index": 1,
            "chunk_total": 1,
            "context_char_limit": 4_000,
            "model_review_reason": (
                f"{chunk.get('model_review_reason') or 'connector source/config change'}; "
                f"{reason_suffix}"
            ),
        }
    )
    return retry


def circuit_requests_new_chat(text: str) -> bool:
    lowered = text.lower()
    if "chat" not in lowered:
        return False
    new_chat_phrases = (
        "begin a new chat",
        "begin new chat",
        "start a new chat",
        "start new chat",
        "new conversation",
        "start a fresh chat",
        "fresh chat",
    )
    return any(phrase in lowered for phrase in new_chat_phrases)


def normalize_deep_concurrency(
    value: int,
    total_chunks: int,
    chunks: list[dict[str, Any]] | None = None,
) -> int:
    if total_chunks <= 1:
        return 1
    requested = int(value or 0)
    explicit_cap = requested if requested > 0 else 3
    recommended = recommended_deep_concurrency(total_chunks, chunks or [])
    return min(max(1, explicit_cap), recommended, total_chunks)


def recommended_deep_concurrency(total_chunks: int, chunks: list[dict[str, Any]]) -> int:
    diff_chars = [int(chunk.get("diff_chars") or len(str(chunk.get("diff") or ""))) for chunk in chunks]
    max_diff = max(diff_chars) if diff_chars else 0
    avg_diff = (sum(diff_chars) / len(diff_chars)) if diff_chars else 0
    high_priority_count = sum(1 for chunk in chunks if chunk.get("model_review_priority") == "high")
    if max_diff >= 28_000 or avg_diff >= 16_000:
        return 1
    if total_chunks <= 3:
        return 3
    if total_chunks >= 40 or high_priority_count >= 18:
        return 2
    if total_chunks >= 7 or max_diff >= 8_000 or avg_diff >= 5_000 or high_priority_count >= 4:
        return 2
    return 3


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
        '      "category": "introduced_bug|introduced_regression|exposed_existing_bug|security_issue|pre_existing_issue|design_observation|maintainability_suggestion|repository_policy_suggestion|release_management_suggestion|insufficient_evidence",\n'
        '      "review_area": "api_auth_correctness|polling_checkpoint|output_schema_mismatch|unsafe_logging|soar_metadata|docs_pr_accuracy|pagination|validation|missing_tests|precommit|merge_conflict|ci_synthesis|general",\n'
        '      "causality": "introduced_by_pr|worsened_by_pr|exposed_by_pr|pre_existing_unrelated|unknown",\n'
        '      "severity": "critical|high|medium|low|info",\n'
        '      "confidence": "high|medium|low",\n'
        '      "confidence_score": 0.95,\n'
        '      "merge_blocking": true,\n'
        '      "publication_destination": "inline_blocking|inline_non_blocking|summary_high_priority|summary_observation|artifact_only|suppress",\n'
        '      "file": "path or null",\n'
        '      "line": 123,\n'
        '      "line_start": 123,\n'
        '      "line_end": null,\n'
        '      "code_reference": "reference or null",\n'
        '      "evidence": "specific evidence",\n'
        '      "changed_line_evidence": "changed line or null",\n'
        '      "execution_path": "entry point to sink or null",\n'
        '      "trigger": "input/state trigger or null",\n'
        '      "observable_failure": "observed failure or null",\n'
        '      "root_cause": "single root cause",\n'
        '      "repository_rule": "explicit rule or null",\n'
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


def build_fresh_chat_retry_prompt(original_user_prompt: str) -> str:
    return (
        "This is a fresh CIRCUIT chat-completions request with no prior conversation history. "
        "Review the original request below and return exactly one valid JSON object using the required schema. "
        "Do not include Markdown, prose outside JSON, headings, or code fences. Preserve review quality and include "
        "only concrete, high-confidence actionable findings.\n\n"
        "Original review request:\n"
        f"{original_user_prompt}"
    )


def extract_chat_completion_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if isinstance(choices, dict):
        message = choices.get("message")
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            return message["content"]
        if isinstance(choices.get("text"), str):
            return choices["text"]
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


def retry_after_delay_seconds(error: HTTPError) -> float | None:
    value = error.headers.get("Retry-After") if error.headers else None
    if not value:
        return None
    try:
        return max(0.0, min(float(value), 120.0))
    except ValueError:
        return None


def transient_retry_delay_seconds(*, base: float, attempt: int) -> float:
    exponential = max(0.0, float(base)) * (2 ** max(0, attempt - 1))
    jitter = random.uniform(0.0, min(2.0, max(0.25, exponential / 2 or 0.25)))
    return min(120.0, exponential + jitter)


def summarize_numeric_samples(values: list[float] | list[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    ordered = sorted(float(value) for value in values)
    return {
        "count": len(ordered),
        "avg": sum(ordered) / len(ordered),
        "p50": percentile(ordered, 0.50),
        "p95": percentile(ordered, 0.95),
        "max": ordered[-1],
    }


def percentile(ordered_values: list[float], quantile: float) -> float:
    if not ordered_values:
        return 0.0
    index = min(len(ordered_values) - 1, max(0, int(round((len(ordered_values) - 1) * quantile))))
    return ordered_values[index]


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
    chunk_notes = []
    for output in chunk_outputs:
        findings.extend(output.get("findings") or [])
        note = str(output.get("model_notes") or "").strip()
        if note:
            chunk_notes.append(note)
    findings = dedupe_findings(findings)
    notes = [f"Synthesis failed: {redact_text(model_notes)}"]
    notes.extend(chunk_notes[:10])
    return normalize_review_output(
        {
            "summary": "Deep chunk review completed, but synthesis failed; using deduplicated chunk findings.",
            "overall_status": "needs_review" if findings else "looks_good",
            "safe_to_publish": True,
            "findings": findings,
            "model_notes": "\n".join(notes),
        },
        deterministic_findings=deterministic_findings,
    )


def fallback_chunk_transient_output(
    chunk: dict[str, Any],
    deterministic_findings: list[dict[str, Any]],
    error: GatewayTransientModelError,
) -> dict[str, Any]:
    path = str(chunk.get("path") or "chunk")
    return normalize_review_output(
        {
            "summary": f"Model review for {path} hit repeated transient gateway failures.",
            "overall_status": "needs_review" if deterministic_findings else "looks_good",
            "safe_to_publish": True,
            "findings": [],
            "model_notes": (
                f"Model review for `{path}` fell back to deterministic findings after a repeated transient "
                f"gateway failure: {redact_text(str(error))}"
            ),
        },
        deterministic_findings=deterministic_findings,
    )


def fallback_chunk_format_output(
    chunk: dict[str, Any],
    deterministic_findings: list[dict[str, Any]],
    error: GatewayModelFormatError,
) -> dict[str, Any]:
    path = str(chunk.get("path") or "chunk")
    return normalize_review_output(
        {
            "summary": f"Model review for {path} returned malformed JSON after recovery attempts.",
            "overall_status": "needs_review" if deterministic_findings else "looks_good",
            "safe_to_publish": True,
            "findings": [],
            "model_notes": (
                f"Model review for `{path}` fell back to deterministic findings after malformed model output: "
                f"{redact_text(str(error))}"
            ),
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
