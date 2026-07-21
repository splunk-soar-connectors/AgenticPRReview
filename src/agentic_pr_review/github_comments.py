"""Plan and publish concise GitHub comments for review findings."""

from __future__ import annotations

import hashlib
import re
from typing import Any

from .github_client import GitHubClient, GitHubError
from .models import is_actionable_review_finding, should_promote_deterministic_finding
from .secret_redactor import redact_text


CODE_SUFFIXES = {".py", ".json", ".yaml", ".yml", ".toml", ".cfg", ".ini", ".txt"}
TEXT_SUFFIXES = {".md", ".rst"}
TEXT_CATEGORIES = {"docs_pr_accuracy", "precommit", "merge_conflict", "ci_synthesis"}
SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
MARKER_PREFIX = "<!-- agentic-pr-review:"
GENERIC_CI_FIXES = (
    "resolve the pre-commit failure",
    "resolve the precommit failure",
    "fix the pre-commit failure",
    "fix the precommit failure",
    "fix the failing check",
    "fix failing ci",
    "address the failing ci",
    "inspect the failing check logs",
    "inspect ci logs",
)
PRECOMMIT_DETAIL_KEYWORDS = (
    "failed -",
    "hook id:",
    "exit code:",
    "ruff",
    "semgrep",
    "detect-secrets",
    "build-docs",
    "release-notes",
    "check-json",
    "check-yaml",
    "mdformat",
    "app_package_name",
    "valid_app_name_and_guid",
    "appid_to_name",
    "appid_to_package_name",
    "degub_print",
    "debug_print",
    "min platform version",
    "min phantom version",
    "min number log statements",
    "action name",
    "additional logging",
    "verbosity",
    "product name on files",
    "playbook missing",
    "integration test results are missing",
    "app-tests",
    "apps-test-playbooks",
    "pytest",
    "coverage",
    "ci-tools",
    "package-app-dependencies",
    "dependency",
    "dependencies",
    "wheel",
    "wheels",
    "uv.lock",
    "assertionerror",
    "traceback",
)

def build_comment_plan(
    review_output: dict[str, Any],
    review_input: dict[str, Any],
    *,
    max_comments: int | None = None,
) -> dict[str, Any]:
    if review_output.get("safe_to_publish") is False:
        return {
            "schema_version": "0.1",
            "repo": review_input.get("repo"),
            "pr_number": (review_input.get("pr") or {}).get("number"),
            "head_sha": ((review_input.get("pr") or {}).get("head") or {}).get("sha"),
            "comments": [],
            "publish_blocked": True,
            "publish_blocked_reason": "review_output.safe_to_publish is false",
        }

    findings = findings_for_comments(review_output)
    findings = [
        finding
        for finding in findings
        if not should_skip_finding_for_pr_context(finding, review_input)
    ]
    findings = group_related_findings_for_comments(findings)
    findings.sort(key=lambda item: SEVERITY_ORDER.get(str(item.get("severity", "medium")), 2))
    if max_comments is not None and max_comments > 0:
        findings = findings[:max_comments]
    diff_index = build_diff_index(review_input.get("changed_files", []))

    comments = []
    for finding in findings:
        if should_skip_posted_comment(finding):
            continue
        if should_skip_finding_for_pr_context(finding, review_input):
            continue
        comment = build_comment_for_finding(finding, review_input, diff_index)
        if comment:
            comments.append(comment)

    return {
        "schema_version": "0.1",
        "repo": review_input.get("repo"),
        "pr_number": (review_input.get("pr") or {}).get("number"),
        "head_sha": ((review_input.get("pr") or {}).get("head") or {}).get("sha"),
        "comments": comments,
    }


def filter_review_output_for_pr_context(review_output: dict[str, Any], review_input: dict[str, Any]) -> dict[str, Any]:
    original_findings = [finding for finding in review_output.get("findings", []) or [] if isinstance(finding, dict)]
    findings = []
    suppressed: list[dict[str, Any]] = []
    for finding in original_findings:
        reason = finding_context_filter_reason(finding, review_input)
        if reason:
            suppressed.append(
                {
                    "id": finding.get("id"),
                    "title": finding.get("title"),
                    "file": finding.get("file"),
                    "line": finding.get("line"),
                    "reason": reason["reason"],
                    "confidence_before": finding.get("confidence"),
                    "confidence_after": reason.get("confidence_after", "suppressed"),
                    "evidence_source": reason.get("evidence_source"),
                    "reason_confidence_was_lowered": reason.get("detail"),
                }
            )
            continue
        findings.append(finding)
    if len(findings) == len(original_findings):
        return review_output

    output = dict(review_output)
    removed_count = len(original_findings) - len(findings)
    output["findings"] = findings
    if not findings and output.get("overall_status") in {"needs_review", "blocked_by_ci"}:
        output["overall_status"] = "looks_good"
    output["behavioral_verification"] = {
        "suppressed_count": removed_count,
        "suppressed_findings": suppressed[:50],
    }
    reason_counts: dict[str, int] = {}
    for item in suppressed:
        reason_key = str(item.get("reason") or "unknown")
        reason_counts[reason_key] = reason_counts.get(reason_key, 0) + 1
    reason_text = ", ".join(f"{reason}={count}" for reason, count in sorted(reason_counts.items()))
    note = f"Filtered {removed_count} finding(s) during PR-scope/evidence verification"
    if reason_text:
        note += f" ({reason_text})"
    note += "."
    existing_notes = str(output.get("model_notes") or "").strip()
    output["model_notes"] = f"{existing_notes}\n{note}".strip() if existing_notes else note
    return output


def findings_for_comments(review_output: dict[str, Any]) -> list[dict[str, Any]]:
    findings = list(review_output.get("findings") or [])
    existing_keys = {finding_identity(finding) for finding in findings if isinstance(finding, dict)}
    for finding in review_output.get("deterministic_findings") or []:
        if not isinstance(finding, dict):
            continue
        if not should_promote_deterministic_finding(finding):
            continue
        key = finding_identity(finding)
        if key in existing_keys:
            continue
        findings.append(finding)
        existing_keys.add(key)
    return findings


def group_related_findings_for_comments(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    passthrough: list[dict[str, Any]] = []
    for finding in findings:
        if not isinstance(finding, dict):
            continue
        if str(finding.get("severity") or "").lower() == "low":
            continue
        if str(finding.get("confidence") or "").lower() != "high":
            continue
        key = comment_group_key(finding)
        if not key:
            passthrough.append(finding)
            continue
        grouped.setdefault(key, []).append(finding)

    output = passthrough[:]
    for key, items in grouped.items():
        if len(items) == 1:
            output.append(items[0])
        else:
            output.append(merge_comment_group(key, items))
    return output


def comment_group_key(finding: dict[str, Any]) -> str | None:
    text = all_comment_finding_text(finding)
    category = str(finding.get("category") or "")
    title = str(finding.get("title") or "").lower()
    file_path = str(finding.get("file") or "")

    if (
        "verify_server_cert" in text
        or ("tls" in text and "verify" in text)
        or "direct requests call does not set a timeout" in title
        or "requests are missing explicit timeouts" in title
        or ("debug data" in text and ("response text" in text or "headers" in text))
        or "http 429" in text
        or "rate limit" in text
        or "fixed indicator" in text
    ):
        return "network_request_safety"
    if "action_result.data" in text and (
        "no data output" in text
        or "declares no data output" in text
        or "no action_result.data" in text
        or "data.* output" in text
    ):
        return "action_metadata_contract"
    if "indicator action parameters" in text and "contains" in text:
        return "action_metadata_contract"
    if category == "output_schema_mismatch" and "summary" in text and "action_result.data" not in text and ("app json" in text or "output" in text):
        return "action_metadata_contract"
    if (
        "custom view" in text
        and ("actionresult" in text or "action result" in text or "handler" in text)
    ) or "dashboard/date" in text:
        return "custom_view_result_rendering"
    if category in {"polling_checkpoint", "soar_metadata"} and any(
        term in text
        for term in (
            "on_poll",
            "save_artifact",
            "save_container",
            "artifact label",
            "dynamic cef",
            "polling artifacts",
            "polling state",
            "poll controls",
        )
    ):
        return "polling_container_artifact_contract"
    if ("pagination has no maximum page guard" in title or ("pagination" in title and "no upper bound" in title)) and any(
        term in text for term in ("on_poll", "polling", "dashboard", "ioc_view", "feed")
    ):
        return "polling_container_artifact_contract"
    if "pagination has no maximum page guard" in title or ("pagination" in title and "no upper bound" in title):
        return "pagination_bounds"
    if (
        "jsonl parser" in text
        or "malformed lines" in text
        or "unknown action identifiers" in text
        or "indexes response data after only checking count" in text
    ):
        return "runtime_error_handling"

    same_title = title.strip()
    if same_title:
        return f"same:{category}:{same_title}:{file_path}"
    return None


def merge_comment_group(key: str, items: list[dict[str, Any]]) -> dict[str, Any]:
    items = sorted(
        items,
        key=lambda item: (
            SEVERITY_ORDER.get(str(item.get("severity", "medium")), 2),
            0 if str(item.get("source") or "") == "deterministic" else 1,
            0 if normalize_line(item.get("line")) else 1,
        ),
    )
    base = dict(items[0])
    base["title"] = grouped_title(key, items)
    base["severity"] = strongest_severity(items)
    base["confidence"] = "high" if all(str(item.get("confidence") or "").lower() == "high" for item in items) else "medium"
    base["evidence"] = grouped_evidence(key, items)
    base["why_it_matters"] = grouped_why(key, items)
    base["suggested_fix"] = grouped_fix(key, items)
    base["code_reference"] = grouped_code_reference(key, items)
    base["suggested_code"] = None
    return base


def grouped_title(key: str, items: list[dict[str, Any]]) -> str:
    titles = {
        "network_request_safety": "External API request handling has TLS, timeout, or debug-data safety gaps",
        "action_metadata_contract": "Action output and indicator metadata are incomplete",
        "custom_view_result_rendering": "Custom views or dashboard wiring do not render action results correctly",
        "polling_container_artifact_contract": "on_poll container/artifact behavior does not meet SOAR polling expectations",
        "pagination_bounds": "Polling or dashboard pagination lacks bounded page/result limits",
        "runtime_error_handling": "Runtime error paths can fail silently or crash with raw exceptions",
    }
    if key in titles:
        return titles[key]
    title = str(items[0].get("title") or "Grouped review finding").strip()
    if len(items) > 1 and not title.lower().startswith("multiple"):
        return f"{title} in multiple locations"
    return title


def grouped_evidence(key: str, items: list[dict[str, Any]]) -> str:
    locations = []
    for item in items:
        location = format_finding_location(item)
        if location:
            locations.append(location)
    location_text = ", ".join(dedupe_text(locations)[:8])
    root = str(items[0].get("evidence") or "").strip()
    if key == "polling_container_artifact_contract":
        root = "The polling implementation has multiple SOAR contract issues around poll controls, labels, save results, state/checkpointing, or artifact metadata."
    elif key == "action_metadata_contract":
        root = "The action metadata is incomplete for the data and indicators the connector exposes."
    elif key == "network_request_safety":
        root = "External API request handling has concrete safety gaps around TLS verification, timeouts, debug data, rate limits, or test-connectivity endpoint choice."
    elif key == "custom_view_result_rendering":
        root = "The custom view/dashboard path does not reliably render the completed action result data."
    elif key == "runtime_error_handling":
        root = "Several runtime edge cases can either crash with raw Python errors or silently report partial/successful results."
    if location_text:
        return f"{root} Related locations: {location_text}."
    return root


def grouped_why(key: str, items: list[dict[str, Any]]) -> str:
    why_by_key = {
        "network_request_safety": "Connector users need bounded, verifiable, and non-leaky API calls so credentials, threat-intel data, and SOAR workers are protected.",
        "action_metadata_contract": "SOAR playbooks and the datapath picker rely on app metadata matching emitted data, summaries, and typed indicator inputs.",
        "custom_view_result_rendering": "Custom views can fail at render time, show empty data, or re-call external APIs with missing parameters.",
        "polling_container_artifact_contract": "Polling bugs create duplicate or unroutable containers/artifacts and can hide ingestion failures from users.",
        "pagination_bounds": "A bad API response or very large feed can consume worker or web-process resources without a clear stopping point.",
        "runtime_error_handling": "Raw crashes and silent partial success make polling/action results unreliable and hard for users to troubleshoot.",
    }
    return why_by_key.get(key) or str(items[0].get("why_it_matters") or "").strip()


def grouped_fix(key: str, items: list[dict[str, Any]]) -> str:
    fix_by_key = {
        "network_request_safety": "Add a secure `verify_server_cert` config path, pass explicit request timeouts, remove full response/header debug data, handle 429s distinctly when applicable, and make test connectivity use a lightweight auth/status path instead of a quota-consuming sample lookup.",
        "action_metadata_contract": "Declare the emitted `action_result.data.*` and `action_result.summary.*` paths, and add `contains` metadata for hash/IP/URL/domain parameters.",
        "custom_view_result_rendering": "Render SOAR's existing action result data in the view, import `ActionResult` explicitly if still needed, and avoid re-calling action handlers from GET parameters.",
        "polling_container_artifact_contract": "Rework on_poll to use standard poll controls, persist checkpoints only after successful ingestion, set container/artifact labels, check save return values, and map indicators to stable CEF fields.",
        "pagination_bounds": "Add a configurable max page/result guard and report clearly when a poll or dashboard view hits that bound.",
        "runtime_error_handling": "Validate response shapes before indexing, return APP_ERROR for unknown actions, and surface malformed JSONL/feed lines as an error or counted warning instead of silently continuing.",
    }
    base_fix = fix_by_key.get(key) or str(items[0].get("suggested_fix") or "").strip()
    concrete_fixes = grouped_concrete_fixes(items)
    if not concrete_fixes:
        return base_fix
    return f"{base_fix} Specific fixes: {'; '.join(concrete_fixes)}."


def grouped_concrete_fixes(items: list[dict[str, Any]], *, limit: int = 5) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for item in items:
        fix = str(item.get("suggested_fix") or "").strip().rstrip(".")
        if not fix:
            continue
        normalized = re.sub(r"\s+", " ", fix.lower())
        if normalized in seen:
            continue
        seen.add(normalized)
        output.append(clamp_text(fix, 220))
        if len(output) >= limit:
            break
    return output


def grouped_code_reference(key: str, items: list[dict[str, Any]]) -> str:
    refs = [
        str(item.get("code_reference") or "").strip()
        for item in items
        if str(item.get("code_reference") or "").strip()
    ]
    if refs:
        return "; ".join(dedupe_text(refs)[:5])
    locations = [format_finding_location(item) for item in items]
    return "; ".join(dedupe_text([item for item in locations if item])[:5])


def strongest_severity(items: list[dict[str, Any]]) -> str:
    return min(
        (str(item.get("severity") or "medium").lower() for item in items),
        key=lambda severity: SEVERITY_ORDER.get(severity, 2),
    )


def format_finding_location(finding: dict[str, Any]) -> str:
    path = str(finding.get("file") or "").strip()
    if not path:
        return ""
    line = finding.get("line")
    if isinstance(line, int) and line > 0:
        return f"{path}:{line}"
    if isinstance(line, str) and line.isdigit():
        return f"{path}:{line}"
    code_reference = str(finding.get("code_reference") or "").strip()
    if code_reference:
        return f"{path} ({code_reference})"
    return path


def dedupe_text(values: list[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = value.strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        output.append(cleaned)
    return output


def all_comment_finding_text(finding: dict[str, Any]) -> str:
    return " ".join(
        str(finding.get(key) or "")
        for key in ("title", "category", "file", "code_reference", "evidence", "why_it_matters", "suggested_fix")
    ).lower()


def finding_identity(finding: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(finding.get("category") or ""),
        str(finding.get("title") or "").lower(),
        str(finding.get("file") or ""),
        str(finding.get("line") or ""),
    )


def finding_context_filter_reason(finding: dict[str, Any], review_input: dict[str, Any]) -> dict[str, str] | None:
    if is_stale_removed_sdk_manifest_finding(finding, review_input):
        return {
            "reason": "stale_removed_sdk_manifest",
            "detail": "deleted legacy app JSON is not the active metadata source for this SDK migration",
            "evidence_source": "pr_scope",
        }
    if is_outside_changed_pr_scope(finding, review_input):
        return {
            "reason": "outside_changed_pr_scope",
            "detail": "finding target is not tied to a changed file, changed hunk, CI blocker, or inferable changed anchor",
            "evidence_source": "pr_scope",
        }
    unsupported_identifier_reason = unconnected_identifier_inference_reason(finding)
    if unsupported_identifier_reason:
        return unsupported_identifier_reason
    return None


def unconnected_identifier_inference_reason(finding: dict[str, Any]) -> dict[str, str] | None:
    """Detect behavioral claims that join unrelated identifier evidence.

    This is a deliberately narrow verifier. It does not second-guess normal
    findings; it only suppresses model claims that use repository examples for
    one identifier role as proof about a different runtime identifier without
    stating a connecting assignment/path.
    """

    source = str(finding.get("source") or "").lower()
    if source in {"deterministic", "static", "ci"}:
        return None
    category = str(finding.get("category") or "")
    if category not in {
        "api_auth_correctness",
        "polling_checkpoint",
        "output_schema_mismatch",
        "soar_metadata",
        "validation",
        "general",
    }:
        return None

    text = " ".join(
        str(finding.get(key) or "")
        for key in ("title", "evidence", "why_it_matters", "suggested_fix", "code_reference")
    )
    lowered = text.lower()
    roles = identifier_roles_in_text(lowered)
    if len(roles) < 2:
        return None
    if has_direct_identifier_connection(lowered):
        return None

    if "uuid" not in lowered and "guid" not in lowered and "format" not in lowered and "validation" not in lowered:
        return None
    if not mentions_repository_example_without_runtime_link(lowered):
        return None

    conflicting_roles = sorted(roles)
    return {
        "reason": "unconnected_identifier_inference",
        "detail": (
            "finding appears to infer runtime behavior across distinct identifier roles "
            f"without an assignment/provenance chain: {', '.join(conflicting_roles)}"
        ),
        "evidence_source": "behavioral_evidence_verifier",
        "confidence_after": "low",
    }


def identifier_roles_in_text(lowered: str) -> set[str]:
    roles: set[str] = set()
    patterns = {
        "asset_id": (
            r"\basset[_\s-]*id\b",
            r"\bassetid\b",
        ),
        "application_id": (
            r"\bapp[_\s-]*id\b",
            r"\bappid\b",
            r"\bapplication[_\s-]*id\b",
            r"\bapplicationid\b",
        ),
        "connector_id": (
            r"\bconnector[_\s-]*id\b",
            r"\bconnectorid\b",
        ),
        "action_id": (
            r"\baction[_\s-]*id\b",
            r"\baction[_\s-]*identifier\b",
        ),
        "user_id": (
            r"\buser[_\s-]*id\b",
            r"\buserid\b",
        ),
        "tenant_id": (
            r"\btenant[_\s-]*id\b",
            r"\btenantid\b",
        ),
        "oauth_client_id": (
            r"\bclient[_\s-]*id\b",
            r"\bclientid\b",
        ),
        "guid_or_uuid": (
            r"\buuid\b",
            r"\bguid\b",
            r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
        ),
    }
    for role, role_patterns in patterns.items():
        if any(re.search(pattern, lowered) for pattern in role_patterns):
            roles.add(role)
    if "guid_or_uuid" in roles and len(roles) == 1:
        roles.remove("guid_or_uuid")
    return roles


def has_direct_identifier_connection(lowered: str) -> bool:
    connection_phrases = (
        "assigned from",
        "assigned to",
        "comes from",
        "derived from",
        "populated from",
        "loaded from",
        "read from",
        "passed from",
        "passed into",
        "mapped from",
        "converted from",
        "copied from",
        "same value",
        "same identifier",
        "stored as",
        "looked up by",
        "assignment trace",
        "provenance",
        "runtime source",
        "request parameter",
        "asset configuration",
        "connector state",
    )
    return any(phrase in lowered for phrase in connection_phrases)


def mentions_repository_example_without_runtime_link(lowered: str) -> bool:
    repository_phrases = (
        ".json",
        "manifest",
        "repository",
        "example",
        "found in",
        "contains",
        "declares",
        "metadata",
    )
    runtime_phrases = (
        "function parameter",
        "request parameter",
        "asset configuration",
        "connector state",
        "api response",
        "assignment trace",
        "source variable",
        "runtime value",
    )
    return any(phrase in lowered for phrase in repository_phrases) and not any(phrase in lowered for phrase in runtime_phrases)


def should_skip_posted_comment(finding: dict[str, Any]) -> bool:
    category = str(finding.get("category") or "")
    confidence = str(finding.get("confidence") or "").lower()
    text = " ".join(
        str(finding.get(key) or "")
        for key in ("title", "evidence", "why_it_matters", "suggested_fix", "code_reference")
    ).lower()
    if not is_actionable_review_finding(finding):
        return True
    if confidence in {"medium", "low"}:
        return True
    if category == "ci_synthesis":
        return True
    if category == "merge_conflict" and is_unhelpful_mergeability_finding(finding):
        return True
    if is_readme_only_finding(finding):
        return True
    if category == "precommit" and not has_actionable_precommit_details(text):
        return True
    speculative_phrases = (
        "if it does, this finding can be closed",
        "if so, this finding can be closed",
        "if it does, close this finding",
        "if this is already handled",
    )
    if any(phrase in text for phrase in speculative_phrases):
        return True
    suggested_fix = str(finding.get("suggested_fix") or "").strip().lower()
    if suggested_fix.startswith(("confirm that ", "verify that ", "confirm whether ", "verify whether ")):
        return True
    if any(suggested_fix.startswith(phrase) for phrase in GENERIC_CI_FIXES) and not has_actionable_precommit_details(text):
        return True
    return False


def should_skip_finding_for_pr_context(finding: dict[str, Any], review_input: dict[str, Any]) -> bool:
    return finding_context_filter_reason(finding, review_input) is not None


def is_outside_changed_pr_scope(finding: dict[str, Any], review_input: dict[str, Any]) -> bool:
    """Return True when a finding is backed only by unchanged full-file context.

    Full files are necessary for comparing schema, docs, and helpers, but the
    bot should not publish old unrelated issues just because a file was fetched.
    A finding is publishable only when it is tied to a changed file/hunk, an
    added/removed file, or a concrete CI/merge signal from this PR.
    """

    category = str(finding.get("category") or "")
    if category in {"precommit", "merge_conflict"}:
        return False

    path = str(finding.get("file") or "").strip()
    if not path:
        return True

    changed = {
        str(item.get("filename")): item
        for item in review_input.get("changed_files", [])
        if isinstance(item, dict) and item.get("filename")
    }
    file_info = changed.get(path)
    if not file_info:
        if category == "docs_pr_accuracy" and can_infer_changed_anchor(finding, review_input):
            return False
        return True

    status = str(file_info.get("status") or "")
    if status in {"added", "removed"}:
        return False

    line = normalize_line(finding.get("line"))
    if line is None:
        return False

    patch = str(file_info.get("patch") or "")
    if not patch:
        return not bool(file_info.get("deep_patch_reconstructed"))

    right_lines = {entry["line"] for entry in parse_right_side_diff_entries(patch)}
    return line not in right_lines


def can_infer_changed_anchor(finding: dict[str, Any], review_input: dict[str, Any]) -> bool:
    if normalize_line(finding.get("line")) is not None:
        return False
    diff_index = build_diff_index(review_input.get("changed_files", []))
    path, line = infer_code_anchor(finding, diff_index, preferred_path=None)
    return bool(path and line)


def is_stale_removed_sdk_manifest_finding(finding: dict[str, Any], review_input: dict[str, Any]) -> bool:
    path = str(finding.get("file") or "").strip()
    if not path or "/" in path or not path.endswith(".json"):
        return False

    changed = {
        str(item.get("filename")): item
        for item in review_input.get("changed_files", [])
        if isinstance(item, dict) and item.get("filename")
    }
    file_info = changed.get(path) or {}
    if file_info.get("status") != "removed":
        return False
    if not is_sdk_migration(review_input):
        return False

    text = " ".join(
        str(finding.get(key) or "")
        for key in ("title", "evidence", "why_it_matters", "suggested_fix", "code_reference")
    ).lower()
    stale_manifest_terms = (
        "main_module",
        "rest_handler",
        "configuration",
        "app_version",
        "python_version",
        "min_phantom_version",
        "asset field",
        "asset configuration",
        "update `main_module`",
        "update main_module",
        "set `\"app_version\"",
        "set `\"python_version\"",
    )
    return str(finding.get("category") or "") == "soar_metadata" and any(term in text for term in stale_manifest_terms)


def is_sdk_migration(review_input: dict[str, Any]) -> bool:
    full_files = review_input.get("full_files") or {}
    changed_paths = {
        str(item.get("filename"))
        for item in review_input.get("changed_files", [])
        if isinstance(item, dict) and item.get("filename")
    }
    pyproject = str(full_files.get("pyproject.toml") or "")
    has_sdk_pyproject = "[tool.soar.app]" in pyproject and "main_module" in pyproject
    has_sdk_app = "src/app.py" in full_files or "src/app.py" in changed_paths
    has_sdk_asset = "src/asset.py" in full_files or "src/asset.py" in changed_paths
    if has_sdk_pyproject and has_sdk_app:
        return True
    return bool(has_sdk_app and has_sdk_asset and "pyproject.toml" in changed_paths)


def is_readme_only_finding(finding: dict[str, Any]) -> bool:
    path = str(finding.get("file") or "").strip().lower()
    if path.rsplit("/", 1)[-1] == "readme.md":
        return True
    if finding.get("category") != "docs_pr_accuracy":
        return False
    text = " ".join(
        str(finding.get(key) or "")
        for key in ("title", "evidence", "why_it_matters", "suggested_fix")
    ).lower()
    return "readme.md" in text and "manual_readme_content.md" not in text and "manual" not in text


def has_actionable_precommit_details(text: str) -> bool:
    if is_ci_boilerplate_text(text):
        return False
    lowered = text.lower()
    if any(keyword in lowered for keyword in PRECOMMIT_DETAIL_KEYWORDS):
        return True
    return re.search(r"\b[FEW]\d{3}\b", text) is not None


def is_ci_boilerplate_text(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) > 1:
        return all(is_ci_boilerplate_text(line) for line in lines)
    lowered = text.lower()
    boilerplate = (
        "pytest-output-raw.log",
        "pytest-output.log",
        "pytest_exit_code",
        "pipestatus",
        "create results directory",
        "running tests and capturing output",
        "test output saved to",
        "shell: /usr/bin/bash",
        "retention-days:",
        "if-no-files-found:",
    )
    return any(phrase in lowered for phrase in boilerplate) or re.search(r"\bpytest\s+suite/apps/", lowered) is not None


def is_unhelpful_mergeability_finding(finding: dict[str, Any]) -> bool:
    text = " ".join(
        str(finding.get(key) or "")
        for key in ("title", "evidence", "why_it_matters", "suggested_fix")
    ).lower()
    return not has_specific_merge_blocker_details(text)


def has_specific_merge_blocker_details(text: str) -> bool:
    if has_actionable_precommit_details(text):
        return True
    concrete_conflict_phrases = (
        "mergeable_state=dirty",
        "mergeable_state is dirty",
        "merge conflict",
        "cannot be automatically merged",
        "resolve conflicts",
        "conflict marker",
        "<<<<<<<",
        ">>>>>>>",
    )
    if any(phrase in text for phrase in concrete_conflict_phrases):
        return True
    if re.search(r"\b[\w./-]+\.(?:py|json|ya?ml|toml|md):\d+", text):
        return True
    specific_failure_phrases = (
        "assertionerror",
        "traceback",
        "failed -",
        "hook id:",
        "conflict in ",
        "test failed:",
        "coverage",
        "semgrep",
        "detect-secrets",
    )
    return any(phrase in text for phrase in specific_failure_phrases)


def build_comment_for_finding(
    finding: dict[str, Any],
    review_input: dict[str, Any],
    diff_index: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    title = str(finding.get("title") or "").strip()
    if not title:
        return None

    path = str(finding.get("file") or "").strip() or None
    line = normalize_line(finding.get("line"))
    if should_infer_code_anchor(finding, path, line, diff_index):
        inferred_path, inferred_line = infer_code_anchor(finding, diff_index, preferred_path=path)
        path = inferred_path or path
        line = inferred_line or line
    finding_type = classify_finding(finding, path=path)
    target = choose_target(path, line, diff_index)
    finding_id = stable_finding_id(finding)
    marker = f"{MARKER_PREFIX}{finding_id} -->"
    body = render_short_comment(finding, finding_type, target)
    body = append_code_suggestion_block(body, finding, finding_type, target)
    body = redact_text(body)

    return {
        "id": finding_id,
        "finding_id": str(finding.get("id") or ""),
        "title": title,
        "severity": finding.get("severity", "medium"),
        "confidence": finding.get("confidence", "high"),
        "category": finding.get("category", "general"),
        "finding_type": finding_type,
        "comment_style": comment_style_for(finding, finding_type, target),
        "github_comment_type": target["github_comment_type"],
        "path": target.get("path"),
        "line": target.get("line"),
        "side": target.get("side"),
        "target_reason": target["target_reason"],
        "body": body,
        "marker": marker,
    }


def build_diff_index(changed_files: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for item in changed_files:
        if not isinstance(item, dict):
            continue
        path = item.get("filename")
        if not path:
            continue
        right_entries = parse_right_side_diff_entries(str(item.get("patch") or ""))
        right_lines = {entry["line"] for entry in right_entries}
        output[str(path)] = {
            "right_lines": right_lines,
            "right_entries": right_entries,
            "first_right_line": min(right_lines) if right_lines else None,
            "status": item.get("status"),
        }
    return output


def parse_right_side_diff_lines(patch: str) -> set[int]:
    return {entry["line"] for entry in parse_right_side_diff_entries(patch)}


def parse_right_side_diff_entries(patch: str) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    right_line: int | None = None
    left_line: int | None = None
    hunk_pattern = re.compile(r"^@@ -(?P<left>\d+)(?:,\d+)? \+(?P<right>\d+)(?:,\d+)? @@")

    for raw_line in patch.splitlines():
        hunk_match = hunk_pattern.match(raw_line)
        if hunk_match:
            left_line = int(hunk_match.group("left"))
            right_line = int(hunk_match.group("right"))
            continue
        if right_line is None or left_line is None:
            continue
        if raw_line.startswith("+") and not raw_line.startswith("+++"):
            entries.append({"line": right_line, "text": raw_line[1:], "kind": "added"})
            right_line += 1
        elif raw_line.startswith("-") and not raw_line.startswith("---"):
            left_line += 1
        elif raw_line.startswith(" "):
            entries.append({"line": right_line, "text": raw_line[1:], "kind": "context"})
            left_line += 1
            right_line += 1
    return entries


def choose_target(path: str | None, line: int | None, diff_index: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if not path or path not in diff_index:
        return {
            "github_comment_type": "conversation",
            "path": path,
            "line": line,
            "side": None,
            "anchor_text": None,
            "target_reason": "finding has no changed-file line target",
        }

    file_info = diff_index[path]
    right_lines = file_info["right_lines"]
    if line and line in right_lines:
        return {
            "github_comment_type": "line",
            "path": path,
            "line": line,
            "side": "RIGHT",
            "anchor_text": entry_text_for_line(file_info, line),
            "target_reason": "finding line is present in the PR diff",
        }
    return {
        "github_comment_type": "conversation",
        "path": path,
        "line": line,
        "side": None,
        "anchor_text": None,
        "target_reason": "finding line is not present in the PR diff",
    }


def entry_text_for_line(file_info: dict[str, Any], line: int) -> str:
    for entry in file_info.get("right_entries", []):
        if entry.get("line") == line:
            return str(entry.get("text") or "")
    return ""


def should_infer_code_anchor(
    finding: dict[str, Any],
    path: str | None,
    line: int | None,
    diff_index: dict[str, dict[str, Any]],
) -> bool:
    if str(finding.get("category") or "") in {"missing_tests", "ci_synthesis", "precommit", "merge_conflict"}:
        return False
    if (not path or path not in diff_index) and str(finding.get("category") or "") == "docs_pr_accuracy" and line is None:
        return bool(target_keywords_for_finding(finding, all_finding_text(finding), diff_index))
    if not path or path not in diff_index:
        return False
    if line is None:
        return bool(target_keywords_for_finding(finding, all_finding_text(finding), diff_index))
    if line not in diff_index[path]["right_lines"]:
        return False
    anchor_text = entry_text_for_line(diff_index[path], line)
    return is_low_signal_anchor_line(anchor_text)


def infer_code_anchor(
    finding: dict[str, Any],
    diff_index: dict[str, dict[str, Any]],
    *,
    preferred_path: str | None = None,
) -> tuple[str | None, int | None]:
    text = all_finding_text(finding)
    keywords = target_keywords_for_finding(finding, text, diff_index)
    if not keywords:
        return None, None

    best: tuple[int, str, int] | None = None
    for path, info in diff_index.items():
        if not is_code_path(path):
            continue
        for entry in info.get("right_entries", []):
            line_text = str(entry.get("text") or "").lower()
            score = score_anchor_line(path, line_text, str(entry.get("kind") or ""), keywords, str(finding.get("category") or ""))
            if preferred_path and path == preferred_path:
                score += 20
            if score <= 0:
                continue
            candidate = (score, path, int(entry["line"]))
            if best is None or candidate[0] > best[0]:
                best = candidate

    if best is None:
        for path, info in diff_index.items():
            if is_code_path(path) and info.get("first_right_line"):
                return path, int(info["first_right_line"])
        return None, None
    return best[1], best[2]


def all_finding_text(finding: dict[str, Any]) -> str:
    return " ".join(
        str(finding.get(key) or "")
        for key in ("title", "evidence", "why_it_matters", "suggested_fix", "code_reference", "category")
    ).lower()


def is_low_signal_anchor_line(line_text: str) -> bool:
    stripped = line_text.strip()
    if not stripped:
        return True
    if stripped in {'"""', "'''", "{", "}", "[", "]", "(", ")"}:
        return True
    if stripped.startswith(('"""', "'''")) and len(stripped) <= 8:
        return True
    return False


def target_keywords_for_finding(finding: dict[str, Any], text: str, diff_index: dict[str, dict[str, Any]]) -> list[str]:
    literal_keywords = extract_code_like_keywords(text)
    if finding.get("category") == "docs_pr_accuracy":
        if literal_keywords:
            return literal_keywords + target_keywords_for_text(text)
        diff_text = all_diff_text(diff_index)
        if any(word in diff_text for word in ("public_cert", "private_key", "certificate", "cert=")):
            return ["public_cert", "private_key", "assetfield", "configuration", "certificate", "cert", "temp_cert_files", "cert="]
    if finding.get("category") == "missing_tests":
        diff_text = all_diff_text(diff_index)
        diff_keywords = target_keywords_for_text(diff_text)
        if diff_keywords:
            return diff_keywords
    return dedupe_keywords(literal_keywords + target_keywords_for_text(text))


def dedupe_keywords(keywords: list[str]) -> list[str]:
    seen = set()
    output = []
    for keyword in keywords:
        if keyword in seen:
            continue
        seen.add(keyword)
        output.append(keyword)
    return output[:12]


def target_keywords_for_text(text: str) -> list[str]:
    if any(word in text for word in ("certificate", "cert", "public_cert", "private_key")):
        return ["temp_cert_files", "cert=", "public_cert", "private_key", "certificate", "cert"]
    if any(word in text for word in ("poll", "checkpoint", "dedup", "state")):
        return ["on_poll", "checkpoint", "save_state", "poll", "source_data_identifier"]
    if any(word in text for word in ("oauth", "token", "auth", "session")):
        return ["oauth", "token", "auth", "session", "headers"]
    if any(word in text for word in ("validation", "validate", "base64", "b64decode")):
        return ["validate", "validation", "b64decode", "base64"]
    return []


def extract_code_like_keywords(text: str) -> list[str]:
    keywords: list[str] = []
    for match in re.findall(r"`([^`]+)`", text):
        if is_useful_keyword(match):
            keywords.append(match.lower())
    for match in re.findall(r"\b[a-z][a-z0-9]+(?:_[a-z0-9]+)+\b", text):
        if is_useful_keyword(match):
            keywords.append(match.lower())
    seen = set()
    output = []
    for keyword in keywords:
        if keyword in seen:
            continue
        seen.add(keyword)
        output.append(keyword)
    return output[:8]


def is_useful_keyword(value: str) -> bool:
    value = value.strip().lower()
    if not value or len(value) < 4:
        return False
    if len(value) > 80:
        return False
    if " " in value:
        return False
    return True


def all_diff_text(diff_index: dict[str, dict[str, Any]]) -> str:
    pieces = []
    for info in diff_index.values():
        for entry in info.get("right_entries", []):
            pieces.append(str(entry.get("text") or ""))
    return "\n".join(pieces).lower()


def score_anchor_line(path: str, line_text: str, kind: str, keywords: list[str], category: str = "") -> int:
    score = 0
    for index, keyword in enumerate(keywords):
        if keyword in line_text:
            score += 50 - index
    if kind == "added":
        score += 8
    if category == "docs_pr_accuracy" and path.endswith(".json"):
        score += 18
    if category == "docs_pr_accuracy" and path.endswith(".py"):
        score -= 8
    if "def " in line_text:
        score += 6
    if path.endswith(".py"):
        score += 4
    return score


def classify_finding(finding: dict[str, Any], *, path: str | None = None) -> str:
    category = str(finding.get("category") or "")
    if category == "docs_pr_accuracy":
        return "text"
    file_path = path if path is not None else str(finding.get("file") or "")
    suffix = "." + file_path.rsplit(".", 1)[-1].lower() if "." in file_path else ""
    if suffix in CODE_SUFFIXES:
        return "code"
    if category in TEXT_CATEGORIES or suffix in TEXT_SUFFIXES:
        return "text"
    return "text"


def is_code_path(path: str) -> bool:
    suffix = "." + path.rsplit(".", 1)[-1].lower() if "." in path else ""
    return suffix in CODE_SUFFIXES


def render_short_comment(finding: dict[str, Any], finding_type: str, target: dict[str, Any]) -> str:
    title = first_sentence(str(finding.get("title") or "Review finding."))
    evidence = concise_comment_text(str(finding.get("evidence") or ""), max_chars=650, max_sentences=3)
    why = concise_comment_text(str(finding.get("why_it_matters") or ""), max_chars=420, max_sentences=2)
    fix = concise_comment_text(str(finding.get("suggested_fix") or ""), max_chars=800, max_sentences=4)
    code_reference = str(finding.get("code_reference") or "").strip()
    location = format_conversation_location(target)
    anchor_text = concise_anchor_text(str(target.get("anchor_text") or ""))

    issue_text = title
    if evidence and evidence.lower() not in title.lower():
        issue_text = f"{issue_text} {evidence}"

    lines = [f"Issue: {issue_text}"]
    if location and target["github_comment_type"] == "conversation":
        lines.append(f"Location: `{location}`")
    if code_reference:
        lines.append(f"Code reference: `{code_reference}`")
    if anchor_text and target["github_comment_type"] == "line":
        lines.append(f"Current line: `{anchor_text}`")
    if why:
        lines.append(f"Impact: {why}")
    if fix:
        lines.append(f"How to fix: {fix}")

    return clamp_text("\n\n".join(lines), 1900)


def comment_style_for(finding: dict[str, Any], finding_type: str, target: dict[str, Any]) -> str:
    if code_suggestion_for(finding, finding_type, target):
        return "code_comment_with_github_suggestion"
    return f"{finding_type}_comment_with_suggested_fix"


def append_code_suggestion_block(
    body: str,
    finding: dict[str, Any],
    finding_type: str,
    target: dict[str, Any],
) -> str:
    suggestion = code_suggestion_for(finding, finding_type, target)
    if not suggestion:
        return body
    return f"{body}\n\n```suggestion\n{suggestion}\n```"


def code_suggestion_for(finding: dict[str, Any], finding_type: str, target: dict[str, Any]) -> str:
    if finding_type != "code":
        return ""
    if target.get("github_comment_type") != "line":
        return ""
    return normalize_suggested_code(finding.get("suggested_code"))


def normalize_suggested_code(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip("\n")
    if not text.strip():
        return ""
    fence_match = re.fullmatch(r"\s*```(?:suggestion|[a-zA-Z0-9_-]+)?\n(?P<code>.*?)\n```\s*", text, flags=re.DOTALL)
    if fence_match:
        text = fence_match.group("code").strip("\n")
    if "```" in text:
        return ""
    return clamp_text(text, 1800)


def format_conversation_location(target: dict[str, Any]) -> str:
    path = target.get("path")
    if not path:
        return ""
    line = target.get("line")
    if isinstance(line, int):
        return f"{path}:{line}"
    return str(path)


def stable_finding_id(finding: dict[str, Any]) -> str:
    raw = "|".join(
        [
            str(finding.get("id") or ""),
            str(finding.get("title") or ""),
            str(finding.get("file") or ""),
            str(finding.get("line") or ""),
            str(finding.get("category") or ""),
        ]
    )
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return f"finding-{digest}"


def normalize_line(value: Any) -> int | None:
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def first_sentence(text: str) -> str:
    text = " ".join(text.strip().split())
    if not text:
        return ""
    for separator in (". ", "; "):
        if separator in text:
            text = text.split(separator, 1)[0] + "."
            break
    return clamp_text(text, 420)


def concise_comment_text(text: str, *, max_chars: int, max_sentences: int) -> str:
    text = " ".join(text.strip().split())
    if not text:
        return ""
    sentences = split_comment_sentences(text)
    if len(sentences) <= 1:
        return clamp_text(text, max_chars)

    selected = []
    total = 0
    for sentence in sentences:
        candidate_length = total + len(sentence) + (1 if selected else 0)
        if selected and (len(selected) >= max_sentences or candidate_length > max_chars):
            break
        selected.append(sentence)
        total = candidate_length
    if not selected:
        selected = [sentences[0]]
    return clamp_text(" ".join(selected), max_chars)


def split_comment_sentences(text: str) -> list[str]:
    """Split prose without breaking common code references like file.py:1:1."""

    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9`])", text)
    return [part.strip() for part in parts if part.strip()]


def concise_anchor_text(text: str) -> str:
    text = " ".join(text.strip().split())
    if not text:
        return ""
    return clamp_text(text, 180)


def clamp_text(text: str, max_chars: int) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "..."


def render_comment_plan(plan: dict[str, Any]) -> str:
    lines = [
        "## Planned GitHub Comments",
        "",
        f"PR: `{plan.get('repo')}#{plan.get('pr_number')}`",
        "",
    ]
    comments = plan.get("comments") or []
    if not comments:
        lines.append("No comments planned.")
        return "\n".join(lines) + "\n"

    for index, comment in enumerate(comments, start=1):
        target = "conversation"
        if comment.get("github_comment_type") == "line":
            target = f"{comment.get('path')}:{comment.get('line')}"
        elif comment.get("path"):
            target = str(comment.get("path"))
        lines.extend(
            [
                f"{index}. `{comment.get('github_comment_type')}` / `{comment.get('finding_type')}` / `{target}`",
                "",
                comment.get("body") or "",
                "",
            ]
        )
    return redact_text("\n".join(lines).rstrip() + "\n")


def publish_comment_plan(
    client: GitHubClient,
    plan: dict[str, Any],
    review_input: dict[str, Any],
    *,
    allow_duplicates: bool = False,
    max_comments: int | None = None,
) -> dict[str, Any]:
    repo = str(plan.get("repo") or "")
    pr_number = int(plan.get("pr_number") or 0)
    head_sha = str(plan.get("head_sha") or "")
    existing_markers = collect_existing_markers(review_input)
    comments = list(plan.get("comments") or [])
    if max_comments is not None:
        comments = comments[:max_comments]

    results = []
    for comment in comments:
        marker = str(comment.get("marker") or "")
        if marker and marker in existing_markers and not allow_duplicates:
            results.append({"id": comment.get("id"), "status": "skipped_duplicate"})
            continue
        body = redact_text(body_with_marker(comment))
        try:
            if comment.get("github_comment_type") == "line":
                response = client.create_pull_request_line_comment(
                    repo,
                    pr_number,
                    body=body,
                    commit_id=head_sha,
                    path=str(comment["path"]),
                    line=int(comment["line"]),
                    side=str(comment.get("side") or "RIGHT"),
                )
                results.append(
                    {
                        "id": comment.get("id"),
                        "status": "posted",
                        "github_comment_type": "line",
                        "url": response.get("html_url") if isinstance(response, dict) else None,
                    }
                )
            else:
                response = client.create_issue_comment(repo, pr_number, body=body)
                results.append(
                    {
                        "id": comment.get("id"),
                        "status": "posted",
                        "github_comment_type": "conversation",
                        "url": response.get("html_url") if isinstance(response, dict) else None,
                    }
                )
        except GitHubError as exc:
            safe_error = redact_text(str(exc))
            if comment.get("github_comment_type") != "line":
                results.append({"id": comment.get("id"), "status": "error", "error": safe_error})
                continue
            try:
                response = client.create_issue_comment(repo, pr_number, body=body)
                results.append(
                    {
                        "id": comment.get("id"),
                        "status": "posted_fallback",
                        "github_comment_type": "conversation",
                        "line_error": safe_error,
                        "url": response.get("html_url") if isinstance(response, dict) else None,
                    }
                )
            except GitHubError as fallback_exc:
                safe_fallback_error = redact_text(str(fallback_exc))
                results.append(
                    {
                        "id": comment.get("id"),
                        "status": "error",
                        "error": safe_error,
                        "fallback_error": safe_fallback_error,
                    }
                )

    posted_count = sum(1 for item in results if str(item.get("status")).startswith("posted"))
    label_result = apply_reviewed_label(client, repo, pr_number, posted_count=posted_count)
    return {
        "schema_version": "0.1",
        "repo": repo,
        "pr_number": pr_number,
        "attempted": len(comments),
        "posted": posted_count,
        "skipped": sum(1 for item in results if str(item.get("status")).startswith("skipped")),
        "errors": sum(1 for item in results if item.get("status") == "error") + (1 if label_result.get("status") == "error" else 0),
        "label": label_result,
        "results": results,
    }


def apply_reviewed_label(
    client: GitHubClient,
    repo: str,
    pr_number: int,
    *,
    posted_count: int,
    label: str = "ai-reviewed",
) -> dict[str, Any]:
    if posted_count <= 0:
        return {"name": label, "status": "skipped_no_posted_comments"}
    try:
        response = client.add_issue_labels(repo, pr_number, [label])
    except GitHubError as exc:
        return {"name": label, "status": "error", "error": redact_text(str(exc))}
    return {
        "name": label,
        "status": "applied",
        "url": response[0].get("url") if isinstance(response, list) and response and isinstance(response[0], dict) else None,
    }


def collect_existing_markers(review_input: dict[str, Any]) -> set[str]:
    markers: set[str] = set()
    comments = review_input.get("comments") or {}
    for key in ("issue_comments", "review_comments", "reviews"):
        for item in comments.get(key, []) or []:
            body = str(item.get("body") or "")
            markers.update(re.findall(r"<!-- agentic-pr-review:[^>]+-->", body))
    return markers


def body_with_marker(comment: dict[str, Any]) -> str:
    marker = str(comment.get("marker") or "")
    body = str(comment.get("body") or "").rstrip()
    return redact_text(f"{body}\n\n{marker}".rstrip() + "\n")
