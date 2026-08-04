"""Deterministic connector-review checks.

These checks are intentionally conservative signals for the LLM and comment
renderer. They do not replace model review; they anchor known recurring SOAR
connector mistakes in concrete code evidence.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any

from .models import normalize_finding
from .sdk_analysis import annotation_is_string_like_text, build_sdk_review_inventory


MUTATING_PREFIXES = (
    "add",
    "block",
    "create",
    "delete",
    "disable",
    "enable",
    "patch",
    "post",
    "put",
    "remove",
    "reset",
    "revoke",
    "rotate",
    "set",
    "take",
    "terminate",
    "unblock",
    "update",
)

SENSITIVE_WORDS = (
    "access_token",
    "apikey",
    "api_key",
    "auth",
    "authorization",
    "bearer",
    "client_secret",
    "credential",
    "header",
    "password",
    "payload",
    "refresh_token",
    "response",
    "secret",
    "tenant",
    "token",
)

PAGINATION_WORDS = (
    "cursor",
    "limit",
    "next",
    "next_page",
    "nextpagetoken",
    "offset",
    "page",
    "pagination",
    "per_page",
    "while ",
)

POLL_PARAM_NAMES = {
    "container_count",
    "artifact_count",
    "start_time",
    "end_time",
    "container_id",
    "poll_now",
}

RISKY_CHANGE_WORDS = (
    "auth",
    "client_secret",
    "checkpoint",
    "delete",
    "oauth",
    "on_poll",
    "patch",
    "poll",
    "post",
    "put",
    "session",
    "state",
    "token",
    "validate",
)

TARGET_PIPELINE_JOB_NAMES = {
    "pre-commit",
    "compile",
    "build",
    "semantic-release-preview",
}


def run_deterministic_checks(review_input: dict[str, Any]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    full_files = active_full_files(review_input, review_input.get("full_files", {}))
    app_jsons = load_app_jsons(full_files)

    findings.extend(check_missing_timeouts(full_files))
    findings.extend(check_oauth_client_credentials_body(full_files))
    findings.extend(check_oauth_json_body_for_token_request(review_input, full_files))
    findings.extend(check_oauth_v1_v2_endpoint_mismatch(review_input, full_files))
    findings.extend(check_tls_verification_defaults(review_input, full_files, app_jsons))
    findings.extend(check_missing_verify_config_field(full_files, app_jsons))
    findings.extend(check_python_config_reads_missing_json_fields(full_files, app_jsons))
    findings.extend(check_unsafe_logging(full_files))
    findings.extend(check_sensitive_debug_data(full_files))
    findings.extend(check_read_only_metadata(app_jsons))
    findings.extend(check_indicator_parameters_missing_contains(app_jsons))
    findings.extend(check_app_metadata_basics(review_input, app_jsons))
    findings.extend(check_version_bump_missing(review_input, app_jsons))
    findings.extend(check_add_data_schema_mismatch(full_files, app_jsons))
    findings.extend(check_summary_schema_mismatch(full_files, app_jsons))
    findings.extend(check_sdk_summary_type_missing(full_files))
    findings.extend(check_sdk_app_contracts(review_input, full_files))
    findings.extend(check_sdk_migration_parity(review_input, full_files))
    findings.extend(check_sdk_full_migration_gate(review_input, full_files))
    findings.extend(check_sdk_large_output_schema_contractions(review_input, full_files))
    findings.extend(check_sdk_generic_strict_output_models(review_input, full_files))
    findings.extend(check_sdk_pat_only_auth_regression(review_input, full_files))
    findings.extend(check_sdk_issue_number_validation_regressions(review_input, full_files))
    findings.extend(check_sdk_make_request_dynamic_output_serialization(full_files))
    findings.extend(check_sdk_make_request_missing_tests(review_input, full_files))
    findings.extend(check_make_request_security_contracts(review_input, full_files))
    findings.extend(check_sdk_tests_import_package_app_as_top_level(review_input, full_files))
    findings.extend(check_sdk_manifest_result(review_input))
    findings.extend(check_sdk_generated_manifest_contract_drift(review_input))
    findings.extend(check_microsoft_graph_sdk_response_modeling(review_input, full_files))
    findings.extend(check_large_file_transfer_patterns(review_input, full_files))
    findings.extend(check_upload_session_resilience(review_input, full_files))
    findings.extend(check_polling_dedup_heuristics(full_files))
    findings.extend(check_polling_container_artifact_contracts(full_files))
    findings.extend(check_unbounded_poll_pagination(full_files))
    findings.extend(check_polling_state_loaded_but_not_checkpointed(full_files))
    findings.extend(check_pagination_heuristics(full_files, app_jsons))
    findings.extend(check_validation_heuristics(full_files))
    findings.extend(check_custom_view_action_result_misuse(full_files))
    findings.extend(check_dashboard_view_not_wired(review_input, full_files))
    findings.extend(check_unsafe_response_index_after_count(full_files))
    findings.extend(check_unknown_action_success_dispatch(full_files))
    findings.extend(check_jsonl_parser_swallows_malformed_lines(full_files))
    findings.extend(check_generic_rate_limit_handling(full_files))
    findings.extend(check_test_connectivity_uses_quota_action_endpoint(full_files))
    findings.extend(check_swallowed_broad_exceptions(review_input, full_files))
    findings.extend(check_unreachable_code(review_input, full_files))
    findings.extend(check_branch_local_exception_imports(full_files))
    findings.extend(check_sdk_numeric_validation_regressions(review_input, full_files))
    findings.extend(check_sdk_validation_cookbook_regressions(review_input, full_files))
    findings.extend(check_graph_target_user_id_normalization(full_files))
    findings.extend(check_obviously_wrong_output_cef_types(full_files))
    findings.extend(check_search_query_user_input_interpolation(full_files))
    findings.extend(check_search_output_pagination_fields(full_files))
    findings.extend(check_search_api_rate_limit_handling(full_files))
    findings.extend(check_stale_url_helper_comments(full_files))
    findings.extend(check_base64_decode_before_try(review_input, full_files))
    findings.extend(check_removed_safety_regressions(review_input))
    findings.extend(check_removed_tests_for_risky_changes(review_input))
    findings.extend(check_removed_release_notes_for_code_changes(review_input))
    findings.extend(check_on_poll_params_and_tags(full_files, app_jsons))
    findings.extend(check_unused_boilerplate(full_files, app_jsons))
    findings.extend(check_missing_tests(review_input))
    findings.extend(check_sdk_live_only_tests(review_input, full_files))
    findings.extend(check_docs_schema_drift(review_input, app_jsons))
    findings.extend(check_pr_claim_diff_mismatch(review_input))
    findings.extend(check_app_mapping_hint(review_input, app_jsons))
    findings.extend(check_release_notes_missing(review_input))
    findings.extend(check_python_syntax_errors(full_files))
    findings.extend(check_conflict_markers(full_files))
    findings.extend(check_known_static_name_errors(full_files))
    findings.extend(check_target_pipeline_failures(review_input))
    findings.extend(check_precommit_failures(review_input))
    findings.extend(check_merge_conflicts(review_input))
    findings.extend(check_ci_failures(review_input))

    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, str | None, int | None]] = set()
    for raw in findings:
        item = normalize_finding(raw, default_source="deterministic")
        key = (item["title"], item.get("file"), item.get("line"))
        if key in seen:
            continue
        seen.add(key)
        if not item["id"]:
            item["id"] = f"det-{len(normalized) + 1}"
        normalized.append(item)
    return normalized


def load_app_jsons(full_files: dict[str, str]) -> dict[str, dict[str, Any]]:
    app_jsons: dict[str, dict[str, Any]] = {}
    for path, text in full_files.items():
        if "/" in path or not path.endswith(".json"):
            continue
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and isinstance(data.get("actions"), list):
            app_jsons[path] = data
    return app_jsons


def active_full_files(review_input: dict[str, Any], full_files: dict[str, str]) -> dict[str, str]:
    """Return files that exist at PR head.

    GitHub PR file entries for removed files can still expose a raw_url for the
    deleted base content. That is useful diff evidence, but it must not be
    treated as current head source for deterministic checks.
    """

    removed_paths = {
        str(item.get("filename"))
        for item in review_input.get("changed_files", [])
        if isinstance(item, dict) and item.get("filename") and item.get("status") == "removed"
    }
    if not removed_paths:
        return full_files
    return {path: text for path, text in full_files.items() if path not in removed_paths}


def iter_lines(text: str):
    for index, line in enumerate(text.splitlines(), start=1):
        yield index, line


def changed_paths(review_input: dict[str, Any]) -> set[str]:
    return {str(item.get("filename")) for item in review_input.get("changed_files", []) if item.get("filename")}


def changed_file_map(review_input: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(item.get("filename")): item
        for item in review_input.get("changed_files", [])
        if isinstance(item, dict) and item.get("filename")
    }


def changed_text(review_input: dict[str, Any]) -> str:
    pieces = []
    for item in review_input.get("changed_files", []):
        if not isinstance(item, dict):
            continue
        pieces.append(str(item.get("filename") or ""))
        pieces.append(str(item.get("patch") or ""))
    return "\n".join(pieces).lower()


def changed_right_line_numbers(review_input: dict[str, Any]) -> dict[str, set[int]]:
    output: dict[str, set[int]] = {}
    hunk_pattern = re.compile(r"^@@ -\d+(?:,\d+)? \+(?P<right>\d+)(?:,\d+)? @@")
    for item in review_input.get("changed_files", []):
        if not isinstance(item, dict):
            continue
        path = str(item.get("filename") or "")
        patch = str(item.get("patch") or "")
        if not path or not patch:
            continue
        right_line: int | None = None
        for raw_line in patch.splitlines():
            hunk_match = hunk_pattern.match(raw_line)
            if hunk_match:
                right_line = int(hunk_match.group("right"))
                continue
            if right_line is None:
                continue
            if raw_line.startswith("+") and not raw_line.startswith("+++"):
                output.setdefault(path, set()).add(right_line)
                right_line += 1
            elif raw_line.startswith("-") and not raw_line.startswith("---"):
                continue
            else:
                right_line += 1
    return output


def patch_line_groups(review_input: dict[str, Any]) -> dict[str, dict[str, list[str]]]:
    groups: dict[str, dict[str, list[str]]] = {}
    for item in review_input.get("changed_files", []):
        if not isinstance(item, dict):
            continue
        path = str(item.get("filename") or "")
        patch = str(item.get("patch") or "")
        if not path or not patch:
            continue
        file_groups = groups.setdefault(path, {"added": [], "removed": [], "context": []})
        for raw_line in patch.splitlines():
            if raw_line.startswith("@@") or raw_line.startswith("+++") or raw_line.startswith("---"):
                continue
            if raw_line.startswith("+"):
                file_groups["added"].append(raw_line[1:])
            elif raw_line.startswith("-"):
                file_groups["removed"].append(raw_line[1:])
            elif raw_line.startswith(" "):
                file_groups["context"].append(raw_line[1:])
    return groups


def removed_line_evidence(review_input: dict[str, Any], path: str, pattern: re.Pattern[str]) -> tuple[int | None, str | None]:
    hunk_pattern = re.compile(r"^@@ -(?P<left>\d+)(?:,\d+)? \+\d+(?:,\d+)? @@")
    item = changed_file_map(review_input).get(path) or {}
    left_line: int | None = None
    for raw_line in str(item.get("patch") or "").splitlines():
        hunk_match = hunk_pattern.match(raw_line)
        if hunk_match:
            left_line = int(hunk_match.group("left"))
            continue
        if left_line is None:
            continue
        if raw_line.startswith("-") and not raw_line.startswith("---"):
            text = raw_line[1:]
            if pattern.search(text):
                return left_line, text.strip()
            left_line += 1
        elif raw_line.startswith("+") and not raw_line.startswith("+++"):
            continue
        else:
            left_line += 1
    return None, None


def is_changed_line(review_input: dict[str, Any], path: str, line: int) -> bool:
    file_info = changed_file_map(review_input).get(path) or {}
    if file_info.get("status") == "added":
        return True
    return line in changed_right_line_numbers(review_input).get(path, set())


def action_output_paths(action: dict[str, Any]) -> set[str]:
    output = action.get("output") or []
    paths = set()
    for item in output:
        if isinstance(item, dict) and item.get("data_path"):
            paths.add(str(item["data_path"]))
    return paths


def action_parameters(action: dict[str, Any]) -> list[dict[str, Any]]:
    params = action.get("parameters") or action.get("params") or []
    if isinstance(params, dict):
        output = []
        for key, value in params.items():
            if not isinstance(value, dict):
                continue
            copied = dict(value)
            copied.setdefault("name", key)
            copied.setdefault("identifier", key)
            output.append(copied)
        return output
    return [item for item in params if isinstance(item, dict)]


def app_configuration(app_json: dict[str, Any]) -> list[dict[str, Any]]:
    config = app_json.get("configuration") or app_json.get("asset_params") or []
    if isinstance(config, dict):
        output = []
        for key, value in config.items():
            if not isinstance(value, dict):
                continue
            copied = dict(value)
            copied.setdefault("name", key)
            copied.setdefault("identifier", key)
            output.append(copied)
        return output
    return [item for item in config if isinstance(item, dict)]


def find_line(text: str, needle: str) -> int | None:
    lowered = needle.lower()
    for index, line in iter_lines(text):
        if lowered in line.lower():
            return index
    return None


def json_key_line(text: str, key: str) -> int | None:
    pattern = re.compile(rf"^[ \t]*{re.escape(json.dumps(key))}\s*:", re.M)
    match = pattern.search(text)
    if not match:
        return None
    return text[: match.start()].count("\n") + 1


def flatten_json_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        output: list[str] = []
        for item in value.values():
            output.extend(flatten_json_strings(item))
        return output
    if isinstance(value, list):
        output = []
        for item in value:
            output.extend(flatten_json_strings(item))
        return output
    if value is None:
        return []
    return [str(value)]


def line_from_match(text: str, start: int) -> str:
    end = text.find("\n", start)
    if end == -1:
        end = len(text)
    return text[start:end].strip()


def check_missing_timeouts(full_files: dict[str, str]) -> list[dict[str, Any]]:
    findings = []
    pattern = re.compile(
        r"\brequests\.(get|post|put|patch|delete|request)\s*\("
        r"|\brequest_func\s*\("
        r"|\b_get_requests_session\(\)\.(get|post|put|patch|delete|request)\s*\("
        r"|\b[A-Za-z_][A-Za-z0-9_]*_session\.(get|post|put|patch|delete|request)\s*\("
    )
    for path, text in full_files.items():
        if not path.endswith(".py"):
            continue
        lines = text.splitlines()
        for idx, line in enumerate(lines):
            if not pattern.search(line):
                continue
            call_lines = [line]
            balance = line.count("(") - line.count(")")
            cursor = idx + 1
            while balance > 0 and cursor < len(lines) and cursor < idx + 12:
                call_lines.append(lines[cursor])
                balance += lines[cursor].count("(") - lines[cursor].count(")")
                cursor += 1
            snippet = "\n".join(call_lines)
            if "timeout=" in snippet:
                continue
            findings.append(
                {
                    "title": "Direct requests call does not set a timeout",
                    "category": "api_auth_correctness",
                    "severity": "medium",
                    "confidence": "high",
                    "file": path,
                    "line": idx + 1,
                    "evidence": snippet.strip(),
                    "why_it_matters": "Connector actions can hang indefinitely when token or API calls omit request timeouts.",
                    "suggested_fix": "Pass an explicit timeout to the requests call or route the call through the connector REST helper if it sets one.",
                }
            )
    return findings


def check_oauth_client_credentials_body(full_files: dict[str, str]) -> list[dict[str, Any]]:
    findings = []
    for path, text in full_files.items():
        if not path.endswith(".py"):
            continue
        lines = text.splitlines()
        for idx, line in enumerate(lines):
            lowered = line.lower()
            if "client_credentials" not in lowered and "grant_type" not in lowered:
                continue
            window = "\n".join(lines[max(0, idx - 3) : min(len(lines), idx + 11)])
            window_lower = window.lower()
            if "client_id" not in window_lower or "client_secret" not in window_lower:
                continue
            if "clientcredentialsflow" in window_lower or "client_credentials_flow" in window_lower:
                continue
            if not any(term in window_lower for term in ("requests.", "httpx.", "_make_rest_call", ".post(", ".request(")):
                continue
            if "authorization" in window_lower or "auth=" in window_lower or "basic " in window_lower:
                continue
            findings.append(
                {
                    "title": "OAuth client credentials are sent in the request body without Basic auth evidence",
                    "category": "api_auth_correctness",
                    "severity": "medium",
                    "confidence": "medium",
                    "file": path,
                    "line": idx + 1,
                    "evidence": window.strip(),
                    "why_it_matters": "Many OAuth client-credentials endpoints require client_id/client_secret in an HTTP Basic Authorization header, not in the form body.",
                    "suggested_fix": "Verify the vendor token endpoint contract. If Basic auth is required, send Authorization: Basic base64(client_id:client_secret) and keep only grant_type in the body.",
                }
            )
            break
    return findings


def check_oauth_json_body_for_token_request(
    review_input: dict[str, Any],
    full_files: dict[str, str],
) -> list[dict[str, Any]]:
    findings = []
    changed = changed_paths(review_input)
    json_body_pattern = re.compile(r"\b(json|json_data)\s*=")
    token_terms = ("token_endpoint", "/token", "grant_type", "client_secret")
    form_terms = ("application/x-www-form-urlencoded", "form-urlencoded")

    for path, text in full_files.items():
        if not path.endswith(".py") or path not in changed:
            continue
        lines = text.splitlines()
        for idx, line in enumerate(lines):
            if not json_body_pattern.search(line):
                continue
            line_no = idx + 1
            if not is_changed_line(review_input, path, line_no):
                continue
            window = "\n".join(lines[max(0, idx - 8) : min(len(lines), idx + 9)])
            lowered = window.lower()
            if not any(term in lowered for term in token_terms):
                continue
            if any(term in lowered for term in form_terms):
                continue
            findings.append(
                {
                    "title": "OAuth/token request appears to send a JSON body without form-encoding evidence",
                    "category": "api_auth_correctness",
                    "severity": "medium",
                    "confidence": "medium",
                    "file": path,
                    "line": line_no,
                    "evidence": window.strip(),
                    "why_it_matters": "Historical connector reviews caught token endpoints where vendor docs required application/x-www-form-urlencoded but the code sent json/json_data.",
                    "suggested_fix": "Verify the vendor token endpoint contract. If it requires form encoding, send the payload through data=, set Content-Type: application/x-www-form-urlencoded, and keep any JSON body only for endpoints that explicitly accept it.",
                }
            )
            break
    return findings


def check_oauth_v1_v2_endpoint_mismatch(
    review_input: dict[str, Any],
    full_files: dict[str, str],
) -> list[dict[str, Any]]:
    findings = []
    changed = changed_paths(review_input)
    for path, text in full_files.items():
        if not path.endswith(".py") or path not in changed:
            continue
        for function_name, start_line, body in extract_functions_matching(text, r"(auth|oauth|token|connectivity|flow)"):
            lowered = body.lower()
            if "oauth2/v2.0/authorize" not in lowered or "oauth2/token" not in lowered:
                continue
            if "oauth2/v2.0/token" in lowered and lowered.index("oauth2/v2.0/token") < lowered.rfind("oauth2/token"):
                continue
            token_line = None
            token_text = None
            for offset, line in enumerate(body.splitlines()):
                if "oauth2/token" not in line or "oauth2/v2.0/token" in line:
                    continue
                token_line = start_line + offset
                token_text = line.strip()
                break
            if token_line is None:
                token_line = start_line
                token_text = f"{function_name}() mixes v2.0 authorize with a v1 token endpoint."
            findings.append(
                {
                    "title": "OAuth authorization code flow mixes v2.0 authorize with v1 token endpoint",
                    "category": "api_auth_correctness",
                    "severity": "high",
                    "confidence": "high",
                    "file": path,
                    "line": token_line,
                    "code_reference": function_name,
                    "evidence": token_text,
                    "why_it_matters": (
                        "Microsoft authorization codes issued by the v2.0 authorize endpoint should be redeemed "
                        "at the matching v2.0 token endpoint; mixing endpoint generations can break delegated "
                        "OAuth and test connectivity."
                    ),
                    "suggested_fix": "Use the v2.0 token path for the authorization-code token endpoint.",
                    "suggested_code": token_text.replace("oauth2/token", "oauth2/v2.0/token") if token_text else None,
                }
            )
            break
    return findings


def check_tls_verification_defaults(
    review_input: dict[str, Any],
    full_files: dict[str, str],
    app_jsons: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    findings = []
    changed = changed_paths(review_input)
    verify_false_pattern = re.compile(r"\bverify\s*=\s*False\b|['\"]verify['\"]\s*:\s*False\b")
    weak_default_pattern = re.compile(r"\b(?:self\.)?_?verify\s*=\s*[^,\n]*\.get\([^,\n]+,\s*False\s*\)")

    for path, text in full_files.items():
        if not path.endswith(".py"):
            continue
        for line_no, line in iter_lines(text):
            if verify_false_pattern.search(line):
                findings.append(
                    {
                        "title": "TLS certificate verification is explicitly disabled",
                        "category": "api_auth_correctness",
                        "severity": "high",
                        "confidence": "high",
                        "file": path,
                        "line": line_no,
                        "evidence": line.strip(),
                        "why_it_matters": "Connectors should verify TLS by default so token and API traffic are not vulnerable to interception.",
                        "suggested_fix": "Default TLS verification to enabled and only allow an explicit asset option to disable it when the connector convention permits that escape hatch.",
                    }
                )
            elif weak_default_pattern.search(line):
                findings.append(
                    {
                        "title": "TLS verification appears to default to disabled",
                        "category": "api_auth_correctness",
                        "severity": "medium",
                        "confidence": "medium",
                        "file": path,
                        "line": line_no,
                        "evidence": line.strip(),
                        "why_it_matters": "A false default weakens auth and API transport security for every new asset.",
                        "suggested_fix": "Default the verify option to true and preserve compatibility only with an explicit documented opt-out.",
                    }
                )

    for json_path, app_json in app_jsons.items():
        if json_path not in changed:
            continue
        for item in app_configuration(app_json):
            name = str(item.get("name") or item.get("key") or item.get("identifier") or "").lower()
            if name not in {"verify_server_cert", "verify_ssl", "verify"}:
                continue
            default = item.get("default")
            if default is False or str(default).lower() == "false":
                findings.append(
                    {
                        "title": "TLS verification asset parameter defaults to false",
                        "category": "api_auth_correctness",
                        "severity": "medium",
                        "confidence": "medium",
                        "file": json_path,
                        "line": None,
                        "evidence": f"Configuration parameter '{name}' has default={default!r}.",
                        "why_it_matters": "TLS verification defaults are a recurring connector review issue and should match current connector security expectations.",
                        "suggested_fix": "Use a secure true default unless the project convention requires preserving a legacy default, and document any compatibility exception.",
                    }
                )
    return findings


def check_missing_verify_config_field(
    full_files: dict[str, str],
    app_jsons: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    python_refs = []
    for path, text in full_files.items():
        if not path.endswith(".py") or "verify_server_cert" not in text:
            continue
        for line_no, line in iter_lines(text):
            if "verify_server_cert" in line:
                python_refs.append((path, line_no, line.strip()))
                break

    if not python_refs or not app_jsons:
        return []

    configured = False
    for app_json in app_jsons.values():
        for item in app_configuration(app_json):
            name = str(item.get("name") or item.get("key") or item.get("identifier") or "").lower()
            if name == "verify_server_cert":
                configured = True
                break
        if configured:
            break

    if configured:
        return []

    json_path = next(iter(app_jsons))
    json_text = full_files.get(json_path, "")
    py_path, py_line, py_evidence = python_refs[0]
    return [
        {
            "title": "Python reads verify_server_cert but app JSON does not expose the configuration field",
            "category": "api_auth_correctness",
            "severity": "high",
            "confidence": "high",
            "file": json_path,
            "line": json_key_line(json_text, "configuration"),
            "code_reference": "verify_server_cert",
            "evidence": f"{py_path}:{py_line} reads `verify_server_cert` (`{py_evidence}`), but no matching asset configuration field exists in the app JSON.",
            "why_it_matters": "Users cannot enable or control TLS certificate verification from the asset UI if the runtime option is absent from configuration metadata.",
            "suggested_fix": "Add a boolean `verify_server_cert` asset parameter with a secure default, and have all external API calls use that value consistently.",
        }
    ]


def check_python_config_reads_missing_json_fields(
    full_files: dict[str, str],
    app_jsons: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    if not app_jsons:
        return []

    configured_names: set[str] = set()
    for app_json in app_jsons.values():
        for item in app_configuration(app_json):
            name = str(item.get("name") or item.get("key") or item.get("identifier") or "").strip().lower()
            if name:
                configured_names.add(name)

    ignored = {
        "asset_id",
        "app_version",
        "directory",
        "ingest",
        "main_module",
        "verify_server_cert",
    }
    config_read_pattern = re.compile(
        r"(?:self\.)?get_config\(\)\s*(?:\.get\(\s*['\"](?P<direct_get>[^'\"]+)['\"]|\[\s*['\"](?P<direct_index>[^'\"]+)['\"]\s*\])"
        r"|\bconfig\s*(?:\.get\(\s*['\"](?P<var_get>[^'\"]+)['\"]|\[\s*['\"](?P<var_index>[^'\"]+)['\"]\s*\])"
    )

    missing_refs: dict[str, tuple[str, int, str]] = {}
    for path, text in full_files.items():
        if not path.endswith(".py") or ("get_config" not in text and "config" not in text):
            continue
        for line_no, line in iter_lines(text):
            for match in config_read_pattern.finditer(line):
                raw_key = next((match.group(name) for name in ("direct_get", "direct_index", "var_get", "var_index") if match.group(name)), "")
                key = raw_key.strip().lower()
                if not key or key in configured_names or key in ignored or key.startswith("_"):
                    continue
                missing_refs.setdefault(key, (path, line_no, line.strip()))

    if not missing_refs:
        return []

    json_path = next(iter(app_jsons))
    json_text = full_files.get(json_path, "")
    evidence_parts = [
        f"`{key}` at {path}:{line_no} (`{line}`)"
        for key, (path, line_no, line) in sorted(missing_refs.items())[:6]
    ]
    more_count = max(0, len(missing_refs) - len(evidence_parts))
    more_text = f" plus {more_count} more" if more_count else ""
    return [
        {
            "title": "Python reads asset configuration fields missing from app JSON",
            "category": "soar_metadata",
            "severity": "high",
            "confidence": "high",
            "file": json_path,
            "line": json_key_line(json_text, "configuration"),
            "code_reference": "configuration",
            "evidence": (
                "Python reads asset config keys that are not exposed in the manifest configuration: "
                + "; ".join(evidence_parts)
                + more_text
                + "."
            ),
            "why_it_matters": "Runtime options that are absent from app JSON cannot be set from the SOAR asset UI, so the connector can silently use hardcoded or insecure fallback behavior.",
            "suggested_fix": "Add matching configuration entries with correct data types/defaults, or remove the Python reads and document the intentionally unsupported behavior.",
        }
    ]


def check_unsafe_logging(full_files: dict[str, str]) -> list[dict[str, Any]]:
    findings = []
    logging_pattern = re.compile(
        r"\b(debug_print|save_progress|send_progress|logger\.(debug|info|warning|error)|logging\.(debug|info|warning|error))\s*\("
    )
    dynamic_sensitive_pattern = re.compile(
        r"(\{[^}]*\b(data|payload|headers?|request_headers|response|resp_json|token_response|self\._headers|self\._state)\b[^}]*\}"
        r"|\((data|payload|headers?|request_headers|response|resp_json|token_response|self\._headers|self\._state)\))"
    )
    for path, text in full_files.items():
        if not path.endswith(".py"):
            continue
        for line_no, line in iter_lines(text):
            lowered = line.lower()
            if not logging_pattern.search(line):
                continue
            if is_safe_status_only_log(line):
                continue
            has_sensitive_word = any(word in lowered for word in SENSITIVE_WORDS)
            has_dynamic_value = (
                "f\"" in line
                or "f'" in line
                or ".format(" in line
                or "%(" in line
                or dynamic_sensitive_pattern.search(line) is not None
            )
            if not has_sensitive_word and not dynamic_sensitive_pattern.search(line):
                continue
            if not has_dynamic_value:
                continue
            findings.append(
                {
                    "title": "Potentially sensitive data is logged",
                    "category": "unsafe_logging",
                    "severity": "high" if any(w in lowered for w in ("token", "secret", "authorization", "password")) else "medium",
                    "confidence": "medium",
                    "file": path,
                    "line": line_no,
                    "evidence": line.strip(),
                    "why_it_matters": "SOAR connector logs can expose tokens, auth headers, request payloads, tenant data, or full API responses.",
                    "suggested_fix": "Remove the log line or redact sensitive fields before logging. Prefer status-only or count-only progress messages.",
                }
            )
    return findings


def check_sensitive_debug_data(full_files: dict[str, str]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    sensitive_debug_pattern = re.compile(
        r"\badd_debug_data\s*\(\s*\{[^}\n]*(?:"
        r"r_text|r_headers|response\.text|response\.content|response\.headers|"
        r"\bheaders?\b|token|authorization|payload|tenant|secret"
        r")[^}\n]*\}",
        re.I,
    )
    for path, text in full_files.items():
        if not path.endswith(".py"):
            continue
        for line_no, line in iter_lines(text):
            if not sensitive_debug_pattern.search(line):
                continue
            findings.append(
                {
                    "title": "Debug data records full response text or headers",
                    "category": "unsafe_logging",
                    "severity": "medium",
                    "confidence": "high",
                    "file": path,
                    "line": line_no,
                    "evidence": line.strip(),
                    "why_it_matters": "SOAR debug data can expose full API responses, tenant intelligence, headers, tokens, or other sensitive operational details.",
                    "suggested_fix": "Do not attach full response bodies or headers to debug data. Keep only safe fields such as status code, request ID, or a redacted error summary.",
                }
            )
    return findings


def is_safe_status_only_log(line: str) -> bool:
    lowered = line.lower()
    if "status_code" not in lowered:
        return False
    risky_terms = (
        "access_token",
        "api_key",
        "authorization",
        "bearer",
        "client_secret",
        "headers",
        "password",
        "payload",
        "refresh_token",
        "response.text",
        "response.content",
        "response.json",
        "secret",
        "tenant",
        "token",
    )
    return not any(term in lowered for term in risky_terms)


def check_read_only_metadata(app_jsons: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    findings = []
    for path, app_json in app_jsons.items():
        for action in app_json.get("actions", []):
            identifier = str(action.get("identifier") or "")
            action_name = str(action.get("action") or "")
            normalized_name = identifier or action_name.replace(" ", "_").lower()
            if not normalized_name.startswith(MUTATING_PREFIXES):
                continue
            if action.get("read_only") is not True:
                continue
            findings.append(
                {
                    "title": "Mutating action is marked read_only",
                    "category": "soar_metadata",
                    "severity": "high",
                    "confidence": "high",
                    "file": path,
                    "line": None,
                    "evidence": f"Action '{action_name or identifier}' has identifier '{identifier}' and read_only=true.",
                    "why_it_matters": "Mutating POST/PATCH/DELETE-style actions should not be advertised as read-only in SOAR metadata.",
                    "suggested_fix": "Set read_only to false for mutating actions and verify the action type matches the behavior.",
                }
            )
    return findings


def check_indicator_parameters_missing_contains(app_jsons: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    contains_by_name = {
        "domain": "domain",
        "hash": "hash",
        "ip": "ip",
        "ipv4": "ip",
        "ipv6": "ip",
        "sha1": "hash",
        "sha256": "hash",
        "md5": "hash",
        "url": "url",
    }
    for path, app_json in app_jsons.items():
        missing: list[tuple[str, str, str]] = []
        for action in app_json.get("actions", []):
            action_name = str(action.get("action") or action.get("identifier") or "action")
            for param in action_parameters(action):
                raw_name = str(param.get("name") or param.get("identifier") or "").strip()
                normalized = raw_name.lower().replace(" ", "_")
                expected = contains_by_name.get(normalized)
                if not expected:
                    continue
                if param.get("contains"):
                    continue
                data_type = str(param.get("data_type") or param.get("type") or "").lower()
                if data_type and data_type not in {"string", "password"}:
                    continue
                missing.append((action_name, raw_name, expected))

        if not missing:
            continue
        examples = ", ".join(f"{action}.{name} -> {contains}" for action, name, contains in missing[:8])
        findings.append(
            {
                "title": "Indicator action parameters are missing contains metadata",
                "category": "soar_metadata",
                "severity": "medium",
                "confidence": "high",
                "file": path,
                "line": None,
                "code_reference": "action parameters contains",
                "evidence": f"Indicator-like parameters have no `contains` metadata: {examples}.",
                "why_it_matters": "`contains` drives typed indicator handling, playbook recommendations, and CEF mapping for common inputs such as hash, IP, URL, and domain.",
                "suggested_fix": "Add the appropriate `contains` value to each indicator parameter, for example hash/hash, ip/ip, url/url, and domain/domain.",
            }
        )
    return findings


def check_app_metadata_basics(review_input: dict[str, Any], app_jsons: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    findings = []
    changed = changed_paths(review_input)
    full_files = review_input.get("full_files") or {}
    invalid_legacy_types = {"phantom", "phantom_app", "legacy", "app", "python"}

    for path, app_json in app_jsons.items():
        if path not in changed:
            continue

        app_type = str(app_json.get("type") or "").strip().lower()
        if not app_type:
            findings.append(
                {
                    "title": "Changed app JSON is missing app type metadata",
                    "category": "soar_metadata",
                    "severity": "medium",
                    "confidence": "medium",
                    "file": path,
                    "line": None,
                    "evidence": "The changed app JSON has no top-level type value.",
                    "why_it_matters": "SOAR metadata controls packaging and compatibility behavior; missing or legacy type values are common connector review failures.",
                    "suggested_fix": "Set the app type to the value required by the current connector conventions for this app.",
                }
            )
        elif app_type in invalid_legacy_types:
            findings.append(
                {
                    "title": "Changed app JSON uses an invalid-looking legacy app type",
                    "category": "soar_metadata",
                    "severity": "medium",
                    "confidence": "medium",
                    "file": path,
                    "line": json_key_line(str(full_files.get(path) or ""), "type"),
                    "evidence": f"Top-level type is '{app_type}'.",
                    "why_it_matters": "Legacy or placeholder app type values can break connector packaging and support classification.",
                    "suggested_fix": "Verify the allowed values in the connector conventions and update the app type accordingly.",
                }
            )

        python_version = str(app_json.get("python_version") or "").strip()
        if not python_version:
            findings.append(
                {
                    "title": "Changed app JSON is missing python_version",
                    "category": "soar_metadata",
                    "severity": "medium",
                    "confidence": "high",
                    "file": path,
                    "line": None,
                    "evidence": "No top-level python_version value is declared.",
                    "why_it_matters": "Connector compatibility checks depend on app JSON declaring the supported Python runtime.",
                    "suggested_fix": "Declare the current supported python_version required by the connector conventions.",
                }
            )
        elif "3.13" not in python_version:
            findings.append(
                {
                    "title": "Changed app JSON python_version may be stale",
                    "category": "soar_metadata",
                    "severity": "medium",
                    "confidence": "medium",
                    "file": path,
                    "line": json_key_line(str(full_files.get(path) or ""), "python_version"),
                    "evidence": f"python_version is '{python_version}', not 3.13.",
                    "why_it_matters": "Python runtime metadata drift is a recurring connector review issue, especially for compatibility changes.",
                    "suggested_fix": "Verify the required runtime in the connector conventions and update python_version if this PR is expected to move the app to the current runtime.",
                }
            )

        app_version = str(app_json.get("app_version") or "").strip()
        product_version = str(app_json.get("product_version") or "").strip()
        if app_version and product_version and app_version == product_version:
            findings.append(
                {
                    "title": "Product version appears to use the SOAR app version",
                    "category": "soar_metadata",
                    "severity": "medium",
                    "confidence": "medium",
                    "file": path,
                    "line": json_key_line(str(full_files.get(path) or ""), "product_version"),
                    "evidence": f"app_version and product_version are both '{app_version}'.",
                    "why_it_matters": "Human connector reviews have caught product_version values that copied the SOAR app version instead of the vendor product/version tested.",
                    "suggested_fix": "Set product_version to the vendor product version tested, or remove/adjust it according to connector metadata conventions.",
                }
            )

        latest_tested_versions = app_json.get("latest_tested_versions")
        latest_values = [value.strip() for value in flatten_json_strings(latest_tested_versions) if value.strip()]
        copied_latest_values = [
            value
            for value in latest_values
            if app_version and (value == app_version or re.search(rf"\b{re.escape(app_version)}\b", value))
        ]
        if copied_latest_values:
            findings.append(
                {
                    "title": "Latest tested version appears to use the SOAR app version",
                    "category": "soar_metadata",
                    "severity": "medium",
                    "confidence": "medium",
                    "file": path,
                    "line": json_key_line(str(full_files.get(path) or ""), "latest_tested_versions"),
                    "evidence": f"latest_tested_versions contains {copied_latest_values[:3]}, matching app_version '{app_version}'.",
                    "why_it_matters": "Human reviews caught latest-tested-version metadata copied from the SOAR app version instead of the vendor product version tested.",
                    "suggested_fix": "Set latest_tested_versions to the vendor product version tested for this app, not the SOAR connector app_version.",
                }
            )
    return findings


def check_version_bump_missing(review_input: dict[str, Any], app_jsons: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    base_files = review_input.get("base_files") or {}
    if not isinstance(base_files, dict):
        return []
    base_jsons = load_app_jsons(base_files)
    if not base_jsons:
        return []

    changed = changed_paths(review_input)
    risky_change = any(
        path.endswith(".py")
        or (path.endswith(".json") and "/" not in path)
        or path in {"pyproject.toml", "uv.lock"}
        for path in changed
    )
    if not risky_change:
        return []

    findings = []
    for path, app_json in app_jsons.items():
        base_json = base_jsons.get(path)
        if not base_json:
            continue
        head_version = app_json.get("app_version")
        base_version = base_json.get("app_version")
        if head_version and base_version and str(head_version) == str(base_version):
            findings.append(
                {
                    "title": "Connector metadata changed without an app_version bump",
                    "category": "soar_metadata",
                    "severity": "medium",
                    "confidence": "medium",
                    "file": path,
                    "line": None,
                    "evidence": f"app_version remains {head_version!r} while code/app metadata/dependencies changed.",
                    "why_it_matters": "Compatibility and packaging changes usually need a version bump so users and release tooling can identify the update.",
                    "suggested_fix": "Bump app_version when the change affects runtime behavior, compatibility, actions, dependencies, or metadata; otherwise document why no bump is needed.",
                }
            )
    return findings


def check_add_data_schema_mismatch(
    full_files: dict[str, str],
    app_jsons: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    findings = []
    actions_by_identifier: dict[str, tuple[str, dict[str, Any]]] = {}
    for json_path, app_json in app_jsons.items():
        for action in app_json.get("actions", []):
            identifier = action.get("identifier")
            if identifier:
                actions_by_identifier[str(identifier)] = (json_path, action)

    if not actions_by_identifier:
        return findings

    for path, text in full_files.items():
        if not path.endswith(".py"):
            continue
        for function_name, start_line, body in extract_handle_functions(text):
            if "add_data(" not in body:
                continue
            action_entry = actions_by_identifier.get(function_name)
            if not action_entry:
                continue
            json_path, action = action_entry
            outputs = action.get("output") or []
            has_data_output = any(
                isinstance(item, dict) and str(item.get("data_path", "")).startswith("action_result.data")
                for item in outputs
            )
            if has_data_output:
                continue
            findings.append(
                {
                    "title": "Code emits action_result.data but app JSON declares no data output",
                    "category": "output_schema_mismatch",
                    "severity": "medium",
                    "confidence": "high",
                    "file": path,
                    "line": start_line,
                    "evidence": f"_handle_{function_name}() calls add_data(), but '{json_path}' action '{action.get('action')}' has no action_result.data.* output paths.",
                    "why_it_matters": "SOAR playbooks, docs, and result views rely on app JSON output paths matching emitted action data.",
                    "suggested_fix": "Declare the emitted action_result.data.* fields in app JSON or stop emitting undeclared data.",
                }
            )
    return findings


def check_summary_schema_mismatch(
    full_files: dict[str, str],
    app_jsons: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    findings = []
    actions_by_identifier: dict[str, tuple[str, dict[str, Any]]] = {}
    for json_path, app_json in app_jsons.items():
        for action in app_json.get("actions", []):
            identifier = action.get("identifier")
            if identifier:
                actions_by_identifier[str(identifier)] = (json_path, action)

    for path, text in full_files.items():
        if not path.endswith(".py"):
            continue
        for function_name, start_line, body in extract_handle_functions(text):
            summary_keys = extract_summary_keys(body)
            if not summary_keys:
                continue
            action_entry = actions_by_identifier.get(function_name)
            if not action_entry:
                continue
            json_path, action = action_entry
            declared_paths = action_output_paths(action)
            missing = sorted(
                key
                for key in summary_keys
                if f"action_result.summary.{key}" not in declared_paths
                and f"summary.{key}" not in declared_paths
            )
            if not missing:
                continue
            findings.append(
                {
                    "title": "Code emits summary fields missing from app JSON output",
                    "category": "output_schema_mismatch",
                    "severity": "medium",
                    "confidence": "medium",
                    "file": path,
                    "line": start_line,
                    "evidence": f"_handle_{function_name}() writes summary keys {missing}, but '{json_path}' does not declare matching action_result.summary.* output paths.",
                    "why_it_matters": "Summary/schema mismatches make SOAR result views and downstream playbooks inconsistent with connector behavior.",
                    "suggested_fix": "Declare the emitted action_result.summary.* paths in app JSON or remove the undeclared summary fields.",
                }
            )
    return findings


def extract_summary_keys(body: str) -> set[str]:
    keys = set(re.findall(r"\bsummary(?:_data)?\s*\[\s*['\"]([A-Za-z0-9_]+)['\"]\s*\]", body))
    for match in re.finditer(r"\bsummary(?:_data)?\.update\s*\(\s*\{(?P<body>.*?)\}\s*\)", body, re.S):
        keys.update(re.findall(r"['\"]([A-Za-z0-9_]+)['\"]\s*:", match.group("body")))
    for match in re.finditer(r"\bupdate_summary\s*\(\s*\{(?P<body>.*?)\}\s*\)", body, re.S):
        keys.update(re.findall(r"['\"]([A-Za-z0-9_]+)['\"]\s*:", match.group("body")))
    return keys


def check_sdk_summary_type_missing(full_files: dict[str, str]) -> list[dict[str, Any]]:
    findings = []
    sdk_markers = ("soar_sdk", "ActionOutput", "OutputField", "summary_type")
    all_text = "\n".join(full_files.values())
    for path, text in full_files.items():
        if not path.endswith(".py") or not any(marker in text for marker in sdk_markers):
            continue
        if "soar.set_summary" not in text and "set_summary(" not in text and "action_result.summary" not in text:
            continue
        if "summary_type" in all_text:
            continue
        lines = text.splitlines()
        for idx, line in enumerate(lines):
            lowered = line.lower()
            if "summary" not in lowered:
                continue
            if "output" not in lowered and "field" not in lowered and "action_result.summary" not in lowered:
                continue
            window = "\n".join(lines[max(0, idx - 3) : min(len(lines), idx + 8)])
            if "summary_type" in window:
                continue
            findings.append(
                {
                    "title": "SDK summary output lacks visible summary_type metadata",
                    "category": "output_schema_mismatch",
                    "severity": "low",
                    "confidence": "low",
                    "file": path,
                    "line": idx + 1,
                    "evidence": window.strip(),
                    "why_it_matters": "SDK actions need summary metadata to line up generated app JSON, docs, and runtime summaries.",
                    "suggested_fix": "If this is an SDK summary output field, add the required summary_type metadata or confirm the generated schema supplies it elsewhere.",
                }
            )
            break
    return findings


def check_sdk_app_contracts(
    review_input: dict[str, Any],
    full_files: dict[str, str],
) -> list[dict[str, Any]]:
    inventory = build_sdk_review_inventory(review_input, full_files)
    if not inventory["is_sdk_app"]:
        return []

    findings: list[dict[str, Any]] = []
    python = inventory["python"]
    if not (
        inventory["is_sdk_migration"]
        or inventory["sdk_markers"].get("has_pyproject_tool")
        or python["app_instances"]
    ):
        return []
    app_instances = python["app_instances"]
    target_file = sdk_target_file(inventory)

    if inventory["is_sdk_migration"] and not inventory["main_module_file_present"]:
        findings.append(
            {
                "title": "SDK pyproject main_module points to a file that is not present in the PR context",
                "category": "soar_metadata",
                "severity": "high",
                "confidence": "medium",
                "file": "pyproject.toml",
                "line": find_line(full_files.get("pyproject.toml", ""), "main_module"),
                "code_reference": inventory["main_module"],
                "evidence": f"`[tool.soar.app].main_module` is `{inventory['main_module']}`, but `{inventory['main_module_file']}` was not available as current SDK source.",
                "why_it_matters": "SDK package/manifest generation imports the App instance from main_module; a wrong or missing path can prevent the connector from building or running.",
                "suggested_fix": "Point `[tool.soar.app].main_module` at the module that defines the single App instance, or add the missing module.",
            }
        )

    if inventory["main_module_file_present"] and not inventory["main_app_instance_present"]:
        findings.append(
            {
                "title": "SDK main_module does not expose the configured App instance",
                "category": "soar_metadata",
                "severity": "high",
                "confidence": "high",
                "file": inventory["main_module_file"],
                "line": 1,
                "code_reference": inventory["main_module_app_name"],
                "evidence": f"`pyproject.toml` points to `{inventory['main_module']}`, but no `App(...)` assignment named `{inventory['main_module_app_name']}` was found in `{inventory['main_module_file']}`.",
                "why_it_matters": "The SDK manifest processor imports this object to build the app manifest, so the package can fail before any action runs.",
                "suggested_fix": "Create exactly one `App(...)` instance with the configured name or update `main_module` to the actual App object.",
            }
        )

    if len(app_instances) > 1:
        first = app_instances[1]
        findings.append(
            {
                "title": "SDK app defines more than one App instance",
                "category": "soar_metadata",
                "severity": "high",
                "confidence": "high",
                "file": first["file"],
                "line": first["line"],
                "evidence": f"Found App instances {[item['name'] for item in app_instances]}; SDK apps are expected to expose one single App instance.",
                "why_it_matters": "Multiple App instances can register different action sets and make generated metadata depend on the imported object rather than the intended app.",
                "suggested_fix": "Keep one App instance and register all actions, views, webhooks, and asset_cls on that object.",
            }
        )

    main_app = next(
        (
            item
            for item in app_instances
            if item["file"] == inventory["main_module_file"]
            and item["name"] == inventory["main_module_app_name"]
        ),
        app_instances[0] if app_instances else None,
    )
    if main_app and python["asset_field_names"] and not main_app.get("asset_cls"):
        findings.append(
            {
                "title": "SDK App instance does not register the asset configuration class",
                "category": "soar_metadata",
                "severity": "high",
                "confidence": "high",
                "file": main_app["file"],
                "line": main_app["line"],
                "evidence": f"Asset fields {python['asset_field_names'][:8]} are defined, but the App(...) call has no `asset_cls=` argument.",
                "why_it_matters": "The SDK generates the asset configuration schema from `asset_cls`; omitting it can produce an empty config form and fail actions that expect asset fields.",
                "suggested_fix": "Pass the app's `BaseAsset` subclass through `asset_cls=` in the App(...) initialization.",
            }
        )

    if not python["test_connectivity"]:
        findings.append(
            {
                "title": "SDK app does not register test connectivity",
                "category": "soar_metadata",
                "severity": "high",
                "confidence": "high",
                "file": target_file,
                "line": 1,
                "code_reference": "@app.test_connectivity",
                "evidence": "No function decorated or registered with `app.test_connectivity()` was found in the current SDK source.",
                "why_it_matters": "The SDK documentation expects every app to implement test connectivity so users can validate connection, authentication, and permissions before running actions.",
                "suggested_fix": "Add exactly one `@app.test_connectivity()` function that exercises the real auth/API path and raises `ActionFailure` on failure.",
            }
        )
    elif python["test_connectivity_count"] > 1:
        extra = python["test_connectivity"][1]
        findings.append(
            {
                "title": "SDK app registers more than one test connectivity action",
                "category": "soar_metadata",
                "severity": "high",
                "confidence": "high",
                "file": extra["file"],
                "line": extra["line"],
                "evidence": f"Found {python['test_connectivity_count']} `@app.test_connectivity()` functions.",
                "why_it_matters": "The SDK test connectivity decorator only allows one registration; duplicates fail during import.",
                "suggested_fix": "Keep one test connectivity function and move any helper logic into undecorated functions.",
            }
        )

    for test in python["test_connectivity"]:
        return_annotation = test.get("return_annotation")
        if return_annotation and return_annotation not in {"None", "NoneType"}:
            findings.append(
                {
                    "title": "SDK test connectivity has a non-None return annotation",
                    "category": "soar_metadata",
                    "severity": "high",
                    "confidence": "high",
                    "file": test["file"],
                    "line": test["line"],
                    "code_reference": test["function"],
                    "evidence": f"{test['function']} is annotated as returning `{return_annotation}`.",
                    "why_it_matters": "The SDK test connectivity decorator rejects non-None return annotations during registration.",
                    "suggested_fix": "Annotate test connectivity as `-> None` and raise `ActionFailure` for failures.",
                }
            )
        if test.get("returns_value"):
            findings.append(
                {
                    "title": "SDK test connectivity returns a value",
                    "category": "soar_metadata",
                    "severity": "high",
                    "confidence": "high",
                    "file": test["file"],
                    "line": test["line"],
                    "code_reference": test["function"],
                    "evidence": f"{test['function']} contains a `return` statement with a value.",
                    "why_it_matters": "A successful SDK test connectivity action should return None; returning a value is treated as a runtime error.",
                    "suggested_fix": "Remove the returned value and raise `ActionFailure` when connectivity, authentication, or permission checks fail.",
                }
            )
        if "params" in test.get("args", []) or any("Params" in str(value) for value in test.get("arg_annotations", {}).values()):
            findings.append(
                {
                    "title": "SDK test connectivity declares action parameters",
                    "category": "soar_metadata",
                    "severity": "high",
                    "confidence": "high",
                    "file": test["file"],
                    "line": test["line"],
                    "code_reference": test["function"],
                    "evidence": f"{test['function']} arguments are {test.get('args', [])}.",
                    "why_it_matters": "SDK test connectivity takes no action params; adding params can break runtime invocation and gives users no way to provide them.",
                    "suggested_fix": "Remove the params argument and use only injected `asset` and/or `soar` arguments as needed.",
                }
            )

    for action in python["action_registrations"]:
        registration_type = action.get("registration_type")
        signature_file = str(action.get("implementation_file") or action["file"])
        signature_line = int(action.get("implementation_line") or action["line"])
        signature_visible = bool(action.get("signature_visible"))
        has_output_type = bool(action.get("return_annotation") or action.get("output_class"))
        has_params_type = bool(action.get("first_param_annotation") or action.get("params_class"))
        registration_label = (
            "app.register_action(...)"
            if registration_type == "register_action"
            else f"@app.{registration_type}()"
        )
        if registration_type in {"action", "make_request", "register_action"} and signature_visible and not has_output_type:
            findings.append(
                {
                    "title": "SDK action is missing a return type annotation",
                    "category": "soar_metadata",
                    "severity": "high",
                    "confidence": "high",
                    "file": signature_file,
                    "line": signature_line,
                    "code_reference": action.get("function") or action.get("identifier"),
                    "evidence": f"`{action.get('function')}` is registered with `{registration_label}` but has no return annotation or `output_class=`.",
                    "why_it_matters": "The SDK uses the return annotation or explicit output_class to validate the action and generate output datapaths; missing output typing fails action registration.",
                    "suggested_fix": "Annotate the action with an `ActionOutput`-derived return type, or a supported list/Iterator/AsyncGenerator of that output type.",
                }
            )
        if registration_type in {"action", "register_action"} and signature_visible and not has_params_type:
            findings.append(
                {
                    "title": "SDK action params argument is missing a type annotation",
                    "category": "soar_metadata",
                    "severity": "high",
                    "confidence": "high",
                    "file": signature_file,
                    "line": signature_line,
                    "code_reference": action.get("function") or action.get("identifier"),
                    "evidence": f"`{action.get('function')}` is registered with `{registration_label}` but its first argument has no Params-derived annotation or `params_class=`.",
                    "why_it_matters": "The SDK validates the action params model at registration and uses it to generate input metadata.",
                    "suggested_fix": "Annotate the first argument with `Params` or an app-specific subclass of `Params`.",
                }
            )
        if registration_type == "make_request" and signature_visible:
            arg_annotations = action.get("arg_annotations") or {}
            make_request_param_names = [
                name
                for name, annotation in arg_annotations.items()
                if "MakeRequestParams" in str(annotation)
            ]
            first_annotation = str(action.get("first_param_annotation") or "")
            if len(make_request_param_names) != 1:
                findings.append(
                    {
                        "title": "SDK make_request action does not declare exactly one MakeRequestParams parameter",
                        "category": "soar_metadata",
                        "severity": "high",
                        "confidence": "high",
                        "file": signature_file,
                        "line": signature_line,
                        "code_reference": action.get("function") or "make_request",
                        "evidence": f"`{action.get('function')}` has MakeRequestParams annotations on {make_request_param_names or 'no parameters'}.",
                        "why_it_matters": "The SDK make_request decorator validates that exactly one parameter is typed as MakeRequestParams or a subclass before registering the special make request action.",
                        "suggested_fix": "Declare one request params argument typed as `MakeRequestParams` or a subclass, and remove any extra MakeRequestParams-typed parameters.",
                    }
                )
            elif "MakeRequestParams" not in first_annotation:
                findings.append(
                    {
                        "title": "SDK make_request action does not use MakeRequestParams as the first argument",
                        "category": "soar_metadata",
                        "severity": "high",
                        "confidence": "high",
                        "file": signature_file,
                        "line": signature_line,
                        "code_reference": action.get("function") or "make_request",
                        "evidence": f"`{action.get('function')}` first parameter annotation is `{action.get('first_param_annotation')}`, while `{make_request_param_names[0]}` is the MakeRequestParams argument.",
                        "why_it_matters": "The SDK wrapper passes validated request params as the first positional argument when executing make_request; putting asset/soar first can bind the wrong object at runtime.",
                        "suggested_fix": "Move the MakeRequestParams argument to the first position, for example `def make_request(params: MakeRequestParams, asset: Asset) -> MakeRequestOutput:`.",
                    }
                )
        if registration_type == "on_poll" and "OnPollParams" not in str(action.get("first_param_annotation") or ""):
            findings.append(
                {
                    "title": "SDK on_poll action does not use OnPollParams",
                    "category": "polling_checkpoint",
                    "severity": "high",
                    "confidence": "high",
                    "file": action["file"],
                    "line": action["line"],
                    "code_reference": action.get("function") or "on_poll",
                    "evidence": f"`on_poll` first parameter annotation is `{action.get('first_param_annotation')}`.",
                    "why_it_matters": "The SDK on_poll decorator expects the standard OnPollParams model so scheduled and manual poll inputs are parsed correctly.",
                    "suggested_fix": "Annotate the first on_poll argument as `OnPollParams`.",
                }
            )
        if registration_type == "on_es_poll" and "OnESPollParams" not in str(action.get("first_param_annotation") or ""):
            findings.append(
                {
                    "title": "SDK on_es_poll action does not use OnESPollParams",
                    "category": "polling_checkpoint",
                    "severity": "high",
                    "confidence": "high",
                    "file": action["file"],
                    "line": action["line"],
                    "code_reference": action.get("function") or "on_es_poll",
                    "evidence": f"`on_es_poll` first parameter annotation is `{action.get('first_param_annotation')}`.",
                    "why_it_matters": "The SDK on_es_poll decorator expects OnESPollParams so ES polling inputs are parsed correctly.",
                    "suggested_fix": "Annotate the first on_es_poll argument as `OnESPollParams`.",
                }
            )
        identifier = str(action.get("identifier") or action.get("function") or "")
        if registration_type in {"action", "register_action"} and is_mutating_identifier(identifier) and action.get("read_only") is not False:
            findings.append(
                {
                    "title": "Mutating SDK action relies on the read_only=True default",
                    "category": "soar_metadata",
                    "severity": "high",
                    "confidence": "high",
                    "file": action["file"],
                    "line": action["line"],
                    "code_reference": identifier,
                    "evidence": f"Action `{identifier}` looks mutating but is registered without `read_only=False`.",
                    "why_it_matters": "SDK App.action/register_action default to read_only=True, so create/update/delete actions can be advertised incorrectly in SOAR metadata.",
                    "suggested_fix": "Set `read_only=False` on mutating SDK action registrations and verify the action type matches the behavior.",
                }
            )

    for field in python["unsupported_param_fields"]:
        findings.append(
            {
                "title": "SDK Params field uses an unsupported list annotation",
                "category": "validation",
                "severity": "high",
                "confidence": "high",
                "file": field["file"],
                "line": field["line"],
                "code_reference": f"{field['class_name']}.{field['field']}",
                "evidence": f"{field['class_name']}.{field['field']} is annotated as `{field['annotation']}`.",
                "why_it_matters": "The SDK Params serializer rejects list annotations; action parameters must be scalar strings, booleans, or numbers even when `allow_list=True` is used for the manifest.",
                "suggested_fix": "Use a scalar Params type and validate/split allow-list input inside the action if multiple values are supported.",
            }
        )

    for field in python["asset_fields"]:
        if field.get("reserved"):
            findings.append(
                {
                    "title": "SDK asset model uses a reserved SOAR config field",
                    "category": "soar_metadata",
                    "severity": "high",
                    "confidence": "high",
                    "file": field["file"],
                    "line": field["line"],
                    "code_reference": field["name"],
                    "evidence": f"`{field['name']}` is reserved by the platform or starts with `_reserved_`.",
                    "why_it_matters": "The SDK BaseAsset validator rejects reserved platform config fields during asset validation.",
                    "suggested_fix": "Rename the asset field or use a non-reserved alias that does not collide with platform app configuration keys.",
                }
            )
        if field.get("sensitive") and not annotation_is_string_like_text(field.get("annotation")):
            findings.append(
                {
                    "title": "SDK sensitive asset field is not typed as string",
                    "category": "validation",
                    "severity": "high",
                    "confidence": "high",
                    "file": field["file"],
                    "line": field["line"],
                    "code_reference": field["name"],
                    "evidence": f"`{field['name']}` is sensitive but annotated as `{field.get('annotation')}`.",
                    "why_it_matters": "The SDK asset serializer only emits sensitive fields as password fields when the field is typed as `str`.",
                    "suggested_fix": "Change the asset field annotation to `str` or `str | None`.",
                }
            )
        if field.get("is_file") and not annotation_is_string_like_text(field.get("annotation")):
            findings.append(
                {
                    "title": "SDK file asset field is not typed as string",
                    "category": "validation",
                    "severity": "high",
                    "confidence": "high",
                    "file": field["file"],
                    "line": field["line"],
                    "code_reference": field["name"],
                    "evidence": f"`{field['name']}` is an AssetField(is_file=True) but annotated as `{field.get('annotation')}`.",
                    "why_it_matters": "The SDK asset serializer requires file-upload asset fields to be typed as `str`.",
                    "suggested_fix": "Change the file asset field annotation to `str`.",
                }
            )

    for field in python["param_fields"]:
        if field.get("sensitive") and not annotation_is_string_like_text(field.get("annotation")):
            findings.append(
                {
                    "title": "SDK sensitive action parameter is not typed as string",
                    "category": "validation",
                    "severity": "high",
                    "confidence": "high",
                    "file": field["file"],
                    "line": field["line"],
                    "code_reference": f"{field['class_name']}.{field['field']}",
                    "evidence": f"`{field['class_name']}.{field['field']}` is sensitive but annotated as `{field.get('annotation')}`.",
                    "why_it_matters": "The SDK Params serializer only emits sensitive parameters as password fields when the parameter is typed as `str`.",
                    "suggested_fix": "Change the parameter annotation to `str` or `str | None`.",
                }
            )

    for field in python["invalid_make_request_param_fields"]:
        findings.append(
            {
                "title": "SDK MakeRequestParams subclass defines unsupported fields",
                "category": "validation",
                "severity": "high",
                "confidence": "high",
                "file": field["file"],
                "line": field["line"],
                "code_reference": f"{field['class_name']}.{field['field']}",
                "evidence": f"`{field['class_name']}` defines `{field['field']}`, which is not a supported MakeRequestParams field.",
                "why_it_matters": "The SDK validates MakeRequestParams subclasses against a fixed set of request fields.",
                "suggested_fix": "Use only the supported make-request fields: http_method, endpoint, headers, query_parameters, body, timeout, and verify_ssl.",
            }
        )

    return findings


def check_sdk_migration_parity(
    review_input: dict[str, Any],
    full_files: dict[str, str],
) -> list[dict[str, Any]]:
    inventory = build_sdk_review_inventory(review_input, full_files)
    if not inventory["is_sdk_migration"]:
        return []

    findings: list[dict[str, Any]] = []
    legacy = inventory["legacy"]
    python = inventory["python"]
    target_file = sdk_target_file(inventory)
    target_line = 1
    if python["app_instances"]:
        target_file = python["app_instances"][0]["file"]
        target_line = python["app_instances"][0]["line"]

    legacy_fields = set(legacy["config_fields"])
    sdk_fields = set(python["asset_field_names"])
    missing_fields = sorted(legacy_fields - sdk_fields)
    if missing_fields:
        findings.append(
            {
                "title": "Legacy asset configuration fields are missing from the SDK asset model",
                "category": "soar_metadata",
                "severity": "high",
                "confidence": "high",
                "file": target_file,
                "line": target_line,
                "code_reference": "Asset(BaseAsset)",
                "evidence": f"Legacy manifest fields {missing_fields[:10]} are not present in SDK Asset fields {sorted(sdk_fields)[:10]}.",
                "why_it_matters": "SDK conversion should preserve asset configuration semantics; missing fields can break existing assets and hide required credentials or options from users.",
                "suggested_fix": "Add the missing fields to the SDK `BaseAsset` model, using `AssetField(alias=...)` when the Python name has to differ from the manifest key.",
            }
        )

    legacy_actions = set(legacy["action_identifiers"])
    sdk_actions = set(python["registered_action_identifiers"])
    missing_actions = sorted(legacy_actions - sdk_actions)
    if missing_actions:
        findings.append(
            {
                "title": "Legacy actions are not registered in the SDK app",
                "category": "soar_metadata",
                "severity": "high",
                "confidence": "high",
                "file": target_file,
                "line": target_line,
                "code_reference": "SDK action registrations",
                "evidence": f"Legacy action identifiers {missing_actions[:12]} are absent from current SDK registrations {sorted(sdk_actions)[:12]}.",
                "why_it_matters": "An SDK migration that drops action registrations removes user-facing connector behavior and can break playbooks after upgrade.",
                "suggested_fix": "Register migrated implementations for the missing action identifiers or document and release-note intentional removals.",
            }
        )

    has_view_handler_reference = python["view_handler_count"] > 0 or any(
        item.get("view_handler") for item in python["action_registrations"]
    )
    if legacy["custom_view_actions"] and not has_view_handler_reference:
        findings.append(
            {
                "title": "Legacy custom views were not visibly migrated to SDK view handlers",
                "category": "soar_metadata",
                "severity": "medium",
                "confidence": "high",
                "file": target_file,
                "line": target_line,
                "code_reference": "view_handler",
                "evidence": f"Legacy custom-view actions {legacy['custom_view_actions'][:8]} exist, but no `@app.view_handler` or `view_handler=` registration was found.",
                "why_it_matters": "The SDK converter does not automatically migrate custom views; without a replacement, users lose the custom action result widgets after upgrade.",
                "suggested_fix": "Implement SDK view handlers and attach them to the migrated actions with `view_handler=` and templates/components as needed.",
            }
        )

    if (legacy["has_rest_handler"] or legacy["has_webhooks"]) and not python["enable_webhooks_count"] and not python["webhook_count"]:
        legacy_bits = []
        if legacy["has_rest_handler"]:
            legacy_bits.append("custom REST handler")
        if legacy["has_webhooks"]:
            legacy_bits.append("webhook handler")
        findings.append(
            {
                "title": "Legacy REST/webhook handling was not visibly migrated to SDK webhooks",
                "category": "soar_metadata",
                "severity": "medium",
                "confidence": "high",
                "file": target_file,
                "line": target_line,
                "code_reference": "app.enable_webhooks",
                "evidence": f"Legacy manifest contains {', '.join(legacy_bits)}, but the SDK source has no `app.enable_webhooks()` or `@app.webhook(...)` registration.",
                "why_it_matters": "The SDK converter does not support custom REST migration and requires custom REST handlers to be converted to webhooks.",
                "suggested_fix": "Enable SDK webhooks and add route handlers for the migrated callback/custom REST behavior.",
            }
        )

    has_summary_metadata = any(item.get("summary_type") for item in python["action_registrations"])
    current_text = "\n".join(full_files.values())
    if legacy["summary_actions"] and not has_summary_metadata and "set_summary" not in current_text:
        findings.append(
            {
                "title": "Legacy action summaries were not visibly migrated to SDK summary metadata",
                "category": "output_schema_mismatch",
                "severity": "medium",
                "confidence": "high",
                "file": target_file,
                "line": target_line,
                "code_reference": "summary_type",
                "evidence": f"Legacy actions {legacy['summary_actions'][:8]} declare action_result.summary paths, but no SDK `summary_type=` or `soar.set_summary(...)` usage was found.",
                "why_it_matters": "The SDK converter does not automatically migrate action summaries; dropping them can break dashboards, playbooks, and user-facing result summaries.",
                "suggested_fix": "Define `ActionOutput` summary models, pass them via `summary_type=`, and set summaries in the migrated actions.",
            }
        )

    findings.extend(check_sdk_output_datapath_regressions(legacy, python, full_files, target_file, target_line))
    findings.extend(check_sdk_summary_field_regressions(legacy, python, full_files, target_file, target_line))
    findings.extend(check_sdk_migration_missing_regression_tests(review_input, full_files, target_file, target_line))

    return findings


def check_sdk_pat_only_auth_regression(
    review_input: dict[str, Any],
    full_files: dict[str, str],
) -> list[dict[str, Any]]:
    inventory = build_sdk_review_inventory(review_input, full_files)
    if not inventory["is_sdk_app"]:
        return []

    asset_fields = set(inventory["python"].get("asset_field_names") or [])
    legacy_or_documented_fields = sorted(asset_fields & {"client_id", "client_secret", "username", "password"})
    if not legacy_or_documented_fields:
        return []

    python_files = {
        path: text
        for path, text in full_files.items()
        if path.endswith(".py") and not is_test_path(path)
    }
    code_text = "\n".join(python_files.values())
    lowered_code = code_text.lower()
    non_pat_runtime_refs = [
        f"asset.{field}"
        for field in legacy_or_documented_fields
        if f"asset.{field}" in lowered_code
    ]
    oauth_runtime_markers = (
        "authorizationcodeflow",
        "clientcredentialsflow",
        "token_endpoint",
        "refresh_token",
        "oauth/token",
    )
    if non_pat_runtime_refs or any(marker in lowered_code for marker in oauth_runtime_markers):
        return []
    if "asset.personal_access_token" not in lowered_code:
        return []

    docs_text = "\n".join(
        str(full_files.get(path) or "")
        for path in ("README.md", "manual_readme_content.md", "release_notes/unreleased.md")
    ).lower()
    pr_text = f"{review_input.get('pr', {}).get('title', '')}\n{review_input.get('pr', {}).get('body', '')}".lower()
    doc_claims_oauth = any(
        term in f"{docs_text}\n{pr_text}"
        for term in ("oauth", "client id", "client_id", "client secret", "client_secret", "username/password")
    )
    app_text = full_files.get(inventory.get("main_module_file") or "src/app.py", "")
    target_file = inventory.get("main_module_file") or "src/app.py"
    target_line = find_line(app_text, "personal_access_token") or find_line(app_text, "test_connectivity") or 1
    doc_evidence = " Docs or release notes still mention OAuth/client credentials." if doc_claims_oauth else ""
    return [
        {
            "title": "SDK auth path exposes non-PAT credential fields but only executes PAT auth",
            "category": "api_auth_correctness",
            "severity": "high",
            "confidence": "high",
            "file": target_file,
            "line": target_line,
            "code_reference": "Asset credentials / resolve_auth",
            "evidence": (
                f"The SDK asset model exposes {legacy_or_documented_fields}, but current Python code only "
                "references `asset.personal_access_token` at runtime and no OAuth/client-credential token "
                f"exchange path was found.{doc_evidence}"
            ),
            "why_it_matters": "Existing assets or users following the docs can configure OAuth/client credentials that test connectivity and actions immediately reject after migration.",
            "suggested_fix": "Either implement the non-PAT auth paths from the legacy connector in test_connectivity and resolve_auth, or remove those asset fields/docs and explicitly release-note the breaking change.",
        }
    ]


def check_sdk_issue_number_validation_regressions(
    review_input: dict[str, Any],
    full_files: dict[str, str],
) -> list[dict[str, Any]]:
    inventory = build_sdk_review_inventory(review_input, full_files)
    if not inventory["is_sdk_migration"]:
        return []

    affected: list[tuple[str, int]] = []
    pattern = re.compile(r"int\s*\(\s*params\.issue_number\s*\)")
    validation_terms = (
        "_validate_integer",
        "validate_integer",
        "validate_issue_number",
        "positive integer",
        "must be positive",
        "must be a positive",
        "issue_number <= 0",
        "issue_number < 1",
        "params.issue_number <= 0",
        "params.issue_number < 1",
        ".is_integer()",
        "ge=1",
        "gt=0",
    )
    for path, text in full_files.items():
        if not path.endswith(".py") or "params.issue_number" not in text:
            continue
        lowered = text.lower()
        if any(term in lowered for term in validation_terms):
            continue
        for match in pattern.finditer(text):
            affected.append((path, text[: match.start()].count("\n") + 1))
            break

    if not affected:
        return []
    first_path, first_line = affected[0]
    examples = ", ".join(f"{path}:{line}" for path, line in affected[:8])
    return [
        {
            "title": "SDK migration drops positive-integer validation for issue_number",
            "category": "validation",
            "severity": "high",
            "confidence": "high",
            "file": first_path,
            "line": first_line,
            "code_reference": "params.issue_number",
            "evidence": f"`params.issue_number` is coerced with `int(...)` without a visible positive-integer validation guard in {examples}.",
            "why_it_matters": "Values like 0, -1, or 1.9 can reach GitHub URLs or be silently truncated instead of failing with a clear SOAR validation error, which changes legacy connector behavior.",
            "suggested_fix": "Restore the legacy positive-integer validator before building the GitHub endpoint for every issue-number action, and raise ActionFailure for non-integer or non-positive input.",
        }
    ]


def check_sdk_make_request_dynamic_output_serialization(full_files: dict[str, str]) -> list[dict[str, Any]]:
    for path, text in full_files.items():
        if not path.endswith(".py"):
            continue
        lowered = text.lower()
        if "@app.make_request" not in text or "object.__setattr__" not in text:
            continue
        if "response_body" not in lowered or "json" not in lowered:
            continue
        line = find_line(text, "object.__setattr__") or find_line(text, "from_response") or 1
        return [
            {
                "title": "make request promotes JSON fields with object.__setattr__ that will not serialize",
                "category": "output_schema_mismatch",
                "severity": "high",
                "confidence": "high",
                "file": path,
                "line": line,
                "code_reference": "MakeRequestOutput.from_response",
                "evidence": "`@app.make_request` output construction uses `object.__setattr__` to add response JSON keys after the Pydantic output model is created.",
                "why_it_matters": "Attributes added this way are not reliable generated output datapaths and can be omitted from serialized action_result.data, so playbooks may only see status_code/response_body despite docs or code claiming promoted fields.",
                "suggested_fix": "Use explicit SDK output fields for promoted data, configure/populate a supported extra-field model path, or stop claiming dynamic promoted fields and keep the raw JSON under response_body.",
            }
        ]
    return []


def check_sdk_make_request_missing_tests(
    review_input: dict[str, Any],
    full_files: dict[str, str],
) -> list[dict[str, Any]]:
    inventory = build_sdk_review_inventory(review_input, full_files)
    if not inventory["is_sdk_app"]:
        return []

    make_request_actions = [
        action
        for action in inventory["python"].get("action_registrations", [])
        if action.get("registration_type") == "make_request" or action.get("identifier") == "make_request"
    ]
    if not make_request_actions:
        return []

    test_files = {
        path: text
        for path, text in full_files.items()
        if is_test_path(path)
    }
    if any("make_request" in text or "make request" in text.lower() for text in test_files.values()):
        return []

    action = make_request_actions[0]
    file_path = str(action.get("implementation_file") or action.get("file") or "src/app.py")
    line = int(action.get("implementation_line") or action.get("line") or 1)
    detail = "No checked-in test file was found." if not test_files else "Checked-in tests do not mention `make_request`."
    return [
        {
            "title": "`make request` action has no focused test coverage",
            "category": "missing_tests",
            "severity": "high",
            "confidence": "high",
            "file": file_path,
            "line": line,
            "code_reference": action.get("function") or "make_request",
            "evidence": f"The SDK app registers a `make request` action. {detail}",
            "why_it_matters": "`make request` usually contains custom endpoint validation, auth/header merging, timeout/SSL handling, error mapping, and output serialization; migration bugs in this action can break broad user workflows.",
            "suggested_fix": "Add mocked unit tests for `make request` covering endpoint validation, JSON/query/header parsing, auth behavior, HTTP errors, timeout/verify_ssl handling, and the serialized action_result.data shape.",
        }
    ]


def check_make_request_security_contracts(
    review_input: dict[str, Any],
    full_files: dict[str, str],
) -> list[dict[str, Any]]:
    inventory = build_sdk_review_inventory(review_input, full_files)
    if not inventory["is_sdk_app"]:
        return []

    actions = [
        action
        for action in inventory["python"].get("action_registrations", [])
        if action.get("registration_type") == "make_request" or action.get("identifier") == "make_request"
    ]
    if not actions:
        return []

    findings: list[dict[str, Any]] = []
    entries = make_request_function_entries(actions, full_files)

    verify_default = make_request_verify_ssl_default_false(full_files)
    if verify_default:
        file_path, line, evidence = verify_default
        findings.append(
            {
                "title": "`make request` verify_ssl defaults to false",
                "category": "api_auth_correctness",
                "severity": "high",
                "confidence": "high",
                "file": file_path,
                "line": line,
                "code_reference": "verify_ssl",
                "evidence": evidence,
                "why_it_matters": "Arbitrary make-request actions are commonly used for live API troubleshooting; silently defaulting TLS verification off weakens every request made through that action.",
                "suggested_fix": "Default `verify_ssl` to true, honor the asset-level TLS setting when one exists, and require an explicit user choice before disabling certificate verification.",
            }
        )

    for entry in entries:
        body = entry["body"]
        lowered = body.lower()
        if (
            ("authorization" in lowered or "api-key" in lowered or "api_key" in lowered or "bearer " in lowered)
            and (
                re.search(r"\bheaders\.update\s*\(\s*params\.headers", body)
                or re.search(r"\{\s*\*\*headers\s*,\s*\*\*params\.headers", body)
                or re.search(r"\bheaders\s*=\s*headers\s*\|\s*params\.headers", body)
            )
        ):
            findings.append(
                {
                    "title": "`make request` lets user headers override connector auth headers",
                    "category": "api_auth_correctness",
                    "severity": "high",
                    "confidence": "high",
                    "file": entry["file"],
                    "line": line_in_body(entry, "headers.update") or line_in_body(entry, "params.headers") or entry["line"],
                    "code_reference": entry["function"],
                    "evidence": "The make-request handler builds connector auth headers and then merges `params.headers` after them.",
                    "why_it_matters": "A user-supplied Authorization or API-key header can replace the connector's authenticated header, changing the auth context or leaking confusing failures through a generic make-request path.",
                    "suggested_fix": "Merge user headers first and then apply connector auth headers, or explicitly reject user-supplied auth header names for make request.",
                }
            )
            break

    for entry in entries:
        body = entry["body"]
        if not make_request_uses_unscoped_endpoint(body):
            continue
        findings.append(
            {
                "title": "`make request` endpoint is not product-scoped",
                "category": "api_auth_correctness",
                "severity": "medium",
                "confidence": "high",
                "file": entry["file"],
                "line": line_in_body(entry, "params.endpoint") or entry["line"],
                "code_reference": entry["function"],
                "evidence": "`params.endpoint` is passed as the request URL without visible base-URL joining or absolute-URL rejection.",
                "why_it_matters": "A connector make-request action should be constrained to the product API boundary; sending arbitrary absolute URLs through the connector auth/session path can create surprising SSRF-like behavior or incorrect auth routing.",
                "suggested_fix": "Require relative endpoints, reject absolute URLs, and join the path to the connector's product base URL before issuing the request.",
            }
        )
        break

    for entry in entries:
        body = entry["body"]
        if not make_request_parse_error_echoes_user_secrets(body):
            continue
        findings.append(
            {
                "title": "`make request` parse errors can echo user-supplied secrets",
                "category": "unsafe_logging",
                "severity": "medium",
                "confidence": "high",
                "file": entry["file"],
                "line": line_in_body(entry, "params.headers") or line_in_body(entry, "params.body") or entry["line"],
                "code_reference": entry["function"],
                "evidence": "The make-request parser formats raw user headers/body into an exception or action-result message.",
                "why_it_matters": "Make-request headers and bodies often contain bearer tokens, API keys, cookies, or tenant data; parse errors should not echo them back into SOAR logs or action messages.",
                "suggested_fix": "Report only the invalid field name and parse error class, and redact or omit the raw header/body value from user-facing messages.",
            }
        )
        break

    return findings


def make_request_function_entries(
    actions: list[dict[str, Any]],
    full_files: dict[str, str],
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for action in actions:
        file_path = str(action.get("implementation_file") or action.get("file") or "")
        function_name = str(action.get("function") or "")
        if not file_path or not function_name:
            continue
        text = full_files.get(file_path, "")
        body = function_body_text(text, function_name)
        if not body:
            continue
        entries.append(
            {
                "file": file_path,
                "line": int(action.get("implementation_line") or action.get("line") or 1),
                "function": function_name,
                "body": body,
            }
        )
    return entries


def make_request_verify_ssl_default_false(full_files: dict[str, str]) -> tuple[str, int, str] | None:
    pattern = re.compile(
        r"verify_ssl\s*:\s*[^=\n]+=\s*(?:Param\s*\([^)]*default\s*=\s*False|False\b)",
        re.S,
    )
    call_pattern = re.compile(r"verify_ssl\s*=\s*Param\s*\([^)]*default\s*=\s*False", re.S)
    for path, text in full_files.items():
        if not path.endswith(".py") or "verify_ssl" not in text:
            continue
        match = pattern.search(text) or call_pattern.search(text)
        if not match:
            continue
        line = text[: match.start()].count("\n") + 1
        return path, line, line_from_match(text, match.start())
    return None


def make_request_uses_unscoped_endpoint(body: str) -> bool:
    if not re.search(r"\bparams\.endpoint\b", body):
        return False
    direct_patterns = (
        r"\burl\s*=\s*params\.endpoint\b",
        r"\bendpoint\s*=\s*params\.endpoint\b",
        r"\b(?:requests|httpx)\.(?:get|post|put|patch|delete|request)\s*\([^)]*params\.endpoint",
        r"\b[A-Za-z_][A-Za-z0-9_]*\.(?:get|post|put|patch|delete|request)\s*\([^)]*params\.endpoint",
    )
    if not any(re.search(pattern, body, flags=re.S) for pattern in direct_patterns):
        return False
    safe_terms = ("urljoin(", "urlparse(", "base_url", "asset.base_url", "startswith('/')", "startswith(\"/\")")
    return not any(term in body for term in safe_terms)


def make_request_parse_error_echoes_user_secrets(body: str) -> bool:
    if not any(term in body for term in ("params.headers", "params.body", "params.query_parameters")):
        return False
    secret_field_pattern = r"params\.(?:headers|body|query_parameters)"
    return bool(
        re.search(r"(?:ActionFailure|ValueError|set_status|raise)\s*\([^)]*" + secret_field_pattern, body, flags=re.S)
        or re.search(r"f[\"'][^\"']*\{" + secret_field_pattern, body)
        or re.search(r"str\s*\(\s*" + secret_field_pattern, body)
    )


def line_in_body(entry: dict[str, Any], needle: str) -> int | None:
    body = str(entry.get("body") or "")
    local = find_line(body, needle)
    if local is None:
        return None
    return int(entry.get("line") or 1) + local - 1


def check_sdk_tests_import_package_app_as_top_level(
    review_input: dict[str, Any],
    full_files: dict[str, str],
) -> list[dict[str, Any]]:
    inventory = build_sdk_review_inventory(review_input, full_files)
    if not inventory["is_sdk_app"]:
        return []
    app_file = str(inventory.get("main_module_file") or "src/app.py")
    app_text = full_files.get(app_file, "")
    if "from ." not in app_text:
        return []

    sys_path_src_files = [
        path
        for path, text in full_files.items()
        if is_test_path(path)
        and "sys.path" in text
        and re.search(r"['\"]src['\"]", text)
    ]
    top_level_imports: list[tuple[str, int, str]] = []
    top_level_pattern = re.compile(r"^\s*(?:from\s+app\s+import\b|import\s+app\b|(?:mock\.)?patch\s*\(\s*['\"]app\.)", re.M)
    for path, text in full_files.items():
        if not is_test_path(path):
            continue
        match = top_level_pattern.search(text)
        if match:
            top_level_imports.append((path, text[: match.start()].count("\n") + 1, match.group(0).strip()))

    if not sys_path_src_files or not top_level_imports:
        return []
    first_path, first_line, first_import = top_level_imports[0]
    return [
        {
            "title": "SDK tests import src/app.py as a top-level module while the app uses package-relative imports",
            "category": "missing_tests",
            "severity": "high",
            "confidence": "high",
            "file": first_path,
            "line": first_line,
            "code_reference": first_import,
            "evidence": f"Tests add `src/` to sys.path in {sys_path_src_files[:3]} and import or patch `app`, but `{app_file}` uses package-relative imports such as `from .client ...`.",
            "why_it_matters": "The test suite can fail during import before exercising connector behavior, so it provides no usable evidence for a large SDK migration.",
            "suggested_fix": "Import and patch the package entry point the same way the SDK does, for example `import src.app` / `src.app.app`, or install the package and import through `[tool.soar.app].main_module`.",
        }
    ]


def check_sdk_output_datapath_regressions(
    legacy: dict[str, Any],
    python: dict[str, Any],
    full_files: dict[str, str],
    target_file: str,
    target_line: int,
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    current_text = "\n".join(full_files.values())
    actions_by_identifier = sdk_actions_by_identifier(python)
    for identifier, paths in (legacy.get("action_output_paths") or {}).items():
        nested_paths = [path for path in paths if re.search(r"\.\*\.\*\.", path)]
        if not nested_paths:
            continue
        flattened_matches = [
            (path, path.replace(".*.*.", ".*."))
            for path in nested_paths
            if (path.replace(".*.*.", ".*.") in current_text or sdk_model_flattens_nested_rows(full_files, identifier))
            and path not in current_text
        ]
        if not flattened_matches:
            continue
        action = actions_by_identifier.get(identifier) or {}
        file_path = str(action.get("implementation_file") or action.get("file") or target_file)
        line = find_sdk_output_model_line(full_files.get(file_path, ""), identifier) or int(
            action.get("implementation_line") or action.get("line") or target_line
        )
        legacy_path, sdk_path = flattened_matches[0]
        findings.append(
            {
                "title": "SDK migration changes a legacy nested output datapath",
                "category": "output_schema_mismatch",
                "severity": "high",
                "confidence": "high",
                "file": file_path,
                "line": line,
                "code_reference": legacy_path,
                "evidence": (
                    f"Legacy action `{identifier}` exposed `{legacy_path}`, but the SDK source/docs now expose "
                    f"`{sdk_path}`. The migration flattened one wildcard level from the playbook-facing path."
                ),
                "why_it_matters": "Existing playbooks and saved result references can stop resolving, and strict SDK output models can reject raw vendor payloads that still use the legacy nested row/cell shape.",
                "suggested_fix": "Preserve the legacy output shape by normalizing raw API rows before SDK validation, or update the SDK output model and migration notes to explicitly handle the legacy nested datapath.",
            }
        )
    return findings


def check_sdk_summary_field_regressions(
    legacy: dict[str, Any],
    python: dict[str, Any],
    full_files: dict[str, str],
    target_file: str,
    target_line: int,
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    summary_by_action = legacy.get("action_summary_fields") or {}
    if not summary_by_action:
        return findings

    actions_by_identifier = sdk_actions_by_identifier(python)
    missing_by_action: dict[str, list[str]] = {}
    first_file = target_file
    first_line = target_line
    for identifier, legacy_fields in summary_by_action.items():
        action = actions_by_identifier.get(identifier) or {}
        if not action:
            continue
        file_path = str(action.get("implementation_file") or action.get("file") or target_file)
        text = full_files.get(file_path, "")
        current_fields = sdk_summary_fields_for_action(text, str(action.get("function") or identifier), identifier)
        missing = sorted(set(legacy_fields) - current_fields)
        if not missing:
            continue
        missing_by_action[identifier] = missing
        if first_file == target_file:
            first_file = file_path
            first_line = find_line(text, f"def {action.get('function') or identifier}") or int(
                action.get("implementation_line") or action.get("line") or target_line
            )

    if missing_by_action:
        examples = [
            f"{identifier}: {fields[:6]}"
            for identifier, fields in list(missing_by_action.items())[:6]
        ]
        findings.append(
            {
                "title": "SDK migration drops legacy action summary fields",
                "category": "output_schema_mismatch",
                "severity": "high",
                "confidence": "high",
                "file": first_file,
                "line": first_line,
                "code_reference": "action_result.summary.*",
                "evidence": f"Legacy manifest summary fields are not emitted by the SDK migration: {'; '.join(examples)}.",
                "why_it_matters": "SOAR playbooks often read action_result.summary.* values directly; dropping those paths in a migration silently breaks existing playbooks and dashboards.",
                "suggested_fix": "Restore the missing summary fields with SDK summary output models and `soar.set_summary(...)`, or document the compatibility break in release notes and the PR body.",
            }
        )
    return findings


def check_sdk_large_output_schema_contractions(
    review_input: dict[str, Any],
    full_files: dict[str, str],
) -> list[dict[str, Any]]:
    inventory = build_sdk_review_inventory(review_input, full_files)
    if not inventory["is_sdk_migration"]:
        return []
    legacy_paths_by_action = inventory["legacy"].get("action_output_paths") or {}
    if not legacy_paths_by_action:
        return []

    output_models = collect_sdk_output_models(full_files)
    if not output_models:
        return []
    actions_by_identifier = sdk_actions_by_identifier(inventory["python"])
    findings: list[dict[str, Any]] = []

    for identifier, legacy_paths in legacy_paths_by_action.items():
        legacy_data_paths = [
            path
            for path in legacy_paths
            if path.startswith("action_result.data.") and not path.endswith("action_result.data")
        ]
        if len(legacy_data_paths) < 40:
            continue
        action = actions_by_identifier.get(identifier) or {}
        output_class = str(action.get("output_class") or action.get("return_annotation") or "").strip()
        if not output_class:
            continue
        sdk_paths = expand_sdk_output_model_paths(output_class, output_models)
        if not sdk_paths:
            continue
        if len(sdk_paths) >= max(30, int(len(legacy_data_paths) * 0.35)):
            continue

        target_file = str(action.get("file") or action.get("implementation_file") or "src/app.py")
        target_line = int(action.get("line") or action.get("implementation_line") or 1)
        model_line = output_models.get(output_class, {}).get("line")
        model_file = output_models.get(output_class, {}).get("file")
        findings.append(
            {
                "title": "SDK migration contracts a large legacy action output schema",
                "category": "output_schema_mismatch",
                "severity": "high",
                "confidence": "high",
                "file": str(model_file or target_file),
                "line": int(model_line or target_line),
                "code_reference": f"{identifier} / {output_class}",
                "evidence": (
                    f"Legacy action `{identifier}` declared {len(legacy_data_paths)} data output paths, "
                    f"but SDK output model `{output_class}` exposes about {len(sdk_paths)} modeled paths."
                ),
                "why_it_matters": (
                    "Even when the action returns raw vendor data, SDK/generated metadata controls output "
                    "discovery, documented datapaths, and playbook authoring. Contracting a large legacy "
                    "schema can silently remove paths users rely on."
                ),
                "suggested_fix": (
                    "Preserve the high-value legacy output paths with permissive nested SDK output models, "
                    "or explicitly document the compatibility break and regenerated output contract."
                ),
            }
        )
    return findings[:3]


def check_sdk_generic_strict_output_models(
    review_input: dict[str, Any],
    full_files: dict[str, str],
) -> list[dict[str, Any]]:
    inventory = build_sdk_review_inventory(review_input, full_files)
    if not inventory["is_sdk_migration"]:
        return []
    if has_sparse_response_tests(full_files):
        return []

    call_sites = raw_output_model_call_sites(full_files)
    if not call_sites:
        return []

    findings: list[dict[str, Any]] = []
    for path, text in full_files.items():
        if not path.endswith(".py") or is_test_path(path):
            continue
        classes = extract_sdk_output_classes(text)
        if not classes:
            continue
        for class_name, info in classes.items():
            if class_name.endswith("SummaryOutput") or class_name.endswith("Params") or class_name == "MakeRequestOutput":
                continue
            if class_name not in call_sites:
                continue
            required_fields = [
                field
                for field in info.get("fields", [])
                if field_is_required_like(field) and not sdk_field_annotation_is_permissive(str(field.get("annotation") or ""))
            ]
            if len(required_fields) < 3:
                continue
            call_site = call_sites[class_name][0]
            examples = [
                f"{field['name']}: {field['annotation']}"
                for field in required_fields[:8]
            ]
            findings.append(
                {
                    "title": "SDK output model validates raw API responses with many required fields",
                    "category": "output_schema_mismatch",
                    "severity": "medium",
                    "confidence": "medium",
                    "file": path,
                    "line": int(required_fields[0]["line"]),
                    "code_reference": class_name,
                    "evidence": (
                        f"`{class_name}` has {len(required_fields)} required modeled fields "
                        f"({examples}) and is built from raw response-shaped data at "
                        f"{call_site['file']}:{call_site['line']}."
                    ),
                    "why_it_matters": "Full SDK migrations replace legacy raw `add_data()` behavior with typed output construction. Vendor APIs often omit optional fields, so strict models can make successful API calls fail locally before SOAR receives data.",
                    "suggested_fix": "Default API response fields to optional/permissive unless the vendor contract proves they are always present, and add sparse/minimal response fixtures that construct the SDK output model.",
                }
            )
            break
    return findings[:3]


def has_sparse_response_tests(full_files: dict[str, str]) -> bool:
    test_text = "\n".join(text for path, text in full_files.items() if is_test_path(path)).lower()
    return any(
        term in test_text
        for term in (
            "sparse",
            "minimal",
            "missing field",
            "missing_field",
            "optional",
            "model_validate",
            "model_dump",
            "payload variant",
            "response variant",
        )
    )


def sdk_field_annotation_is_permissive(annotation: str) -> bool:
    lowered = annotation.lower().replace("typing.", "")
    return bool(re.search(r"\b(any|dict|mapping|object)\b", lowered)) or "permissive" in lowered


def raw_output_model_call_sites(full_files: dict[str, str]) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    raw_names = {
        "response",
        "response_body",
        "response_data",
        "response_json",
        "result",
        "result_json",
        "payload",
        "payload_json",
        "data",
        "item",
        "row",
        "record",
    }
    for path, text in full_files.items():
        if not path.endswith(".py") or is_test_path(path):
            continue
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            class_name: str | None = None
            raw_argument = False
            if isinstance(node.func, ast.Name) and node.func.id.endswith("Output"):
                class_name = node.func.id
                raw_argument = any(keyword.arg is None for keyword in node.keywords)
            elif (
                isinstance(node.func, ast.Attribute)
                and node.func.attr in {"model_validate", "parse_obj"}
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id.endswith("Output")
            ):
                class_name = node.func.value.id
                raw_argument = any(
                    isinstance(arg, ast.Name) and arg.id in raw_names
                    for arg in node.args[:1]
                )
            if not class_name or not raw_argument:
                continue
            output.setdefault(class_name, []).append({"file": path, "line": node.lineno})
    return output


def collect_sdk_output_models(full_files: dict[str, str]) -> dict[str, dict[str, Any]]:
    models: dict[str, dict[str, Any]] = {}
    for path, text in full_files.items():
        if not path.endswith(".py"):
            continue
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            bases = {base_name(base) for base in node.bases}
            fields: dict[str, str] = {}
            for stmt in node.body:
                if not isinstance(stmt, ast.AnnAssign) or not isinstance(stmt.target, ast.Name):
                    continue
                output_name = stmt.target.id
                if isinstance(stmt.value, ast.Call) and base_name(stmt.value.func) == "OutputField":
                    alias = literal_keyword_from_ast_call(stmt.value, "alias")
                    if isinstance(alias, str) and alias:
                        output_name = alias
                fields[output_name] = safe_unparse_for_checks(stmt.annotation) or ""
            if fields or bases & {"ActionOutput", "PermissiveActionOutput", "MakeRequestOutput"}:
                models[node.name] = {
                    "file": path,
                    "line": node.lineno,
                    "bases": bases,
                    "fields": fields,
                }
    return models


def expand_sdk_output_model_paths(
    class_name: str,
    models: dict[str, dict[str, Any]],
    *,
    prefix: str = "",
    seen: set[str] | None = None,
) -> set[str]:
    if seen is None:
        seen = set()
    class_name = class_name.strip("'\"")
    if class_name in seen or class_name not in models:
        return set()
    seen = {*seen, class_name}
    output: set[str] = set()
    for base in models[class_name].get("bases") or set():
        if base in models:
            output.update(expand_sdk_output_model_paths(base, models, prefix=prefix, seen=seen))
    for field, annotation in (models[class_name].get("fields") or {}).items():
        path = f"{prefix}.{field}" if prefix else field
        output.add(path)
        nested = first_output_model_annotation(annotation, models)
        if nested:
            output.update(expand_sdk_output_model_paths(nested, models, prefix=path, seen=seen))
    return output


def first_output_model_annotation(annotation: str, models: dict[str, dict[str, Any]]) -> str | None:
    for name in models:
        if re.search(rf"\b{re.escape(name)}\b", annotation):
            return name
    return None


def safe_unparse_for_checks(node: ast.AST | None) -> str | None:
    if node is None:
        return None
    try:
        return ast.unparse(node)
    except Exception:
        return None


def literal_keyword_from_ast_call(call: ast.Call, name: str) -> Any:
    for keyword in call.keywords:
        if keyword.arg != name:
            continue
        try:
            return ast.literal_eval(keyword.value)
        except Exception:
            return safe_unparse_for_checks(keyword.value)
    return None


def check_sdk_migration_missing_regression_tests(
    review_input: dict[str, Any],
    full_files: dict[str, str],
    target_file: str,
    target_line: int,
) -> list[dict[str, Any]]:
    changed = changed_paths(review_input)
    has_test_file = any(is_test_path(path) for path in changed) or any(is_test_path(path) for path in full_files)
    if has_test_file:
        return []
    risky_markers = (
        "model_validate(",
        "soar.set_summary",
        "TaniumClient",
        "BaseAsset",
        "ActionOutput",
        "Param(",
        "app.action(",
    )
    current_text = "\n".join(full_files.values())
    if not any(marker in current_text for marker in risky_markers):
        return []
    return [
        {
            "title": "Large SDK migration has no checked-in regression tests",
            "category": "missing_tests",
            "severity": "high",
            "confidence": "high",
            "file": target_file,
            "line": target_line,
            "code_reference": "tests/",
            "evidence": "This PR removes a legacy connector/manifest and adds SDK App, Params, ActionOutput, and client code, but no tests/, test_*.py, or *_test.py files are present.",
            "why_it_matters": "SDK migrations can regress output validation, summary datapaths, parameter validation, and failure paths without changing action names; those are exactly the behaviors existing playbooks depend on.",
            "suggested_fix": "Add mocked regression tests for migrated output models, action summaries, parameter validation/failure paths, and representative API responses before merge.",
        }
    ]


def check_sdk_full_migration_gate(
    review_input: dict[str, Any],
    full_files: dict[str, str],
) -> list[dict[str, Any]]:
    inventory = build_sdk_review_inventory(review_input, full_files)
    if not inventory["is_sdk_migration"]:
        return []

    target_file = sdk_target_file(inventory)
    target_line = 1
    if inventory["python"].get("app_instances"):
        target_file = inventory["python"]["app_instances"][0]["file"]
        target_line = inventory["python"]["app_instances"][0]["line"]

    findings: list[dict[str, Any]] = []
    if sdk_pytest_collected_zero_tests(review_input):
        findings.append(
            {
                "title": "Full SDK migration CI collected zero tests",
                "category": "missing_tests",
                "severity": "high",
                "confidence": "high",
                "file": "pyproject.toml" if "pyproject.toml" in full_files else target_file,
                "line": find_line(full_files.get("pyproject.toml", ""), "[tool.pytest") or target_line,
                "code_reference": "pytest",
                "evidence": "CI output indicates pytest collected no tests or exited with the no-tests status for this SDK migration.",
                "why_it_matters": "A full SDK migration rewrites auth, metadata, output serialization, validation, action dispatch, and error handling. A green-looking migration with zero collected tests gives no regression evidence.",
                "suggested_fix": "Add checked-in pytest tests and fix the pytest discovery configuration so CI collects them before merge.",
            }
        )

    manifest_result = review_input.get("sdk_manifest")
    if isinstance(manifest_result, dict) and manifest_result.get("attempted") is False:
        findings.append(
            {
                "title": "Full SDK migration skipped generated manifest inspection",
                "category": "soar_metadata",
                "severity": "high",
                "confidence": "medium",
                "file": "pyproject.toml" if "pyproject.toml" in full_files else target_file,
                "line": find_line(full_files.get("pyproject.toml", ""), "main_module") or target_line,
                "code_reference": "soarapps manifests create",
                "evidence": "The review context explicitly says SDK manifest generation was not attempted for a full SDK migration.",
                "why_it_matters": "The generated manifest is the packaged metadata users receive; SDK migrations can look correct in Python while generated actions, config fields, summary metadata, or output paths are wrong.",
                "suggested_fix": "Run `soarapps manifests create` or package build for the migrated SDK app and review the generated manifest before merge.",
            }
        )

    test_files = {
        path: text
        for path, text in full_files.items()
        if is_test_path(path) and path.endswith(".py") and not path.endswith("__init__.py")
    }
    if not test_files:
        return findings

    coverage = sdk_migration_test_coverage_topics(test_files, inventory)
    missing_topics = [name for name, present in coverage.items() if not present]
    if len(missing_topics) >= 3:
        first_test = sorted(test_files)[0]
        findings.append(
            {
                "title": "SDK migration tests miss core migration-risk areas",
                "category": "missing_tests",
                "severity": "medium",
                "confidence": "medium",
                "file": first_test,
                "line": 1,
                "code_reference": "SDK migration test coverage",
                "evidence": f"Checked-in tests are present, but these migration gate areas were not visibly covered: {missing_topics}.",
                "why_it_matters": "Human reviews repeatedly found SDK migration bugs in auth modes, generated metadata, output serialization, parameter validation, representative action behavior, and API error paths.",
                "suggested_fix": "Add focused tests for the missing areas: auth/test_connectivity, generated manifest/metadata, output model serialization, validation failures, representative migrated actions, and non-2xx/error handling.",
            }
        )
    return findings


def sdk_pytest_collected_zero_tests(review_input: dict[str, Any]) -> bool:
    ci = review_input.get("ci") or {}
    if not isinstance(ci, dict):
        return False
    pieces: list[str] = []
    for run in ci.get("check_runs", []):
        if not isinstance(run, dict):
            continue
        pieces.append(str(run.get("name") or ""))
        output = run.get("output") if isinstance(run.get("output"), dict) else {}
        pieces.append(str(output.get("summary") or ""))
        pieces.append(str(output.get("text") or ""))
    for log in ci.get("failed_check_logs", []):
        if not isinstance(log, dict):
            continue
        pieces.append(str(log.get("name") or ""))
        pieces.append(str(log.get("log_excerpt") or ""))
    text = "\n".join(pieces).lower()
    return any(
        phrase in text
        for phrase in (
            "collected 0 items",
            "collected 0 tests",
            "no tests ran",
            "exit code 5",
            "pytest_exit_code=5",
        )
    )


def sdk_migration_test_coverage_topics(
    test_files: dict[str, str],
    inventory: dict[str, Any],
) -> dict[str, bool]:
    text = "\n".join(test_files.values()).lower()
    action_names = [
        str(item.get("identifier") or item.get("function") or "").lower()
        for item in inventory["python"].get("action_registrations", [])
        if item.get("identifier") not in {None, "", "test_connectivity"}
    ]
    action_mentions = sum(1 for name in set(action_names) if name and name in text)
    return {
        "auth": any(term in text for term in ("auth", "oauth", "token", "client_secret", "personal_access_token", "test_connectivity")),
        "metadata": any(term in text for term in ("manifest", "pyproject", "assetfield", "read_only", "summary_type", "app_version")),
        "output_serialization": any(term in text for term in ("actionoutput", "model_validate", "model_dump", "output", "summary", "action_result.data")),
        "validation": any(term in text for term in ("pytest.raises", "actionfailure", "invalid", "validation", "positive", "non-empty", "empty string")),
        "representative_actions": action_mentions >= min(2, max(1, len(set(action_names)))),
        "error_handling": any(term in text for term in ("status_code", "httpstatuserror", "raise_for_status", "non-2xx", "4xx", "5xx", "error response", "actionfailure")),
    }


def sdk_actions_by_identifier(python: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(item.get("identifier") or ""): item
        for item in python.get("action_registrations", [])
        if item.get("identifier")
    }


def find_sdk_output_model_line(text: str, identifier: str) -> int | None:
    prefix = snake_to_pascal(identifier)
    return find_line(text, f"class {prefix}RowsOutput") or find_line(text, f"class {prefix}Output")


def sdk_model_flattens_nested_rows(full_files: dict[str, str], identifier: str) -> bool:
    prefix = snake_to_pascal(identifier)
    rows_class = f"{prefix}RowsOutput"
    cell_class = f"{prefix}CellOutput"
    for text in full_files.values():
        body = class_body_text(text, rows_class)
        if not body:
            continue
        if re.search(rf"\bdata\s*:\s*list\s*\[\s*{re.escape(cell_class)}\s*\]", body):
            return True
    return False


def class_body_text(text: str, class_name: str) -> str:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return ""
    lines = text.splitlines()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            end = getattr(node, "end_lineno", node.lineno)
            return "\n".join(lines[node.lineno - 1 : end])
    return ""


def sdk_summary_fields_for_action(text: str, function_name: str, identifier: str) -> set[str]:
    fields: set[str] = set()
    prefix = snake_to_pascal(identifier)
    fields.update(class_annotation_fields(text, f"{prefix}SummaryOutput"))
    body = function_body_text(text, function_name)
    for match in re.finditer(r"set_summary\s*\(\s*(?P<body>.*?)\)\s*", body, flags=re.S):
        fields.update(re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*=", match.group("body")))
    return fields


def class_annotation_fields(text: str, class_name: str) -> set[str]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        fields: set[str] = set()
        for stmt in node.body:
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                fields.add(stmt.target.id)
        return fields
    return set()


def function_body_text(text: str, function_name: str) -> str:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return ""
    lines = text.splitlines()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
            end = getattr(node, "end_lineno", node.lineno)
            return "\n".join(lines[node.lineno - 1 : end])
    return ""


def snake_to_pascal(value: str) -> str:
    return "".join(part[:1].upper() + part[1:] for part in value.split("_") if part)


def is_test_path(path: str) -> bool:
    lowered = path.lower()
    return (
        lowered.startswith("tests/")
        or lowered.startswith("test")
        or "/tests/" in lowered
        or lowered.endswith("_test.py")
        or lowered.startswith("app-tests/")
    )


def check_sdk_manifest_result(review_input: dict[str, Any]) -> list[dict[str, Any]]:
    manifest_result = review_input.get("sdk_manifest")
    if not isinstance(manifest_result, dict) or not manifest_result.get("attempted"):
        return []

    findings: list[dict[str, Any]] = []
    status = str(manifest_result.get("status") or "")
    if status == "failed" and sdk_manifest_failure_is_project_blocker(manifest_result):
        file_path, line = sdk_manifest_failure_location(manifest_result)
        findings.append(
            {
                "title": "SDK generated manifest build fails",
                "category": "precommit",
                "severity": "high",
                "confidence": "high",
                "file": file_path,
                "line": line,
                "code_reference": "soarapps manifests create",
                "evidence": sdk_manifest_failure_evidence(manifest_result),
                "why_it_matters": "SDK packaging and app metadata generation import the App instance and serialize asset/action/output metadata; if manifest generation fails, the connector package cannot be trusted.",
                "suggested_fix": "Fix the SDK registration/import/metadata error reported by `soarapps manifests create`, then rerun manifest generation or package build.",
            }
        )
        return findings

    if status != "success":
        return findings

    manifest = manifest_result.get("manifest")
    if not isinstance(manifest, dict):
        return findings

    base_files = review_input.get("base_files") or {}
    if not isinstance(base_files, dict):
        base_files = {}
    legacy = build_sdk_review_inventory({**review_input, "full_files": {}})["legacy"]
    if not legacy.get("manifest_paths"):
        return findings

    generated_actions = {
        str(action.get("identifier") or "")
        for action in manifest.get("actions", [])
        if isinstance(action, dict) and action.get("identifier")
    }
    missing_actions = sorted(set(legacy.get("action_identifiers") or []) - generated_actions)
    if missing_actions:
        findings.append(
            {
                "title": "Generated SDK manifest drops legacy actions",
                "category": "soar_metadata",
                "severity": "high",
                "confidence": "high",
                "file": "pyproject.toml",
                "line": 1,
                "code_reference": "generated manifest actions",
                "evidence": f"`soarapps manifests create` generated actions {sorted(generated_actions)[:12]}, missing legacy actions {missing_actions[:12]}.",
                "why_it_matters": "The generated SDK manifest is what users receive; dropped actions remove user-facing behavior and can break existing playbooks.",
                "suggested_fix": "Register the missing action implementations in the SDK app or explicitly document and release-note intentional removals.",
            }
        )

    generated_config = generated_manifest_config_names(manifest)
    missing_config = sorted(set(legacy.get("config_fields") or []) - generated_config)
    if missing_config:
        findings.append(
            {
                "title": "Generated SDK manifest drops legacy asset configuration fields",
                "category": "soar_metadata",
                "severity": "high",
                "confidence": "high",
                "file": "pyproject.toml",
                "line": 1,
                "code_reference": "generated manifest configuration",
                "evidence": f"`soarapps manifests create` generated config fields {sorted(generated_config)[:12]}, missing legacy fields {missing_config[:12]}.",
                "why_it_matters": "Missing generated asset fields can break existing assets or hide required credentials/options from users after SDK migration.",
                "suggested_fix": "Add the missing fields to the SDK `BaseAsset` model and verify the generated manifest includes them.",
            }
        )

    for action in manifest.get("actions", []):
        if not isinstance(action, dict):
            continue
        identifier = str(action.get("identifier") or "")
        if is_mutating_identifier(identifier) and action.get("read_only") is not False:
            findings.append(
                {
                    "title": "Generated SDK manifest marks a mutating action read-only",
                    "category": "soar_metadata",
                    "severity": "high",
                    "confidence": "high",
                    "file": "pyproject.toml",
                    "line": 1,
                    "code_reference": identifier,
                    "evidence": f"Generated action `{identifier}` has read_only={action.get('read_only')!r}.",
                    "why_it_matters": "SOAR uses generated action metadata to advertise action behavior; mutating actions should not be marked read-only.",
                    "suggested_fix": "Set `read_only=False` on the SDK action registration and regenerate the manifest.",
                }
            )
            break

    return findings


def check_sdk_generated_manifest_contract_drift(review_input: dict[str, Any]) -> list[dict[str, Any]]:
    manifest_result = review_input.get("sdk_manifest")
    if not isinstance(manifest_result, dict) or manifest_result.get("status") != "success":
        return []
    generated = manifest_result.get("manifest")
    if not isinstance(generated, dict):
        return []

    legacy_app_jsons = legacy_root_app_jsons(review_input)
    if not legacy_app_jsons:
        return []

    findings: list[dict[str, Any]] = []
    generated_actions = {
        str(action.get("identifier") or ""): action
        for action in generated.get("actions", [])
        if isinstance(action, dict) and action.get("identifier")
    }
    legacy_actions = legacy_actions_by_identifier(legacy_app_jsons)

    output_drops = generated_manifest_output_drops(legacy_actions, generated_actions)
    if output_drops:
        first_identifier, first_detail = output_drops[0]
        findings.append(
            {
                "title": "Generated SDK manifest drops legacy output datapaths",
                "category": "output_schema_mismatch",
                "severity": "high",
                "confidence": "high",
                "file": "pyproject.toml",
                "line": 1,
                "code_reference": first_identifier,
                "evidence": "; ".join(detail for _, detail in output_drops[:4]),
                "why_it_matters": "The generated SDK manifest controls README/datapath discovery and playbook-visible output paths. Dropping legacy data or summary paths in a migration can silently break existing playbooks even if Python returns raw data.",
                "suggested_fix": "Preserve the legacy datapaths in SDK ActionOutput/summary models and regenerate the manifest, or add an explicit migration note for every intentionally changed output path.",
            }
        )

    param_drops = generated_manifest_parameter_drift(legacy_actions, generated_actions)
    if param_drops:
        first_identifier, first_param, first_detail = param_drops[0]
        findings.append(
            {
                "title": "Generated SDK manifest changes legacy action parameter metadata",
                "category": "soar_metadata",
                "severity": "high",
                "confidence": "high",
                "file": "pyproject.toml",
                "line": 1,
                "code_reference": f"{first_identifier}.{first_param}",
                "evidence": "; ".join(detail for _, _, detail in param_drops[:6]),
                "why_it_matters": "Required flags, defaults, and contains/CEF metadata are part of the SOAR action contract. Changing them during an SDK migration can break playbooks, prompts, and typed indicator handling.",
                "suggested_fix": "Align SDK Param metadata with the legacy manifest or explicitly document and release-note each intentional required/default/contains change.",
            }
        )

    return findings


def legacy_root_app_jsons(review_input: dict[str, Any]) -> dict[str, dict[str, Any]]:
    base_files = review_input.get("base_files") if isinstance(review_input.get("base_files"), dict) else {}
    manifests = load_app_jsons(base_files or {})
    full_files = review_input.get("full_files") if isinstance(review_input.get("full_files"), dict) else {}
    removed_paths = {
        str(item.get("filename"))
        for item in review_input.get("changed_files", [])
        if isinstance(item, dict) and item.get("status") == "removed" and "/" not in str(item.get("filename") or "") and str(item.get("filename") or "").endswith(".json")
    }
    removed_manifests = load_app_jsons({path: str(full_files.get(path) or "") for path in removed_paths})
    manifests.update(removed_manifests)
    return manifests


def legacy_actions_by_identifier(app_jsons: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for app_json in app_jsons.values():
        for action in app_json.get("actions", []):
            if not isinstance(action, dict):
                continue
            identifier = str(action.get("identifier") or "").strip()
            if identifier:
                output[identifier] = action
    return output


def generated_manifest_output_drops(
    legacy_actions: dict[str, dict[str, Any]],
    generated_actions: dict[str, dict[str, Any]],
) -> list[tuple[str, str]]:
    drops: list[tuple[str, str]] = []
    for identifier, legacy_action in legacy_actions.items():
        generated_action = generated_actions.get(identifier)
        if not generated_action:
            continue
        legacy_paths = action_output_paths(legacy_action)
        generated_paths = action_output_paths(generated_action)
        legacy_interesting = {
            path
            for path in legacy_paths
            if path.startswith("action_result.data.") or path.startswith("action_result.summary.")
        }
        if not legacy_interesting:
            continue
        missing = sorted(path for path in legacy_interesting if path not in generated_paths)
        if not missing:
            continue
        legacy_data_count = sum(1 for path in legacy_interesting if path.startswith("action_result.data."))
        legacy_summary_count = sum(1 for path in legacy_interesting if path.startswith("action_result.summary."))
        missing_data_count = sum(1 for path in missing if path.startswith("action_result.data."))
        missing_summary_count = sum(1 for path in missing if path.startswith("action_result.summary."))
        if missing_summary_count or missing_data_count >= 3 or missing_data_count >= max(1, int(legacy_data_count * 0.4)):
            drops.append(
                (
                    identifier,
                    (
                        f"`{identifier}` generated output is missing {missing_data_count}/{legacy_data_count} "
                        f"legacy data paths and {missing_summary_count}/{legacy_summary_count} summary paths; "
                        f"examples: {missing[:6]}"
                    ),
                )
            )
    return drops


def generated_manifest_parameter_drift(
    legacy_actions: dict[str, dict[str, Any]],
    generated_actions: dict[str, dict[str, Any]],
) -> list[tuple[str, str, str]]:
    drift: list[tuple[str, str, str]] = []
    for identifier, legacy_action in legacy_actions.items():
        generated_action = generated_actions.get(identifier)
        if not generated_action:
            continue
        legacy_params = params_by_name(legacy_action)
        generated_params = params_by_name(generated_action)
        for name, legacy_param in legacy_params.items():
            generated_param = generated_params.get(name)
            if not generated_param:
                continue
            changes = []
            for key in ("required", "default"):
                legacy_value = normalized_param_value(legacy_param.get(key))
                generated_value = normalized_param_value(generated_param.get(key))
                if legacy_value != generated_value:
                    changes.append(f"{key}: legacy={legacy_value!r}, generated={generated_value!r}")
            legacy_contains = normalized_contains_value(legacy_param)
            generated_contains = normalized_contains_value(generated_param)
            if legacy_contains and legacy_contains != generated_contains:
                changes.append(f"contains/cef_types: legacy={legacy_contains!r}, generated={generated_contains!r}")
            if changes:
                drift.append((identifier, name, f"`{identifier}.{name}` changed {', '.join(changes)}"))
    return drift


def params_by_name(action: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(item.get("name") or item.get("identifier") or "").strip(): item
        for item in action_parameters(action)
        if str(item.get("name") or item.get("identifier") or "").strip()
    }


def normalized_param_value(value: Any) -> Any:  # noqa: ANN401
    if isinstance(value, str):
        return value.strip().lower()
    return value


def normalized_contains_value(param: dict[str, Any]) -> tuple[str, ...]:
    raw = param.get("contains")
    if raw is None:
        raw = param.get("cef_types")
    if raw is None:
        return ()
    if isinstance(raw, str):
        return (raw.strip().lower(),) if raw.strip() else ()
    if isinstance(raw, list):
        return tuple(sorted(str(item).strip().lower() for item in raw if str(item).strip()))
    return (str(raw).strip().lower(),)


def sdk_manifest_failure_is_project_blocker(manifest_result: dict[str, Any]) -> bool:
    if manifest_result.get("stage") != "manifest_create":
        return False
    text = f"{manifest_result.get('stdout') or ''}\n{manifest_result.get('stderr') or ''}".lower()
    local_toolchain_markers = (
        "could not resolve host",
        "failed to download",
        "network is unreachable",
        "temporary failure in name resolution",
        "connection refused",
        "no module named soar_sdk",
        "command not found",
    )
    if any(marker in text for marker in local_toolchain_markers):
        return False
    project_markers = (
        "action function must",
        "actionregistrationerror",
        "asset field",
        "can only define these fields",
        "duplicate output field",
        "failed to serialize",
        "file parameter",
        "invalid type annotation",
        "manifest",
        "model_validate",
        "no package",
        "not supported",
        "params type",
        "pydantic",
        "reserved by the platform",
        "return type",
        "sensitive parameter",
        "sensitive field",
        "traceback",
        "typeerror",
        "valueerror",
        "uv.lock",
    )
    return any(marker in text for marker in project_markers)


def sdk_manifest_failure_location(manifest_result: dict[str, Any]) -> tuple[str, int]:
    text = f"{manifest_result.get('stderr') or ''}\n{manifest_result.get('stdout') or ''}"
    for match in re.finditer(r'File "<project>/(?P<path>[^"]+)", line (?P<line>\d+)', text):
        path = match.group("path")
        if "/.venv/" in path or path.startswith("."):
            continue
        return path, int(match.group("line"))
    return "pyproject.toml", 1


def sdk_manifest_failure_evidence(manifest_result: dict[str, Any]) -> str:
    text = "\n".join(
        part
        for part in (str(manifest_result.get("stderr") or ""), str(manifest_result.get("stdout") or ""))
        if part.strip()
    )
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    interesting = [
        line
        for line in lines
        if any(
            marker in line.lower()
            for marker in ("traceback", "typeerror", "valueerror", "action", "params", "output", "asset", "uv.lock", "no package")
        )
    ]
    return "\n".join((interesting or lines)[-8:])


def generated_manifest_config_names(manifest: dict[str, Any]) -> set[str]:
    config = manifest.get("configuration") or {}
    if isinstance(config, dict):
        return {str(key) for key in config}
    if isinstance(config, list):
        return {
            str(item.get("name") or item.get("key") or item.get("identifier"))
            for item in config
            if isinstance(item, dict) and (item.get("name") or item.get("key") or item.get("identifier"))
        }
    return set()


def sdk_target_file(inventory: dict[str, Any]) -> str:
    python = inventory.get("python") or {}
    app_instances = python.get("app_instances") or []
    if app_instances:
        return str(app_instances[0]["file"])
    if inventory.get("main_module_file_present"):
        return str(inventory["main_module_file"])
    return "src/app.py"


def is_mutating_identifier(identifier: str) -> bool:
    normalized = identifier.replace("-", "_").lower()
    return normalized.startswith(MUTATING_PREFIXES)


GRAPH_OPTIONAL_FIELDS = {
    "@microsoft.graph.downloadUrl",
    "@odata.context",
    "@odata.etag",
    "@odata.type",
    "application",
    "blocksDownload",
    "cTag",
    "childCount",
    "createdBy",
    "createdDateTime",
    "deleted",
    "description",
    "displayName",
    "driveId",
    "driveType",
    "eTag",
    "email",
    "file",
    "fileSystemInfo",
    "folder",
    "folderPath",
    "height",
    "hashes",
    "image",
    "lastModifiedBy",
    "lastModifiedDateTime",
    "mimeType",
    "owner",
    "package",
    "parentReference",
    "quickXorHash",
    "quota",
    "readOnly",
    "remaining",
    "root",
    "sha1Hash",
    "shared",
    "size",
    "state",
    "total",
    "used",
    "user",
    "webUrl",
    "width",
}


def check_microsoft_graph_sdk_response_modeling(
    review_input: dict[str, Any],
    full_files: dict[str, str],
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    changed = changed_paths(review_input)

    for path, text in full_files.items():
        if not path.endswith(".py") or path not in changed:
            continue
        lowered = text.lower()
        graphish_upload = "upload" in path.lower() and ("upload_url" in lowered or "upload session" in lowered or "upload_session" in lowered)
        if not graphish_upload and "microsoft graph" not in lowered and "graph.microsoft" not in lowered and "onedrive" not in lowered:
            continue
        if "actionoutput" not in lowered:
            continue

        classes = extract_sdk_output_classes(text)
        if not classes:
            continue

        raw_instantiations = raw_action_output_instantiations(text)
        sparse_fields = [
            (class_name, field)
            for class_name, info in classes.items()
            for field in info["fields"]
            if is_graph_optional_field(field["name"]) and field_is_required_like(field)
        ]

        if (
            sparse_fields
            and any(name in raw_instantiations for name in classes)
            and path.rsplit("/", 1)[-1] in {"list_drive.py", "list_items.py"}
        ):
            samples = [
                f"{class_name}.{field['name']}: {field['annotation']}"
                for class_name, field in sparse_fields[:8]
            ]
            first_class, first_field = sparse_fields[0]
            findings.append(
                {
                    "title": "Microsoft Graph output models require fields that Graph can omit",
                    "category": "output_schema_mismatch",
                    "severity": "high",
                    "confidence": "high",
                    "file": path,
                    "line": int(first_field["line"]),
                    "code_reference": first_class,
                    "evidence": (
                        "The action constructs SDK output models directly from Graph response dictionaries, "
                        f"while these Graph-shaped fields are declared without an explicit `= None` default: {samples}."
                    ),
                    "why_it_matters": (
                        "Microsoft Graph drive and driveItem resources vary by backing store and response shape; "
                        "a successful API call can fail locally during SDK model construction instead of "
                        "returning data to SOAR."
                    ),
                    "suggested_fix": (
                        "Give Graph fields and nested facet fields a real optional default, then keep or add "
                        "sparse-payload unit tests for document libraries, app-created items, folders, and "
                        "minimal driveItem responses."
                    ),
                }
            )

        if "upload" in path.lower() and "UploadFileOutput" in classes:
            upload_fields = {
                field["name"]: field
                for field in classes["UploadFileOutput"]["fields"]
            }
            parent_reference = upload_fields.get("parentReference")
            if (
                parent_reference
                and field_is_required_like(parent_reference)
                and "UploadFileOutput" in raw_instantiations
            ):
                findings.append(
                    {
                        "title": "Upload completion output requires parentReference from Graph response",
                        "category": "output_schema_mismatch",
                        "severity": "high",
                        "confidence": "high",
                        "file": path,
                        "line": int(parent_reference["line"]),
                        "code_reference": "UploadFileOutput.parentReference",
                        "evidence": (
                            "`upload_file()` returns `UploadFileOutput(**upload_response)`, but "
                            "`UploadFileOutput.parentReference` has no explicit default even though Graph "
                            "upload-session completion can return a minimal driveItem payload."
                        ),
                        "why_it_matters": (
                            "The file upload can succeed in Microsoft Graph and then be reported as failed by "
                            "SOAR because local output model construction rejects the returned shape."
                        ),
                        "suggested_fix": (
                            "Make `parentReference` optional with a default of `None`, or fetch complete "
                            "driveItem metadata after upload before constructing `UploadFileOutput`."
                        ),
                    }
                )

    return findings


def extract_sdk_output_classes(text: str) -> dict[str, dict[str, Any]]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return {}

    classes: dict[str, dict[str, Any]] = {}
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        base_names = {base_name(base) for base in node.bases}
        if not base_names.intersection({"ActionOutput", "PermissiveActionOutput"}):
            continue
        fields = []
        for statement in node.body:
            if not isinstance(statement, ast.AnnAssign) or not isinstance(statement.target, ast.Name):
                continue
            fields.append(
                {
                    "name": statement.target.id,
                    "line": statement.lineno,
                    "annotation": ast.unparse(statement.annotation),
                    "value": ast.unparse(statement.value) if statement.value is not None else "",
                    "has_default_none": isinstance(statement.value, ast.Constant) and statement.value.value is None,
                }
            )
        classes[node.name] = {"line": node.lineno, "bases": base_names, "fields": fields}
    return classes


def base_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Subscript):
        return base_name(node.value)
    return ""


def is_graph_optional_field(name: str) -> bool:
    return name in GRAPH_OPTIONAL_FIELDS


def field_is_required_like(field: dict[str, Any]) -> bool:
    value = str(field.get("value") or "")
    if field.get("has_default_none"):
        return False
    if re.search(r"\bdefault\s*=\s*None\b", value):
        return False
    return True


def raw_action_output_instantiations(text: str) -> set[str]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return set()
    output: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if not node.func.id.endswith("Output"):
            continue
        if any(isinstance(keyword.value, ast.Name) for keyword in node.keywords if keyword.arg is None):
            output.add(node.func.id)
    return output


def check_large_file_transfer_patterns(
    review_input: dict[str, Any],
    full_files: dict[str, str],
) -> list[dict[str, Any]]:
    findings = []
    changed = changed_paths(review_input)

    for path, text in full_files.items():
        if not path.endswith(".py") or path not in changed:
            continue
        lowered = text.lower()
        if "vault" not in lowered and "upload_session" not in lowered and "downloadurl" not in lowered:
            continue

        download_match = re.search(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*=\s*httpx\.get\([^)]*\)", text)
        if download_match:
            response_name = download_match.group(1)
            content_pattern = re.compile(rf"\b([A-Za-z_][A-Za-z0-9_]*)\s*=\s*{re.escape(response_name)}\.content\b")
            content_match = content_pattern.search(text)
            if content_match and "create_attachment" in text[content_match.end() : content_match.end() + 1200]:
                line_no = text[: content_match.start()].count("\n") + 1
                findings.append(
                    {
                        "title": "OneDrive download buffers the entire file before vault attachment",
                        "category": "general",
                        "severity": "medium",
                        "confidence": "high",
                        "file": path,
                        "line": line_no,
                        "code_reference": content_match.group(0),
                        "evidence": (
                            f"`{content_match.group(0)}` stores the complete HTTP response body before calling "
                            "`soar.vault.create_attachment(...)`."
                        ),
                        "why_it_matters": (
                            "Large OneDrive files can consume memory proportional to file size before SOAR can "
                            "persist them to vault."
                        ),
                        "suggested_fix": (
                            "Stream the HTTP response into a temporary file or vault-compatible stream instead "
                            "of materializing `response.content` in memory."
                        ),
                    }
                )

        whole_file_read = re.search(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*=\s*[^=\n]*\.read\(\)", text)
        if whole_file_read and re.search(rf"{re.escape(whole_file_read.group(1))}\s*\[", text):
            line_no = text[: whole_file_read.start()].count("\n") + 1
            findings.append(
                {
                    "title": "Upload path reads the whole vault file before chunking",
                    "category": "general",
                    "severity": "medium",
                    "confidence": "high",
                    "file": path,
                    "line": line_no,
                    "code_reference": whole_file_read.group(0),
                    "evidence": f"`{whole_file_read.group(0)}` reads the whole file and later slices that buffer for upload chunks.",
                    "why_it_matters": (
                        "Microsoft Graph upload sessions are meant for large files; buffering the full vault "
                        "file duplicates memory and can fail under large uploads."
                    ),
                    "suggested_fix": (
                        "Seek and read bounded chunks from the vault file handle for each upload request "
                        "instead of slicing a full-file byte buffer."
                    ),
                }
            )

    return findings


def check_upload_session_resilience(
    review_input: dict[str, Any],
    full_files: dict[str, str],
) -> list[dict[str, Any]]:
    findings = []
    changed = changed_paths(review_input)

    for path, text in full_files.items():
        if not path.endswith(".py") or path not in changed:
            continue
        if "upload" not in path.lower() and "uploadsession" not in text.lower() and "upload_session" not in text.lower():
            continue
        for function_name, start_line, body in extract_functions_matching(text, r"upload.*chunk|chunk.*upload|upload"):
            lowered = body.lower()
            if "httpx.put" not in lowered or "content-range" not in lowered:
                continue
            has_retry = any(term in lowered for term in ("retry", "backoff", "retry-after", "time.sleep", "429", "500", "502", "503", "504"))
            if has_retry:
                continue
            findings.append(
                {
                    "title": "Upload-session chunk loop has no retry or backoff path for transient Graph failures",
                    "category": "general",
                    "severity": "medium",
                    "confidence": "high",
                    "file": path,
                    "line": start_line,
                    "code_reference": function_name,
                    "evidence": (
                        f"{function_name}() sends chunked `httpx.put(...)` requests with `Content-Range` and "
                        "immediately calls `raise_for_status()` without visible retry/backoff handling."
                    ),
                    "why_it_matters": (
                        "Large Microsoft Graph upload sessions commonly need retry handling for transient "
                        "429/5xx responses; failing immediately can turn recoverable uploads into failed SOAR "
                        "actions."
                    ),
                    "suggested_fix": (
                        "Handle transient 429/5xx responses with bounded retry/backoff, honor `Retry-After` "
                        "when present, and continue from Graph's returned `nextExpectedRanges`."
                    ),
                }
            )
            break

    return findings


def extract_handle_functions(text: str) -> list[tuple[str, int, str]]:
    lines = text.splitlines()
    starts: list[tuple[str, int, int]] = []
    pattern = re.compile(r"^(\s*)def\s+_handle_([A-Za-z0-9_]+)\s*\(")
    for index, line in enumerate(lines):
        match = pattern.match(line)
        if match:
            starts.append((match.group(2), index, len(match.group(1))))

    functions: list[tuple[str, int, str]] = []
    for offset, (name, start_idx, indent) in enumerate(starts):
        end_idx = len(lines)
        for _, next_start_idx, next_indent in starts[offset + 1 :]:
            if next_indent <= indent:
                end_idx = next_start_idx
                break
        functions.append((name, start_idx + 1, "\n".join(lines[start_idx:end_idx])))
    return functions


def extract_functions_matching(text: str, name_pattern: str) -> list[tuple[str, int, str]]:
    lines = text.splitlines()
    starts: list[tuple[str, int, int, bool]] = []
    pattern = re.compile(r"^(\s*)def\s+([A-Za-z0-9_]+)\s*\(")
    wanted = re.compile(name_pattern)
    for index, line in enumerate(lines):
        match = pattern.match(line)
        if match:
            name = match.group(2)
            starts.append((name, index, len(match.group(1)), wanted.search(name) is not None))

    functions: list[tuple[str, int, str]] = []
    for offset, (name, start_idx, indent, include) in enumerate(starts):
        if not include:
            continue
        end_idx = len(lines)
        for _, next_start_idx, next_indent, _ in starts[offset + 1 :]:
            if next_indent <= indent:
                end_idx = next_start_idx
                break
        functions.append((name, start_idx + 1, "\n".join(lines[start_idx:end_idx])))
    return functions


def check_polling_dedup_heuristics(full_files: dict[str, str]) -> list[dict[str, Any]]:
    findings = []
    timestamp_sdi = re.compile(
        r"source_data_identifier[^=\n:}]*(=|:)\s*[^#\n}]*(datetime|time\.time|strftime|timestamp|created_at|updated_at|last_modified|current_time|now)",
        re.I,
    )
    for path, text in full_files.items():
        if not path.endswith(".py"):
            continue
        for function_name, start_line, body in extract_functions_matching(text, r"poll"):
            lowered = body.lower()
            sdi_match = timestamp_sdi.search(body)
            if sdi_match:
                findings.append(
                    {
                        "title": "Polling uses timestamp-like source_data_identifier",
                        "category": "polling_checkpoint",
                        "severity": "high",
                        "confidence": "medium",
                        "file": path,
                        "line": start_line + body[: sdi_match.start()].count("\n"),
                        "evidence": line_from_match(body, sdi_match.start()),
                        "why_it_matters": "Timestamp-based SDIs are unstable for dedup because multiple events can share a timestamp and replay behavior becomes ambiguous.",
                        "suggested_fix": "Use a stable vendor event ID for source_data_identifier, or combine timestamp with a unique event identifier.",
                    }
                )

            if re.search(r"last_[A-Za-z0-9_]*id", body) and any(word in lowered for word in ("timestamp", "created_at", "updated_at", "last_modified", "start_time", "end_time")):
                findings.append(
                    {
                        "title": "Polling cursor mixes time boundary with a single saved ID",
                        "category": "polling_checkpoint",
                        "severity": "high",
                        "confidence": "medium",
                        "file": path,
                        "line": start_line,
                        "evidence": f"{function_name}() references both last_*id state and timestamp/time filters.",
                        "why_it_matters": "A single boundary ID with a time-based server cursor can reprocess or skip events that share the same timestamp.",
                        "suggested_fix": "Track a complete boundary set for the timestamp, use a vendor cursor, or advance the checkpoint only after all same-boundary events are safely ingested.",
                    }
                )

            if "is_poll_now" in lowered and "save_state" in lowered and not re.search(r"if\s+not\s+[^:\n]*poll_now", lowered):
                findings.append(
                    {
                        "title": "Poll Now appears able to write scheduled-poll state",
                        "category": "polling_checkpoint",
                        "severity": "high",
                        "confidence": "medium",
                        "file": path,
                        "line": start_line,
                        "evidence": f"{function_name}() references is_poll_now and save_state without a visible not-poll-now guard.",
                        "why_it_matters": "Poll Now should not contaminate scheduled polling checkpoints, or later scheduled polls can miss data.",
                        "suggested_fix": "Keep Poll Now state isolated and only persist scheduled checkpoints when the run is not Poll Now and ingestion has succeeded.",
                    }
                )

            if (
                "save_state" in lowered
                and ("save_artifact" in lowered or "save_container" in lowered)
                and any(word in lowered for word in ("except", "failed", "failure", "continue"))
            ):
                findings.append(
                    {
                        "title": "Polling may advance checkpoint state after partial ingestion failure",
                        "category": "polling_checkpoint",
                        "severity": "high",
                        "confidence": "low",
                        "file": path,
                        "line": start_line,
                        "evidence": f"{function_name}() saves state and also handles artifact/container save failures.",
                        "why_it_matters": "Advancing checkpoints after a failed artifact/container write can silently drop events.",
                        "suggested_fix": "Only persist checkpoint state after all containers/artifacts for the batch were created successfully, or persist enough retry state to recover failed events.",
                    }
                )

            if has_unbounded_container_lookup_pattern(lowered):
                findings.append(
                    {
                        "title": "Polling container lookup may scale with source group cardinality",
                        "category": "polling_checkpoint",
                        "severity": "medium",
                        "confidence": "medium",
                        "file": path,
                        "line": start_line,
                        "evidence": f"{function_name}() appears to do SOAR container discovery while looping over source groups/events.",
                        "why_it_matters": "Historical reviews caught polling designs where one /rest/container lookup per rule/group made platform load depend on source-side cardinality and could hit rate limits.",
                        "suggested_fix": "Use a deterministic SDI/container model, or do one bounded container query per poll run and build an in-memory lookup before processing groups/events.",
                    }
                )
    return findings


def has_unbounded_container_lookup_pattern(lowered_body: str) -> bool:
    has_container_lookup = any(
        phrase in lowered_body
        for phrase in (
            "/rest/container",
            "rest/container",
            "_check_for_existing_container",
            "check_for_existing_container",
            "get_container",
        )
    )
    if not has_container_lookup:
        return False
    has_source_group_loop = re.search(
        r"for\s+[^:\n]*(rule|group|detection|event|alert|container)[^:\n]*:",
        lowered_body,
    )
    has_grouping_state = any(
        phrase in lowered_body
        for phrase in (
            "rule_id",
            "ruleid",
            "grouped",
            "group_by",
            "source_data_identifier",
            "detections_by",
        )
    )
    return bool(has_source_group_loop and has_grouping_state)


def check_polling_container_artifact_contracts(full_files: dict[str, str]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for path, text in full_files.items():
        if not path.endswith(".py"):
            continue
        for function_name, start_line, body in extract_functions_matching(text, r"poll"):
            lowered = body.lower()
            if "save_container" not in lowered and "save_artifact" not in lowered:
                continue

            if "save_container" in lowered and re.search(r"['\"]label['\"]\s*:", body) is None:
                line = start_line + body[: body.lower().find("save_container")].count("\n") if "save_container" in lowered else start_line
                findings.append(
                    {
                        "title": "on_poll creates containers without a label",
                        "category": "polling_checkpoint",
                        "severity": "high",
                        "confidence": "high",
                        "file": path,
                        "line": line,
                        "code_reference": function_name,
                        "evidence": f"{function_name}() calls save_container(), but no `label` key was found in the container body.",
                        "why_it_matters": "SOAR container labels are required for routing and queueing; unlabeled containers can be hard to triage or fail deployment-specific routing expectations.",
                        "suggested_fix": "Add a configurable container label from asset config or a documented connector default before calling save_container().",
                    }
                )

            standard_poll_controls = ("is_poll_now", "container_count", "artifact_count", "start_time", "end_time")
            missing_controls = [name for name in standard_poll_controls if name not in lowered]
            if len(missing_controls) >= 4:
                findings.append(
                    {
                        "title": "on_poll ignores standard SOAR poll controls",
                        "category": "polling_checkpoint",
                        "severity": "high",
                        "confidence": "high",
                        "file": path,
                        "line": start_line,
                        "code_reference": function_name,
                        "evidence": f"{function_name}() creates poll artifacts/containers but does not visibly use {missing_controls}.",
                        "why_it_matters": "Scheduled and manual poll runs should respect SOAR poll parameters so users can bound time windows and ingestion volume.",
                        "suggested_fix": "Use `is_poll_now()`, start/end time, container_count, and artifact_count to bound manual and scheduled polling behavior, and keep scheduled checkpoint state separate from Poll Now.",
                    }
                )

            ignored_artifact_match = re.search(r"^\s*self\.save_artifact\s*\(\s*artifact\s*\)", body, re.M)
            if ignored_artifact_match:
                findings.append(
                    {
                        "title": "on_poll ignores save_artifact return values",
                        "category": "polling_checkpoint",
                        "severity": "high",
                        "confidence": "high",
                        "file": path,
                        "line": start_line + body[: ignored_artifact_match.start()].count("\n"),
                        "code_reference": function_name,
                        "evidence": line_from_match(body, ignored_artifact_match.start()),
                        "why_it_matters": "Failed artifact creation can be silently skipped while the poll still reports success and may advance state.",
                        "suggested_fix": "Check the `(success, message, artifact_id)` result from save_artifact(), fail or count the poll on errors, and only advance checkpoint state after successful ingestion.",
                    }
                )

            dynamic_cef_match = re.search(r"['\"]cef['\"]\s*:\s*\{\s*[A-Za-z_][A-Za-z0-9_]*\s*:", body)
            if dynamic_cef_match:
                findings.append(
                    {
                        "title": "Polling artifacts use dynamic CEF keys",
                        "category": "soar_metadata",
                        "severity": "medium",
                        "confidence": "high",
                        "file": path,
                        "line": start_line + body[: dynamic_cef_match.start()].count("\n"),
                        "code_reference": function_name,
                        "evidence": line_from_match(body, dynamic_cef_match.start()),
                        "why_it_matters": "CEF keys should be stable, standard fields with contains mappings. Using vendor-provided indicator types as keys weakens playbook recommendations and can create inconsistent artifact schemas.",
                        "suggested_fix": "Map each vendor indicator type to a fixed CEF field such as `fileHash`, `sourceAddress`, `requestURL`, or `destinationDnsDomain`, and set matching `contains` metadata.",
                    }
                )

            if "save_artifact" in lowered and re.search(r"['\"]label['\"]\s*:", body) is None:
                findings.append(
                    {
                        "title": "Polling artifacts are missing artifact labels",
                        "category": "soar_metadata",
                        "severity": "medium",
                        "confidence": "high",
                        "file": path,
                        "line": start_line + body[: lowered.find("save_artifact")].count("\n"),
                        "code_reference": function_name,
                        "evidence": f"{function_name}() saves artifacts but no artifact `label` key was found.",
                        "why_it_matters": "Artifact labels help SOAR route, display, and triage ingested data consistently.",
                        "suggested_fix": "Set a stable artifact label, preferably from configuration or a documented connector default, before saving each artifact.",
                    }
                )

            if "save_artifact" in lowered and re.search(r"['\"]name['\"]\s*:\s*indicator_value\b", body):
                findings.append(
                    {
                        "title": "Polling artifact names are raw indicator values",
                        "category": "soar_metadata",
                        "severity": "low",
                        "confidence": "high",
                        "file": path,
                        "line": find_line(body, "indicator_value") + start_line - 1 if find_line(body, "indicator_value") else start_line,
                        "code_reference": function_name,
                        "evidence": "Artifact `name` is assigned directly from `indicator_value`.",
                        "why_it_matters": "Raw indicator values make artifact lists noisy and can expose sensitive or high-cardinality values where a stable descriptive name would be clearer.",
                        "suggested_fix": "Use a descriptive artifact name based on indicator type or vendor object type, and keep the raw indicator in CEF/source_data_identifier.",
                    }
                )
    return findings


def check_unbounded_poll_pagination(full_files: dict[str, str]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    bound_terms = ("max_pages", "max_results", "container_count", "artifact_count", "page_count", "total_limit")
    for path, text in full_files.items():
        if not path.endswith(".py"):
            continue
        for function_name, start_line, body in extract_functions_matching(text, r"poll|ioc|dashboard"):
            lowered = body.lower()
            if "while true" not in lowered:
                continue
            if not any(term in lowered for term in ("offset", "page", "limit")):
                continue
            if any(term in lowered for term in bound_terms):
                continue
            match = re.search(r"while\s+True\s*:", body)
            findings.append(
                {
                    "title": "Polling or dashboard pagination has no maximum page guard",
                    "category": "pagination",
                    "severity": "medium",
                    "confidence": "high",
                    "file": path,
                    "line": start_line + body[: match.start()].count("\n") if match else start_line,
                    "code_reference": function_name,
                    "evidence": f"{function_name}() loops through pages until the API returns fewer records, with no visible max page/result guard.",
                    "why_it_matters": "A bad API response or very large feed can keep a poll/view loop running for too many pages and consume worker or web-process resources.",
                    "suggested_fix": "Add a configurable maximum page/result guard and stop with a clear warning or partial-result summary when the limit is reached.",
                }
            )
    return findings


def check_polling_state_loaded_but_not_checkpointed(full_files: dict[str, str]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    all_python = "\n".join(text for path, text in full_files.items() if path.endswith(".py")).lower()
    if "load_state" not in all_python and "save_state" not in all_python and "self._state" not in all_python:
        return []
    for path, text in full_files.items():
        if not path.endswith(".py"):
            continue
        for function_name, start_line, body in extract_functions_matching(text, r"poll"):
            lowered = body.lower()
            if not any(term in lowered for term in ("save_container", "save_artifact", "feed", "offset", "poll")):
                continue
            uses_state = "self._state" in lowered or "save_state" in lowered or "load_state" in lowered
            has_today_window = any(term in lowered for term in ("datetime.now", "date.today", "today =", "utcnow"))
            has_paging = "offset" in lowered or "limit" in lowered or "while true" in lowered
            if uses_state or not (has_today_window or has_paging):
                continue
            findings.append(
                {
                    "title": "Polling state is loaded but not used as an ingestion checkpoint",
                    "category": "polling_checkpoint",
                    "severity": "high",
                    "confidence": "high",
                    "file": path,
                    "line": start_line,
                    "code_reference": function_name,
                    "evidence": f"The connector has state handling elsewhere, but {function_name}() does not read or update `self._state` while it polls by time/window or offset.",
                    "why_it_matters": "Scheduled polling can repeatedly reprocess the same feed window and cannot recover cleanly after a partial ingestion failure.",
                    "suggested_fix": "Persist a checkpoint such as last successful vendor cursor/date/offset after successful ingestion, and keep Poll Now state separate from scheduled polling state.",
                }
            )
    return findings


def check_pagination_heuristics(
    full_files: dict[str, str],
    app_jsons: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    findings = []
    list_identifiers: set[str] = set()
    for app_json in app_jsons.values():
        for action in app_json.get("actions", []):
            identifier = str(action.get("identifier") or "")
            name = str(action.get("action") or "")
            normalized = identifier or name.replace(" ", "_").lower()
            if normalized.startswith(("list", "search", "query")) or normalized.endswith("_list"):
                list_identifiers.add(identifier)

    if not list_identifiers:
        return findings

    for path, text in full_files.items():
        if not path.endswith(".py"):
            continue
        for function_name, start_line, body in extract_handle_functions(text):
            if function_name not in list_identifiers:
                continue
            lowered = body.lower()
            has_rest_call = any(word in lowered for word in ("requests.", "_make_rest_call", "make_request", "session."))
            has_pagination = any(word in lowered for word in PAGINATION_WORDS)
            if not has_rest_call or has_pagination:
                continue
            findings.append(
                {
                    "title": "List/search action has no visible pagination handling",
                    "category": "pagination",
                    "severity": "medium",
                    "confidence": "low",
                    "file": path,
                    "line": start_line,
                    "evidence": f"_handle_{function_name}() performs a REST call but has no visible page/offset/limit/cursor/next handling.",
                    "why_it_matters": "List actions that only fetch the first page can silently truncate results and mislead users.",
                    "suggested_fix": "Verify the vendor API pagination contract and implement page/offset/cursor traversal or document a deliberate limit.",
                }
            )
    return findings


def check_validation_heuristics(full_files: dict[str, str]) -> list[dict[str, Any]]:
    findings = []
    fstring_url = re.compile(r"\b(endpoint|url|uri|path)\s*=\s*f[\"'][^\"'\n]*[/][^\"'\n]*\{[^}]+\}[^\"'\n]*[\"']")

    for path, text in full_files.items():
        if not path.endswith(".py"):
            continue

        for function_name, start_line, body in extract_handle_functions(text):
            if "quote(" in body or "urlencode(" in body or "quote_plus(" in body:
                continue
            match = fstring_url.search(body)
            if not match:
                continue
            findings.append(
                {
                    "title": "Dynamic ID is interpolated into a URL without visible encoding",
                    "category": "validation",
                    "severity": "medium",
                    "confidence": "medium",
                    "file": path,
                    "line": start_line + body[: match.start()].count("\n"),
                    "evidence": line_from_match(body, match.start()),
                    "why_it_matters": "Unencoded IDs containing spaces, slashes, or reserved characters can call the wrong endpoint or fail against valid vendor IDs.",
                    "suggested_fix": "Encode path components with urllib.parse.quote or build query strings with urlencode before making the request.",
                }
            )
    return findings


def check_custom_view_action_result_misuse(full_files: dict[str, str]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for path, text in full_files.items():
        if not path.endswith(".py") or "view" not in path.lower():
            continue
        lowered = text.lower()
        if "phantom.action_result.actionresult" in lowered:
            has_action_result_import = (
                "import phantom.action_result" in lowered
                or "from phantom.action_result import" in lowered
            )
            if not has_action_result_import:
                line = find_line(text, "phantom.action_result.ActionResult") or 1
                findings.append(
                    {
                        "title": "Custom view references phantom.action_result without importing it",
                        "category": "validation",
                        "severity": "high",
                        "confidence": "high",
                        "file": path,
                        "line": line,
                        "code_reference": "ActionResult",
                        "evidence": "`phantom.action_result.ActionResult(...)` is used, but the view module does not import `phantom.action_result` or `ActionResult` directly.",
                        "why_it_matters": "The custom view can fail before rendering unless `phantom.app` happens to expose an unrelated `action_result` attribute.",
                        "suggested_fix": "Import `ActionResult` explicitly with the same pattern used by the connector, then instantiate that imported class.",
                    }
                )

        if "add_action_result" in text and "get_data()" in text and re.search(r"\bhandler\s*\(\s*param\s*\)", text):
            line = find_line(text, "add_action_result") or find_line(text, "handler(param)") or 1
            findings.append(
                {
                    "title": "Custom view renders a different ActionResult than the handler populates",
                    "category": "validation",
                    "severity": "high",
                    "confidence": "high",
                    "file": path,
                    "line": line,
                    "code_reference": "_render_enrichment_view",
                    "evidence": "The view creates one ActionResult, calls a connector handler that creates/populates its own ActionResult, then renders data from the original empty ActionResult.",
                    "why_it_matters": "The custom view can show no data even when the enrichment API call succeeds.",
                    "suggested_fix": "Render the completed action result data passed by SOAR instead of re-calling the handler, or have the helper use the same ActionResult object that the view renders.",
                }
            )

        api_recall_terms = ("_handle_", "_make_rest_call", "requests.", "session.", "connector.")
        if "request.get.get" in lowered and any(term in lowered for term in api_recall_terms):
            for function_name, start_line, body in extract_functions_matching(text, r"view"):
                body_lower = body.lower()
                if "request.get.get" not in body_lower or not any(term in body_lower for term in api_recall_terms):
                    continue
                if re.search(r"if\s+not\s+[^:\n]+:", body) or "missing required" in body_lower:
                    continue
                findings.append(
                    {
                        "title": "Custom view re-calls API with unchecked GET parameters",
                        "category": "validation",
                        "severity": "high",
                        "confidence": "medium",
                        "file": path,
                        "line": start_line,
                        "code_reference": function_name,
                        "evidence": f"{function_name}() reads request.GET values and calls connector behavior without a visible missing-value guard.",
                        "why_it_matters": "SOAR custom action views should render completed action result data. Re-calling APIs from GET parameters can send None values, consume quota, and show different data than the action result.",
                        "suggested_fix": "Use existing action result data for custom views; if a standalone view must call the API, validate each required GET parameter and return HTTP 400 before making the call.",
                    }
                )
                break
    return findings


def check_dashboard_view_not_wired(
    review_input: dict[str, Any],
    full_files: dict[str, str],
) -> list[dict[str, Any]]:
    dashboard_paths = [
        str(item.get("filename") or "")
        for item in review_input.get("changed_files", [])
        if isinstance(item, dict) and "dashboard" in str(item.get("filename") or "").lower()
    ]
    if not dashboard_paths:
        return []

    dashboard_text = "\n".join(file_text_from_review(review_input, full_files, path) for path in dashboard_paths).lower()
    if "<dashboard" not in dashboard_text:
        return []

    template_text = "\n".join(
        file_text_from_review(review_input, full_files, str(item.get("filename") or ""))
        for item in review_input.get("changed_files", [])
        if isinstance(item, dict) and str(item.get("filename") or "").endswith(".html")
    ).lower()
    view_candidates = [
        (path, text)
        for path, text in full_files.items()
        if path.endswith(".py") and "view" in path.lower()
    ]
    if not view_candidates:
        return []

    has_empty_dashboard_panel = re.search(r"<html(?:\s+[^>]*)?>\s*(?:<div[^>]*>\s*</div>)?\s*</html>", dashboard_text, re.S) is not None
    has_template_date_input = "type=\"date\"" in template_text or "type='date'" in template_text or "name=\"date\"" in template_text

    for path, text in view_candidates:
        for function_name, start_line, body in extract_functions_matching(text, r"ioc|dashboard|view"):
            body_lower = body.lower()
            if "datetime.now" not in body_lower and "date.today" not in body_lower and "today =" not in body_lower:
                continue
            reads_date_input = any(
                term in body_lower
                for term in (
                    "request.get.get(\"date\"",
                    "request.get.get('date'",
                    "request.post.get(\"date\"",
                    "request.post.get('date'",
                    "request.args.get(\"date\"",
                    "request.args.get('date'",
                )
            )
            if reads_date_input:
                continue
            if not has_empty_dashboard_panel and not has_template_date_input:
                continue
            evidence_bits = []
            if has_empty_dashboard_panel:
                evidence_bits.append("dashboard XML contains empty HTML panels instead of wiring the custom view")
            if has_template_date_input:
                evidence_bits.append("the HTML template includes a date input")
            evidence_bits.append(f"{function_name}() always derives the feed date from the current day")
            return [
                {
                    "title": "Dashboard/date filter UI is not wired to the custom view",
                    "category": "docs_pr_accuracy",
                    "severity": "medium",
                    "confidence": "high",
                    "file": path,
                    "line": start_line,
                    "code_reference": function_name,
                    "evidence": "; ".join(evidence_bits) + ".",
                    "why_it_matters": "A dashboard that claims or shows date-bounded IOC filtering will not actually honor the selected date, and users can get a blank or misleading dashboard.",
                    "suggested_fix": "Wire the dashboard XML to the custom view endpoint and read/validate the date input in the view before building the feed start/end timestamps.",
                }
            ]
    return []


def file_text_from_review(review_input: dict[str, Any], full_files: dict[str, str], path: str) -> str:
    if path in full_files:
        return str(full_files.get(path) or "")
    for item in review_input.get("changed_files", []):
        if isinstance(item, dict) and item.get("filename") == path:
            patch = str(item.get("patch") or "")
            lines = []
            for raw_line in patch.splitlines():
                if raw_line.startswith("+") and not raw_line.startswith("+++"):
                    lines.append(raw_line[1:])
            return "\n".join(lines)
    return ""


def check_unsafe_response_index_after_count(full_files: dict[str, str]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for path, text in full_files.items():
        if not path.endswith(".py"):
            continue
        for function_name, start_line, body in extract_functions_matching(text, r"poll|container|lookup"):
            if not re.search(r"(?:data|response_json)\.get\(\s*['\"]count['\"]\s*,\s*0\s*\)\s*>\s*0", body):
                continue
            index_match = re.search(r"(?:data|response_json)\s*\[\s*['\"]data['\"]\s*\]\s*\[\s*0\s*\]\s*\[\s*['\"]id['\"]\s*\]", body)
            if not index_match:
                continue
            findings.append(
                {
                    "title": "Container lookup indexes response data after only checking count",
                    "category": "validation",
                    "severity": "medium",
                    "confidence": "high",
                    "file": path,
                    "line": start_line + body[: index_match.start()].count("\n"),
                    "code_reference": function_name,
                    "evidence": line_from_match(body, index_match.start()),
                    "why_it_matters": "A stale or malformed API response with count > 0 but an empty data array can crash polling with KeyError/IndexError instead of returning a controlled connector error.",
                    "suggested_fix": "Validate that `data` is a non-empty list and the first item has an `id` before indexing, otherwise return a clear ActionResult error.",
                }
            )
    return findings


def check_unknown_action_success_dispatch(full_files: dict[str, str]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for path, text in full_files.items():
        if not path.endswith(".py"):
            continue
        body = function_body_text(text, "handle_action")
        if not body:
            continue
        lowered = body.lower()
        if "ret_val = phantom.app_success" not in lowered:
            continue
        if "action_id in action_mapping" not in body and "action_id in action_map" not in body:
            continue
        if re.search(r"else\s*:\s*(?:\n\s+)*(?:return\s+)?(?:action_result\.)?set_status\([^)]*APP_ERROR", body, re.I | re.S):
            continue
        if "app_error" in lowered and "unknown" in lowered:
            continue
        findings.append(
            {
                "title": "Unknown action identifiers return success silently",
                "category": "validation",
                "severity": "medium",
                "confidence": "high",
                "file": path,
                "line": find_line(text, "ret_val = phantom.APP_SUCCESS") or find_line(text, "def handle_action") or 1,
                "code_reference": "handle_action",
                "evidence": "handle_action initializes ret_val to APP_SUCCESS and only dispatches known actions without a visible APP_ERROR path for unknown action_id values.",
                "why_it_matters": "Manifest/dispatch mistakes can be hidden as successful actions instead of surfacing a clear unsupported-action error.",
                "suggested_fix": "Return APP_ERROR with an explicit unknown-action message when action_id is not present in the action mapping.",
            }
        )
    return findings


def check_jsonl_parser_swallows_malformed_lines(full_files: dict[str, str]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for path, text in full_files.items():
        if not path.endswith(".py"):
            continue
        for function_name, start_line, body in extract_functions_matching(text, r"jsonl|parse"):
            if "json.loads" not in body:
                continue
            if not re.search(r"except\s+(?:ValueError|json\.JSONDecodeError)", body):
                continue
            if not re.search(r"except\s+(?:ValueError|json\.JSONDecodeError)[^:]*:\s*\n\s*continue\b", body):
                continue
            findings.append(
                {
                    "title": "JSONL parser silently drops malformed lines",
                    "category": "validation",
                    "severity": "medium",
                    "confidence": "high",
                    "file": path,
                    "line": start_line,
                    "code_reference": function_name,
                    "evidence": f"{function_name}() catches JSON decode errors and continues without recording or failing the malformed line.",
                    "why_it_matters": "Feed corruption or vendor response changes can produce partial ingestion while the action still reports success.",
                    "suggested_fix": "Count malformed lines in the summary/debug output or fail the action with a clear error instead of silently dropping them.",
                }
            )
    return findings


def check_generic_rate_limit_handling(full_files: dict[str, str]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    combined = "\n".join(text for path, text in full_files.items() if path.endswith(".py")).lower()
    if "status_code" not in combined or "response" not in combined:
        return []
    if "429" in combined or "retry-after" in combined or "x-rate-limit" in combined or "ratelimit" in combined:
        return []
    if not any(term in combined for term in ("requests.", "_make_rest_call", "request_func", "httpx.", "response.status_code")):
        return []
    for path, text in full_files.items():
        if not path.endswith(".py"):
            continue
        if "status_code" not in text or "response" not in text:
            continue
        findings.append(
            {
                "title": "HTTP 429 rate limits are handled as generic API errors",
                "category": "api_auth_correctness",
                "severity": "medium",
                "confidence": "medium",
                "file": path,
                "line": find_line(text, "status_code") or 1,
                "code_reference": "HTTP 429",
                "evidence": "The connector handles HTTP status codes, but no explicit 429, Retry-After, or rate-limit header handling was found.",
                "why_it_matters": "Rate-limit responses need clear retry/backoff messaging; generic server errors make failures harder for contributors and users to diagnose.",
                "suggested_fix": "Handle HTTP 429 separately, surface Retry-After or rate-limit reset details when present, and add bounded retry/backoff where the action can safely retry.",
            }
        )
        break
    return findings


def check_test_connectivity_uses_quota_action_endpoint(full_files: dict[str, str]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    sample_indicator_pattern = re.compile(r"['\"](?:cyberint\.com|example\.com|test\.com|8\.8\.8\.8|1\.1\.1\.1)['\"]", re.I)
    for path, text in full_files.items():
        if not path.endswith(".py"):
            continue
        for function_name, start_line, body in extract_handle_functions(text):
            if function_name != "test_connectivity":
                continue
            lowered = body.lower()
            if not any(term in lowered for term in ("enrich", "lookup", "detonate", "search")):
                continue
            if not sample_indicator_pattern.search(body):
                continue
            findings.append(
                {
                    "title": "Test connectivity consumes a real action endpoint with a fixed indicator",
                    "category": "api_auth_correctness",
                    "severity": "medium",
                    "confidence": "high",
                    "file": path,
                    "line": start_line,
                    "code_reference": "_handle_test_connectivity",
                    "evidence": f"{function_name}() calls a real lookup/enrichment-style endpoint using a fixed sample indicator.",
                    "why_it_matters": "Test connectivity can consume action quota and may validate only one endpoint while missing the feed/polling API path the asset also depends on.",
                    "suggested_fix": "Use the vendor's lightweight auth/status/quota endpoint when available, or explicitly test both required API paths without consuming enrichment/detonation quota.",
                }
            )
    return findings


def check_branch_local_exception_imports(full_files: dict[str, str]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    exception_names = {"ActionFailure", "AssetMisconfiguration", "ConnectorError"}
    for path, text in full_files.items():
        if not path.endswith(".py"):
            continue
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        functions = [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
        for function in functions:
            loads_by_name: dict[str, list[int]] = {}
            for node in ast.walk(function):
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                    loads_by_name.setdefault(node.id, []).append(node.lineno)

            for imported_name, import_line, block_end in nested_imported_names(function):
                if imported_name not in exception_names:
                    continue
                later_loads = [line for line in loads_by_name.get(imported_name, []) if line > block_end]
                if not later_loads:
                    continue
                findings.append(
                    {
                        "title": "Branch-local exception import can raise UnboundLocalError",
                        "category": "validation",
                        "severity": "high",
                        "confidence": "high",
                        "file": path,
                        "line": import_line,
                        "code_reference": f"{function.name}.{imported_name}",
                        "evidence": f"`{imported_name}` is imported inside a nested branch of `{function.name}` and then referenced after that branch at line {later_loads[0]}.",
                        "why_it_matters": "Python treats the imported name as local to the whole function; if the import branch is skipped, later error paths raise UnboundLocalError instead of the intended SOAR connector error.",
                        "suggested_fix": f"Move `from soar_sdk.exceptions import {imported_name}` to module scope or the top of `{function.name}` before any branch that can raise it.",
                    }
                )
                break
    return findings


def nested_imported_names(function: ast.FunctionDef | ast.AsyncFunctionDef) -> list[tuple[str, int, int]]:
    output: list[tuple[str, int, int]] = []

    def visit_statements(statements: list[ast.stmt], *, nested_block_end: int | None = None) -> None:
        for stmt in statements:
            if isinstance(stmt, (ast.Import, ast.ImportFrom)) and nested_block_end is not None:
                for alias in stmt.names:
                    name = alias.asname or alias.name.rsplit(".", 1)[-1]
                    output.append((name, stmt.lineno, nested_block_end))
                continue
            child_block_end = getattr(stmt, "end_lineno", stmt.lineno)
            for field_name in ("body", "orelse", "finalbody"):
                value = getattr(stmt, field_name, None)
                if isinstance(value, list):
                    visit_statements(value, nested_block_end=child_block_end)
            handlers = getattr(stmt, "handlers", None)
            if isinstance(handlers, list):
                for handler in handlers:
                    visit_statements(
                        getattr(handler, "body", []),
                        nested_block_end=getattr(handler, "end_lineno", child_block_end),
                    )

    visit_statements(function.body)
    return output


def check_sdk_numeric_validation_regressions(
    review_input: dict[str, Any],
    full_files: dict[str, str],
) -> list[dict[str, Any]]:
    if not build_sdk_review_inventory(review_input, full_files).get("is_sdk_migration"):
        return []

    risky_param_pattern = re.compile(
        r"int\s*\(\s*params\.(?P<name>(?:return_when|wait_for|timeout|count|percent|percentage)[A-Za-z0-9_]*)\s*\)"
    )
    for path, text in full_files.items():
        if not path.endswith(".py"):
            continue
        for match in risky_param_pattern.finditer(text):
            param_name = match.group("name")
            window = text[max(0, match.start() - 700) : min(len(text), match.end() + 700)]
            validation_terms = (
                f"{param_name} < 0",
                f"{param_name} <= 0",
                ".is_integer()",
                "must be an integer",
                "non-negative",
            )
            if any(term in window for term in validation_terms):
                continue
            return [
                {
                    "title": "SDK migration coerces numeric action parameters without validation",
                    "category": "validation",
                    "severity": "high",
                    "confidence": "high",
                    "file": path,
                    "line": text[: match.start()].count("\n") + 1,
                    "code_reference": f"params.{param_name}",
                    "evidence": f"`params.{param_name}` is converted with `int(...)` without a nearby integer/non-negative validation guard.",
                    "why_it_matters": "Silently truncating fractional values changes user input semantics from the legacy connector and can send invalid polling thresholds or timeouts into API wait loops.",
                    "suggested_fix": "Reject fractional and negative values with a clear ActionFailure before coercion, and preserve the legacy validation behavior for timeout/result-threshold parameters.",
                }
            ]

    for path, text in full_files.items():
        if not path.endswith(".py") or "sleep(timeout_seconds - 1)" not in text:
            continue
        return [
            {
                "title": "SDK migration allows timeout values that can make sleep negative",
                "category": "validation",
                "severity": "high",
                "confidence": "high",
                "file": path,
                "line": find_line(text, "sleep(timeout_seconds - 1)") or 1,
                "code_reference": "timeout_seconds",
                "evidence": "`sleep(timeout_seconds - 1)` can receive a negative value if action parameter validation allows values between 0 and 1.",
                "why_it_matters": "A user-supplied timeout should fail validation with a connector error, not surface as an unhandled ValueError from the Python runtime.",
                "suggested_fix": "Validate timeout_seconds before polling so it is an integer or bounded positive value large enough for the wait loop.",
            }
        ]
    return []


def check_sdk_validation_cookbook_regressions(
    review_input: dict[str, Any],
    full_files: dict[str, str],
) -> list[dict[str, Any]]:
    if not build_sdk_review_inventory(review_input, full_files).get("is_sdk_migration"):
        return []

    findings: list[dict[str, Any]] = []
    numeric = sdk_bounded_numeric_without_validation(full_files)
    if numeric:
        path, line, param_name = numeric
        findings.append(
            {
                "title": "SDK action coerces bounded numeric parameter without range validation",
                "category": "validation",
                "severity": "medium",
                "confidence": "high",
                "file": path,
                "line": line,
                "code_reference": f"params.{param_name}",
                "evidence": f"`params.{param_name}` is coerced with `int(...)` without a nearby range/positive-integer validation guard.",
                "why_it_matters": "SDK type hints do not preserve legacy validation semantics by themselves; page sizes, limits, offsets, counts, and durations can be silently truncated or sent outside vendor/API bounds.",
                "suggested_fix": "Validate integer-ness and the allowed range before coercion, then raise ActionFailure with the parameter name when the value is fractional, zero/negative, or outside the documented bounds.",
            }
        )

    split_list = sdk_list_split_without_normalization(full_files)
    if split_list:
        path, line, param_name = split_list
        findings.append(
            {
                "title": "SDK action splits a list parameter without removing blank values",
                "category": "validation",
                "severity": "medium",
                "confidence": "high",
                "file": path,
                "line": line,
                "code_reference": f"params.{param_name}",
                "evidence": f"`params.{param_name}.split(',')` is used without nearby stripping/filtering of empty entries.",
                "why_it_matters": "Allow-list and comma-separated parameters often arrive with whitespace or trailing commas; sending blank entries to an API can change filters, produce confusing errors, or bypass intended validation.",
                "suggested_fix": "Normalize with `[item.strip() for item in value.split(',') if item.strip()]`, reject an empty normalized list when required, and add validation tests for whitespace/trailing-comma inputs.",
            }
        )

    path_segment = sdk_path_segment_without_encoding(full_files)
    if path_segment:
        path, line, param_name = path_segment
        findings.append(
            {
                "title": "SDK action interpolates a parameter into a URL path without encoding",
                "category": "validation",
                "severity": "medium",
                "confidence": "high",
                "file": path,
                "line": line,
                "code_reference": f"params.{param_name}",
                "evidence": f"`params.{param_name}` is formatted directly into a URL path segment without visible `quote(...)` or equivalent encoding.",
                "why_it_matters": "Identifiers such as folders, repository names, tenants, users, and vendor object IDs can contain `/`, spaces, `#`, or query characters; unencoded path interpolation can route the request to the wrong object or fail outside connector validation.",
                "suggested_fix": "URL-encode path parameters with `urllib.parse.quote(value, safe='')` before formatting them into endpoints, or validate the parameter against the exact vendor path-segment grammar.",
            }
        )

    return findings


def sdk_bounded_numeric_without_validation(full_files: dict[str, str]) -> tuple[str, int, str] | None:
    risky_names = re.compile(
        r"(?:page|page_size|per_page|limit|offset|max|size|count|days|hours|minutes|timeout|duration|results?)",
        re.I,
    )
    int_pattern = re.compile(r"int\s*\(\s*params\.(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\)")
    skip_prefixes = ("return_when", "wait_for", "issue_number")
    for path, text in full_files.items():
        if not path.endswith(".py") or is_test_path(path):
            continue
        for function_name, start_line, body in python_function_bodies(text):
            del function_name
            for match in int_pattern.finditer(body):
                name = match.group("name")
                if name.startswith(skip_prefixes) or not risky_names.search(name):
                    continue
                window = body[max(0, match.start() - 900) : min(len(body), match.end() + 900)]
                if sdk_numeric_validation_present(window, name):
                    continue
                return path, start_line + body[: match.start()].count("\n"), name
    return None


def sdk_numeric_validation_present(window: str, name: str) -> bool:
    lowered = window.lower()
    normalized = name.lower()
    validation_terms = (
        f"{normalized} <",
        f"{normalized}<",
        f"{normalized} >",
        f"{normalized}>",
        f"{normalized} <=",
        f"{normalized}<=",
        f"{normalized} >=",
        f"{normalized}>=",
        f"params.{normalized} <",
        f"params.{normalized}<",
        f"params.{normalized} >",
        f"params.{normalized}>",
        ".is_integer()",
        "positive integer",
        "non-negative",
        "out of range",
        "must be",
        "actionfailure",
        "ge=",
        "gt=",
        "le=",
        "lt=",
    )
    return any(term in lowered for term in validation_terms)


def sdk_list_split_without_normalization(full_files: dict[str, str]) -> tuple[str, int, str] | None:
    split_pattern = re.compile(r"params\.(?P<name>[A-Za-z_][A-Za-z0-9_]*)\.split\s*\(\s*['\"],['\"]\s*\)")
    for path, text in full_files.items():
        if not path.endswith(".py") or is_test_path(path):
            continue
        for function_name, start_line, body in python_function_bodies(text):
            del function_name
            for match in split_pattern.finditer(body):
                window = body[max(0, match.start() - 500) : min(len(body), match.end() + 900)]
                if any(term in window for term in (".strip()", "if item", "if value", "filter(None", "not item.strip")):
                    continue
                return path, start_line + body[: match.start()].count("\n"), match.group("name")
    return None


def sdk_path_segment_without_encoding(full_files: dict[str, str]) -> tuple[str, int, str] | None:
    path_param_pattern = re.compile(
        r"f[\"'][^\"'\n]*[/][^\"'\n]*\{[^}]*params\.(?P<name>[A-Za-z_][A-Za-z0-9_]*)[^}]*\}[^\"'\n]*[\"']"
    )
    path_name_terms = (
        "id",
        "name",
        "path",
        "folder",
        "file",
        "drive",
        "repo",
        "owner",
        "user",
        "tenant",
        "domain",
        "url",
        "ip",
        "group",
    )
    for path, text in full_files.items():
        if not path.endswith(".py") or is_test_path(path):
            continue
        for function_name, start_line, body in python_function_bodies(text):
            del function_name
            for match in path_param_pattern.finditer(body):
                name = match.group("name")
                if not any(term in name.lower() for term in path_name_terms):
                    continue
                window = body[max(0, match.start() - 800) : min(len(body), match.end() + 800)]
                if any(term in window for term in ("quote(", "quote_plus(", "urlencode(", "urllib.parse.quote")):
                    continue
                return path, start_line + body[: match.start()].count("\n"), name
    return None


def python_function_bodies(text: str) -> list[tuple[str, int, str]]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    lines = text.splitlines()
    bodies: list[tuple[str, int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        end = getattr(node, "end_lineno", node.lineno)
        bodies.append((node.name, node.lineno, "\n".join(lines[node.lineno - 1 : end])))
    return bodies


def check_graph_target_user_id_normalization(full_files: dict[str, str]) -> list[dict[str, Any]]:
    combined_text = "\n".join(full_files.values()).lower()
    if "target_user_id" not in combined_text:
        return []
    graph_markers = ("microsoft graph", "graph.microsoft", "/users/", "onedrive", "client credentials")
    if not any(marker in combined_text for marker in graph_markers):
        return []

    for path, text in full_files.items():
        if not path.endswith(".py") or "target_user_id" not in text:
            continue
        lines = text.splitlines()
        for index, line in enumerate(lines, start=1):
            compact = line.strip()
            if "/users/" not in compact or "target_user_id" not in compact:
                continue
            if ".strip(" in compact:
                continue
            before = "\n".join(lines[max(0, index - 18) : index])
            if target_user_id_is_normalized_nearby(before):
                continue
            return [
                {
                    "title": "Client Credentials target_user_id is used in a Graph URL without normalization",
                    "category": "validation",
                    "severity": "medium",
                    "confidence": "high",
                    "file": path,
                    "line": index,
                    "code_reference": compact,
                    "evidence": (
                        "`target_user_id` is formatted into a Microsoft Graph `/users/{...}` route "
                        "without a nearby `.strip()` or normalization step."
                    ),
                    "why_it_matters": (
                        "Client Credentials assets often rely on target_user_id for user-drive routes. "
                        "Leading or trailing spaces from asset configuration can produce invalid Graph URLs "
                        "even though the rest of the asset validates."
                    ),
                    "suggested_fix": (
                        "Normalize the asset value once, for example "
                        "`target_user_id = (asset.target_user_id or '').strip()`, validate it is non-empty, "
                        "and use that normalized value for every `/users/{target_user_id}` route."
                    ),
                }
            ]
    return []


def target_user_id_is_normalized_nearby(text: str) -> bool:
    assignment_lines = [
        line.strip()
        for line in text.splitlines()
        if "target_user_id" in line and ("=" in line or ":=" in line)
    ]
    return any(".strip(" in line for line in assignment_lines)


def check_obviously_wrong_output_cef_types(full_files: dict[str, str]) -> list[dict[str, Any]]:
    for path, text in full_files.items():
        if not path.endswith(".py") or "file path" not in text:
            continue
        lines = text.splitlines()
        for index, line in enumerate(lines, start=1):
            if not re.search(r"\bdisplay_?name\b|displayName", line):
                continue
            window = "\n".join(lines[index - 1 : min(len(lines), index + 8)])
            if "OutputField" not in window or "file path" not in window:
                continue
            return [
                {
                    "title": "Output CEF type marks a display name as a file path",
                    "category": "output_schema_mismatch",
                    "severity": "low",
                    "confidence": "high",
                    "file": path,
                    "line": index,
                    "code_reference": line.strip(),
                    "evidence": "`displayName` is modeled with a `file path` CEF type.",
                    "why_it_matters": (
                        "CEF types drive SOAR field recommendations and playbook ergonomics. "
                        "A user display name should not be advertised as a filesystem path."
                    ),
                    "suggested_fix": (
                        "Remove the `file path` CEF type from the display-name output field, or replace it "
                        "with a semantically correct CEF type if one exists."
                    ),
                }
            ]
    return []


def check_search_query_user_input_interpolation(full_files: dict[str, str]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    raw_patterns = (
        r"q\s*=\s*[^'\"]*domain:\{?\s*params\.domain",
        r"q\s*=\s*[^'\"]*ip:\{?\s*params\.ip",
        r"search/\?q=[^'\"]*\{?\s*params\.(?:domain|ip|query)",
        r"domain:\{params\.domain\}",
        r"ip:\{params\.ip\}",
        r"\.format\(\s*params\.(?:domain|ip|query)\s*\)",
    )
    safety_terms = (
        "_escape",
        "quote(",
        "quote_plus(",
        "urlencode(",
        "params={",
        "params = {",
        "field_validator",
        "ipaddress.ip_address",
    )
    for path, text in full_files.items():
        if not path.endswith(".py"):
            continue
        lowered = text.lower()
        if "search" not in lowered or not any(term in lowered for term in ("params.domain", "params.ip", "params.query")):
            continue
        if any(term in text for term in safety_terms):
            continue
        for line_no, line in iter_lines(text):
            compact = line.strip()
            if any(re.search(pattern, compact) for pattern in raw_patterns):
                findings.append(
                    {
                        "title": "Search action interpolates raw user input into query syntax",
                        "category": "validation",
                        "severity": "high",
                        "confidence": "high",
                        "file": path,
                        "line": line_no,
                        "code_reference": compact,
                        "evidence": f"`{compact}` builds a search query directly from action parameters without visible URL encoding, escaping, or strict validation.",
                        "why_it_matters": "Search APIs that support operators, grouping, ranges, or reserved characters can change query semantics when user input is inserted directly.",
                        "suggested_fix": "Build the request with query parameters and escape the vendor search syntax, or strictly validate the parameter before formatting it into the search expression.",
                    }
                )
                break
    return findings


def check_search_output_pagination_fields(full_files: dict[str, str]) -> list[dict[str, Any]]:
    combined_text = "\n".join(full_files.values()).lower()
    if "search_api" not in combined_text and "search api" not in combined_text:
        return []
    if "api/v1/search" not in combined_text:
        return []
    for path, text in full_files.items():
        if not path.endswith(".py") or "SearchResultItemOutput" not in text:
            continue
        class_body, line = class_body_by_name(text, "SearchResultItemOutput")
        if not class_body:
            continue
        missing = []
        if "_id" not in class_body and "alias=\"_id\"" not in class_body and "alias='_id'" not in class_body:
            missing.append("_id")
        if re.search(r"^\s*sort\s*:", class_body, re.M) is None:
            missing.append("sort")
        if not missing:
            return []
        return [
            {
                "title": "Search result output model omits pagination identity fields",
                "category": "output_schema_mismatch",
                "severity": "medium",
                "confidence": "high",
                "file": path,
                "line": line,
                "code_reference": "SearchResultItemOutput",
                "evidence": f"`SearchResultItemOutput` does not model {', '.join(missing)} even though the search API exposes stable result identity/pagination fields.",
                "why_it_matters": "Fields such as `_id` and `sort` are useful for output discovery and search_after-style pagination; dropping them in an SDK migration can break playbook authors who rely on documented result paths.",
                "suggested_fix": "Add these fields to the SDK output model, using an alias for `_id` if needed, and regenerate docs/manifest output metadata.",
            }
        ]
    return []


def check_search_api_rate_limit_handling(full_files: dict[str, str]) -> list[dict[str, Any]]:
    combined_text = "\n".join(full_files.values()).lower()
    if "search_api" not in combined_text and "search api" not in combined_text:
        return []
    if "httpx" not in combined_text:
        return []
    if "429" in combined_text and "x-rate-limit" in combined_text:
        return []
    for path, text in full_files.items():
        if not path.endswith(".py"):
            continue
        lowered = text.lower()
        if "status_code" not in lowered or "response" not in lowered:
            continue
        return [
            {
                "title": "Search API client does not surface rate-limit responses distinctly",
                "category": "api_auth_correctness",
                "severity": "medium",
                "confidence": "high",
                "file": path,
                "line": find_line(text, "status_code") or 1,
                "code_reference": "HTTP 429 / X-Rate-Limit-*",
                "evidence": "The Search API HTTP client handles response status codes, but no explicit 429 or X-Rate-Limit header handling was found.",
                "why_it_matters": "Search APIs often document rate limits through HTTP 429 and X-Rate-Limit headers. Generic server errors make retry/backoff and contributor debugging harder, especially for detonation and report polling flows.",
                "suggested_fix": "Handle HTTP 429 separately, include X-Rate-Limit-Limit/Remaining/Reset in the ActionFailure message, and use bounded backoff or Retry-After-style delay when retrying.",
            }
        ]
    return []


def check_stale_url_helper_comments(full_files: dict[str, str]) -> list[dict[str, Any]]:
    combined_python = "\n".join(text for path, text in full_files.items() if path.endswith(".py"))
    if "def _build_lookup_url" in combined_python:
        return []
    for path, text in full_files.items():
        if not path.endswith(".py") or "_build_lookup_url" not in text:
            continue
        for line_no, line in iter_lines(text):
            if "_build_lookup_url" not in line:
                continue
            stripped = line.strip()
            if not stripped.startswith("#"):
                continue
            return [
                {
                    "title": "Comment references a missing lookup URL helper",
                    "category": "docs_pr_accuracy",
                    "severity": "low",
                    "confidence": "high",
                    "file": path,
                    "line": line_no,
                    "code_reference": stripped,
                    "evidence": "A comment refers to `_build_lookup_url()`, but no helper with that name exists in the collected Python files.",
                    "why_it_matters": "Stale comments can hide the real URL-construction behavior during review, especially around search escaping and query parameter handling.",
                    "suggested_fix": "Update the comment to describe the current request-building code, or add the helper and route lookup URL construction through it.",
                }
            ]
    return []


def class_body_by_name(text: str, class_name: str) -> tuple[str, int]:
    lines = text.splitlines()
    start_index = None
    indent = 0
    pattern = re.compile(rf"^(\s*)class\s+{re.escape(class_name)}\b")
    for index, line in enumerate(lines):
        match = pattern.match(line)
        if match:
            start_index = index
            indent = len(match.group(1))
            break
    if start_index is None:
        return "", 0
    end_index = len(lines)
    for index in range(start_index + 1, len(lines)):
        line = lines[index]
        if not line.strip():
            continue
        leading = len(line) - len(line.lstrip())
        if leading <= indent and re.match(r"\s*(?:class|def)\s+", line):
            end_index = index
            break
    return "\n".join(lines[start_index:end_index]), start_index + 1


def check_swallowed_broad_exceptions(review_input: dict[str, Any], full_files: dict[str, str]) -> list[dict[str, Any]]:
    findings = []
    broad_except = re.compile(r"^(?P<indent>\s*)except\s*(?:Exception(?:\s+as\s+\w+)?|BaseException(?:\s+as\s+\w+)?|\([^)]*Exception[^)]*\))?\s*:\s*(?P<trailing>.*)$")

    for path, text in full_files.items():
        if not path.endswith(".py"):
            continue
        lines = text.splitlines()
        for index, line in enumerate(lines, start=1):
            match = broad_except.match(line)
            if not match or not is_changed_line(review_input, path, index):
                continue
            trailing = match.group("trailing").strip()
            block = exception_block_lines(lines, index, len(match.group("indent")))
            block_text = "\n".join([trailing] + block).lower()
            if not swallows_exception(block_text):
                continue
            findings.append(
                {
                    "title": "Broad exception handler swallows connector errors",
                    "category": "validation",
                    "severity": "medium",
                    "confidence": "high",
                    "file": path,
                    "line": index,
                    "code_reference": line.strip(),
                    "evidence": line.strip(),
                    "why_it_matters": "Swallowing broad exceptions can turn API, auth, parsing, or validation failures into false-success SOAR action results.",
                    "suggested_fix": "Catch the expected exception type and return or raise a connector failure with the original error context.",
                }
            )
    return findings


def exception_block_lines(lines: list[str], except_line: int, except_indent: int) -> list[str]:
    block: list[str] = []
    for line in lines[except_line:]:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        if indent <= except_indent:
            break
        block.append(line.strip())
    return block[:8]


def swallows_exception(block_text: str) -> bool:
    if any(phrase in block_text for phrase in ("raise", "actionfailure", "app_error", "set_status", "error_print")):
        return False
    return any(
        re.search(pattern, block_text)
        for pattern in (
            r"\bpass\b",
            r"\bcontinue\b",
            r"\bbreak\b",
            r"\breturn\s+(?:none|true|false|\[\]|\{\}|phantom\.app_success|action_result\.set_status\s*\(\s*phantom\.app_success)",
        )
    )


def check_unreachable_code(review_input: dict[str, Any], full_files: dict[str, str]) -> list[dict[str, Any]]:
    findings = []
    terminator = re.compile(r"^(?P<indent>\s*)(return|raise|break|continue)\b")

    for path, text in full_files.items():
        if not path.endswith(".py"):
            continue
        lines = text.splitlines()
        for index, line in enumerate(lines[:-1], start=1):
            match = terminator.match(line)
            if not match:
                continue
            next_index, next_line = next_code_line(lines, index)
            if next_index is None or next_line is None:
                continue
            if not is_changed_line(review_input, path, next_index):
                continue
            indent = len(match.group("indent"))
            next_indent = len(next_line) - len(next_line.lstrip(" "))
            if next_indent != indent:
                continue
            if next_line.lstrip().startswith(("elif ", "else:", "except ", "finally:")):
                continue
            findings.append(
                {
                    "title": "Code after a terminating statement is unreachable",
                    "category": "validation",
                    "severity": "medium",
                    "confidence": "high",
                    "file": path,
                    "line": next_index,
                    "code_reference": next_line.strip(),
                    "evidence": f"`{next_line.strip()}` follows `{line.strip()}` at the same indentation level.",
                    "why_it_matters": "Unreachable connector code can hide missing validation, skipped cleanup, or outputs that SOAR never emits.",
                    "suggested_fix": "Move the statement before the terminator, remove it, or restructure the branch so the code can execute.",
                }
            )
    return findings


def next_code_line(lines: list[str], current_index: int) -> tuple[int | None, str | None]:
    for index in range(current_index + 1, len(lines) + 1):
        line = lines[index - 1]
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        return index, line
    return None, None


def check_base64_decode_before_try(review_input: dict[str, Any], full_files: dict[str, str]) -> list[dict[str, Any]]:
    findings = []
    changed = changed_paths(review_input)
    for path, text in full_files.items():
        if not path.endswith(".py") or path not in changed:
            continue
        for function_name, start_line, body in extract_functions_matching(text, r".*"):
            lines = body.splitlines()
            first_try_offset = first_try_line_offset(lines)
            if first_try_offset is None or first_try_offset <= 1:
                continue
            has_connector_error_handling = "ActionFailure" in body or "actionfailure" in body.lower() or "set_status" in body
            if not has_connector_error_handling:
                continue
            for offset, line in enumerate(lines[:first_try_offset]):
                if "base64.b64decode" not in line:
                    continue
                absolute_line = start_line + offset
                if not is_changed_line(review_input, path, absolute_line):
                    continue
                findings.append(
                    {
                        "title": "base64.b64decode runs before connector error handling",
                        "category": "error_handling",
                        "severity": "medium",
                        "confidence": "high",
                        "file": path,
                        "line": absolute_line,
                        "evidence": (
                            f"{function_name}() decodes base64 before the first try block, so decode failures bypass "
                            "the connector's cleanup/error-handling path."
                        ),
                        "why_it_matters": "Invalid certificate/key input or malformed base64 can surface as a raw exception instead of a clear SOAR ActionFailure.",
                        "suggested_fix": "Move the base64.b64decode call inside the existing try/except, catch decode errors, and raise ActionFailure with a concise validation message.",
                    }
                )
                break
    return findings


def first_try_line_offset(lines: list[str]) -> int | None:
    for offset, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "try:" or stripped.startswith("try:"):
            return offset
    return None


def check_removed_safety_regressions(review_input: dict[str, Any]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    sdk_migration = build_sdk_review_inventory(review_input).get("is_sdk_migration")
    for path, groups in patch_line_groups(review_input).items():
        if not path.endswith(".py"):
            continue
        if sdk_migration and (changed_file_map(review_input).get(path) or {}).get("status") == "removed":
            continue
        removed_text = "\n".join(groups["removed"])
        added_text = "\n".join(groups["added"])
        if not removed_text.strip():
            continue
        lowered_added = added_text.lower()

        timeout_pattern = re.compile(r"\btimeout\s*=")
        if timeout_pattern.search(removed_text) and "timeout=" not in lowered_added:
            line, evidence = removed_line_evidence(review_input, path, timeout_pattern)
            findings.append(
                {
                    "title": "Diff removes request timeout handling without visible replacement",
                    "category": "api_auth_correctness",
                    "severity": "medium",
                    "confidence": "medium",
                    "file": path,
                    "line": line,
                    "evidence": evidence or "A removed line included timeout=, but no added line in the same file adds timeout=.",
                    "why_it_matters": "Removing request timeouts can make token or API calls hang indefinitely and is easy to miss when only added code is reviewed.",
                    "suggested_fix": "Restore an explicit timeout or show the replacement helper/default that guarantees bounded network calls.",
                }
            )

        tls_pattern = re.compile(r"\bverify\s*=\s*(True|self\.[A-Za-z0-9_]+|[^,\n)]+)")
        if tls_pattern.search(removed_text) and not re.search(r"\bverify\s*=", added_text):
            line, evidence = removed_line_evidence(review_input, path, tls_pattern)
            findings.append(
                {
                    "title": "Diff removes TLS verification handling without visible replacement",
                    "category": "api_auth_correctness",
                    "severity": "medium",
                    "confidence": "medium",
                    "file": path,
                    "line": line,
                    "evidence": evidence or "A removed line included verify=, but no added line in the same file adds verify=.",
                    "why_it_matters": "TLS verification defaults are part of the connector auth/security behavior; removing them can silently weaken API and token calls.",
                    "suggested_fix": "Restore the verify argument or ensure the replacement request helper preserves the same secure TLS verification behavior.",
                }
            )

        validation_pattern = re.compile(r"\b(validate|is_valid|quote|quote_plus|urlencode|ActionFailure|AssetMisconfiguration)\b")
        added_has_validation = validation_pattern.search(added_text) is not None
        if validation_pattern.search(removed_text) and not added_has_validation:
            line, evidence = removed_line_evidence(review_input, path, validation_pattern)
            findings.append(
                {
                    "title": "Diff removes validation or connector-friendly error handling without visible replacement",
                    "category": "validation",
                    "severity": "medium",
                    "confidence": "medium",
                    "file": path,
                    "line": line,
                    "evidence": evidence or "Removed lines included validation/error-handling code, but the added lines do not show a replacement.",
                    "why_it_matters": "Deleting validators, URL encoding, or ActionFailure/AssetMisconfiguration handling can turn bad input or API errors into raw failures.",
                    "suggested_fix": "Keep the validation/error-handling behavior, or replace it with an equivalent path that returns clear connector errors for invalid input and API failures.",
                }
            )
    return findings


def check_removed_tests_for_risky_changes(review_input: dict[str, Any]) -> list[dict[str, Any]]:
    groups = patch_line_groups(review_input)
    if not groups:
        return []
    changed = changed_paths(review_input)
    risky_code_changed = any(
        path.endswith(".py")
        and not is_test_path(path)
        and any(word in f"{path}\n{changed_file_map(review_input).get(path, {}).get('patch', '')}".lower() for word in RISKY_CHANGE_WORDS)
        for path in changed
    )
    if not risky_code_changed:
        return []

    evidence_parts: list[str] = []
    first_path: str | None = None
    first_line: int | None = None
    test_pattern = re.compile(r"\b(def\s+test_|assert|pytest|unittest|mock|responses?)\b", re.I)
    for path, file_groups in groups.items():
        removed_text = "\n".join(file_groups["removed"])
        if not is_test_path(path):
            continue
        file_info = changed_file_map(review_input).get(path) or {}
        if file_info.get("status") != "removed" and not test_pattern.search(removed_text):
            continue
        line, evidence = removed_line_evidence(review_input, path, test_pattern)
        evidence_parts.append(f"{path}: {evidence or 'test file/coverage was removed'}")
        first_path = first_path or path
        first_line = first_line or line

    if not evidence_parts:
        return []
    return [
        {
            "title": "Risky connector behavior changed while test coverage was removed",
            "category": "missing_tests",
            "severity": "medium",
            "confidence": "medium",
            "file": first_path,
            "line": first_line,
            "evidence": "; ".join(evidence_parts[:4]),
            "why_it_matters": "A PR can regress coverage by deleting tests even if it also modifies code; reviewing only added lines misses that risk.",
            "suggested_fix": "Keep or replace the removed tests so the changed auth/session/polling/state/validation/mutating behavior still has coverage.",
        }
    ]


def is_test_path(path: str) -> bool:
    lowered = path.lower()
    return (
        "/test" in lowered
        or "/tests/" in lowered
        or lowered.startswith("test")
        or lowered.startswith("tests/")
        or "test_params" in lowered
        or "app-tests" in lowered
    )


def check_removed_release_notes_for_code_changes(review_input: dict[str, Any]) -> list[dict[str, Any]]:
    files = changed_file_map(review_input)
    has_code_or_json = any(
        path.endswith(".py")
        or (path.endswith(".json") and "/" not in path)
        or path in {"pyproject.toml", "uv.lock"}
        for path in files
    )
    if not has_code_or_json:
        return []
    release_info = files.get("release_notes/unreleased.md")
    if not release_info or release_info.get("status") != "removed":
        return []
    return [
        {
            "title": "Code or metadata changed while release notes were removed",
            "category": "docs_pr_accuracy",
            "severity": "medium",
            "confidence": "medium",
            "file": "release_notes/unreleased.md",
            "line": None,
            "evidence": "release_notes/unreleased.md is removed in a PR that also changes connector code or metadata.",
            "why_it_matters": "Deleting the release note can leave user-visible behavior, dependency, metadata, or action changes undocumented.",
            "suggested_fix": "Restore or replace the release note entry unless this PR is intentionally release-note-free and that exemption is clear.",
        }
    ]


def check_on_poll_params_and_tags(
    full_files: dict[str, str],
    app_jsons: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    findings = []
    on_poll_params: dict[str, set[str]] = {}

    for json_path, app_json in app_jsons.items():
        for action in app_json.get("actions", []):
            identifier = str(action.get("identifier") or "").lower()
            name = str(action.get("action") or "").lower()
            if identifier != "on_poll" and name != "on poll":
                continue
            param_names = {
                str(param.get("name") or param.get("identifier") or "").strip()
                for param in action_parameters(action)
                if param.get("name") or param.get("identifier")
            }
            on_poll_params[json_path] = {name for name in param_names if name}

        for item in app_configuration(app_json):
            name = str(item.get("name") or item.get("key") or item.get("identifier") or "").lower()
            if "poll_now" in name or (name.startswith("poll_") and any(word in name for word in ("start", "end", "count"))):
                findings.append(
                    {
                        "title": "Asset configuration appears to add workaround Poll Now parameters",
                        "category": "polling_checkpoint",
                        "severity": "medium",
                        "confidence": "medium",
                        "file": json_path,
                        "line": None,
                        "evidence": f"Asset parameter '{name}' looks specific to Poll Now/on_poll behavior.",
                        "why_it_matters": "Connector conventions provide default on_poll parameters; workaround asset params can confuse scheduled polling and Poll Now behavior.",
                        "suggested_fix": "Use the standard on_poll parameters where available and remove redundant asset-level Poll Now controls unless explicitly required.",
                    }
                )

    on_poll_bodies: list[tuple[str, int, str]] = []
    for path, text in full_files.items():
        if not path.endswith(".py"):
            continue
        for function_name, start_line, body in extract_functions_matching(text, r"poll"):
            if "on_poll" in function_name or "poll" in function_name:
                on_poll_bodies.append((path, start_line, body))
                for line_no, line in iter_lines(body):
                    lowered = line.lower()
                    if "\"tags\"" not in lowered and "'tags'" not in lowered and ".tags" not in lowered:
                        continue
                    if "f\"" in line or "f'" in line or ".format(" in line or "append(" in line or "extend(" in line:
                        findings.append(
                            {
                                "title": "on_poll appears to generate tags dynamically",
                                "category": "polling_checkpoint",
                                "severity": "medium",
                                "confidence": "medium",
                                "file": path,
                                "line": start_line + line_no - 1,
                                "evidence": line.strip(),
                                "why_it_matters": "Dynamic tag generation in on_poll can create inconsistent container/artifact metadata and has been called out in prior connector reviews.",
                                "suggested_fix": "Keep on_poll tags deterministic and documented, or remove dynamic tag construction unless it is explicitly required.",
                            }
                        )

    if not on_poll_params:
        return findings

    combined_body = "\n".join(body for _, _, body in on_poll_bodies).lower()
    for json_path, params in on_poll_params.items():
        relevant = sorted(name for name in params if name in POLL_PARAM_NAMES)
        unused = [name for name in relevant if name.lower() not in combined_body]
        if not unused:
            continue
        findings.append(
            {
                "title": "on_poll parameters are declared but not visibly used",
                "category": "polling_checkpoint",
                "severity": "medium",
                "confidence": "medium" if on_poll_bodies else "low",
                "file": json_path,
                "line": None,
                "evidence": f"Declared on_poll parameters not found in polling code: {unused}.",
                "why_it_matters": "Declared on_poll parameters should affect polling behavior; unused params produce misleading UI and review drift.",
                "suggested_fix": "Use the declared parameters in on_poll or remove them from app JSON if the defaults are sufficient.",
            }
        )
    return findings


def check_unused_boilerplate(
    full_files: dict[str, str],
    app_jsons: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    findings = []
    action_identifiers = {
        str(action.get("identifier"))
        for app_json in app_jsons.values()
        for action in app_json.get("actions", [])
        if action.get("identifier")
    }
    for path, text in full_files.items():
        if not path.endswith(".py"):
            continue
        for function_name, start_line, body in extract_handle_functions(text):
            if action_identifiers and function_name not in action_identifiers:
                continue
            body_lower = body.lower()
            stripped_body = "\n".join(
                line.strip()
                for line in body.splitlines()[1:]
                if line.strip() and not line.strip().startswith("#")
            )
            if (
                "notimplementederror" in body_lower
                or "todo: implement" in body_lower
                or stripped_body in {"pass", "return phantom.APP_SUCCESS", "return self.set_status(phantom.APP_SUCCESS)"}
            ):
                findings.append(
                    {
                        "title": "Action handler appears to contain unused boilerplate",
                        "category": "general",
                        "severity": "low",
                        "confidence": "medium",
                        "file": path,
                        "line": start_line,
                        "evidence": f"_handle_{function_name}() looks like placeholder or unused boilerplate.",
                        "why_it_matters": "Leftover boilerplate can expose unsupported actions or make generated connector code look complete when behavior is missing.",
                        "suggested_fix": "Remove unused handlers or implement/test the action behavior before exposing it in app JSON.",
                    }
                )
    return findings


def check_missing_tests(review_input: dict[str, Any]) -> list[dict[str, Any]]:
    changed = changed_paths(review_input)
    if not changed:
        return []
    test_changed = any(
        "/test" in path.lower()
        or path.lower().startswith("test")
        or "/tests/" in path.lower()
        or path.lower().startswith("tests/")
        or "test_params" in path.lower()
        or "app-tests" in path.lower()
        for path in changed
    )
    if test_changed:
        return []

    text = changed_text(review_input)
    risky = any(word in text for word in RISKY_CHANGE_WORDS)
    code_changed = any(path.endswith(".py") or (path.endswith(".json") and "/" not in path) for path in changed)
    if not code_changed or not risky:
        return []

    return [
        {
            "title": "Risky connector behavior changed without visible tests",
            "category": "missing_tests",
            "severity": "medium",
            "confidence": "medium",
            "file": None,
            "line": None,
            "evidence": "The diff changes auth/session/polling/state/validation or mutating behavior but no test files were changed.",
            "why_it_matters": "Historical high-value bugs cluster around auth, mutating actions, API errors, dedup/checkpointing, and validation; those changes need regression coverage.",
            "suggested_fix": "Add or update tests for the changed behavior, especially failure paths, validation failures, checkpoint behavior, and mutating action behavior.",
        }
    ]


def check_sdk_live_only_tests(
    review_input: dict[str, Any],
    full_files: dict[str, str],
) -> list[dict[str, Any]]:
    inventory = build_sdk_review_inventory(review_input, full_files)
    if not inventory["is_sdk_migration"]:
        return []

    test_files = {
        path: text
        for path, text in full_files.items()
        if path.startswith("tests/") and path.endswith(".py") and not path.endswith("__init__.py")
    }
    if not test_files:
        return []

    live_paths = [
        path
        for path, text in test_files.items()
        if path.endswith("_live.py")
        or "_live_" in path
        or "live_asset_config" in text
        or "Missing required live test environment variables" in text
        or ("client_secret" in text and "tenant_id" in text)
    ]
    if not live_paths:
        return []

    offline_markers = (
        "monkeypatch",
        "unittest.mock",
        "from unittest import mock",
        "mocker",
        "MockTransport",
        "pytest_httpx",
        "responses",
        "respx",
        "model_validate",
        "accepts_sparse",
        "BytesIO",
    )
    offline_paths = [
        path
        for path, text in test_files.items()
        if path not in live_paths and any(marker in text for marker in offline_markers)
    ]
    if offline_paths:
        return []

    return [
        {
            "title": "SDK migration tests are live-only and require external credentials",
            "category": "missing_tests",
            "severity": "medium",
            "confidence": "high",
            "file": live_paths[0],
            "line": 1,
            "code_reference": "tests",
            "evidence": f"Detected live/env-bound test files {live_paths[:8]} but no offline mock/model tests.",
            "why_it_matters": "SDK migrations can fail during local model construction, endpoint generation, or manifest serialization before live API calls; live-only tests are difficult to run in CI/review and miss sparse/error payload regressions.",
            "suggested_fix": "Add offline unit tests for SDK output model parsing, endpoint construction, manifest-sensitive metadata, Graph error responses, and upload-session response variants.",
        }
    ]


def check_docs_schema_drift(review_input: dict[str, Any], app_jsons: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    changed_paths = {item.get("filename") for item in review_input.get("changed_files", [])}
    docs_changed = any(path in changed_paths for path in {"README.md", "manual_readme_content.md"})
    app_json_changed = any(path in changed_paths for path in app_jsons)
    pr_text = f"{review_input.get('pr', {}).get('title', '')}\n{review_input.get('pr', {}).get('body', '')}".lower()
    output_claim = any(word in pr_text for word in ("json", "output", "field", "schema"))
    if not docs_changed or app_json_changed or not output_claim:
        return []
    return [
        {
            "title": "PR/docs claim output or JSON changes but app JSON was not changed",
            "category": "docs_pr_accuracy",
            "severity": "medium",
            "confidence": "medium",
            "file": "README.md" if "README.md" in changed_paths else "manual_readme_content.md",
            "line": None,
            "evidence": "README/manual docs changed and the PR text mentions JSON/output/schema, but no top-level app JSON changed.",
            "why_it_matters": "Docs can claim outputs that SOAR does not expose through app JSON, creating playbook and user confusion.",
            "suggested_fix": "Update the app JSON output declarations or adjust the docs/PR body to match the implementation.",
        }
    ]


def check_pr_claim_diff_mismatch(review_input: dict[str, Any]) -> list[dict[str, Any]]:
    changed = changed_paths(review_input)
    pr_text = f"{review_input.get('pr', {}).get('title', '')}\n{review_input.get('pr', {}).get('body', '')}".lower()
    release_text = str((review_input.get("full_files") or {}).get("release_notes/unreleased.md") or "").lower()
    claim_text = f"{pr_text}\n{release_text}"
    findings = []

    dependency_claim = any(word in claim_text for word in ("dependency", "dependencies", "package", "requirements", "version pin"))
    dependency_changed = any(
        path in {"pyproject.toml", "uv.lock", "requirements.txt", "setup.py"}
        or "requirements" in path.lower()
        or path.endswith("Pipfile.lock")
        for path in changed
    )
    if dependency_claim and not dependency_changed:
        findings.append(
            {
                "title": "PR or release notes claim dependency changes but no dependency file changed",
                "category": "docs_pr_accuracy",
                "severity": "medium",
                "confidence": "medium",
                "file": None,
                "line": None,
                "evidence": "PR/release text mentions dependencies/packages/requirements, but no known dependency manifest changed.",
                "why_it_matters": "Review comments and release notes must describe behavior implemented by the diff; dependency claims are especially important for packaging.",
                "suggested_fix": "Update the dependency manifest or revise the PR/release text to match the actual change.",
            }
        )

    default_claim = "default" in claim_text or "defaults" in claim_text
    default_changed = any(path.endswith(".json") or path.endswith(".py") for path in changed) and "default" in changed_text(review_input)
    if default_claim and not default_changed:
        findings.append(
            {
                "title": "PR or release notes claim default behavior without matching diff evidence",
                "category": "docs_pr_accuracy",
                "severity": "medium",
                "confidence": "low",
                "file": None,
                "line": None,
                "evidence": "PR/release text mentions defaults, but the changed code/app JSON patches do not show a default change.",
                "why_it_matters": "Misleading defaults in PR bodies or release notes have caused review escapes in historical connector PRs.",
                "suggested_fix": "Either implement the claimed default change or update the PR body/release note to describe the actual behavior.",
            }
        )

    return findings


def check_app_mapping_hint(review_input: dict[str, Any], app_jsons: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    files = changed_file_map(review_input)
    changed = set(files)
    added_app_jsons = [
        path
        for path in app_jsons
        if path in files and files[path].get("status") in {"added", "renamed"}
    ]
    if not added_app_jsons:
        return []
    mapping_changed = any(
        path.startswith(".github/")
        or "appid_to_name" in path.lower()
        or "appid_to_package" in path.lower()
        or "ci-metadata" in path.lower()
        for path in changed
    )
    if mapping_changed:
        return []
    names = []
    for path in added_app_jsons:
        app_json = app_jsons[path]
        names.append(str(app_json.get("appid") or app_json.get("package_name") or app_json.get("name") or path))
    return [
        {
            "title": "New app may need app-id/app-name mapping updates",
            "category": "precommit",
            "severity": "medium",
            "confidence": "medium",
            "file": added_app_jsons[0],
            "line": None,
            "evidence": f"New app JSON added for {names}, but no .github/ci-metadata mapping file appears in this PR.",
            "why_it_matters": "New connector apps often need appid-to-name/package mappings in support repos; missing mappings show up as connector hook/pre-commit failures.",
            "suggested_fix": "Run the connector pre-commit hooks and update the required .github/ci-metadata app mapping files or create the follow-up PR expected by the team workflow.",
        }
    ]


def check_release_notes_missing(review_input: dict[str, Any]) -> list[dict[str, Any]]:
    changed_paths = {item.get("filename") for item in review_input.get("changed_files", [])}
    has_code_or_json = any(
        path
        and (
            path.endswith(".py")
            or (path.endswith(".json") and "/" not in path)
            or path in {"pyproject.toml", "uv.lock"}
        )
        for path in changed_paths
    )
    if not has_code_or_json:
        return []
    if "release_notes/unreleased.md" in changed_paths:
        return []
    return [
        {
            "title": "Code or metadata changed without release notes",
            "category": "docs_pr_accuracy",
            "severity": "low",
            "confidence": "medium",
            "file": None,
            "line": None,
            "evidence": "The PR changes code/app metadata but does not include release_notes/unreleased.md.",
            "why_it_matters": "Connector PRs usually need release notes for user-visible behavior, dependency, metadata, or action changes.",
            "suggested_fix": "Add a concise release note or explain why the change is intentionally release-note-free.",
        }
    ]


def check_python_syntax_errors(full_files: dict[str, str]) -> list[dict[str, Any]]:
    findings = []
    for path, text in full_files.items():
        if not path.endswith(".py"):
            continue
        try:
            ast.parse(text, filename=path)
        except SyntaxError as exc:
            line = exc.lineno or 1
            evidence = f"{exc.__class__.__name__}: {exc.msg} at {path}:{line}"
            if exc.offset:
                evidence = f"{evidence}:{exc.offset}"
            findings.append(
                {
                    "title": "Changed Python file does not compile",
                    "category": "precommit",
                    "severity": "high",
                    "confidence": "high",
                    "file": path,
                    "line": line,
                    "evidence": evidence,
                    "why_it_matters": "A connector with a Python syntax error cannot pass pre-commit, compile, or runtime loading.",
                    "suggested_fix": "Fix the Python syntax at the reported line, then run the compile/pre-commit hook locally before updating the PR.",
                }
            )
    return findings


def check_conflict_markers(full_files: dict[str, str]) -> list[dict[str, Any]]:
    findings = []
    marker_pattern = re.compile(r"^(<<<<<<<|=======|>>>>>>>)")
    for path, text in full_files.items():
        for line_no, line in iter_lines(text):
            if not marker_pattern.match(line):
                continue
            findings.append(
                {
                    "title": "Unresolved merge conflict marker is present",
                    "category": "merge_conflict",
                    "severity": "high",
                    "confidence": "high",
                    "file": path,
                    "line": line_no,
                    "evidence": line.strip(),
                    "why_it_matters": "Conflict markers make the connector invalid and indicate the branch was not cleanly resolved against the base branch.",
                    "suggested_fix": "Resolve the conflicting block against the base branch, remove all conflict markers, and rerun CI on the resolved commit.",
                }
            )
            break
    return findings


def check_known_static_name_errors(full_files: dict[str, str]) -> list[dict[str, Any]]:
    findings = []
    for path, text in full_files.items():
        if not path.endswith(".py"):
            continue
        for line_no, line in iter_lines(text):
            if "degub_print" not in line:
                continue
            findings.append(
                {
                    "title": "Connector logging method appears misspelled",
                    "category": "precommit",
                    "severity": "high",
                    "confidence": "high",
                    "file": path,
                    "line": line_no,
                    "evidence": line.strip(),
                    "why_it_matters": "Historical static reviews caught misspelled connector logging calls; this will fail at runtime because degub_print is not a connector method.",
                    "suggested_fix": "Rename degub_print to debug_print and rerun the static/pre-commit checks.",
                    "suggested_code": line.replace("degub_print", "debug_print").strip(),
                }
            )
    return findings


def check_target_pipeline_failures(review_input: dict[str, Any]) -> list[dict[str, Any]]:
    ci = review_input.get("ci") or {}
    if not isinstance(ci, dict):
        return []

    failures = ci.get("target_job_failures") or []
    if not isinstance(failures, list):
        failures = []
    if not failures:
        failures = target_pipeline_failures_from_check_runs(ci)

    findings: list[dict[str, Any]] = []
    seen: set[str] = set()
    for failure in failures:
        if not isinstance(failure, dict):
            continue
        target_name = normalize_target_pipeline_job(str(failure.get("target_name") or failure.get("name") or ""))
        if target_name not in TARGET_PIPELINE_JOB_NAMES:
            continue
        conclusion = str(failure.get("conclusion") or "").lower()
        if conclusion not in {"failure", "timed_out", "cancelled", "action_required"}:
            continue
        if target_name in seen:
            continue
        seen.add(target_name)

        reason = target_pipeline_failure_reason(failure)
        fix = target_pipeline_suggested_fix(target_name, reason, conclusion)
        url = str(failure.get("html_url") or failure.get("details_url") or failure.get("url") or "").strip() or None
        evidence_parts = [
            f"`{failure.get('name') or target_name}` concluded `{conclusion}`.",
        ]
        if reason:
            evidence_parts.append(reason)
        if url:
            evidence_parts.append(f"Failed job: {url}")
        findings.append(
            {
                "title": f"{target_name} pipeline job failed",
                "category": "ci_pipeline_failure",
                "finding_category": "introduced_bug",
                "causality": "exposed_by_pr",
                "severity": "high",
                "confidence": "high",
                "merge_blocking": True,
                "publication_destination": "inline_blocking",
                "file": None,
                "line": None,
                "code_reference": f"GitHub Actions job `{target_name}`",
                "evidence": " ".join(part for part in evidence_parts if part),
                "why_it_matters": (
                    "This review job runs after the connector pipeline, so a failed required pipeline job blocks "
                    "a clean merge signal even if the code review also finds issues."
                ),
                "suggested_fix": fix,
                "url": url,
            }
        )
    return findings


def target_pipeline_failures_from_check_runs(ci: dict[str, Any]) -> list[dict[str, Any]]:
    log_by_name: dict[str, dict[str, Any]] = {}
    for log in ci.get("failed_check_logs", []):
        if not isinstance(log, dict):
            continue
        target_name = normalize_target_pipeline_job(str(log.get("name") or ""))
        if target_name in TARGET_PIPELINE_JOB_NAMES:
            log_by_name[target_name] = log

    failures: list[dict[str, Any]] = []
    for run in ci.get("check_runs", []):
        if not isinstance(run, dict):
            continue
        target_name = normalize_target_pipeline_job(str(run.get("name") or ""))
        if target_name not in TARGET_PIPELINE_JOB_NAMES:
            continue
        conclusion = str(run.get("conclusion") or "").lower()
        if conclusion not in {"failure", "timed_out", "cancelled", "action_required"}:
            continue
        failure = dict(run)
        failure["target_name"] = target_name
        if target_name in log_by_name:
            failure["log_excerpt"] = log_by_name[target_name].get("log_excerpt")
            failure["html_url"] = failure.get("html_url") or log_by_name[target_name].get("details_url")
        failures.append(failure)
    return failures


def normalize_target_pipeline_job(job_name: str) -> str:
    lowered = re.sub(r"\s+", " ", job_name.strip().lower())
    lowered = lowered.split(" (", 1)[0].strip()
    if lowered in TARGET_PIPELINE_JOB_NAMES:
        return lowered
    for target in TARGET_PIPELINE_JOB_NAMES:
        if lowered.startswith(f"{target} ") or lowered.startswith(f"{target} /"):
            return target
    return lowered


def target_pipeline_failure_reason(job: dict[str, Any]) -> str:
    failed_steps = [
        str(step.get("name") or "").strip()
        for step in job.get("steps") or []
        if isinstance(step, dict) and str(step.get("conclusion") or "").lower() in {"failure", "timed_out", "cancelled"}
    ]
    excerpt = str(job.get("log_excerpt") or "")
    snippet = summarize_precommit_body(excerpt) or summarize_ci_failure_excerpt(excerpt)
    parts: list[str] = []
    if failed_steps:
        parts.append(f"Failed step(s): {', '.join(step for step in failed_steps[:3] if step)}.")
    if snippet:
        parts.append(f"Log excerpt: {snippet}")
    return " ".join(parts).strip()


def target_pipeline_suggested_fix(job_name: str, reason: str, conclusion: str) -> str:
    lowered = reason.lower()
    if conclusion == "timed_out":
        return (
            f"Open the `{job_name}` job log linked above, find the last command that was still running, "
            "and either fix the hanging operation or add a bounded timeout/retry around that command."
        )
    if conclusion == "cancelled":
        return (
            f"Rerun `{job_name}` if it was cancelled by a newer push. If it cancels repeatedly, inspect "
            "the linked job for the cancellation source before merging."
        )
    if job_name == "pre-commit":
        return precommit_suggested_fix(reason)
    if job_name == "compile":
        if any(term in lowered for term in ("syntaxerror", "py_compile", "compileerror", "could not compile")):
            return "Fix the Python syntax at the reported file/line and rerun the compile job."
        if any(term in lowered for term in ("importerror", "modulenotfounderror", "no module named")):
            return "Fix the missing import or packaging dependency named in the compile log, then rerun the compile job."
        if "soarapps" in lowered or "manifest" in lowered:
            return "Fix the SDK app metadata or import-time error reported by the compile/manifest step, then rerun compile."
        return "Open the compile job log, fix the first concrete Python/package/metadata error shown there, and rerun compile."
    if job_name == "build":
        if any(term in lowered for term in ("soarapps package build", "package build", ".tgz", "tar")):
            return "Fix the packaging error reported by the build command, regenerate the app package, and rerun build."
        if any(term in lowered for term in ("dependency", "dependencies", "wheel", "uv.lock", "requirements")):
            return "Fix the dependency or lockfile issue named by the build log, then rerun the build job."
        return "Open the build job log, fix the first concrete packaging/build error shown there, and rerun build."
    if job_name == "semantic-release-preview":
        if any(term in lowered for term in ("release note", "release_notes", "unreleased.md")):
            return "Fix the release_notes/unreleased.md entry or generated release-note content, then rerun semantic-release-preview."
        if any(term in lowered for term in ("conventional commit", "semantic-release", "tag format", "branch")):
            return "Fix the release metadata, branch/tag configuration, or commit message issue named in the semantic-release log."
        if any(term in lowered for term in ("npm", "node", "module not found", "enoent")):
            return "Fix the Node/npm setup or semantic-release dependency error shown in the job log, then rerun the preview."
        return "Open the semantic-release-preview log, fix the first release metadata or semantic-release configuration error, and rerun the job."
    return "Open the failed job log linked above, fix the first concrete error shown there, and rerun the workflow."


def check_precommit_failures(review_input: dict[str, Any]) -> list[dict[str, Any]]:
    evidence: list[str] = []
    urls: list[str] = []
    ci = review_input.get("ci") or {}

    for run in ci.get("check_runs", []) if isinstance(ci, dict) else []:
        name = str(run.get("name") or "")
        conclusion = str(run.get("conclusion") or "")
        summary = str((run.get("output") or {}).get("summary") or "")
        combined = f"{name}\n{summary}"
        if not is_precommit_signal(combined):
            continue
        if conclusion and conclusion not in {"success", "skipped", "neutral"}:
            evidence.append(f"{name}: {conclusion}")
            if run.get("details_url"):
                urls.append(str(run["details_url"]))

    for status in ci.get("statuses", []) if isinstance(ci, dict) else []:
        context = str(status.get("context") or "")
        state = str(status.get("state") or "")
        description = str(status.get("description") or "")
        combined = f"{context}\n{description}"
        if is_precommit_signal(combined) and state in {"failure", "error"}:
            evidence.append(f"{context}: {state} - {description}".strip())
            if status.get("target_url"):
                urls.append(str(status["target_url"]))

    for log in ci.get("failed_check_logs", []) if isinstance(ci, dict) else []:
        name = str(log.get("name") or "")
        excerpt = str(log.get("log_excerpt") or "")
        combined = f"{name}\n{excerpt}"
        if not is_precommit_signal(combined):
            continue
        snippet = summarize_precommit_body(excerpt) or excerpt.strip()[:500]
        if snippet:
            evidence.append(f"{name}: {snippet}".strip())
            if log.get("details_url"):
                urls.append(str(log["details_url"]))

    for comment in all_comments(review_input):
        body = str(comment.get("body") or "")
        if not is_precommit_signal(body):
            continue
        snippet = summarize_precommit_body(body)
        if snippet:
            evidence.append(snippet)
            if comment.get("url"):
                urls.append(str(comment["url"]))

    if not evidence:
        return []

    severity = "high" if any("FAILED -" in item or "failure" in item.lower() for item in evidence) else "medium"
    evidence_text = "; ".join(evidence[:6])
    return [
        {
            "title": "Pre-commit or connector hook failures need to be resolved",
            "category": "precommit",
            "severity": severity,
            "confidence": "high",
            "file": None,
            "line": None,
            "evidence": evidence_text,
            "why_it_matters": "Connector PRs are expected to pass pre-commit and local hooks before merge; these hooks catch app mappings, JSON metadata, docs generation, secrets, formatting, and release-note issues.",
            "suggested_fix": precommit_suggested_fix(evidence_text),
            "url": urls[0] if urls else None,
        }
    ]


def precommit_suggested_fix(evidence: str) -> str:
    lowered = evidence.lower()
    if "syntaxerror" in lowered or "py_compile" in lowered or "could not compile" in lowered:
        return "Fix the Python syntax at the reported file/line, then rerun the compile or pre-commit hook that reported it."
    if "ruff" in lowered or re.search(r"\b[FEW]\d{3}\b", evidence):
        return "Apply the reported ruff lint or formatting fix at the named file/line, then rerun pre-commit."
    if "detect-secrets" in lowered:
        return "Remove the secret-like value from the diff, or update the secrets baseline only after confirming the hit is a false positive."
    if "semgrep" in lowered:
        return "Apply the semgrep-reported code change, or suppress it only with a clear false-positive justification in the PR."
    if "build-docs" in lowered or "mdformat" in lowered:
        return "Run the named docs hook locally, keep the generated documentation changes, and commit those generated files with the source change."
    if "release-notes" in lowered or "release_notes/unreleased" in lowered:
        return "Add or fix the release_notes/unreleased.md entry so the hook sees a valid release note for this behavior change."
    if "check-json" in lowered or "check-yaml" in lowered:
        return "Fix the JSON/YAML syntax or formatting at the reported file/line, then rerun the named hook."
    if any(word in lowered for word in ("app_package_name", "valid_app_name_and_guid", "appid_to_name", "appid_to_package_name")):
        return "Update the app JSON metadata and the required .github/ci-metadata app-id/name/package mapping so the GUID, app name, and package name agree."
    if "min platform version" in lowered or "min phantom version" in lowered:
        return "Update the app JSON minimum platform/min_phantom_version metadata and regenerate generated docs so the static test sees the expected value."
    if "action coverage" in lowered or "test coverage" in lowered or "coverage" in lowered:
        return "Add tests for the actions or changed connector path named by the coverage output, then rerun the coverage/static test."
    if "action name" in lowered or "action names" in lowered:
        return "Update the action name, identifier, or display text to match connector naming conventions, then regenerate docs if action docs changed."
    if "additional logging" in lowered or "min number log statements" in lowered:
        return "Add useful debug_print, error_print, save_progress, or send_progress statements to each named action outside tight loops, then rerun the static test."
    if "verbosity" in lowered:
        return "Remove unnecessary verbose output or convert it into useful connector progress/error logging, then rerun the static test."
    if "license" in lowered or "notice" in lowered:
        return "Update LICENSE/NOTICE or dependency license metadata to match the packaged dependency files, then rerun the static test."
    if "product name on files" in lowered:
        return "Align product/app naming in app JSON, docs, and generated files so the static product-name check passes."
    if "playbook missing" in lowered or "app-tests" in lowered or "apps-test-playbooks" in lowered:
        return "Add or link the required app test playbook so the integration/static test can find the expected test artifact."
    if "integration test results are missing" in lowered:
        return "Attach or link the integration test results, or rerun the integration test once the external blocker is resolved."
    return "Run the named pre-commit hook locally, apply the reported file-level failure, and commit the resulting fix."


def check_merge_conflicts(review_input: dict[str, Any]) -> list[dict[str, Any]]:
    pr = review_input.get("pr") or {}
    mergeable = pr.get("mergeable")
    mergeable_state = str(pr.get("mergeable_state") or "").lower()
    evidence: list[str] = []
    blocker_details = merge_blocker_details(review_input)

    if pr.get("state") == "open" and mergeable is False:
        evidence.append(f"GitHub reports mergeable=false, mergeable_state={mergeable_state or 'unknown'}.")
    if mergeable_state == "blocked" and blocker_details:
        evidence.append(f"GitHub mergeable_state is blocked by: {blocker_details[0]}")
    elif mergeable_state == "dirty":
        evidence.append(f"GitHub mergeable_state is {mergeable_state}.")

    for comment in all_comments(review_input):
        body = str(comment.get("body") or "")
        lowered = body.lower()
        if "merge conflict" in lowered or "cannot be automatically merged" in lowered or "resolve conflicts" in lowered:
            evidence.append(summarize_line_matches(body, ("merge conflict", "cannot be automatically merged", "resolve conflicts")))

    if not evidence:
        return []

    return [
        {
            "title": "PR appears to have merge conflicts or mergeability blockers",
            "category": "merge_conflict",
            "severity": "high",
            "confidence": "high" if mergeable is False or mergeable_state == "dirty" or blocker_details else "medium",
            "file": None,
            "line": None,
            "evidence": "; ".join(filter(None, evidence[:4])),
            "why_it_matters": "The CI-end reviewer cannot give a clean merge recommendation if GitHub cannot merge the branch into the target base.",
            "suggested_fix": merge_blocker_fix(blocker_details),
        }
    ]


def merge_blocker_details(review_input: dict[str, Any]) -> list[str]:
    ci = review_input.get("ci") or {}
    details: list[str] = []
    if not isinstance(ci, dict):
        return details

    for log in ci.get("failed_check_logs", []):
        if not isinstance(log, dict):
            continue
        name = str(log.get("name") or "failed check")
        excerpt = str(log.get("log_excerpt") or "")
        snippet = summarize_precommit_body(excerpt) or summarize_ci_failure_excerpt(excerpt)
        if snippet and has_specific_ci_detail(snippet):
            details.append(f"{name}: {snippet}")

    for run in ci.get("check_runs", []):
        if not isinstance(run, dict):
            continue
        conclusion = str(run.get("conclusion") or "").lower()
        if conclusion not in {"failure", "timed_out", "cancelled", "action_required"}:
            continue
        name = str(run.get("name") or "failed check")
        summary = str((run.get("output") or {}).get("summary") or "")
        snippet = summarize_precommit_body(summary) or summarize_ci_failure_excerpt(summary)
        if snippet and has_specific_ci_detail(snippet):
            details.append(f"{name}: {snippet}")

    for status in ci.get("statuses", []):
        if not isinstance(status, dict):
            continue
        state = str(status.get("state") or "").lower()
        if state not in {"failure", "error"}:
            continue
        context = str(status.get("context") or "failed status")
        description = str(status.get("description") or "")
        if description and has_specific_ci_detail(description):
            details.append(f"{context}: {state} - {description}")

    output: list[str] = []
    seen: set[str] = set()
    for detail in details:
        if detail in seen:
            continue
        seen.add(detail)
        output.append(detail)
    return output[:3]


def has_specific_ci_detail(text: str) -> bool:
    lowered = text.lower()
    if is_precommit_signal(text) or is_precommit_failure_detail(text):
        return True
    if re.search(r"\b[FEW]\d{3}\b", text):
        return True
    if re.search(r"\b[\w./-]+\.(?:py|json|ya?ml|toml|md):\d+", text):
        return True
    return any(word in lowered for word in ("assertionerror", "traceback", "failed -", "hook id:", "coverage", "semgrep", "detect-secrets"))


def merge_blocker_fix(blocker_details: list[str]) -> str:
    if blocker_details:
        return "Fix the named blocker above, using the reported hook/test/file detail as the source of truth, then rerun the required check."
    return "Update the PR branch from the base branch and resolve the concrete conflict or required-check blocker before relying on CI or review output."


def all_comments(review_input: dict[str, Any]) -> list[dict[str, Any]]:
    comments = review_input.get("comments") or {}
    output: list[dict[str, Any]] = []
    for key in ("issue_comments", "review_comments", "reviews"):
        items = comments.get(key) or []
        if isinstance(items, list):
            output.extend(item for item in items if isinstance(item, dict))
    return output


def is_precommit_signal(text: str) -> bool:
    lowered = text.lower()
    needles = (
        "pre-commit",
        "precommit",
        "failed -",
        "valid_app_name_and_guid",
        "app_package_name",
        "appid_to_name",
        "appid_to_package_name",
        "semgrep",
        "detect-secrets",
        "ruff",
        "release-notes",
        "build-docs",
        "check-json",
        "check-yaml",
        "mdformat",
        "local_hooks",
        "static test",
        "static tests",
        "sanity test",
        "sanity tests",
        "integration test",
        "integration tests",
        "jenkins",
        "syntaxerror",
        "importerror",
        "py_compile",
        "could not compile",
    )
    return any(needle in lowered for needle in needles)


def summarize_precommit_body(body: str) -> str:
    matches = []
    for line in body.splitlines():
        if is_ci_boilerplate_line(line):
            continue
        if is_precommit_signal(line) or is_precommit_failure_detail(line):
            stripped = line.strip()
            if stripped:
                matches.append(stripped)
    if matches:
        return " / ".join(matches[:4])
    return body.strip()[:500]


def is_precommit_failure_detail(line: str) -> bool:
    if is_ci_boilerplate_line(line):
        return False
    lowered = line.lower()
    if re.search(r"\b[FEW]\d{3}\b", line):
        return True
    if re.search(r"\.py:\d+(?::\d+)?:", line):
        return True
    static_failure_terms = (
        "min platform version",
        "min phantom version",
        "action name",
        "action names",
        "additional logging",
        "verbosity",
        "license",
        "product name on files",
        "playbook missing",
        "integration test results are missing",
        "app-tests",
        "apps-test-playbooks",
    )
    return any(word in lowered for word in ("hook id:", "exit code:", "assertionerror", "syntaxerror", "importerror")) or any(
        term in lowered for term in static_failure_terms
    )


def is_ci_boilerplate_line(line: str) -> bool:
    lowered = line.lower()
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
    if any(phrase in lowered for phrase in boilerplate):
        return True
    if re.search(r"\bpytest\s+suite/apps/", lowered):
        return True
    return False


def summarize_line_matches(body: str, needles: tuple[str, ...]) -> str:
    for line in body.splitlines():
        lowered = line.lower()
        if any(needle in lowered for needle in needles):
            return line.strip()[:500]
    return body.strip()[:500]


def check_ci_failures(review_input: dict[str, Any]) -> list[dict[str, Any]]:
    ci = review_input.get("ci") or {}
    failures = []
    for run in ci.get("check_runs", []) if isinstance(ci, dict) else []:
        conclusion = run.get("conclusion")
        if conclusion in {"failure", "timed_out", "cancelled", "action_required"}:
            failures.append(f"{run.get('name')}: {conclusion}")
    for status in ci.get("statuses", []) if isinstance(ci, dict) else []:
        state = status.get("state")
        if state in {"failure", "error"}:
            failures.append(f"{status.get('context')}: {state}")
    for log in ci.get("failed_check_logs", []) if isinstance(ci, dict) else []:
        name = str(log.get("name") or "failed check")
        excerpt = summarize_ci_failure_excerpt(str(log.get("log_excerpt") or ""))
        if excerpt:
            failures.append(f"{name}: {excerpt}")
    if not failures:
        return []
    return [
        {
            "title": "CI checks are failing or incomplete",
            "category": "ci_synthesis",
            "severity": "medium",
            "confidence": "high",
            "file": None,
            "line": None,
            "evidence": "; ".join(failures[:10]),
            "why_it_matters": "The final review should synthesize prior pipeline failures instead of reviewing code in isolation.",
            "suggested_fix": "Use the named failing hook or test output above to fix the reported file, assertion, metadata, or generated-doc issue before merge.",
        }
    ]


def summarize_ci_failure_excerpt(excerpt: str) -> str:
    for line in excerpt.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        lowered = stripped.lower()
        if is_precommit_signal(stripped) or any(word in lowered for word in ("assertionerror", "traceback", "error:", "failed")):
            return stripped[:500]
    return excerpt.strip()[:500]
