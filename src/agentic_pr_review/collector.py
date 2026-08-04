"""Collect and compact PR context from GitHub."""

from __future__ import annotations

import io
import json
import os
from pathlib import PurePosixPath
import re
from typing import Any
import zipfile

from .config import RuntimeConfig
from .github_client import GitHubClient, GitHubError
from .json_utils import truncate_text


BINARY_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".svg",
    ".ico",
    ".whl",
    ".zip",
    ".gz",
    ".pdf",
}

ALWAYS_FETCH_FULL_FILES = {
    "README.md",
    "manual_readme_content.md",
    "pyproject.toml",
    "uv.lock",
    "release_notes/unreleased.md",
}

FAILED_CHECK_CONCLUSIONS = {"failure", "timed_out", "cancelled", "action_required"}
TARGET_PIPELINE_JOB_NAMES = {
    "pre-commit",
    "compile",
    "build",
    "semantic-release-preview",
}
CI_LOG_KEYWORDS = (
    "failed -",
    "failed:",
    "failures",
    "hook id:",
    "exit code:",
    "error:",
    "traceback",
    "assertionerror",
    "syntaxerror",
    "importerror",
    "modulenotfounderror",
    "ruff",
    "semgrep",
    "detect-secrets",
    "build-docs",
    "build failed",
    "release-notes",
    "semantic-release",
    "check-json",
    "check-yaml",
    "mdformat",
    "compile failed",
    "pytest",
    "coverage",
    "npm err!",
    "package build",
    "soarapps",
    "app_package_name",
    "valid_app_name_and_guid",
    "appid_to_name",
    "appid_to_package_name",
)


def is_probably_text(path: str) -> bool:
    suffix = PurePosixPath(path).suffix.lower()
    return suffix not in BINARY_SUFFIXES


def is_relevant_full_file(path: str) -> bool:
    name = PurePosixPath(path).name.lower()
    suffix = PurePosixPath(path).suffix.lower()
    if name in {"readme.md", "manual_readme_content.md", "pyproject.toml", "uv.lock"}:
        return True
    if path.startswith("release_notes/"):
        return True
    return suffix in {
        ".py",
        ".json",
        ".md",
        ".yml",
        ".yaml",
        ".toml",
        ".html",
        ".xml",
        ".jinja",
        ".j2",
        ".css",
    }


def compact_user(user: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(user, dict):
        return None
    return {"login": user.get("login"), "id": user.get("id")}


def compact_comment(comment: dict[str, Any], *, max_body: int = 3000) -> dict[str, Any]:
    return {
        "id": comment.get("id"),
        "url": comment.get("html_url") or comment.get("url"),
        "path": comment.get("path"),
        "line": comment.get("line") or comment.get("original_line"),
        "in_reply_to_id": comment.get("in_reply_to_id"),
        "created_at": comment.get("created_at"),
        "user": compact_user(comment.get("user")),
        "body": truncate_text(comment.get("body") or "", max_body),
    }


class PRCollector:
    def __init__(self, client: GitHubClient, config: RuntimeConfig) -> None:
        self.client = client
        self.config = config

    def collect(
        self,
        repo: str,
        pr_number: int,
    ) -> dict[str, Any]:
        pr = self.client.get_pr(repo, pr_number)
        head_sha = pr["head"]["sha"]
        base_sha = pr["base"]["sha"]

        files = self.client.list_pr_files(repo, pr_number)
        issue_comments = self.client.list_issue_comments(repo, pr_number)
        review_comments = self.client.list_review_comments(repo, pr_number)
        reviews = self.client.list_reviews(repo, pr_number)
        checks = self._safe_checks(repo, head_sha)
        workflow_jobs = self._safe_workflow_run_jobs(repo)
        if workflow_jobs.get("workflow_job_errors"):
            checks["errors"].extend(str(error) for error in workflow_jobs["workflow_job_errors"])
        checks.update(workflow_jobs)

        root_json_files = self._root_json_files(repo, head_sha)
        full_file_paths = self._select_full_files(files, root_json_files)
        raw_urls = {file.get("filename"): file.get("raw_url") for file in files if file.get("filename") and file.get("raw_url")}
        full_files, full_file_errors = self._fetch_full_files(repo, head_sha, full_file_paths, raw_urls)
        doc_context = self._fetch_doc_context(repo, head_sha, files)

        base_root_json_files = self._root_json_files(repo, base_sha)
        base_full_file_paths = set(base_root_json_files)
        base_full_file_paths.update(ALWAYS_FETCH_FULL_FILES)
        base_files, base_full_file_errors = self._fetch_full_files(repo, base_sha, base_full_file_paths, {})
        compact_files = [self._compact_file(file) for file in files]

        return {
            "schema_version": "0.1",
            "repo": repo,
            "pr": {
                "number": pr_number,
                "title": pr.get("title"),
                "body": truncate_text(pr.get("body") or "", 12_000),
                "state": pr.get("state"),
                "merged": pr.get("merged"),
                "mergeable": pr.get("mergeable"),
                "mergeable_state": pr.get("mergeable_state"),
                "rebaseable": pr.get("rebaseable"),
                "draft": pr.get("draft"),
                "html_url": pr.get("html_url"),
                "created_at": pr.get("created_at"),
                "updated_at": pr.get("updated_at"),
                "user": compact_user(pr.get("user")),
                "base": {"ref": pr["base"]["ref"], "sha": base_sha},
                "head": {"ref": pr["head"]["ref"], "sha": head_sha},
            },
            "changed_files": compact_files,
            "full_files": full_files,
            "doc_context": doc_context,
            "base_files": base_files,
            "comments": {
                "issue_comments": [compact_comment(comment) for comment in issue_comments],
                "review_comments": [compact_comment(comment) for comment in review_comments],
                "reviews": [self._compact_review(review) for review in reviews],
            },
            "ci": checks,
            "collector_notes": {
                "github_auth_mode": getattr(self.client, "auth_mode", "unknown"),
                "changed_file_count": len(files),
                "changed_file_patch_count": sum(1 for file in compact_files if file.get("patch")),
                "changed_file_missing_patch_count": sum(1 for file in compact_files if not file.get("patch")),
                "changed_file_truncated_patch_count": sum(1 for file in compact_files if file.get("patch_truncated")),
                "changed_file_total_additions": sum(int(file.get("additions") or 0) for file in compact_files),
                "changed_file_total_deletions": sum(int(file.get("deletions") or 0) for file in compact_files),
                "full_file_count": len(full_files),
                "full_file_missing_count": len(full_file_paths - set(full_files)),
                "base_full_file_count": len(base_files),
                "base_full_file_missing_count": len(base_full_file_paths - set(base_files)),
                "root_json_files": root_json_files,
                "base_root_json_files": base_root_json_files,
                "full_file_paths": sorted(full_file_paths),
                "full_file_missing_paths": sorted(full_file_paths - set(full_files))[:50],
                "full_file_fetch_errors": full_file_errors[:50],
                "base_full_file_paths": sorted(base_full_file_paths),
                "base_full_file_missing_paths": sorted(base_full_file_paths - set(base_files))[:50],
                "base_full_file_fetch_errors": base_full_file_errors[:50],
                "ci_error_count": len(checks.get("errors", [])) if isinstance(checks, dict) else 0,
                "target_pipeline_failure_count": len(checks.get("target_job_failures", [])) if isinstance(checks, dict) else 0,
            },
        }

    def _safe_checks(self, repo: str, head_sha: str) -> dict[str, Any]:
        output: dict[str, Any] = {"check_runs": [], "statuses": [], "errors": []}
        try:
            data = self.client.get_check_runs(repo, head_sha)
            runs = data.get("check_runs", []) if isinstance(data, dict) else []
            output["check_runs"] = [
                {
                    "name": run.get("name"),
                    "status": run.get("status"),
                    "conclusion": run.get("conclusion"),
                    "started_at": run.get("started_at"),
                    "completed_at": run.get("completed_at"),
                    "details_url": run.get("details_url"),
                    "html_url": run.get("html_url"),
                    "output": {
                        "title": (run.get("output") or {}).get("title"),
                        "summary": truncate_text((run.get("output") or {}).get("summary") or "", 2500),
                    },
                }
                for run in runs
            ]
            failed_logs, log_errors = self._failed_check_logs(repo, output["check_runs"])
            output["failed_check_logs"] = failed_logs
            output["errors"].extend(log_errors)
        except GitHubError as exc:
            output["errors"].append(str(exc))
        try:
            status = self.client.get_combined_status(repo, head_sha)
            output["combined_status"] = {
                "state": status.get("state"),
                "total_count": status.get("total_count"),
            }
            output["statuses"] = [
                {
                    "context": item.get("context"),
                    "state": item.get("state"),
                    "description": item.get("description"),
                    "target_url": item.get("target_url"),
                }
                for item in status.get("statuses", [])
            ]
        except GitHubError as exc:
            output["errors"].append(str(exc))
        return output

    def _failed_check_logs(
        self,
        repo: str,
        check_runs: list[dict[str, Any]],
        *,
        max_logs: int = 5,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        logs: list[dict[str, Any]] = []
        errors: list[str] = []
        for run in check_runs:
            conclusion = str(run.get("conclusion") or "").lower()
            if conclusion not in FAILED_CHECK_CONCLUSIONS:
                continue
            job_id = extract_actions_job_id(str(run.get("details_url") or ""))
            if not job_id:
                continue
            try:
                raw_log = self.client.get_actions_job_logs(repo, job_id)
            except GitHubError as exc:
                errors.append(f"Could not fetch GitHub Actions log for {run.get('name') or job_id}: {exc}")
                continue
            log_text = decode_actions_job_log(raw_log)
            excerpt = summarize_ci_log(log_text)
            if not excerpt:
                continue
            logs.append(
                {
                    "name": run.get("name"),
                    "conclusion": run.get("conclusion"),
                    "details_url": run.get("details_url"),
                    "job_id": job_id,
                    "log_excerpt": excerpt,
                }
            )
            if len(logs) >= max_logs:
                break
        return logs, errors

    def _safe_workflow_run_jobs(self, repo: str) -> dict[str, Any]:
        workflow_needs_results = previous_pipeline_results_from_env()
        output: dict[str, Any] = {
            "workflow_run_id": os.getenv("GITHUB_RUN_ID"),
            "workflow_run_attempt": os.getenv("GITHUB_RUN_ATTEMPT"),
            "workflow_needs_results": workflow_needs_results,
            "workflow_jobs": [],
            "target_job_failures": previous_pipeline_failures_from_env(),
            "workflow_job_errors": [],
        }
        run_id = str(output.get("workflow_run_id") or "").strip()
        if not run_id:
            return output
        attempt = str(output.get("workflow_run_attempt") or "").strip() or None
        try:
            jobs = self.client.list_workflow_run_jobs(repo, run_id, attempt=attempt)
        except GitHubError as exc:
            output["workflow_job_errors"].append(f"Could not list workflow jobs for run {run_id}: {exc}")
            return output

        compact_jobs = [compact_workflow_job(job) for job in jobs]
        output["workflow_jobs"] = compact_jobs
        for job in compact_jobs:
            target_name = target_pipeline_job_name(str(job.get("name") or ""))
            if not target_name:
                continue
            conclusion = str(job.get("conclusion") or "").lower()
            if conclusion not in FAILED_CHECK_CONCLUSIONS:
                continue
            failed = dict(job)
            failed["target_name"] = target_name
            output["target_job_failures"] = [
                existing
                for existing in output["target_job_failures"]
                if not isinstance(existing, dict) or existing.get("target_name") != target_name
            ]
            job_id = failed.get("id")
            if job_id:
                try:
                    raw_log = self.client.get_actions_job_logs(repo, job_id)
                except GitHubError as exc:
                    output["workflow_job_errors"].append(
                        f"Could not fetch GitHub Actions log for {failed.get('name') or job_id}: {exc}"
                    )
                else:
                    failed["log_excerpt"] = summarize_ci_log(decode_actions_job_log(raw_log))
            output["target_job_failures"].append(failed)
        return output

    def _root_json_files(self, repo: str, head_sha: str) -> list[str]:
        try:
            contents = self.client.list_root_contents(repo, head_sha)
        except GitHubError:
            return []
        output = []
        for item in contents:
            path = item.get("path")
            if not path or item.get("type") != "file":
                continue
            name = PurePosixPath(path).name.lower()
            if name.endswith(".json") and "postman" not in name and name not in {"package-lock.json"}:
                output.append(path)
        return output

    def _select_full_files(self, files: list[dict[str, Any]], root_json_files: list[str]) -> set[str]:
        selected = set(root_json_files)
        selected.update(ALWAYS_FETCH_FULL_FILES)
        for file in files:
            path = file.get("filename")
            if not path or not is_probably_text(path):
                continue
            if is_relevant_full_file(path):
                selected.add(path)
        return selected

    def _fetch_full_files(
        self,
        repo: str,
        head_sha: str,
        paths: set[str],
        raw_urls: dict[str, str],
    ) -> tuple[dict[str, str], list[str]]:
        output: dict[str, str] = {}
        errors: list[str] = []
        for path in sorted(paths):
            text = None
            try:
                text = self.client.fetch_text_file(repo, path, head_sha)
            except GitHubError as exc:
                errors.append(f"{path}: contents API failed: {exc}")
                raw_url = raw_urls.get(path)
                if raw_url:
                    try:
                        text = self.client.get_text_url(raw_url)
                    except GitHubError as raw_exc:
                        errors.append(f"{path}: raw_url fallback failed: {raw_exc}")
                        text = None
            if text is None:
                if path not in raw_urls:
                    errors.append(f"{path}: file was unavailable from contents API and no PR raw_url fallback was present")
                continue
            output[path] = truncate_text(text, self.config.max_file_chars)
        return output, errors

    def _fetch_doc_context(self, repo: str, head_sha: str, files: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        terms = self._doc_terms(files)
        if not terms:
            return {}
        output: dict[str, list[dict[str, Any]]] = {}
        for path in ("README.md", "manual_readme_content.md"):
            try:
                text = self.client.fetch_text_file(repo, path, head_sha, max_bytes=1_200_000)
            except GitHubError:
                continue
            if not text:
                continue
            snippets = snippets_for_terms(text, terms)
            if snippets:
                output[path] = snippets
        return output

    def _doc_terms(self, files: list[dict[str, Any]]) -> list[str]:
        terms: list[str] = []
        for file in files:
            path = str(file.get("filename") or "")
            patch = str(file.get("patch") or "")
            if not patch:
                continue
            if path.endswith((".json", ".py", ".md", ".toml")):
                terms.extend(re.findall(r"\b[a-z][a-z0-9]+(?:_[a-z0-9]+)+\b", patch))
                terms.extend(re.findall(r"`([^`]{4,80})`", patch))
        ignored = {
            "action_result",
            "data_path",
            "data_type",
            "input_file",
            "app_version",
            "pyproject",
        }
        output = []
        seen = set()
        for term in terms:
            normalized = term.strip().lower()
            if len(normalized) < 4 or normalized in ignored or normalized in seen:
                continue
            seen.add(normalized)
            output.append(normalized)
        return output[:25]

    def _compact_file(self, file: dict[str, Any]) -> dict[str, Any]:
        patch = file.get("patch") or ""
        patch_truncated = len(patch) > self.config.max_patch_chars
        return {
            "filename": file.get("filename"),
            "status": file.get("status"),
            "additions": file.get("additions"),
            "deletions": file.get("deletions"),
            "changes": file.get("changes"),
            "blob_url": file.get("blob_url"),
            "raw_url": file.get("raw_url"),
            "patch_chars": len(patch),
            "patch_truncated": patch_truncated,
            "patch_missing": not bool(patch),
            "patch": truncate_text(patch, self.config.max_patch_chars),
        }

    def _compact_review(self, review: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": review.get("id"),
            "state": review.get("state"),
            "submitted_at": review.get("submitted_at"),
            "user": compact_user(review.get("user")),
            "body": truncate_text(review.get("body") or "", 2500),
        }


def extract_actions_job_id(details_url: str) -> str | None:
    match = re.search(r"/actions/(?:runs/\d+/job|jobs)/(?P<job_id>\d+)", details_url)
    if not match:
        return None
    return match.group("job_id")


def target_pipeline_job_name(job_name: str) -> str | None:
    normalized = normalize_job_name(job_name)
    if normalized in TARGET_PIPELINE_JOB_NAMES:
        return normalized
    for target in TARGET_PIPELINE_JOB_NAMES:
        if normalized.startswith(f"{target} ") or normalized.startswith(f"{target} /"):
            return target
    return None


def normalize_job_name(job_name: str) -> str:
    normalized = re.sub(r"\s+", " ", job_name.strip().lower())
    normalized = normalized.split(" (", 1)[0].strip()
    return normalized


def previous_pipeline_failures_from_env() -> list[dict[str, Any]]:
    results = previous_pipeline_results_from_env()
    failures: list[dict[str, Any]] = []
    default_url = workflow_run_url_from_env()
    for raw_name, raw_value in results.items():
        target_name = target_pipeline_job_name(str(raw_name))
        if not target_name:
            continue
        if isinstance(raw_value, dict):
            conclusion = str(raw_value.get("result") or raw_value.get("conclusion") or "").lower()
            url = str(raw_value.get("url") or raw_value.get("html_url") or default_url or "").strip()
        else:
            conclusion = str(raw_value or "").lower()
            url = default_url
        if conclusion not in FAILED_CHECK_CONCLUSIONS:
            continue
        failures.append(
            {
                "name": target_name,
                "target_name": target_name,
                "status": "completed",
                "conclusion": conclusion,
                "html_url": url or None,
                "steps": [],
                "source": "workflow_needs",
            }
        )
    return failures


def previous_pipeline_results_from_env() -> dict[str, Any]:
    raw = os.getenv("PREVIOUS_PIPELINE_RESULTS_JSON")
    if not raw or not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if not isinstance(parsed, dict):
        return {}
    jobs = parsed.get("jobs", parsed)
    return jobs if isinstance(jobs, dict) else {}


def workflow_run_url_from_env() -> str | None:
    server = os.getenv("GITHUB_SERVER_URL") or "https://github.com"
    repo = os.getenv("GITHUB_REPOSITORY")
    run_id = os.getenv("GITHUB_RUN_ID")
    if not repo or not run_id:
        return None
    return f"{server.rstrip('/')}/{repo}/actions/runs/{run_id}"


def decode_actions_job_log(raw: bytes) -> str:
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            parts = []
            for name in sorted(archive.namelist()):
                if name.endswith("/"):
                    continue
                parts.append(archive.read(name).decode("utf-8", errors="replace"))
            return "\n".join(parts)
    except zipfile.BadZipFile:
        return raw.decode("utf-8", errors="replace")


def summarize_ci_log(text: str, *, max_lines: int = 80, max_chars: int = 7000) -> str:
    lines = [strip_github_log_prefix(strip_ansi(line)).rstrip() for line in text.splitlines()]
    if not lines:
        return ""

    failure_section = extract_ci_failure_section_indices(lines, max_lines=max_lines)
    if failure_section:
        excerpt = "\n".join(lines[index] for index in failure_section if lines[index].strip())
        return truncate_text(excerpt, max_chars)

    selected: list[int] = []
    seen: set[int] = set()
    priority_indices = [index for index, line in enumerate(lines) if is_failure_ci_line(line)]
    candidate_indices = priority_indices or [
        index for index, line in enumerate(lines) if is_actionable_ci_line(line)
    ]
    for index in candidate_indices:
        line = lines[index]
        if priority_indices and is_ci_boilerplate_line(line):
            continue
        left = max(0, index - 1)
        right = min(len(lines), index + (10 if priority_indices else 3))
        for candidate in range(left, right):
            if candidate in seen:
                continue
            if is_ci_boilerplate_line(lines[candidate]):
                continue
            seen.add(candidate)
            selected.append(candidate)
        if len(selected) >= max_lines:
            break

    if not selected:
        start = max(0, len(lines) - max_lines)
        selected = list(range(start, len(lines)))

    excerpt = "\n".join(lines[index] for index in selected if lines[index].strip())
    return truncate_text(excerpt, max_chars)


def strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text)


def strip_github_log_prefix(text: str) -> str:
    return re.sub(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z\s+", "", text)


def extract_ci_failure_section_indices(lines: list[str], *, max_lines: int) -> list[int]:
    selected: list[int] = []
    seen: set[int] = set()
    for index, line in enumerate(lines):
        if not is_ci_result_line(line, result="failed"):
            continue
        right = min(len(lines), index + 80)
        for candidate in range(index, right):
            candidate_line = lines[candidate]
            if candidate > index and is_ci_result_line(candidate_line):
                break
            if candidate in seen or is_ci_boilerplate_line(candidate_line) or not candidate_line.strip():
                continue
            seen.add(candidate)
            selected.append(candidate)
            if len(selected) >= max_lines:
                return selected
    return selected


def is_ci_result_line(line: str, *, result: str | None = None) -> bool:
    pattern = r"\.{5,}\s*(passed|failed|skipped|cancelled)\b"
    match = re.search(pattern, line.strip(), flags=re.IGNORECASE)
    if not match:
        return False
    if result is None:
        return True
    return match.group(1).lower() == result.lower()


def is_failure_ci_line(line: str) -> bool:
    if is_ci_boilerplate_line(line):
        return False
    lowered = line.lower()
    if re.search(r"\.{5,}\s*failed\b", lowered):
        return True
    if re.search(r"\b[FEW]\d{3}\b", line):
        return True
    return any(
        phrase in lowered
        for phrase in (
            "- hook id:",
            "- exit code:",
            "error:",
            "traceback",
            "assertionerror",
            "syntaxerror",
            "importerror",
            "modulenotfounderror",
            "secret type:",
            "location:",
            "potential secrets",
            "process completed with exit code",
        )
    )


def is_actionable_ci_line(line: str) -> bool:
    if is_ci_boilerplate_line(line):
        return False
    lowered = line.lower()
    if any(keyword in lowered for keyword in CI_LOG_KEYWORDS):
        return True
    return re.search(r"\b[FEW]\d{3}\b", line) is not None


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
        "initializing environment for ",
        "installing environment for ",
        "once installed this environment will be reused",
        "this may take a few minutes",
        "run pre-commit run --all-files",
    )
    if any(phrase in lowered for phrase in boilerplate):
        return True
    if re.search(r"\bpytest\s+suite/apps/", lowered):
        return True
    return False


def compact_workflow_job(job: dict[str, Any]) -> dict[str, Any]:
    steps = []
    for step in job.get("steps") or []:
        if not isinstance(step, dict):
            continue
        steps.append(
            {
                "name": step.get("name"),
                "status": step.get("status"),
                "conclusion": step.get("conclusion"),
                "number": step.get("number"),
                "started_at": step.get("started_at"),
                "completed_at": step.get("completed_at"),
            }
        )
    return {
        "id": job.get("id"),
        "run_id": job.get("run_id"),
        "run_attempt": job.get("run_attempt"),
        "name": job.get("name"),
        "status": job.get("status"),
        "conclusion": job.get("conclusion"),
        "started_at": job.get("started_at"),
        "completed_at": job.get("completed_at"),
        "html_url": job.get("html_url"),
        "steps": steps,
    }


def snippets_for_terms(text: str, terms: list[str], *, window: int = 700, max_snippets: int = 12) -> list[dict[str, Any]]:
    lowered = text.lower()
    snippets = []
    seen_ranges: list[tuple[int, int]] = []
    for term in terms:
        start = lowered.find(term.lower())
        if start < 0:
            continue
        left = max(0, start - window)
        right = min(len(text), start + len(term) + window)
        if any(abs(left - old_left) < 150 for old_left, _ in seen_ranges):
            continue
        seen_ranges.append((left, right))
        line = text.count("\n", 0, start) + 1
        snippets.append(
            {
                "term": term,
                "line": line,
                "snippet": truncate_text(text[left:right], 2 * window + 200),
            }
        )
        if len(snippets) >= max_snippets:
            break
    return snippets
