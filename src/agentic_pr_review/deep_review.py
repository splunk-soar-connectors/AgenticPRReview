"""Deep PR collection and chunked review helpers."""

from __future__ import annotations

import difflib
from pathlib import PurePosixPath
from typing import Any

from .collector import PRCollector, is_probably_text, is_relevant_full_file
from .config import RuntimeConfig
from .github_client import GitHubClient, GitHubError
from .historical_context import filter_historical_context_for_path
from .json_utils import truncate_text


DEEP_CONTEXT_FILES = {
    "manual_readme_content.md",
    "release_notes/unreleased.md",
    "pyproject.toml",
    "uv.lock",
}


class DeepPRCollector(PRCollector):
    def __init__(
        self,
        client: GitHubClient,
        config: RuntimeConfig,
        *,
        deep_max_file_bytes: int = 5_000_000,
        deep_max_file_chars: int = 250_000,
        deep_chunk_chars: int = 60_000,
    ) -> None:
        super().__init__(client, config)
        self.deep_max_file_bytes = deep_max_file_bytes
        self.deep_max_file_chars = deep_max_file_chars
        self.deep_chunk_chars = deep_chunk_chars

    def collect(
        self,
        repo: str,
        pr_number: int,
    ) -> dict[str, Any]:
        review_input = super().collect(repo, pr_number)
        return self.enrich_deep_context(repo, pr_number, review_input)

    def enrich_deep_context(self, repo: str, pr_number: int, review_input: dict[str, Any]) -> dict[str, Any]:
        pr = review_input.get("pr") or {}
        base_sha = ((pr.get("base") or {}).get("sha")) or ""
        head_sha = ((pr.get("head") or {}).get("sha")) or ""
        notes = review_input.setdefault("collector_notes", {})
        errors: list[str] = []

        try:
            files = self.client.list_pr_files(repo, pr_number)
        except GitHubError as exc:
            notes["deep_collection_enabled"] = True
            notes["deep_collection_error"] = str(exc)
            return review_input

        chunks: list[dict[str, Any]] = []
        full_diffs: dict[str, str] = {}
        head_files: dict[str, str] = {}
        base_files: dict[str, str] = {}
        skipped_binary: list[str] = []
        skipped_large_or_unavailable: list[str] = []

        for file_info in files:
            path = str(file_info.get("filename") or "")
            status = str(file_info.get("status") or "")
            previous_path = str(file_info.get("previous_filename") or path)
            if not path or not is_probably_text(path):
                if path:
                    skipped_binary.append(path)
                continue

            head_text = None if status == "removed" else self._fetch_deep_file(repo, path, head_sha, errors)
            base_text = None if status == "added" else self._fetch_deep_file(repo, previous_path, base_sha, errors)
            if head_text is None and base_text is None:
                skipped_large_or_unavailable.append(path)
                continue

            if head_text is not None and is_relevant_full_file(path):
                head_files[path] = truncate_text(head_text, self.deep_max_file_chars)
            if base_text is not None and is_relevant_full_file(previous_path):
                base_files[previous_path] = truncate_text(base_text, self.deep_max_file_chars)

            diff = generate_unified_diff(
                base_text or "",
                head_text or "",
                base_path=previous_path,
                head_path=path,
            )
            if not diff.strip():
                continue
            full_diffs[path] = diff
            chunks.extend(
                chunk_unified_diff(
                    file_info,
                    diff,
                    max_chars=self.deep_chunk_chars,
                )
            )

        self._merge_deep_files(review_input, head_files, base_files, chunks, full_diffs)
        review_input["deep_review"] = {
            "enabled": True,
            "strategy": "github_app_api_base_head_contents_local_diff_chunked",
            "chunk_chars": self.deep_chunk_chars,
            "changed_file_count": len(files),
            "chunk_count": len(chunks),
            "chunks": chunks,
            "skipped_binary_paths": skipped_binary[:100],
            "skipped_large_or_unavailable_paths": skipped_large_or_unavailable[:100],
            "errors": errors[:100],
        }
        notes.update(
            {
                "deep_collection_enabled": True,
                "deep_changed_file_count": len(files),
                "deep_chunk_count": len(chunks),
                "deep_head_file_count": len(head_files),
                "deep_base_file_count": len(base_files),
                "deep_skipped_binary_count": len(skipped_binary),
                "deep_skipped_large_or_unavailable_count": len(skipped_large_or_unavailable),
                "deep_error_count": len(errors),
                "deep_errors": errors[:20],
            }
        )
        return review_input

    def _fetch_deep_file(self, repo: str, path: str, ref: str, errors: list[str]) -> str | None:
        if not ref:
            return None
        try:
            return self.client.fetch_text_file_deep(repo, path, ref, max_bytes=self.deep_max_file_bytes)
        except GitHubError as exc:
            if "error 404" in str(exc).lower() or '"status":"404"' in str(exc).replace(" ", "").lower():
                return None
            errors.append(f"{path}@{ref}: {exc}")
            return None

    def _merge_deep_files(
        self,
        review_input: dict[str, Any],
        head_files: dict[str, str],
        base_files: dict[str, str],
        chunks: list[dict[str, Any]],
        full_diffs: dict[str, str],
    ) -> None:
        full_files = review_input.setdefault("full_files", {})
        existing_base_files = review_input.setdefault("base_files", {})
        for path, text in head_files.items():
            full_files[path] = text
        for path, text in base_files.items():
            existing_base_files[path] = text

        changed_by_path = {
            str(item.get("filename")): item
            for item in review_input.get("changed_files", [])
            if isinstance(item, dict) and item.get("filename")
        }
        replaced_patch_count = 0
        for path, diff in full_diffs.items():
            if path not in changed_by_path or not diff:
                continue
            file_info = changed_by_path[path]
            previous_patch = str(file_info.get("patch") or "")
            previous_missing = not previous_patch
            previous_truncated = bool(file_info.get("patch_truncated"))
            if previous_missing or previous_truncated or len(diff) > len(previous_patch):
                file_info["patch"] = diff
                file_info["patch_chars"] = len(diff)
                file_info["patch_missing"] = False
                file_info["patch_truncated"] = False
                file_info["deep_patch_reconstructed"] = True
                file_info["github_patch_missing"] = previous_missing
                file_info["github_patch_truncated"] = previous_truncated
                replaced_patch_count += 1

        notes = review_input.setdefault("collector_notes", {})
        notes["deep_reconstructed_patch_count"] = replaced_patch_count


def generate_unified_diff(base_text: str, head_text: str, *, base_path: str, head_path: str) -> str:
    base_lines = base_text.splitlines()
    head_lines = head_text.splitlines()
    return "\n".join(
        difflib.unified_diff(
            base_lines,
            head_lines,
            fromfile=f"a/{base_path}",
            tofile=f"b/{head_path}",
            lineterm="",
            n=6,
        )
    )


def chunk_unified_diff(file_info: dict[str, Any], diff: str, *, max_chars: int) -> list[dict[str, Any]]:
    path = str(file_info.get("filename") or "")
    previous_path = str(file_info.get("previous_filename") or path)
    headers, hunks = split_unified_diff(diff)
    if not hunks:
        hunks = [diff]

    chunks: list[str] = []
    current = headers
    for hunk in hunks:
        candidate = f"{current}\n{hunk}".strip() if current else hunk
        if len(candidate) <= max_chars or current == headers:
            current = candidate
            if len(current) > max_chars:
                chunks.extend(split_large_hunk(headers, hunk, max_chars=max_chars))
                current = headers
            continue
        if current.strip() and current.strip() != headers.strip():
            chunks.append(current)
        current = f"{headers}\n{hunk}".strip()
    if current.strip() and current.strip() != headers.strip():
        chunks.append(current)

    output = []
    total = len(chunks)
    for index, chunk in enumerate(chunks, start=1):
        output.append(
            {
                "id": f"{path}:{index}",
                "path": path,
                "previous_path": previous_path if previous_path != path else None,
                "status": file_info.get("status"),
                "additions": file_info.get("additions"),
                "deletions": file_info.get("deletions"),
                "changes": file_info.get("changes"),
                "file_type": PurePosixPath(path).suffix.lower().lstrip(".") or "unknown",
                "chunk_index": index,
                "chunk_total": total,
                "diff_chars": len(chunk),
                "diff": chunk,
            }
        )
    return output


def split_unified_diff(diff: str) -> tuple[str, list[str]]:
    lines = diff.splitlines()
    header_lines = []
    hunks: list[list[str]] = []
    current: list[str] | None = None
    for line in lines:
        if line.startswith("@@"):
            current = [line]
            hunks.append(current)
        elif current is None:
            header_lines.append(line)
        else:
            current.append(line)
    return "\n".join(header_lines), ["\n".join(hunk) for hunk in hunks]


def split_large_hunk(headers: str, hunk: str, *, max_chars: int) -> list[str]:
    lines = hunk.splitlines()
    if not lines:
        return []
    hunk_header = lines[0]
    chunks: list[str] = []
    current = f"{headers}\n{hunk_header}".strip()
    for line in lines[1:]:
        candidate = f"{current}\n{line}"
        if len(candidate) > max_chars and current.strip() != f"{headers}\n{hunk_header}".strip():
            chunks.append(current)
            current = f"{headers}\n{hunk_header}\n{line}".strip()
        else:
            current = candidate
    if current.strip():
        chunks.append(current)
    return chunks


def build_chunk_review_input(review_input: dict[str, Any], chunk: dict[str, Any]) -> dict[str, Any]:
    path = str(chunk.get("path") or "")
    previous_path = str(chunk.get("previous_path") or path)
    full_files = review_input.get("full_files") or {}
    base_files = review_input.get("base_files") or {}
    context_files = select_context_files(full_files, path)
    base_context_files = select_context_files(base_files, previous_path)

    return {
        "schema_version": "0.1",
        "review_scope": {
            "type": "deep_file_chunk",
            "chunk_id": chunk.get("id"),
            "path": path,
            "previous_path": chunk.get("previous_path"),
            "chunk_index": chunk.get("chunk_index"),
            "chunk_total": chunk.get("chunk_total"),
            "instruction": "Only report concrete issues proven by this chunk plus supplied PR context.",
        },
        "repo": review_input.get("repo"),
        "pr": review_input.get("pr"),
        "changed_files": [
            {
                "filename": path,
                "previous_filename": chunk.get("previous_path"),
                "status": chunk.get("status"),
                "additions": chunk.get("additions"),
                "deletions": chunk.get("deletions"),
                "changes": chunk.get("changes"),
                "patch": chunk.get("diff"),
                "deep_chunk": True,
                "chunk_index": chunk.get("chunk_index"),
                "chunk_total": chunk.get("chunk_total"),
            }
        ],
        "full_files": context_files,
        "base_files": base_context_files,
        "comments": review_input.get("comments") or {},
        "ci": review_input.get("ci") or {},
        "historical_context": filter_historical_context_for_path(
            review_input.get("historical_context"),
            path,
        ),
        "sdk_review_inventory": review_input.get("sdk_review_inventory"),
        "collector_notes": {
            **(review_input.get("collector_notes") or {}),
            "deep_chunk_review": True,
            "deep_chunk_id": chunk.get("id"),
        },
    }


def select_context_files(files: dict[str, str], primary_path: str) -> dict[str, str]:
    output: dict[str, str] = {}
    for path, text in files.items():
        name = PurePosixPath(path).name
        if (
            path == primary_path
            or name in DEEP_CONTEXT_FILES
            or ("/" not in path and path.endswith(".json"))
            or is_related_view_context_file(path, primary_path)
        ):
            output[path] = truncate_text(text, 80_000)
    return output


def is_related_view_context_file(path: str, primary_path: str) -> bool:
    path_lower = path.lower()
    primary_lower = primary_path.lower()
    path_suffix = PurePosixPath(path).suffix.lower()
    primary_suffix = PurePosixPath(primary_path).suffix.lower()
    template_suffixes = {".html", ".xml", ".jinja", ".j2"}

    if path == primary_path:
        return False
    if primary_suffix in template_suffixes and path_suffix == ".py" and "view" in path_lower:
        return True
    if path_suffix in template_suffixes and (
        "view" in primary_lower
        or "dashboard" in primary_lower
        or "template" in primary_lower
        or "/templates/" in path_lower
        or "dashboard" in path_lower
        or "/default/data/ui/" in path_lower
    ):
        return True
    return False


def filter_findings_for_chunk(deterministic_findings: list[dict[str, Any]], chunk: dict[str, Any]) -> list[dict[str, Any]]:
    path = str(chunk.get("path") or "")
    return [
        finding
        for finding in deterministic_findings
        if not finding.get("file") or str(finding.get("file")) == path
    ]


def dedupe_findings(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    seen: set[tuple[str, str, str, str]] = set()
    for finding in findings:
        key = (
            str(finding.get("category") or ""),
            str(finding.get("title") or "").lower(),
            str(finding.get("file") or ""),
            str(finding.get("line") or finding.get("code_reference") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        output.append(finding)
    return output
