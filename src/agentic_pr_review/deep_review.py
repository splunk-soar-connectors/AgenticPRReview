"""Deep PR collection and chunked review helpers."""

from __future__ import annotations

import ast
import difflib
from pathlib import PurePosixPath
import re
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

CIRCUIT_PACKET_DIFF_CHARS = 30_000
CIRCUIT_PACKET_PRIMARY_CONTEXT_CHARS = 10_000
CIRCUIT_PACKET_RELATED_CONTEXT_CHARS = 5_000
CIRCUIT_PACKET_RELATED_FILE_LIMIT = 8
CONTEXT_SOURCE_FILE_CHARS = 250_000
CONTEXT_AWARE_HUNK_RADIUS = 45
PYTHON_SEMANTIC_CONTEXT_CHARS = 12_000
PYTHON_SEMANTIC_MAX_SYMBOLS = 16
PYTHON_SEMANTIC_MAX_SNIPPETS = 12
PYTHON_FUNCTION_CHUNK_TARGET_CHARS = 12_000
STRUCTURED_UNIT_CHUNK_TARGET_CHARS = 12_000
PYTHON_RELATED_CONTEXT_FILE_LIMIT = 20
CONTEXT_AWARE_REVIEW_STRATEGY = "changed_hunks_enclosing_scope_semantic_context"

IGNORED_EXTERNAL_PYTHON_MODULES = {
    "abc",
    "argparse",
    "ast",
    "base64",
    "collections",
    "contextlib",
    "copy",
    "csv",
    "datetime",
    "decimal",
    "functools",
    "hashlib",
    "hmac",
    "http",
    "io",
    "ipaddress",
    "itertools",
    "json",
    "logging",
    "math",
    "os",
    "pathlib",
    "phantom",
    "pydantic",
    "random",
    "re",
    "requests",
    "shutil",
    "ssl",
    "sys",
    "tempfile",
    "time",
    "typing",
    "urllib",
    "uuid",
    "zipfile",
}

LOW_SIGNAL_MODEL_FILENAMES = {
    "license",
    "notice",
    "readme.md",
}

RISKY_MODEL_TERMS = {
    "action_result",
    "add_data",
    "summary",
    "save_artifact",
    "save_container",
    "source_data_identifier",
    "on_poll",
    "is_poll_now",
    "checkpoint",
    "save_state",
    "load_state",
    "requests.",
    "httpx.",
    "timeout",
    "verify",
    "verify_server_cert",
    "token",
    "oauth",
    "authorization",
    "client_secret",
    "debug_print",
    "logger.",
    "read_only",
    "param(",
    "assetfield",
    "actionoutput",
    "model_validate",
    "base64",
    "b64decode",
    "json.loads",
    "pagination",
    "nextlink",
    "next_page",
    "rate limit",
    "429",
    "pull_request_target",
    "secrets.",
    "self-hosted",
    "<<<<<<<",
    "=======",
    ">>>>>>>",
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
                    head_text=head_text or "",
                    base_text=base_text or "",
                )
            )

        related_paths = sorted(collect_related_python_context_paths(head_files) - set(head_files))
        related_fetched: list[str] = []
        for related_path in related_paths[:PYTHON_RELATED_CONTEXT_FILE_LIMIT]:
            related_text = self._fetch_deep_file(repo, related_path, head_sha, errors)
            if related_text is None or not is_relevant_full_file(related_path):
                continue
            head_files[related_path] = truncate_text(related_text, self.deep_max_file_chars)
            related_fetched.append(related_path)

        self._merge_deep_files(review_input, head_files, base_files, chunks, full_diffs)
        review_input["deep_review"] = {
            "enabled": True,
            "strategy": CONTEXT_AWARE_REVIEW_STRATEGY,
            "chunk_chars": self.deep_chunk_chars,
            "changed_file_count": len(files),
            "chunk_count": len(chunks),
            "chunks": chunks,
            "skipped_binary_paths": skipped_binary[:100],
            "skipped_large_or_unavailable_paths": skipped_large_or_unavailable[:100],
            "errors": errors[:100],
            "related_context_paths": related_fetched[:100],
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
                "deep_related_context_file_count": len(related_fetched),
                "deep_related_context_paths": related_fetched[:20],
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


def chunk_unified_diff(
    file_info: dict[str, Any],
    diff: str,
    *,
    max_chars: int,
    head_text: str = "",
    base_text: str = "",
) -> list[dict[str, Any]]:
    path = str(file_info.get("filename") or "")
    previous_path = str(file_info.get("previous_filename") or path)
    function_chunks = chunk_python_diff_by_function(
        file_info,
        diff,
        max_chars=max_chars,
        head_text=head_text,
        base_text=base_text,
    )
    if function_chunks:
        return function_chunks

    structured_chunks = chunk_structured_diff_by_unit(
        file_info,
        diff,
        max_chars=max_chars,
        head_text=head_text,
        base_text=base_text,
    )
    if structured_chunks:
        return structured_chunks

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
                "chunk_strategy": "changed_hunks",
                "context_strategy": CONTEXT_AWARE_REVIEW_STRATEGY,
            }
        )
    return output


def chunk_structured_diff_by_unit(
    file_info: dict[str, Any],
    diff: str,
    *,
    max_chars: int,
    head_text: str,
    base_text: str,
) -> list[dict[str, Any]]:
    path = str(file_info.get("filename") or "")
    suffix = PurePosixPath(path).suffix.lower()
    if suffix not in {".json", ".yml", ".yaml", ".toml", ".xml"}:
        return []
    if not (head_text.strip() or base_text.strip()):
        return []

    head_scopes = structured_scopes_from_source(head_text, suffix)
    base_scopes = structured_scopes_from_source(base_text, suffix)
    if not head_scopes and not base_scopes:
        return []

    headers, hunks = split_unified_diff(diff)
    if not hunks:
        return []

    grouped: dict[str, dict[str, Any]] = {}
    for hunk in hunks:
        for scope, scoped_hunk in split_hunk_by_structured_scope(hunk, head_scopes, base_scopes):
            key = structured_scope_key(scope)
            item = grouped.setdefault(key, {"scope": scope, "hunks": [], "hunk_count": 0})
            item["hunks"].append(scoped_hunk)
            item["hunk_count"] = int(item.get("hunk_count") or 0) + 1

    if not grouped:
        return []

    target_chars = min(max_chars, STRUCTURED_UNIT_CHUNK_TARGET_CHARS)
    path_suffix = suffix.lstrip(".") or "unknown"
    previous_path = str(file_info.get("previous_filename") or path)
    output: list[dict[str, Any]] = []
    for item in grouped.values():
        scope = item.get("scope") or {}
        chunk_diff = f"{headers}\n" + "\n".join(str(hunk) for hunk in item.get("hunks") or [])
        chunk_diff = chunk_diff.strip()
        if not chunk_diff or chunk_diff == headers.strip():
            continue
        parts: list[str]
        if len(chunk_diff) <= target_chars:
            parts = [chunk_diff]
        else:
            parts = []
            for hunk in item.get("hunks") or []:
                parts.extend(split_large_hunk(headers, str(hunk), max_chars=target_chars))
        for part in parts:
            output.append(
                {
                    "id": "",
                    "path": path,
                    "previous_path": previous_path if previous_path != path else None,
                    "status": file_info.get("status"),
                    "additions": file_info.get("additions"),
                    "deletions": file_info.get("deletions"),
                    "changes": file_info.get("changes"),
                    "file_type": path_suffix,
                    "chunk_index": 0,
                    "chunk_total": 0,
                    "diff_chars": len(part),
                    "diff": part,
                    "chunk_strategy": str(scope.get("chunk_strategy") or "structured_unit"),
                    "context_strategy": CONTEXT_AWARE_REVIEW_STRATEGY,
                    "changed_hunk_count": item.get("hunk_count"),
                    "logical_unit": scope.get("name") or "document",
                    "logical_unit_kind": scope.get("kind") or "document",
                    "logical_unit_start_line": scope.get("start_line"),
                    "logical_unit_end_line": scope.get("end_line"),
                }
            )

    total = len(output)
    for index, chunk in enumerate(output, start=1):
        chunk["id"] = f"{path}:{index}"
        chunk["chunk_index"] = index
        chunk["chunk_total"] = total
    return output


def chunk_python_diff_by_function(
    file_info: dict[str, Any],
    diff: str,
    *,
    max_chars: int,
    head_text: str,
    base_text: str,
) -> list[dict[str, Any]]:
    path = str(file_info.get("filename") or "")
    if not path.endswith(".py"):
        return []

    has_meaningful_context = bool(head_text.strip() or base_text.strip())
    if not has_meaningful_context:
        return []

    head_scopes = python_scopes_from_source(head_text)
    base_scopes = python_scopes_from_source(base_text)
    if not head_scopes and not base_scopes:
        return []

    headers, hunks = split_unified_diff(diff)
    if not hunks:
        return []

    grouped: dict[str, dict[str, Any]] = {}
    for hunk in hunks:
        for scope, scoped_hunk in split_python_hunk_by_scope(hunk, head_scopes, base_scopes):
            key = python_scope_key(scope)
            item = grouped.setdefault(key, {"scope": scope, "hunks": [], "hunk_count": 0})
            item["hunks"].append(scoped_hunk)
            item["hunk_count"] = int(item.get("hunk_count") or 0) + 1

    if not grouped:
        return []

    target_chars = min(max_chars, PYTHON_FUNCTION_CHUNK_TARGET_CHARS)
    path_suffix = PurePosixPath(path).suffix.lower().lstrip(".") or "unknown"
    previous_path = str(file_info.get("previous_filename") or path)
    output: list[dict[str, Any]] = []

    for item in grouped.values():
        scope = item.get("scope")
        chunk_diff = f"{headers}\n" + "\n".join(str(hunk) for hunk in item.get("hunks") or [])
        chunk_diff = chunk_diff.strip()
        if not chunk_diff or chunk_diff == headers.strip():
            continue
        parts: list[str]
        if len(chunk_diff) <= target_chars:
            parts = [chunk_diff]
        else:
            parts = []
            for hunk in item.get("hunks") or []:
                parts.extend(split_large_hunk(headers, str(hunk), max_chars=target_chars))
        for part in parts:
            output.append(
                {
                    "id": "",
                    "path": path,
                    "previous_path": previous_path if previous_path != path else None,
                    "status": file_info.get("status"),
                    "additions": file_info.get("additions"),
                    "deletions": file_info.get("deletions"),
                    "changes": file_info.get("changes"),
                    "file_type": path_suffix,
                    "chunk_index": 0,
                    "chunk_total": 0,
                    "diff_chars": len(part),
                    "diff": part,
                    "chunk_strategy": "python_function",
                    "context_strategy": CONTEXT_AWARE_REVIEW_STRATEGY,
                    "changed_hunk_count": item.get("hunk_count"),
                    "python_scope": scope.get("name") if scope else "module",
                    "python_scope_kind": scope.get("kind") if scope else "module",
                    "python_scope_start_line": scope.get("start_line") if scope else None,
                    "python_scope_end_line": scope.get("end_line") if scope else None,
                }
            )

    total = len(output)
    for index, chunk in enumerate(output, start=1):
        chunk["id"] = f"{path}:{index}"
        chunk["chunk_index"] = index
        chunk["chunk_total"] = total
    return output


def python_scopes_from_source(source: str) -> list[dict[str, Any]]:
    if not source.strip():
        return []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    scopes: list[dict[str, Any]] = []

    def visit_body(body: list[ast.stmt], prefix: str = "") -> None:
        for node in body:
            if not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            name = f"{prefix}.{node.name}" if prefix else node.name
            kind = "class" if isinstance(node, ast.ClassDef) else "function"
            start_line = int(getattr(node, "lineno", 1) or 1)
            end_line = int(getattr(node, "end_lineno", start_line) or start_line)
            scopes.append(
                {
                    "name": name,
                    "kind": kind,
                    "start_line": start_line,
                    "end_line": end_line,
                }
            )
            visit_body(list(getattr(node, "body", []) or []), name)

    visit_body(list(tree.body))
    return scopes


def split_python_hunk_by_scope(
    hunk: str,
    head_scopes: list[dict[str, Any]],
    base_scopes: list[dict[str, Any]],
) -> list[tuple[dict[str, Any] | None, str]]:
    lines = hunk.splitlines()
    if not lines:
        return []
    parsed = parse_hunk_header(lines[0])
    if parsed is None:
        return [(None, hunk)]

    old_line = int(parsed["old_start"])
    new_line = int(parsed["new_start"])
    grouped: dict[str, dict[str, Any]] = {}

    for line in lines[1:]:
        is_added = line.startswith("+") and not line.startswith("+++")
        is_removed = line.startswith("-") and not line.startswith("---")
        old_for_line = None if is_added else old_line
        new_for_line = None if is_removed else new_line
        scope = None
        if new_for_line is not None:
            scope = find_python_scope_for_line(head_scopes, new_for_line)
        if scope is None and old_for_line is not None:
            scope = find_python_scope_for_line(base_scopes, old_for_line)

        key = python_scope_key(scope)
        item = grouped.setdefault(key, {"scope": scope, "lines": [lines[0]], "changed": False})
        item["lines"].append(line)
        if (is_added or is_removed) and line[1:].strip():
            item["changed"] = True

        if not is_added:
            old_line += 1
        if not is_removed:
            new_line += 1

    return [
        (item.get("scope"), "\n".join(item.get("lines") or []))
        for item in grouped.values()
        if item.get("changed") and len(item.get("lines") or []) > 1
    ]


def find_python_scope_for_line(scopes: list[dict[str, Any]], line_no: int) -> dict[str, Any] | None:
    candidates = [
        scope
        for scope in scopes
        if int(scope.get("start_line") or 0) <= line_no <= int(scope.get("end_line") or 0)
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda scope: (
            int(scope.get("end_line") or 0) - int(scope.get("start_line") or 0),
            -int(scope.get("start_line") or 0),
        ),
    )


def python_scope_key(scope: dict[str, Any] | None) -> str:
    if not scope:
        return "module"
    return f"{scope.get('kind') or 'scope'}:{scope.get('name') or 'unknown'}"


def collect_related_python_context_paths(files: dict[str, str]) -> set[str]:
    """Infer small, local Python modules worth fetching for semantic context."""

    output: set[str] = set()
    known_roots = project_roots_from_python_paths(files)
    for path, text in files.items():
        if not path.endswith(".py") or not text:
            continue
        output.update(infer_local_python_import_paths(path, text, known_roots=known_roots))
    return output


def project_roots_from_python_paths(files: dict[str, str]) -> set[str]:
    roots: set[str] = {""}
    for path in files:
        parts = PurePosixPath(path).parts
        if len(parts) >= 2 and parts[-1].endswith(".py"):
            roots.add(parts[0])
    return roots


def infer_local_python_import_paths(path: str, text: str, *, known_roots: set[str] | None = None) -> set[str]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return set()

    current_parts = list(PurePosixPath(path).parts[:-1])
    roots = known_roots or {""}
    output: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            output.update(resolve_import_from_paths(node, current_parts=current_parts, known_roots=roots))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                output.update(resolve_absolute_module_paths(alias.name, known_roots=roots))

    return {
        candidate
        for candidate in output
        if candidate
        and candidate != path
        and candidate.endswith(".py")
        and not candidate.endswith("/__init__.py")
        and is_probably_text(candidate)
    }


def resolve_import_from_paths(
    node: ast.ImportFrom,
    *,
    current_parts: list[str],
    known_roots: set[str],
) -> set[str]:
    module_parts = [part for part in (node.module or "").split(".") if part]
    output: set[str] = set()
    if node.level:
        package_len = max(0, len(current_parts) - node.level + 1)
        base_parts = current_parts[:package_len] + module_parts
        output.update(module_path_candidates(base_parts))
        if not module_parts:
            for alias in node.names:
                output.update(module_path_candidates(base_parts + [alias.name]))
        return output

    if not module_parts:
        return output
    output.update(resolve_absolute_module_paths(".".join(module_parts), known_roots=known_roots))
    return output


def resolve_absolute_module_paths(module_name: str, *, known_roots: set[str]) -> set[str]:
    parts = [part for part in module_name.split(".") if part]
    if not parts:
        return set()
    if parts[0] in IGNORED_EXTERNAL_PYTHON_MODULES:
        return set()

    output = set(module_path_candidates(parts))
    for root in known_roots:
        if not root or parts[0] == root:
            continue
        output.update(module_path_candidates([root, *parts]))
    return output


def module_path_candidates(parts: list[str]) -> set[str]:
    if not parts:
        return set()
    path = "/".join(parts)
    return {f"{path}.py", f"{path}/__init__.py"}


def structured_scopes_from_source(source: str, suffix: str) -> list[dict[str, Any]]:
    if not source.strip():
        return []
    if suffix == ".json":
        return json_scopes_from_source(source)
    if suffix in {".yml", ".yaml"}:
        return yaml_scopes_from_source(source)
    if suffix == ".toml":
        return toml_scopes_from_source(source)
    if suffix == ".xml":
        return xml_scopes_from_source(source)
    return []


def json_scopes_from_source(source: str) -> list[dict[str, Any]]:
    scopes: list[dict[str, Any]] = []
    stack: list[dict[str, Any]] = []
    in_string = False
    escaped = False
    lines = source.splitlines()
    for line_no, line in enumerate(lines, start=1):
        for index, char in enumerate(line):
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == "\"":
                    in_string = False
                continue
            if char == "\"":
                in_string = True
                continue
            if char not in "{}[]":
                continue
            if char in "{[":
                name = json_key_before_position(line, index)
                if not name:
                    name = "$root" if not stack else f"{stack[-1]['name']}[]"
                stack.append(
                    {
                        "kind": "json_object" if char == "{" else "json_array",
                        "chunk_strategy": "json_object" if char == "{" else "json_array",
                        "name": name,
                        "start_line": line_no,
                        "open": char,
                    }
                )
                continue
            expected_open = "{" if char == "}" else "["
            while stack:
                scope = stack.pop()
                if scope.get("open") != expected_open:
                    continue
                scope = {
                    key: value
                    for key, value in scope.items()
                    if key != "open"
                }
                scope["end_line"] = line_no
                scope["name"] = json_scope_name(source, scope)
                scopes.append(scope)
                break

    last_line = max(1, len(lines))
    for scope in stack:
        scope = {key: value for key, value in scope.items() if key != "open"}
        scope["end_line"] = last_line
        scope["name"] = json_scope_name(source, scope)
        scopes.append(scope)

    if not scopes:
        scopes.append(document_scope("json_document", "json_document", len(lines)))
    return sorted(scopes, key=lambda item: (int(item.get("start_line") or 0), int(item.get("end_line") or 0)))


def json_key_before_position(line: str, index: int) -> str | None:
    prefix = line[:index]
    matches = list(re.finditer(r'"((?:\\.|[^"\\])*)"\s*:\s*$', prefix))
    if not matches:
        return None
    return matches[-1].group(1)


def json_scope_name(source: str, scope: dict[str, Any]) -> str:
    name = str(scope.get("name") or "")
    if name and name != "$root" and not name.endswith("[]"):
        return name
    start = max(1, int(scope.get("start_line") or 1))
    end = max(start, int(scope.get("end_line") or start))
    snippet = "\n".join(source.splitlines()[start - 1:end])
    for key in ("identifier", "action", "name", "contains", "data_path", "parameter", "type"):
        match = re.search(rf'"{re.escape(key)}"\s*:\s*"((?:\\.|[^"\\])*)"', snippet)
        if match:
            return f"{key}:{match.group(1)}"
    return name or "$root"


def yaml_scopes_from_source(source: str) -> list[dict[str, Any]]:
    lines = source.splitlines()
    candidates: list[dict[str, Any]] = []
    stack: list[dict[str, Any]] = []
    key_pattern = re.compile(r"^(?P<indent>\s*)(?:-\s*)?(?P<key>[A-Za-z0-9_.-][^:#{}\[\]]{0,80}?):(?:\s|$)")
    for line_no, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = key_pattern.match(line)
        if not match:
            continue
        key = match.group("key").strip().strip("'\"")
        if not key:
            continue
        indent = len(match.group("indent").replace("\t", "    "))
        while stack and int(stack[-1]["indent"]) >= indent:
            stack.pop()
        name = ".".join([str(item["key"]) for item in stack] + [key])
        item = {
            "kind": "yaml_section",
            "chunk_strategy": "yaml_section",
            "name": name,
            "key": key,
            "indent": indent,
            "start_line": line_no,
            "end_line": len(lines),
        }
        candidates.append(item)
        value_after_colon = line[match.end():].strip()
        is_container = not value_after_colon or value_after_colon.startswith(("#", "|", ">"))
        if is_container:
            stack.append(item)

    for index, item in enumerate(candidates):
        indent = int(item["indent"])
        for later in candidates[index + 1:]:
            if int(later["indent"]) <= indent:
                item["end_line"] = int(later["start_line"]) - 1
                break

    return [
        {key: value for key, value in item.items() if key not in {"indent", "key"}}
        for item in candidates
    ] or [document_scope("yaml_document", "yaml_section", len(lines))]


def toml_scopes_from_source(source: str) -> list[dict[str, Any]]:
    lines = source.splitlines()
    scopes: list[dict[str, Any]] = []
    section_pattern = re.compile(r"^\s*(?P<header>\[\[?[^\]]+\]?\])")
    matches: list[tuple[int, str]] = []
    for line_no, line in enumerate(lines, start=1):
        match = section_pattern.match(line)
        if match:
            matches.append((line_no, match.group("header").strip()))
    for index, (start_line, header) in enumerate(matches):
        end_line = (matches[index + 1][0] - 1) if index + 1 < len(matches) else len(lines)
        scopes.append(
            {
                "kind": "toml_section",
                "chunk_strategy": "toml_section",
                "name": header.strip("[]"),
                "start_line": start_line,
                "end_line": end_line,
            }
        )
    return scopes or [document_scope("toml_document", "toml_section", len(lines))]


def xml_scopes_from_source(source: str) -> list[dict[str, Any]]:
    lines = source.splitlines()
    scopes: list[dict[str, Any]] = []
    stack: list[dict[str, Any]] = []
    tag_pattern = re.compile(r"<(?P<closing>/)?(?P<tag>[A-Za-z_][\w:.-]*)(?P<attrs>[^>]*)>")
    for line_no, line in enumerate(lines, start=1):
        for match in tag_pattern.finditer(line):
            token = match.group(0)
            if token.startswith(("<?", "<!")):
                continue
            tag = match.group("tag")
            if match.group("closing"):
                for index in range(len(stack) - 1, -1, -1):
                    if stack[index]["tag"] != tag:
                        continue
                    scope = stack.pop(index)
                    scope["end_line"] = line_no
                    scopes.append({key: value for key, value in scope.items() if key != "tag"})
                    break
                continue
            if token.endswith("/>"):
                scopes.append(
                    {
                        "kind": "xml_element",
                        "chunk_strategy": "xml_element",
                        "name": xml_scope_name(tag, match.group("attrs") or ""),
                        "start_line": line_no,
                        "end_line": line_no,
                    }
                )
                continue
            stack.append(
                {
                    "kind": "xml_element",
                    "chunk_strategy": "xml_element",
                    "name": xml_scope_name(tag, match.group("attrs") or ""),
                    "tag": tag,
                    "start_line": line_no,
                }
            )

    last_line = max(1, len(lines))
    for scope in stack:
        scope["end_line"] = last_line
        scopes.append({key: value for key, value in scope.items() if key != "tag"})
    return scopes or [document_scope("xml_document", "xml_element", len(lines))]


def xml_scope_name(tag: str, attrs: str) -> str:
    for attr in ("id", "name", "label"):
        match = re.search(rf"\b{attr}\s*=\s*['\"]([^'\"]+)['\"]", attrs)
        if match:
            return f"{tag}#{match.group(1)}"
    return tag


def document_scope(name: str, strategy: str, line_count: int) -> dict[str, Any]:
    return {
        "kind": "document",
        "chunk_strategy": strategy,
        "name": name,
        "start_line": 1,
        "end_line": max(1, line_count),
    }


def split_hunk_by_structured_scope(
    hunk: str,
    head_scopes: list[dict[str, Any]],
    base_scopes: list[dict[str, Any]],
) -> list[tuple[dict[str, Any] | None, str]]:
    lines = hunk.splitlines()
    if not lines:
        return []
    parsed = parse_hunk_header(lines[0])
    if parsed is None:
        return [(None, hunk)]

    old_line = int(parsed["old_start"])
    new_line = int(parsed["new_start"])
    grouped: dict[str, dict[str, Any]] = {}
    for line in lines[1:]:
        is_added = line.startswith("+") and not line.startswith("+++")
        is_removed = line.startswith("-") and not line.startswith("---")
        old_for_line = None if is_added else old_line
        new_for_line = None if is_removed else new_line
        scope = None
        if new_for_line is not None:
            scope = find_structured_scope_for_line(head_scopes, new_for_line)
        if scope is None and old_for_line is not None:
            scope = find_structured_scope_for_line(base_scopes, old_for_line)

        key = structured_scope_key(scope)
        item = grouped.setdefault(key, {"scope": scope, "lines": [lines[0]], "changed": False})
        item["lines"].append(line)
        if (is_added or is_removed) and line[1:].strip():
            item["changed"] = True

        if not is_added:
            old_line += 1
        if not is_removed:
            new_line += 1

    return [
        (item.get("scope"), "\n".join(item.get("lines") or []))
        for item in grouped.values()
        if item.get("changed") and len(item.get("lines") or []) > 1
    ]


def find_structured_scope_for_line(scopes: list[dict[str, Any]], line_no: int) -> dict[str, Any] | None:
    candidates = [
        scope
        for scope in scopes
        if int(scope.get("start_line") or 0) <= line_no <= int(scope.get("end_line") or 0)
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda scope: (
            int(scope.get("end_line") or 0) - int(scope.get("start_line") or 0),
            -int(scope.get("start_line") or 0),
        ),
    )


def structured_scope_key(scope: dict[str, Any] | None) -> str:
    if not scope:
        return "document"
    return f"{scope.get('kind') or 'unit'}:{scope.get('name') or 'document'}"


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
    context_char_limit = int(chunk.get("context_char_limit") or 14_000)
    if path.endswith(".py") and not chunk.get("adaptive_retry"):
        context_char_limit = min(context_char_limit, 8_000)
    source_char_limit = int(chunk.get("source_char_limit") or CONTEXT_SOURCE_FILE_CHARS)
    context_files = select_context_files(
        full_files,
        path,
        max_chars=source_char_limit,
        include_all_source=True,
    )
    base_context_files = select_context_files(
        base_files,
        previous_path,
        max_chars=source_char_limit,
        include_all_source=True,
    )
    review_packet = build_circuit_review_packet(
        review_input,
        chunk,
        context_files,
        base_context_files,
    )
    instruction = "Only report concrete issues proven by this chunk plus supplied PR context."
    if chunk.get("adaptive_retry"):
        if chunk.get("proactive_review"):
            instruction += (
                " This is a proactive smaller slice of a timeout-risk file chunk; review this slice fully "
                "and do not assume sibling slices are already represented here."
            )
        else:
            instruction += (
                " This is a smaller retry slice of a larger file chunk after a transient model gateway failure; "
                "review this slice fully and do not assume sibling slices are already represented here."
            )
    if chunk.get("model_review_reason"):
        instruction += f" Circuit packet planner reason: {chunk.get('model_review_reason')}."

    return {
        "schema_version": "0.1",
        "review_scope": {
            "type": "circuit_review_packet",
            "chunk_id": chunk.get("id"),
            "parent_chunk_id": chunk.get("parent_chunk_id"),
            "adaptive_retry": bool(chunk.get("adaptive_retry")),
            "proactive_review": bool(chunk.get("proactive_review")),
            "path": path,
            "previous_path": chunk.get("previous_path"),
            "chunk_index": chunk.get("chunk_index"),
            "chunk_total": chunk.get("chunk_total"),
            "chunk_strategy": chunk.get("chunk_strategy"),
            "context_strategy": chunk.get("context_strategy"),
            "changed_hunk_count": chunk.get("changed_hunk_count"),
            "python_scope": chunk.get("python_scope"),
            "python_scope_kind": chunk.get("python_scope_kind"),
            "python_scope_start_line": chunk.get("python_scope_start_line"),
            "python_scope_end_line": chunk.get("python_scope_end_line"),
            "logical_unit": chunk.get("logical_unit"),
            "logical_unit_kind": chunk.get("logical_unit_kind"),
            "logical_unit_start_line": chunk.get("logical_unit_start_line"),
            "logical_unit_end_line": chunk.get("logical_unit_end_line"),
            "model_review_priority": chunk.get("model_review_priority"),
            "model_review_reason": chunk.get("model_review_reason"),
            "instruction": instruction,
        },
        "review_packet": review_packet,
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
                "patch": review_packet.get("diff"),
                "deep_chunk": True,
                "chunk_index": chunk.get("chunk_index"),
                "chunk_total": chunk.get("chunk_total"),
            }
        ],
        "full_files": review_packet.get("head_context_files") or context_files,
        "base_files": review_packet.get("base_context_files") or base_context_files,
        "comments": select_chunk_comments(review_input.get("comments") or {}, path),
        "ci": select_chunk_ci(review_input.get("ci") or {}, path),
        "historical_context": filter_historical_context_for_path(
            review_input.get("historical_context"),
            path,
        ),
        "sdk_review_inventory": review_input.get("sdk_review_inventory"),
        "collector_notes": {
            **(review_input.get("collector_notes") or {}),
            "deep_chunk_review": True,
            "deep_chunk_id": chunk.get("id"),
            "deep_parent_chunk_id": chunk.get("parent_chunk_id"),
            "deep_adaptive_retry": bool(chunk.get("adaptive_retry")),
            "deep_proactive_review": bool(chunk.get("proactive_review")),
        },
    }


def plan_circuit_review_chunks(
    chunks: list[dict[str, Any]],
    deterministic_findings: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    planned: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for chunk in chunks:
        decision = classify_chunk_for_circuit(chunk, deterministic_findings)
        annotated = dict(chunk)
        annotated["model_review_priority"] = decision["priority"]
        annotated["model_review_reason"] = decision["reason"]
        if decision["action"] == "skip":
            skipped.append(
                {
                    "id": chunk.get("id"),
                    "path": chunk.get("path"),
                    "reason": decision["reason"],
                    "priority": decision["priority"],
                }
            )
            continue
        planned.append(annotated)
    return planned, skipped


def classify_chunk_for_circuit(
    chunk: dict[str, Any],
    deterministic_findings: list[dict[str, Any]],
) -> dict[str, str]:
    path = str(chunk.get("path") or "")
    name = PurePosixPath(path).name.lower()
    suffix = PurePosixPath(path).suffix.lower()
    diff = str(chunk.get("diff") or "")
    diff_lower = diff.lower()

    if name in LOW_SIGNAL_MODEL_FILENAMES:
        return {"action": "skip", "priority": "low", "reason": f"{name} is low-signal for model review"}
    if any(str(finding.get("file") or "") == path for finding in deterministic_findings if isinstance(finding, dict)):
        return {"action": "review", "priority": "high", "reason": "deterministic finding targets this file"}
    if name == "__init__.py" and is_metadata_only_diff(diff):
        return {"action": "skip", "priority": "low", "reason": "__init__.py change is metadata-only"}
    if any(term in diff_lower for term in RISKY_MODEL_TERMS):
        return {"action": "review", "priority": "high", "reason": "diff contains connector-risk keywords"}
    if chunk.get("status") == "removed" and suffix in {".py", ".json", ".toml", ".yml", ".yaml"}:
        return {"action": "review", "priority": "high", "reason": "removed code/config can be a regression"}
    if is_test_path(path):
        return {"action": "review", "priority": "medium", "reason": "test change can affect review evidence"}
    if path.startswith(".github/workflows/"):
        return {"action": "review", "priority": "medium", "reason": "workflow changes can affect CI/secrets behavior"}
    if suffix in {".py", ".json", ".toml", ".yml", ".yaml", ".xml", ".html", ".jinja", ".j2"}:
        return {"action": "review", "priority": "medium", "reason": "connector source/config change"}
    if path.startswith("release_notes/") or name == "manual_readme_content.md":
        return {"action": "review", "priority": "medium", "reason": "user-visible docs/release-note source"}
    return {"action": "skip", "priority": "low", "reason": "low-signal non-connector chunk"}


def is_metadata_only_diff(diff: str) -> bool:
    changed_lines: list[str] = []
    for raw_line in diff.splitlines():
        if raw_line.startswith(("+++", "---", "@@")):
            continue
        if raw_line.startswith(("+", "-")):
            changed_lines.append(raw_line[1:].strip())
    if not changed_lines:
        return True
    metadata_terms = ("copyright", "license", "generated", "__version__", "version", "year")
    for line in changed_lines:
        lowered = line.lower()
        if not line or line in {'"""', "'''"}:
            continue
        if lowered.startswith(("#", "\"\"\"", "'''")) and any(term in lowered for term in metadata_terms):
            continue
        if re.fullmatch(r"[,\[\]\{\}\(\)]*", line):
            continue
        if re.fullmatch(r"\d{4}(?:-\d{4})?", line):
            continue
        return False
    return True


def is_test_path(path: str) -> bool:
    lowered = path.lower()
    name = PurePosixPath(path).name.lower()
    return lowered.startswith("tests/") or "/tests/" in lowered or name.startswith("test_") or name.endswith("_test.py")


def build_circuit_review_packet(
    review_input: dict[str, Any],
    chunk: dict[str, Any],
    context_files: dict[str, str],
    base_context_files: dict[str, str],
) -> dict[str, Any]:
    path = str(chunk.get("path") or "")
    previous_path = str(chunk.get("previous_path") or path)
    diff = str(chunk.get("diff") or "")
    hunk_summaries = summarize_diff_hunks(diff)
    head_ranges = [
        (int(item["new_start"]), int(item["new_count"]))
        for item in hunk_summaries
        if int(item.get("new_count") or 0) > 0
    ]
    base_ranges = [
        (int(item["old_start"]), int(item["old_count"]))
        for item in hunk_summaries
        if int(item.get("old_count") or 0) > 0
    ]
    scope_start = int(chunk.get("python_scope_start_line") or 0)
    scope_end = int(chunk.get("python_scope_end_line") or 0)
    if path.endswith(".py") and scope_start > 0 and scope_end >= scope_start:
        scope_range = (scope_start, scope_end - scope_start + 1)
        if chunk.get("status") == "removed":
            if scope_range not in base_ranges:
                base_ranges.insert(0, scope_range)
        elif scope_range not in head_ranges:
            head_ranges.insert(0, scope_range)
    logical_start = int(chunk.get("logical_unit_start_line") or 0)
    logical_end = int(chunk.get("logical_unit_end_line") or 0)
    if logical_start > 0 and logical_end >= logical_start:
        logical_range = (logical_start, logical_end - logical_start + 1)
        if chunk.get("status") == "removed":
            if logical_range not in base_ranges:
                base_ranges.insert(0, logical_range)
        elif logical_range not in head_ranges:
            head_ranges.insert(0, logical_range)
    terms = interesting_terms_from_diff(diff, path)
    primary_context_max_chars = min(
        CIRCUIT_PACKET_PRIMARY_CONTEXT_CHARS,
        max(80, int(chunk.get("context_char_limit") or CIRCUIT_PACKET_PRIMARY_CONTEXT_CHARS)),
    )
    related_context_max_chars = min(
        CIRCUIT_PACKET_RELATED_CONTEXT_CHARS,
        max(80, primary_context_max_chars // 2),
    )
    head_context = focused_context_for_packet(
        context_files,
        primary_path=path,
        primary_ranges=head_ranges,
        terms=terms,
        primary_max_chars=primary_context_max_chars,
        related_max_chars=related_context_max_chars,
    )
    base_context = focused_context_for_packet(
        base_context_files,
        primary_path=previous_path,
        primary_ranges=base_ranges,
        terms=terms,
        primary_max_chars=max(80, min(primary_context_max_chars, CIRCUIT_PACKET_PRIMARY_CONTEXT_CHARS // 2)),
        related_max_chars=max(80, min(related_context_max_chars, CIRCUIT_PACKET_RELATED_CONTEXT_CHARS // 2)),
    )
    semantic_context = build_semantic_context_for_packet(
        context_files,
        primary_path=path,
        chunk=chunk,
        diff=diff,
    )
    return {
        "packet_type": "focused_circuit_pr_review_packet",
        "packet_goal": (
            "Judge this context-aware changed-code packet for concrete SOAR connector issues. "
            "The packet is centered on changed hunks, enclosing logical units, and selected semantic context. "
            "Use local deterministic findings as candidate evidence, and do not report speculative issues."
        ),
        "path": path,
        "previous_path": previous_path if previous_path != path else None,
        "status": chunk.get("status"),
        "change_stats": {
            "additions": chunk.get("additions"),
            "deletions": chunk.get("deletions"),
            "changes": chunk.get("changes"),
            "chunk_index": chunk.get("chunk_index"),
            "chunk_total": chunk.get("chunk_total"),
            "chunk_strategy": chunk.get("chunk_strategy"),
            "context_strategy": chunk.get("context_strategy"),
            "changed_hunk_count": chunk.get("changed_hunk_count"),
            "python_scope": chunk.get("python_scope"),
            "python_scope_kind": chunk.get("python_scope_kind"),
            "python_scope_start_line": chunk.get("python_scope_start_line"),
            "python_scope_end_line": chunk.get("python_scope_end_line"),
            "logical_unit": chunk.get("logical_unit"),
            "logical_unit_kind": chunk.get("logical_unit_kind"),
            "logical_unit_start_line": chunk.get("logical_unit_start_line"),
            "logical_unit_end_line": chunk.get("logical_unit_end_line"),
        },
        "planner": {
            "priority": chunk.get("model_review_priority"),
            "reason": chunk.get("model_review_reason"),
            "diff_chars_before_packet": len(diff),
            "diff_truncated_for_packet": len(diff) > CIRCUIT_PACKET_DIFF_CHARS,
        },
        "diff": truncate_text(diff, CIRCUIT_PACKET_DIFF_CHARS),
        "changed_hunks": hunk_summaries[:24],
        "removed_code_focus": removed_line_excerpts(diff),
        "head_context_files": head_context,
        "base_context_files": base_context,
        "semantic_context": semantic_context,
        "terms_used_for_context": terms[:25],
        "collection_diagnostics": {
            "repo": review_input.get("repo"),
            "pr_number": (review_input.get("pr") or {}).get("number"),
            "collector_notes": review_input.get("collector_notes") or {},
        },
    }


def summarize_diff_hunks(diff: str) -> list[dict[str, Any]]:
    _, hunks = split_unified_diff(diff)
    output: list[dict[str, Any]] = []
    for hunk in hunks:
        lines = hunk.splitlines()
        if not lines:
            continue
        parsed = parse_hunk_header(lines[0])
        if parsed is None:
            continue
        added = []
        removed = []
        for line in lines[1:]:
            if line.startswith("+") and not line.startswith("+++"):
                added.append(line)
            elif line.startswith("-") and not line.startswith("---"):
                removed.append(line)
        output.append(
            {
                **parsed,
                "added_excerpt": truncate_text("\n".join(added), 2_000),
                "removed_excerpt": truncate_text("\n".join(removed), 2_000),
            }
        )
    return output


def parse_hunk_header(header: str) -> dict[str, Any] | None:
    match = re.match(
        r"@@\s+-(?P<old_start>\d+)(?:,(?P<old_count>\d+))?\s+\+(?P<new_start>\d+)(?:,(?P<new_count>\d+))?",
        header,
    )
    if not match:
        return None
    old_count = int(match.group("old_count") or "1")
    new_count = int(match.group("new_count") or "1")
    return {
        "header": header,
        "old_start": int(match.group("old_start")),
        "old_count": old_count,
        "new_start": int(match.group("new_start")),
        "new_count": new_count,
    }


def focused_context_for_packet(
    files: dict[str, str],
    *,
    primary_path: str,
    primary_ranges: list[tuple[int, int]],
    terms: list[str],
    primary_max_chars: int,
    related_max_chars: int,
) -> dict[str, str]:
    output: dict[str, str] = {}
    related_count = 0
    for path, text in files.items():
        if path == primary_path:
            output[path] = line_window_snippets(
                text,
                primary_ranges,
                radius=CONTEXT_AWARE_HUNK_RADIUS,
                max_chars=primary_max_chars,
            )
            continue
        if related_count >= CIRCUIT_PACKET_RELATED_FILE_LIMIT:
            continue
        snippet = keyword_snippets(text, terms, radius=8, max_snippets=5, max_chars=related_max_chars)
        if snippet:
            output[path] = snippet
            related_count += 1
        elif should_keep_context_file_without_term(path) or is_related_view_context_file(path, primary_path):
            output[path] = truncate_text(text, min(related_max_chars, 3_000))
            related_count += 1
    return output


def line_window_snippets(
    text: str,
    ranges: list[tuple[int, int]],
    *,
    radius: int,
    max_chars: int,
) -> str:
    if not text:
        return ""
    lines = text.splitlines()
    if not ranges:
        return truncate_text(text, max_chars)
    windows: list[str] = []
    seen: set[tuple[int, int]] = set()
    for start, count in ranges[:10]:
        if start <= 0:
            start = 1
        end_line = start + max(count, 1) - 1
        window_start = max(1, start - radius)
        window_end = min(len(lines), end_line + radius)
        key = (window_start, window_end)
        if key in seen:
            continue
        seen.add(key)
        numbered = [
            f"{line_no}: {lines[line_no - 1]}"
            for line_no in range(window_start, window_end + 1)
        ]
        windows.append(f"lines {window_start}-{window_end}\n" + "\n".join(numbered))
    return truncate_text("\n\n---\n\n".join(windows), max_chars)


def iter_lines(text: str):
    for index, line in enumerate(text.splitlines(), start=1):
        yield index, line


def keyword_snippets(
    text: str,
    terms: list[str],
    *,
    radius: int,
    max_snippets: int,
    max_chars: int,
) -> str:
    if not text or not terms:
        return ""
    lines = text.splitlines()
    lowered_terms = [term.lower() for term in terms if len(term) >= 4]
    snippets: list[str] = []
    seen: set[tuple[int, int]] = set()
    for index, line in enumerate(lines, start=1):
        lowered = line.lower()
        if not any(term in lowered for term in lowered_terms):
            continue
        start = max(1, index - radius)
        end = min(len(lines), index + radius)
        key = (start, end)
        if key in seen:
            continue
        seen.add(key)
        snippets.append(
            f"lines {start}-{end}\n"
            + "\n".join(f"{line_no}: {lines[line_no - 1]}" for line_no in range(start, end + 1))
        )
        if len(snippets) >= max_snippets:
            break
    return truncate_text("\n\n---\n\n".join(snippets), max_chars)


def build_semantic_context_for_packet(
    files: dict[str, str],
    *,
    primary_path: str,
    chunk: dict[str, Any],
    diff: str,
) -> dict[str, Any]:
    if not primary_path.endswith(".py"):
        return {}
    primary_text = files.get(primary_path, "")
    if not primary_text:
        return {}

    symbols = extract_python_referenced_symbols(diff)
    scope_name = str(chunk.get("python_scope") or "")
    if scope_name and scope_name != "module":
        symbols = [symbol for symbol in symbols if symbol != scope_name.rsplit(".", 1)[-1]]
    symbols = symbols[:PYTHON_SEMANTIC_MAX_SYMBOLS]

    current_start = int(chunk.get("python_scope_start_line") or 0)
    current_end = int(chunk.get("python_scope_end_line") or 0)
    snippets: list[dict[str, Any]] = []
    if symbols:
        snippets.extend(
            python_import_constant_snippets(
                primary_text,
                primary_path,
                symbols,
                max_snippets=4,
            )
        )
        snippets.extend(
            python_symbol_definition_snippets(
                files,
                symbols,
                primary_path=primary_path,
                current_start=current_start,
                current_end=current_end,
                max_snippets=PYTHON_SEMANTIC_MAX_SNIPPETS,
            )
        )

    if scope_name and scope_name != "module":
        snippets.extend(
            python_enclosing_class_snippets(
                primary_text,
                primary_path,
                scope_name,
                max_snippets=max(1, PYTHON_SEMANTIC_MAX_SNIPPETS - len(snippets)),
            )
        )
        function_name = scope_name.rsplit(".", 1)[-1]
        snippets.extend(
            python_caller_snippets_in_files(
                files,
                function_name,
                primary_path=primary_path,
                current_start=current_start,
                current_end=current_end,
                max_snippets=max(2, PYTHON_SEMANTIC_MAX_SNIPPETS - len(snippets)),
            )
        )

    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int | None]] = set()
    total_chars = 0
    for item in snippets:
        key = (str(item.get("path") or ""), str(item.get("symbol") or item.get("kind") or ""), item.get("line"))
        if key in seen:
            continue
        seen.add(key)
        excerpt = truncate_text(str(item.get("excerpt") or ""), 2_000)
        if not excerpt:
            continue
        total_chars += len(excerpt)
        if total_chars > PYTHON_SEMANTIC_CONTEXT_CHARS:
            break
        copied = dict(item)
        copied["excerpt"] = excerpt
        deduped.append(copied)

    if not symbols and not deduped:
        return {}
    return {
        "strategy": "static_symbol_expansion_from_changed_code",
        "referenced_symbols": symbols,
        "snippet_count": len(deduped),
        "snippets": deduped,
    }


def extract_python_referenced_symbols(diff: str) -> list[str]:
    lines: list[str] = []
    for raw_line in diff.splitlines():
        if raw_line.startswith(("+++", "---", "@@")):
            continue
        if raw_line.startswith(("+", "-", " ")):
            lines.append(raw_line[1:])
        else:
            lines.append(raw_line)
    text = "\n".join(lines)
    ignored = {
        "and",
        "assert",
        "class",
        "def",
        "elif",
        "else",
        "except",
        "false",
        "finally",
        "for",
        "from",
        "if",
        "import",
        "in",
        "none",
        "not",
        "or",
        "return",
        "self",
        "true",
        "with",
        "yield",
        "dict",
        "append",
        "decode",
        "encode",
        "endswith",
        "extend",
        "format",
        "get",
        "items",
        "join",
        "keys",
        "lower",
        "int",
        "len",
        "list",
        "max",
        "min",
        "open",
        "range",
        "read",
        "replace",
        "set",
        "split",
        "startswith",
        "str",
        "strip",
        "sum",
        "update",
        "values",
    }
    candidates: list[str] = []
    candidates.extend(re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", text))
    candidates.extend(re.findall(r"\.([A-Za-z_][A-Za-z0-9_]*)\s*\(", text))
    candidates.extend(re.findall(r"\bself\.([A-Za-z_][A-Za-z0-9_]*)\b", text))
    candidates.extend(re.findall(r"\b([A-Z][A-Z0-9_]{3,})\b", text))

    output: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized = candidate.strip()
        lowered = normalized.lower()
        if len(normalized) < 3 or lowered in ignored or normalized.startswith("__"):
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        output.append(normalized)
    return output


def python_import_constant_snippets(
    text: str,
    path: str,
    symbols: list[str],
    *,
    max_snippets: int,
) -> list[dict[str, Any]]:
    if not symbols:
        return []
    symbol_pattern = "|".join(re.escape(symbol) for symbol in symbols)
    line_pattern = re.compile(
        rf"^\s*(?:from\s+[\w.]+\s+import\s+.*\b(?:{symbol_pattern})\b|import\s+.*\b(?:{symbol_pattern})\b|(?:{symbol_pattern})\s*=)",
        re.M,
    )
    output: list[dict[str, Any]] = []
    for match in line_pattern.finditer(text):
        line_no = text[: match.start()].count("\n") + 1
        output.append(
            {
                "path": path,
                "kind": "import_or_constant",
                "symbol": matched_symbol(match.group(0), symbols),
                "line": line_no,
                "reason": "import or constant referenced by changed code",
                "excerpt": line_window_snippets(text, [(line_no, 1)], radius=2, max_chars=800),
            }
        )
        if len(output) >= max_snippets:
            break
    return output


def matched_symbol(text: str, symbols: list[str]) -> str:
    lowered = text.lower()
    for symbol in symbols:
        if symbol.lower() in lowered:
            return symbol
    return symbols[0] if symbols else ""


def python_symbol_definition_snippets(
    files: dict[str, str],
    symbols: list[str],
    *,
    primary_path: str,
    current_start: int,
    current_end: int,
    max_snippets: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    symbol_set = {symbol.lower() for symbol in symbols}
    ordered_files = sorted(files.items(), key=lambda item: 0 if item[0] == primary_path else 1)
    for path, text in ordered_files:
        if not path.endswith(".py") or not text:
            continue
        for scope in python_scopes_from_source(text):
            name = str(scope.get("name") or "")
            short_name = name.rsplit(".", 1)[-1].lower()
            if short_name not in symbol_set:
                continue
            start = int(scope.get("start_line") or 0)
            end = int(scope.get("end_line") or start)
            if path == primary_path and current_start and current_end and current_start <= start <= current_end:
                continue
            output.append(
                {
                    "path": path,
                    "kind": str(scope.get("kind") or "function"),
                    "symbol": name,
                    "line": start,
                    "reason": "definition referenced by changed code",
                    "excerpt": line_window_snippets(text, [(start, end - start + 1)], radius=4, max_chars=2_000),
                }
            )
            if len(output) >= max_snippets:
                return output
    return output


def python_enclosing_class_snippets(
    text: str,
    path: str,
    scope_name: str,
    *,
    max_snippets: int,
) -> list[dict[str, Any]]:
    if "." not in scope_name or max_snippets <= 0:
        return []
    class_name = scope_name.rsplit(".", 1)[0]
    scopes = python_scopes_from_source(text)
    class_scope = next(
        (scope for scope in scopes if scope.get("kind") == "class" and scope.get("name") == class_name),
        None,
    )
    if not class_scope:
        return []

    output: list[dict[str, Any]] = []
    class_start = int(class_scope.get("start_line") or 1)
    output.append(
        {
            "path": path,
            "kind": "enclosing_class",
            "symbol": class_name,
            "line": class_start,
            "reason": "enclosing class for changed method",
            "excerpt": line_window_snippets(text, [(class_start, 1)], radius=8, max_chars=1_200),
        }
    )
    if len(output) >= max_snippets:
        return output

    init_name = f"{class_name}.__init__"
    init_scope = next(
        (scope for scope in scopes if scope.get("kind") == "function" and scope.get("name") == init_name),
        None,
    )
    if init_scope:
        start = int(init_scope.get("start_line") or class_start)
        end = int(init_scope.get("end_line") or start)
        output.append(
            {
                "path": path,
                "kind": "enclosing_class_initializer",
                "symbol": init_name,
                "line": start,
                "reason": "initializer for changed method's class",
                "excerpt": line_window_snippets(text, [(start, end - start + 1)], radius=3, max_chars=1_800),
            }
        )
    return output[:max_snippets]


def python_caller_snippets_in_files(
    files: dict[str, str],
    function_name: str,
    *,
    primary_path: str,
    current_start: int,
    current_end: int,
    max_snippets: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    ordered_files = sorted(files.items(), key=lambda item: 0 if item[0] == primary_path else 1)
    for path, text in ordered_files:
        if not path.endswith(".py") or not text:
            continue
        remaining = max_snippets - len(output)
        if remaining <= 0:
            break
        output.extend(
            python_caller_snippets(
                text,
                path,
                function_name,
                current_start=current_start if path == primary_path else 0,
                current_end=current_end if path == primary_path else 0,
                max_snippets=remaining,
            )
        )
    return output[:max_snippets]


def python_caller_snippets(
    text: str,
    path: str,
    function_name: str,
    *,
    current_start: int,
    current_end: int,
    max_snippets: int,
) -> list[dict[str, Any]]:
    if not function_name or not text or max_snippets <= 0:
        return []
    scopes = python_scopes_from_source(text)
    call_pattern = re.compile(rf"\b(?:self\.)?{re.escape(function_name)}\s*\(")
    output: list[dict[str, Any]] = []
    seen_scope_names: set[str] = set()
    for line_no, line in iter_lines(text):
        if not call_pattern.search(line) or re.match(r"\s*def\s+", line):
            continue
        if current_start and current_end and current_start <= line_no <= current_end:
            continue
        scope = find_python_scope_for_line(scopes, line_no)
        if not scope:
            continue
        scope_name = str(scope.get("name") or "")
        if scope_name in seen_scope_names:
            continue
        seen_scope_names.add(scope_name)
        start = int(scope.get("start_line") or line_no)
        end = int(scope.get("end_line") or line_no)
        output.append(
            {
                "path": path,
                "kind": "caller",
                "symbol": scope_name,
                "line": start,
                "reason": f"caller of changed function `{function_name}`",
                "excerpt": line_window_snippets(text, [(start, end - start + 1)], radius=4, max_chars=2_000),
            }
        )
        if len(output) >= max_snippets:
            break
    return output


def should_keep_context_file_without_term(path: str) -> bool:
    name = PurePosixPath(path).name.lower()
    if name in {"readme.md", "uv.lock", "license", "notice"}:
        return False
    return (
        name in {"manual_readme_content.md", "pyproject.toml"}
        or path.startswith("release_notes/")
        or ("/" not in path and path.endswith(".json"))
    )


def interesting_terms_from_diff(diff: str, path: str) -> list[str]:
    ignored = {
        "self",
        "none",
        "true",
        "false",
        "return",
        "import",
        "from",
        "class",
        "def",
        "with",
        "data",
        "result",
        "response",
        "action",
    }
    raw_terms = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]{3,}\b", diff)
    raw_terms.extend(re.findall(r"`([^`]{4,80})`", diff))
    raw_terms.extend(PurePosixPath(path).stem.split("_"))
    output: list[str] = []
    seen: set[str] = set()
    for term in raw_terms:
        normalized = str(term).strip().lower()
        if len(normalized) < 4 or normalized in ignored or normalized in seen:
            continue
        seen.add(normalized)
        output.append(normalized)
        if len(output) >= 40:
            break
    return output


def removed_line_excerpts(diff: str, *, max_chars: int = 4_000) -> str:
    removed = []
    for raw_line in diff.splitlines():
        if raw_line.startswith("---") or raw_line.startswith("@@"):
            continue
        if raw_line.startswith("-"):
            removed.append(raw_line)
    return truncate_text("\n".join(removed), max_chars)


def select_context_files(
    files: dict[str, str],
    primary_path: str,
    *,
    max_chars: int = 80_000,
    include_all_source: bool = False,
) -> dict[str, str]:
    output: dict[str, str] = {}
    for path, text in files.items():
        name = PurePosixPath(path).name
        if (
            path == primary_path
            or name in DEEP_CONTEXT_FILES
            or ("/" not in path and path.endswith(".json"))
            or is_related_view_context_file(path, primary_path)
            or (include_all_source and is_semantic_context_source_file(path))
        ):
            output[path] = truncate_text(text, max_chars)
    return output


def is_semantic_context_source_file(path: str) -> bool:
    suffix = PurePosixPath(path).suffix.lower()
    name = PurePosixPath(path).name.lower()
    if name in {"license", "notice", "readme.md", "uv.lock"}:
        return False
    return suffix in {".py", ".json", ".toml", ".yml", ".yaml", ".xml", ".html", ".jinja", ".j2"}


def select_chunk_comments(comments: dict[str, Any], path: str) -> dict[str, list[dict[str, Any]]]:
    if not isinstance(comments, dict):
        return {"issue_comments": [], "review_comments": [], "reviews": []}
    basename = PurePosixPath(path).name
    output: dict[str, list[dict[str, Any]]] = {}
    for key, limit in (("review_comments", 40), ("issue_comments", 12), ("reviews", 12)):
        items = comments.get(key) or []
        if not isinstance(items, list):
            output[key] = []
            continue
        selected = []
        for item in items:
            if not isinstance(item, dict):
                continue
            if comment_matches_path(item, path, basename) or key == "reviews":
                compact = dict(item)
                if "body" in compact:
                    compact["body"] = truncate_text(str(compact.get("body") or ""), 1_200)
                selected.append(compact)
            if len(selected) >= limit:
                break
        output[key] = selected
    return output


def comment_matches_path(item: dict[str, Any], path: str, basename: str) -> bool:
    candidate_paths = [
        str(item.get(key) or "")
        for key in ("path", "file", "filename", "original_path")
    ]
    if any(candidate == path for candidate in candidate_paths):
        return True
    body = str(item.get("body") or "")
    return path in body or (basename and basename in body)


def select_chunk_ci(ci: dict[str, Any], path: str) -> dict[str, Any]:
    if not isinstance(ci, dict):
        return {}
    basename = PurePosixPath(path).name
    output: dict[str, Any] = {}
    for key in ("errors", "statuses"):
        value = ci.get(key)
        if isinstance(value, list):
            output[key] = value[:20]
    check_runs = ci.get("check_runs") or []
    if isinstance(check_runs, list):
        output["check_runs"] = [
            compact_check_run(item)
            for item in check_runs[:80]
            if isinstance(item, dict)
        ]
    failed_logs = ci.get("failed_check_logs") or []
    if isinstance(failed_logs, list):
        selected_logs = []
        for item in failed_logs:
            if not isinstance(item, dict):
                continue
            text = " ".join(str(item.get(key) or "") for key in ("name", "path", "body", "log", "text", "summary"))
            if path in text or (basename and basename in text):
                compact = dict(item)
                for body_key in ("body", "log", "text", "summary"):
                    if body_key in compact:
                        compact[body_key] = truncate_text(str(compact.get(body_key) or ""), 1_500)
                selected_logs.append(compact)
            if len(selected_logs) >= 8:
                break
        output["failed_check_logs"] = selected_logs
    return output


def compact_check_run(item: dict[str, Any]) -> dict[str, Any]:
    compact = {
        key: item.get(key)
        for key in ("name", "status", "conclusion", "html_url", "details_url", "started_at", "completed_at")
        if key in item
    }
    output = item.get("output")
    if isinstance(output, dict):
        compact["output"] = {
            "title": output.get("title"),
            "summary": truncate_text(str(output.get("summary") or ""), 800),
            "text": truncate_text(str(output.get("text") or ""), 1_200),
        }
    return compact


def split_chunk_for_adaptive_retry(
    chunk: dict[str, Any],
    *,
    max_chars: int = 18_000,
    context_char_limit: int = 30_000,
) -> list[dict[str, Any]]:
    diff = str(chunk.get("diff") or "")
    if not diff or len(diff) <= max_chars:
        return []
    headers, hunks = split_unified_diff(diff)
    pieces: list[str] = []
    for hunk in hunks or [diff]:
        if len(f"{headers}\n{hunk}".strip()) <= max_chars:
            pieces.append(f"{headers}\n{hunk}".strip() if headers else hunk)
        else:
            pieces.extend(split_large_hunk(headers, hunk, max_chars=max_chars))
    pieces = [piece for piece in pieces if piece.strip()]
    if len(pieces) <= 1:
        return []

    parent_id = str(chunk.get("id") or chunk.get("path") or "chunk")
    output: list[dict[str, Any]] = []
    total = len(pieces)
    for index, piece in enumerate(pieces, start=1):
        child = dict(chunk)
        child.update(
            {
                "id": f"{parent_id}:retry-{index}",
                "parent_chunk_id": parent_id,
                "adaptive_retry": True,
                "chunk_index": index,
                "chunk_total": total,
                "diff": piece,
                "diff_chars": len(piece),
                "context_char_limit": context_char_limit,
            }
        )
        output.append(child)
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
