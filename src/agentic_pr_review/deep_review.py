"""Deep PR collection and chunked review helpers."""

from __future__ import annotations

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
    context_char_limit = int(chunk.get("context_char_limit") or 14_000)
    if path.endswith(".py") and not chunk.get("adaptive_retry"):
        context_char_limit = min(context_char_limit, 8_000)
    context_files = select_context_files(full_files, path, max_chars=context_char_limit)
    base_context_files = select_context_files(base_files, previous_path, max_chars=context_char_limit)
    review_packet = build_circuit_review_packet(
        review_input,
        chunk,
        context_files,
        base_context_files,
    )
    instruction = "Only report concrete issues proven by this chunk plus supplied PR context."
    if chunk.get("adaptive_retry"):
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
            "path": path,
            "previous_path": chunk.get("previous_path"),
            "chunk_index": chunk.get("chunk_index"),
            "chunk_total": chunk.get("chunk_total"),
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
    terms = interesting_terms_from_diff(diff, path)
    head_context = focused_context_for_packet(
        context_files,
        primary_path=path,
        primary_ranges=head_ranges,
        terms=terms,
        primary_max_chars=CIRCUIT_PACKET_PRIMARY_CONTEXT_CHARS,
        related_max_chars=CIRCUIT_PACKET_RELATED_CONTEXT_CHARS,
    )
    base_context = focused_context_for_packet(
        base_context_files,
        primary_path=previous_path,
        primary_ranges=base_ranges,
        terms=terms,
        primary_max_chars=max(4_000, CIRCUIT_PACKET_PRIMARY_CONTEXT_CHARS // 2),
        related_max_chars=max(2_500, CIRCUIT_PACKET_RELATED_CONTEXT_CHARS // 2),
    )
    return {
        "packet_type": "focused_circuit_pr_review_packet",
        "packet_goal": (
            "Judge this focused changed-file packet for concrete SOAR connector issues. "
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
    for path, text in files.items():
        if path == primary_path:
            output[path] = line_window_snippets(text, primary_ranges, radius=35, max_chars=primary_max_chars)
            continue
        snippet = keyword_snippets(text, terms, radius=8, max_snippets=5, max_chars=related_max_chars)
        if snippet:
            output[path] = snippet
        elif should_keep_context_file_without_term(path) or is_related_view_context_file(path, primary_path):
            output[path] = truncate_text(text, min(related_max_chars, 3_000))
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


def should_keep_context_file_without_term(path: str) -> bool:
    name = PurePosixPath(path).name
    return name in DEEP_CONTEXT_FILES or ("/" not in path and path.endswith(".json"))


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


def select_context_files(files: dict[str, str], primary_path: str, *, max_chars: int = 80_000) -> dict[str, str]:
    output: dict[str, str] = {}
    for path, text in files.items():
        name = PurePosixPath(path).name
        if (
            path == primary_path
            or name in DEEP_CONTEXT_FILES
            or ("/" not in path and path.endswith(".json"))
            or is_related_view_context_file(path, primary_path)
        ):
            output[path] = truncate_text(text, max_chars)
    return output


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
