"""Runtime retrieval of historically human-reviewed PR examples."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any

from .json_utils import truncate_text


DEFAULT_EXAMPLE_GLOBS = (
    "runs/training-mining-full/*/training_examples.jsonl",
    "runs/training-mining-pipeline-validation/*/training_examples.jsonl",
    "runs/training-mining-known/*/training_examples.jsonl",
    "runs/training-mining/*/training_examples.jsonl",
)

TOKEN_RE = re.compile(r"[a-z][a-z0-9_]{2,}|[a-z0-9_.-]+\.(?:py|json|ya?ml|toml|md)")
STOPWORDS = {
    "about",
    "action",
    "actions",
    "added",
    "again",
    "already",
    "also",
    "all",
    "and",
    "any",
    "app",
    "apps",
    "are",
    "asset",
    "aggregate",
    "aws",
    "because",
    "been",
    "being",
    "body",
    "but",
    "can",
    "characters",
    "change",
    "changed",
    "check",
    "cicd",
    "code",
    "comment",
    "com",
    "connector",
    "could",
    "class",
    "cloud_next",
    "default",
    "def",
    "description",
    "diff",
    "does",
    "download",
    "dict",
    "else",
    "error",
    "expected",
    "failure",
    "file",
    "files",
    "fix",
    "false",
    "found",
    "for",
    "from",
    "get",
    "github",
    "have",
    "had",
    "has",
    "here",
    "into",
    "import",
    "int",
    "its",
    "issue",
    "json",
    "line",
    "list",
    "license",
    "need",
    "needs",
    "none",
    "not",
    "our",
    "please",
    "pull",
    "return",
    "review",
    "self",
    "set",
    "should",
    "sanity",
    "str",
    "src",
    "that",
    "the",
    "their",
    "them",
    "there",
    "they",
    "this",
    "test",
    "tests",
    "true",
    "was",
    "were",
    "will",
    "with",
    "would",
    "you",
    "your",
}

BOT_COMMENT_MARKERS = (
    "<!-- agentic-pr-review",
    "agentic-pr-review:",
    "agentic-pr-review/finding",
)

LOW_VALUE_COMMENT_PHRASES = (
    "combine using walrus",
    "switch to walrus",
    "friendlier",
    "nit:",
    "minor:",
    "style:",
    "formatting",
    "can simplify",
    "could simplify",
    "prefer this syntax",
    "do we need the pipeline",
    "please advise where tests",
    "other prs do not have tests",
    "passes successfully",
    "passing successfully",
    "can process with release",
    "i'll approve",
    " done",
    "manual runs",
    "manual run",
    "pipeline links",
    "waiting for tests to re-run",
    "not sure how to get test coverage",
)

CI_RELEVANT_CATEGORIES = {
    "precommit",
    "static_sanity_tests",
    "missing_tests",
    "dependency_packaging",
    "docs_pr_accuracy",
}

CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "api_auth_correctness": (
        "oauth",
        "token",
        "client_credentials",
        "client_id",
        "client_secret",
        "basic",
        "authorization",
        "timeout",
        "test_connectivity",
        "verify",
        "tls",
    ),
    "api_contract": (
        "api",
        "endpoint",
        "content-type",
        "form-urlencoded",
        "json_data",
        "required",
        "vendor",
        "documentation",
        "response",
    ),
    "polling_checkpoint": (
        "on_poll",
        "poll",
        "poll_now",
        "checkpoint",
        "save_state",
        "source_data_identifier",
        "dedup",
        "timestamp",
        "container",
        "artifact",
    ),
    "output_schema_mismatch": (
        "add_data",
        "output",
        "data_path",
        "summary",
        "summary_type",
        "readme",
        "schema",
        "action_result",
    ),
    "unsafe_logging": (
        "debug_print",
        "log",
        "logger",
        "token",
        "secret",
        "authorization",
        "headers",
        "password",
        "payload",
        "response",
        "tenant",
        "pii",
    ),
    "soar_metadata": (
        "read_only",
        "python_version",
        "app_version",
        "appid",
        "package_name",
        "product_version",
        "latest_tested_versions",
        "metadata",
    ),
    "pagination": (
        "pagination",
        "next_page",
        "nextpagetoken",
        "offset",
        "limit",
        "cursor",
        "page",
        "truncat",
    ),
    "validation": (
        "validate",
        "validation",
        "blank",
        "empty",
        "allow_list",
        "cross-field",
        "quote",
        "urlencode",
        "unencoded",
    ),
    "missing_tests": (
        "test",
        "tests",
        "coverage",
        "playbook",
        "mock",
        "unit",
        "integration",
        "untested",
    ),
    "docs_pr_accuracy": (
        "manual_readme_content",
        "release_notes",
        "release",
        "readme",
        "docs",
        "documentation",
        "default",
        "claim",
    ),
    "precommit": (
        "pre-commit",
        "precommit",
        "ruff",
        "semgrep",
        "detect-secrets",
        "build-docs",
        "release-notes",
        "check-json",
        "check-yaml",
        "mdformat",
        "valid_app_name_and_guid",
        "app_package_name",
    ),
    "merge_conflict": (
        "conflict",
        "mergeable",
        "blocked",
        "dirty",
        "merge",
    ),
    "error_handling": (
        "try",
        "except",
        "exception",
        "actionfailure",
        "cleanup",
        "set_status",
        "b64decode",
    ),
    "dependency_packaging": (
        "requirements",
        "dependency",
        "dependencies",
        "license",
        "notice",
        "wheel",
        "package",
        "pypi",
    ),
    "app_mapping": (
        "appid_to_name",
        "appid_to_package_name",
        "app_mapping",
        "valid_app_name_and_guid",
        "guid",
    ),
    "static_sanity_tests": (
        "min_phantom_version",
        "minimum",
        "platform",
        "logging",
        "verbosity",
        "product-name-on-files",
        "playbook",
    ),
}


def enrich_with_historical_context(
    review_input: dict[str, Any],
    *,
    examples_path: str | None = None,
    max_examples: int = 8,
    min_score: int = 8,
) -> dict[str, Any]:
    """Attach top historical examples to review_input for model review."""

    notes = review_input.setdefault("collector_notes", {})
    path = resolve_historical_examples_path(examples_path)
    if path is None:
        review_input["historical_context"] = {
            "enabled": False,
            "reason": "No training_examples.jsonl file found.",
            "matches": [],
        }
        notes["historical_context_enabled"] = False
        notes["historical_context_match_count"] = 0
        return review_input

    examples, errors = load_training_examples(path)
    matches = select_historical_examples(
        review_input,
        examples,
        max_examples=max_examples,
        min_score=min_score,
    )
    category_counts = Counter(str(item.get("category") or "general") for item in matches)
    review_input["historical_context"] = {
        "enabled": True,
        "source_path": str(path),
        "loaded_example_count": len(examples),
        "matched_example_count": len(matches),
        "min_score": min_score,
        "usage": (
            "Use these as human-review precedents only. Do not report an issue "
            "unless current PR code, docs, comments, or CI evidence independently proves it."
        ),
        "category_counts": dict(category_counts),
        "matches": matches,
        "errors": errors[:10],
    }
    notes["historical_context_enabled"] = True
    notes["historical_context_source_path"] = str(path)
    notes["historical_context_loaded_example_count"] = len(examples)
    notes["historical_context_match_count"] = len(matches)
    notes["historical_context_categories"] = dict(category_counts)
    if errors:
        notes["historical_context_error_count"] = len(errors)
        notes["historical_context_errors"] = errors[:5]
    return review_input


def resolve_historical_examples_path(examples_path: str | None) -> Path | None:
    if examples_path:
        path = Path(examples_path)
        if not path.is_file():
            raise ValueError(f"Historical examples file does not exist: {examples_path}")
        return path

    roots = [Path.cwd(), Path(__file__).resolve().parents[2]]
    candidates: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        for pattern in DEFAULT_EXAMPLE_GLOBS:
            for candidate in root.glob(pattern):
                resolved = candidate.resolve()
                if resolved in seen or not candidate.is_file():
                    continue
                seen.add(resolved)
                candidates.append(candidate)
    if not candidates:
        return None
    return max(candidates, key=lambda item: item.stat().st_mtime)


def load_training_examples(path: Path, *, max_examples: int = 10_000) -> tuple[list[dict[str, Any]], list[str]]:
    examples: list[dict[str, Any]] = []
    errors: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"{path}:{line_number}: {exc}")
                continue
            if isinstance(value, dict):
                examples.append(value)
            if len(examples) >= max_examples:
                break
    return examples, errors


def select_historical_examples(
    review_input: dict[str, Any],
    examples: list[dict[str, Any]],
    *,
    max_examples: int,
    min_score: int,
) -> list[dict[str, Any]]:
    if max_examples <= 0:
        return []

    features = build_current_features(review_input)
    scored: list[tuple[int, dict[str, Any]]] = []
    seen: set[tuple[str, str, str | None, str]] = set()
    for example in examples:
        if not is_usable_historical_example(example):
            continue
        if is_same_pr(example, review_input):
            continue
        score, reasons = score_example(example, features)
        if score < min_score:
            continue
        compacted = compact_historical_example(example, score=score, reasons=reasons)
        key = (
            str(compacted.get("pattern_id") or ""),
            str(compacted.get("repo") or ""),
            compacted.get("path"),
            str(compacted.get("human_comment") or "")[:160],
        )
        if key in seen:
            continue
        seen.add(key)
        scored.append((score, compacted))

    scored.sort(key=lambda item: (-item[0], str(item[1].get("pattern_id") or ""), str(item[1].get("repo") or "")))
    return diversify_historical_matches(scored, max_examples=max_examples)


def diversify_historical_matches(scored: list[tuple[int, dict[str, Any]]], *, max_examples: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    pattern_counts: Counter[str] = Counter()
    seen_pr_patterns: set[tuple[str, str, str]] = set()
    for _, item in scored:
        pattern_id = str(item.get("pattern_id") or "")
        repo = str(item.get("repo") or "")
        pr_number = str(item.get("pr_number") or "")
        pr_pattern = (pattern_id, repo, pr_number)
        if pr_pattern in seen_pr_patterns:
            continue
        if pattern_counts[pattern_id] >= 3:
            continue
        seen_pr_patterns.add(pr_pattern)
        pattern_counts[pattern_id] += 1
        selected.append(item)
        if len(selected) >= max_examples:
            break
    return selected


def build_current_features(review_input: dict[str, Any]) -> dict[str, Any]:
    current_text = current_review_text(review_input)
    paths = current_paths(review_input)
    terms = tokenize(current_text)
    categories = infer_categories(current_text)
    ci_terms = tokenize(ci_text(review_input))
    return {
        "repo": str(review_input.get("repo") or ""),
        "pr_number": int((review_input.get("pr") or {}).get("number") or 0),
        "text": current_text.lower(),
        "terms": terms,
        "paths": paths,
        "basenames": {PurePosixPath(path).name.lower() for path in paths},
        "extensions": {PurePosixPath(path).suffix.lower() for path in paths if PurePosixPath(path).suffix},
        "categories": categories,
        "ci_terms": ci_terms,
    }


def current_review_text(review_input: dict[str, Any]) -> str:
    pieces: list[str] = []
    pr = review_input.get("pr") or {}
    pieces.extend([str(pr.get("title") or ""), str(pr.get("body") or ""), str(review_input.get("repo") or "")])
    for item in review_input.get("changed_files", []) or []:
        if not isinstance(item, dict):
            continue
        pieces.append(str(item.get("filename") or ""))
        pieces.append(str(item.get("status") or ""))
        pieces.append(str(item.get("patch") or ""))
    deep_review = review_input.get("deep_review") or {}
    for chunk in deep_review.get("chunks", []) or []:
        if isinstance(chunk, dict):
            pieces.append(str(chunk.get("path") or ""))
            pieces.append(str(chunk.get("diff") or ""))
    for group in (review_input.get("comments") or {}).values():
        if isinstance(group, list):
            for comment in group[:120]:
                if isinstance(comment, dict):
                    pieces.append(str(comment.get("body") or ""))
                    pieces.append(str(comment.get("path") or ""))
    pieces.append(ci_text(review_input))
    return "\n".join(pieces)


def ci_text(review_input: dict[str, Any]) -> str:
    pieces: list[str] = []
    ci = review_input.get("ci") or {}
    for key in ("check_runs", "statuses", "failed_check_logs", "errors"):
        values = ci.get(key) or []
        if not isinstance(values, list):
            continue
        for item in values[:120]:
            if isinstance(item, dict):
                pieces.extend(str(item.get(field) or "") for field in ("name", "context", "conclusion", "state", "summary", "log_excerpt", "description"))
            else:
                pieces.append(str(item))
    return "\n".join(pieces)


def current_paths(review_input: dict[str, Any]) -> set[str]:
    paths = {
        str(item.get("filename") or "")
        for item in review_input.get("changed_files", []) or []
        if isinstance(item, dict) and item.get("filename")
    }
    deep_review = review_input.get("deep_review") or {}
    for chunk in deep_review.get("chunks", []) or []:
        if isinstance(chunk, dict) and chunk.get("path"):
            paths.add(str(chunk["path"]))
    return paths


def tokenize(text: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    for match in TOKEN_RE.finditer(text.lower()):
        token = match.group(0).strip("_-.")
        if len(token) < 3 or token in STOPWORDS:
            continue
        counts[token] += 1
    return counts


def infer_categories(text: str) -> set[str]:
    lowered = text.lower()
    categories = set()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            categories.add(category)
    return categories


def is_same_pr(example: dict[str, Any], review_input: dict[str, Any]) -> bool:
    repo = str(review_input.get("repo") or "")
    pr_number = int((review_input.get("pr") or {}).get("number") or 0)
    try:
        example_number = int(example.get("pr_number") or 0)
    except (TypeError, ValueError):
        example_number = 0
    return bool(repo and pr_number and str(example.get("repo") or "") == repo and example_number == pr_number)


def is_usable_historical_example(example: dict[str, Any]) -> bool:
    raw_body = str(example.get("body") or example.get("human_comment") or "")
    lowered_raw = raw_body.lower()
    if any(marker in lowered_raw for marker in BOT_COMMENT_MARKERS):
        return False
    if raw_body.lstrip().startswith(">"):
        return False
    body = clean_historical_comment_body(raw_body).lower()
    if any(phrase in body for phrase in LOW_VALUE_COMMENT_PHRASES):
        return False
    user = str(example.get("user") or "").lower()
    if user.endswith("[bot]") or "github-actions" in user or "pre-commit-ci" in user:
        return False
    return bool(str(example.get("pattern_id") or "").strip() and body.strip())


def clean_historical_comment_body(body: str) -> str:
    lines = []
    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line or line.startswith(">"):
            continue
        lines.append(line)
    text = " ".join(lines) if lines else body
    text = re.sub(r"<!--\s*agentic-pr-review:[^>]*-->", "", text, flags=re.IGNORECASE)
    return " ".join(text.split())


def score_example(example: dict[str, Any], features: dict[str, Any]) -> tuple[int, list[str]]:
    score = 0
    reasons: list[str] = []
    category = str(example.get("category") or "")
    example_body = example_text(example)
    if category and category in features["categories"]:
        score += 10
        reasons.append(f"category:{category}")

    if str(example.get("repo") or "") == features["repo"]:
        score += 6
        reasons.append("same_repo")

    example_paths = example_file_paths(example)
    exact_paths = example_paths & features["paths"]
    if exact_paths:
        score += min(18, 9 * len(exact_paths))
        reasons.append("same_path:" + ",".join(sorted(exact_paths)[:3]))

    example_basenames = {PurePosixPath(path).name.lower() for path in example_paths}
    basename_matches = example_basenames & features["basenames"]
    if basename_matches:
        score += min(10, 5 * len(basename_matches))
        reasons.append("same_filename:" + ",".join(sorted(basename_matches)[:3]))

    kind_matches = {
        kind
        for kind in {path_kind(path) for path in example_paths}
        if kind and kind in {path_kind(path) for path in features["paths"]}
    }
    if kind_matches and not basename_matches:
        score += min(8, 4 * len(kind_matches))
        reasons.append("same_file_kind:" + ",".join(sorted(kind_matches)[:3]))

    example_extensions = {PurePosixPath(path).suffix.lower() for path in example_paths if PurePosixPath(path).suffix}
    extension_matches = example_extensions & features["extensions"]
    if extension_matches:
        score += min(4, 2 * len(extension_matches))
        reasons.append("same_file_type:" + ",".join(sorted(extension_matches)[:3]))

    example_terms = tokenize(example_body)
    overlap = set(example_terms) & set(features["terms"])
    weighted_overlap = sum(min(example_terms[term], features["terms"][term], 3) for term in overlap)
    if weighted_overlap:
        score += min(18, weighted_overlap)
        top_terms = sorted(overlap, key=lambda term: -(example_terms[term] + features["terms"][term]))[:6]
        reasons.append("shared_terms:" + ",".join(top_terms))

    ci_overlap = set()
    if category in CI_RELEVANT_CATEGORIES:
        ci_overlap = set(example_terms) & set(features["ci_terms"])
    if ci_overlap:
        score += min(8, 2 * len(ci_overlap))
        reasons.append("ci_terms:" + ",".join(sorted(ci_overlap)[:5]))

    category_term_matches = []
    for keyword in CATEGORY_KEYWORDS.get(category, ()):
        if keyword in features["text"] and keyword in example_body.lower():
            category_term_matches.append(keyword)
    if category_term_matches:
        score += min(12, 3 * len(category_term_matches))
        reasons.append("category_terms:" + ",".join(category_term_matches[:5]))
    return score, reasons[:8]


def example_file_paths(example: dict[str, Any]) -> set[str]:
    paths = {str(example.get("path") or "")}
    for path in example.get("changed_files") or []:
        if path:
            paths.add(str(path))
    for item in example.get("file_evidence") or []:
        if isinstance(item, dict) and item.get("filename"):
            paths.add(str(item["filename"]))
    return {path for path in paths if path}


def path_kind(path: str) -> str | None:
    lowered = str(path or "").lower()
    if not lowered:
        return None
    basename = PurePosixPath(lowered).name
    suffix = PurePosixPath(lowered).suffix
    if basename == "connector.py" or basename.endswith("_connector.py"):
        return "connector.py"
    if basename in {"app.json", "readme.md", "manual_readme_content.md", "requirements.txt", "pyproject.toml"}:
        return basename
    if "release_notes" in lowered or basename == "unreleased.md":
        return "release_notes/unreleased.md"
    if "test" in lowered and suffix == ".py":
        return "tests/test_connector.py"
    if suffix == ".json" and ("app" in basename or "connector" in lowered):
        return "app.json"
    return None


def example_text(example: dict[str, Any]) -> str:
    pieces = [
        str(example.get("pattern_id") or ""),
        str(example.get("pattern_title") or ""),
        str(example.get("category") or ""),
        str(example.get("implementation_hint") or ""),
        str(example.get("pr_title") or ""),
        clean_historical_comment_body(str(example.get("body") or "")),
        str(example.get("diff_hunk") or ""),
        str(example.get("path") or ""),
    ]
    for item in example.get("file_evidence") or []:
        if isinstance(item, dict):
            pieces.extend([str(item.get("filename") or ""), str(item.get("patch_excerpt") or "")])
    for item in example.get("check_evidence") or []:
        if isinstance(item, dict):
            pieces.extend([str(item.get("name") or ""), str(item.get("summary") or ""), str(item.get("conclusion") or "")])
    return "\n".join(pieces)


def compact_historical_example(example: dict[str, Any], *, score: int, reasons: list[str]) -> dict[str, Any]:
    file_evidence = []
    for item in (example.get("file_evidence") or [])[:2]:
        if not isinstance(item, dict):
            continue
        file_evidence.append(
            {
                "filename": item.get("filename"),
                "status": item.get("status"),
                "patch_excerpt": truncate_text(str(item.get("patch_excerpt") or ""), 700),
            }
        )

    check_evidence = []
    for item in (example.get("check_evidence") or [])[:3]:
        if not isinstance(item, dict):
            continue
        check_evidence.append(
            {
                "kind": item.get("kind"),
                "name": item.get("name"),
                "conclusion": item.get("conclusion"),
                "summary": truncate_text(str(item.get("summary") or item.get("error") or ""), 500),
            }
        )

    return {
        "score": score,
        "why_selected": reasons,
        "pattern_id": example.get("pattern_id"),
        "category": example.get("category"),
        "pattern_title": example.get("pattern_title"),
        "implementation_hint": example.get("implementation_hint"),
        "repo": example.get("repo"),
        "pr_number": example.get("pr_number"),
        "pr_title": example.get("pr_title"),
        "pr_url": example.get("pr_url"),
        "comment_kind": example.get("comment_kind"),
        "path": example.get("path"),
        "line": example.get("line"),
        "human_comment": truncate_text(clean_historical_comment_body(str(example.get("body") or "")), 800),
        "diff_hunk": truncate_text(str(example.get("diff_hunk") or ""), 800),
        "file_evidence": file_evidence,
        "check_evidence": check_evidence,
    }


def filter_historical_context_for_path(
    historical_context: dict[str, Any] | None,
    path: str,
    *,
    max_examples: int = 5,
) -> dict[str, Any] | None:
    if not historical_context or not historical_context.get("enabled"):
        return historical_context

    matches = [item for item in historical_context.get("matches", []) if isinstance(item, dict)]
    if not matches:
        return {**historical_context, "matches": []}

    exact = []
    related = []
    basename = PurePosixPath(path).name.lower()
    suffix = PurePosixPath(path).suffix.lower()
    for item in matches:
        item_paths = {str(item.get("path") or "")}
        for evidence in item.get("file_evidence") or []:
            if isinstance(evidence, dict) and evidence.get("filename"):
                item_paths.add(str(evidence["filename"]))
        if path in item_paths:
            exact.append(item)
            continue
        item_basenames = {PurePosixPath(item_path).name.lower() for item_path in item_paths if item_path}
        item_suffixes = {PurePosixPath(item_path).suffix.lower() for item_path in item_paths if item_path}
        if basename in item_basenames or (suffix and suffix in item_suffixes):
            related.append(item)

    selected = dedupe_context_matches(exact + related + matches)[:max_examples]
    return {
        **historical_context,
        "matches": selected,
        "matched_example_count": len(selected),
        "chunk_filtered_for_path": path,
    }


def dedupe_context_matches(matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    seen: set[tuple[str, str, str | None, str]] = set()
    for item in matches:
        key = (
            str(item.get("pattern_id") or ""),
            str(item.get("repo") or ""),
            item.get("path"),
            str(item.get("human_comment") or "")[:160],
        )
        if key in seen:
            continue
        seen.add(key)
        output.append(item)
    return output
