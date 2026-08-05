"""Shared path exclusions for PR review content."""

from __future__ import annotations


REVIEW_EXCLUDED_PATHS = {
    ".github/workflows/agentic-pr-review.yml",
    ".github/workflows/agentic-pr-review.yaml",
}


def normalize_review_path(path: object) -> str:
    return str(path or "").strip().replace("\\", "/").removeprefix("./").lower()


def is_review_excluded_path(path: object) -> bool:
    return normalize_review_path(path) in REVIEW_EXCLUDED_PATHS


def review_exclusion_reason(path: object) -> str:
    if is_review_excluded_path(path):
        return "bot invocation workflow is excluded from review"
    return ""


def text_mentions_review_excluded_path(text: object) -> bool:
    normalized_text = normalize_review_path(text)
    return any(path in normalized_text for path in REVIEW_EXCLUDED_PATHS)


def finding_mentions_review_excluded_path(finding: dict[str, object]) -> bool:
    if is_review_excluded_path(finding.get("file")):
        return True
    fields = (
        "title",
        "code_reference",
        "evidence",
        "changed_line_evidence",
        "execution_path",
        "trigger",
        "observable_failure",
        "root_cause",
        "why_it_matters",
        "suggested_fix",
        "url",
    )
    return any(text_mentions_review_excluded_path(finding.get(field)) for field in fields)
