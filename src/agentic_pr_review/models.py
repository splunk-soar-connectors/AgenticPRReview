"""Small schema helpers for review output."""

from __future__ import annotations

import re
from typing import Any


ALLOWED_SEVERITIES = {"critical", "high", "medium", "low", "info"}
ALLOWED_STATUSES = {"needs_review", "looks_good", "blocked_by_ci", "error"}
ALLOWED_CATEGORIES = {
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
}
REQUIRED_TEXT_FIELDS = ("evidence", "why_it_matters", "suggested_fix")
SPECULATIVE_PHRASES = (
    "cannot confirm",
    "could not confirm",
    "do not appear",
    "does not appear",
    "it is unclear",
    "may be",
    "might be",
    "needs to be verified",
    "not enough evidence",
    "should verify",
    "unable to confirm",
    "verify whether",
    "verify that",
    "confirm whether",
    "confirm that",
)


def normalize_finding(raw: dict[str, Any], *, default_source: str) -> dict[str, Any]:
    severity = str(raw.get("severity", "medium")).lower()
    if severity not in ALLOWED_SEVERITIES:
        severity = "medium"
    confidence = str(raw.get("confidence", "medium")).lower()
    if confidence not in {"high", "medium", "low"}:
        confidence = "medium"
    category = str(raw.get("category") or "general").strip()
    if category not in ALLOWED_CATEGORIES:
        category = "general"
    return {
        "id": str(raw.get("id") or ""),
        "title": str(raw.get("title") or "Untitled finding").strip(),
        "category": category,
        "severity": severity,
        "confidence": confidence,
        "file": raw.get("file"),
        "line": raw.get("line"),
        "code_reference": normalize_optional_text(raw.get("code_reference")),
        "evidence": str(raw.get("evidence") or "").strip(),
        "why_it_matters": str(raw.get("why_it_matters") or "").strip(),
        "suggested_fix": str(raw.get("suggested_fix") or "").strip(),
        "suggested_code": normalize_optional_text(raw.get("suggested_code")),
        "source": str(raw.get("source") or default_source),
        "url": raw.get("url"),
    }


def normalize_review_output(raw: dict[str, Any], *, deterministic_findings: list[dict[str, Any]]) -> dict[str, Any]:
    findings = raw.get("findings", [])
    if not isinstance(findings, list):
        findings = []
    normalized = [
        finding
        for finding in (normalize_finding(item, default_source="claude") for item in findings if isinstance(item, dict))
        if is_actionable_review_finding(finding)
    ]
    normalized = merge_promotable_deterministic_findings(normalized, deterministic_findings)
    status = str(raw.get("overall_status") or "").lower()
    if status not in ALLOWED_STATUSES:
        status = "needs_review" if normalized else "looks_good"
    elif normalized and status == "looks_good":
        status = "needs_review"
    elif not normalized and status in {"needs_review", "blocked_by_ci"}:
        status = "looks_good"
    summary = str(raw.get("summary") or "").strip()
    safe_to_publish = bool(raw.get("safe_to_publish", True))
    if normalized and status != "error":
        safe_to_publish = True
    return {
        "summary": summary,
        "overall_status": status,
        "safe_to_publish": safe_to_publish,
        "findings": normalized,
        "deterministic_findings": deterministic_findings,
        "model_notes": str(raw.get("model_notes") or "").strip(),
    }


def deterministic_only_output(deterministic_findings: list[dict[str, Any]]) -> dict[str, Any]:
    return normalize_review_output(
        {
            "summary": "Model review was skipped; deterministic checks only.",
            "overall_status": "needs_review" if deterministic_findings else "looks_good",
            "safe_to_publish": True,
            "findings": deterministic_findings,
            "model_notes": "Run without --skip-model to include model review.",
        },
        deterministic_findings=deterministic_findings,
    )


def merge_promotable_deterministic_findings(
    model_findings: list[dict[str, Any]],
    deterministic_findings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged = list(model_findings)
    seen = {finding_merge_key(finding) for finding in merged}
    for raw in deterministic_findings:
        if not isinstance(raw, dict):
            continue
        finding = normalize_finding(raw, default_source="deterministic")
        if not should_promote_deterministic_finding(finding):
            continue
        key = finding_merge_key(finding)
        if key in seen:
            continue
        merged.append(finding)
        seen.add(key)
    return merged


def should_promote_deterministic_finding(finding: dict[str, Any]) -> bool:
    if str(finding.get("confidence") or "").lower() != "high":
        return False
    if str(finding.get("category") or "") in {"ci_synthesis"}:
        return False
    return is_actionable_review_finding(finding)


def finding_merge_key(finding: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(finding.get("category") or ""),
        str(finding.get("title") or "").lower(),
        str(finding.get("file") or ""),
        str(finding.get("line") or ""),
    )


def normalize_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip("\n")
    return text if text.strip() else None


def is_actionable_review_finding(finding: dict[str, Any]) -> bool:
    """Return True only for findings with enough evidence to show a user."""

    file_path = str(finding.get("file") or "").strip()
    if not file_path or file_path.lower() in {"none", "null", "conversation"}:
        return False

    if any(not str(finding.get(field) or "").strip() for field in REQUIRED_TEXT_FIELDS):
        return False

    if not has_line_or_code_reference(finding):
        return False

    text = " ".join(
        str(finding.get(field) or "")
        for field in ("title", "evidence", "why_it_matters", "suggested_fix", "code_reference")
    ).lower()
    if any(phrase in text for phrase in SPECULATIVE_PHRASES):
        return False

    suggested_fix = str(finding.get("suggested_fix") or "").strip().lower()
    if suggested_fix.startswith(("verify ", "confirm ", "consider ", "inspect ")):
        return False

    return True


def has_line_or_code_reference(finding: dict[str, Any]) -> bool:
    line = finding.get("line")
    if isinstance(line, int) and line > 0:
        return True
    if isinstance(line, str) and line.isdigit() and int(line) > 0:
        return True
    if str(finding.get("code_reference") or "").strip():
        return True
    evidence = str(finding.get("evidence") or "")
    file_path = str(finding.get("file") or "")
    if file_path and file_path in evidence:
        return True
    if re.search(r"`[^`]{3,120}`", evidence):
        return True
    if re.search(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\s*\(", evidence):
        return True
    if re.search(r"\b(action_result\.summary|summary\.|data\.|parameters?\.)[A-Za-z0-9_.*-]*", evidence):
        return True
    return False
