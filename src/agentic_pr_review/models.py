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
    "ci_pipeline_failure",
    "general",
}
ALLOWED_FINDING_CATEGORIES = {
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
}
ALLOWED_CAUSALITIES = {
    "introduced_by_pr",
    "worsened_by_pr",
    "exposed_by_pr",
    "pre_existing_unrelated",
    "unknown",
}
ALLOWED_PUBLICATION_DESTINATIONS = {
    "inline_blocking",
    "inline_non_blocking",
    "summary_high_priority",
    "summary_observation",
    "artifact_only",
    "suppress",
}
REQUIRED_TEXT_FIELDS = ("evidence", "why_it_matters", "suggested_fix")
MIN_PUBLISH_CONFIDENCE_SCORE = 0.8
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
    confidence, confidence_score = normalize_confidence(raw.get("confidence"), raw.get("confidence_score"))
    raw_category = str(raw.get("category") or "general").strip()
    if raw_category in ALLOWED_FINDING_CATEGORIES:
        finding_category = raw_category
        category = str(raw.get("review_area") or raw.get("area") or raw.get("legacy_category") or "").strip()
    else:
        category = raw_category
        finding_category = str(raw.get("finding_category") or raw.get("defect_category") or "").strip()
    if finding_category not in ALLOWED_FINDING_CATEGORIES:
        finding_category = infer_finding_category(raw, category=category)
    causality = str(raw.get("causality") or "").strip()
    if causality not in ALLOWED_CAUSALITIES:
        causality = "unknown"
    publication_destination = str(raw.get("publication_destination") or "").strip()
    if publication_destination not in ALLOWED_PUBLICATION_DESTINATIONS:
        publication_destination = "artifact_only" if finding_category in {"design_observation", "maintainability_suggestion"} else "inline_non_blocking"
    if confidence not in {"high", "medium", "low"}:
        confidence = "medium"
    if category not in ALLOWED_CATEGORIES:
        category = infer_review_area(raw)
    merge_blocking = raw.get("merge_blocking")
    if not isinstance(merge_blocking, bool):
        merge_blocking = bool(severity in {"critical"} or category in {"precommit", "merge_conflict"})
    return {
        "id": str(raw.get("id") or ""),
        "title": str(raw.get("title") or "Untitled finding").strip(),
        "category": category,
        "review_area": category,
        "finding_category": finding_category,
        "causality": causality,
        "severity": severity,
        "confidence": confidence,
        "confidence_score": confidence_score,
        "merge_blocking": merge_blocking,
        "publication_destination": publication_destination,
        "file": raw.get("file"),
        "line": raw.get("line_start", raw.get("line")),
        "line_start": raw.get("line_start", raw.get("line")),
        "line_end": raw.get("line_end"),
        "code_reference": normalize_optional_text(raw.get("code_reference")),
        "evidence": str(raw.get("evidence") or "").strip(),
        "why_it_matters": str(raw.get("why_it_matters") or "").strip(),
        "suggested_fix": str(raw.get("suggested_fix") or "").strip(),
        "suggested_code": normalize_optional_text(raw.get("suggested_code")),
        "changed_line_evidence": normalize_optional_text(raw.get("changed_line_evidence")),
        "execution_path": normalize_optional_text(raw.get("execution_path")),
        "trigger": normalize_optional_text(raw.get("trigger")),
        "observable_failure": normalize_optional_text(raw.get("observable_failure")),
        "root_cause": normalize_optional_text(raw.get("root_cause")),
        "repository_rule": normalize_optional_text(raw.get("repository_rule")),
        "source": str(raw.get("source") or default_source),
        "url": raw.get("url"),
        "ci_diagnosis_confidence": raw.get("ci_diagnosis_confidence"),
        "ci_diagnosis_confidence_score": raw.get("ci_diagnosis_confidence_score"),
        "ci_diagnosis_needs_model": raw.get("ci_diagnosis_needs_model"),
        "pipeline_failed_steps": raw.get("pipeline_failed_steps"),
        "pipeline_failed_hooks": raw.get("pipeline_failed_hooks"),
        "pipeline_failure_tool": raw.get("pipeline_failure_tool"),
        "pipeline_source_location": raw.get("pipeline_source_location"),
        "pipeline_log_excerpt": raw.get("pipeline_log_excerpt"),
        "pipeline_failed_jobs": raw.get("pipeline_failed_jobs"),
        "pipeline_jobs": raw.get("pipeline_jobs"),
    }


def normalize_confidence(confidence: Any, confidence_score: Any = None) -> tuple[str, float]:
    if isinstance(confidence, (int, float)):
        score = max(0.0, min(1.0, float(confidence)))
        if score >= 0.85:
            return "high", score
        if score >= 0.55:
            return "medium", score
        return "low", score
    score_provided = False
    parsed_confidence_score = parse_confidence_score(confidence_score)
    if parsed_confidence_score is not None:
        score = parsed_confidence_score
        score_provided = True
        if not confidence:
            if score >= 0.85:
                return "high", score
            if score >= 0.55:
                return "medium", score
            return "low", score
    else:
        score = 0.5
    label = str(confidence or "medium").lower()
    if score_provided and label in {"high", "medium", "low"}:
        return label, score
    if label == "high":
        return "high", max(score, 0.9)
    if label == "low":
        return "low", min(score, 0.35)
    return "medium", score if confidence_score is not None else 0.65


def parse_confidence_score(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, float(value)))
    if isinstance(value, str):
        try:
            return max(0.0, min(1.0, float(value)))
        except ValueError:
            return None
    return None


def infer_finding_category(raw: dict[str, Any], *, category: str) -> str:
    text = " ".join(
        str(raw.get(field) or "")
        for field in ("title", "evidence", "why_it_matters", "suggested_fix", "code_reference")
    ).lower()
    if any(phrase in text for phrase in ("pudb.set_trace", "breakpoint(", "interactive debugger", "set_trace()")):
        return "introduced_bug"
    if category == "unsafe_logging" or any(term in text for term in ("token", "secret", "authorization header", "credentials exposed")):
        return "security_issue"
    if category in {"precommit", "merge_conflict"}:
        return "introduced_bug"
    if any(term in text for term in ("app_version", "version bump", "release note", "release_notes", "unreleased.md")):
        return "release_management_suggestion"
    if category in {"docs_pr_accuracy", "missing_tests"}:
        return "repository_policy_suggestion"
    if any(phrase in text for phrase in SPECULATIVE_PHRASES):
        return "insufficient_evidence"
    if category in {"polling_checkpoint", "output_schema_mismatch", "api_auth_correctness", "pagination", "validation", "soar_metadata"}:
        return "introduced_regression"
    return "introduced_bug"


def infer_review_area(raw: dict[str, Any]) -> str:
    text = " ".join(
        str(raw.get(field) or "")
        for field in ("title", "evidence", "why_it_matters", "suggested_fix", "code_reference")
    ).lower()
    if any(term in text for term in ("oauth", "token", "auth", "client credentials")):
        return "api_auth_correctness"
    if any(term in text for term in ("poll", "checkpoint", "save_artifact", "save_container", "source_data_identifier")):
        return "polling_checkpoint"
    if any(term in text for term in ("output", "action_result.data", "summary", "schema")):
        return "output_schema_mismatch"
    if any(term in text for term in ("log", "debug", "secret", "authorization header", "token")):
        return "unsafe_logging"
    if any(term in text for term in ("read_only", "app json", "app_version", "python_version", "manifest")):
        return "soar_metadata"
    if any(term in text for term in ("readme", "docs", "release note")):
        return "docs_pr_accuracy"
    if "pagination" in text or "page" in text:
        return "pagination"
    if "test" in text:
        return "missing_tests"
    return "general"


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
    if str(finding.get("category") or "") == "ci_pipeline_failure":
        return is_actionable_review_finding(finding)
    if not has_publishable_confidence(finding):
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

    if str(finding.get("finding_category") or "") == "insufficient_evidence":
        return False
    if str(finding.get("publication_destination") or "") == "suppress":
        return False
    if not has_publishable_confidence(finding):
        return False

    file_path = str(finding.get("file") or "").strip()
    if not file_path or file_path.lower() in {"none", "null", "conversation"}:
        return is_actionable_fileless_finding(finding)

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


def is_actionable_fileless_finding(finding: dict[str, Any]) -> bool:
    category = str(finding.get("category") or "")
    if category != "ci_pipeline_failure":
        return False
    if not has_publishable_confidence(finding):
        return False
    if any(not str(finding.get(field) or "").strip() for field in REQUIRED_TEXT_FIELDS):
        return False
    if not str(finding.get("url") or "").strip() and not has_pipeline_job_link(finding):
        return False
    text = " ".join(
        str(finding.get(field) or "")
        for field in ("title", "evidence", "why_it_matters", "suggested_fix", "code_reference")
    ).lower()
    if any(phrase in text for phrase in SPECULATIVE_PHRASES):
        return False
    return True


def has_publishable_confidence(finding: dict[str, Any]) -> bool:
    return review_confidence_score(finding) >= MIN_PUBLISH_CONFIDENCE_SCORE


def review_confidence_score(finding: dict[str, Any]) -> float:
    for key in ("confidence_score", "ci_diagnosis_confidence_score"):
        value = finding.get(key)
        if isinstance(value, (int, float)):
            return max(0.0, min(1.0, float(value)))
        if isinstance(value, str):
            try:
                return max(0.0, min(1.0, float(value)))
            except ValueError:
                pass

    for key in ("confidence", "ci_diagnosis_confidence"):
        label = str(finding.get(key) or "").lower()
        if label == "high":
            return 0.9
        if label == "medium":
            return 0.65
        if label == "low":
            return 0.35
    return 0.0


def has_pipeline_job_link(finding: dict[str, Any]) -> bool:
    pipeline_jobs = finding.get("pipeline_jobs")
    if not isinstance(pipeline_jobs, list):
        return False
    return any(isinstance(item, dict) and str(item.get("url") or "").strip() for item in pipeline_jobs)


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
