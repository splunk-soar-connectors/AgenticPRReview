"""Render review JSON as a GitHub-friendly Markdown comment."""

from __future__ import annotations

from typing import Any


SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}


def render_comment(review_output: dict[str, Any], review_input: dict[str, Any], *, publishing_requested: bool = False) -> str:
    pr = review_input.get("pr", {})
    findings = list(review_output.get("findings") or [])
    findings.sort(key=lambda item: SEVERITY_ORDER.get(item.get("severity", "medium"), 2))

    lines = [
        "## Agentic SOAR Connector Review",
        "",
        f"PR: `{review_input.get('repo')}#{pr.get('number')}`",
        f"Status: `{review_output.get('overall_status', 'needs_review')}`",
        "",
    ]

    if not findings:
        lines.extend(
            [
                "No actionable connector issues with file-backed evidence were found.",
                "",
            ]
        )
    else:
        lines.append("### Findings")
        lines.append("")
        for index, finding in enumerate(findings, start=1):
            location = format_location(finding)
            confidence = finding.get("confidence", "medium")
            severity = finding.get("severity", "medium")
            category = finding.get("category", "general")
            lines.extend(
                [
                    f"{index}. **{finding.get('title', 'Finding')}**",
                    f"   - Severity: `{severity}` | Confidence: `{confidence}` | Category: `{category}`",
                ]
            )
            if location:
                lines.append(f"   - Location: {location}")
            if finding.get("code_reference"):
                lines.append(f"   - Code reference: `{finding['code_reference']}`")
            lines.append(f"   - Evidence: {finding['evidence']}")
            lines.append(f"   - Why it matters: {finding['why_it_matters']}")
            lines.append(f"   - Suggested fix: {finding['suggested_fix']}")
            lines.append("")

    deterministic_count = len(review_output.get("deterministic_findings") or [])
    if deterministic_count:
        lines.extend(
            [
                "### Review Inputs",
                "",
                f"- Deterministic connector checks produced `{deterministic_count}` candidate finding(s).",
            ]
        )
        usage = review_output.get("usage")
        if usage:
            lines.append(f"- Model usage: `{usage}`")
        if review_output.get("model"):
            lines.append(f"- Model: `{review_output['model']}`")
        lines.append("")

    if review_output.get("model_notes"):
        lines.extend(["### Notes", "", review_output["model_notes"], ""])

    if publishing_requested:
        lines.append("_Local review summary. GitHub comment publishing was requested; see publish_result.json._")
    else:
        lines.append("_Dry-run prototype comment. No GitHub comment was posted._")
    return "\n".join(lines).rstrip() + "\n"


def format_location(finding: dict[str, Any]) -> str:
    file_path = finding.get("file")
    if not file_path:
        return ""
    line = finding.get("line")
    if isinstance(line, int):
        return f"`{file_path}:{line}`"
    return f"`{file_path}`"
