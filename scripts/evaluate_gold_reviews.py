#!/usr/bin/env python3
"""Evaluate saved review artifacts against human-review gold expectations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from agentic_pr_review.checks import run_deterministic_checks
from agentic_pr_review.github_comments import build_comment_plan
from agentic_pr_review.models import deterministic_only_output
from agentic_pr_review.sdk_analysis import build_sdk_review_inventory


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "gold_file",
        nargs="?",
        default="examples/gold_reviews.connector_sdk.json",
        help="Gold-review expectation JSON file.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of a text report.",
    )
    args = parser.parse_args(argv)

    repo_root = Path.cwd()
    gold_path = resolve_path(args.gold_file, repo_root)
    gold = load_json(gold_path)
    results = evaluate_gold_file(gold, repo_root=repo_root)
    if args.json:
        print(json.dumps(results, indent=2, sort_keys=True))
    else:
        print(render_report(results))
    return 0 if results["passed"] else 1


def evaluate_gold_file(gold: dict[str, Any], *, repo_root: Path) -> dict[str, Any]:
    case_results = [evaluate_case(case, repo_root=repo_root) for case in gold.get("cases", [])]
    active_cases = [case for case in case_results if not case.get("skipped")]
    total_expected = sum(len(case.get("expected_results", [])) for case in active_cases)
    matched_expected = sum(
        1
        for case in active_cases
        for item in case.get("expected_results", [])
        if item.get("matched")
    )
    total_forbidden = sum(len(case.get("forbidden_results", [])) for case in active_cases)
    forbidden_hits = sum(
        1
        for case in active_cases
        for item in case.get("forbidden_results", [])
        if item.get("matched")
    )
    return {
        "schema_version": "0.1",
        "passed": matched_expected == total_expected and forbidden_hits == 0,
        "summary": {
            "case_count": len(case_results),
            "active_case_count": len(active_cases),
            "skipped_case_count": len(case_results) - len(active_cases),
            "expected_matched": matched_expected,
            "expected_total": total_expected,
            "forbidden_hits": forbidden_hits,
            "forbidden_total": total_forbidden,
        },
        "cases": case_results,
    }


def evaluate_case(case: dict[str, Any], *, repo_root: Path) -> dict[str, Any]:
    artifacts_dir = resolve_path(str(case.get("artifacts_dir") or ""), repo_root)
    review_input_path = artifacts_dir / "review_input.json"
    review_output_path = artifacts_dir / "review_output.json"
    if not review_input_path.exists():
        return {
            "id": case.get("id"),
            "repo": case.get("repo"),
            "pr_number": case.get("pr_number"),
            "artifacts_dir": str(artifacts_dir),
            "skipped": True,
            "candidate_count": 0,
            "deterministic_count": 0,
            "stored_model_count": 0,
            "deterministic_publishable_count": 0,
            "artifact_quality": {
                "status": "missing",
                "notes": [f"review_input.json not found at {review_input_path}"],
            },
            "expected_results": [],
            "forbidden_results": [],
            "passed": True,
        }
    review_input = load_json(review_input_path)
    stored_output = load_json(review_output_path) if review_output_path.exists() else {}
    deterministic_findings = run_deterministic_checks(review_input)
    stored_model_findings = stored_output.get("findings") or []
    source_candidates = {
        "deterministic": normalize_candidates(deterministic_findings, "deterministic"),
        "stored_model": normalize_candidates(stored_model_findings, "stored_model"),
    }
    source_candidates["combined"] = merge_candidates(
        source_candidates["deterministic"],
        source_candidates["stored_model"],
    )
    candidates = source_candidates["combined"]
    deterministic_plan = build_comment_plan(deterministic_only_output(deterministic_findings), review_input)
    artifact_quality = analyze_artifact_quality(review_input, deterministic_findings)

    expected_results = [
        evaluate_pattern(pattern, source_candidates, default_sources=["combined"])
        for pattern in case.get("expected", [])
    ]
    forbidden_results = [
        evaluate_pattern(pattern, source_candidates, default_sources=["deterministic"])
        for pattern in case.get("forbidden", [])
    ]
    return {
        "id": case.get("id"),
        "repo": case.get("repo"),
        "pr_number": case.get("pr_number"),
        "artifacts_dir": str(artifacts_dir),
        "candidate_count": len(candidates),
        "deterministic_count": len(deterministic_findings),
        "stored_model_count": len(stored_model_findings),
        "deterministic_publishable_count": len(deterministic_plan.get("comments") or []),
        "artifact_quality": artifact_quality,
        "expected_results": expected_results,
        "forbidden_results": forbidden_results,
        "passed": all(item["matched"] for item in expected_results)
        and not any(item["matched"] for item in forbidden_results)
        and artifact_quality["status"] != "bad",
    }


def normalize_candidates(group: list[dict[str, Any]], source: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for item in group:
        if not isinstance(item, dict):
            continue
        copied = dict(item)
        copied["_source"] = source
        output.append(copied)
    return output


def merge_candidates(*groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for group in groups:
        for item in group:
            if not isinstance(item, dict):
                continue
            key = (
                str(item.get("title") or "").lower(),
                str(item.get("file") or ""),
                str(item.get("line") or ""),
            )
            if key in seen:
                continue
            seen.add(key)
            output.append(item)
    return output


def evaluate_pattern(
    pattern: dict[str, Any],
    source_candidates: dict[str, list[dict[str, Any]]],
    *,
    default_sources: list[str],
) -> dict[str, Any]:
    terms = [str(term).lower() for term in pattern.get("terms", [])]
    sources = pattern_sources(pattern, default_sources)
    candidates = merge_candidates(*(source_candidates.get(source, []) for source in sources))
    candidate_texts = [candidate_text(item) for item in candidates]
    for index, text in enumerate(candidate_texts):
        if all(term in text for term in terms):
            candidate = candidates[index]
            return {
                "id": pattern.get("id"),
                "terms": pattern.get("terms", []),
                "sources": sources,
                "matched": True,
                "matched_title": candidate.get("title"),
                "matched_file": candidate.get("file"),
                "matched_line": candidate.get("line"),
                "matched_source": candidate.get("_source"),
            }
    return {
        "id": pattern.get("id"),
        "terms": pattern.get("terms", []),
        "sources": sources,
        "matched": False,
    }


def pattern_sources(pattern: dict[str, Any], default_sources: list[str]) -> list[str]:
    raw = pattern.get("sources")
    if raw is None:
        return default_sources
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list):
        return [str(item) for item in raw if str(item).strip()]
    return default_sources


def analyze_artifact_quality(
    review_input: dict[str, Any],
    deterministic_findings: list[dict[str, Any]],
) -> dict[str, Any]:
    """Detect saved-run context problems that make a human comparison unfair."""

    notes: list[str] = []
    collector_notes = review_input.get("collector_notes") or {}
    if isinstance(collector_notes, dict):
        missing = int(collector_notes.get("full_file_missing_count") or 0)
        if missing:
            notes.append(f"{missing} selected full files were not collected")
        truncated = int(collector_notes.get("changed_file_truncated_patch_count") or 0)
        if truncated:
            notes.append(f"{truncated} changed-file patches were truncated")
        missing_patches = int(collector_notes.get("changed_file_missing_patch_count") or 0)
        if missing_patches:
            notes.append(f"{missing_patches} changed-file patches were missing")

    inventory = build_sdk_review_inventory(review_input)
    if inventory.get("is_sdk_app"):
        python_inventory = inventory.get("python") or {}
        action_count = len(python_inventory.get("action_registrations") or [])
        has_pyproject = bool((review_input.get("full_files") or {}).get("pyproject.toml"))
        if has_pyproject and action_count == 0:
            notes.append("SDK inventory found zero action registrations")
        main_file = str(inventory.get("main_module_file") or "")
        for finding in deterministic_findings:
            if (
                finding.get("title") == "Changed Python file does not compile"
                and str(finding.get("file") or "") == main_file
            ):
                notes.append(f"SDK main module does not compile: {finding.get('evidence')}")
                break

    lowered_errors = "\n".join(
        str(error) for error in (collector_notes.get("full_file_fetch_errors") or [])
    ).lower() if isinstance(collector_notes, dict) else ""
    if "github api error" in lowered_errors or "raw_url fallback failed" in lowered_errors:
        notes.append("full-file fetch errors were recorded")

    status = "ok"
    if any(
        phrase in note
        for note in notes
        for phrase in (
            "SDK main module does not compile",
            "SDK inventory found zero action registrations",
        )
    ):
        status = "bad"
    elif notes:
        status = "warn"

    return {"status": status, "notes": notes}


def candidate_text(finding: dict[str, Any]) -> str:
    return "\n".join(
        str(finding.get(key) or "")
        for key in (
            "title",
            "category",
            "file",
            "line",
            "code_reference",
            "evidence",
            "why_it_matters",
            "suggested_fix",
            "suggested_code",
        )
    ).lower()


def render_report(results: dict[str, Any]) -> str:
    lines = [
        "# Gold Review Evaluation",
        "",
        f"Passed: `{results['passed']}`",
        (
            "Expected: "
            f"{results['summary']['expected_matched']}/{results['summary']['expected_total']} matched; "
            "Forbidden: "
            f"{results['summary']['forbidden_hits']}/{results['summary']['forbidden_total']} hit"
        ),
        "",
    ]
    for case in results.get("cases", []):
        status = "SKIP" if case.get("skipped") else ("PASS" if case.get("passed") else "FAIL")
        lines.extend(
            [
                f"## {case.get('id')} - {status}",
                f"- Candidates: {case.get('candidate_count')} "
                f"(deterministic {case.get('deterministic_count')}, stored model {case.get('stored_model_count')})",
                f"- Deterministic publishable comments: {case.get('deterministic_publishable_count')}",
            ]
        )
        artifact_quality = case.get("artifact_quality") or {}
        if artifact_quality.get("status") != "ok":
            notes = "; ".join(artifact_quality.get("notes") or [])
            lines.append(f"- Artifact quality: {artifact_quality.get('status')} - {notes}")
        for item in case.get("expected_results", []):
            marker = "OK" if item.get("matched") else "MISS"
            source = f" [{item.get('matched_source')}]" if item.get("matched_source") else ""
            detail = f" -> {item.get('matched_title')}{source}" if item.get("matched") else ""
            lines.append(f"- [{marker}] expected `{item.get('id')}`{detail}")
        for item in case.get("forbidden_results", []):
            marker = "HIT" if item.get("matched") else "OK"
            source = f" [{item.get('matched_source')}]" if item.get("matched_source") else ""
            detail = f" -> {item.get('matched_title')}{source}" if item.get("matched") else ""
            lines.append(f"- [{marker}] forbidden `{item.get('id')}`{detail}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def resolve_path(value: str, repo_root: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return repo_root / path


if __name__ == "__main__":
    raise SystemExit(main())
