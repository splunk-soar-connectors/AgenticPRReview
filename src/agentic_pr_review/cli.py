"""Command-line interface for the local dry-run reviewer."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import Any

from .checks import run_deterministic_checks
from .gateway_claude import GatewayClaudeReviewer, GatewayReviewError
from .collector import PRCollector
from .config import DEFAULT_MAX_FILE_CHARS, DEFAULT_MAX_MODEL_INPUT_CHARS, DEFAULT_MAX_PATCH_CHARS, RuntimeConfig
from .deep_review import DeepPRCollector
from .github_comments import build_comment_plan, filter_review_output_for_pr_context, publish_comment_plan, render_comment_plan
from .github_client import GitHubError
from .github_client_factory import build_github_client
from .historical_context import enrich_with_historical_context
from .json_utils import dump_json
from .models import deterministic_only_output
from .progress import format_duration, ProgressReporter
from .renderer import render_comment
from .sdk_analysis import build_sdk_review_inventory
from .sdk_manifest import enrich_with_sdk_manifest, mark_sdk_manifest_disabled
from .secret_redactor import redact_obj, redact_text


DEFAULT_RUNS_DIR = Path(__file__).resolve().parents[2] / "runs"
TARGET_PIPELINE_JOB_ALIASES = {
    "pre-commit": (
        "pre-commit",
        "precommit",
        "pre commit",
        "hook id:",
        "detect-secrets",
        "detect secrets",
        "secret keyword",
        "potential secrets",
        "ruff",
        "semgrep",
        "djlint",
        "mdformat",
    ),
    "compile": (
        "compile",
        "compile application",
        "soar instance",
        "phantom instance",
        "phantom_instance",
        "connecttimeout",
        "connect timeout",
        "connection timed out",
        "timed out connecting",
        "installing app on",
        "app installation failed",
    ),
    "build": (
        "build",
        "build application",
        "build sdk app",
        "package build",
        "app-tar",
        "tarball",
        "upload app tar",
    ),
    "semantic-release-preview": (
        "semantic-release-preview",
        "semantic-release",
        "semantic release",
        "release preview",
        "release notes",
        "release_version.txt",
        "release_notes/unreleased.md",
        "unreleased.md",
    ),
}
PIPELINE_DUPLICATE_CATEGORIES = {"ci_pipeline_failure", "ci_synthesis", "precommit"}
PIPELINE_PUBLICATION_SUCCESS_STATUSES = ("posted", "skipped_duplicate")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "review":
        return run_review(args)
    parser.print_help()
    return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agentic-pr-review",
        description="Local dry-run SOAR connector PR review prototype.",
    )
    subparsers = parser.add_subparsers(dest="command")

    review = subparsers.add_parser("review", help="Review a GitHub pull request.")
    review.add_argument("repo", help="GitHub repository in owner/name form.")
    review.add_argument("pr_number", type=int, help="Pull request number.")
    review.add_argument(
        "--output-dir",
        default=str(DEFAULT_RUNS_DIR),
        help="Directory for review artifacts. Defaults to the AgenticPRReview repo runs directory.",
    )
    review.add_argument("--skip-model", action="store_true", help="Run deterministic checks only.")
    review.add_argument("--max-patch-chars", type=int, default=DEFAULT_MAX_PATCH_CHARS)
    review.add_argument("--max-file-chars", type=int, default=DEFAULT_MAX_FILE_CHARS)
    review.add_argument("--max-model-input-chars", type=int, default=DEFAULT_MAX_MODEL_INPUT_CHARS)
    review.add_argument("--max-model-tokens", type=int, default=4500)
    review.add_argument("--shallow", action="store_true", help="Disable deep base/head collection and chunked review.")
    review.add_argument("--deep-max-file-bytes", type=int, default=5_000_000)
    review.add_argument("--deep-max-file-chars", type=int, default=DEFAULT_MAX_FILE_CHARS)
    review.add_argument("--deep-chunk-chars", type=int, default=35_000)
    review.add_argument(
        "--deep-concurrency",
        type=int,
        default=0,
        help="Maximum concurrent deep model chunk reviews. 0 selects concurrency from workload size/risk.",
    )
    review.add_argument("--deep-max-chunks", type=int, default=0, help="Limit deep model chunk reviews. 0 means all chunks.")
    review.add_argument(
        "--enable-sdk-manifest",
        action="store_true",
        help=(
            "Fetch the PR archive and run soarapps manifests create for SDK apps. "
            "This executes PR code in a sanitized subprocess, so it is opt-in."
        ),
    )
    review.add_argument(
        "--disable-sdk-manifest",
        action="store_true",
        help="Deprecated compatibility flag. SDK manifest generation is disabled unless --enable-sdk-manifest is set.",
    )
    review.add_argument(
        "--sdk-manifest-timeout",
        type=int,
        default=120,
        help="Timeout in seconds for soarapps manifests create.",
    )
    review.add_argument(
        "--historical-examples",
        help="Optional training_examples.jsonl file for runtime historical retrieval. Defaults to the newest local mining output.",
    )
    review.add_argument(
        "--historical-max-examples",
        type=int,
        default=8,
        help="Maximum historical examples to inject into model review.",
    )
    review.add_argument(
        "--historical-min-score",
        type=int,
        default=8,
        help="Minimum relevance score for runtime historical examples.",
    )
    review.add_argument(
        "--disable-historical-context",
        action="store_true",
        help="Disable runtime retrieval from mined historical review examples.",
    )
    review.add_argument(
        "--publish-comments",
        action="store_true",
        help="Post one concise GitHub comment per finding. Without this flag, only local artifacts are written.",
    )
    review.add_argument(
        "--allow-duplicate-comments",
        action="store_true",
        help="Post comments even when a prior bot marker is already present on the PR.",
    )
    review.add_argument(
        "--max-published-comments",
        type=int,
        default=0,
        help="Maximum number of comments to post when --publish-comments is set. 0 means no fixed cap.",
    )
    return parser


def run_review(args: argparse.Namespace) -> int:
    if args.enable_sdk_manifest and args.disable_sdk_manifest:
        print("error: --enable-sdk-manifest and --disable-sdk-manifest cannot both be set", file=sys.stderr)
        return 2

    config = RuntimeConfig.from_env(
        max_patch_chars=args.max_patch_chars,
        max_file_chars=args.max_file_chars,
        max_model_input_chars=args.max_model_input_chars,
    )

    run_dir = make_run_dir(Path(args.output_dir).expanduser(), args.repo, args.pr_number)
    run_dir.mkdir(parents=True, exist_ok=True)
    progress = ProgressReporter()
    review_started = time.monotonic()
    review_mode = "shallow" if args.shallow else "deep"
    publish_mode = "publishing enabled" if args.publish_comments else "dry run"
    progress(f"Review started for {args.repo}#{args.pr_number} ({review_mode}; {publish_mode}).")

    try:
        stage_started = time.monotonic()
        progress("Collecting pull request, diff, comments, and CI context.")
        client = build_github_client(config)
        collector = (
            PRCollector(client, config)
            if args.shallow
            else DeepPRCollector(
                client,
                config,
                deep_max_file_bytes=args.deep_max_file_bytes,
                deep_max_file_chars=args.deep_max_file_chars,
                deep_chunk_chars=args.deep_chunk_chars,
            )
        )
        review_input = collector.collect(args.repo, args.pr_number)
        changed_file_count = len(review_input.get("changed_files") or [])
        deep_chunk_count = len(((review_input.get("deep_review") or {}).get("chunks")) or [])
        progress(
            f"Collection completed in {format_duration(time.monotonic() - stage_started)}: "
            f"{changed_file_count} changed file(s), {deep_chunk_count} deep-review chunk(s)."
        )
        if args.enable_sdk_manifest:
            stage_started = time.monotonic()
            progress("SDK manifest generation started.")
            review_input = enrich_with_sdk_manifest(
                review_input,
                client,
                args.repo,
                timeout_seconds=args.sdk_manifest_timeout,
            )
            progress(f"SDK manifest generation completed in {format_duration(time.monotonic() - stage_started)}.")
        else:
            review_input = mark_sdk_manifest_disabled(
                review_input,
                reason=(
                    "Generated SDK manifest evidence is disabled by default because "
                    "soarapps manifests create imports PR code. Pass --enable-sdk-manifest "
                    "for trusted local analysis."
                ),
            )
        review_input["sdk_review_inventory"] = build_sdk_review_inventory(review_input)
        if not args.disable_historical_context:
            stage_started = time.monotonic()
            progress("Historical review context retrieval started.")
            review_input = enrich_with_historical_context(
                review_input,
                examples_path=args.historical_examples,
                max_examples=args.historical_max_examples,
                min_score=args.historical_min_score,
            )
            progress(
                f"Historical review context retrieval completed in "
                f"{format_duration(time.monotonic() - stage_started)}."
            )
        stage_started = time.monotonic()
        progress("Deterministic checks started.")
        deterministic_findings = run_deterministic_checks(review_input)
        progress(
            f"Deterministic checks completed in {format_duration(time.monotonic() - stage_started)} "
            f"with {len(deterministic_findings)} finding(s)."
        )
        reviewer: GatewayClaudeReviewer | None = None
        if not args.skip_model:
            def get_reviewer() -> GatewayClaudeReviewer:
                nonlocal reviewer
                if reviewer is None:
                    reviewer = build_model_reviewer(config, max_tokens=args.max_model_tokens, progress=progress)
                return reviewer

            deterministic_findings = refine_low_confidence_pipeline_findings(
                deterministic_findings,
                reviewer_factory=get_reviewer,
                progress=progress,
            )
        else:
            deterministic_findings = conservatively_downgrade_low_confidence_pipeline_findings(deterministic_findings)
        publish_target_pipeline_failure_comments(
            client,
            review_input,
            deterministic_findings,
            run_dir=run_dir,
            publish_comments=args.publish_comments,
            allow_duplicates=args.allow_duplicate_comments,
            progress=progress,
        )

        if args.skip_model:
            progress("Model review skipped by --skip-model.")
            review_output = deterministic_only_output(deterministic_findings)
        else:
            stage_started = time.monotonic()
            progress(f"Model review started with provider {config.model_provider}.")
            reviewer = reviewer or build_model_reviewer(config, max_tokens=args.max_model_tokens, progress=progress)
            if args.shallow:
                review_output = reviewer.review(review_input, deterministic_findings)
            else:
                review_output = reviewer.review_deep(
                    review_input,
                    deterministic_findings,
                    max_chunks=args.deep_max_chunks,
                    deep_concurrency=args.deep_concurrency,
                    checkpoint_path=run_dir / "deep_review_checkpoint.json",
                )
            progress(f"Model review completed in {format_duration(time.monotonic() - stage_started)}.")

        stage_started = time.monotonic()
        progress("Filtering findings and building the GitHub comment plan.")
        review_output = filter_review_output_for_pr_context(review_output, review_input)
        review_output = suppress_redundant_published_pipeline_findings(
            review_output,
            review_input,
            progress=progress,
        )
        comment = render_comment(review_output, review_input, publishing_requested=args.publish_comments)
        max_comments = args.max_published_comments if args.max_published_comments > 0 else None
        comment_plan = build_comment_plan(review_output, review_input, max_comments=max_comments)
        planned_comment_count = len(comment_plan.get("comments") or [])
        progress(
            f"Comment plan completed in {format_duration(time.monotonic() - stage_started)} "
            f"with {planned_comment_count} publishable comment(s)."
        )
        publish_result = None
        if args.publish_comments:
            stage_started = time.monotonic()
            progress(f"GitHub publication started for {planned_comment_count} planned comment(s).")
            publish_result = publish_comment_plan(
                client,
                comment_plan,
                review_input,
                allow_duplicates=args.allow_duplicate_comments,
                max_comments=max_comments,
            )
            progress(
                f"GitHub publication completed in {format_duration(time.monotonic() - stage_started)}: "
                f"{publish_result.get('posted', 0)} posted, {publish_result.get('skipped', 0)} skipped, "
                f"{publish_result.get('errors', 0)} error(s)."
            )
        progress("Writing review artifacts.")
        write_artifacts(run_dir, review_input, review_output, comment, comment_plan, publish_result)
    except (GitHubError, GatewayReviewError, RuntimeError, OSError, ValueError) as exc:
        safe_error = redact_text(str(exc))
        error_payload = {
            "summary": "Review failed.",
            "overall_status": "error",
            "safe_to_publish": False,
            "findings": [],
            "deterministic_findings": [],
            "model_notes": safe_error,
        }
        write_json(run_dir / "review_output.json", error_payload)
        (run_dir / "comment.md").write_text(f"Review failed: {safe_error}\n", encoding="utf-8")
        progress(f"Review failed after {format_duration(time.monotonic() - review_started)}: {safe_error}")
        print(f"error: {safe_error}", file=sys.stderr)
        print(f"artifacts: {run_dir}")
        return 1

    progress(f"Review completed successfully in {format_duration(time.monotonic() - review_started)}.")
    print(f"artifacts: {run_dir}")
    print(f"comment: {run_dir / 'comment.md'}")
    print(f"planned comments: {run_dir / 'planned_comments.md'}")
    if args.publish_comments:
        print(f"publish result: {run_dir / 'publish_result.json'}")
    return 0


def publish_target_pipeline_failure_comments(
    client: Any,
    review_input: dict[str, Any],
    deterministic_findings: list[dict[str, Any]],
    *,
    run_dir: Path,
    publish_comments: bool,
    allow_duplicates: bool,
    progress: ProgressReporter,
) -> dict[str, Any] | None:
    pipeline_findings = [
        finding
        for finding in deterministic_findings
        if isinstance(finding, dict) and str(finding.get("category") or "") == "ci_pipeline_failure"
    ]
    if not pipeline_findings:
        return None

    progress(f"Target pipeline failure notification planned for {len(pipeline_findings)} failed job(s).")
    review_output = deterministic_only_output(pipeline_findings)
    comment_plan = build_comment_plan(review_output, review_input, max_comments=None)
    write_json(run_dir / "ci_pipeline_comment_plan.json", comment_plan)
    planned_count = len(comment_plan.get("comments") or [])
    if not publish_comments or planned_count == 0:
        progress(f"Target pipeline failure notification prepared with {planned_count} publishable comment(s).")
        return {"planned": planned_count, "published": False}

    progress(f"Publishing {planned_count} target pipeline failure comment(s) before model review.")
    publish_result = publish_comment_plan(
        client,
        comment_plan,
        review_input,
        allow_duplicates=allow_duplicates,
        max_comments=None,
        apply_label=False,
    )
    write_json(run_dir / "ci_pipeline_publish_result.json", publish_result)
    mark_successfully_published_markers_as_existing(review_input, comment_plan, publish_result)
    record_successfully_published_target_pipeline_jobs(review_input, comment_plan, publish_result)
    progress(
        f"Target pipeline failure publication completed: "
        f"{publish_result.get('posted', 0)} posted, {publish_result.get('skipped', 0)} skipped, "
        f"{publish_result.get('errors', 0)} error(s)."
    )
    return publish_result


def refine_low_confidence_pipeline_findings(
    deterministic_findings: list[dict[str, Any]],
    *,
    reviewer_factory: Any,
    progress: ProgressReporter,
) -> list[dict[str, Any]]:
    needs_refinement = [
        finding
        for finding in deterministic_findings
        if isinstance(finding, dict)
        and str(finding.get("category") or "") == "ci_pipeline_failure"
        and finding.get("ci_diagnosis_needs_model") is True
    ]
    if not needs_refinement:
        return deterministic_findings

    progress(
        f"CI failure diagnosis fallback needed for {len(needs_refinement)} low-confidence pipeline finding(s)."
    )
    try:
        reviewer = reviewer_factory()
    except Exception as exc:  # noqa: BLE001 - model fallback should not break deterministic review
        progress(f"CI failure diagnosis fallback unavailable: {redact_text(str(exc))}.")
        return [
            conservatively_downgrade_pipeline_finding(finding)
            if isinstance(finding, dict)
            and str(finding.get("category") or "") == "ci_pipeline_failure"
            and finding.get("ci_diagnosis_needs_model") is True
            else finding
            for finding in deterministic_findings
        ]

    refined: list[dict[str, Any]] = []
    for finding in deterministic_findings:
        if (
            not isinstance(finding, dict)
            or str(finding.get("category") or "") != "ci_pipeline_failure"
            or finding.get("ci_diagnosis_needs_model") is not True
        ):
            refined.append(finding)
            continue

        try:
            diagnosis = reviewer.diagnose_ci_failure(
                job_name=target_job_name_from_finding(finding),
                failed_steps=[str(item) for item in finding.get("pipeline_failed_steps") or []],
                conclusion=target_job_conclusion_from_evidence(str(finding.get("evidence") or "")),
                deterministic_root_cause=str(finding.get("root_cause") or ""),
                deterministic_fix=str(finding.get("suggested_fix") or ""),
                log_excerpt=str(finding.get("pipeline_log_excerpt") or finding.get("evidence") or ""),
            )
        except GatewayReviewError as exc:
            progress(f"CI failure diagnosis fallback failed: {redact_text(str(exc))}.")
            refined.append(conservatively_downgrade_pipeline_finding(finding))
            continue

        if diagnosis.get("diagnosis_status") != "confirmed":
            progress(
                f"CI failure diagnosis fallback could not confirm {target_job_name_from_finding(finding)} root cause."
            )
            refined.append(conservatively_downgrade_pipeline_finding(finding))
            continue

        updated = dict(finding)
        updated["root_cause"] = diagnosis["root_cause"]
        updated["observable_failure"] = diagnosis["failure_scenario"] or (
            f"The `{target_job_name_from_finding(finding)}` job cannot pass because {diagnosis['root_cause']}."
        )
        updated["suggested_fix"] = diagnosis["suggested_fix"]
        updated["ci_diagnosis_confidence"] = diagnosis["confidence"]
        updated["ci_diagnosis_confidence_score"] = diagnosis["confidence_score"]
        updated["ci_diagnosis_needs_model"] = False
        updated["confidence"] = diagnosis["confidence"]
        updated["confidence_score"] = diagnosis["confidence_score"]
        updated["evidence"] = pipeline_evidence_with_model_diagnosis(updated, diagnosis)
        progress(
            f"CI failure diagnosis fallback confirmed {target_job_name_from_finding(finding)} "
            f"with {diagnosis['confidence']} confidence."
        )
        refined.append(updated)

    return refined


def suppress_redundant_published_pipeline_findings(
    review_output: dict[str, Any],
    review_input: dict[str, Any],
    *,
    progress: ProgressReporter | None = None,
) -> dict[str, Any]:
    published_jobs = published_target_pipeline_jobs(review_input)
    if not published_jobs:
        return review_output

    original_findings = [finding for finding in review_output.get("findings", []) or [] if isinstance(finding, dict)]
    findings = []
    suppressed = []
    for finding in original_findings:
        duplicate_job = redundant_published_pipeline_job_for_finding(finding, published_jobs)
        if duplicate_job:
            suppressed.append(
                {
                    "id": finding.get("id"),
                    "title": finding.get("title"),
                    "category": finding.get("category"),
                    "code_reference": finding.get("code_reference"),
                    "duplicate_of_pipeline_job": duplicate_job,
                }
            )
            continue
        findings.append(finding)

    if not suppressed:
        return review_output

    output = dict(review_output)
    output["findings"] = findings
    verification = dict(output.get("behavioral_verification") or {})
    verification["suppressed_duplicate_pipeline_comment_count"] = len(suppressed)
    verification["suppressed_duplicate_pipeline_comments"] = suppressed[:50]
    output["behavioral_verification"] = verification

    suppressed_jobs = sorted(set(item["duplicate_of_pipeline_job"] for item in suppressed))
    jobs_text = ", ".join(f"`{job}`" for job in suppressed_jobs)
    note = (
        f"Suppressed {len(suppressed)} duplicate final finding(s) already covered by "
        f"early target pipeline comment(s): {jobs_text}."
    )
    existing_notes = str(output.get("model_notes") or "").strip()
    output["model_notes"] = f"{existing_notes}\n{note}".strip() if existing_notes else note
    if progress:
        progress(note)
    return output


def redundant_published_pipeline_job_for_finding(finding: dict[str, Any], published_jobs: set[str]) -> str:
    category = str(finding.get("category") or "").lower()
    text = pipeline_duplicate_match_text(finding)
    own_job = normalize_pipeline_job_name(target_job_name_from_finding(finding))

    if category == "ci_pipeline_failure" and own_job in published_jobs:
        return own_job
    if category == "precommit" and "pre-commit" in published_jobs:
        return "pre-commit"
    if category == "ci_synthesis" and published_jobs:
        return sorted(published_jobs)[0]

    if not looks_like_pipeline_finding(category, text):
        return ""

    for job in sorted(published_jobs):
        if job == own_job or finding_mentions_pipeline_job(text, job):
            return job
    return ""


def looks_like_pipeline_finding(category: str, text: str) -> bool:
    if category in PIPELINE_DUPLICATE_CATEGORIES:
        return True
    pipeline_terms = (
        "github actions job",
        "pipeline",
        "failed job",
        "failed check",
        "pre-commit",
        "precommit",
        "hook id:",
        "detect-secrets",
        "semantic-release",
        "connecttimeout",
        "app installation failed",
    )
    return any(term in text for term in pipeline_terms)


def finding_mentions_pipeline_job(text: str, job: str) -> bool:
    aliases = TARGET_PIPELINE_JOB_ALIASES.get(job, (job,))
    return any(alias in text for alias in aliases)


def pipeline_duplicate_match_text(finding: dict[str, Any]) -> str:
    fields = (
        "title",
        "category",
        "finding_category",
        "review_area",
        "code_reference",
        "evidence",
        "observable_failure",
        "root_cause",
        "why_it_matters",
        "suggested_fix",
        "url",
    )
    return " ".join(str(finding.get(field) or "") for field in fields).lower()


def published_target_pipeline_jobs(review_input: dict[str, Any]) -> set[str]:
    ci = review_input.get("ci") if isinstance(review_input, dict) else {}
    if not isinstance(ci, dict):
        return set()
    published_jobs = ci.get("published_target_pipeline_jobs", []) or []
    return {normalized for normalized in (normalize_pipeline_job_name(job) for job in published_jobs) if normalized}


def record_successfully_published_target_pipeline_jobs(
    review_input: dict[str, Any],
    comment_plan: dict[str, Any],
    publish_result: dict[str, Any],
) -> None:
    successful_ids = {
        str(item.get("id") or "")
        for item in publish_result.get("results", []) or []
        if str(item.get("status") or "").startswith(PIPELINE_PUBLICATION_SUCCESS_STATUSES)
    }
    if not successful_ids:
        return

    jobs = published_target_pipeline_jobs(review_input)
    for comment in comment_plan.get("comments", []) or []:
        if str(comment.get("id") or "") not in successful_ids:
            continue
        job = normalize_pipeline_job_name(target_job_name_from_finding(comment))
        if job:
            jobs.add(job)

    if jobs:
        ci = review_input.get("ci")
        if not isinstance(ci, dict):
            ci = {}
            review_input["ci"] = ci
        ci["published_target_pipeline_jobs"] = sorted(jobs)


def normalize_pipeline_job_name(name: Any) -> str:
    normalized = " ".join(str(name or "").strip().lower().replace("_", "-").split())
    if normalized in {"precommit", "pre commit"}:
        return "pre-commit"
    if normalized in TARGET_PIPELINE_JOB_ALIASES:
        return normalized
    for job, aliases in TARGET_PIPELINE_JOB_ALIASES.items():
        if normalized in aliases:
            return job
    return normalized


def conservatively_downgrade_low_confidence_pipeline_findings(
    deterministic_findings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        conservatively_downgrade_pipeline_finding(finding)
        if isinstance(finding, dict)
        and str(finding.get("category") or "") == "ci_pipeline_failure"
        and finding.get("ci_diagnosis_needs_model") is True
        else finding
        for finding in deterministic_findings
    ]


def target_job_name_from_finding(finding: dict[str, Any]) -> str:
    code_reference = str(finding.get("code_reference") or "")
    if code_reference.startswith("GitHub Actions job "):
        return code_reference.removeprefix("GitHub Actions job ").strip()
    title = str(finding.get("title") or "")
    if title.endswith(" pipeline job failed"):
        return title.removesuffix(" pipeline job failed").strip()
    return "pipeline"


def target_job_conclusion_from_evidence(evidence: str) -> str:
    for conclusion in ("failure", "timed_out", "cancelled", "action_required"):
        if f"concluded `{conclusion}`" in evidence:
            return conclusion
    return "failure"


def pipeline_evidence_with_model_diagnosis(finding: dict[str, Any], diagnosis: dict[str, Any]) -> str:
    parts = [str(finding.get("evidence") or "").strip()]
    evidence_lines = [str(line).strip() for line in diagnosis.get("evidence_lines") or [] if str(line).strip()]
    if diagnosis.get("root_cause"):
        parts.append(f"Verified root cause: {diagnosis['root_cause']}.")
    if evidence_lines:
        parts.append(f"Supporting log line(s): {' / '.join(evidence_lines[:5])}")
    return " ".join(part for part in parts if part)


def conservatively_downgrade_pipeline_finding(finding: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(finding, dict):
        return finding
    updated = dict(finding)
    job_name = target_job_name_from_finding(updated)
    updated["confidence"] = "medium"
    updated["confidence_score"] = min(float(updated.get("confidence_score") or 0.55), 0.55)
    updated["ci_diagnosis_confidence"] = "medium"
    updated["ci_diagnosis_confidence_score"] = updated["confidence_score"]
    updated["ci_diagnosis_needs_model"] = False
    updated["root_cause"] = (
        f"`{job_name}` failed, but the compact log excerpt did not prove a more specific root cause."
    )
    updated["observable_failure"] = f"The `{job_name}` job ended without a passing result."
    updated["suggested_fix"] = (
        f"Open the linked `{job_name}` job log and fix the first terminal failure summary, retry summary, "
        "or tool-specific error line shown there before rerunning the workflow."
    )
    return updated


def mark_successfully_published_markers_as_existing(
    review_input: dict[str, Any],
    comment_plan: dict[str, Any],
    publish_result: dict[str, Any],
) -> None:
    successful_ids = {
        str(item.get("id") or "")
        for item in publish_result.get("results", []) or []
        if str(item.get("status") or "").startswith(("posted", "skipped_duplicate"))
    }
    if not successful_ids:
        return
    comments = review_input.setdefault("comments", {})
    issue_comments = comments.setdefault("issue_comments", [])
    for comment in comment_plan.get("comments", []) or []:
        marker = str(comment.get("marker") or "")
        if marker and str(comment.get("id") or "") in successful_ids:
            issue_comments.append({"body": marker})


def build_model_reviewer(
    config: RuntimeConfig,
    *,
    max_tokens: int,
    progress: ProgressReporter,
) -> GatewayClaudeReviewer:
    if config.model_provider == "gateway":
        return GatewayClaudeReviewer(config, max_tokens=max_tokens, progress=progress)
    raise RuntimeError(f"Unsupported MODEL_PROVIDER '{config.model_provider}'. Use circuit or gateway.")


def make_run_dir(base: Path, repo: str, pr_number: int) -> Path:
    owner, name = repo.split("/", 1)
    return base / f"{owner}-{name}-{pr_number}"


def write_artifacts(
    run_dir: Path,
    review_input: dict[str, Any],
    review_output: dict[str, Any],
    comment: str,
    comment_plan: dict[str, Any] | None = None,
    publish_result: dict[str, Any] | None = None,
) -> None:
    write_json(run_dir / "review_input.json", review_input)
    write_json(run_dir / "review_output.json", review_output)
    routing_summary = (review_output.get("deep_review") or {}).get("model_routing_summary")
    if isinstance(routing_summary, dict):
        write_json(run_dir / "model_routing_summary.json", routing_summary)
    sdk_manifest = review_input.get("sdk_manifest") if isinstance(review_input, dict) else None
    if isinstance(sdk_manifest, dict):
        write_json(run_dir / "sdk_manifest_result.json", sdk_manifest)
        if isinstance(sdk_manifest.get("manifest"), dict):
            write_json(run_dir / "generated_manifest.json", sdk_manifest["manifest"])
    (run_dir / "comment.md").write_text(redact_text(comment), encoding="utf-8")
    if comment_plan is not None:
        write_json(run_dir / "planned_comments.json", comment_plan)
        (run_dir / "planned_comments.md").write_text(redact_text(render_comment_plan(comment_plan)), encoding="utf-8")
    if publish_result is not None:
        write_json(run_dir / "publish_result.json", publish_result)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(dump_json(redact_obj(data)) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
