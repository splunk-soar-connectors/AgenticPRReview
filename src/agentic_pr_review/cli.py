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
    review.add_argument("--deep-chunk-chars", type=int, default=60_000)
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

        if args.skip_model:
            progress("Model review skipped by --skip-model.")
            review_output = deterministic_only_output(deterministic_findings)
        else:
            stage_started = time.monotonic()
            progress(f"Model review started with provider {config.model_provider}.")
            reviewer = build_model_reviewer(config, max_tokens=args.max_model_tokens, progress=progress)
            if args.shallow:
                review_output = reviewer.review(review_input, deterministic_findings)
            else:
                review_output = reviewer.review_deep(
                    review_input,
                    deterministic_findings,
                    max_chunks=args.deep_max_chunks,
                )
            progress(f"Model review completed in {format_duration(time.monotonic() - stage_started)}.")

        stage_started = time.monotonic()
        progress("Filtering findings and building the GitHub comment plan.")
        review_output = filter_review_output_for_pr_context(review_output, review_input)
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
