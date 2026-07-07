"""Mine historical connector PRs for recurring review patterns.

This is a local development helper. It does not post to GitHub.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_pr_review.config import RuntimeConfig, load_local_env_files  # noqa: E402
from agentic_pr_review.github_client import GitHubClient, GitHubError  # noqa: E402
from agentic_pr_review.github_client_factory import build_github_client  # noqa: E402
from agentic_pr_review.json_utils import truncate_text  # noqa: E402


DEFAULT_REPOS = [
    "example-connectors/microsoftteams",
    "example-connectors/webproxy",
    "example-connectors/nmap",
    "example-connectors/mysql",
    "example-connectors/mimecast",
]

DEFAULT_ORG = "example-connectors"

PATTERNS: dict[str, tuple[str, ...]] = {
    "api_auth_correctness": (
        "oauth",
        "client credentials",
        "basic auth",
        "authorization header",
        "token",
        "test_connectivity",
        "timeout",
        "verify=false",
        "tls",
        "retry",
    ),
    "polling_checkpoint": (
        "on_poll",
        "poll now",
        "checkpoint",
        "save_state",
        "source_data_identifier",
        "dedup",
        "duplicate",
        "timestamp",
    ),
    "output_schema_mismatch": (
        "add_data",
        "output",
        "data_path",
        "summary",
        "summary_type",
        "schema",
        "data type",
        "readme output",
    ),
    "unsafe_logging": (
        "log",
        "debug_print",
        "token",
        "secret",
        "authorization",
        "header",
        "password",
        "pii",
        "payload",
        "response",
    ),
    "soar_metadata": (
        "read_only",
        "python_version",
        "app_version",
        "appid",
        "package_name",
        "app type",
        "version bump",
        "metadata",
    ),
    "pagination": (
        "pagination",
        "next page",
        "next_page",
        "limit",
        "offset",
        "page token",
        "cursor",
        "truncat",
    ),
    "validation": (
        "validate",
        "validation",
        "blank",
        "empty string",
        "allow_list",
        "cross-field",
        "urlencode",
        "quote(",
        "unencoded",
    ),
    "missing_tests": (
        "test",
        "coverage",
        "unit test",
        "mock",
        "untested",
    ),
    "docs_pr_accuracy": (
        "readme",
        "manual_readme_content",
        "release note",
        "release_notes",
        "pr body",
        "documentation",
        "doc",
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
        "merge conflict",
        "cannot be automatically merged",
        "mergeable_state",
        "blocked",
    ),
    "api_contract": (
        "api contract",
        "api docs",
        "documentation says",
        "vendor docs",
        "endpoint",
        "wrong path",
        "wrong endpoint",
        "required header",
        "required parameter",
        "response shape",
    ),
    "error_handling": (
        "try",
        "except",
        "exception",
        "actionfailure",
        "set_status",
        "app_error",
        "raw exception",
        "cleanup",
        "timeout",
    ),
    "static_sanity_tests": (
        "min phantom version",
        "min platform version",
        "action coverage",
        "action name",
        "additional logging",
        "min number log statements",
        "verbosity",
        "playbook missing",
        "integration test results are missing",
    ),
    "dependency_packaging": (
        "requirements",
        "dependencies",
        "pip",
        "wheel",
        "license",
        "notice",
        "package",
        "pypi",
    ),
    "app_mapping": (
        "appid_to_name",
        "appid_to_package_name",
        "app mapping",
        "ci-metadata",
        ".github",
        "valid_app_name_and_guid",
    ),
}

REVIEWER_WORDS = (
    "should",
    "need",
    "missing",
    "incorrect",
    "wrong",
    "bug",
    "fail",
    "break",
    "unsafe",
    "leak",
    "not handled",
    "doesn't",
    "does not",
    "please",
    "can you",
    "could you",
    "nit:",
    "question:",
    "suggestion:",
)

BOT_LOGIN_MARKERS = (
    "[bot]",
    "dependabot",
    "github-actions",
    "renovate",
    "pre-commit-ci",
    "sonarcloud",
    "snyk",
)

COMMENT_PATTERN_RULES: tuple[dict[str, Any], ...] = (
    {
        "id": "app_mapping_missing",
        "category": "app_mapping",
        "title": "Missing app-id/app-name/package mapping",
        "regex": r"(appid_to_name|appid_to_package_name|app_package_name|valid_app_name_and_guid|app mappings?|guid and app)",
        "implementation_hint": "Check new app JSON files and CI hook logs for required .github/ci-metadata mapping updates.",
    },
    {
        "id": "oauth_basic_auth_contract",
        "category": "api_auth_correctness",
        "title": "OAuth/client-credentials flow does not match API contract",
        "regex": r"(authorization:\s*basic|basic\s+<base64|client credentials|client_id.*client_secret|token creation flow|grant_type=client_credentials)",
        "implementation_hint": "Check token requests for vendor-required Basic auth, body fields, timeout, and test_connectivity path.",
    },
    {
        "id": "mutating_read_only",
        "category": "soar_metadata",
        "title": "Mutating action marked read_only",
        "regex": r"(read_only.*false|should be false|mutating action|post/patch/delete|delete.*read_only)",
        "implementation_hint": "Check app JSON actions whose identifiers imply create/update/delete/post/patch/put but read_only is true.",
    },
    {
        "id": "product_version_vendor",
        "category": "soar_metadata",
        "title": "Product version copied from SOAR app version",
        "regex": r"(version of .* not the soar app|not the soar app'?s version|product_version|vendor version|version.*tested)",
        "implementation_hint": "Check app JSON product_version against app_version and docs/release metadata.",
    },
    {
        "id": "release_note_missing",
        "category": "docs_pr_accuracy",
        "title": "Release note missing or stale",
        "regex": r"(add release note|release_notes/unreleased|release note.*missing|missing release note)",
        "implementation_hint": "Check behavior/code/metadata changes have release_notes/unreleased.md unless explicitly exempt.",
    },
    {
        "id": "precommit_failure_specific",
        "category": "precommit",
        "title": "Specific pre-commit/static hook failure",
        "regex": r"(pre-commit|failed -|hook id:|ruff|semgrep|detect-secrets|build-docs|check-json|check-yaml|mdformat)",
        "implementation_hint": "Parse CI/comment hook output and post only named hook/file-level fixes.",
    },
    {
        "id": "unsafe_logging_sensitive",
        "category": "unsafe_logging",
        "title": "Sensitive value logged",
        "regex": r"(logs? secrets?|token exposure|cleartext logging|logs?.*(token|secret|password|authorization|payload|response|tenant|pii)|remove this log.*(token|secret|password|authorization|payload|response))",
        "implementation_hint": "Check added logging of params, headers, tokens, payloads, full responses, tenant data, or PII.",
    },
    {
        "id": "compile_or_import_failure",
        "category": "precommit",
        "title": "Python compile/import failure",
        "regex": r"(syntaxerror|importerror|nameerror|undefined|not defined|py_compile|compile failure|could not compile)",
        "implementation_hint": "Parse changed Python files and CI logs for syntax/import failures with exact file and line evidence.",
    },
    {
        "id": "minimum_platform_static_failure",
        "category": "static_sanity_tests",
        "title": "Minimum platform/min_phantom_version static failure",
        "regex": r"(min(?:imum)? (?:platform|phantom) version|min_phantom_version)",
        "implementation_hint": "Parse CI/comment failures for minimum platform metadata and suggest updating app JSON plus regenerated docs.",
    },
    {
        "id": "generated_docs_stale",
        "category": "docs_pr_accuracy",
        "title": "Generated docs or manual content are stale",
        "regex": r"(build-docs|manual_readme_content|readme file didn'?t run|generated docs?|run pre-commit.*readme|docs hook)",
        "implementation_hint": "Use CI build-docs output and changed manual/app JSON files to point contributors at the source doc input, not generated README.md.",
    },
    {
        "id": "logging_coverage_static_failure",
        "category": "static_sanity_tests",
        "title": "Static tests require useful connector logging",
        "regex": r"(min number log statements|additional logging|action implementations should have at least 2 statements|verbosity)",
        "implementation_hint": "Parse static-test output for action names and suggest targeted debug_print/error_print/save_progress/send_progress updates.",
    },
    {
        "id": "action_test_playbook_missing",
        "category": "missing_tests",
        "title": "Action test playbook or integration-test artifact is missing",
        "regex": r"(playbook missing|test playbook|app-tests|apps-test-playbooks|integration test results are missing)",
        "implementation_hint": "Parse static/integration-test output for missing playbook/result artifacts and comment only when the missing artifact is named.",
    },
    {
        "id": "oauth_content_type_contract",
        "category": "api_contract",
        "title": "OAuth/API request content type does not match vendor contract",
        "regex": r"(application/x-www-form-urlencoded|form-urlencoded|content[- ]type|json_data)",
        "implementation_hint": "For auth/token endpoints, compare json/json_data usage against vendor-required form-encoded bodies and headers.",
    },
    {
        "id": "pagination_single_get",
        "category": "pagination",
        "title": "Paginated API only fetches one page",
        "regex": r"(handle pagination|endpoint.*paginated|paginated.*single|get.*only.*first page|only does a single get|next_page)",
        "implementation_hint": "Check list/search actions and reviewer-mentioned endpoints for page/offset/cursor traversal.",
    },
    {
        "id": "output_schema_mismatch",
        "category": "output_schema_mismatch",
        "title": "Python output/docs do not match app JSON",
        "regex": r"(add_data|output.*not.*json|not declared|summary_type|summary.*missing|readme.*output|output fields?)",
        "implementation_hint": "Compare add_data/update_summary/SDK outputs and docs claims against app JSON output paths/types.",
    },
    {
        "id": "polling_sdi_dedup",
        "category": "polling_checkpoint",
        "title": "Polling SDI/checkpoint design can duplicate or skip events",
        "regex": r"(source_data_identifier|sdi|dedup|duplicate container|checkpoint|same timestamp|last_.*id|poll now.*state)",
        "implementation_hint": "Check SDI stability, same-boundary checkpointing, Poll Now state isolation, and save_state timing.",
    },
    {
        "id": "poll_now_params",
        "category": "polling_checkpoint",
        "title": "Poll Now/on_poll parameters are misleading or unused",
        "regex": r"(poll now|on_poll parameters?|start_time_poll_now|default parameters|asset settings page|lookback)",
        "implementation_hint": "Check declared on_poll params, workaround asset params, absolute lookback aging, and docs/tests.",
    },
    {
        "id": "container_lookup_scaling",
        "category": "polling_checkpoint",
        "title": "Polling container lookup scales with source group cardinality",
        "regex": r"(/rest/container|container lookup|rule groups?|rule_id|rate limit|deterministic sdi|bounded container query)",
        "implementation_hint": "Check on_poll loops that query SOAR containers once per rule/group/event instead of using a bounded lookup.",
    },
    {
        "id": "missing_tests",
        "category": "missing_tests",
        "title": "Risky path missing test coverage",
        "regex": r"(test coverage|action coverage|add tests?|missing tests?|unit tests?|no tests?|untested)",
        "implementation_hint": "Check risky auth/session/polling/state/validation/mutating changes for targeted tests or explicit test evidence.",
    },
    {
        "id": "docs_default_or_private",
        "category": "docs_pr_accuracy",
        "title": "Docs/defaults expose wrong or private behavior",
        "regex": r"(default value.*incorrect|private.*readme|readme file|manual_readme_content|docs.*wrong|not be mentioned)",
        "implementation_hint": "Compare docs/manual/app JSON defaults and hide private-only params unless documented as user-facing.",
    },
    {
        "id": "validation_gap",
        "category": "validation",
        "title": "Validation misses invalid/blank/cross-field input",
        "regex": r"(input validation|validate (this|that|the input|parameter)|blank strings?|empty strings?|cross-field|url encode|urlencode|quote\(|malformed input|invalid (input|value|format|parameter))",
        "implementation_hint": "Check blank string rejection, allow_list typing, cross-field validation, and URL path encoding.",
    },
    {
        "id": "dependency_license_packaging",
        "category": "dependency_packaging",
        "title": "Dependency/license/package metadata drift",
        "regex": r"(license|notice|requirements\.txt|dependency|dependencies|wheel|pypi|semgrep hook version)",
        "implementation_hint": "Check dependency manifests, wheel changes, license/NOTICE updates, and hook-version compatibility.",
    },
    {
        "id": "error_handling_cleanup",
        "category": "error_handling",
        "title": "Error path misses cleanup or connector-friendly failure",
        "regex": r"(try/except|except block|cleanup path|actionfailure|raw exception|raise.*actionfailure|error path)",
        "implementation_hint": "Check risky operations are inside try/except, cleanup runs, and failures return ActionFailure/status messages.",
    },
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Mine connector PRs for review-training patterns.")
    parser.add_argument("repos", nargs="*", default=None)
    parser.add_argument("--org", default=DEFAULT_ORG, help="GitHub org used with --all-org-repos.")
    parser.add_argument("--all-org-repos", action="store_true", help="Discover and mine every repo installed/visible in the org.")
    parser.add_argument("--include-archived", action="store_true", help="Include archived repos when discovering org repos.")
    parser.add_argument("--repo-limit", type=int, default=0, help="Limit discovered repos for validation runs. 0 means no limit.")
    parser.add_argument(
        "--pr-ref",
        action="append",
        default=[],
        help="Mine an exact PR in owner/repo#number form. Can be passed multiple times.",
    )
    parser.add_argument("--max-prs-per-repo", type=int, default=0, help="0 means all PRs returned by GitHub.")
    parser.add_argument("--max-pr-pages-per-repo", type=int, default=30, help="100 PRs per page. Increase for deep history.")
    parser.add_argument("--max-comment-pages-per-repo", type=int, default=100, help="100 comments per page for repo-wide comment scans.")
    parser.add_argument("--max-comment-samples", type=int, default=12)
    parser.add_argument(
        "--comment-scan-mode",
        choices=("repo", "pr"),
        default="repo",
        help="repo uses repo-wide comment APIs for org-scale mining; pr fetches comments per PR.",
    )
    parser.add_argument(
        "--enrich-matched-prs",
        action="store_true",
        help="For PRs with human-review matches, also fetch changed-file summaries and check-run/status metadata.",
    )
    parser.add_argument(
        "--max-enriched-prs-per-repo",
        type=int,
        default=0,
        help="Limit enriched matched PRs per repo. 0 means no limit.",
    )
    parser.add_argument("--output-dir", default="runs/training-mining")
    parser.add_argument(
        "--auth-mode",
        choices=("app", "auto"),
        default="auto",
        help="Authentication mode for this mining run. Only GitHub App/API auth is supported.",
    )
    args = parser.parse_args()

    client, resolved_auth_mode = build_training_client(args.auth_mode)
    repos = resolve_repos(client, args)
    started = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(args.output_dir) / started
    output_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "generated_at": started,
        "auth_mode": resolved_auth_mode,
        "repo_count_requested": len(repos),
        "exact_pr_refs_requested": list(args.pr_ref),
        "repos": {},
        "category_counts": Counter(),
        "review_comment_category_counts": Counter(),
        "comment_pattern_counts": Counter(),
        "comment_pattern_examples": defaultdict(list),
        "reviewer_counts": Counter(),
        "examples": [],
        "training_examples": [],
        "top_comment_terms": Counter(),
    }

    for repo in repos:
        print(f"mining repo {repo}", file=sys.stderr, flush=True)
        repo_result = mine_repo(
            client,
            repo,
            max_prs=args.max_prs_per_repo,
            max_pr_pages=args.max_pr_pages_per_repo,
            max_comment_pages=args.max_comment_pages_per_repo,
            max_comment_samples=args.max_comment_samples,
            comment_scan_mode=args.comment_scan_mode,
            enrich_matched_prs=args.enrich_matched_prs,
            max_enriched_prs_per_repo=args.max_enriched_prs_per_repo,
        )
        result["repos"][repo] = repo_result
        result["category_counts"].update(repo_result["category_counts"])
        result["review_comment_category_counts"].update(repo_result["review_comment_category_counts"])
        result["comment_pattern_counts"].update(repo_result["comment_pattern_counts"])
        merge_pattern_examples(result["comment_pattern_examples"], repo_result["comment_pattern_examples"])
        result["reviewer_counts"].update(repo_result["reviewer_counts"])
        result["top_comment_terms"].update(repo_result["top_comment_terms"])
        result["examples"].extend(repo_result["examples"])
        result["training_examples"].extend(repo_result["training_examples"])

    for repo, number in parse_pr_refs(args.pr_ref):
        print(f"mining exact PR {repo}#{number}", file=sys.stderr, flush=True)
        pr_result = mine_exact_pr(client, repo, number, max_comment_samples=args.max_comment_samples)
        if pr_result.get("error"):
            result["repos"].setdefault(
                repo,
                {
                    "pr_count_scanned": 0,
                    "category_counts": {},
                    "review_comment_category_counts": {},
                    "reviewer_counts": {},
                    "top_comment_terms": {},
                    "examples": [],
                    "errors": [],
                },
            )["errors"].append(f"PR {number}: {pr_result['error']}")
            continue
        result["category_counts"].update(pr_result["category_counts"])
        result["review_comment_category_counts"].update(pr_result["review_comment_category_counts"])
        result["comment_pattern_counts"].update(pr_result["comment_pattern_counts"])
        merge_pattern_examples(result["comment_pattern_examples"], pr_result["comment_pattern_examples"])
        result["reviewer_counts"].update(pr_result["reviewer_counts"])
        result["top_comment_terms"].update(pr_result["top_comment_terms"])
        if pr_result.get("example"):
            result["examples"].append(pr_result["example"])
        result["training_examples"].extend(pr_result.get("training_examples") or [])

    serializable = {
        **result,
        "category_counts": dict(result["category_counts"]),
        "review_comment_category_counts": dict(result["review_comment_category_counts"]),
        "comment_pattern_counts": dict(result["comment_pattern_counts"].most_common()),
        "comment_pattern_examples": dict(result["comment_pattern_examples"]),
        "reviewer_counts": dict(result["reviewer_counts"].most_common(50)),
        "training_examples": result["training_examples"],
        "top_comment_terms": dict(result["top_comment_terms"].most_common(75)),
    }
    (output_dir / "mined_pr_patterns.json").write_text(json.dumps(serializable, indent=2, sort_keys=True), encoding="utf-8")
    write_training_jsonl(output_dir / "training_examples.jsonl", result["training_examples"])
    (output_dir / "implementation_candidates.md").write_text(render_implementation_candidates(serializable), encoding="utf-8")
    (output_dir / "mined_pr_patterns.md").write_text(render_markdown(serializable), encoding="utf-8")
    print(output_dir)
    return 0


def build_training_client(auth_mode: str) -> tuple[GitHubClient, str]:
    load_local_env_files()
    resolved = resolve_training_auth_mode(auth_mode)
    config = RuntimeConfig.from_env()
    return build_github_client(config), "app"


def resolve_training_auth_mode(auth_mode: str) -> str:
    if auth_mode in {"app", "auto"}:
        return "app"
    raise RuntimeError("Only GitHub App/API auth is supported.")


def resolve_repos(client: GitHubClient, args: argparse.Namespace) -> list[str]:
    if args.all_org_repos:
        repos = list_org_repos(client, args.org, include_archived=args.include_archived)
    elif args.repos:
        repos = list(args.repos)
    elif args.pr_ref:
        repos = []
    else:
        repos = list(DEFAULT_REPOS)
    if args.repo_limit > 0:
        repos = repos[: args.repo_limit]
    return repos


def parse_pr_refs(values: list[str]) -> list[tuple[str, int]]:
    refs = []
    for value in values:
        match = re.fullmatch(r"(?P<repo>[^#]+/[^#]+)#(?P<number>\d+)", value.strip())
        if not match:
            raise ValueError(f"Invalid --pr-ref {value!r}; expected owner/repo#number")
        refs.append((match.group("repo"), int(match.group("number"))))
    return refs


def list_org_repos(client: GitHubClient, org: str, *, include_archived: bool) -> list[str]:
    repos = client.paginate(
        f"/orgs/{org}/repos",
        params={"type": "all", "sort": "full_name", "direction": "asc"},
        limit_pages=50,
    )
    output = []
    for repo in repos:
        if not include_archived and repo.get("archived"):
            continue
        full_name = str(repo.get("full_name") or "")
        if full_name:
            output.append(full_name)
    return sorted(output)


def mine_repo(
    client: GitHubClient,
    repo: str,
    *,
    max_prs: int,
    max_pr_pages: int,
    max_comment_pages: int,
    max_comment_samples: int,
    comment_scan_mode: str,
    enrich_matched_prs: bool,
    max_enriched_prs_per_repo: int,
) -> dict[str, Any]:
    if comment_scan_mode == "repo":
        return mine_repo_with_repo_comment_indexes(
            client,
            repo,
            max_prs=max_prs,
            max_pr_pages=max_pr_pages,
            max_comment_pages=max_comment_pages,
            max_comment_samples=max_comment_samples,
            enrich_matched_prs=enrich_matched_prs,
            max_enriched_prs_per_repo=max_enriched_prs_per_repo,
        )

    try:
        pulls = client.paginate(
            f"/repos/{repo}/pulls",
            params={"state": "all", "sort": "updated", "direction": "desc"},
            limit_pages=max_pr_pages,
        )
    except GitHubError as exc:
        return empty_repo_result(error=f"list PRs: {exc}")
    if max_prs > 0:
        pulls = pulls[:max_prs]

    category_counts: Counter[str] = Counter()
    review_comment_category_counts: Counter[str] = Counter()
    comment_pattern_counts: Counter[str] = Counter()
    comment_pattern_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    reviewer_counts: Counter[str] = Counter()
    top_comment_terms: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    training_examples: list[dict[str, Any]] = []
    errors: list[str] = []
    enriched_count = 0

    for pr in pulls:
        number = int(pr.get("number") or 0)
        if number <= 0:
            continue
        try:
            issue_comments = client.list_issue_comments(repo, number)
            review_comments = client.list_review_comments(repo, number)
            reviews = client.list_reviews(repo, number)
            files = client.list_pr_files(repo, number)
        except GitHubError as exc:
            errors.append(f"PR {number}: {exc}")
            continue

        comments = compact_pr_text(issue_comments, review_comments, reviews)
        human_comments = [comment for comment in comments if is_human_comment(comment)]
        reviewer_counts.update(str(comment.get("user") or "unknown") for comment in human_comments)
        patch_text = "\n".join(str(item.get("patch") or "") for item in files)
        categories = classify_pr(pr, human_comments, patch_text)
        comment_categories = classify_comments(human_comments)
        categories = sorted(set(categories) | set(comment_categories))
        should_enrich = enrich_matched_prs and (categories or match_any_comment_pattern(human_comments))
        pr_evidence = enrich_pr_evidence(client, repo, pr, files=files) if should_enrich else None
        if pr_evidence:
            enriched_count += 1
        pattern_matches = collect_training_examples(repo, pr, human_comments, pr_evidence=pr_evidence)
        update_pattern_summaries(pattern_matches, comment_pattern_counts, comment_pattern_examples)
        training_examples.extend(pattern_matches)
        if not categories:
            continue

        category_counts.update(categories)
        review_comment_category_counts.update(comment_categories)
        top_comment_terms.update(extract_comment_terms(human_comments))
        samples = extract_samples(human_comments, categories, max_samples=max_comment_samples)
        examples.append(
            {
                "repo": repo,
                "number": number,
                "title": pr.get("title"),
                "state": pr.get("state"),
                "merged_at": pr.get("merged_at"),
                "url": pr.get("html_url"),
                "categories": categories,
                "human_review_comment_categories": comment_categories,
                "human_review_comment_count": len(human_comments),
                "changed_files": [item.get("filename") for item in files[:30]],
                "check_evidence": (pr_evidence or {}).get("check_evidence", []),
                "samples": samples,
            }
        )
        if max_enriched_prs_per_repo > 0 and enriched_count >= max_enriched_prs_per_repo:
            enrich_matched_prs = False

    return {
        "pr_count_scanned": len(pulls),
        "enriched_pr_count": enriched_count,
        "category_counts": dict(category_counts),
        "review_comment_category_counts": dict(review_comment_category_counts),
        "comment_pattern_counts": dict(comment_pattern_counts),
        "comment_pattern_examples": dict(comment_pattern_examples),
        "reviewer_counts": dict(reviewer_counts.most_common(25)),
        "training_examples": training_examples,
        "top_comment_terms": dict(top_comment_terms.most_common(50)),
        "examples": examples,
        "errors": errors,
    }


def mine_repo_with_repo_comment_indexes(
    client: GitHubClient,
    repo: str,
    *,
    max_prs: int,
    max_pr_pages: int,
    max_comment_pages: int,
    max_comment_samples: int,
    enrich_matched_prs: bool,
    max_enriched_prs_per_repo: int,
) -> dict[str, Any]:
    try:
        pulls = client.paginate(
            f"/repos/{repo}/pulls",
            params={"state": "all", "sort": "updated", "direction": "desc"},
            limit_pages=max_pr_pages,
        )
    except GitHubError as exc:
        return empty_repo_result(error=f"list PRs: {exc}")
    if max_prs > 0:
        pulls = pulls[:max_prs]

    pr_by_number = {int(pr.get("number") or 0): pr for pr in pulls if int(pr.get("number") or 0) > 0}
    if not pr_by_number:
        return empty_repo_result(pr_count=0)

    try:
        issue_comments = client.paginate(f"/repos/{repo}/issues/comments", limit_pages=max_comment_pages)
        review_comments = client.paginate(f"/repos/{repo}/pulls/comments", limit_pages=max_comment_pages)
    except GitHubError as exc:
        return empty_repo_result(pr_count=len(pulls), error=f"list repo comments: {exc}")

    comments_by_pr: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in issue_comments:
        number = pr_number_from_url(str(item.get("issue_url") or ""))
        if number not in pr_by_number:
            continue
        comments_by_pr[number].append(compact_issue_comment(item))
    for item in review_comments:
        number = pr_number_from_url(str(item.get("pull_request_url") or ""))
        if number not in pr_by_number:
            continue
        comments_by_pr[number].append(compact_review_comment(item))

    category_counts: Counter[str] = Counter()
    review_comment_category_counts: Counter[str] = Counter()
    comment_pattern_counts: Counter[str] = Counter()
    comment_pattern_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    reviewer_counts: Counter[str] = Counter()
    top_comment_terms: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    training_examples: list[dict[str, Any]] = []
    errors: list[str] = []
    enriched_count = 0

    for number, pr in pr_by_number.items():
        comments = comments_by_pr.get(number, [])
        human_comments = [comment for comment in comments if is_human_comment(comment)]
        if not human_comments:
            continue
        reviewer_counts.update(str(comment.get("user") or "unknown") for comment in human_comments)
        categories = classify_pr(pr, human_comments, patch_text="")
        comment_categories = classify_comments(human_comments)
        categories = sorted(set(categories) | set(comment_categories))
        pr_evidence = None
        should_enrich = enrich_matched_prs and (categories or match_any_comment_pattern(human_comments))
        if should_enrich:
            try:
                pr_evidence = enrich_pr_evidence(client, repo, pr)
                enriched_count += 1
                patch_text = "\n".join(str(item.get("patch") or "") for item in pr_evidence.get("raw_files", []))
                if patch_text:
                    categories = sorted(set(categories) | set(classify_pr(pr, human_comments, patch_text)))
            except GitHubError as exc:
                errors.append(f"PR {number}: enrich evidence: {exc}")
        pattern_matches = collect_training_examples(repo, pr, human_comments, pr_evidence=pr_evidence)
        update_pattern_summaries(pattern_matches, comment_pattern_counts, comment_pattern_examples)
        training_examples.extend(pattern_matches)
        if not categories and not pattern_matches:
            continue
        category_counts.update(categories)
        review_comment_category_counts.update(comment_categories)
        top_comment_terms.update(extract_comment_terms(human_comments))
        samples = extract_samples(human_comments, categories, max_samples=max_comment_samples)
        examples.append(
            {
                "repo": repo,
                "number": number,
                "title": pr.get("title"),
                "state": pr.get("state"),
                "merged_at": pr.get("merged_at"),
                "url": pr.get("html_url"),
                "categories": categories,
                "human_review_comment_categories": comment_categories,
                "human_review_comment_count": len(human_comments),
                "changed_files": [item.get("filename") for item in (pr_evidence or {}).get("changed_files", [])[:30]],
                "check_evidence": (pr_evidence or {}).get("check_evidence", []),
                "samples": samples,
            }
        )
        if max_enriched_prs_per_repo > 0 and enriched_count >= max_enriched_prs_per_repo:
            enrich_matched_prs = False

    return {
        "pr_count_scanned": len(pulls),
        "comment_count_scanned": len(issue_comments) + len(review_comments),
        "enriched_pr_count": enriched_count,
        "category_counts": dict(category_counts),
        "review_comment_category_counts": dict(review_comment_category_counts),
        "comment_pattern_counts": dict(comment_pattern_counts),
        "comment_pattern_examples": dict(comment_pattern_examples),
        "reviewer_counts": dict(reviewer_counts.most_common(25)),
        "training_examples": training_examples,
        "top_comment_terms": dict(top_comment_terms.most_common(50)),
        "examples": examples,
        "errors": errors,
    }


def empty_repo_result(*, pr_count: int = 0, error: str | None = None) -> dict[str, Any]:
    return {
        "pr_count_scanned": pr_count,
        "comment_count_scanned": 0,
        "enriched_pr_count": 0,
        "category_counts": {},
        "review_comment_category_counts": {},
        "comment_pattern_counts": {},
        "comment_pattern_examples": {},
        "reviewer_counts": {},
        "training_examples": [],
        "top_comment_terms": {},
        "examples": [],
        "errors": [error] if error else [],
    }


def mine_exact_pr(
    client: GitHubClient,
    repo: str,
    number: int,
    *,
    max_comment_samples: int,
) -> dict[str, Any]:
    try:
        pr = client.get_pr(repo, number)
        issue_comments = client.list_issue_comments(repo, number)
        review_comments = client.list_review_comments(repo, number)
        reviews = client.list_reviews(repo, number)
        files = client.list_pr_files(repo, number)
    except GitHubError as exc:
        return {"error": str(exc)}

    comments = compact_pr_text(issue_comments, review_comments, reviews)
    human_comments = [comment for comment in comments if is_human_comment(comment)]
    patch_text = "\n".join(str(item.get("patch") or "") for item in files)
    categories = classify_pr(pr, human_comments, patch_text)
    comment_categories = classify_comments(human_comments)
    categories = sorted(set(categories) | set(comment_categories))
    reviewer_counts: Counter[str] = Counter(str(comment.get("user") or "unknown") for comment in human_comments)
    top_comment_terms = extract_comment_terms(human_comments)
    pattern_matches = collect_training_examples(repo, pr, human_comments)
    comment_pattern_counts: Counter[str] = Counter()
    comment_pattern_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    update_pattern_summaries(pattern_matches, comment_pattern_counts, comment_pattern_examples)
    samples = extract_samples(human_comments, categories, max_samples=max_comment_samples)
    example = {
        "repo": repo,
        "number": number,
        "title": pr.get("title"),
        "state": pr.get("state"),
        "merged_at": pr.get("merged_at"),
        "url": pr.get("html_url"),
        "categories": categories,
        "human_review_comment_categories": comment_categories,
        "human_review_comment_count": len(human_comments),
        "changed_files": [item.get("filename") for item in files[:30]],
        "samples": samples,
    }
    return {
        "category_counts": dict(Counter(categories)),
        "review_comment_category_counts": dict(Counter(comment_categories)),
        "comment_pattern_counts": dict(comment_pattern_counts),
        "comment_pattern_examples": dict(comment_pattern_examples),
        "reviewer_counts": dict(reviewer_counts.most_common(25)),
        "training_examples": pattern_matches,
        "top_comment_terms": dict(top_comment_terms.most_common(50)),
        "example": example,
    }


def compact_pr_text(
    issue_comments: list[dict[str, Any]],
    review_comments: list[dict[str, Any]],
    reviews: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for item in issue_comments:
        output.append(compact_issue_comment(item))
    for item in review_comments:
        output.append(compact_review_comment(item))
    for item in reviews:
        body = str(item.get("body") or "")
        if body.strip():
            output.append(
                {
                    "kind": "review",
                    "path": None,
                    "line": None,
                    "body": body,
                    "user": ((item.get("user") or {}).get("login")),
                    "url": item.get("html_url"),
                }
            )
    return output


def compact_issue_comment(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": "issue_comment",
        "path": None,
        "line": None,
        "body": str(item.get("body") or ""),
        "diff_hunk": None,
        "user": ((item.get("user") or {}).get("login")),
        "url": item.get("html_url"),
    }


def compact_review_comment(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": "review_comment",
        "path": item.get("path"),
        "line": item.get("line") or item.get("original_line"),
        "body": str(item.get("body") or ""),
        "diff_hunk": item.get("diff_hunk"),
        "user": ((item.get("user") or {}).get("login")),
        "url": item.get("html_url"),
    }


def pr_number_from_url(url: str) -> int | None:
    match = re.search(r"/(?:issues|pulls)/(?P<number>\d+)$", url)
    if not match:
        return None
    return int(match.group("number"))


def is_human_comment(comment: dict[str, Any]) -> bool:
    user = str(comment.get("user") or "").lower()
    if not user:
        return True
    return not any(marker in user for marker in BOT_LOGIN_MARKERS)


def classify_pr(pr: dict[str, Any], comments: list[dict[str, Any]], patch_text: str) -> list[str]:
    review_text = "\n".join(comment["body"] for comment in comments)
    text = f"{pr.get('title') or ''}\n{pr.get('body') or ''}\n{review_text}\n{patch_text}".lower()
    categories = []
    for category, needles in PATTERNS.items():
        if any(needle in text for needle in needles):
            if category in {"unsafe_logging", "missing_tests", "docs_pr_accuracy"} and not looks_actionable(review_text):
                continue
            categories.append(category)
    return categories


def classify_comments(comments: list[dict[str, Any]]) -> list[str]:
    categories = set()
    for comment in comments:
        body = str(comment.get("body") or "").lower()
        if not looks_actionable(body):
            continue
        for category, needles in PATTERNS.items():
            if any(needle in body for needle in needles):
                categories.add(category)
    return sorted(categories)


def collect_training_examples(
    repo: str,
    pr: dict[str, Any],
    comments: list[dict[str, Any]],
    *,
    pr_evidence: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    pr_number = int(pr.get("number") or 0)
    for comment in comments:
        if not is_human_comment(comment):
            continue
        body = str(comment.get("body") or "")
        if not looks_actionable(body):
            continue
        matches = match_comment_patterns(body)
        if not matches:
            continue
        for rule in matches:
            examples.append(
                {
                    "pattern_id": rule["id"],
                    "category": rule["category"],
                    "pattern_title": rule["title"],
                    "implementation_hint": rule["implementation_hint"],
                    "repo": repo,
                    "pr_number": pr_number,
                    "pr_title": pr.get("title"),
                    "pr_state": pr.get("state"),
                    "pr_url": pr.get("html_url"),
                    "comment_kind": comment.get("kind"),
                    "path": comment.get("path"),
                    "line": comment.get("line"),
                    "user": comment.get("user"),
                    "url": comment.get("url"),
                    "body": truncate_text(" ".join(body.split()), 1800),
                    "diff_hunk": truncate_text(str(comment.get("diff_hunk") or ""), 1200),
                    "changed_files": [item.get("filename") for item in (pr_evidence or {}).get("changed_files", [])[:30]],
                    "file_evidence": select_file_evidence(comment, pr_evidence),
                    "check_evidence": (pr_evidence or {}).get("check_evidence", []),
                }
            )
    return examples


def match_any_comment_pattern(comments: list[dict[str, Any]]) -> bool:
    for comment in comments:
        body = str(comment.get("body") or "")
        if looks_actionable(body) and match_comment_patterns(body):
            return True
    return False


def enrich_pr_evidence(
    client: GitHubClient,
    repo: str,
    pr: dict[str, Any],
    *,
    files: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    number = int(pr.get("number") or 0)
    raw_files = files if files is not None else client.list_pr_files(repo, number)
    head_sha = str(((pr.get("head") or {}).get("sha")) or "")
    return {
        "changed_files": [compact_mined_file(item) for item in raw_files],
        "raw_files": raw_files,
        "check_evidence": collect_check_evidence(client, repo, head_sha) if head_sha else [],
    }


def compact_mined_file(item: dict[str, Any]) -> dict[str, Any]:
    patch = str(item.get("patch") or "")
    return {
        "filename": item.get("filename"),
        "status": item.get("status"),
        "additions": item.get("additions"),
        "deletions": item.get("deletions"),
        "changes": item.get("changes"),
        "patch_excerpt": truncate_text(patch, 1200),
    }


def collect_check_evidence(client: GitHubClient, repo: str, head_sha: str) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    try:
        runs_data = client.get_check_runs(repo, head_sha)
        for run in (runs_data or {}).get("check_runs", [])[:100]:
            conclusion = str(run.get("conclusion") or "").lower()
            if conclusion not in {"failure", "timed_out", "cancelled", "action_required"}:
                continue
            output = run.get("output") or {}
            evidence.append(
                {
                    "kind": "check_run",
                    "name": run.get("name"),
                    "conclusion": conclusion,
                    "details_url": run.get("details_url") or run.get("html_url"),
                    "summary": truncate_text(str(output.get("summary") or output.get("title") or ""), 700),
                }
            )
    except GitHubError as exc:
        evidence.append({"kind": "check_error", "error": truncate_text(str(exc), 500)})

    try:
        status_data = client.get_combined_status(repo, head_sha)
        for status in (status_data or {}).get("statuses", [])[:100]:
            state = str(status.get("state") or "").lower()
            if state not in {"failure", "error"}:
                continue
            evidence.append(
                {
                    "kind": "status",
                    "name": status.get("context"),
                    "conclusion": state,
                    "details_url": status.get("target_url"),
                    "summary": truncate_text(str(status.get("description") or ""), 700),
                }
            )
    except GitHubError as exc:
        evidence.append({"kind": "status_error", "error": truncate_text(str(exc), 500)})

    return evidence[:12]


def select_file_evidence(comment: dict[str, Any], pr_evidence: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not pr_evidence:
        return []
    files = [item for item in pr_evidence.get("changed_files", []) if isinstance(item, dict)]
    if not files:
        return []

    comment_path = str(comment.get("path") or "")
    if comment_path:
        exact = [item for item in files if item.get("filename") == comment_path]
        if exact:
            return exact[:1]

    body = str(comment.get("body") or "")
    keywords = evidence_keywords(body)
    if not keywords:
        return []
    scored: list[tuple[int, dict[str, Any]]] = []
    for item in files:
        haystack = f"{item.get('filename') or ''}\n{item.get('patch_excerpt') or ''}".lower()
        score = sum(1 for keyword in keywords if keyword in haystack)
        if score:
            scored.append((score, item))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [item for _, item in scored[:3]]


def evidence_keywords(text: str) -> list[str]:
    candidates = []
    candidates.extend(re.findall(r"`([^`]{3,80})`", text))
    candidates.extend(re.findall(r"\b[a-z][a-z0-9]+(?:_[a-z0-9]+)+\b", text.lower()))
    candidates.extend(re.findall(r"\b[a-z0-9_.-]+\.(?:py|json|ya?ml|toml|md)\b", text.lower()))
    ignored = {"action_result", "source_data_identifier", "release_notes", "manual_readme_content"}
    output = []
    seen = set()
    for candidate in candidates:
        keyword = candidate.strip().lower()
        if not keyword or keyword in ignored or len(keyword) > 80:
            continue
        if " " in keyword and not keyword.endswith((".py", ".json", ".yaml", ".yml", ".toml", ".md")):
            continue
        if keyword in seen:
            continue
        seen.add(keyword)
        output.append(keyword)
    return output[:12]


def match_comment_patterns(body: str) -> list[dict[str, Any]]:
    matched = []
    lowered = body.lower()
    for rule in COMMENT_PATTERN_RULES:
        if re.search(str(rule["regex"]), lowered, flags=re.IGNORECASE | re.DOTALL):
            matched.append(rule)
    return matched


def update_pattern_summaries(
    examples: list[dict[str, Any]],
    counts: Counter[str],
    pattern_examples: dict[str, list[dict[str, Any]]],
    *,
    max_examples_per_pattern: int = 8,
) -> None:
    for example in examples:
        pattern_id = str(example["pattern_id"])
        counts[pattern_id] += 1
        if len(pattern_examples[pattern_id]) >= max_examples_per_pattern:
            continue
        pattern_examples[pattern_id].append(
            {
                "repo": example["repo"],
                "pr_number": example["pr_number"],
                "path": example.get("path"),
                "line": example.get("line"),
                "url": example.get("url"),
                "body": truncate_text(str(example.get("body") or ""), 500),
            }
        )


def merge_pattern_examples(
    target: dict[str, list[dict[str, Any]]],
    source: dict[str, list[dict[str, Any]]],
    *,
    max_examples_per_pattern: int = 8,
) -> None:
    for pattern_id, examples in (source or {}).items():
        for example in examples:
            if len(target[pattern_id]) >= max_examples_per_pattern:
                break
            target[pattern_id].append(example)


def looks_actionable(text: str) -> bool:
    lowered = text.lower()
    return any(word in lowered for word in REVIEWER_WORDS)


def extract_samples(comments: list[dict[str, Any]], categories: list[str], *, max_samples: int = 8) -> list[dict[str, Any]]:
    samples = []
    for comment in comments:
        body = comment["body"]
        lowered = body.lower()
        matched = [
            category
            for category in categories
            if any(needle in lowered for needle in PATTERNS[category])
        ]
        if not matched or not looks_actionable(body):
            continue
        samples.append(
            {
                "categories": matched,
                "kind": comment["kind"],
                "path": comment["path"],
                "line": comment["line"],
                "user": comment["user"],
                "url": comment["url"],
                "body": truncate_text(" ".join(body.split()), 800),
                "diff_hunk": truncate_text(str(comment.get("diff_hunk") or ""), 500),
            }
        )
        if len(samples) >= max_samples:
            break
    return samples


def extract_comment_terms(comments: list[dict[str, Any]]) -> Counter[str]:
    terms: Counter[str] = Counter()
    for comment in comments:
        body = str(comment.get("body") or "")
        if not looks_actionable(body):
            continue
        lowered = body.lower()
        for category, needles in PATTERNS.items():
            if any(needle in lowered for needle in needles):
                terms[category] += 1
        for phrase in recurring_phrases(lowered):
            terms[phrase] += 1
    return terms


def recurring_phrases(text: str) -> list[str]:
    phrases = []
    phrase_patterns = (
        r"\bmissing [a-z0-9_ -]{3,40}",
        r"\bshould [a-z0-9_ -]{3,50}",
        r"\bneed(?:s)? to [a-z0-9_ -]{3,50}",
        r"\bplease [a-z0-9_ -]{3,50}",
        r"\bnot [a-z0-9_ -]{3,40}",
    )
    for pattern in phrase_patterns:
        for match in re.findall(pattern, text):
            phrase = " ".join(match.split())
            if len(phrase) <= 80:
                phrases.append(phrase)
    return phrases[:12]


def write_training_jsonl(path: Path, examples: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example, sort_keys=True) + "\n")


def render_implementation_candidates(result: dict[str, Any]) -> str:
    lines = [
        "# Implementation Candidates From Human Review Comments",
        "",
        f"Generated: `{result['generated_at']}`",
        f"Auth mode: `{result.get('auth_mode')}`",
        "",
        "These are mined from human-written PR comments and review comments. Treat them as candidates: implement only patterns with concrete code evidence and low false-positive risk.",
        "",
    ]
    pattern_counts = result.get("comment_pattern_counts") or {}
    if not pattern_counts:
        lines.append("No comment patterns matched.")
        return "\n".join(lines) + "\n"

    rules_by_id = {str(rule["id"]): rule for rule in COMMENT_PATTERN_RULES}
    examples_by_pattern = result.get("comment_pattern_examples") or {}
    for pattern_id, count in sorted(pattern_counts.items(), key=lambda item: (-int(item[1]), item[0])):
        rule = rules_by_id.get(pattern_id, {})
        lines.extend(
            [
                f"## {rule.get('title', pattern_id)}",
                "",
                f"- Pattern ID: `{pattern_id}`",
                f"- Category: `{rule.get('category', 'general')}`",
                f"- Count: `{count}`",
                f"- Implementation hint: {rule.get('implementation_hint', 'Review examples and add a conservative check.')}",
                "",
            ]
        )
        examples = list(examples_by_pattern.get(pattern_id) or [])[:5]
        if examples:
            lines.append("Examples:")
            for example in examples:
                target = example.get("path") or "conversation"
                if example.get("line"):
                    target = f"{target}:{example['line']}"
                lines.append(
                    f"- `{example.get('repo')}#{example.get('pr_number')}` `{target}`: {example.get('body')}"
                )
                files = example.get("changed_files") or []
                if files:
                    lines.append(f"  Changed files: {', '.join(str(path) for path in files[:5])}")
                checks = example.get("check_evidence") or []
                if checks:
                    rendered_checks = ", ".join(
                        f"{item.get('name') or item.get('kind')}={item.get('conclusion') or 'error'}"
                        for item in checks[:3]
                    )
                    lines.append(f"  Failed checks/statuses: {rendered_checks}")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Mined PR Patterns",
        "",
        f"Generated: `{result['generated_at']}`",
        f"Auth mode: `{result.get('auth_mode')}`",
        f"Repos requested: `{result.get('repo_count_requested')}`",
        f"Exact PR refs requested: `{len(result.get('exact_pr_refs_requested') or [])}`",
        "",
        "## Category Counts",
        "",
    ]
    for category, count in sorted(result["category_counts"].items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- `{category}`: {count}")
    lines.extend(["", "## Human Review Comment Category Counts", ""])
    for category, count in sorted(result.get("review_comment_category_counts", {}).items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- `{category}`: {count}")
    lines.extend(["", "## Human Comment Pattern Counts", ""])
    for pattern_id, count in sorted(result.get("comment_pattern_counts", {}).items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- `{pattern_id}`: {count}")
    lines.extend(["", "## Top Recurring Comment Terms", ""])
    for term, count in list(result.get("top_comment_terms", {}).items())[:40]:
        lines.append(f"- `{term}`: {count}")
    lines.extend(["", "## Repos", ""])
    for repo, repo_result in result["repos"].items():
        lines.append(f"### {repo}")
        lines.append("")
        lines.append(f"- PRs scanned: `{repo_result['pr_count_scanned']}`")
        if repo_result.get("enriched_pr_count") is not None:
            lines.append(f"- Enriched matched PRs: `{repo_result.get('enriched_pr_count')}`")
        for category, count in sorted(repo_result["category_counts"].items(), key=lambda item: (-item[1], item[0])):
            lines.append(f"- `{category}`: {count}")
        if repo_result.get("review_comment_category_counts"):
            rendered = ", ".join(
                f"`{category}`={count}"
                for category, count in sorted(repo_result["review_comment_category_counts"].items(), key=lambda item: (-item[1], item[0]))[:8]
            )
            lines.append(f"- Human review categories: {rendered}")
        if repo_result["errors"]:
            lines.append("- Errors:")
            lines.extend(f"  - {error}" for error in repo_result["errors"])
        lines.append("")
    lines.extend(["## Examples", ""])
    for example in result["examples"][:120]:
        lines.append(f"### {example['repo']} PR #{example['number']}: {example.get('title') or ''}")
        lines.append("")
        lines.append(f"- URL: {example.get('url')}")
        lines.append(f"- Categories: {', '.join(f'`{category}`' for category in example['categories'])}")
        if example["changed_files"]:
            lines.append(f"- Changed files: {', '.join(str(path) for path in example['changed_files'][:8])}")
        if example.get("check_evidence"):
            rendered_checks = ", ".join(
                f"{item.get('name') or item.get('kind')}={item.get('conclusion') or 'error'}"
                for item in example.get("check_evidence", [])[:5]
            )
            lines.append(f"- Failed checks/statuses: {rendered_checks}")
        for sample in example["samples"][:3]:
            target = sample["path"] or sample["kind"]
            if sample["line"]:
                target = f"{target}:{sample['line']}"
            lines.append(f"- Sample `{target}`: {sample['body']}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
