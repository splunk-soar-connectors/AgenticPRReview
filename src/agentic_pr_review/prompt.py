"""Prompt construction for the model-backed reviewer."""

from __future__ import annotations

import re

from .json_utils import dump_json, truncate_text
from .secret_redactor import redact_text
from .sdk_analysis import build_sdk_review_inventory


SYSTEM_PROMPT = """You are a strict SOAR connectors pull-request reviewer.

Your job is to review GitHub PR context for example-connectors apps. You are
not doing a generic style review. Focus on correctness, connector conventions,
CI readiness, user-visible docs accuracy, and generalized high-value failure
patterns learned from connector reviews.

Good connector PR baseline:
- One focused app/change per PR.
- PR body, manual docs/generated-doc inputs, release notes, app JSON, and code
  agree.
- App metadata is correct: appid, name, package_name, product_name, type,
  python_version, app_version, main_module, publisher, dependencies.
- SDK-format apps may generate the final app JSON from pyproject.toml,
  src/app.py, src/asset.py, action decorators/models, and SDK metadata. For
  SDK migrations, follow the active SDK source-of-truth instead of treating a
  removed legacy root app JSON as current head metadata.
- For SDK migrations, compare the legacy manifest/base connector intent to the
  new SDK source. The official SDK converter migrates asset configuration,
  action names/descriptions/metadata, action parameters, and outputs, but custom
  views, webhook handlers, custom REST handlers, initialization code, and action
  summaries require manual migration.
- For SDK migrations, your primary review job is backward compatibility:
  compare removed legacy `action_result.data.*` and `action_result.summary.*`
  paths, parameter validation, action failure paths, and real vendor payload
  shapes against the new SDK models and `model_validate(...)` calls. CI/config
  failures matter, but do not let them crowd out concrete runtime/playbook
  regressions.
- SDK apps should expose one App instance through `[tool.soar.app].main_module`,
  register one `test connectivity` action, pass the BaseAsset subclass through
  `asset_cls`, and keep action params/returns typed so generated metadata is
  correct.
- SDK `App.action()` and `App.register_action()` default to `read_only=True`.
  Mutating create/update/delete/upload actions need explicit `read_only=False`
  in SDK registrations.
- SDK `App.make_request()` is a special decorator and does not accept
  `read_only=False`; do not recommend `@app.make_request(read_only=False)`.
  Review make-request actions for params/output serialization, tests, and
  endpoint/header/body validation instead.
- SDK `Params` models are scalar models; do not assume `Param(allow_list=True)`
  should use a Python list annotation.
- Action metadata is correct: identifier, type, read_only, params, output
  paths, summary fields, render definitions.
- Mutating actions are not read_only.
- Compatibility metadata changes include the right app_version, python_version,
  and app type values.
- User-visible behavior changes have release notes and docs when needed.
- New/risky behavior has tests or clear test coverage.
- CI failures are considered part of the review, especially pre-commit and
  mergeability blockers.
- Secrets, tokens, auth headers, full request payloads, full API responses,
  tenant data, and PII must not be logged.
- Legacy custom action views should render the completed action result data
  passed by SOAR. Be suspicious of view modules that instantiate a connector,
  create a fresh ActionResult, or re-call enrichment/search APIs from GET query
  parameters; that can show empty data, consume quota, or call APIs with None.

High-value review pattern library:
- API/auth correctness: wrong OAuth/client-credentials flow, stale auth docs,
  missing token request timeout, TLS verification defaults, retry behavior.
- test_connectivity must validate the auth/API path the asset actually needs.
- External API TLS verification should be configurable from asset metadata and
  should not default to disabled. If Python reads `verify_server_cert`, the app
  JSON/SDK asset model must expose it.
- HTTP calls should set timeouts, including calls through connector helpers,
  session objects, token requests, and platform REST lookups.
- Do not add full response text, response headers, token payloads, or API
  payloads to debug data.
- SOAR metadata drift: mutating POST/PATCH/DELETE actions marked read_only,
  invalid legacy app type, incorrect python_version, product_version copied
  from the SOAR app version instead of the vendor product version,
  latest_tested_versions copied from the SOAR app version, missing version bump.
- Polling/checkpoint/dedup defects: timestamp-only cursors, single-ID boundary
  cursors, Poll Now contaminating scheduled state, unstable source_data_identifier,
  checkpoint/state saved after partial ingestion failure.
- on_poll implementations should use standard SOAR controls such as Poll Now,
  start_time, end_time, container_count, and artifact_count where available.
  They should create containers with a routable label, check save_container and
  save_artifact return values, use stable artifact labels/metadata, and avoid
  dynamic CEF keys derived directly from vendor indicator types.
- JSONL/feed parsers should not silently discard malformed lines and still
  report success; surface corruption in summary/debug output or fail clearly.
- HTTP 429 and rate-limit headers should be handled distinctly from generic
  server errors when the vendor API can rate-limit polling/enrichment flows.
- Polling container/artifact modeling matters: avoid SDI/container designs that
  require a SOAR `/rest/container` lookup per vendor rule/group/event. Prefer a
  stable vendor object identity or one bounded lookup per poll run with an
  in-memory map.
- Poll Now-specific lookback asset parameters and absolute lookback values can
  age into invalid configurations; prefer standard on_poll parameters unless
  there is a clear connector limitation and docs/tests cover it.
- Pagination/truncation: list actions only fetching the first page or ignoring
  offset/limit/pageToken contracts.
- Output/schema mismatch: add_data() emits fields not declared in app JSON,
  summaries emit undeclared fields, wrong output data types, SDK actions miss
  summary_type, README claims outputs not exposed.
- SDK migrations can break playbooks even when action identifiers are
  preserved. Flag dropped or renamed `action_result.summary.*` fields, flattened
  or reshaped nested output datapaths, strict SDK models that validate raw API
  responses without normalization, and migration PR bodies that overclaim output
  compatibility.
- SDK migration tests must import the app through the package entry point used
  by `[tool.soar.app].main_module`; adding `src/` to `sys.path` and importing
  `app` as a top-level module can fail when `src/app.py` uses package-relative
  imports.
- Make-request output models must serialize the fields they claim to expose.
  Dynamic attributes added with `object.__setattr__()` after Pydantic model
  construction are suspicious unless the model explicitly supports and emits
  those extra fields.
- SDK output models built directly from vendor API dictionaries must tolerate
  valid sparse responses. For Microsoft Graph OneDrive/driveItem responses,
  fields and facets such as owner, quota, created/modified timestamps, webUrl,
  parentReference, file/folder/package/image facets, application/user identity
  details, and upload-session completion fields can be absent depending on
  backing store and response shape. Flag strict models only when the current
  code constructs them directly from raw Graph responses.
- Microsoft Graph Client Credentials routes that use asset `target_user_id`
  should normalize/strip that value before formatting `/users/{...}` URLs.
- Large file actions should stream downloads/uploads where practical. Microsoft
  Graph upload-session code should use nextExpectedRanges and bounded
  retry/backoff for transient 429/5xx responses.
- Missing validation: blank strings, missing
  cross-field validators, unencoded IDs interpolated into URLs.
- Search APIs with query syntax need query parameter
  construction plus escaping/validation of user values before inserting them
  into `domain:...`, `ip:...`, or similar expressions.
- CEF/output metadata should match the field semantics. A user display name,
  message, tenant value, or API identifier should not be tagged as a file path
  unless it is actually a filesystem path.
- PR/docs accuracy: release notes, PR body, manual docs/generated-doc inputs,
  dependencies, defaults, and output claims must match the diff.
- SDK package builds source dependency wheels from `uv.lock`; dependency
  changes need a current lock file.
- on_poll params declared in JSON should be used, default available on_poll
  params should not be replaced by workaround asset params, and on_poll should
  avoid dynamic tag generation unless clearly justified.
- Unused boilerplate handlers or methods should be removed or implemented.
- Use PR comments or supplied docs links to check whether code follows vendor
  API contracts, especially auth, pagination, validation, and response shapes.
- New app mapping checks may be needed in the .github/ci-metadata support repos.
- review_input.historical_context may include similar human-reviewed historical
  PR examples from mined comments. Use these examples as reviewer memory and
  pattern guidance only. They are not evidence that the current PR has the same
  bug, and they do not outrank current PR code, docs, comments, CI output, or
  deterministic checks. Report a finding only when current PR evidence
  independently proves the issue.
- Pre-commit failures are high-signal review findings. Pay special attention to
  local hook failures such as app_package_name, valid_app_name_and_guid,
  build-docs, release-notes, detect-secrets, ruff, semgrep, check-json, and
  check-yaml.
- Compile/import failures and unresolved conflict markers are merge blockers.
  Mention the exact file/line and the concrete syntax/import/conflict marker
  that needs to be fixed.
- Historical static-test blockers include min platform/min_phantom_version,
  action name conventions, additional logging/verbosity, license/NOTICE,
  product-name-on-files, missing app test playbooks, and missing integration
  test results. Mention these only when the exact failed static/sanity test is
  available; do not post a generic "tests are failing" finding.
- When review_input.ci.failed_check_logs is present, use those excerpts to
  name the failing hook/test and the concrete file/assertion/metadata problem.
  Do not produce a precommit finding whose fix is only "resolve the failure" or
  "inspect the logs"; the fix must say how to address the named failure.
- Concrete merge conflicts and named hook/test merge blockers should be called
  out separately from ordinary CI failures.

Strict output rules:
- The default outcome is zero findings. Return [] unless the supplied context
  proves a concrete, actionable issue.
- Do not praise the PR.
- Do not summarize obvious changes, migrations, file moves, or generated output.
- Do not report "verify", "confirm", "consider", or "unclear" findings. Put
  uncertainty in model_notes, not findings.
- Do not report a finding just because a risky area changed. Report it only
  when the diff, full file, docs, app JSON, comments, or CI excerpts show the
  exact bug or missing artifact.
- Every finding must include a non-null file plus either a changed-line number
  or a concrete code reference such as a function, action identifier, parameter,
  output path, hook name with file path, or commit SHA when commit metadata is
  available.
- Every finding must include evidence, why_it_matters, and suggested_fix. If you
  cannot fill those with concrete facts from the supplied context, omit the
  finding.
- Only report issues in these areas: app JSON parameter/action/output schema
  mismatch; connector implementation mismatch; README/manual docs mismatch;
  missing or weak tests with a concrete changed behavior; invalid conventional
  commits when commit metadata is supplied; obvious pre-commit/lint issues with
  exact file or hook evidence; secrets or unsafe logging; broad exception
  handling; unreachable code; missing validation.

Evidence rules:
- Use the supplied PR context and deterministic findings as evidence.
- Historical examples can help you decide what to inspect and how to phrase a
  concise human-style comment, but do not cite historical PRs as evidence in an
  active finding. Cite the current file, line, diff, docs, comments, or CI log.
- Historical examples are not a separate review source to optimize for. Treat
  them as part of the same checklist/pattern library as the deterministic
  checks, then verify against the current PR before reporting anything.
- Historical categories can be more specific than the required output schema;
  map them to the nearest allowed finding category.
- Review the whole diff, including removed lines. A deletion can be the bug if
  it removes tests, timeouts, TLS verification, pagination, checkpoint guards,
  validation, error handling, release notes, or docs without an equivalent
  replacement.
- When a removed block is risky, compare the removed lines with the added lines
  and current full file before reporting it. Do not assume every deletion is a
  problem; only flag removals that create a concrete regression or missing
  coverage/docs.
- When a top-level app JSON file is removed as part of an SDK migration, do not
  report stale fields from the deleted JSON as active findings. Only report
  metadata drift if the active SDK source-of-truth or a generated manifest in
  the PR is wrong, or if CI proves the generated packaging metadata is wrong.
- Omit deterministic findings that are false positives or require no action; do
  not include "no action required" items in findings.
- Do not invent vendor API requirements. If a PR comment or doc link indicates
  an API contract, use it. Otherwise omit the API-contract concern.
- Do not return "verify/confirm this" as an active finding. If the evidence only
  supports a verification reminder, put it in model_notes instead.
- For docs accuracy findings, compare the supplied README/manual docs content
  against the diff. Do not claim docs are missing when the relevant docs content
  was not supplied or already contains the changed parameter/action behavior.
- README.md is generated for these connector PRs. Do not create README.md-only
  active findings or ask contributors to hand-edit README.md. If generated docs
  are wrong, target manual_readme_content.md, app JSON metadata, release notes,
  or the generator input that needs to change.
- Use review_input.doc_context snippets when README/manual files are too large
  for full inclusion. Treat matching snippets as current docs evidence.
- Only return active findings you would be comfortable posting on an external
  contributor's PR. If the evidence is incomplete or you are unsure, put that
  concern in model_notes instead of findings.
- Do not return a finding that only says GitHub reports mergeable_state=blocked,
  behind, or otherwise not mergeable. A mergeability finding is useful only when
  it explains the concrete blocker: where it appears, why it blocks merge, and
  the specific fix, such as a conflict file or a named failing hook/test.
- Avoid nitpicks and broad style comments.
- Do not artificially cap the review at a small number of findings. Report all
  high-confidence correctness, security, SOAR contract, metadata, polling,
  output-schema, and test-evidence issues you would expect a human connector
  reviewer to raise. Group repeated instances under one finding when they share
  the same root cause, but do not omit a serious issue just because other
  serious issues already exist.
- For each finding, set file and line to the exact GitHub comment target when
  possible. Keep suggested_fix short, direct, and contributor-facing.
- For code findings, include suggested_code only when you can provide the exact
  replacement code for the targeted changed line or block. Do not put prose in
  suggested_code. Omit it or set it to null when the fix needs contributor
  judgment.
- If a finding is already fixed in the current head, do not report it as active.
- If only CI data is missing, say so in model_notes, not as a blocking finding.
- Return only valid JSON. No Markdown fences.

Required JSON shape:
{
  "summary": "",
  "overall_status": "needs_review|looks_good|blocked_by_ci|error",
  "safe_to_publish": true,
  "findings": [
    {
      "id": "short-stable-id",
      "title": "clear finding title",
      "category": "api_auth_correctness|polling_checkpoint|output_schema_mismatch|unsafe_logging|soar_metadata|docs_pr_accuracy|pagination|validation|missing_tests|precommit|merge_conflict|ci_synthesis|general",
      "severity": "critical|high|medium|low|info",
      "confidence": "high|medium|low",
      "file": "path or null",
      "line": 123,
      "code_reference": "function/action/parameter/output path/hook/commit reference when line is null, or null",
      "evidence": "specific evidence from code/comment/CI",
      "why_it_matters": "why this matters for a SOAR connector",
      "suggested_fix": "actionable fix",
      "suggested_code": "exact replacement code for the GitHub suggestion block, or null",
      "source": "claude"
    }
  ],
  "model_notes": "brief notes about uncertainty or unavailable context"
}
"""


def build_user_prompt(review_input: dict, deterministic_findings: list[dict], *, max_chars: int) -> str:
    payload = build_model_payload(review_input, deterministic_findings, max_chars=max_chars)
    text = redact_text(dump_json(payload))
    if len(text) > max_chars:
        text = truncate_text(text, max_chars)
    return (
        "Review this PR context and produce the required JSON object. "
        "Treat deterministic findings as candidate evidence: confirm, merge, "
        "downgrade, or omit them based on the full context. Return no findings "
        "when the context does not prove an actionable file-backed issue.\n\n"
        f"{text}"
    )


def build_synthesis_prompt(
    review_input: dict,
    deterministic_findings: list[dict],
    chunk_review_outputs: list[dict],
    *,
    max_chars: int,
) -> str:
    candidate_findings = []
    for output in chunk_review_outputs:
        for finding in output.get("findings", []) or []:
            if isinstance(finding, dict):
                candidate_findings.append(finding)

    payload = {
        "task": "Synthesize final PR review findings from deterministic checks and chunk-level model reviews.",
        "instructions": [
            "Merge duplicate findings across chunks.",
            "Drop speculative, weak, already-fixed, README-only, or generic CI findings.",
            "Keep only concrete findings that should be posted to an external contributor.",
            "Prefer exact file/line targets from chunk findings or deterministic findings.",
            "Use CI/check/comment context to confirm whether findings are real blockers.",
            "Return the required JSON object only.",
        ],
        "deterministic_findings": deterministic_findings,
        "chunk_candidate_findings": candidate_findings,
        "chunk_model_notes": [
            {
                "chunk_id": output.get("chunk_id"),
                "model_notes": output.get("model_notes"),
                "usage": output.get("usage"),
            }
            for output in chunk_review_outputs
        ],
        "collection_diagnostics": build_collection_diagnostics(review_input),
        "pr_context": compact_review_input_for_model(
            review_input,
            limits={"patch": 1_500, "file": 3_500, "comment": 700, "review": 600},
        ),
    }
    text = redact_text(dump_json(payload))
    if len(text) > max_chars:
        payload["chunk_candidate_findings"] = candidate_findings[:120]
        payload["pr_context"] = compact_review_input_for_model(
            review_input,
            limits={"patch": 800, "file": 2_000, "comment": 350, "review": 300},
        )
        text = redact_text(dump_json(payload))
    if len(text) > max_chars:
        text = truncate_text(text, max_chars)
    return (
        "Synthesize these chunk-level review results into one final PR review. "
        "Return only the required JSON object. Do not add findings unless the "
        "candidate evidence proves a concrete issue.\n\n"
        f"{text}"
    )


def build_model_payload(review_input: dict, deterministic_findings: list[dict], *, max_chars: int) -> dict:
    """Pack model context so high-signal data survives large PRs.

    Deterministic findings and collection diagnostics are intentionally placed
    before bulky patches/files. If the PR is too large, we progressively shrink
    patches, full files, and comments instead of blindly cutting the whole JSON
    after whichever field happens to come first.
    """

    for limits in (
        None,
        {"patch": 8_000, "file": 12_000, "comment": 1_500, "review": 1_200},
        {"patch": 4_000, "file": 8_000, "comment": 900, "review": 800},
        {"patch": 1_800, "file": 4_000, "comment": 500, "review": 500},
    ):
        compacted = compact_review_input_for_model(review_input, limits=limits)
        payload = {
            "deterministic_findings": deterministic_findings,
            "collection_diagnostics": build_collection_diagnostics(review_input),
            "review_input": compacted,
        }
        text = dump_json(payload)
        if len(text) <= max_chars:
            if limits is not None:
                payload["model_context_notes"] = {
                    "compacted_for_model": True,
                    "limits": limits,
                    "uncompacted_chars": len(dump_json({"review_input": review_input, "deterministic_findings": deterministic_findings})),
                    "compacted_chars": len(text),
                }
            return payload

    payload = {
        "deterministic_findings": deterministic_findings,
        "collection_diagnostics": build_collection_diagnostics(review_input),
        "review_input": compact_review_input_for_model(
            review_input,
            limits={"patch": 1_000, "file": 2_000, "comment": 300, "review": 300},
        ),
        "model_context_notes": {
            "compacted_for_model": True,
            "hard_truncation_expected": True,
            "reason": "PR context still exceeds max_model_input_chars after aggressive compaction.",
        },
    }
    return payload


def compact_review_input_for_model(review_input: dict, *, limits: dict[str, int] | None) -> dict:
    if limits is None:
        return review_input

    compacted = dict(review_input)
    compacted["changed_files"] = [
        compact_changed_file(item, patch_limit=limits["patch"])
        for item in review_input.get("changed_files", [])
        if isinstance(item, dict)
    ]
    compacted["full_files"] = {
        path: truncate_text(text, limits["file"])
        for path, text in (review_input.get("full_files") or {}).items()
    }
    compacted["base_files"] = {
        path: truncate_text(text, min(limits["file"], 3_000))
        for path, text in (review_input.get("base_files") or {}).items()
    }
    compacted["comments"] = compact_comments_for_model(review_input.get("comments") or {}, limits=limits)
    compacted["doc_context"] = compact_doc_context(review_input.get("doc_context") or {}, limits=limits)
    compacted["historical_context"] = compact_historical_context(
        review_input.get("historical_context") or {},
        limits=limits,
    )
    compacted["sdk_manifest"] = compact_sdk_manifest_context(review_input.get("sdk_manifest") or {})
    if isinstance(review_input.get("sdk_review_inventory"), dict):
        compacted["sdk_review_inventory"] = compact_sdk_review_inventory(review_input["sdk_review_inventory"])
    return compacted


def compact_changed_file(item: dict, *, patch_limit: int) -> dict:
    output = dict(item)
    patch = str(item.get("patch") or "")
    output["patch"] = truncate_text(patch, patch_limit)
    output["model_patch_truncated"] = len(patch) > patch_limit
    return output


def compact_comments_for_model(comments: dict, *, limits: dict[str, int]) -> dict:
    output = {}
    for key, body_limit in (
        ("issue_comments", limits["comment"]),
        ("review_comments", limits["comment"]),
        ("reviews", limits["review"]),
    ):
        items = comments.get(key) or []
        if not isinstance(items, list):
            output[key] = []
            continue
        output[key] = [compact_comment_item(item, body_limit=body_limit) for item in items[:80] if isinstance(item, dict)]
    return output


def compact_comment_item(item: dict, *, body_limit: int) -> dict:
    output = dict(item)
    output["body"] = truncate_text(str(item.get("body") or ""), body_limit)
    return output


def compact_doc_context(doc_context: dict, *, limits: dict[str, int]) -> dict:
    output = {}
    for path, snippets in doc_context.items():
        if not isinstance(snippets, list):
            continue
        compacted = []
        for snippet in snippets[:8]:
            if not isinstance(snippet, dict):
                continue
            item = dict(snippet)
            item["snippet"] = truncate_text(str(snippet.get("snippet") or ""), min(limits["file"], 1_800))
            compacted.append(item)
        output[path] = compacted
    return output


def compact_historical_context(historical_context: dict, *, limits: dict[str, int]) -> dict:
    if not historical_context:
        return {}
    output = dict(historical_context)
    matches = historical_context.get("matches") or []
    if not isinstance(matches, list):
        output["matches"] = []
        return output

    compacted = []
    for item in matches[:10]:
        if not isinstance(item, dict):
            continue
        compact_item = dict(item)
        compact_item["human_comment"] = truncate_text(
            str(item.get("human_comment") or ""),
            min(limits["comment"], 800),
        )
        compact_item["diff_hunk"] = truncate_text(
            str(item.get("diff_hunk") or ""),
            min(limits["patch"], 900),
        )
        file_evidence = []
        for evidence in (item.get("file_evidence") or [])[:2]:
            if not isinstance(evidence, dict):
                continue
            compact_evidence = dict(evidence)
            compact_evidence["patch_excerpt"] = truncate_text(
                str(evidence.get("patch_excerpt") or ""),
                min(limits["patch"], 700),
            )
            file_evidence.append(compact_evidence)
        compact_item["file_evidence"] = file_evidence
        compacted.append(compact_item)
    output["matches"] = compacted
    output["matched_example_count"] = len(compacted)
    return output


def build_collection_diagnostics(review_input: dict) -> dict:
    notes = review_input.get("collector_notes") or {}
    changed_files = review_input.get("changed_files") or []
    changed_file_map = {
        str(item.get("filename")): item
        for item in changed_files
        if isinstance(item, dict) and item.get("filename")
    }
    removed_root_json_files = [
        path
        for path, item in changed_file_map.items()
        if item.get("status") == "removed" and "/" not in path and path.endswith(".json")
    ]
    full_files = review_input.get("full_files") or {}
    sdk_metadata_sources = [
        path
        for path in ("pyproject.toml", "src/app.py", "src/asset.py")
        if path in full_files or path in changed_file_map
    ]
    raw_sdk_review_inventory = review_input.get("sdk_review_inventory")
    if not isinstance(raw_sdk_review_inventory, dict) or not raw_sdk_review_inventory:
        raw_sdk_review_inventory = build_sdk_review_inventory(review_input)
    sdk_review_inventory = compact_sdk_review_inventory(raw_sdk_review_inventory)
    return {
        "github_auth_mode": notes.get("github_auth_mode"),
        "changed_file_count": notes.get("changed_file_count", len(changed_files)),
        "changed_file_patch_count": notes.get("changed_file_patch_count"),
        "changed_file_missing_patch_count": notes.get("changed_file_missing_patch_count"),
        "changed_file_truncated_patch_count": notes.get("changed_file_truncated_patch_count"),
        "changed_file_total_additions": notes.get("changed_file_total_additions"),
        "changed_file_total_deletions": notes.get("changed_file_total_deletions"),
        "deep_reconstructed_patch_count": notes.get("deep_reconstructed_patch_count"),
        "full_file_count": notes.get("full_file_count"),
        "full_file_missing_count": notes.get("full_file_missing_count"),
        "removed_root_json_files": removed_root_json_files,
        "sdk_migration_detected": bool(removed_root_json_files and {"pyproject.toml", "src/app.py"}.issubset(set(sdk_metadata_sources))),
        "sdk_metadata_sources": sdk_metadata_sources,
        "sdk_review_inventory": sdk_review_inventory,
        "sdk_manifest": compact_sdk_manifest_context(review_input.get("sdk_manifest") or {}),
        "ci_error_count": notes.get("ci_error_count"),
        "historical_context_enabled": notes.get("historical_context_enabled"),
        "historical_context_match_count": notes.get("historical_context_match_count"),
        "historical_context_categories": notes.get("historical_context_categories"),
        "ci_errors": (review_input.get("ci") or {}).get("errors", [])[:10]
        if isinstance(review_input.get("ci"), dict)
        else [],
    }


def compact_sdk_manifest_context(sdk_manifest: dict) -> dict:
    if not isinstance(sdk_manifest, dict) or not sdk_manifest:
        return {}
    manifest = sdk_manifest.get("manifest")
    output = {
        "attempted": sdk_manifest.get("attempted"),
        "status": sdk_manifest.get("status"),
        "stage": sdk_manifest.get("stage"),
        "reason": sdk_manifest.get("reason"),
        "returncode": sdk_manifest.get("returncode"),
        "command": sdk_manifest.get("command"),
        "manifest_summary": sdk_manifest.get("manifest_summary"),
        "stdout": truncate_text(str(sdk_manifest.get("stdout") or ""), 1200),
        "stderr": truncate_text(str(sdk_manifest.get("stderr") or ""), 1600),
    }
    if isinstance(manifest, dict):
        output["generated_action_count"] = len(manifest.get("actions") or [])
        config = manifest.get("configuration") or {}
        output["generated_config_count"] = len(config) if isinstance(config, (dict, list)) else 0
    return output


def compact_sdk_review_inventory(inventory: dict) -> dict:
    python = inventory.get("python") or {}
    legacy = inventory.get("legacy") or {}
    registrations = python.get("action_registrations") or []
    return {
        "is_sdk_app": inventory.get("is_sdk_app"),
        "is_sdk_migration": inventory.get("is_sdk_migration"),
        "main_module": inventory.get("main_module"),
        "main_module_file": inventory.get("main_module_file"),
        "main_app_instance_present": inventory.get("main_app_instance_present"),
        "app_instances": [
            {
                "file": item.get("file"),
                "line": item.get("line"),
                "name": item.get("name"),
                "asset_cls": item.get("asset_cls"),
                "app_type": item.get("app_type"),
            }
            for item in (python.get("app_instances") or [])[:5]
            if isinstance(item, dict)
        ],
        "asset_field_names": (python.get("asset_field_names") or [])[:80],
        "registered_action_identifiers": (python.get("registered_action_identifiers") or [])[:120],
        "action_registration_samples": [
            {
                "file": item.get("file"),
                "line": item.get("line"),
                "identifier": item.get("identifier"),
                "registration_type": item.get("registration_type"),
                "read_only": item.get("read_only"),
                "return_annotation": item.get("return_annotation"),
                "summary_type": item.get("summary_type"),
            }
            for item in registrations[:40]
            if isinstance(item, dict)
        ],
        "test_connectivity_count": python.get("test_connectivity_count"),
        "view_handler_count": python.get("view_handler_count"),
        "webhook_count": python.get("webhook_count"),
        "enable_webhooks_count": python.get("enable_webhooks_count"),
        "cli_invocation_present": python.get("cli_invocation_present"),
        "unsupported_param_fields": (python.get("unsupported_param_fields") or [])[:20],
        "legacy_manifest_paths": legacy.get("manifest_paths") or [],
        "legacy_action_identifiers": (legacy.get("action_identifiers") or [])[:120],
        "legacy_config_fields": (legacy.get("config_fields") or [])[:80],
        "legacy_custom_view_actions": legacy.get("custom_view_actions") or [],
        "legacy_summary_actions": legacy.get("summary_actions") or [],
        "legacy_action_summary_fields": legacy.get("action_summary_fields") or {},
        "legacy_action_output_path_samples": compact_legacy_output_paths(legacy.get("action_output_paths") or {}),
        "legacy_has_rest_handler": legacy.get("has_rest_handler"),
        "legacy_has_webhooks": legacy.get("has_webhooks"),
    }


def compact_legacy_output_paths(paths_by_action: dict) -> dict:
    output = {}
    for identifier, paths in list(paths_by_action.items())[:40]:
        if not isinstance(paths, list):
            continue
        interesting = [
            path
            for path in paths
            if "action_result.summary." in str(path) or re.search(r"\.\*\.\*\.", str(path))
        ]
        if interesting:
            output[identifier] = interesting[:20]
    return output
