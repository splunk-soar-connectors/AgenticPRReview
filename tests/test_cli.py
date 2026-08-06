import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agentic_pr_review.cli import (
    build_parser,
    publish_target_pipeline_failure_comments,
    refine_low_confidence_pipeline_findings,
    suppress_redundant_published_pipeline_findings,
    write_artifacts,
)
from agentic_pr_review.secret_redactor import REDACTED_SECRET


class CLITest(unittest.TestCase):
    def test_sdk_manifest_generation_is_opt_in(self):
        args = build_parser().parse_args(["review", "owner/repo", "1"])

        self.assertFalse(args.enable_sdk_manifest)
        self.assertFalse(args.disable_sdk_manifest)

    def test_sdk_manifest_generation_can_be_enabled(self):
        args = build_parser().parse_args(["review", "owner/repo", "1", "--enable-sdk-manifest"])

        self.assertTrue(args.enable_sdk_manifest)

    def test_default_output_dir_is_bot_repo_runs(self):
        args = build_parser().parse_args(["review", "owner/repo", "1"])

        self.assertEqual(Path(args.output_dir), Path(__file__).resolve().parents[1] / "runs")

    def test_default_publish_comment_budget_has_no_fixed_cap(self):
        args = build_parser().parse_args(["review", "owner/repo", "1"])

        self.assertEqual(args.max_published_comments, 0)

    def test_write_artifacts_redacts_runtime_secrets(self):
        env = {"CIRCUIT_CLIENT_SECRET": "canary-artifact-secret"}
        review_input = {
            "repo": "owner/repo",
            "pr": {"number": 1},
            "full_files": {"connector.py": "secret='canary-artifact-secret'"},
        }
        review_output = {
            "summary": "canary-artifact-secret",
            "overall_status": "needs_review",
            "safe_to_publish": True,
            "findings": [{"title": "Leak", "evidence": "canary-artifact-secret"}],
        }
        comment_plan = {"comments": [{"body": "canary-artifact-secret"}]}
        publish_result = {"results": [{"error": "canary-artifact-secret"}]}

        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, env, clear=True):
            run_dir = Path(tmp)
            write_artifacts(
                run_dir,
                review_input,
                review_output,
                "comment canary-artifact-secret",
                comment_plan,
                publish_result,
            )
            rendered = "\n".join(path.read_text(encoding="utf-8") for path in run_dir.iterdir())
            output = json.loads((run_dir / "review_output.json").read_text(encoding="utf-8"))

        self.assertIn(REDACTED_SECRET, rendered)
        self.assertNotIn("canary-artifact-secret", rendered)
        self.assertEqual(output["summary"], REDACTED_SECRET)

    def test_write_artifacts_writes_model_routing_summary(self):
        review_input = {"repo": "owner/repo", "pr": {"number": 1}}
        review_output = {
            "summary": "ok",
            "overall_status": "looks_good",
            "safe_to_publish": True,
            "findings": [],
            "deep_review": {
                "model_routing_summary": {
                    "original_chunk_count": 3,
                    "planned_model_chunk_count": 2,
                    "cached_chunk_count": 1,
                }
            },
        }

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            write_artifacts(run_dir, review_input, review_output, "comment")
            routing = json.loads((run_dir / "model_routing_summary.json").read_text(encoding="utf-8"))

        self.assertEqual(routing["original_chunk_count"], 3)
        self.assertEqual(routing["cached_chunk_count"], 1)

    def test_target_pipeline_failures_publish_before_model_without_label(self):
        class FakeClient:
            def __init__(self):
                self.bodies = []
                self.labels = []

            def create_issue_comment(self, repo, number, *, body):
                self.bodies.append(body)
                return {"html_url": f"https://github.example/{repo}/pull/{number}#issuecomment-1"}

            def add_issue_labels(self, repo, number, labels):
                self.labels.append((repo, number, labels))
                return [{"name": labels[0]}]

        progress_messages = []
        review_input = {
            "repo": "owner/repo",
            "pr": {"number": 1, "head": {"sha": "abc"}},
            "changed_files": [],
            "comments": {"issue_comments": [], "review_comments": [], "reviews": []},
        }
        deterministic_findings = [
            {
                "id": "ci-1",
                "title": "build pipeline job failed",
                "category": "ci_pipeline_failure",
                "finding_category": "introduced_bug",
                "causality": "exposed_by_pr",
                "severity": "high",
                "confidence": "high",
                "merge_blocking": True,
                "publication_destination": "inline_blocking",
                "file": None,
                "line": None,
                "code_reference": "GitHub Actions job build",
                "evidence": "`build` concluded `failure`. Failed job: https://github.example/actions/runs/123/job/99",
                "why_it_matters": "The build failure blocks merge.",
                "suggested_fix": "Open the build job log, fix the first concrete packaging/build error shown there, and rerun build.",
                "url": "https://github.example/actions/runs/123/job/99",
            }
        ]

        with tempfile.TemporaryDirectory() as tmp:
            result = publish_target_pipeline_failure_comments(
                FakeClient(),
                review_input,
                deterministic_findings,
                run_dir=Path(tmp),
                publish_comments=True,
                allow_duplicates=False,
                progress=progress_messages.append,
            )

        self.assertEqual(result["posted"], 1)
        self.assertEqual(result["label"]["status"], "skipped_disabled")
        self.assertEqual(len(review_input["comments"]["issue_comments"]), 1)
        self.assertEqual(review_input["ci"]["published_target_pipeline_jobs"], ["build"])
        self.assertIn("agentic-pr-review:", review_input["comments"]["issue_comments"][0]["body"])
        self.assertTrue(any("before model review" in message for message in progress_messages))

    def test_target_pipeline_publish_suppresses_low_score_compile_failure(self):
        class FakeClient:
            def __init__(self):
                self.bodies = []

            def create_issue_comment(self, repo, number, *, body):
                self.bodies.append(body)
                return {"html_url": f"https://github.example/{repo}/pull/{number}#issuecomment-{len(self.bodies)}"}

            def add_issue_labels(self, repo, number, labels):
                return [{"name": labels[0]}]

        progress_messages = []
        client = FakeClient()
        review_input = {
            "repo": "owner/repo",
            "pr": {"number": 1, "head": {"sha": "abc"}},
            "changed_files": [],
            "comments": {"issue_comments": [], "review_comments": [], "reviews": []},
        }
        deterministic_findings = [
            {
                "id": "ci-precommit",
                "title": "pre-commit pipeline job failed",
                "category": "ci_pipeline_failure",
                "finding_category": "introduced_bug",
                "causality": "exposed_by_pr",
                "severity": "high",
                "confidence": "high",
                "merge_blocking": True,
                "publication_destination": "inline_blocking",
                "file": None,
                "line": None,
                "code_reference": "GitHub Actions job pre-commit",
                "evidence": "`pre-commit` concluded `failure`. Failed job: https://github.example/actions/runs/123",
                "why_it_matters": "The pre-commit failure blocks merge.",
                "suggested_fix": "Fix the detect-secrets failure and rerun pre-commit.",
                "url": "https://github.example/actions/runs/123",
            },
            {
                "id": "ci-compile",
                "title": "compile pipeline job failed",
                "category": "ci_pipeline_failure",
                "finding_category": "introduced_bug",
                "causality": "exposed_by_pr",
                "severity": "high",
                "confidence": "medium",
                "confidence_score": 0.55,
                "merge_blocking": True,
                "publication_destination": "inline_blocking",
                "file": None,
                "line": None,
                "code_reference": "GitHub Actions job compile",
                "evidence": "`compile` concluded `failure`. Failed job: https://github.example/actions/runs/123",
                "why_it_matters": "The compile failure blocks merge.",
                "suggested_fix": (
                    "Open the linked `compile` job log and fix the first terminal failure summary "
                    "before rerunning the workflow."
                ),
                "url": "https://github.example/actions/runs/123",
            },
        ]

        with tempfile.TemporaryDirectory() as tmp:
            result = publish_target_pipeline_failure_comments(
                client,
                review_input,
                deterministic_findings,
                run_dir=Path(tmp),
                publish_comments=True,
                allow_duplicates=False,
                progress=progress_messages.append,
            )
            plan = json.loads((Path(tmp) / "ci_pipeline_comment_plan.json").read_text(encoding="utf-8"))

        self.assertEqual(result["posted"], 1)
        self.assertEqual(len(plan["comments"]), 1)
        self.assertEqual(len(client.bodies), 1)
        self.assertEqual(review_input["ci"]["published_target_pipeline_jobs"], ["pre-commit"])
        self.assertTrue(any("Publishing 1 target pipeline" in message for message in progress_messages))

    def test_target_pipeline_publish_groups_same_root_cause_jobs_with_all_job_links(self):
        class FakeClient:
            def __init__(self):
                self.bodies = []

            def create_issue_comment(self, repo, number, *, body):
                self.bodies.append(body)
                return {"html_url": f"https://github.example/{repo}/pull/{number}#issuecomment-{len(self.bodies)}"}

            def add_issue_labels(self, repo, number, labels):
                return [{"name": labels[0]}]

        progress_messages = []
        client = FakeClient()
        review_input = {
            "repo": "owner/repo",
            "pr": {"number": 1, "head": {"sha": "abc"}},
            "changed_files": [],
            "comments": {"issue_comments": [], "review_comments": [], "reviews": []},
        }
        deterministic_findings = [
            {
                "id": "ci-compile",
                "title": "compile pipeline job failed",
                "category": "ci_pipeline_failure",
                "finding_category": "introduced_bug",
                "causality": "exposed_by_pr",
                "severity": "high",
                "confidence": "high",
                "merge_blocking": True,
                "publication_destination": "inline_blocking",
                "file": None,
                "line": None,
                "code_reference": "GitHub Actions job compile",
                "evidence": "`compile` concluded `failure`. Failed job: https://github.example/actions/runs/123/job/1",
                "root_cause": (
                    "`compile` failed because the SOAR instance at 10.1.66.159 timed out "
                    "during app installation."
                ),
                "why_it_matters": "The compile failure blocks merge.",
                "suggested_fix": "Verify network access to the SOAR instance and rerun compile.",
                "url": "https://github.example/actions/runs/123/job/1",
            },
            {
                "id": "ci-build",
                "title": "build pipeline job failed",
                "category": "ci_pipeline_failure",
                "finding_category": "introduced_bug",
                "causality": "exposed_by_pr",
                "severity": "high",
                "confidence": "high",
                "merge_blocking": True,
                "publication_destination": "inline_blocking",
                "file": None,
                "line": None,
                "code_reference": "GitHub Actions job build",
                "evidence": "`build` concluded `failure`. Failed job: https://github.example/actions/runs/123/job/2",
                "root_cause": (
                    "`build` failed because the SOAR instance at 10.1.66.159 timed out "
                    "during app installation."
                ),
                "why_it_matters": "The build failure blocks merge.",
                "suggested_fix": "Verify network access to the SOAR instance and rerun build.",
                "url": "https://github.example/actions/runs/123/job/2",
            },
        ]

        with tempfile.TemporaryDirectory() as tmp:
            result = publish_target_pipeline_failure_comments(
                client,
                review_input,
                deterministic_findings,
                run_dir=Path(tmp),
                publish_comments=True,
                allow_duplicates=False,
                progress=progress_messages.append,
            )
            plan = json.loads((Path(tmp) / "ci_pipeline_comment_plan.json").read_text(encoding="utf-8"))

        self.assertEqual(result["posted"], 1)
        self.assertEqual(len(plan["comments"]), 1)
        self.assertEqual(len(client.bodies), 1)
        self.assertEqual(plan["comments"][0]["pipeline_failed_jobs"], ["build", "compile"])
        self.assertEqual(review_input["ci"]["published_target_pipeline_jobs"], ["build", "compile"])
        self.assertIn("`build` [failed job](https://github.example/actions/runs/123/job/2)", client.bodies[0])
        self.assertIn("`compile` [failed job](https://github.example/actions/runs/123/job/1)", client.bodies[0])
        self.assertTrue(any("grouped 2 failed job(s) into 1 root-cause comment(s)" in message for message in progress_messages))

    def test_published_precommit_pipeline_comment_suppresses_final_duplicates(self):
        review_input = {
            "ci": {"published_target_pipeline_jobs": ["pre-commit"]},
        }
        review_output = {
            "summary": "Review found issues.",
            "overall_status": "blocked_by_ci",
            "safe_to_publish": True,
            "findings": [
                {
                    "id": "precommit-duplicate",
                    "title": "Pre-commit or connector hook failures need to be resolved",
                    "category": "precommit",
                    "confidence": "high",
                    "evidence": "Detect secrets failed: Secret Keyword in .github/workflows/agentic-pr-review.yml:206",
                    "suggested_fix": "Remove the secret-like value from the diff.",
                },
                {
                    "id": "model-duplicate",
                    "title": "Potential secret detected in workflow",
                    "category": "introduced_bug",
                    "confidence": "high",
                    "evidence": "The detect-secrets hook flagged Secret Keyword in the workflow.",
                    "suggested_fix": "Remove the secret-like value or allowlist a verified false positive.",
                },
                {
                    "id": "real-code-finding",
                    "title": "Request timeout is missing",
                    "category": "introduced_bug",
                    "confidence": "high",
                    "evidence": "The changed request call has no timeout argument.",
                    "suggested_fix": "Pass a bounded timeout to the request.",
                },
            ],
        }
        progress_messages = []

        filtered = suppress_redundant_published_pipeline_findings(
            review_output,
            review_input,
            progress=progress_messages.append,
        )

        self.assertEqual([finding["id"] for finding in filtered["findings"]], ["real-code-finding"])
        self.assertEqual(filtered["overall_status"], "blocked_by_ci")
        self.assertEqual(filtered["behavioral_verification"]["suppressed_duplicate_pipeline_comment_count"], 2)
        self.assertTrue(any("early target pipeline comment" in message for message in progress_messages))

    def test_published_compile_pipeline_comment_suppresses_ci_synthesis_duplicate(self):
        review_input = {
            "ci": {"published_target_pipeline_jobs": ["compile"]},
        }
        review_output = {
            "summary": "Review found issues.",
            "overall_status": "blocked_by_ci",
            "safe_to_publish": True,
            "findings": [
                {
                    "id": "compile-wrapper",
                    "title": "Compile job blocks merge",
                    "category": "ci_synthesis",
                    "confidence": "high",
                    "evidence": "The compile job failed with ConnectTimeout while installing the app.",
                    "suggested_fix": "Verify SOAR instance network access and rerun compile.",
                },
                {
                    "id": "metadata",
                    "title": "Output metadata is missing",
                    "category": "introduced_bug",
                    "confidence": "high",
                    "evidence": "The new action returns data without matching metadata.",
                    "suggested_fix": "Add the missing output datapath.",
                },
            ],
        }

        filtered = suppress_redundant_published_pipeline_findings(
            review_output,
            review_input,
            progress=None,
        )

        self.assertEqual([finding["id"] for finding in filtered["findings"]], ["metadata"])
        self.assertIn("compile", filtered["model_notes"])

    def test_high_confidence_pipeline_failure_does_not_call_model_fallback(self):
        deterministic_findings = [
            {
                "title": "compile pipeline job failed",
                "category": "ci_pipeline_failure",
                "confidence": "high",
                "ci_diagnosis_needs_model": False,
                "root_cause": "`compile` reported `SyntaxError: invalid syntax`",
            }
        ]

        def fail_factory():
            raise AssertionError("model fallback should not be called")

        refined = refine_low_confidence_pipeline_findings(
            deterministic_findings,
            reviewer_factory=fail_factory,
            progress=lambda _message: None,
        )

        self.assertEqual(refined, deterministic_findings)

    def test_low_confidence_pipeline_failure_uses_model_fallback(self):
        class FakeReviewer:
            def diagnose_ci_failure(self, **kwargs):
                self.kwargs = kwargs
                return {
                    "diagnosis_status": "confirmed",
                    "root_cause": "`compile` failed because the SOAR instance was unreachable",
                    "failure_scenario": "The compile job timed out connecting to https://10.1.66.159/.",
                    "suggested_fix": "Verify the SOAR instance IP and runner network access, then rerun compile.",
                    "evidence_lines": [
                        "ConnectTimeout",
                        "SDKfied app installation failed on 10.1.66.159 after 3 attempts",
                    ],
                    "confidence": "high",
                    "confidence_score": 0.94,
                }

        reviewer = FakeReviewer()
        deterministic_findings = [
            {
                "title": "compile pipeline job failed",
                "category": "ci_pipeline_failure",
                "code_reference": "GitHub Actions job compile",
                "confidence": "low",
                "confidence_score": 0.4,
                "ci_diagnosis_needs_model": True,
                "evidence": "`compile` concluded `failure`. Root cause: `compile` failed on traceback noise.",
                "root_cause": "`compile` failed on traceback noise",
                "suggested_fix": "Open the compile job log, fix the first concrete Python/package/metadata error.",
                "pipeline_failed_steps": ["Compile Application"],
                "pipeline_log_excerpt": (
                    "ConnectTimeout\n"
                    "SDKfied app installation failed on 10.1.66.159 after 3 attempts"
                ),
            }
        ]

        refined = refine_low_confidence_pipeline_findings(
            deterministic_findings,
            reviewer_factory=lambda: reviewer,
            progress=lambda _message: None,
        )

        self.assertFalse(refined[0]["ci_diagnosis_needs_model"])
        self.assertEqual(refined[0]["confidence"], "high")
        self.assertIn("SOAR instance was unreachable", refined[0]["root_cause"])
        self.assertIn("SDKfied app installation failed", refined[0]["evidence"])
        self.assertIn("runner network access", refined[0]["suggested_fix"])

    def test_failed_model_fallback_downgrades_only_ambiguous_pipeline_findings(self):
        deterministic_findings = [
            {
                "title": "compile pipeline job failed",
                "category": "ci_pipeline_failure",
                "code_reference": "GitHub Actions job compile",
                "confidence": "low",
                "confidence_score": 0.4,
                "ci_diagnosis_needs_model": True,
                "evidence": "`compile` concluded `failure`.",
                "root_cause": "`compile` failed on traceback noise",
                "suggested_fix": "Open the compile job log.",
            },
            {
                "title": "Pre-commit found a secret",
                "category": "precommit",
                "confidence": "high",
                "evidence": "detect-secrets failed",
            },
        ]

        refined = refine_low_confidence_pipeline_findings(
            deterministic_findings,
            reviewer_factory=lambda: (_ for _ in ()).throw(RuntimeError("missing model env")),
            progress=lambda _message: None,
        )

        self.assertIn("did not prove a more specific root cause", refined[0]["root_cause"])
        self.assertEqual(refined[0]["confidence"], "medium")
        self.assertEqual(refined[1], deterministic_findings[1])


if __name__ == "__main__":
    unittest.main()
