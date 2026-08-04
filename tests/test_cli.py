import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agentic_pr_review.cli import build_parser, publish_target_pipeline_failure_comments, write_artifacts
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
                "code_reference": "GitHub Actions job `build`",
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
        self.assertIn("agentic-pr-review:", review_input["comments"]["issue_comments"][0]["body"])
        self.assertTrue(any("before model review" in message for message in progress_messages))


if __name__ == "__main__":
    unittest.main()
