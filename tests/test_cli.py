import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agentic_pr_review.cli import build_parser, write_artifacts
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


if __name__ == "__main__":
    unittest.main()
