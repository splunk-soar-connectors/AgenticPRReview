import unittest
from pathlib import Path

from agentic_pr_review.cli import build_parser


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


if __name__ == "__main__":
    unittest.main()
