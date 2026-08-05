import unittest
from types import SimpleNamespace
from unittest.mock import patch

from agentic_pr_review.collector import (
    PRCollector,
    extract_actions_job_id,
    is_actionable_ci_line,
    is_relevant_full_file,
    previous_pipeline_failures_from_env,
    snippets_for_terms,
    summarize_ci_log,
    target_pipeline_job_name,
)


class CollectorHelpersTest(unittest.TestCase):
    def test_snippets_for_terms_finds_late_doc_content(self):
        text = ("intro\n" * 1000) + "The parent_message_ts parameter replies in a thread.\n"

        snippets = snippets_for_terms(text, ["parent_message_ts"], window=40)

        self.assertEqual(len(snippets), 1)
        self.assertEqual(snippets[0]["term"], "parent_message_ts")
        self.assertIn("replies in a thread", snippets[0]["snippet"])

    def test_extract_actions_job_id_from_details_url(self):
        url = "https://github.com/example-connectors/slack/actions/runs/123456/job/987654"

        self.assertEqual(extract_actions_job_id(url), "987654")

    def test_target_pipeline_job_name_matches_known_jobs(self):
        self.assertEqual(target_pipeline_job_name("pre-commit"), "pre-commit")
        self.assertEqual(target_pipeline_job_name("compile / Compile Application"), "compile")
        self.assertEqual(target_pipeline_job_name("build (sdkfied)"), "build")
        self.assertEqual(target_pipeline_job_name("semantic-release-preview"), "semantic-release-preview")
        self.assertIsNone(target_pipeline_job_name("sanity-test"))

    def test_previous_pipeline_results_env_creates_target_failures(self):
        env = {
            "PREVIOUS_PIPELINE_RESULTS_JSON": (
                '{"jobs":{'
                '"pre-commit":{"result":"success","url":"https://github.example/run"},'
                '"compile":{"result":"failure","url":"https://github.example/run"},'
                '"sanity-test":{"result":"failure","url":"https://github.example/run"},'
                '"semantic-release-preview":{"result":"cancelled","url":"https://github.example/run"}'
                "}}"
            )
        }

        with patch.dict("os.environ", env, clear=True):
            failures = previous_pipeline_failures_from_env()

        self.assertEqual([failure["target_name"] for failure in failures], ["compile", "semantic-release-preview"])
        self.assertEqual(failures[0]["html_url"], "https://github.example/run")
        self.assertEqual(failures[0]["source"], "workflow_needs")

    def test_workflow_run_jobs_collects_target_failure_log(self):
        class FakeClient:
            def list_workflow_run_jobs(self, repo, run_id, *, attempt=None):
                self.args = (repo, run_id, attempt)
                return [
                    {
                        "id": 99,
                        "run_id": int(run_id),
                        "run_attempt": int(attempt),
                        "name": "pre-commit",
                        "status": "completed",
                        "conclusion": "failure",
                        "html_url": "https://github.example/actions/runs/123/job/99",
                        "steps": [
                            {"name": "Setup", "conclusion": "success"},
                            {"name": "Pre-commit", "conclusion": "failure"},
                        ],
                    },
                    {
                        "id": 100,
                        "name": "sanity-test",
                        "status": "completed",
                        "conclusion": "failure",
                        "html_url": "https://github.example/actions/runs/123/job/100",
                    },
                ]

            def get_actions_job_logs(self, repo, job_id):
                self.log_args = (repo, job_id)
                return (
                    b"normal setup\n"
                    b"ruff.....................................................................Failed\n"
                    b"hook id: ruff\n"
                    b"connector.py:1:1: F401 `os` imported but unused\n"
                )

        client = FakeClient()
        collector = PRCollector(client, config=object())

        with patch.dict("os.environ", {"GITHUB_RUN_ID": "123", "GITHUB_RUN_ATTEMPT": "2"}, clear=True):
            result = collector._safe_workflow_run_jobs("owner/repo")

        self.assertEqual(client.args, ("owner/repo", "123", "2"))
        self.assertEqual(client.log_args, ("owner/repo", 99))
        self.assertEqual(len(result["target_job_failures"]), 1)
        failure = result["target_job_failures"][0]
        self.assertEqual(failure["target_name"], "pre-commit")
        self.assertIn("F401", failure["log_excerpt"])

    def test_summarize_ci_log_keeps_actionable_failure_lines(self):
        text = "\n".join(
            [
                "setup line",
                "lots of normal output",
                "ruff.....................................................................Failed",
                "hook id: ruff",
                "connector.py:1:1: F401 `os` imported but unused",
                "final normal output",
            ]
        )

        summary = summarize_ci_log(text)

        self.assertIn("hook id: ruff", summary)
        self.assertIn("F401", summary)
        self.assertNotIn("setup line", summary)

    def test_summarize_ci_log_prioritizes_detect_secrets_failure_over_setup_noise(self):
        text = "\n".join(
            [
                "2026-08-04T19:16:09.1474127Z [INFO] Initializing environment for https://github.com/pre-commit/pre-commit-hooks.",
                "2026-08-04T19:16:09.6462018Z [INFO] Initializing environment for https://github.com/astral-sh/ruff-pre-commit.",
                "2026-08-04T19:16:11.4801993Z [INFO] Initializing environment for https://github.com/hukkin/mdformat.",
                "2026-08-04T19:16:12.1132660Z [INFO] Initializing environment for https://github.com/returntocorp/semgrep.",
                "Detect secrets...........................................................Failed",
                "- hook id: detect-secrets",
                "- exit code: 1",
                "",
                "ERROR: Potential secrets about to be committed to git repo!",
                "",
                "Secret Type: Secret Keyword",
                "Location:    .github/workflows/agentic-pr-review.yml:206",
                "",
                "build docs...............................................................Passed",
            ]
        )

        summary = summarize_ci_log(text)

        self.assertIn("Detect secrets", summary)
        self.assertIn("detect-secrets", summary)
        self.assertIn("Secret Keyword", summary)
        self.assertIn(".github/workflows/agentic-pr-review.yml:206", summary)
        self.assertNotIn("Initializing environment", summary)
        self.assertNotIn("ruff-pre-commit", summary)

    def test_summarize_ci_log_keeps_terminal_soar_install_timeout(self):
        text = "\n".join(
            [
                "Building SDKfied app using soarapps CLI",
                "✓ Package successfully built and saved to: Microsoft 365.tgz",
                *[f"│ {line} │ │ │ traceback frame noise │" for line in range(100, 140)],
                "Installing app on current phantom instance (10.1.66.159)",
                "ConnectTimeout",
                "SDKfied app installation failed on 10.1.66.159 after 3 attempts",
                "Error: Process completed with exit code 1.",
            ]
        )

        summary = summarize_ci_log(text, max_lines=12)

        self.assertIn("Installing app on current phantom instance (10.1.66.159)", summary)
        self.assertIn("ConnectTimeout", summary)
        self.assertIn("SDKfied app installation failed on 10.1.66.159 after 3 attempts", summary)

    def test_ci_log_wrapper_line_is_not_actionable(self):
        self.assertFalse(
            is_actionable_ci_line(
                'echo "Test output saved to /tmp/pytest-output-raw.log (exit code: $PYTEST_EXIT_CODE)"'
            )
        )

    def test_connector_view_assets_are_relevant_full_files(self):
        self.assertTrue(is_relevant_full_file("templates/get_report.html"))
        self.assertTrue(is_relevant_full_file("default/data/ui/dashboards/ioc_view.xml"))
        self.assertTrue(is_relevant_full_file("templates/result.jinja"))

    def test_collect_excludes_agentic_pr_review_workflow_from_review_context(self):
        class FakeClient:
            auth_mode = "fake"

            def __init__(self):
                self.fetch_paths = []

            def get_pr(self, repo, pr_number):
                return {
                    "head": {"sha": "head-sha", "ref": "feature"},
                    "base": {"sha": "base-sha", "ref": "main"},
                    "title": "Test PR",
                    "body": "",
                    "state": "open",
                }

            def list_pr_files(self, repo, pr_number):
                return [
                    {
                        "filename": ".github/workflows/agentic-pr-review.yml",
                        "status": "added",
                        "patch": "+name: bot\n",
                        "additions": 1,
                        "deletions": 0,
                        "changes": 1,
                        "raw_url": "https://example.invalid/workflow",
                    },
                    {
                        "filename": "connector.py",
                        "status": "modified",
                        "patch": "+def action():\n+    return 1\n",
                        "additions": 2,
                        "deletions": 0,
                        "changes": 2,
                    },
                ]

            def list_issue_comments(self, repo, pr_number):
                return []

            def list_review_comments(self, repo, pr_number):
                return []

            def list_reviews(self, repo, pr_number):
                return []

            def get_check_runs(self, repo, head_sha):
                return {"check_runs": []}

            def get_combined_status(self, repo, head_sha):
                return {"state": "success", "total_count": 0, "statuses": []}

            def list_root_contents(self, repo, sha):
                return []

            def fetch_text_file(self, repo, path, sha, max_bytes=None):
                self.fetch_paths.append(path)
                if path == "connector.py":
                    return "def action():\n    return 1\n"
                return ""

        client = FakeClient()
        config = SimpleNamespace(max_patch_chars=10_000, max_file_chars=10_000)
        collector = PRCollector(client, config=config)

        with patch.dict("os.environ", {}, clear=True):
            review_input = collector.collect("owner/repo", 1)

        changed_paths = {item["filename"] for item in review_input["changed_files"]}
        self.assertEqual(changed_paths, {"connector.py"})
        self.assertEqual(
            [item["filename"] for item in review_input["comment_anchor_files"]],
            [".github/workflows/agentic-pr-review.yml"],
        )
        self.assertNotIn(".github/workflows/agentic-pr-review.yml", review_input["full_files"])
        self.assertNotIn(".github/workflows/agentic-pr-review.yml", client.fetch_paths)
        self.assertEqual(
            review_input["collector_notes"]["excluded_review_files"],
            [
                {
                    "path": ".github/workflows/agentic-pr-review.yml",
                    "reason": "bot invocation workflow is excluded from review",
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
