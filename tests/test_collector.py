import unittest

from agentic_pr_review.collector import (
    extract_actions_job_id,
    is_actionable_ci_line,
    is_relevant_full_file,
    snippets_for_terms,
    summarize_ci_log,
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


if __name__ == "__main__":
    unittest.main()
