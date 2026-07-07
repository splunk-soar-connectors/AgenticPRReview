import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from agentic_pr_review.historical_context import enrich_with_historical_context, select_historical_examples


class HistoricalContextTest(unittest.TestCase):
    def test_selects_relevant_historical_examples_and_excludes_same_pr(self):
        review_input = {
            "repo": "example-connectors/remoteaccess",
            "pr": {"number": 2, "title": "Fix OAuth token exchange", "body": ""},
            "changed_files": [
                {
                    "filename": "remoteaccess_connector.py",
                    "patch": (
                        "@@ -10,2 +10,4 @@\n"
                        "+payload = {'client_id': client_id, 'client_secret': client_secret}\n"
                        "+response = requests.post(token_url, json=payload)\n"
                    ),
                }
            ],
            "comments": {"issue_comments": [], "review_comments": [], "reviews": []},
            "ci": {"check_runs": [], "statuses": []},
        }
        examples = [
            {
                "pattern_id": "oauth_basic_auth_contract",
                "category": "api_auth_correctness",
                "pattern_title": "OAuth/client-credentials flow does not match API contract",
                "implementation_hint": "Check token requests for required Basic auth and body encoding.",
                "repo": "example-connectors/remoteaccess",
                "pr_number": 1,
                "body": "The token exchange needs HTTP Basic auth instead of putting client_id/client_secret in the body.",
                "path": "remoteaccess_connector.py",
                "diff_hunk": "+response = requests.post(token_url, json=payload)",
                "file_evidence": [
                    {
                        "filename": "remoteaccess_connector.py",
                        "patch_excerpt": "+payload = {'client_id': client_id, 'client_secret': client_secret}",
                    }
                ],
            },
            {
                "pattern_id": "oauth_basic_auth_contract",
                "category": "api_auth_correctness",
                "repo": "example-connectors/remoteaccess",
                "pr_number": 2,
                "body": "same PR should not leak into its own review",
                "path": "remoteaccess_connector.py",
            },
            {
                "pattern_id": "precommit_failure_specific",
                "category": "precommit",
                "repo": "example-connectors/remoteaccess",
                "pr_number": 3,
                "body": "Issue: CI shows pre-commit failure. <!-- agentic-pr-review:finding-123 -->",
                "path": "remoteaccess_connector.py",
            },
            {
                "pattern_id": "generated_docs_stale",
                "category": "docs_pr_accuracy",
                "repo": "example-connectors/slack",
                "pr_number": 9,
                "body": "Run build-docs for README drift.",
                "path": "manual_readme_content.md",
            },
        ]

        matches = select_historical_examples(review_input, examples, max_examples=3, min_score=8)

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["pattern_id"], "oauth_basic_auth_contract")
        self.assertEqual(matches[0]["pr_number"], 1)
        self.assertIn("same_path:remoteaccess_connector.py", matches[0]["why_selected"])

    def test_enrich_loads_jsonl_and_records_collector_notes(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "training_examples.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "pattern_id": "polling_sdi_dedup",
                        "category": "polling_checkpoint",
                        "pattern_title": "Polling SDI/checkpoint design can duplicate or skip events",
                        "implementation_hint": "Check SDI stability and checkpoint timing.",
                        "repo": "example-connectors/example",
                        "pr_number": 7,
                        "body": "Saving state before ingestion succeeds can drop events.",
                        "path": "example_connector.py",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            review_input = {
                "repo": "example-connectors/example",
                "pr": {"number": 8, "title": "on_poll checkpoint update", "body": ""},
                "changed_files": [
                    {
                        "filename": "example_connector.py",
                        "patch": "+self.save_state({'checkpoint': latest_timestamp})\n+return self.save_artifact(event)",
                    }
                ],
                "comments": {},
                "ci": {},
                "collector_notes": {},
            }

            enriched = enrich_with_historical_context(review_input, examples_path=str(path), max_examples=2)

        self.assertTrue(enriched["historical_context"]["enabled"])
        self.assertEqual(enriched["collector_notes"]["historical_context_match_count"], 1)
        self.assertEqual(enriched["historical_context"]["matches"][0]["pattern_id"], "polling_sdi_dedup")

    def test_generic_sanitized_path_matches_connector_files(self):
        review_input = {
            "repo": "example/repo",
            "pr": {"number": 9, "title": "OAuth connector update", "body": ""},
            "changed_files": [{"filename": "vendor_connector.py", "patch": "+response = requests.post(token_url, json=payload)"}],
            "comments": {},
            "ci": {},
        }
        examples = [
            {
                "pattern_id": "oauth_basic_auth_contract",
                "category": "api_auth_correctness",
                "pattern_title": "OAuth/client-credentials flow does not match API contract",
                "implementation_hint": "Check token requests for required Basic auth.",
                "repo": "repo-0001",
                "pr_number": 1,
                "body": "OAuth token requests need correct authorization handling.",
                "path": "connector.py",
            }
        ]

        matches = select_historical_examples(review_input, examples, max_examples=1, min_score=8)

        self.assertEqual(len(matches), 1)
        self.assertIn("same_file_kind:connector.py", matches[0]["why_selected"])


if __name__ == "__main__":
    unittest.main()
