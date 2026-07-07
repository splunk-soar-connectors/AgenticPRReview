import json
import unittest

from agentic_pr_review.prompt import SYSTEM_PROMPT, build_collection_diagnostics, build_user_prompt


class PromptPackingTest(unittest.TestCase):
    def test_historical_reviews_are_reference_not_authority(self):
        self.assertIn("pattern guidance only", SYSTEM_PROMPT)
        self.assertIn("do not outrank current PR code", SYSTEM_PROMPT)
        self.assertIn("verify against the current PR", SYSTEM_PROMPT)
        self.assertNotIn("Historical high-value review patterns to prioritize", SYSTEM_PROMPT)

    def test_large_pr_prompt_keeps_deterministic_findings_before_bulky_context(self):
        review_input = {
            "repo": "owner/repo",
            "pr": {"number": 1, "title": "large", "body": "", "head": {"sha": "abc"}},
            "changed_files": [
                {
                    "filename": "connector.py",
                    "patch": "+" + ("x" * 50_000),
                    "additions": 4000,
                    "deletions": 0,
                    "patch_truncated": True,
                }
            ],
            "full_files": {"connector.py": "x" * 80_000},
            "base_files": {},
            "comments": {"issue_comments": [], "review_comments": [], "reviews": []},
            "ci": {"check_runs": [], "statuses": [], "errors": []},
            "historical_context": {
                "enabled": True,
                "matched_example_count": 1,
                "matches": [
                    {
                        "pattern_id": "oauth_basic_auth_contract",
                        "category": "api_auth_correctness",
                        "human_comment": "Historical reviewer caught missing Basic auth on OAuth token exchange.",
                        "diff_hunk": "+" + ("oauth " * 2000),
                        "file_evidence": [
                            {
                                "filename": "connector.py",
                                "patch_excerpt": "+" + ("client_secret " * 1000),
                            }
                        ],
                    }
                ],
            },
            "collector_notes": {
                "github_auth_mode": "app",
                "changed_file_count": 1,
                "changed_file_patch_count": 1,
                "changed_file_missing_patch_count": 0,
                "changed_file_truncated_patch_count": 1,
                "full_file_count": 1,
                "full_file_missing_count": 0,
            },
        }
        deterministic_findings = [
            {
                "title": "Connector logging method appears misspelled",
                "category": "precommit",
                "confidence": "high",
                "file": "connector.py",
                "line": 1,
                "evidence": "degub_print",
                "suggested_fix": "Rename degub_print to debug_print.",
            }
        ]

        prompt = build_user_prompt(review_input, deterministic_findings, max_chars=16_000)

        self.assertIn("deterministic_findings", prompt)
        self.assertIn("Connector logging method appears misspelled", prompt)
        self.assertIn('"github_auth_mode": "app"', prompt)
        self.assertIn("oauth_basic_auth_contract", prompt)
        self.assertIn("Historical reviewer caught missing Basic auth", prompt)
        self.assertIn("compacted_for_model", prompt)

    def test_collection_diagnostics_include_sdk_review_inventory(self):
        review_input = {
            "pr": {"title": "sdk migration", "body": ""},
            "changed_files": [
                {"filename": "legacy.json", "status": "removed", "patch": ""},
                {"filename": "pyproject.toml", "status": "added", "patch": "+[tool.soar.app]\n+main_module = \"src.app:app\""},
                {"filename": "src/app.py", "status": "added", "patch": "+app = App(...)"},
            ],
            "full_files": {
                "pyproject.toml": "[tool.soar.app]\nmain_module = \"src.app:app\"\n",
                "src/app.py": (
                    "from soar_sdk.app import App\n"
                    "app = App(name='sample', appid='12345678-1234-5678-9012-123456789012', app_type='sandbox', logo='logo.svg', logo_dark='logo_dark.svg', product_vendor='ExampleVendor', product_name='Sample', publisher='ExampleVendor')\n"
                    "@app.test_connectivity()\n"
                    "def test_connectivity() -> None:\n"
                    "    pass\n"
                ),
            },
            "base_files": {
                "legacy.json": json.dumps(
                    {
                        "configuration": {"api_key": {"data_type": "password"}},
                        "actions": [
                            {
                                "identifier": "lookup",
                                "action": "lookup",
                                "render": {"type": "custom"},
                                "output": [],
                            }
                        ],
                    }
                )
            },
            "collector_notes": {},
            "comments": {},
            "ci": {"check_runs": [], "statuses": []},
        }

        diagnostics = build_collection_diagnostics(review_input)
        inventory = diagnostics["sdk_review_inventory"]

        self.assertTrue(inventory["is_sdk_migration"])
        self.assertEqual(inventory["main_module"], "src.app:app")
        self.assertEqual(inventory["legacy_config_fields"], ["api_key"])
        self.assertEqual(inventory["legacy_custom_view_actions"], ["lookup"])
        self.assertEqual(inventory["test_connectivity_count"], 1)


if __name__ == "__main__":
    unittest.main()
