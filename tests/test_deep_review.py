import unittest

from agentic_pr_review.deep_review import DeepPRCollector, build_chunk_review_input, chunk_unified_diff, generate_unified_diff


class DeepReviewHelpersTest(unittest.TestCase):
    def test_generate_unified_diff_preserves_removed_and_added_lines(self):
        diff = generate_unified_diff(
            "def f():\n    return requests.get(url, timeout=30)\n",
            "def f():\n    return requests.get(url)\n",
            base_path="connector.py",
            head_path="connector.py",
        )

        self.assertIn("-    return requests.get(url, timeout=30)", diff)
        self.assertIn("+    return requests.get(url)", diff)

    def test_chunk_unified_diff_keeps_file_metadata(self):
        diff = generate_unified_diff(
            "\n".join(f"old_{idx}" for idx in range(20)),
            "\n".join(f"new_{idx}" for idx in range(20)),
            base_path="connector.py",
            head_path="connector.py",
        )
        chunks = chunk_unified_diff(
            {"filename": "connector.py", "status": "modified", "additions": 20, "deletions": 20, "changes": 40},
            diff,
            max_chars=220,
        )

        self.assertGreaterEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["path"], "connector.py")
        self.assertIn("@@", chunks[0]["diff"])

    def test_build_chunk_review_input_includes_primary_and_app_context(self):
        review_input = {
            "repo": "owner/repo",
            "pr": {"number": 1, "title": "test"},
            "full_files": {
                "connector.py": "def f():\n    return 1\n",
                "sample.json": '{"actions": []}',
                "threatintel_view.py": "def ioc_view(request):\n    return None\n",
                "templates/ioc_view.html": "<input type=\"date\" name=\"date\">",
                "unrelated.txt": "ignore",
            },
            "base_files": {"connector.py": "def f():\n    return 0\n"},
            "comments": {"issue_comments": [], "review_comments": [], "reviews": []},
            "ci": {"check_runs": []},
            "historical_context": {
                "enabled": True,
                "matches": [
                    {
                        "pattern_id": "api_auth_correctness",
                        "path": "connector.py",
                        "human_comment": "Add a timeout to this auth request.",
                        "file_evidence": [{"filename": "connector.py"}],
                    },
                    {
                        "pattern_id": "docs_pr_accuracy",
                        "path": "manual_readme_content.md",
                        "human_comment": "Docs drift.",
                        "file_evidence": [{"filename": "manual_readme_content.md"}],
                    },
                ],
            },
            "collector_notes": {},
        }
        chunk = {
            "id": "connector.py:1",
            "path": "connector.py",
            "status": "modified",
            "chunk_index": 1,
            "chunk_total": 1,
            "diff": "@@ -1 +1 @@\n-return 0\n+return 1",
        }

        chunk_input = build_chunk_review_input(review_input, chunk)

        self.assertEqual(chunk_input["review_scope"]["type"], "deep_file_chunk")
        self.assertIn("connector.py", chunk_input["full_files"])
        self.assertIn("sample.json", chunk_input["full_files"])
        self.assertNotIn("templates/ioc_view.html", chunk_input["full_files"])
        self.assertNotIn("unrelated.txt", chunk_input["full_files"])
        self.assertEqual(chunk_input["historical_context"]["matched_example_count"], 2)
        self.assertEqual(chunk_input["historical_context"]["matches"][0]["path"], "connector.py")

    def test_build_chunk_review_input_includes_related_view_template_context(self):
        review_input = {
            "repo": "owner/repo",
            "pr": {"number": 1, "title": "test"},
            "full_files": {
                "sample.json": '{"actions": []}',
                "threatintel_view.py": "def ioc_view(request):\n    return None\n",
                "templates/ioc_view.html": "<input type=\"date\" name=\"date\">",
                "default/data/ui/dashboards/ioc_view.xml": "<dashboard><row /></dashboard>",
                "connector.py": "ignore",
            },
            "base_files": {},
            "comments": {"issue_comments": [], "review_comments": [], "reviews": []},
            "ci": {"check_runs": []},
            "collector_notes": {},
        }
        chunk = {
            "id": "threatintel_view.py:1",
            "path": "threatintel_view.py",
            "status": "modified",
            "chunk_index": 1,
            "chunk_total": 1,
            "diff": "@@ -1 +1 @@\n-def old\n+def ioc_view",
        }

        chunk_input = build_chunk_review_input(review_input, chunk)

        self.assertIn("threatintel_view.py", chunk_input["full_files"])
        self.assertIn("templates/ioc_view.html", chunk_input["full_files"])
        self.assertIn("default/data/ui/dashboards/ioc_view.xml", chunk_input["full_files"])
        self.assertNotIn("connector.py", chunk_input["full_files"])

    def test_deep_merge_replaces_truncated_github_patch_with_full_diff(self):
        collector = object.__new__(DeepPRCollector)
        full_diff = generate_unified_diff(
            "def f():\n    return requests.get(url, timeout=30)\n",
            "def f():\n    return requests.get(url)\n",
            base_path="connector.py",
            head_path="connector.py",
        )
        review_input = {
            "changed_files": [
                {
                    "filename": "connector.py",
                    "status": "modified",
                    "patch": "@@ -1 +1 @@\n-truncated\n+truncated",
                    "patch_chars": 30,
                    "patch_truncated": True,
                    "patch_missing": False,
                }
            ],
            "full_files": {},
            "base_files": {},
            "collector_notes": {},
        }

        collector._merge_deep_files(  # pylint: disable=protected-access
            review_input,
            {"connector.py": "def f():\n    return requests.get(url)\n"},
            {"connector.py": "def f():\n    return requests.get(url, timeout=30)\n"},
            [],
            {"connector.py": full_diff},
        )

        changed = review_input["changed_files"][0]
        self.assertEqual(changed["patch"], full_diff)
        self.assertFalse(changed["patch_truncated"])
        self.assertTrue(changed["deep_patch_reconstructed"])
        self.assertTrue(changed["github_patch_truncated"])
        self.assertEqual(review_input["collector_notes"]["deep_reconstructed_patch_count"], 1)


if __name__ == "__main__":
    unittest.main()
