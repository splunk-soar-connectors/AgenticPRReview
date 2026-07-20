import unittest

from agentic_pr_review.deep_review import (
    DeepPRCollector,
    build_chunk_review_input,
    chunk_unified_diff,
    generate_unified_diff,
    plan_circuit_review_chunks,
    split_chunk_for_adaptive_retry,
)


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

    def test_large_python_diff_is_chunked_by_function_scope(self):
        def function_source(name: str, value: int) -> str:
            body = [f"def {name}():"]
            body.extend(f"    item_{idx} = {value + idx}" for idx in range(90))
            body.append("    return item_0")
            return "\n".join(body)

        base = "\n\n".join(
            [
                function_source("alpha", 1),
                function_source("beta", 1000),
                function_source("gamma", 2000),
            ]
        )
        head = base.replace("    item_7 = 8", "    item_7 = 9008").replace(
            "    item_12 = 2012",
            "    item_12 = 9912",
        )
        diff = generate_unified_diff(base, head, base_path="connector.py", head_path="connector.py")

        chunks = chunk_unified_diff(
            {
                "filename": "connector.py",
                "status": "modified",
                "additions": 2,
                "deletions": 2,
                "changes": 200,
            },
            diff,
            max_chars=60_000,
            head_text=head,
            base_text=base,
        )

        scopes = {chunk.get("python_scope") for chunk in chunks}
        self.assertGreaterEqual(len(chunks), 2)
        self.assertIn("alpha", scopes)
        self.assertIn("gamma", scopes)
        self.assertTrue(all(chunk.get("chunk_strategy") == "python_function" for chunk in chunks))

        review_input = {
            "repo": "owner/repo",
            "pr": {"number": 1, "title": "test"},
            "full_files": {"connector.py": head},
            "base_files": {"connector.py": base},
            "comments": {"issue_comments": [], "review_comments": [], "reviews": []},
            "ci": {"check_runs": []},
            "collector_notes": {},
        }
        scoped_input = build_chunk_review_input(review_input, next(chunk for chunk in chunks if chunk.get("python_scope") == "alpha"))

        self.assertEqual(scoped_input["review_scope"]["chunk_strategy"], "python_function")
        self.assertEqual(scoped_input["review_scope"]["python_scope"], "alpha")
        self.assertEqual(scoped_input["review_packet"]["change_stats"]["python_scope"], "alpha")

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

        self.assertEqual(chunk_input["review_scope"]["type"], "circuit_review_packet")
        self.assertEqual(chunk_input["review_packet"]["packet_type"], "focused_circuit_pr_review_packet")
        self.assertIn("connector.py", chunk_input["full_files"])
        self.assertIn("sample.json", chunk_input["full_files"])
        self.assertNotIn("templates/ioc_view.html", chunk_input["full_files"])
        self.assertNotIn("unrelated.txt", chunk_input["full_files"])
        self.assertEqual(chunk_input["historical_context"]["matched_example_count"], 2)
        self.assertEqual(chunk_input["historical_context"]["matches"][0]["path"], "connector.py")

    def test_adaptive_retry_chunk_uses_smaller_context_window(self):
        review_input = {
            "repo": "owner/repo",
            "pr": {"number": 1, "title": "test"},
            "full_files": {
                "connector.py": "x" * 1000,
                "sample.json": "y" * 1000,
            },
            "base_files": {},
            "comments": {"issue_comments": [], "review_comments": [], "reviews": []},
            "ci": {"check_runs": []},
            "collector_notes": {},
        }
        chunk = {
            "id": "connector.py:1:retry-1",
            "parent_chunk_id": "connector.py:1",
            "adaptive_retry": True,
            "context_char_limit": 100,
            "path": "connector.py",
            "status": "modified",
            "chunk_index": 1,
            "chunk_total": 2,
            "diff": "@@ -1 +1 @@\n-old\n+new",
        }

        chunk_input = build_chunk_review_input(review_input, chunk)

        self.assertTrue(chunk_input["review_scope"]["adaptive_retry"])
        self.assertEqual(chunk_input["review_scope"]["parent_chunk_id"], "connector.py:1")
        self.assertLess(len(chunk_input["full_files"]["connector.py"]), 220)
        self.assertIn("smaller retry slice", chunk_input["review_scope"]["instruction"])

    def test_chunk_review_input_keeps_only_path_relevant_comments_and_ci_logs(self):
        review_input = {
            "repo": "owner/repo",
            "pr": {"number": 1, "title": "test"},
            "full_files": {"connector.py": "def f():\n    return 1\n"},
            "base_files": {},
            "comments": {
                "review_comments": [
                    {"path": "connector.py", "body": "Fix connector.py timeout."},
                    {"path": "other.py", "body": "Unrelated comment."},
                ],
                "issue_comments": [
                    {"body": "connector.py still has a failure path."},
                    {"body": "General unrelated discussion."},
                ],
                "reviews": [{"body": "overall review body"}],
            },
            "ci": {
                "check_runs": [
                    {
                        "name": "pre-commit",
                        "status": "completed",
                        "conclusion": "failure",
                        "output": {"summary": "s" * 2000, "text": "t" * 3000},
                    }
                ],
                "failed_check_logs": [
                    {"name": "ruff", "body": "connector.py:10: error"},
                    {"name": "ruff", "body": "other.py:10: error"},
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

        self.assertEqual(len(chunk_input["comments"]["review_comments"]), 1)
        self.assertEqual(chunk_input["comments"]["review_comments"][0]["path"], "connector.py")
        self.assertEqual(len(chunk_input["comments"]["issue_comments"]), 1)
        self.assertIn("connector.py", chunk_input["comments"]["issue_comments"][0]["body"])
        self.assertEqual(len(chunk_input["comments"]["reviews"]), 1)
        self.assertEqual(len(chunk_input["ci"]["failed_check_logs"]), 1)
        self.assertIn("connector.py", chunk_input["ci"]["failed_check_logs"][0]["body"])
        self.assertLess(len(chunk_input["ci"]["check_runs"][0]["output"]["summary"]), 900)
        self.assertLess(len(chunk_input["ci"]["check_runs"][0]["output"]["text"]), 1300)

    def test_split_chunk_for_adaptive_retry_preserves_all_diff_lines(self):
        diff = generate_unified_diff(
            "\n".join(f"old_{idx}" for idx in range(40)),
            "\n".join(f"new_{idx}" for idx in range(40)),
            base_path="connector.py",
            head_path="connector.py",
        )
        chunk = {
            "id": "connector.py:1",
            "path": "connector.py",
            "status": "modified",
            "chunk_index": 1,
            "chunk_total": 1,
            "diff": diff,
        }

        retry_chunks = split_chunk_for_adaptive_retry(chunk, max_chars=350, context_char_limit=123)

        self.assertGreater(len(retry_chunks), 1)
        self.assertTrue(all(item["adaptive_retry"] for item in retry_chunks))
        self.assertTrue(all(item["parent_chunk_id"] == "connector.py:1" for item in retry_chunks))
        self.assertTrue(all(item["context_char_limit"] == 123 for item in retry_chunks))
        joined = "\n".join(item["diff"] for item in retry_chunks)
        self.assertIn("+new_0", joined)
        self.assertIn("+new_39", joined)

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

    def test_circuit_planner_skips_generated_and_metadata_only_chunks(self):
        chunks = [
            {
                "id": "README.md:1",
                "path": "README.md",
                "status": "modified",
                "diff": "@@ -1 +1 @@\n-old\n+new",
            },
            {
                "id": "LICENSE:1",
                "path": "LICENSE",
                "status": "modified",
                "diff": "@@ -1 +1 @@\n-2025\n+2025-2026",
            },
            {
                "id": "__init__.py:1",
                "path": "__init__.py",
                "status": "modified",
                "diff": "@@ -1 +1 @@\n-# Copyright 2025\n+# Copyright 2025-2026",
            },
            {
                "id": "connector.py:1",
                "path": "connector.py",
                "status": "modified",
                "diff": "@@ -1 +1 @@\n-response = requests.get(url, timeout=30)\n+response = requests.get(url)",
            },
        ]

        planned, skipped = plan_circuit_review_chunks(chunks, [])

        self.assertEqual([item["path"] for item in planned], ["connector.py"])
        self.assertEqual({item["path"] for item in skipped}, {"README.md", "LICENSE", "__init__.py"})
        self.assertEqual(planned[0]["model_review_priority"], "high")

    def test_circuit_planner_keeps_manual_docs_with_deterministic_finding(self):
        chunks = [
            {
                "id": "manual_readme_content.md:1",
                "path": "manual_readme_content.md",
                "status": "modified",
                "diff": "@@ -1 +1 @@\n-old\n+new",
            }
        ]

        planned, skipped = plan_circuit_review_chunks(
            chunks,
            [{"file": "manual_readme_content.md", "title": "docs generator input mismatch"}],
        )

        self.assertEqual(len(planned), 1)
        self.assertEqual(skipped, [])
        self.assertEqual(planned[0]["model_review_reason"], "deterministic finding targets this file")

    def test_chunk_review_input_builds_focused_packet_context(self):
        big_file = "\n".join(f"line_{idx}" for idx in range(1, 1_200))
        review_input = {
            "repo": "owner/repo",
            "pr": {"number": 1, "title": "test"},
            "full_files": {
                "connector.py": big_file,
                "sample.json": '{"actions": [{"identifier": "lookup_ioc"}]}',
            },
            "base_files": {
                "connector.py": big_file.replace("line_100", "old_line_100"),
            },
            "comments": {"issue_comments": [], "review_comments": [], "reviews": []},
            "ci": {"check_runs": []},
            "collector_notes": {},
        }
        chunk = {
            "id": "connector.py:1",
            "path": "connector.py",
            "status": "modified",
            "chunk_index": 1,
            "chunk_total": 1,
            "diff": "@@ -100,3 +100,3 @@\n-line_100\n+lookup_ioc = line_100\n line_101\n line_102",
        }

        chunk_input = build_chunk_review_input(review_input, chunk)

        packet = chunk_input["review_packet"]
        self.assertIn("lines 65-", packet["head_context_files"]["connector.py"])
        self.assertIn("100:", packet["head_context_files"]["connector.py"])
        self.assertIn("lookup_ioc", packet["changed_hunks"][0]["added_excerpt"])
        self.assertLess(len(chunk_input["full_files"]["connector.py"]), len(big_file))

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
