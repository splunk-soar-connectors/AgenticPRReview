import json
import tempfile
from pathlib import Path
import unittest

from scripts.evaluate_gold_reviews import evaluate_gold_file


class GoldReviewEvalTest(unittest.TestCase):
    def test_packaged_gold_file_has_unique_ids(self):
        gold_path = Path(__file__).parents[1] / "examples" / "gold_reviews.connector_sdk.json"
        gold = json.loads(gold_path.read_text(encoding="utf-8"))

        pattern_ids = [item["id"] for item in gold["pattern_catalog"]]
        case_ids = [item["id"] for item in gold["cases"]]
        expected_ids = [
            expected["id"]
            for case in gold["cases"]
            for expected in case.get("expected", [])
        ]

        self.assertEqual(len(pattern_ids), len(set(pattern_ids)))
        self.assertEqual(len(case_ids), len(set(case_ids)))
        self.assertEqual(len(expected_ids), len(set(expected_ids)))
        self.assertGreaterEqual(len(case_ids), 10)

    def test_matches_expected_and_forbidden_patterns(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = root / "runs" / "case"
            artifacts.mkdir(parents=True)
            (artifacts / "review_input.json").write_text(
                json.dumps(
                    {
                        "repo": "owner/repo",
                        "pr": {"number": 1, "head": {"sha": "abc"}},
                        "changed_files": [],
                        "full_files": {},
                        "ci": {"check_runs": [], "statuses": []},
                    }
                ),
                encoding="utf-8",
            )
            (artifacts / "review_output.json").write_text(
                json.dumps(
                    {
                        "findings": [
                            {
                                "title": "OAuth path is PAT-only",
                                "category": "api_auth_correctness",
                                "file": "src/app.py",
                                "line": 1,
                                "evidence": "Only PAT auth is implemented.",
                                "why_it_matters": "OAuth assets fail.",
                                "suggested_fix": "Implement OAuth or remove docs.",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            gold = {
                "cases": [
                    {
                        "id": "case",
                        "artifacts_dir": "runs/case",
                        "expected": [{"id": "pat-only", "terms": ["pat-only", "oauth"]}],
                        "forbidden": [{"id": "read-only", "terms": ["read_only=false"]}],
                    }
                ]
            }

            result = evaluate_gold_file(gold, repo_root=root)

        self.assertTrue(result["passed"])
        case = result["cases"][0]
        self.assertTrue(case["expected_results"][0]["matched"])
        self.assertFalse(case["forbidden_results"][0]["matched"])

    def test_forbidden_patterns_default_to_current_deterministic_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = root / "runs" / "case"
            artifacts.mkdir(parents=True)
            (artifacts / "review_input.json").write_text(
                json.dumps(
                    {
                        "repo": "owner/repo",
                        "pr": {"number": 1, "head": {"sha": "abc"}},
                        "changed_files": [],
                        "full_files": {},
                        "ci": {"check_runs": [], "statuses": []},
                    }
                ),
                encoding="utf-8",
            )
            (artifacts / "review_output.json").write_text(
                json.dumps(
                    {
                        "findings": [
                            {
                                "title": "Old stale model finding",
                                "evidence": "make_request should use read_only=False",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            gold = {
                "cases": [
                    {
                        "id": "case",
                        "artifacts_dir": "runs/case",
                        "forbidden": [
                            {
                                "id": "stale-read-only",
                                "terms": ["make_request", "read_only=False"],
                            }
                        ],
                    }
                ]
            }

            result = evaluate_gold_file(gold, repo_root=root)

        self.assertTrue(result["passed"])
        self.assertFalse(result["cases"][0]["forbidden_results"][0]["matched"])

    def test_artifact_quality_marks_unparseable_sdk_app_as_bad(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = root / "runs" / "case"
            artifacts.mkdir(parents=True)
            (artifacts / "review_input.json").write_text(
                json.dumps(
                    {
                        "repo": "owner/repo",
                        "pr": {"number": 1, "head": {"sha": "abc"}},
                        "changed_files": [
                            {"filename": "pyproject.toml", "status": "added"},
                            {"filename": "src/app.py", "status": "added"},
                        ],
                        "full_files": {
                            "pyproject.toml": "[tool.soar.app]\nmain_module = \"src.app:app\"\n",
                            "src/app.py": "from soar_sdk.app import App\napp = App(\n",
                        },
                        "ci": {"check_runs": [], "statuses": []},
                    }
                ),
                encoding="utf-8",
            )
            gold = {"cases": [{"id": "case", "artifacts_dir": "runs/case"}]}

            result = evaluate_gold_file(gold, repo_root=root)

        case = result["cases"][0]
        self.assertFalse(case["passed"])
        self.assertEqual(case["artifact_quality"]["status"], "bad")
        self.assertTrue(any("does not compile" in note for note in case["artifact_quality"]["notes"]))


if __name__ == "__main__":
    unittest.main()
