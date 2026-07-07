import unittest
from copy import deepcopy

from agentic_pr_review.github_comments import (
    build_comment_plan as real_build_comment_plan,
    collect_existing_markers,
    parse_right_side_diff_lines,
    publish_comment_plan,
)


def build_comment_plan(review_output, review_input, **kwargs):
    review_output = deepcopy(review_output)
    for key in ("findings", "deterministic_findings"):
        for finding in review_output.get(key, []) or []:
            finding.setdefault("confidence", "high")
            finding.setdefault("why_it_matters", "This can break SOAR connector behavior.")
    return real_build_comment_plan(review_output, review_input, **kwargs)


class GitHubCommentsTest(unittest.TestCase):
    def test_parse_right_side_diff_lines(self):
        patch = (
            "@@ -1,3 +1,4 @@\n"
            " import os\n"
            "-old = 1\n"
            "+new = 2\n"
            " keep = 3\n"
        )

        self.assertEqual(parse_right_side_diff_lines(patch), {1, 2, 3})

    def test_safe_to_publish_false_blocks_comment_plan(self):
        review_input = {
            "repo": "owner/repo",
            "pr": {"number": 1, "head": {"sha": "abc"}},
            "changed_files": [
                {
                    "filename": "connector.py",
                    "patch": "@@ -1,1 +1,1 @@\n+bad = True\n",
                }
            ],
        }
        review_output = {
            "safe_to_publish": False,
            "findings": [
                {
                    "id": "f1",
                    "title": "Wrong value",
                    "category": "validation",
                    "severity": "medium",
                    "file": "connector.py",
                    "line": 1,
                    "evidence": "bad is wrong.",
                    "suggested_fix": "Use the right value.",
                }
            ],
        }

        plan = build_comment_plan(review_output, review_input)

        self.assertEqual(plan["comments"], [])
        self.assertTrue(plan["publish_blocked"])

    def test_removed_sdk_manifest_metadata_finding_is_not_planned(self):
        review_input = {
            "repo": "owner/repo",
            "pr": {"number": 1, "head": {"sha": "abc"}},
            "changed_files": [
                {
                    "filename": "cloudstorage.json",
                    "status": "removed",
                    "patch": "@@ -1,3 +0,0 @@\n-{\"main_module\": \"cloudstorage_connector.py\"}\n",
                },
                {
                    "filename": "pyproject.toml",
                    "status": "modified",
                    "patch": "+[tool.soar.app]\n+main_module = \"src.app:app\"\n",
                },
                {"filename": "src/app.py", "status": "added", "patch": "+app = App(...)"},
                {"filename": "src/asset.py", "status": "added", "patch": "+class Asset(BaseAsset): pass"},
            ],
            "full_files": {
                "pyproject.toml": '[tool.soar.app]\nmain_module = "src.app:app"\n',
                "src/app.py": "from soar_sdk.app import App\napp = App(name='x')\n",
                "src/asset.py": "from soar_sdk.asset import BaseAsset\nclass Asset(BaseAsset): pass\n",
            },
        }
        review_output = {
            "safe_to_publish": True,
            "findings": [
                {
                    "id": "f1",
                    "title": "cloudstorage.json main_module still points to deleted legacy connector file",
                    "category": "soar_metadata",
                    "severity": "critical",
                    "file": "cloudstorage.json",
                    "line": None,
                    "code_reference": "main_module field",
                    "evidence": "cloudstorage.json has `main_module` pointing at cloudstorage_connector.py.",
                    "suggested_fix": "Update `main_module` in cloudstorage.json.",
                }
            ],
        }

        plan = build_comment_plan(review_output, review_input)

        self.assertEqual(plan["comments"], [])

    def test_code_finding_targets_changed_line(self):
        review_input = {
            "repo": "owner/repo",
            "pr": {"number": 1, "head": {"sha": "abc"}},
            "changed_files": [
                {
                    "filename": "connector.py",
                    "patch": "@@ -9,2 +9,2 @@\n context\n-old\n+new\n",
                }
            ],
        }
        review_output = {
            "findings": [
                {
                    "id": "f1",
                    "title": "Wrong value",
                    "category": "validation",
                    "severity": "medium",
                    "file": "connector.py",
                    "line": 10,
                    "evidence": "new uses the wrong value.",
                    "suggested_fix": "Use the validated value instead.",
                    "suggested_code": "new = validated_value",
                }
            ]
        }

        plan = build_comment_plan(review_output, review_input)
        comment = plan["comments"][0]

        self.assertEqual(comment["github_comment_type"], "line")
        self.assertEqual(comment["finding_type"], "code")
        self.assertEqual(comment["confidence"], "high")
        self.assertEqual(comment["path"], "connector.py")
        self.assertEqual(comment["line"], 10)
        self.assertEqual(comment["comment_style"], "code_comment_with_github_suggestion")
        self.assertIn("How to fix:", comment["body"])
        self.assertIn("Current line: `new`", comment["body"])
        self.assertIn("```suggestion\nnew = validated_value\n```", comment["body"])

    def test_code_finding_without_exact_code_does_not_add_suggestion_block(self):
        review_input = {
            "repo": "owner/repo",
            "pr": {"number": 1, "head": {"sha": "abc"}},
            "changed_files": [
                {
                    "filename": "connector.py",
                    "patch": "@@ -9,2 +9,2 @@\n context\n-old\n+new\n",
                }
            ],
        }
        review_output = {
            "findings": [
                {
                    "id": "f1",
                    "title": "Wrong value",
                    "category": "validation",
                    "severity": "medium",
                    "file": "connector.py",
                    "line": 10,
                    "evidence": "new uses the wrong value.",
                    "suggested_fix": "Use the validated value instead.",
                }
            ]
        }

        plan = build_comment_plan(review_output, review_input)
        comment = plan["comments"][0]

        self.assertEqual(comment["comment_style"], "code_comment_with_suggested_fix")
        self.assertNotIn("```suggestion", comment["body"])

    def test_finding_outside_changed_hunk_is_not_planned(self):
        review_input = {
            "repo": "owner/repo",
            "pr": {"number": 1, "head": {"sha": "abc"}},
            "changed_files": [
                {
                    "filename": "connector.py",
                    "status": "modified",
                    "patch": "@@ -9,2 +9,2 @@\n context\n-old\n+new\n",
                }
            ],
        }
        review_output = {
            "findings": [
                {
                    "id": "f1",
                    "title": "Old missing timeout elsewhere in file",
                    "category": "api_auth_correctness",
                    "severity": "medium",
                    "confidence": "high",
                    "file": "connector.py",
                    "line": 200,
                    "evidence": "`requests.get(url)` at line 200 has no timeout.",
                    "why_it_matters": "A hung request can tie up a worker.",
                    "suggested_fix": "Add a bounded `timeout=` to this request.",
                }
            ]
        }

        plan = build_comment_plan(review_output, review_input)

        self.assertEqual(plan["comments"], [])

    def test_finding_in_changed_hunk_is_planned(self):
        review_input = {
            "repo": "owner/repo",
            "pr": {"number": 1, "head": {"sha": "abc"}},
            "changed_files": [
                {
                    "filename": "connector.py",
                    "status": "modified",
                    "patch": "@@ -9,2 +9,2 @@\n context\n-old\n+new\n",
                }
            ],
        }
        review_output = {
            "findings": [
                {
                    "id": "f1",
                    "title": "Changed request is missing timeout",
                    "category": "api_auth_correctness",
                    "severity": "medium",
                    "confidence": "high",
                    "file": "connector.py",
                    "line": 10,
                    "evidence": "`new` starts a request without a timeout.",
                    "why_it_matters": "A hung request can tie up a worker.",
                    "suggested_fix": "Add a bounded `timeout=` to this request.",
                }
            ]
        }

        plan = build_comment_plan(review_output, review_input)

        self.assertEqual(len(plan["comments"]), 1)

    def test_low_signal_docstring_line_reanchors_to_named_changed_code(self):
        review_input = {
            "repo": "owner/repo",
            "pr": {"number": 1, "head": {"sha": "abc"}},
            "changed_files": [
                {
                    "filename": "connector.py",
                    "patch": (
                        "@@ -50,0 +50,13 @@\n"
                        "+    def _is_user_inactive(self, user):\n"
                        "+        \"\"\"\n"
                        "+        Function to calculate whether user is inactive.\n"
                        "+        \"\"\"\n"
                        "+        last_auth = user.get(\"last_authentication\")\n"
                        "+        ref_date_str = last_auth if last_auth else user.get(\"created_at\")\n"
                        "+\n"
                        "+        ref_date = datetime.fromisoformat(ref_date_str.replace(\"Z\", \"+00:00\"))\n"
                    ),
                }
            ],
        }
        review_output = {
            "findings": [
                {
                    "id": "f1",
                    "title": "Null dereference in _is_user_inactive",
                    "category": "validation",
                    "severity": "medium",
                    "file": "connector.py",
                    "line": 53,
                    "code_reference": "_is_user_inactive",
                    "evidence": "`ref_date_str` can be None when both `last_authentication` and `created_at` are missing.",
                    "why_it_matters": "A single sparse user record can abort the whole action.",
                    "suggested_fix": "Guard `ref_date_str` before calling `replace(...)` and return False or skip the user when both dates are absent.",
                }
            ]
        }

        plan = build_comment_plan(review_output, review_input)
        comment = plan["comments"][0]

        self.assertEqual(comment["github_comment_type"], "line")
        self.assertEqual(comment["path"], "connector.py")
        self.assertEqual(comment["line"], 55)
        self.assertIn("Current line: `ref_date_str =", comment["body"])

    def test_comment_preserves_concrete_multi_sentence_fix_details(self):
        review_input = {
            "repo": "owner/repo",
            "pr": {"number": 1, "head": {"sha": "abc"}},
            "changed_files": [
                {
                    "filename": "connector.py",
                    "patch": "@@ -20,1 +20,1 @@\n+response = requests.post(token_url, json=payload)\n",
                }
            ],
        }
        review_output = {
            "findings": [
                {
                    "id": "f1",
                    "title": "OAuth token request uses the wrong body format",
                    "category": "api_auth_correctness",
                    "severity": "high",
                    "file": "connector.py",
                    "line": 20,
                    "evidence": "The token request sends `json=payload`, but this OAuth endpoint expects a form-encoded body.",
                    "why_it_matters": "The connector can fail test_connectivity and all token refreshes even when the asset credentials are valid.",
                    "suggested_fix": (
                        "Send the grant fields with `data=payload` and set the content type to "
                        "`application/x-www-form-urlencoded`. Keep `timeout=` on the request. "
                        "If the API requires Basic auth, move client_id/client_secret into the "
                        "Authorization header instead of the body."
                    ),
                }
            ]
        }

        plan = build_comment_plan(review_output, review_input)
        body = plan["comments"][0]["body"]

        self.assertIn("Issue: OAuth token request uses the wrong body format", body)
        self.assertIn("Impact:", body)
        self.assertIn("How to fix:", body)
        self.assertIn("application/x-www-form-urlencoded", body)
        self.assertIn("Keep `timeout=` on the request", body)
        self.assertIn("Authorization header", body)

    def test_publish_applies_ai_reviewed_label_after_posted_comment(self):
        class FakeClient:
            def __init__(self):
                self.labels = []

            def create_issue_comment(self, repo, number, *, body):
                return {"html_url": f"https://github.example/{repo}/pull/{number}#issuecomment-1"}

            def add_issue_labels(self, repo, number, labels):
                self.labels.append((repo, number, labels))
                return [{"name": labels[0], "url": f"https://github.example/{repo}/labels/{labels[0]}"}]

        client = FakeClient()
        plan = {
            "repo": "owner/repo",
            "pr_number": 1,
            "head_sha": "abc",
            "comments": [
                {
                    "id": "f1",
                    "github_comment_type": "conversation",
                    "body": "Issue: real finding\n\nHow to fix: fix it.",
                    "marker": "<!-- agentic-pr-review:f1 -->",
                }
            ],
        }

        result = publish_comment_plan(client, plan, {"comments": {}})

        self.assertEqual(result["posted"], 1)
        self.assertEqual(result["label"]["status"], "applied")
        self.assertEqual(client.labels, [("owner/repo", 1, ["ai-reviewed"])])

    def test_publish_skips_ai_reviewed_label_when_no_comment_was_posted(self):
        class FakeClient:
            def __init__(self):
                self.labels = []

            def create_issue_comment(self, repo, number, *, body):
                raise AssertionError("duplicate comment should not be reposted")

            def add_issue_labels(self, repo, number, labels):
                self.labels.append((repo, number, labels))
                return []

        plan = {
            "repo": "owner/repo",
            "pr_number": 1,
            "head_sha": "abc",
            "comments": [
                {
                    "id": "f1",
                    "github_comment_type": "conversation",
                    "body": "Issue: real finding",
                    "marker": "<!-- agentic-pr-review:f1 -->",
                }
            ],
        }
        review_input = {
            "comments": {
                "issue_comments": [{"body": "already posted\n<!-- agentic-pr-review:f1 -->"}],
                "review_comments": [],
                "reviews": [],
            }
        }
        client = FakeClient()

        result = publish_comment_plan(client, plan, review_input)

        self.assertEqual(result["posted"], 0)
        self.assertEqual(result["skipped"], 1)
        self.assertEqual(result["label"]["status"], "skipped_no_posted_comments")
        self.assertEqual(client.labels, [])

    def test_text_finding_without_changed_anchor_is_not_planned(self):
        review_input = {
            "repo": "owner/repo",
            "pr": {"number": 1, "head": {"sha": "abc"}},
            "changed_files": [],
        }
        review_output = {
            "findings": [
                {
                    "id": "f1",
                    "title": "Docs missing",
                    "category": "docs_pr_accuracy",
                    "severity": "medium",
                    "file": "manual_readme_content.md",
                    "line": None,
                    "evidence": "manual_readme_content.md does not document the new asset field.",
                    "suggested_fix": "Document the new asset field and expected format.",
                }
            ]
        }

        plan = build_comment_plan(review_output, review_input)

        self.assertEqual(plan["comments"], [])

    def test_line_not_in_diff_is_not_planned(self):
        review_input = {
            "repo": "owner/repo",
            "pr": {"number": 1, "head": {"sha": "abc"}},
            "changed_files": [
                {
                    "filename": "connector.py",
                    "patch": "@@ -9,1 +9,1 @@\n+new\n",
                }
            ],
        }
        review_output = {
            "findings": [
                {
                    "id": "f1",
                    "title": "Wrong value",
                    "category": "validation",
                    "severity": "medium",
                    "file": "connector.py",
                    "line": 99,
                    "evidence": "Line 99 is not in the PR diff.",
                    "suggested_fix": "Move the issue onto a changed line or handle it manually.",
                }
            ]
        }

        plan = build_comment_plan(review_output, review_input)

        self.assertEqual(plan["comments"], [])

    def test_code_file_wins_over_text_category(self):
        review_input = {
            "repo": "owner/repo",
            "pr": {"number": 1, "head": {"sha": "abc"}},
            "changed_files": [
                {
                    "filename": "connector.py",
                    "patch": "@@ -1,1 +1,1 @@\n+import os\n",
                }
            ],
        }
        review_output = {
            "findings": [
                {
                    "id": "f1",
                    "title": "Unused import",
                    "category": "precommit",
                    "severity": "medium",
                    "file": "connector.py",
                    "line": 1,
                    "evidence": "ruff F401 reports import os is unused.",
                    "suggested_fix": "Remove the unused import.",
                }
            ]
        }

        plan = build_comment_plan(review_output, review_input)

        self.assertEqual(plan["comments"][0]["finding_type"], "code")

    def test_missing_cert_tests_anchor_to_cert_code(self):
        review_input = {
            "repo": "owner/repo",
            "pr": {"number": 1, "head": {"sha": "abc"}},
            "changed_files": [
                {
                    "filename": "asset.py",
                    "patch": "@@ -50,1 +50,2 @@\n+public_cert = AssetField()\n",
                },
                {
                    "filename": "helpers.py",
                    "patch": (
                        "@@ -120,1 +120,4 @@\n"
                        "+@contextmanager\n"
                        "+def temp_cert_files(cert_b64: str, key_b64: str):\n"
                        "+    cert_bytes = base64.b64decode(cert_b64)\n"
                    ),
                },
            ],
        }
        review_output = {
            "findings": [
                {
                    "id": "f1",
                    "title": "New client certificate auth path has no test coverage",
                    "category": "missing_tests",
                    "severity": "medium",
                    "file": "helpers.py",
                    "line": 121,
                    "code_reference": "temp_cert_files",
                    "evidence": "The new temp_cert_files path and cert parameter threading are untested.",
                    "suggested_fix": "Add tests for valid certs, invalid Base64, and no-cert behavior.",
                }
            ]
        }

        plan = build_comment_plan(review_output, review_input)
        comment = plan["comments"][0]

        self.assertEqual(comment["github_comment_type"], "line")
        self.assertEqual(comment["finding_type"], "code")
        self.assertEqual(comment["path"], "helpers.py")
        self.assertEqual(comment["line"], 121)
        self.assertIn("temp_cert_files", comment["body"])

    def test_deterministic_missing_tests_without_file_evidence_is_not_used(self):
        review_input = {
            "repo": "owner/repo",
            "pr": {"number": 1, "head": {"sha": "abc"}},
            "changed_files": [
                {
                    "filename": "helpers.py",
                    "patch": (
                        "@@ -120,1 +120,4 @@\n"
                        "+@contextmanager\n"
                        "+def temp_cert_files(cert_b64: str, key_b64: str):\n"
                        "+    cert_bytes = base64.b64decode(cert_b64)\n"
                    ),
                }
            ],
        }
        review_output = {
            "findings": [],
            "deterministic_findings": [
                {
                    "id": "det-1",
                    "title": "Risky connector behavior changed without visible tests",
                    "category": "missing_tests",
                    "severity": "medium",
                    "confidence": "high",
                    "file": None,
                    "line": None,
                    "evidence": "The diff changes auth/session/polling/state/validation or mutating behavior but no test files were changed.",
                    "suggested_fix": "Add or update tests for the changed behavior.",
                }
            ],
        }

        plan = build_comment_plan(review_output, review_input)

        self.assertEqual(plan["comments"], [])

    def test_high_confidence_deterministic_finding_is_used_when_model_omits_it(self):
        review_input = {
            "repo": "owner/repo",
            "pr": {"number": 1, "head": {"sha": "abc"}},
            "changed_files": [
                {
                    "filename": "connector.py",
                    "patch": "@@ -1,1 +1,1 @@\n+self.degub_print('hello')\n",
                }
            ],
        }
        review_output = {
            "findings": [],
            "deterministic_findings": [
                {
                    "id": "det-1",
                    "title": "Connector logging method appears misspelled",
                    "category": "precommit",
                    "severity": "high",
                    "confidence": "high",
                    "file": "connector.py",
                    "line": 1,
                    "evidence": "self.degub_print('hello')",
                    "suggested_fix": "Rename degub_print to debug_print.",
                    "suggested_code": "self.debug_print('hello')",
                }
            ],
        }

        plan = build_comment_plan(review_output, review_input)
        comment = plan["comments"][0]

        self.assertEqual(comment["path"], "connector.py")
        self.assertIn("debug_print", comment["body"])
        self.assertIn("```suggestion", comment["body"])

    def test_model_review_still_uses_high_confidence_deterministic_findings(self):
        review_input = {
            "repo": "owner/repo",
            "pr": {"number": 1, "head": {"sha": "abc"}},
            "changed_files": [
                {
                    "filename": "src/app.py",
                    "patch": "@@ -1,1 +1,1 @@\n+app = App(...)\n",
                }
            ],
        }
        review_output = {
            "model": "claude-sonnet",
            "findings": [],
            "deterministic_findings": [
                {
                    "id": "det-1",
                    "title": "SDK app does not register test connectivity",
                    "category": "soar_metadata",
                    "severity": "high",
                    "confidence": "high",
                    "file": "src/app.py",
                    "line": 1,
                    "code_reference": "@app.test_connectivity",
                    "evidence": "No function decorated with @app.test_connectivity() was found.",
                    "suggested_fix": "Add test connectivity.",
                }
            ],
        }

        plan = build_comment_plan(review_output, review_input)

        self.assertEqual(len(plan["comments"]), 1)
        self.assertEqual(plan["comments"][0]["path"], "src/app.py")
        self.assertIn("test connectivity", plan["comments"][0]["body"])

    def test_missing_auth_tests_without_file_evidence_is_not_planned(self):
        review_input = {
            "repo": "owner/repo",
            "pr": {"number": 1, "head": {"sha": "abc"}},
            "changed_files": [
                {
                    "filename": "auth.py",
                    "patch": (
                        "@@ -30,1 +30,3 @@\n"
                        "+def refresh_oauth_token(session):\n"
                        "+    return session.get_token(force=True)\n"
                    ),
                }
            ],
        }
        review_output = {
            "findings": [
                {
                    "id": "f1",
                    "title": "OAuth refresh has no tests",
                    "category": "missing_tests",
                    "severity": "medium",
                    "file": None,
                    "line": None,
                    "evidence": "The OAuth token refresh path is not covered by tests.",
                    "suggested_fix": "Add tests for token refresh success and failure cases.",
                }
            ]
        }

        plan = build_comment_plan(review_output, review_input)

        self.assertEqual(plan["comments"], [])

    def test_docs_finding_uses_changed_code_anchor_when_docs_file_is_not_changed(self):
        review_input = {
            "repo": "owner/repo",
            "pr": {"number": 1, "head": {"sha": "abc"}},
            "changed_files": [
                {
                    "filename": "asset.py",
                    "patch": (
                        "@@ -50,1 +50,3 @@\n"
                        "+public_cert: str = AssetField(required=False)\n"
                        "+private_key: str = AssetField(required=False)\n"
                    ),
                }
            ],
        }
        review_output = {
            "findings": [
                {
                    "id": "f1",
                    "title": "New cert fields are not documented",
                    "category": "docs_pr_accuracy",
                    "severity": "medium",
                    "file": "manual_readme_content.md",
                    "line": None,
                    "evidence": "manual_readme_content.md does not mention public_cert or private_key.",
                    "suggested_fix": "Document the new fields.",
                }
            ]
        }

        plan = build_comment_plan(review_output, review_input)
        comment = plan["comments"][0]

        self.assertEqual(comment["github_comment_type"], "line")
        self.assertEqual(comment["finding_type"], "text")
        self.assertEqual(comment["path"], "asset.py")
        self.assertEqual(comment["line"], 50)
        self.assertIn("manual_readme_content.md does not mention public_cert or private_key", comment["body"])
        self.assertNotIn("```suggestion", comment["body"])

    def test_docs_finding_uses_exact_schema_line_when_supplied(self):
        review_input = {
            "repo": "owner/repo",
            "pr": {"number": 1, "head": {"sha": "abc"}},
            "changed_files": [
                {
                    "filename": "slack.json",
                    "patch": (
                        "@@ -10,1 +10,5 @@\n"
                        "+\"parent_message_ts\": {\n"
                        "+  \"description\": \"Parent message timestamp\"\n"
                        "+}\n"
                    ),
                }
            ],
        }
        review_output = {
            "findings": [
                {
                    "id": "f1",
                    "title": "Manual docs missing parent_message_ts docs",
                    "category": "docs_pr_accuracy",
                    "severity": "medium",
                    "file": "slack.json",
                    "line": 10,
                    "code_reference": "parent_message_ts",
                    "evidence": "`parent_message_ts` is added to slack.json, but manual_readme_content.md does not document it.",
                    "suggested_fix": "Document `parent_message_ts` in manual docs or remove the undocumented parameter.",
                }
            ]
        }

        plan = build_comment_plan(review_output, review_input)
        comment = plan["comments"][0]

        self.assertEqual(comment["github_comment_type"], "line")
        self.assertEqual(comment["finding_type"], "text")
        self.assertEqual(comment["path"], "slack.json")
        self.assertEqual(comment["line"], 10)

    def test_docs_finding_prefers_changed_schema_anchor_over_code_anchor(self):
        review_input = {
            "repo": "owner/repo",
            "pr": {"number": 1, "head": {"sha": "abc"}},
            "changed_files": [
                {
                    "filename": "connector.py",
                    "patch": (
                        "@@ -20,1 +20,3 @@\n"
                        "+if \"parent_message_ts\" in param:\n"
                        "+    params[\"thread_ts\"] = param.get(\"parent_message_ts\")\n"
                    ),
                },
                {
                    "filename": "slack.json",
                    "patch": (
                        "@@ -10,1 +10,5 @@\n"
                        "+\"parent_message_ts\": {\n"
                        "+  \"description\": \"Parent message timestamp\"\n"
                        "+}\n"
                    ),
                },
            ],
        }
        review_output = {
            "findings": [
                {
                    "id": "f1",
                    "title": "Manual docs missing parent_message_ts docs",
                    "category": "docs_pr_accuracy",
                    "severity": "medium",
                    "file": "manual_readme_content.md",
                    "line": None,
                    "evidence": "manual_readme_content.md does not document `parent_message_ts`.",
                    "suggested_fix": "Document `parent_message_ts` in manual docs.",
                }
            ]
        }

        plan = build_comment_plan(review_output, review_input)
        comment = plan["comments"][0]

        self.assertEqual(comment["github_comment_type"], "line")
        self.assertEqual(comment["path"], "slack.json")
        self.assertEqual(comment["line"], 10)
        self.assertIn("parent_message_ts", comment["body"])

    def test_speculative_confirm_finding_is_not_planned(self):
        review_input = {
            "repo": "owner/repo",
            "pr": {"number": 1, "head": {"sha": "abc"}},
            "changed_files": [
                {
                    "filename": "connector.py",
                    "patch": "@@ -10,1 +10,2 @@\n+return self._helper(action_result, data)\n",
                }
            ],
        }
        review_output = {
            "findings": [
                {
                    "id": "f1",
                    "title": "Verify helper still emits data",
                    "category": "output_schema_mismatch",
                    "severity": "medium",
                    "file": "connector.py",
                    "line": 10,
                    "evidence": "The helper may emit data.",
                    "suggested_fix": "Confirm that the helper calls action_result.add_data(). If it does, this finding can be closed.",
                }
            ]
        }

        plan = build_comment_plan(review_output, review_input)

        self.assertEqual(plan["comments"], [])

    def test_generic_missing_tests_without_file_evidence_is_not_planned(self):
        review_input = {
            "repo": "owner/repo",
            "pr": {"number": 1, "head": {"sha": "abc"}},
            "changed_files": [
                {
                    "filename": "connector.py",
                    "patch": (
                        "@@ -1400,1 +1400,4 @@\n"
                        "+def _poll_for_question_response(self, action_result, resp_json):\n"
                        "+    return action_result.set_status(phantom.APP_SUCCESS)\n"
                    ),
                }
            ],
        }
        review_output = {
            "findings": [],
            "deterministic_findings": [
                {
                    "id": "det-1",
                    "title": "Risky connector behavior changed without visible tests",
                    "category": "missing_tests",
                    "severity": "medium",
                    "file": None,
                    "line": None,
                    "evidence": "The diff changes auth/session/polling/state/validation or mutating behavior but no test files were changed.",
                    "suggested_fix": "Add or update tests for the changed behavior.",
                }
            ],
        }

        plan = build_comment_plan(review_output, review_input)

        self.assertEqual(plan["comments"], [])

    def test_base64_try_comment_uses_short_template(self):
        review_input = {
            "repo": "owner/repo",
            "pr": {"number": 1, "head": {"sha": "abc"}},
            "changed_files": [
                {
                    "filename": "helpers.py",
                    "patch": (
                        "@@ -134,1 +134,4 @@\n"
                        "+cert_path, key_path = None, None\n"
                        "+cert_bytes = base64.b64decode(cert_b64)\n"
                        "+try:\n"
                    ),
                }
            ],
        }
        review_output = {
            "findings": [
                {
                    "id": "f1",
                    "title": "base64.b64decode calls occur before the try block",
                    "category": "general",
                    "severity": "medium",
                    "file": "helpers.py",
                    "line": 135,
                    "evidence": "base64.b64decode runs before try.",
                    "suggested_fix": "Wrap base64.b64decode in try/except and raise ActionFailure.",
                }
            ]
        }

        plan = build_comment_plan(review_output, review_input)
        body = plan["comments"][0]["body"]

        self.assertIn("base64.b64decode runs before try", body)
        self.assertIn("raise ActionFailure", body)

    def test_undefined_reference_is_not_rewritten_as_try_template(self):
        review_input = {
            "repo": "owner/repo",
            "pr": {"number": 1, "head": {"sha": "abc"}},
            "changed_files": [
                {
                    "filename": "helpers.py",
                    "patch": (
                        "@@ -134,1 +134,3 @@\n"
                        "+cert_bytes = base64.b64decode(asset.public_cert)\n"
                    ),
                }
            ],
        }
        review_output = {
            "findings": [
                {
                    "id": "f1",
                    "title": "NameError: asset is undefined",
                    "category": "general",
                    "severity": "critical",
                    "file": "helpers.py",
                    "line": 134,
                    "evidence": "asset is not in scope here.",
                    "suggested_fix": "Use the cert_b64 parameter instead.",
                }
            ]
        }

        plan = build_comment_plan(review_output, review_input)
        body = plan["comments"][0]["body"]

        self.assertIn("asset is not in scope", body)
        self.assertIn("cert_b64", body)
        self.assertNotIn("Decode errors", body)

    def test_generic_try_comment_uses_error_handling_template(self):
        review_input = {
            "repo": "owner/repo",
            "pr": {"number": 1, "head": {"sha": "abc"}},
            "changed_files": [
                {
                    "filename": "connector.py",
                    "patch": (
                        "@@ -10,1 +10,3 @@\n"
                        "+try:\n"
                        "+    response = client.fetch()\n"
                    ),
                }
            ],
        }
        review_output = {
            "findings": [
                {
                    "id": "f1",
                    "title": "API error is not converted to ActionFailure",
                    "category": "validation",
                    "severity": "medium",
                    "file": "connector.py",
                    "line": 10,
                    "evidence": "This try block can let a raw exception escape.",
                    "suggested_fix": "Catch the client exception and raise ActionFailure.",
                }
            ]
        }

        plan = build_comment_plan(review_output, review_input)
        body = plan["comments"][0]["body"]

        self.assertIn("raw exception", body)
        self.assertIn("Catch the client exception", body)

    def test_concrete_merge_conflict_is_posted(self):
        review_input = {
            "repo": "owner/repo",
            "pr": {"number": 1, "head": {"sha": "abc"}},
            "changed_files": [],
        }
        review_output = {
            "findings": [
                {
                    "id": "f1",
                    "title": "PR branch has merge conflicts",
                    "category": "merge_conflict",
                    "severity": "high",
                    "confidence": "high",
                    "file": "connector.py",
                    "line": None,
                    "code_reference": "merge conflict in connector.py",
                    "evidence": "GitHub reports mergeable_state=dirty with a merge conflict in connector.py.",
                    "suggested_fix": "Rebase the branch and resolve conflicts.",
                }
            ]
        }

        plan = build_comment_plan(review_output, review_input)
        body = plan["comments"][0]["body"]

        self.assertIn("mergeable_state=dirty", body)
        self.assertNotIn("raw exception", body)

    def test_generic_blocked_mergeability_due_to_precommit_is_not_posted(self):
        review_input = {
            "repo": "owner/repo",
            "pr": {"number": 1, "head": {"sha": "abc"}},
            "changed_files": [],
        }
        review_output = {
            "findings": [
                {
                    "id": "f1",
                    "title": "PR mergeability is blocked",
                    "category": "merge_conflict",
                    "severity": "high",
                    "confidence": "high",
                    "file": None,
                    "line": None,
                    "evidence": "GitHub mergeable_state is reported as blocked. The pre-commit required check is failing.",
                    "suggested_fix": "Resolve the pre-commit failure.",
                }
            ]
        }

        plan = build_comment_plan(review_output, review_input)

        self.assertEqual(plan["comments"], [])

    def test_specific_blocked_mergeability_due_to_precommit_is_posted(self):
        review_input = {
            "repo": "owner/repo",
            "pr": {"number": 1, "head": {"sha": "abc"}},
            "changed_files": [],
        }
        review_output = {
            "findings": [
                {
                    "id": "f1",
                    "title": "PR mergeability is blocked by ruff",
                    "category": "merge_conflict",
                    "severity": "high",
                    "confidence": "high",
                    "file": "connector.py",
                    "line": 1,
                    "evidence": "GitHub mergeable_state is blocked by: pre-commit: hook id: ruff / connector.py:1:1: F401 `os` imported but unused.",
                    "suggested_fix": "Remove the unused import and rerun pre-commit.",
                }
            ]
        }

        plan = build_comment_plan(review_output, review_input)
        body = plan["comments"][0]["body"]

        self.assertIn("blocked by", body)
        self.assertIn("connector.py:1:1", body)
        self.assertIn("Remove the unused import", body)

    def test_low_ci_synthesis_is_not_posted(self):
        review_input = {
            "repo": "owner/repo",
            "pr": {"number": 1, "head": {"sha": "abc"}},
            "changed_files": [],
        }
        review_output = {
            "findings": [
                {
                    "id": "f1",
                    "title": "Old CI failures should be re-evaluated",
                    "category": "ci_synthesis",
                    "severity": "low",
                    "file": None,
                    "line": None,
                    "evidence": "Older CI failures may not reflect current head.",
                    "suggested_fix": "Re-check after fixing pre-commit.",
                }
            ]
        }

        plan = build_comment_plan(review_output, review_input)

        self.assertEqual(plan["comments"], [])

    def test_medium_confidence_finding_is_not_posted(self):
        review_input = {
            "repo": "owner/repo",
            "pr": {"number": 1, "head": {"sha": "abc"}},
            "changed_files": [
                {
                    "filename": "connector.py",
                    "patch": "@@ -1,1 +1,1 @@\n+value = call_api()\n",
                }
            ],
        }
        review_output = {
            "findings": [
                {
                    "id": "f1",
                    "title": "Possible API issue",
                    "category": "api_auth_correctness",
                    "severity": "medium",
                    "confidence": "medium",
                    "file": "connector.py",
                    "line": 1,
                    "evidence": "This may not match the API docs.",
                    "suggested_fix": "Verify the API contract.",
                }
            ]
        }

        plan = build_comment_plan(review_output, review_input)

        self.assertEqual(plan["comments"], [])

    def test_readme_only_docs_finding_is_not_posted(self):
        review_input = {
            "repo": "owner/repo",
            "pr": {"number": 1, "head": {"sha": "abc"}},
            "changed_files": [],
        }
        review_output = {
            "findings": [
                {
                    "id": "f1",
                    "title": "README docs missing",
                    "category": "docs_pr_accuracy",
                    "severity": "medium",
                    "confidence": "high",
                    "file": "README.md",
                    "line": None,
                    "evidence": "README.md does not describe the generated output.",
                    "suggested_fix": "Update README.md.",
                }
            ]
        }

        plan = build_comment_plan(review_output, review_input)

        self.assertEqual(plan["comments"], [])

    def test_generic_precommit_finding_is_not_posted(self):
        review_input = {
            "repo": "owner/repo",
            "pr": {"number": 1, "head": {"sha": "abc"}},
            "changed_files": [],
        }
        review_output = {
            "findings": [
                {
                    "id": "f1",
                    "title": "Pre-commit failed",
                    "category": "precommit",
                    "severity": "high",
                    "confidence": "high",
                    "file": None,
                    "line": None,
                    "evidence": "The pre-commit check failed.",
                    "suggested_fix": "Resolve the pre-commit failure.",
                }
            ]
        }

        plan = build_comment_plan(review_output, review_input)

        self.assertEqual(plan["comments"], [])

    def test_ci_wrapper_precommit_finding_is_not_posted(self):
        review_input = {
            "repo": "owner/repo",
            "pr": {"number": 1, "head": {"sha": "abc"}},
            "changed_files": [],
        }
        review_output = {
            "findings": [
                {
                    "id": "f1",
                    "title": "Pre-commit failed",
                    "category": "precommit",
                    "severity": "high",
                    "confidence": "high",
                    "file": None,
                    "line": None,
                    "evidence": 'pytest-output-raw.log (exit code: $PYTEST_EXIT_CODE)"',
                    "suggested_fix": "Run the named pre-commit hook locally.",
                }
            ]
        }

        plan = build_comment_plan(review_output, review_input)

        self.assertEqual(plan["comments"], [])

    def test_actionable_precommit_finding_has_specific_fix(self):
        review_input = {
            "repo": "owner/repo",
            "pr": {"number": 1, "head": {"sha": "abc"}},
            "changed_files": [
                {
                    "filename": "connector.py",
                    "patch": "@@ -1,1 +1,1 @@\n+import os\n",
                }
            ],
        }
        review_output = {
            "findings": [
                {
                    "id": "f1",
                    "title": "Ruff hook failed",
                    "category": "precommit",
                    "severity": "high",
                    "confidence": "high",
                    "file": "connector.py",
                    "line": 1,
                    "evidence": "pre-commit: ruff failed - F401 `os` imported but unused.",
                    "suggested_fix": "Remove the unused import.",
                }
            ]
        }

        plan = build_comment_plan(review_output, review_input)
        body = plan["comments"][0]["body"]

        self.assertIn("ruff", body.lower())
        self.assertIn("Remove the unused import", body)

    def test_package_dependency_precommit_finding_is_publishable(self):
        review_input = {
            "repo": "owner/repo",
            "pr": {"number": 1, "head": {"sha": "abc"}},
            "changed_files": [
                {
                    "filename": ".pre-commit-config.yaml",
                    "patch": "@@ -1,4 +1,2 @@\n-      - id: package-app-dependencies\n+    rev: v2.1.0\n",
                }
            ],
        }
        review_output = {
            "findings": [
                {
                    "id": "f1",
                    "title": "`package-app-dependencies` hook removed by ci-tools downgrade",
                    "category": "precommit",
                    "severity": "high",
                    "confidence": "high",
                    "file": ".pre-commit-config.yaml",
                    "line": 1,
                    "code_reference": "ci-tools rev / package-app-dependencies",
                    "evidence": "The diff removes the package-app-dependencies hook and downgrades ci-tools from v2.1.4 to v2.1.0.",
                    "why_it_matters": "Without package-app-dependencies, dependency wheels will be missing from the app package.",
                    "suggested_fix": "Restore ci-tools v2.1.4 and the package-app-dependencies hook.",
                }
            ]
        }

        plan = build_comment_plan(review_output, review_input)

        self.assertEqual(len(plan["comments"]), 1)
        self.assertIn("package-app-dependencies", plan["comments"][0]["body"])

    def test_related_output_schema_findings_are_grouped_into_one_comment(self):
        review_input = {
            "repo": "owner/repo",
            "pr": {"number": 1, "head": {"sha": "abc"}},
            "changed_files": [
                {
                    "filename": "connector.py",
                    "patch": (
                        "@@ -10,1 +10,5 @@\n"
                        "+def _handle_enrich_hash(self, param):\n"
                        "+    action_result.add_data(response)\n"
                        "+def _handle_enrich_ip(self, param):\n"
                        "+    action_result.add_data(response)\n"
                    ),
                }
            ],
        }
        review_output = {
            "findings": [
                {
                    "id": "f1",
                    "title": "Code emits action_result.data but app JSON declares no data output",
                    "category": "output_schema_mismatch",
                    "severity": "medium",
                    "confidence": "high",
                    "file": "connector.py",
                    "line": 11,
                    "evidence": "_handle_enrich_hash calls action_result.add_data(response).",
                    "why_it_matters": "SOAR playbooks cannot select undeclared outputs.",
                    "suggested_fix": "Declare action_result.data.* output paths.",
                },
                {
                    "id": "f2",
                    "title": "Code emits action_result.data but app JSON declares no data output",
                    "category": "output_schema_mismatch",
                    "severity": "medium",
                    "confidence": "high",
                    "file": "connector.py",
                    "line": 13,
                    "evidence": "_handle_enrich_ip calls action_result.add_data(response).",
                    "why_it_matters": "SOAR playbooks cannot select undeclared outputs.",
                    "suggested_fix": "Declare action_result.data.* output paths.",
                },
            ]
        }

        plan = build_comment_plan(review_output, review_input)

        self.assertEqual(len(plan["comments"]), 1)
        self.assertIn("Action output and indicator metadata", plan["comments"][0]["title"])
        self.assertIn("Related locations", plan["comments"][0]["body"])

    def test_group_excludes_medium_confidence_member_before_posting(self):
        review_input = {
            "repo": "owner/repo",
            "pr": {"number": 1, "head": {"sha": "abc"}},
            "changed_files": [
                {
                    "filename": "connector.py",
                    "patch": (
                        "@@ -10,1 +10,5 @@\n"
                        "+requests.get(url, verify=self._verify)\n"
                        "+requests.post(url, json=payload)\n"
                    ),
                }
            ],
        }
        review_output = {
            "findings": [
                {
                    "id": "f1",
                    "title": "External API TLS verification is disabled by default",
                    "category": "api_auth_correctness",
                    "severity": "high",
                    "confidence": "high",
                    "file": "connector.py",
                    "line": 11,
                    "evidence": "`requests.get(..., verify=self._verify)` uses a false default.",
                    "why_it_matters": "External API calls can be made without certificate validation.",
                    "suggested_fix": "Expose `verify_server_cert` in asset config and default it to true.",
                },
                {
                    "id": "f2",
                    "title": "External API requests are missing explicit timeouts",
                    "category": "api_auth_correctness",
                    "severity": "medium",
                    "confidence": "medium",
                    "file": "connector.py",
                    "line": 12,
                    "evidence": "`requests.post(...)` has no timeout.",
                    "why_it_matters": "A hung request can tie up a SOAR worker.",
                    "suggested_fix": "Pass a bounded `timeout=` to the request.",
                },
            ]
        }

        plan = real_build_comment_plan(review_output, review_input)

        self.assertEqual(len(plan["comments"]), 1)
        self.assertIn("TLS verification", plan["comments"][0]["body"])
        self.assertNotIn("timeout", plan["comments"][0]["body"].lower())

    def test_comment_plan_respects_max_comments_after_grouping(self):
        review_input = {
            "repo": "owner/repo",
            "pr": {"number": 1, "head": {"sha": "abc"}},
            "changed_files": [
                {"filename": "connector.py", "patch": "@@ -1,20 +1,20 @@\n" + "\n".join(f"+value_{idx} = {idx}" for idx in range(1, 21))}
            ],
        }
        findings = [
            {
                "id": f"f{idx}",
                "title": f"Distinct issue {idx}",
                "category": "validation",
                "severity": "medium",
                "confidence": "high",
                "file": "connector.py",
                "line": idx,
                "evidence": f"`value_{idx}` is wrong.",
                "why_it_matters": "This can break connector behavior.",
                "suggested_fix": "Fix this specific value.",
            }
            for idx in range(1, 21)
        ]

        plan = build_comment_plan({"findings": findings}, review_input, max_comments=15)

        self.assertEqual(len(plan["comments"]), 15)

    def test_low_severity_findings_are_not_planned_as_github_comments(self):
        review_input = {
            "repo": "owner/repo",
            "pr": {"number": 1, "head": {"sha": "abc"}},
            "changed_files": [
                {"filename": "connector.py", "patch": "@@ -1,1 +1,1 @@\n+name = indicator_value\n"}
            ],
        }
        review_output = {
            "findings": [
                {
                    "id": "f1",
                    "title": "Polling artifact names are raw indicator values",
                    "category": "soar_metadata",
                    "severity": "low",
                    "confidence": "high",
                    "file": "connector.py",
                    "line": 1,
                    "evidence": "Artifact `name` is assigned directly from `indicator_value`.",
                    "why_it_matters": "This makes artifact lists noisy.",
                    "suggested_fix": "Use a descriptive artifact name.",
                }
            ]
        }

        plan = build_comment_plan(review_output, review_input)

        self.assertEqual(plan["comments"], [])

    def test_static_test_precommit_without_file_evidence_is_not_planned(self):
        review_input = {
            "repo": "owner/repo",
            "pr": {"number": 1, "head": {"sha": "abc"}},
            "changed_files": [],
        }
        review_output = {
            "findings": [
                {
                    "id": "f1",
                    "title": "Static tests failed",
                    "category": "precommit",
                    "severity": "high",
                    "confidence": "high",
                    "file": None,
                    "line": None,
                    "evidence": "Static test failures: Additional logging, license, min phantom version & verbosity",
                    "suggested_fix": "Fix the named static test failures.",
                }
            ]
        }

        plan = build_comment_plan(review_output, review_input)

        self.assertEqual(plan["comments"], [])

    def test_min_number_log_statements_suggests_adding_useful_logging(self):
        review_input = {
            "repo": "owner/repo",
            "pr": {"number": 1, "head": {"sha": "abc"}},
            "changed_files": [],
        }
        review_output = {
            "findings": [
                {
                    "id": "f1",
                    "title": "Static tests failed",
                    "category": "precommit",
                    "severity": "high",
                    "confidence": "high",
                    "file": "connector.py",
                    "line": None,
                    "code_reference": "_handle_update_incident",
                    "evidence": "Min Number Log Statements: _handle_update_incident has too few logging statements.",
                    "suggested_fix": "Add useful debug_print or save_progress statements to _handle_update_incident.",
                }
            ]
        }

        plan = build_comment_plan(review_output, review_input)
        body = plan["comments"][0]["body"]

        self.assertIn("Add useful debug_print", body)
        self.assertNotIn("Remove unnecessary", body)

    def test_collect_existing_markers(self):
        review_input = {
            "comments": {
                "issue_comments": [
                    {"body": "comment\n\n<!-- agentic-pr-review:finding-abc123 -->"}
                ],
                "review_comments": [],
                "reviews": [],
            }
        }

        self.assertEqual(collect_existing_markers(review_input), {"<!-- agentic-pr-review:finding-abc123 -->"})


if __name__ == "__main__":
    unittest.main()
