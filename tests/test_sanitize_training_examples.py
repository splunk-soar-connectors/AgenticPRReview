import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import sanitize_training_examples  # noqa: E402


class SanitizeTrainingExamplesTest(unittest.TestCase):
    def test_sanitizer_drops_raw_repo_user_urls_and_network_data(self):
        url_scheme = "https" + "://"
        fake_url = url_scheme + "example.invalid/org/repo/pull/123"
        fake_ip = "10." + "1.2.3"
        fake_host = "private." + "host.example.invalid"
        fake_check = "Code" + "Build static"
        raw = {
            "pattern_id": "unsafe_logging_sensitive",
            "category": "unsafe_logging",
            "pattern_title": "PrivateConnector sensitive value logged",
            "implementation_hint": "Check PrivateConnector token logging in CI metadata output.",
            "repo": "example-org/privateconnector",
            "pr_number": 123,
            "pr_title": "Private connector change",
            "pr_url": fake_url,
            "comment_kind": "review_comment",
            "path": "privateconnector_connector.py",
            "line": 99,
            "user": "real-user",
            "url": fake_url + "#discussion_r1",
            "body": f"Do not log token values from {fake_ip} on {fake_host}.",
            "diff_hunk": "+self.debug_print(secret_token)",
            "changed_files": ["privateconnector_connector.py"],
            "file_evidence": [{"filename": "privateconnector_connector.py", "patch_excerpt": "+self.debug_print(secret_token)"}],
            "check_evidence": [{"kind": "check_run", "name": fake_check, "conclusion": "failure"}],
        }

        sanitized = sanitize_training_examples.sanitize_examples([raw])[0]
        rendered = json.dumps(sanitized, sort_keys=True).lower()

        self.assertEqual(sanitized["repo"], "repo-0001")
        self.assertEqual(sanitized["path"], "connector.py")
        self.assertEqual(sanitized["user"], "human-reviewer")
        self.assertNotIn("privateconnector", rendered)
        self.assertIn("example-repo", rendered)
        self.assertNotIn(url_scheme, rendered)
        self.assertNotIn(fake_ip, rendered)
        self.assertNotIn(fake_host, rendered)
        self.assertNotIn(fake_check.lower(), rendered)
        self.assertEqual(sanitize_training_examples.find_public_blockers(json.dumps(sanitized)), [])

    def test_replacement_file_overrides_auto_replacements(self):
        raw = {
            "pattern_id": "docs_drift",
            "category": "docs_pr_accuracy",
            "pattern_title": "AcmeConnector docs drift",
            "implementation_hint": "AcmeConnector release notes need the changed default.",
            "repo": "example-org/acmeconnector",
            "pr_number": 7,
        }

        sanitized = sanitize_training_examples.sanitize_examples(
            [raw],
            replacements={"AcmeConnector": "storage connector"},
        )[0]
        rendered = json.dumps(sanitized, sort_keys=True)

        self.assertIn("storage connector", rendered)
        self.assertNotIn("AcmeConnector", rendered)


if __name__ == "__main__":
    unittest.main()
