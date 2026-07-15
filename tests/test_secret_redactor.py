import base64
import os
from unittest.mock import patch
import unittest

from agentic_pr_review.secret_redactor import (
    REDACTED_AUTH,
    REDACTED_PRIVATE_KEY,
    REDACTED_SECRET,
    REDACTED_TOKEN,
    known_secret_values,
    redact_obj,
    redact_text,
    secret_fingerprint,
)


class SecretRedactorTest(unittest.TestCase):
    def test_redacts_runtime_env_secrets_and_basic_auth_derivative(self):
        env = {
            "CIRCUIT_CLIENT_ID": "canary-client-id-12345",
            "CIRCUIT_CLIENT_SECRET": "canary-client-secret-67890",
            "CIRCUIT_APP_KEY": "canary-app-key-abcdef",
            "AI_REVIEW_PRIVATE_KEY": "-----BEGIN PRIVATE KEY-----\\ncanary-private\\n-----END PRIVATE KEY-----",
            "AI_REVIEW_APP_ID": "123456",
            "PLAIN_SETTING": "canary-not-secret",
        }
        basic = base64.b64encode(f"{env['CIRCUIT_CLIENT_ID']}:{env['CIRCUIT_CLIENT_SECRET']}".encode()).decode()
        text = (
            f"id={env['CIRCUIT_CLIENT_ID']} secret={env['CIRCUIT_CLIENT_SECRET']} "
            f"app={env['CIRCUIT_APP_KEY']} basic={basic} "
            f"key={env['AI_REVIEW_PRIVATE_KEY']} app_id={env['AI_REVIEW_APP_ID']} "
            f"plain={env['PLAIN_SETTING']}"
        )

        with patch.dict(os.environ, env, clear=True):
            redacted = redact_text(text)
            known = known_secret_values()

        self.assertIn(REDACTED_SECRET, redacted)
        self.assertNotIn(env["CIRCUIT_CLIENT_ID"], redacted)
        self.assertNotIn(env["CIRCUIT_CLIENT_SECRET"], redacted)
        self.assertNotIn(env["CIRCUIT_APP_KEY"], redacted)
        self.assertNotIn(basic, redacted)
        self.assertNotIn("canary-private", redacted)
        self.assertIn(env["AI_REVIEW_APP_ID"], redacted)
        self.assertIn(env["PLAIN_SETTING"], redacted)
        self.assertIn(basic, known)

    def test_redacts_common_token_shapes_without_redacting_variable_names(self):
        text = "\n".join(
            [
                "Authorization: Bearer abcdefghijklmnopqrstuvwxyz123456",
                "x-api-key: abcdefghijklmnopqrstuvwxyz123456",
                "access_token='abcdefghijklmnopqrstuvwxyz123456'",
                "github_pat_1234567890abcdef1234567890abcdef1234567890abcdef",
                "ghp_1234567890abcdef1234567890abcdef123456",
                "sk-abcdefghijklmnopqrstuvwxyz123456",
                "AKIA1234567890ABCDEF",
                "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.signature123456789",
                "client_secret = os.getenv('CLIENT_SECRET')",
            ]
        )

        redacted = redact_text(text)

        self.assertIn(REDACTED_AUTH, redacted)
        self.assertIn(REDACTED_SECRET, redacted)
        self.assertIn(REDACTED_TOKEN, redacted)
        self.assertIn("client_secret = os.getenv('CLIENT_SECRET')", redacted)
        self.assertNotIn("abcdefghijklmnopqrstuvwxyz123456", redacted)
        self.assertNotIn("github_pat_", redacted)
        self.assertNotIn("AKIA1234567890ABCDEF", redacted)

    def test_redacts_nested_json_like_objects(self):
        env = {"CIRCUIT_CLIENT_SECRET": "canary-nested-secret"}
        payload = {
            "outer": [{"message": "do not leak canary-nested-secret"}],
            "safe": "normal text",
        }

        with patch.dict(os.environ, env, clear=True):
            redacted = redact_obj(payload)

        self.assertEqual(redacted["outer"][0]["message"], f"do not leak {REDACTED_SECRET}")
        self.assertEqual(redacted["safe"], "normal text")

    def test_secret_fingerprint_is_stable_and_non_reversible(self):
        fingerprint = secret_fingerprint("canary-app-key-abcdef")

        self.assertEqual(fingerprint, secret_fingerprint("canary-app-key-abcdef"))
        self.assertNotIn("canary", fingerprint)
        self.assertEqual(len(fingerprint), 16)

    def test_redacts_private_key_blocks(self):
        text = "-----BEGIN PRIVATE KEY-----\nabc123\n-----END PRIVATE KEY-----"

        redacted = redact_text(text)

        self.assertEqual(redacted, REDACTED_PRIVATE_KEY)


if __name__ == "__main__":
    unittest.main()
