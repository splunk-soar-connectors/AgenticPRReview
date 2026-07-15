import json
import os
from io import BytesIO
from io import StringIO
from urllib.error import HTTPError
from urllib.parse import parse_qs
from unittest.mock import patch
import unittest

from agentic_pr_review.gateway_claude import GatewayClaudeReviewer, extract_chat_completion_text, mask_secret_for_github_actions
from agentic_pr_review.config import RuntimeConfig
from agentic_pr_review.secret_redactor import REDACTED_AUTH, REDACTED_SECRET


class GatewayClaudeTest(unittest.TestCase):
    def test_runtime_config_selects_gateway_provider(self):
        env = {
            "AGENTIC_PR_REVIEW_ENV_FILE": "missing.env",
            "MODEL_PROVIDER": "gateway",
            "GATEWAY_BASE_URL": "https://gateway.example/deployments/claude/chat/completions",
            "GATEWAY_MODEL": "claude-sonnet-4-6",
            "GATEWAY_APP_KEY": "app-key-test",
            "GATEWAY_CLIENT_ID": "client-id-test",
            "GATEWAY_CLIENT_SECRET": "client-secret-test",
            "GATEWAY_TOKEN_URL": "https://gateway.example/oauth2/default/v1/token",
        }

        with patch.dict(os.environ, env, clear=True):
            config = RuntimeConfig.from_env()

        self.assertEqual(config.model_provider, "gateway")
        self.assertEqual(config.gateway_model, "claude-sonnet-4-6")
        config.require_model()

    def test_runtime_config_accepts_legacy_circuit_env_names(self):
        env = {
            "AGENTIC_PR_REVIEW_ENV_FILE": "missing.env",
            "MODEL_PROVIDER": "circuit",
            "CIRCUIT_BASE_URL": "https://gateway.example/deployments/claude/chat/completions",
            "CIRCUIT_MODEL": "claude-sonnet-4-6",
            "CIRCUIT_APP_KEY": "app-key-test",
            "CIRCUIT_CLIENT_ID": "client-id-test",
            "CIRCUIT_CLIENT_SECRET": "client-secret-test",
            "CIRCUIT_TOKEN_URL": "https://gateway.example/oauth2/default/v1/token",
        }

        with patch.dict(os.environ, env, clear=True):
            config = RuntimeConfig.from_env()

        self.assertEqual(config.model_provider, "gateway")
        self.assertEqual(config.gateway_base_url, env["CIRCUIT_BASE_URL"])
        self.assertEqual(config.gateway_model, env["CIRCUIT_MODEL"])
        self.assertEqual(config.gateway_app_key, env["CIRCUIT_APP_KEY"])
        self.assertEqual(config.gateway_client_id, env["CIRCUIT_CLIENT_ID"])
        self.assertEqual(config.gateway_client_secret, env["CIRCUIT_CLIENT_SECRET"])
        self.assertEqual(config.gateway_token_url, env["CIRCUIT_TOKEN_URL"])
        config.require_model()

    def test_gateway_review_exchanges_token_and_posts_chat_completion(self):
        env = {
            "AGENTIC_PR_REVIEW_ENV_FILE": "missing.env",
            "MODEL_PROVIDER": "gateway",
            "GATEWAY_BASE_URL": "https://gateway.example/deployments/claude/chat/completions",
            "GATEWAY_MODEL": "claude-sonnet-4-6",
            "GATEWAY_APP_KEY": "app-key-test",
            "GATEWAY_CLIENT_ID": "client-id-test",
            "GATEWAY_CLIENT_SECRET": "client-secret-test",
            "GATEWAY_TOKEN_URL": "https://gateway.example/oauth2/default/v1/token",
        }
        calls = []

        class Response:
            def __init__(self, payload):
                self.payload = payload

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

            def read(self):
                return json.dumps(self.payload).encode("utf-8")

        def fake_urlopen(request, timeout):
            body = request.data.decode("utf-8")
            calls.append((request, timeout, body))
            if request.full_url.endswith("/token"):
                return Response({"access_token": "access-token-test", "token_type": "Bearer", "expires_in": 3600})
            return Response(
                {
                    "model": "claude-sonnet-4-6",
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "summary": "ok",
                                        "overall_status": "looks_good",
                                        "safe_to_publish": True,
                                        "findings": [],
                                    }
                                )
                            }
                        }
                    ],
                }
            )

        with patch.dict(os.environ, env, clear=True):
            config = RuntimeConfig.from_env()
        reviewer = GatewayClaudeReviewer(config, max_tokens=123)

        with patch("agentic_pr_review.gateway_claude.urlopen", fake_urlopen):
            output = reviewer._invoke_review("review this", [])

        self.assertEqual(output["overall_status"], "looks_good")
        self.assertEqual(output["model"], "claude-sonnet-4-6")
        self.assertEqual(len(calls), 2)

        token_request, token_timeout, token_body = calls[0]
        token_headers = {key.lower(): value for key, value in token_request.header_items()}
        self.assertEqual(token_timeout, 60)
        self.assertEqual(token_request.full_url, "https://gateway.example/oauth2/default/v1/token")
        self.assertEqual(parse_qs(token_body), {"grant_type": ["client_credentials"]})
        self.assertTrue(token_headers["authorization"].startswith("Basic "))
        self.assertEqual(token_headers["accept"], "*/*")

        model_request, model_timeout, model_body = calls[1]
        model_headers = {key.lower(): value for key, value in model_request.header_items()}
        model_payload = json.loads(model_body)
        self.assertEqual(model_timeout, 180)
        self.assertEqual(model_request.full_url, "https://gateway.example/deployments/claude/chat/completions")
        self.assertEqual(model_headers["authorization"], "Bearer access-token-test")
        self.assertEqual(model_headers["api-key"], "access-token-test")
        self.assertEqual(json.loads(model_payload["user"]), {"appkey": "app-key-test"})
        self.assertNotIn("model", model_payload)
        self.assertEqual(model_payload["max_tokens"], 123)
        self.assertEqual(model_payload["messages"][0]["role"], "system")
        self.assertEqual(model_payload["messages"][1]["content"], "review this")

    def test_gateway_review_refreshes_token_after_model_auth_failure(self):
        env = {
            "AGENTIC_PR_REVIEW_ENV_FILE": "missing.env",
            "MODEL_PROVIDER": "gateway",
            "GATEWAY_BASE_URL": "https://gateway.example/deployments/claude/chat/completions",
            "GATEWAY_MODEL": "claude-sonnet-4-6",
            "GATEWAY_APP_KEY": "app-key-test",
            "GATEWAY_CLIENT_ID": "client-id-test",
            "GATEWAY_CLIENT_SECRET": "client-secret-test",
            "GATEWAY_TOKEN_URL": "https://gateway.example/oauth2/default/v1/token",
        }
        calls = []
        token_index = 0

        class Response:
            def __init__(self, payload):
                self.payload = payload

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

            def read(self):
                return json.dumps(self.payload).encode("utf-8")

        def fake_urlopen(request, timeout):
            nonlocal token_index
            calls.append((request, timeout))
            if request.full_url.endswith("/token"):
                token_index += 1
                return Response({"access_token": f"access-token-{token_index}", "expires_in": 3600})
            model_headers = {key.lower(): value for key, value in request.header_items()}
            if model_headers["api-key"] == "access-token-1":
                raise HTTPError(
                    request.full_url,
                    401,
                    "Unauthorized",
                    {},
                    BytesIO(b'{"error":"expired token"}'),
                )
            return Response(
                {
                    "model": "claude-sonnet-4-6",
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "summary": "ok",
                                        "overall_status": "looks_good",
                                        "safe_to_publish": True,
                                        "findings": [],
                                    }
                                )
                            }
                        }
                    ],
                }
            )

        with patch.dict(os.environ, env, clear=True):
            config = RuntimeConfig.from_env()
        reviewer = GatewayClaudeReviewer(config)

        with patch("agentic_pr_review.gateway_claude.urlopen", fake_urlopen):
            output = reviewer._invoke_review("review this", [])

        self.assertEqual(output["overall_status"], "looks_good")
        self.assertEqual([call[0].full_url.endswith("/token") for call in calls], [True, False, True, False])
        final_headers = {key.lower(): value for key, value in calls[-1][0].header_items()}
        self.assertEqual(final_headers["authorization"], "Bearer access-token-2")
        self.assertEqual(final_headers["api-key"], "access-token-2")

    def test_gateway_token_refreshes_when_expiry_is_inside_skew_window(self):
        env = {
            "AGENTIC_PR_REVIEW_ENV_FILE": "missing.env",
            "MODEL_PROVIDER": "gateway",
            "GATEWAY_BASE_URL": "https://gateway.example/deployments/claude/chat/completions",
            "GATEWAY_MODEL": "claude-sonnet-4-6",
            "GATEWAY_APP_KEY": "app-key-test",
            "GATEWAY_CLIENT_ID": "client-id-test",
            "GATEWAY_CLIENT_SECRET": "client-secret-test",
            "GATEWAY_TOKEN_URL": "https://gateway.example/oauth2/default/v1/token",
        }
        token_index = 0

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

            def read(self):
                nonlocal token_index
                token_index += 1
                return json.dumps({"access_token": f"access-token-{token_index}", "expires_in": 1}).encode("utf-8")

        with patch.dict(os.environ, env, clear=True):
            config = RuntimeConfig.from_env()
        reviewer = GatewayClaudeReviewer(config)

        with patch("agentic_pr_review.gateway_claude.urlopen", return_value=Response()):
            first = reviewer._get_access_token()
            second = reviewer._get_access_token()

        self.assertEqual(first, "access-token-1")
        self.assertEqual(second, "access-token-2")

    def test_gateway_errors_redact_secret_values(self):
        env = {
            "AGENTIC_PR_REVIEW_ENV_FILE": "missing.env",
            "MODEL_PROVIDER": "gateway",
            "GATEWAY_BASE_URL": "https://gateway.example/deployments/claude/chat/completions",
            "GATEWAY_MODEL": "claude-sonnet-4-6",
            "GATEWAY_APP_KEY": "canary-app-key-secret",
            "GATEWAY_CLIENT_ID": "canary-client-id-secret",
            "GATEWAY_CLIENT_SECRET": "canary-client-secret-value",
            "GATEWAY_TOKEN_URL": "https://gateway.example/oauth2/default/v1/token",
        }

        def fake_urlopen(request, timeout):
            raise HTTPError(
                request.full_url,
                400,
                "Bad Request",
                {},
                BytesIO(b'{"error":"canary-client-secret-value Authorization: Bearer generated-secret-token-123456"}'),
            )

        with patch.dict(os.environ, env, clear=True):
            config = RuntimeConfig.from_env()
            reviewer = GatewayClaudeReviewer(config)
            with patch("agentic_pr_review.gateway_claude.urlopen", fake_urlopen):
                with self.assertRaises(Exception) as caught:
                    reviewer._get_access_token()

        message = str(caught.exception)
        self.assertIn(REDACTED_SECRET, message)
        self.assertIn(REDACTED_AUTH, message)
        self.assertNotIn("canary-client-secret-value", message)
        self.assertNotIn("generated-secret-token-123456", message)

    def test_gateway_invocation_redacts_prompt_before_http_body(self):
        env = {
            "AGENTIC_PR_REVIEW_ENV_FILE": "missing.env",
            "MODEL_PROVIDER": "gateway",
            "GATEWAY_BASE_URL": "https://gateway.example/deployments/claude/chat/completions",
            "GATEWAY_MODEL": "claude-sonnet-4-6",
            "GATEWAY_APP_KEY": "canary-app-key-direct",
            "GATEWAY_CLIENT_ID": "canary-client-id-direct",
            "GATEWAY_CLIENT_SECRET": "canary-direct-secret",
            "GATEWAY_TOKEN_URL": "https://gateway.example/oauth2/default/v1/token",
        }
        bodies = []

        class Response:
            def __init__(self, payload):
                self.payload = payload

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

            def read(self):
                return json.dumps(self.payload).encode("utf-8")

        def fake_urlopen(request, timeout):
            body = request.data.decode("utf-8")
            bodies.append(body)
            if request.full_url.endswith("/token"):
                return Response({"access_token": "access-token-direct", "expires_in": 3600})
            return Response(
                {
                    "model": "claude-sonnet-4-6",
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "summary": "ok",
                                        "overall_status": "looks_good",
                                        "safe_to_publish": True,
                                        "findings": [],
                                    }
                                )
                            }
                        }
                    ],
                }
            )

        with patch.dict(os.environ, env, clear=True):
            config = RuntimeConfig.from_env()
            reviewer = GatewayClaudeReviewer(config)
            with patch("agentic_pr_review.gateway_claude.urlopen", fake_urlopen):
                reviewer._invoke_review("please inspect canary-direct-secret", [])

        model_body = bodies[1]
        self.assertIn(REDACTED_SECRET, model_body)
        self.assertNotIn("canary-direct-secret", model_body)
        self.assertEqual(json.loads(json.loads(model_body)["user"]), {"appkey": "canary-app-key-direct"})

    def test_extract_chat_completion_text(self):
        payload = {"choices": [{"message": {"content": "{\"summary\":\"ok\"}"}}]}

        self.assertEqual(extract_chat_completion_text(payload), '{"summary":"ok"}')

    def test_mask_secret_for_github_actions_only_in_actions(self):
        with patch.dict(os.environ, {"GITHUB_ACTIONS": "true"}, clear=True), patch("sys.stdout", new_callable=StringIO) as out:
            mask_secret_for_github_actions("computed-token")

        self.assertEqual(out.getvalue().strip(), "::add-mask::computed-token")

        with patch.dict(os.environ, {}, clear=True), patch("sys.stdout", new_callable=StringIO) as out:
            mask_secret_for_github_actions("computed-token")

        self.assertEqual(out.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
