import json
import os
import threading
from io import BytesIO
from io import StringIO
from urllib.error import HTTPError
from urllib.parse import parse_qs
from unittest.mock import patch
import unittest

from agentic_pr_review.gateway_claude import (
    CHUNK_MODEL_INPUT_CHARS,
    CIRCUIT_CHAT_TRANSACTION_LIMIT,
    GatewayClaudeReviewer,
    GatewayModelFormatError,
    GatewayTransientModelError,
    extract_chat_completion_text,
    mask_secret_for_github_actions,
)
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
        self.assertEqual(config.gateway_request_timeout_seconds, 180)
        self.assertEqual(config.gateway_request_max_attempts, 2)
        config.require_model()

    def test_runtime_config_accepts_gateway_retry_settings(self):
        env = {
            "AGENTIC_PR_REVIEW_ENV_FILE": "missing.env",
            "MODEL_PROVIDER": "gateway",
            "GATEWAY_BASE_URL": "https://gateway.example/deployments/claude/chat/completions",
            "GATEWAY_MODEL": "claude-sonnet-4-6",
            "GATEWAY_APP_KEY": "app-key-test",
            "GATEWAY_CLIENT_ID": "client-id-test",
            "GATEWAY_CLIENT_SECRET": "client-secret-test",
            "GATEWAY_TOKEN_URL": "https://gateway.example/oauth2/default/v1/token",
            "GATEWAY_REQUEST_TIMEOUT_SECONDS": "420",
            "GATEWAY_REQUEST_MAX_ATTEMPTS": "4",
            "GATEWAY_REQUEST_RETRY_BACKOFF_SECONDS": "0.25",
        }

        with patch.dict(os.environ, env, clear=True):
            config = RuntimeConfig.from_env()

        self.assertEqual(config.gateway_request_timeout_seconds, 420)
        self.assertEqual(config.gateway_request_max_attempts, 4)
        self.assertEqual(config.gateway_request_retry_backoff_seconds, 0.25)

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
        user_metadata = json.loads(model_payload["user"])
        self.assertEqual(user_metadata["appkey"], "app-key-test")
        self.assertTrue(user_metadata["chat_id"].startswith("agentic-pr-review-"))
        self.assertEqual(user_metadata["session_id"], user_metadata["chat_id"])
        self.assertEqual(user_metadata["conversation_id"], user_metadata["chat_id"])
        self.assertNotIn("model", model_payload)
        self.assertEqual(model_payload["response_format"]["type"], "json_schema")
        self.assertEqual(model_payload["response_format"]["json_schema"]["name"], "agentic_pr_review_output")
        self.assertTrue(model_payload["response_format"]["json_schema"]["strict"])
        self.assertEqual(model_payload["max_tokens"], 123)
        self.assertEqual(len(model_payload["messages"]), 2)
        self.assertEqual(model_payload["messages"][0]["role"], "system")
        self.assertEqual(model_payload["messages"][1]["role"], "user")
        self.assertIn("strict SOAR connectors", model_payload["messages"][0]["content"])
        self.assertEqual(model_payload["messages"][1]["content"], "review this")

    def test_gateway_reuses_chat_metadata_until_circuit_transaction_limit(self):
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
        model_user_metadata = []

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
            if request.full_url.endswith("/token"):
                return Response({"access_token": "access-token-test", "token_type": "Bearer", "expires_in": 3600})
            payload = json.loads(request.data.decode("utf-8"))
            model_user_metadata.append(json.loads(payload["user"]))
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
            for index in range(CIRCUIT_CHAT_TRANSACTION_LIMIT + 1):
                reviewer._invoke_review(f"review {index}", [])

        self.assertEqual(len(model_user_metadata), CIRCUIT_CHAT_TRANSACTION_LIMIT + 1)
        self.assertEqual({item["appkey"] for item in model_user_metadata}, {"app-key-test"})
        first_chat_id = model_user_metadata[0]["chat_id"]
        self.assertTrue(first_chat_id.startswith("agentic-pr-review-"))
        self.assertEqual(
            {item["chat_id"] for item in model_user_metadata[:CIRCUIT_CHAT_TRANSACTION_LIMIT]},
            {first_chat_id},
        )
        self.assertNotEqual(model_user_metadata[CIRCUIT_CHAT_TRANSACTION_LIMIT]["chat_id"], first_chat_id)
        for metadata in model_user_metadata:
            self.assertEqual(metadata["session_id"], metadata["chat_id"])
            self.assertEqual(metadata["conversation_id"], metadata["chat_id"])

    def test_gateway_uses_separate_chat_metadata_for_concurrent_worker_lanes(self):
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
        reviewer = GatewayClaudeReviewer(config)
        barrier = threading.Barrier(2)
        results = []
        lock = threading.Lock()

        def build_body():
            barrier.wait(timeout=2)
            body = reviewer._model_body("review from worker")
            with lock:
                results.append(json.loads(body["user"]))

        workers = [
            threading.Thread(target=build_body, name="deep-review_0"),
            threading.Thread(target=build_body, name="deep-review_1"),
        ]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(timeout=2)

        self.assertEqual(len(results), 2)
        self.assertEqual(len({item["chat_id"] for item in results}), 2)
        for metadata in results:
            self.assertIn("-deep-review_", metadata["chat_id"])
            self.assertEqual(metadata["session_id"], metadata["chat_id"])
            self.assertEqual(metadata["conversation_id"], metadata["chat_id"])

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

    def test_gateway_review_retries_transient_model_timeout(self):
        env = {
            "AGENTIC_PR_REVIEW_ENV_FILE": "missing.env",
            "MODEL_PROVIDER": "gateway",
            "GATEWAY_BASE_URL": "https://gateway.example/deployments/claude/chat/completions",
            "GATEWAY_MODEL": "claude-sonnet-4-6",
            "GATEWAY_APP_KEY": "app-key-test",
            "GATEWAY_CLIENT_ID": "client-id-test",
            "GATEWAY_CLIENT_SECRET": "client-secret-test",
            "GATEWAY_TOKEN_URL": "https://gateway.example/oauth2/default/v1/token",
            "GATEWAY_REQUEST_TIMEOUT_SECONDS": "420",
            "GATEWAY_REQUEST_MAX_ATTEMPTS": "3",
            "GATEWAY_REQUEST_RETRY_BACKOFF_SECONDS": "0",
        }
        calls = []
        model_attempt = 0

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
            nonlocal model_attempt
            calls.append((request, timeout))
            if request.full_url.endswith("/token"):
                return Response({"access_token": "access-token-retry", "expires_in": 3600})
            model_attempt += 1
            if model_attempt == 1:
                raise TimeoutError("timed out")
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
        self.assertEqual([call[0].full_url.endswith("/token") for call in calls], [True, False, False])
        self.assertEqual([call[1] for call in calls], [60, 420, 420])

    def test_gateway_downgrades_response_format_when_gateway_rejects_schema_mode(self):
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
        model_response_formats = []

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
            if request.full_url.endswith("/token"):
                return Response({"access_token": "access-token-test", "expires_in": 3600})
            model_payload = json.loads(request.data.decode("utf-8"))
            model_response_formats.append(model_payload.get("response_format"))
            if model_payload.get("response_format", {}).get("type") == "json_schema":
                raise HTTPError(
                    request.full_url,
                    400,
                    "Bad Request",
                    {},
                    BytesIO(b'{"error":"unsupported response_format json_schema"}'),
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
        self.assertEqual([item["type"] for item in model_response_formats], ["json_schema", "json_object"])
        self.assertEqual(reviewer._response_format_mode, "json_object")

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
        user_metadata = json.loads(json.loads(model_body)["user"])
        self.assertEqual(user_metadata["appkey"], "canary-app-key-direct")
        self.assertTrue(user_metadata["chat_id"].startswith("agentic-pr-review-"))

    def test_gateway_repairs_non_json_model_response(self):
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
        bodies = []
        model_attempt = 0

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
            nonlocal model_attempt
            body = request.data.decode("utf-8")
            bodies.append(body)
            if request.full_url.endswith("/token"):
                return Response({"access_token": "access-token-test", "expires_in": 3600})
            model_attempt += 1
            if model_attempt == 1:
                return Response({"choices": [{"message": {"content": "I found no concrete issue here."}}]})
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
        self.assertIn("Recovered from a non-JSON model response", output["model_notes"])
        self.assertEqual(len(bodies), 3)
        repair_body = json.loads(bodies[2])
        self.assertEqual(repair_body["messages"][0]["role"], "system")
        self.assertEqual(repair_body["messages"][1]["role"], "user")
        self.assertIn("Your previous response was not valid JSON", repair_body["messages"][1]["content"])
        self.assertIn("I found no concrete issue here.", repair_body["messages"][1]["content"])

    def test_gateway_retries_original_review_when_json_repair_fails(self):
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
        bodies = []
        model_attempt = 0

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
            nonlocal model_attempt
            body = request.data.decode("utf-8")
            bodies.append(body)
            if request.full_url.endswith("/token"):
                return Response({"access_token": "access-token-test", "expires_in": 3600})
            model_attempt += 1
            if model_attempt < 3:
                return Response({"choices": [{"message": {"content": "still not json"}}]})
            return Response(
                {
                    "model": "claude-sonnet-4-6",
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "summary": "strict retry ok",
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
            output = reviewer._invoke_review("review this exact chunk", [])

        self.assertEqual(output["summary"], "strict retry ok")
        self.assertIn("strict JSON retry", output["model_notes"])
        self.assertEqual(len(bodies), 4)
        retry_body = json.loads(bodies[3])
        self.assertEqual(retry_body["messages"][0]["role"], "system")
        self.assertEqual(retry_body["messages"][1]["role"], "user")
        self.assertIn("Retry the original review request", retry_body["messages"][1]["content"])
        self.assertIn("review this exact chunk", retry_body["messages"][1]["content"])

    def test_gateway_invalid_json_after_repair_raises_format_error(self):
        env = {
            "AGENTIC_PR_REVIEW_ENV_FILE": "missing.env",
            "MODEL_PROVIDER": "gateway",
            "GATEWAY_BASE_URL": "https://gateway.example/deployments/claude/chat/completions",
            "GATEWAY_MODEL": "claude-sonnet-4-6",
            "GATEWAY_APP_KEY": "app-key-test",
            "GATEWAY_CLIENT_ID": "client-id-test",
            "GATEWAY_CLIENT_SECRET": "canary-format-secret",
            "GATEWAY_TOKEN_URL": "https://gateway.example/oauth2/default/v1/token",
        }
        model_attempt = 0

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
            nonlocal model_attempt
            if request.full_url.endswith("/token"):
                return Response({"access_token": "access-token-test", "expires_in": 3600})
            model_attempt += 1
            return Response(
                {
                    "choices": [
                        {
                            "message": {
                                "content": f"still prose with canary-format-secret attempt {model_attempt}"
                            }
                        }
                    ]
                }
            )

        with patch.dict(os.environ, env, clear=True):
            config = RuntimeConfig.from_env()
            reviewer = GatewayClaudeReviewer(config)
            with patch("agentic_pr_review.gateway_claude.urlopen", fake_urlopen):
                with self.assertRaises(GatewayModelFormatError) as caught:
                    reviewer._invoke_review("review this", [])

        self.assertIn("Gateway model did not return valid JSON after repair and strict retry", str(caught.exception))
        self.assertIn(REDACTED_SECRET, str(caught.exception))
        self.assertNotIn("canary-format-secret", str(caught.exception))

    def test_deep_review_does_not_skip_chunks_when_format_recovery_fails(self):
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
        review_input = {
            "repository": {"full_name": "owner/repo"},
            "pull_request": {"number": 1, "title": "test"},
            "deep_review": {
                "chunks": [
                    {
                        "id": "chunk-1",
                        "path": "__init__.py",
                        "chunk_index": 1,
                        "chunk_total": 1,
                        "diff": "-# 2025\n+# 2025-2026\n",
                    }
                ]
            },
        }

        with patch.dict(os.environ, env, clear=True):
            config = RuntimeConfig.from_env()
        reviewer = GatewayClaudeReviewer(config)

        with patch.object(reviewer, "_invoke_review", side_effect=GatewayModelFormatError("bad json")):
            with self.assertRaises(GatewayModelFormatError):
                reviewer.review_deep(review_input, [])

    def test_deep_review_splits_chunk_after_transient_gateway_failure(self):
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
        large_diff = (
            "--- a/connector.py\n"
            "+++ b/connector.py\n"
            "@@ -1,300 +1,300 @@\n"
            + "\n".join(f"+def generated_{idx}(): return '{'x' * 160}'" for idx in range(300))
        )
        review_input = {
            "repo": "owner/repo",
            "pr": {"number": 1, "title": "test"},
            "full_files": {"connector.py": "def helper():\n    return 1\n" + "x" * 1000},
            "base_files": {},
            "comments": {"issue_comments": [], "review_comments": [], "reviews": []},
            "ci": {"check_runs": []},
            "collector_notes": {},
            "deep_review": {
                "chunks": [
                    {
                        "id": "connector.py:1",
                        "path": "connector.py",
                        "status": "modified",
                        "chunk_index": 1,
                        "chunk_total": 1,
                        "diff": large_diff,
                    }
                ]
            },
        }
        subchunk_output = {
            "summary": "subchunk ok",
            "overall_status": "looks_good",
            "safe_to_publish": True,
            "findings": [],
            "model_notes": "",
        }
        with patch.dict(os.environ, env, clear=True):
            config = RuntimeConfig.from_env()
        reviewer = GatewayClaudeReviewer(config)
        invoke_count = 0

        def fake_invoke(*args, **kwargs):
            nonlocal invoke_count
            invoke_count += 1
            if invoke_count == 1:
                raise GatewayTransientModelError("Gateway model invocation failed: timed out")
            return dict(subchunk_output)

        with patch.object(reviewer, "_invoke_review", side_effect=fake_invoke):
            output = reviewer.review_deep(review_input, [])

        self.assertEqual(output["overall_status"], "looks_good")
        self.assertEqual(output["deep_review"]["reviewed_chunk_count"], 1)
        self.assertIn("smaller focused subchunks", output["chunk_review_outputs"][0]["model_notes"])
        self.assertGreaterEqual(invoke_count, 4)

    def test_deep_review_runs_independent_chunks_concurrently(self):
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
        review_input = {
            "repo": "owner/repo",
            "pr": {"number": 1, "title": "test"},
            "full_files": {
                "one.py": "def one():\n    return 1\n",
                "two.py": "def two():\n    return 2\n",
            },
            "base_files": {},
            "comments": {"issue_comments": [], "review_comments": [], "reviews": []},
            "ci": {"check_runs": []},
            "collector_notes": {},
            "deep_review": {
                "chunks": [
                    {
                        "id": "one.py:1",
                        "path": "one.py",
                        "status": "modified",
                        "chunk_index": 1,
                        "chunk_total": 1,
                        "diff": "@@ -1 +1 @@\n-return 0\n+return 1",
                    },
                    {
                        "id": "two.py:1",
                        "path": "two.py",
                        "status": "modified",
                        "chunk_index": 1,
                        "chunk_total": 1,
                        "diff": "@@ -1 +1 @@\n-return 0\n+return 2",
                    },
                ]
            },
        }
        output_template = {
            "summary": "ok",
            "overall_status": "looks_good",
            "safe_to_publish": True,
            "findings": [],
            "model_notes": "",
        }
        barrier = threading.Barrier(2)
        lock = threading.Lock()
        invoke_count = 0

        def fake_invoke(*args, **kwargs):
            nonlocal invoke_count
            with lock:
                invoke_count += 1
                current_call = invoke_count
            if current_call <= 2:
                barrier.wait(timeout=2)
            return dict(output_template)

        with patch.dict(os.environ, env, clear=True):
            config = RuntimeConfig.from_env()
        reviewer = GatewayClaudeReviewer(config)

        with (
            patch.object(reviewer, "_get_access_token", return_value="access-token-test"),
            patch.object(reviewer, "_invoke_review", side_effect=fake_invoke),
        ):
            output = reviewer.review_deep(review_input, [], deep_concurrency=2)

        self.assertEqual(output["overall_status"], "looks_good")
        self.assertEqual(output["deep_review"]["reviewed_chunk_count"], 2)
        self.assertEqual([item["chunk_path"] for item in output["chunk_review_outputs"]], ["one.py", "two.py"])
        self.assertEqual(invoke_count, 3)

    def test_deep_review_chunk_uses_smaller_prompt_budget_than_global_limit(self):
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
        review_input = {
            "repo": "owner/repo",
            "pr": {"number": 1, "title": "test"},
            "full_files": {
                "connector.py": "def one():\n    return 1\n" + ("x" * 200_000),
                "sample.json": "y" * 80_000,
            },
            "base_files": {"connector.py": "z" * 120_000},
            "comments": {
                "review_comments": [{"path": "connector.py", "body": "comment " * 20_000}],
                "issue_comments": [{"body": "connector.py " + ("issue " * 20_000)}],
                "reviews": [{"body": "review " * 20_000}],
            },
            "ci": {
                "check_runs": [
                    {
                        "name": "pre-commit",
                        "status": "completed",
                        "conclusion": "failure",
                        "output": {"summary": "s" * 100_000, "text": "t" * 100_000},
                    }
                ],
                "failed_check_logs": [{"body": "connector.py " + ("log " * 50_000)}],
            },
            "collector_notes": {},
            "deep_review": {
                "chunks": [
                    {
                        "id": "connector.py:1",
                        "path": "connector.py",
                        "status": "modified",
                        "chunk_index": 1,
                        "chunk_total": 1,
                        "diff": "@@ -1 +1 @@\n-return 0\n+return 1",
                    }
                ]
            },
        }
        output_template = {
            "summary": "ok",
            "overall_status": "looks_good",
            "safe_to_publish": True,
            "findings": [],
            "model_notes": "",
        }
        prompt_lengths = []

        def fake_invoke(user_prompt, *args, **kwargs):
            prompt_lengths.append(len(user_prompt))
            return dict(output_template)

        with patch.dict(os.environ, env, clear=True):
            config = RuntimeConfig.from_env(max_model_input_chars=145_000)
        reviewer = GatewayClaudeReviewer(config)

        with patch.object(reviewer, "_invoke_review", side_effect=fake_invoke):
            reviewer.review_deep(review_input, [], deep_concurrency=1)

        self.assertGreater(config.max_model_input_chars, CHUNK_MODEL_INPUT_CHARS)
        self.assertLessEqual(prompt_lengths[0], CHUNK_MODEL_INPUT_CHARS + 500)

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
