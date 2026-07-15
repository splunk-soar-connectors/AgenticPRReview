import os
from io import BytesIO
from unittest.mock import Mock, patch
import unittest
from urllib.error import HTTPError

from agentic_pr_review.config import DEFAULT_ENV_FILES, RuntimeConfig
from agentic_pr_review.github_auth import (
    GitHubAppInstallationTokenProvider,
    build_jwt_payload,
    normalize_private_key,
    parse_github_datetime,
)
from agentic_pr_review.github_client import GitHubClient, GitHubError
from agentic_pr_review.github_client_factory import build_github_client
from agentic_pr_review.secret_redactor import REDACTED_SECRET


class GitHubAuthTest(unittest.TestCase):
    def test_normalize_private_key_accepts_escaped_newlines(self):
        begin = "-----BEGIN " + "PRIVATE KEY-----"
        end = "-----END " + "PRIVATE KEY-----"
        raw = f"{begin}\\nabc\\n{end}"

        normalized = normalize_private_key(raw)

        self.assertIn("\nabc\n", normalized)
        self.assertNotIn("\\n", normalized)

    def test_parse_github_datetime_returns_utc_datetime(self):
        parsed = parse_github_datetime("2099-01-01T12:30:00Z")

        self.assertEqual(parsed.isoformat(), "2099-01-01T12:30:00+00:00")

    def test_build_jwt_payload_uses_app_id_as_issuer(self):
        payload = build_jwt_payload("12345")

        self.assertEqual(payload["iss"], "12345")
        self.assertLess(payload["iat"], payload["exp"])

    def test_installation_token_provider_posts_with_app_jwt_and_caches_token(self):
        calls = []

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

            def read(self):
                return b'{"token": "installation-token", "expires_at": "2099-01-01T00:00:00Z"}'

        def fake_urlopen(request, timeout):
            calls.append((request, timeout))
            return Response()

        provider = GitHubAppInstallationTokenProvider(
            app_id="123",
            installation_id="456",
            private_key="PRIVATE KEY",
            jwt_factory=lambda payload, private_key: "app-jwt",
        )

        with patch("agentic_pr_review.github_auth.urlopen", fake_urlopen):
            first = provider.get_token()
            second = provider.get_token()

        self.assertEqual(first, "installation-token")
        self.assertEqual(second, "installation-token")
        self.assertEqual(len(calls), 1)
        request, timeout = calls[0]
        headers = dict(request.header_items())
        self.assertEqual(timeout, 45)
        self.assertEqual(request.full_url, "https://api.github.com/app/installations/456/access_tokens")
        self.assertEqual(headers["Authorization"], "Bearer app-jwt")
        self.assertEqual(headers["Content-type"], "application/json")

    def test_client_uses_token_provider_for_authorization_header(self):
        class Provider:
            def get_token(self):
                return "provider-token"

        client = GitHubClient(token_provider=Provider())

        self.assertEqual(client._headers()["Authorization"], "Bearer provider-token")

    def test_fetch_text_file_encodes_contents_path(self):
        client = GitHubClient()
        with patch.object(
            client,
            "get",
            return_value={"type": "file", "size": 5, "encoding": "base64", "content": "aGVsbG8="},
        ) as get:
            text = client.fetch_text_file("owner/repo", "templates/Tanium Rest.postman_collection.json", "abc123")

        self.assertEqual(text, "hello")
        self.assertEqual(
            get.call_args.args[0],
            "/repos/owner/repo/contents/templates/Tanium%20Rest.postman_collection.json",
        )
        self.assertEqual(get.call_args.kwargs["params"], {"ref": "abc123"})

    def test_actions_log_redirect_drops_github_authorization_header(self):
        class Provider:
            def get_token(self):
                return "provider-token"

        class RedirectError(Exception):
            pass

        client = GitHubClient(token_provider=Provider())
        opener = Mock()

        from urllib.error import HTTPError
        from email.message import Message

        headers = Message()
        headers["Location"] = "https://signed.example/log.zip"
        opener.open.side_effect = HTTPError(
            "https://api.github.com/repos/o/r/actions/jobs/1/logs",
            302,
            "Found",
            headers,
            None,
        )

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

            def read(self):
                return b"log-bytes"

        redirected_requests = []

        def fake_urlopen(request, timeout):
            redirected_requests.append(request)
            return Response()

        with patch("agentic_pr_review.github_client.build_opener", return_value=opener), patch(
            "agentic_pr_review.github_client.urlopen", fake_urlopen
        ):
            data = client.get_actions_job_logs("o/r", 1)

        self.assertEqual(data, b"log-bytes")
        first_headers = dict(opener.open.call_args.args[0].header_items())
        redirected_headers = dict(redirected_requests[0].header_items())
        self.assertEqual(first_headers["Authorization"], "Bearer provider-token")
        self.assertNotIn("Authorization", redirected_headers)

    def test_github_client_redacts_secret_values_from_error_bodies(self):
        env = {"CIRCUIT_CLIENT_SECRET": "canary-github-error-secret"}
        client = GitHubClient(token="provider-token")

        def fake_urlopen(request, timeout):
            raise HTTPError(
                request.full_url,
                500,
                "Server Error",
                {},
                BytesIO(b'{"message":"canary-github-error-secret"}'),
            )

        with patch.dict(os.environ, env, clear=True), patch("agentic_pr_review.github_client.urlopen", fake_urlopen):
            with self.assertRaises(GitHubError) as caught:
                client.get("/repos/owner/repo/pulls/1")

        message = str(caught.exception)
        self.assertIn(REDACTED_SECRET, message)
        self.assertNotIn("canary-github-error-secret", message)

    def test_runtime_config_auto_selects_github_app_auth(self):
        env = {
            "AGENTIC_PR_REVIEW_ENV_FILE": "missing.env",
            "AI_REVIEW_APP_ID": "123",
            "AI_REVIEW_INSTALLATION_ID": "456",
            "AI_REVIEW_PRIVATE_KEY": "key",
        }

        with patch.dict(os.environ, env, clear=True):
            config = RuntimeConfig.from_env()

        self.assertEqual(config.github_auth_mode, "app")
        self.assertEqual(config.github_app_id, "123")
        self.assertEqual(config.github_app_installation_id, "456")
        self.assertEqual(config.github_app_private_key, "key")

    def test_runtime_config_does_not_auto_load_local_token_files(self):
        self.assertEqual(DEFAULT_ENV_FILES, ())

    def test_runtime_config_requires_app_auth_even_if_pat_env_exists(self):
        env = {
            "AGENTIC_PR_REVIEW_ENV_FILE": "missing.env",
            "GITHUB_TOKEN": "pat-token",
        }

        with patch.dict(os.environ, env, clear=True):
            config = RuntimeConfig.from_env()

        self.assertEqual(config.github_auth_mode, "app")
        with self.assertRaisesRegex(RuntimeError, "AI_REVIEW_APP_ID"):
            build_github_client(config)

    def test_runtime_config_ignores_pat_env_when_app_auth_is_complete(self):
        env = {
            "AGENTIC_PR_REVIEW_ENV_FILE": "missing.env",
            "AI_REVIEW_APP_ID": "123",
            "AI_REVIEW_INSTALLATION_ID": "456",
            "AI_REVIEW_PRIVATE_KEY": "key",
            "GITHUB_TOKEN": "pat-token",
        }

        with patch.dict(os.environ, env, clear=True):
            config = RuntimeConfig.from_env()

        self.assertEqual(config.github_auth_mode, "app")
        client = build_github_client(config)
        self.assertEqual(client.auth_mode, "app")


if __name__ == "__main__":
    unittest.main()
