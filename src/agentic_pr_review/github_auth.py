"""GitHub authentication helpers."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import time
from typing import Any, Callable, Protocol
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from .secret_redactor import redact_obj, redact_text


class GitHubAuthError(RuntimeError):
    pass


class TokenProvider(Protocol):
    def get_token(self) -> str | None:
        pass


class GitHubAppInstallationTokenProvider:
    def __init__(
        self,
        *,
        app_id: str,
        installation_id: str,
        private_key: str,
        api_url: str = "https://api.github.com",
        jwt_factory: Callable[[dict[str, Any], str], str] | None = None,
    ) -> None:
        self.app_id = str(app_id).strip()
        self.installation_id = str(installation_id).strip()
        self.private_key = normalize_private_key(private_key)
        self.api_url = api_url.rstrip("/")
        self.jwt_factory = jwt_factory or create_github_app_jwt
        self._cached_token: str | None = None
        self._expires_at: datetime | None = None

    def get_token(self) -> str:
        if self._cached_token and self._expires_at:
            if datetime.now(timezone.utc) < self._expires_at - timedelta(minutes=5):
                return self._cached_token
        token, expires_at = self._create_installation_token()
        self._cached_token = token
        self._expires_at = expires_at
        return token

    def _create_installation_token(self) -> tuple[str, datetime]:
        app_jwt = self.jwt_factory(build_jwt_payload(self.app_id), self.private_key)
        url = f"{self.api_url}/app/installations/{self.installation_id}/access_tokens"
        request = Request(
            url,
            data=b"{}",
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {app_jwt}",
                "Content-Type": "application/json",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "agentic-pr-review-prototype",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=45) as response:
                raw = response.read()
        except HTTPError as exc:
            body = redact_text(exc.read().decode("utf-8", errors="replace"))
            raise GitHubAuthError(f"GitHub App token request failed {exc.code} for {redact_text(url)}: {body}") from exc

        payload = json.loads(raw.decode("utf-8"))
        token = str(payload.get("token") or "")
        expires_at_raw = str(payload.get("expires_at") or "")
        if not token or not expires_at_raw:
            raise GitHubAuthError(f"GitHub App token response was missing token or expires_at: {redact_obj(payload)}")
        return token, parse_github_datetime(expires_at_raw)


def build_jwt_payload(app_id: str) -> dict[str, Any]:
    now = int(time.time())
    return {
        "iat": now - 60,
        "exp": now + 9 * 60,
        "iss": app_id,
    }


def create_github_app_jwt(payload: dict[str, Any], private_key: str) -> str:
    try:
        import jwt
    except ImportError as exc:  # pragma: no cover - exercised only without optional dependency
        raise GitHubAuthError(
            "PyJWT with crypto support is required for GitHub App auth. "
            "Install project dependencies with `pip install -e .`."
        ) from exc
    return jwt.encode(payload, private_key, algorithm="RS256")


def normalize_private_key(private_key: str) -> str:
    value = private_key.strip()
    if "\\n" in value and "\n" not in value:
        value = value.replace("\\n", "\n")
    return value


def parse_github_datetime(value: str) -> datetime:
    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)
