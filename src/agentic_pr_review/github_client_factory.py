"""Build GitHub clients from runtime configuration."""

from __future__ import annotations

from .config import RuntimeConfig
from .github_auth import GitHubAppInstallationTokenProvider
from .github_client import GitHubClient


def build_github_client(config: RuntimeConfig) -> GitHubClient:
    config.require_github_app()
    return GitHubClient(
        token_provider=GitHubAppInstallationTokenProvider(
            app_id=str(config.github_app_id),
            installation_id=str(config.github_app_installation_id),
            private_key=str(config.github_app_private_key),
        ),
        auth_mode="app",
    )
