"""Runtime configuration for the local prototype."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


DEFAULT_MAX_PATCH_CHARS = 50_000
DEFAULT_MAX_FILE_CHARS = 650_000
DEFAULT_MAX_MODEL_INPUT_CHARS = 145_000
DEFAULT_GATEWAY_REQUEST_TIMEOUT_SECONDS = 600
DEFAULT_GATEWAY_REVIEW_REQUEST_TIMEOUT_SECONDS = 240
DEFAULT_GATEWAY_REQUEST_MAX_ATTEMPTS = 2
DEFAULT_GATEWAY_REQUEST_RETRY_BACKOFF_SECONDS = 5.0
DEFAULT_ENV_FILES: tuple[str, ...] = ()


def load_local_env_files(paths: tuple[str, ...] = DEFAULT_ENV_FILES) -> None:
    explicit = os.getenv("AGENTIC_PR_REVIEW_ENV_FILE")
    candidates = (explicit,) if explicit else paths
    for item in candidates:
        if not item:
            continue
        path = Path(item)
        if not path.is_file():
            continue
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip().removeprefix("export ").strip()
            value = value.strip().strip("'\"")
            if key and key not in os.environ:
                os.environ[key] = value


@dataclass(frozen=True)
class RuntimeConfig:
    model_provider: str
    github_auth_mode: str
    github_app_id: str | None
    github_app_installation_id: str | None
    github_app_private_key: str | None
    gateway_base_url: str | None
    gateway_model: str | None
    gateway_app_key: str | None
    gateway_client_id: str | None
    gateway_client_secret: str | None
    gateway_token_url: str | None
    gateway_request_timeout_seconds: int = DEFAULT_GATEWAY_REQUEST_TIMEOUT_SECONDS
    gateway_review_request_timeout_seconds: int = DEFAULT_GATEWAY_REVIEW_REQUEST_TIMEOUT_SECONDS
    gateway_request_max_attempts: int = DEFAULT_GATEWAY_REQUEST_MAX_ATTEMPTS
    gateway_request_retry_backoff_seconds: float = DEFAULT_GATEWAY_REQUEST_RETRY_BACKOFF_SECONDS
    max_patch_chars: int = DEFAULT_MAX_PATCH_CHARS
    max_file_chars: int = DEFAULT_MAX_FILE_CHARS
    max_model_input_chars: int = DEFAULT_MAX_MODEL_INPUT_CHARS
    model_cache_dir: str | None = None

    @classmethod
    def from_env(
        cls,
        *,
        max_patch_chars: int = DEFAULT_MAX_PATCH_CHARS,
        max_file_chars: int = DEFAULT_MAX_FILE_CHARS,
        max_model_input_chars: int = DEFAULT_MAX_MODEL_INPUT_CHARS,
    ) -> "RuntimeConfig":
        load_local_env_files()
        app_private_key = os.getenv("AI_REVIEW_PRIVATE_KEY") or os.getenv("GITHUB_APP_PRIVATE_KEY")
        private_key_path = os.getenv("AI_REVIEW_PRIVATE_KEY_PATH") or os.getenv("GITHUB_APP_PRIVATE_KEY_PATH")
        if not app_private_key and private_key_path:
            app_private_key = Path(private_key_path).read_text(encoding="utf-8")
        app_id = os.getenv("AI_REVIEW_APP_ID") or os.getenv("GITHUB_APP_ID")
        installation_id = os.getenv("AI_REVIEW_INSTALLATION_ID") or os.getenv("GITHUB_APP_INSTALLATION_ID")
        github_auth_mode = "app"
        raw_provider = (
            os.getenv("MODEL_PROVIDER")
            or os.getenv("AI_REVIEW_MODEL_PROVIDER")
            or "circuit"
        )
        model_provider = normalize_model_provider(raw_provider)
        gateway_request_timeout_seconds = parse_positive_int_env(
            "GATEWAY_REQUEST_TIMEOUT_SECONDS",
            "CIRCUIT_REQUEST_TIMEOUT_SECONDS",
            default=DEFAULT_GATEWAY_REQUEST_TIMEOUT_SECONDS,
        )
        gateway_review_request_timeout_seconds = parse_positive_int_env(
            "GATEWAY_REVIEW_REQUEST_TIMEOUT_SECONDS",
            "CIRCUIT_REVIEW_REQUEST_TIMEOUT_SECONDS",
            default=min(gateway_request_timeout_seconds, DEFAULT_GATEWAY_REVIEW_REQUEST_TIMEOUT_SECONDS),
        )
        return cls(
            model_provider=model_provider,
            github_auth_mode=github_auth_mode,
            github_app_id=app_id,
            github_app_installation_id=installation_id,
            github_app_private_key=app_private_key,
            gateway_base_url=os.getenv("GATEWAY_BASE_URL") or os.getenv("CIRCUIT_BASE_URL"),
            gateway_model=os.getenv("GATEWAY_MODEL") or os.getenv("CIRCUIT_MODEL"),
            gateway_app_key=os.getenv("GATEWAY_APP_KEY") or os.getenv("CIRCUIT_APP_KEY"),
            gateway_client_id=os.getenv("GATEWAY_CLIENT_ID") or os.getenv("CIRCUIT_CLIENT_ID"),
            gateway_client_secret=os.getenv("GATEWAY_CLIENT_SECRET") or os.getenv("CIRCUIT_CLIENT_SECRET"),
            gateway_token_url=os.getenv("GATEWAY_TOKEN_URL") or os.getenv("CIRCUIT_TOKEN_URL"),
            gateway_request_timeout_seconds=gateway_request_timeout_seconds,
            gateway_review_request_timeout_seconds=gateway_review_request_timeout_seconds,
            gateway_request_max_attempts=parse_positive_int_env(
                "GATEWAY_REQUEST_MAX_ATTEMPTS",
                "CIRCUIT_REQUEST_MAX_ATTEMPTS",
                default=DEFAULT_GATEWAY_REQUEST_MAX_ATTEMPTS,
            ),
            gateway_request_retry_backoff_seconds=parse_nonnegative_float_env(
                "GATEWAY_REQUEST_RETRY_BACKOFF_SECONDS",
                "CIRCUIT_REQUEST_RETRY_BACKOFF_SECONDS",
                default=DEFAULT_GATEWAY_REQUEST_RETRY_BACKOFF_SECONDS,
            ),
            max_patch_chars=max_patch_chars,
            max_file_chars=max_file_chars,
            max_model_input_chars=max_model_input_chars,
            model_cache_dir=os.getenv("AGENTIC_PR_REVIEW_MODEL_CACHE_DIR") or None,
        )

    def require_model(self) -> None:
        missing = []
        if self.model_provider == "gateway":
            if not self.gateway_base_url:
                missing.append("GATEWAY_BASE_URL/CIRCUIT_BASE_URL")
            if not self.gateway_model:
                missing.append("GATEWAY_MODEL/CIRCUIT_MODEL")
            if not self.gateway_app_key:
                missing.append("GATEWAY_APP_KEY/CIRCUIT_APP_KEY")
            if not self.gateway_client_id:
                missing.append("GATEWAY_CLIENT_ID/CIRCUIT_CLIENT_ID")
            if not self.gateway_client_secret:
                missing.append("GATEWAY_CLIENT_SECRET/CIRCUIT_CLIENT_SECRET")
            if not self.gateway_token_url:
                missing.append("GATEWAY_TOKEN_URL/CIRCUIT_TOKEN_URL")
        else:
            raise RuntimeError(f"Unsupported MODEL_PROVIDER '{self.model_provider}'. Use circuit or gateway.")
        if missing:
            raise RuntimeError(f"Missing required model environment variable(s): {', '.join(missing)}")

    def require_github_app(self) -> None:
        missing = []
        if not self.github_app_id:
            missing.append("AI_REVIEW_APP_ID")
        if not self.github_app_installation_id:
            missing.append("AI_REVIEW_INSTALLATION_ID")
        if not self.github_app_private_key:
            missing.append("AI_REVIEW_PRIVATE_KEY")
        if missing:
            raise RuntimeError(f"Missing required GitHub App environment variable(s): {', '.join(missing)}")

    def require_github_auth(self) -> None:
        self.require_github_app()


def normalize_model_provider(value: str) -> str:
    provider = value.strip().lower().replace("_", "-")
    aliases = {
        "circuit": "gateway",
        "circuit-api": "gateway",
        "circuit-gateway": "gateway",
        "custom-gateway": "gateway",
        "gateway-api": "gateway",
    }
    return aliases.get(provider, provider)


def parse_positive_int_env(*names: str, default: int) -> int:
    for name in names:
        raw = os.getenv(name)
        if raw is None or raw.strip() == "":
            continue
        try:
            value = int(raw)
        except ValueError:
            return default
        return max(1, value)
    return default


def parse_nonnegative_float_env(*names: str, default: float) -> float:
    for name in names:
        raw = os.getenv(name)
        if raw is None or raw.strip() == "":
            continue
        try:
            value = float(raw)
        except ValueError:
            return default
        return max(0.0, value)
    return default
