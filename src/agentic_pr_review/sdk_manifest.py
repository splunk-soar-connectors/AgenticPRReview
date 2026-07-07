"""Build generated SOAR SDK manifests for PR head snapshots."""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from .github_client import GitHubClient, GitHubError
from .json_utils import truncate_text
from .sdk_analysis import build_sdk_review_inventory


DEFAULT_SDK_SOURCE_ROOT = Path("/path/to/soar-sdk")


def mark_sdk_manifest_disabled(review_input: dict[str, Any], *, reason: str) -> dict[str, Any]:
    """Record why generated manifest evidence was not collected for an SDK app."""

    inventory = build_sdk_review_inventory(review_input)
    if inventory.get("is_sdk_app"):
        review_input["sdk_manifest"] = {
            "attempted": False,
            "status": "skipped",
            "reason": reason,
            "requires_explicit_enable": True,
        }
    return review_input


def enrich_with_sdk_manifest(
    review_input: dict[str, Any],
    client: GitHubClient,
    repo: str,
    *,
    timeout_seconds: int = 120,
) -> dict[str, Any]:
    """Attach generated SDK manifest evidence when the PR appears to be an SDK app."""

    inventory = build_sdk_review_inventory(review_input)
    if not inventory.get("is_sdk_app"):
        review_input["sdk_manifest"] = {
            "attempted": False,
            "status": "skipped",
            "reason": "No SDK app markers were detected.",
        }
        return review_input

    ref = str(((review_input.get("pr") or {}).get("head") or {}).get("sha") or "")
    if not ref:
        review_input["sdk_manifest"] = {
            "attempted": False,
            "status": "skipped",
            "reason": "PR head SHA is unavailable.",
        }
        return review_input

    try:
        raw_zip = client.get_repository_zipball(repo, ref)
    except GitHubError as exc:
        review_input["sdk_manifest"] = {
            "attempted": True,
            "status": "failed",
            "stage": "download",
            "reason": str(exc),
        }
        return review_input

    with tempfile.TemporaryDirectory(prefix="agentic-pr-sdk-manifest-") as tmp:
        tmp_path = Path(tmp)
        try:
            project_dir = extract_repository_zipball(raw_zip, tmp_path)
        except (ValueError, zipfile.BadZipFile) as exc:
            review_input["sdk_manifest"] = {
                "attempted": True,
                "status": "failed",
                "stage": "extract",
                "reason": str(exc),
            }
            return review_input

        result = build_sdk_manifest_from_project(
            project_dir,
            timeout_seconds=timeout_seconds,
        )

    review_input["sdk_manifest"] = result
    return review_input


def extract_repository_zipball(raw_zip: bytes, destination: Path) -> Path:
    with zipfile.ZipFile(io.BytesIO(raw_zip)) as archive:
        members = [name for name in archive.namelist() if name and not name.endswith("/")]
        if not members:
            raise ValueError("Repository archive is empty.")
        archive.extractall(destination)

    roots = [path for path in destination.iterdir() if path.is_dir()]
    if len(roots) != 1:
        raise ValueError(f"Repository archive extracted to {len(roots)} top-level directories.")
    return roots[0]


def build_sdk_manifest_from_project(
    project_dir: Path,
    *,
    timeout_seconds: int = 120,
) -> dict[str, Any]:
    if not (project_dir / "pyproject.toml").exists():
        return {
            "attempted": True,
            "status": "skipped",
            "reason": "No pyproject.toml was present in the extracted PR head.",
            "project_dir_name": project_dir.name,
        }

    output_path = project_dir / ".agentic_pr_review_manifest.json"
    command = choose_manifest_command(project_dir, output_path)
    env = safe_manifest_env(project_dir)

    try:
        completed = subprocess.run(
            command,
            cwd=project_dir,
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
    except FileNotFoundError as exc:
        return {
            "attempted": True,
            "status": "failed",
            "stage": "execute",
            "command": redact_project_paths(command, project_dir),
            "reason": str(exc),
            "project_dir_name": project_dir.name,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "attempted": True,
            "status": "failed",
            "stage": "timeout",
            "command": redact_project_paths(command, project_dir),
            "reason": f"Timed out after {timeout_seconds} seconds.",
            "stdout": truncate_text(redact_project_text(exc.stdout or "", project_dir), 8000),
            "stderr": truncate_text(redact_project_text(exc.stderr or "", project_dir), 8000),
            "project_dir_name": project_dir.name,
        }

    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    base_result = {
        "attempted": True,
        "command": redact_project_paths(command, project_dir),
        "env_sanitized": True,
        "returncode": completed.returncode,
        "stdout": truncate_text(redact_project_text(stdout, project_dir), 8000),
        "stderr": truncate_text(redact_project_text(stderr, project_dir), 8000),
        "project_dir_name": project_dir.name,
    }
    if completed.returncode != 0:
        return {
            **base_result,
            "status": "failed",
            "stage": "manifest_create",
            "reason": "soarapps manifests create returned a non-zero exit code.",
        }

    try:
        manifest = json.loads(output_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            **base_result,
            "status": "failed",
            "stage": "manifest_read",
            "reason": str(exc),
        }

    return {
        **base_result,
        "status": "success",
        "manifest": manifest,
        "manifest_summary": summarize_manifest(manifest),
    }


def choose_manifest_command(project_dir: Path, output_path: Path) -> list[str]:
    explicit = os.getenv("SOARAPPS_BIN")
    if explicit:
        return [explicit, "manifests", "create", str(output_path), str(project_dir)]

    uv = shutil.which("uv")
    sdk_root = sdk_source_root()
    if uv and sdk_root:
        return [
            uv,
            "run",
            "--project",
            str(project_dir),
            "--with-editable",
            str(sdk_root),
            "python",
            "-m",
            "soar_sdk.cli.cli",
            "manifests",
            "create",
            str(output_path),
            str(project_dir),
        ]

    if uv and DEFAULT_SDK_SOURCE_ROOT.exists():
        return [
            uv,
            "run",
            "--project",
            str(DEFAULT_SDK_SOURCE_ROOT),
            "python",
            "-m",
            "soar_sdk.cli.cli",
            "manifests",
            "create",
            str(output_path),
            str(project_dir),
        ]

    return [
        sys.executable,
        "-m",
        "soar_sdk.cli.cli",
        "manifests",
        "create",
        str(output_path),
        str(project_dir),
    ]


def safe_manifest_env(project_dir: Path) -> dict[str, str]:
    """Return a minimal child-process env for executing untrusted PR manifest code."""

    home = project_dir / ".agentic_pr_review_home"
    cache = project_dir / ".agentic_pr_review_cache"
    tmp = project_dir / ".agentic_pr_review_tmp"
    for path in (home, cache, tmp):
        path.mkdir(exist_ok=True)

    env = {
        "HOME": str(home),
        "TMPDIR": str(tmp),
        "XDG_CACHE_HOME": str(cache),
        "UV_CACHE_DIR": str(cache / "uv"),
        "PYTHONUNBUFFERED": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    if os.getenv("PATH"):
        env["PATH"] = str(os.environ["PATH"])
    if os.getenv("LANG"):
        env["LANG"] = str(os.environ["LANG"])
    if os.getenv("LC_ALL"):
        env["LC_ALL"] = str(os.environ["LC_ALL"])
    return env


def sdk_source_root() -> Path | None:
    configured = os.getenv("SOAR_SDK_SOURCE_ROOT")
    if configured:
        path = Path(configured)
        return path if path.exists() else None
    return DEFAULT_SDK_SOURCE_ROOT if DEFAULT_SDK_SOURCE_ROOT.exists() else None


def summarize_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    actions = manifest.get("actions") if isinstance(manifest.get("actions"), list) else []
    configuration = manifest.get("configuration") or {}
    if isinstance(configuration, dict):
        config_names = sorted(str(key) for key in configuration)
    elif isinstance(configuration, list):
        config_names = sorted(
            str(item.get("name") or item.get("key") or item.get("identifier"))
            for item in configuration
            if isinstance(item, dict)
        )
    else:
        config_names = []
    return {
        "name": manifest.get("name"),
        "package_name": manifest.get("package_name"),
        "app_version": manifest.get("app_version"),
        "python_version": manifest.get("python_version"),
        "min_phantom_version": manifest.get("min_phantom_version"),
        "action_identifiers": [
            str(action.get("identifier") or "")
            for action in actions
            if isinstance(action, dict)
        ],
        "configuration_fields": config_names,
        "has_webhook": bool(manifest.get("webhook")),
        "supports_es_polling": bool(manifest.get("supports_es_polling")),
    }


def redact_project_paths(command: list[str], project_dir: Path) -> list[str]:
    project = str(project_dir)
    return [part.replace(project, "<project>") for part in command]


def redact_project_text(text: str, project_dir: Path) -> str:
    return text.replace(str(project_dir), "<project>")
