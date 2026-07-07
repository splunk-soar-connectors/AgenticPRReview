import io
import json
import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from agentic_pr_review.sdk_manifest import (
    build_sdk_manifest_from_project,
    extract_repository_zipball,
    safe_manifest_env,
)


class SDKManifestTest(unittest.TestCase):
    def test_extract_repository_zipball_returns_single_root(self):
        raw = io.BytesIO()
        with zipfile.ZipFile(raw, "w") as archive:
            archive.writestr("owner-repo-sha/pyproject.toml", "[project]\nname='x'\n")
            archive.writestr("owner-repo-sha/src/app.py", "app = None\n")

        with tempfile.TemporaryDirectory() as tmp:
            root = extract_repository_zipball(raw.getvalue(), Path(tmp))

        self.assertEqual(root.name, "owner-repo-sha")

    def test_build_sdk_manifest_from_project_success_reads_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            (project / "pyproject.toml").write_text("[tool.soar.app]\nmain_module='src.app:app'\n")

            def fake_run(command, **kwargs):
                env = kwargs["env"]
                self.assertEqual(env["HOME"], str(project / ".agentic_pr_review_home"))
                self.assertNotIn("AI_REVIEW_PRIVATE_KEY", env)
                self.assertNotIn("GITHUB_TOKEN", env)
                self.assertNotIn("CIRCUIT_CLIENT_SECRET", env)
                self.assertNotIn("MODEL_PROVIDER", env)
                output_path = Path(command[-2])
                output_path.write_text(
                    json.dumps(
                        {
                            "name": "sample",
                            "package_name": "phantom_sample",
                            "app_version": "1.0.0",
                            "configuration": {"api_key": {"data_type": "password"}},
                            "actions": [
                                {
                                    "identifier": "lookup",
                                    "action": "lookup",
                                    "read_only": True,
                                }
                            ],
                        }
                    )
                )

                class Completed:
                    returncode = 0
                    stdout = "created"
                    stderr = ""

                return Completed()

            env = {
                "AI_REVIEW_PRIVATE_KEY": "github-app-private-key",
                "GITHUB_TOKEN": "pat-should-not-be-used",
                "MODEL_PROVIDER": "circuit",
                "CIRCUIT_CLIENT_SECRET": "secret",
                "PATH": os.environ.get("PATH", ""),
            }
            with patch.dict(os.environ, env, clear=True), patch(
                "agentic_pr_review.sdk_manifest.choose_manifest_command"
            ) as choose, patch("agentic_pr_review.sdk_manifest.subprocess.run", side_effect=fake_run):
                choose.side_effect = lambda project_dir, output_path: [
                    "soarapps",
                    "manifests",
                    "create",
                    str(output_path),
                    str(project_dir),
                ]
                result = build_sdk_manifest_from_project(project)

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["manifest_summary"]["action_identifiers"], ["lookup"])
        self.assertEqual(result["manifest_summary"]["configuration_fields"], ["api_key"])
        self.assertTrue(result["env_sanitized"])

    def test_safe_manifest_env_uses_isolated_home_and_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            env = safe_manifest_env(project)

        self.assertEqual(env["HOME"], str(project / ".agentic_pr_review_home"))
        self.assertEqual(env["XDG_CACHE_HOME"], str(project / ".agentic_pr_review_cache"))
        self.assertEqual(env["UV_CACHE_DIR"], str(project / ".agentic_pr_review_cache" / "uv"))
        self.assertEqual(env["TMPDIR"], str(project / ".agentic_pr_review_tmp"))

    def test_build_sdk_manifest_from_project_failure_redacts_project_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            (project / "pyproject.toml").write_text("[tool.soar.app]\nmain_module='src.app:app'\n")

            class Completed:
                returncode = 1
                stdout = ""
                stderr = f'Traceback\n  File "{project}/src/app.py", line 7, in <module>\nTypeError: Action function must specify a return type\n'

            with patch("agentic_pr_review.sdk_manifest.choose_manifest_command") as choose, patch(
                "agentic_pr_review.sdk_manifest.subprocess.run", return_value=Completed()
            ):
                choose.side_effect = lambda project_dir, output_path: [
                    "soarapps",
                    "manifests",
                    "create",
                    str(output_path),
                    str(project_dir),
                ]
                result = build_sdk_manifest_from_project(project)

        self.assertEqual(result["status"], "failed")
        self.assertNotIn(str(project), result["stderr"])
        self.assertIn("<project>/src/app.py", result["stderr"])


if __name__ == "__main__":
    unittest.main()
