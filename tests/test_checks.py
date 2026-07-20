import unittest

from agentic_pr_review.checks import run_deterministic_checks


class DeterministicChecksTest(unittest.TestCase):
    def categories(self, review_input):
        return {finding["category"] for finding in run_deterministic_checks(review_input)}

    def titles(self, review_input):
        return {finding["title"] for finding in run_deterministic_checks(review_input)}

    def test_oauth_body_credentials_without_basic_auth_is_flagged(self):
        review_input = {
            "pr": {"title": "test", "body": ""},
            "changed_files": [{"filename": "connector.py"}],
            "full_files": {
                "connector.py": (
                    "def _get_token(self):\n"
                    "    payload = {\n"
                    "        \"grant_type\": \"client_credentials\",\n"
                    "        \"client_id\": self._client_id,\n"
                    "        \"client_secret\": self._client_secret,\n"
                    "    }\n"
                    "    return self._make_rest_call('/oauth', data=payload, method='post')\n"
                )
            },
            "ci": {"check_runs": [], "statuses": []},
        }

        findings = run_deterministic_checks(review_input)

        self.assertTrue(any(finding["category"] == "api_auth_correctness" for finding in findings))

    def test_oauth_token_json_body_signal_is_flagged(self):
        review_input = {
            "pr": {"title": "oauth", "body": ""},
            "changed_files": [
                {
                    "filename": "connector.py",
                    "patch": (
                        "@@ -1,1 +1,3 @@\n"
                        "+def _get_oauth_token(self):\n"
                        "+    payload = {'grant_type': 'client_credentials'}\n"
                        "+    return self._make_rest_call('/oauth/token', json_data=payload, method='post')\n"
                    ),
                }
            ],
            "full_files": {
                "connector.py": (
                    "def _get_oauth_token(self):\n"
                    "    payload = {'grant_type': 'client_credentials'}\n"
                    "    return self._make_rest_call('/oauth/token', json_data=payload, method='post')\n"
                )
            },
            "ci": {"check_runs": [], "statuses": []},
        }

        self.assertIn("OAuth/token request appears to send a JSON body without form-encoding evidence", self.titles(review_input))

    def test_oauth_v2_authorize_with_v1_token_endpoint_is_flagged(self):
        review_input = {
            "pr": {"title": "oauth", "body": ""},
            "changed_files": [
                {
                    "filename": "src/auth.py",
                    "patch": (
                        "@@ -1,1 +1,9 @@\n"
                        "+def get_auth_code_flow(asset):\n"
                        "+    return AuthorizationCodeFlow(\n"
                        "+        authorization_endpoint=f\"https://login.microsoftonline.com/{tenant}/oauth2/v2.0/authorize\",\n"
                        "+        token_endpoint=f\"https://login.microsoftonline.com/{tenant}/oauth2/token\",\n"
                        "+    )\n"
                    ),
                }
            ],
            "full_files": {
                "src/auth.py": (
                    "def get_auth_code_flow(asset):\n"
                    "    return AuthorizationCodeFlow(\n"
                    "        authorization_endpoint=f\"https://login.microsoftonline.com/{tenant}/oauth2/v2.0/authorize\",\n"
                    "        token_endpoint=f\"https://login.microsoftonline.com/{tenant}/oauth2/token\",\n"
                    "    )\n"
                )
            },
            "ci": {"check_runs": [], "statuses": []},
        }

        self.assertIn("OAuth authorization code flow mixes v2.0 authorize with v1 token endpoint", self.titles(review_input))

    def test_removed_sdk_manifest_is_not_treated_as_current_app_json(self):
        review_input = {
            "pr": {"title": "sdk migration", "body": ""},
            "changed_files": [
                {"filename": "cloudstorage.json", "status": "removed", "patch": "@@ -1 +0,0 @@\n-{}"},
                {"filename": "pyproject.toml", "status": "modified", "patch": "+[tool.soar.app]\n+main_module = \"src.app:app\""},
                {"filename": "src/app.py", "status": "added", "patch": "+app = App(...)"},
                {"filename": "src/asset.py", "status": "added", "patch": "+class Asset(BaseAsset): pass"},
            ],
            "full_files": {
                "cloudstorage.json": '{"type": "sandbox", "python_version": "3.9", "app_version": "2.4.1", "actions": []}',
                "pyproject.toml": '[tool.soar.app]\nmain_module = "src.app:app"\n',
                "src/app.py": "from soar_sdk.app import App\napp = App(name='Microsoft OneDrive')\n",
                "src/asset.py": "from soar_sdk.asset import BaseAsset, AssetField\nclass Asset(BaseAsset): pass\n",
            },
            "ci": {"check_runs": [], "statuses": []},
        }

        titles = self.titles(review_input)

        self.assertNotIn("Changed app JSON python_version may be stale", titles)
        self.assertNotIn("Connector metadata changed without an app_version bump", titles)

    def test_graph_output_model_required_like_fields_are_flagged(self):
        review_input = {
            "pr": {"title": "sdk graph", "body": ""},
            "changed_files": [{"filename": "src/actions/list_drive.py", "status": "added"}],
            "full_files": {
                "src/actions/list_drive.py": (
                    "from soar_sdk.action import ActionOutput, OutputField\n"
                    "# Microsoft Graph OneDrive output models\n"
                    "class OwnerUserOutput(ActionOutput):\n"
                    "    email: str = OutputField(example_values=['test@example.com'])\n"
                    "class OwnerOutput(ActionOutput):\n"
                    "    user: OwnerUserOutput | None\n"
                    "class ListDriveOutput(ActionOutput):\n"
                    "    id: str = OutputField(example_values=['drive-id'])\n"
                    "    owner: OwnerOutput | None\n"
                    "    lastModifiedDateTime: str = OutputField(example_values=['2026-01-01T00:00:00Z'])\n"
                    "def list_drive():\n"
                    "    return [ListDriveOutput(**drive) for drive in drives]\n"
                )
            },
            "ci": {"check_runs": [], "statuses": []},
        }

        self.assertIn("Microsoft Graph output models require fields that Graph can omit", self.titles(review_input))

    def test_upload_output_minimal_response_and_retry_gaps_are_flagged(self):
        review_input = {
            "pr": {"title": "upload", "body": ""},
            "changed_files": [{"filename": "src/actions/upload_file.py", "status": "added"}],
            "full_files": {
                "src/actions/upload_file.py": (
                    "from typing import Any, BinaryIO\n"
                    "import httpx\n"
                    "from soar_sdk.action import ActionOutput\n"
                    "class ParentreferenceOutput(ActionOutput):\n"
                    "    driveId: str | None\n"
                    "class UploadFileOutput(ActionOutput):\n"
                    "    id: str\n"
                    "    parentReference: ParentreferenceOutput | None\n"
                    "def _upload_file_chunks(upload_url: str, file_obj: BinaryIO, file_size: int) -> dict[str, Any]:\n"
                    "    chunk_start = 0\n"
                    "    while chunk_start < file_size:\n"
                    "        headers = {'Content-Range': f'bytes {chunk_start}-3/{file_size}'}\n"
                    "        response = httpx.put(upload_url, headers=headers, content=b'data')\n"
                    "        response.raise_for_status()\n"
                    "        response_json = response.json()\n"
                    "        next_expected_ranges = response_json.get('nextExpectedRanges')\n"
                    "        if not next_expected_ranges:\n"
                    "            return response_json\n"
                    "        chunk_start = int(next_expected_ranges[0].split('-', 1)[0])\n"
                    "def upload_file():\n"
                    "    upload_response = _upload_file_chunks(upload_url, file_obj, file_size)\n"
                    "    return UploadFileOutput(**upload_response)\n"
                )
            },
            "ci": {"check_runs": [], "statuses": []},
        }

        titles = self.titles(review_input)

        self.assertIn("Upload completion output requires parentReference from Graph response", titles)
        self.assertIn("Upload-session chunk loop has no retry or backoff path for transient Graph failures", titles)

    def test_download_buffer_before_vault_attachment_is_flagged(self):
        review_input = {
            "pr": {"title": "download", "body": ""},
            "changed_files": [{"filename": "src/actions/get_file.py", "status": "added"}],
            "full_files": {
                "src/actions/get_file.py": (
                    "import httpx\n"
                    "def get_file(soar):\n"
                    "    file_response = httpx.get(download_url, timeout=30.0)\n"
                    "    file_response.raise_for_status()\n"
                    "    file_content = file_response.content\n"
                    "    return soar.vault.create_attachment(1, file_content, 'sample.txt')\n"
                )
            },
            "ci": {"check_runs": [], "statuses": []},
        }

        self.assertIn("OneDrive download buffers the entire file before vault attachment", self.titles(review_input))

    def test_literal_token_status_message_is_not_sensitive_logging(self):
        review_input = {
            "pr": {"title": "test", "body": ""},
            "changed_files": [{"filename": "connector.py"}],
            "full_files": {
                "connector.py": 'def f(self):\n    self.debug_print("Token not found. Refreshing it.")\n'
            },
            "ci": {"check_runs": [], "statuses": []},
        }

        findings = run_deterministic_checks(review_input)

        self.assertFalse(any(finding["category"] == "unsafe_logging" for finding in findings))

    def test_status_code_only_log_is_not_sensitive_logging(self):
        review_input = {
            "pr": {"title": "test", "body": ""},
            "changed_files": [{"filename": "connector.py"}],
            "full_files": {
                "connector.py": (
                    "def f(response):\n"
                    "    logger.info(f\"Successfully processed data. Status: {response.status_code}\")\n"
                )
            },
            "ci": {"check_runs": [], "statuses": []},
        }

        findings = run_deterministic_checks(review_input)

        self.assertFalse(any(finding["category"] == "unsafe_logging" for finding in findings))

    def test_precommit_comment_failures_are_flagged(self):
        review_input = {
            "pr": {"title": "test", "body": ""},
            "changed_files": [],
            "full_files": {},
            "comments": {
                "issue_comments": [
                    {
                        "body": (
                            "FAILED - app_package_name : GUID and app package name not listed\n"
                            "FAILED - valid_app_name_and_guid : GUID and app not listed\n"
                        ),
                        "url": "https://github.example/comment",
                    }
                ]
            },
            "ci": {"check_runs": [], "statuses": []},
        }

        findings = run_deterministic_checks(review_input)

        self.assertTrue(any(finding["category"] == "precommit" for finding in findings))

    def test_static_test_failure_comment_is_flagged_with_specific_details(self):
        review_input = {
            "pr": {"title": "test", "body": ""},
            "changed_files": [],
            "full_files": {},
            "comments": {
                "issue_comments": [
                    {
                        "body": "Static test failures: Additional logging, license, min phantom version & verbosity",
                        "url": "https://github.example/comment",
                    }
                ]
            },
            "ci": {"check_runs": [], "statuses": []},
        }

        findings = run_deterministic_checks(review_input)
        precommit = [finding for finding in findings if finding["category"] == "precommit"]

        self.assertEqual(len(precommit), 1)
        self.assertIn("min phantom version", precommit[0]["evidence"].lower())
        self.assertIn("minimum platform", precommit[0]["suggested_fix"].lower())
        self.assertEqual(precommit[0]["confidence"], "high")

    def test_merge_conflict_state_is_flagged(self):
        review_input = {
            "pr": {
                "title": "test",
                "body": "",
                "state": "open",
                "mergeable": False,
                "mergeable_state": "dirty",
            },
            "changed_files": [],
            "full_files": {},
            "comments": {},
            "ci": {"check_runs": [], "statuses": []},
        }

        findings = run_deterministic_checks(review_input)

        self.assertTrue(any(finding["category"] == "merge_conflict" for finding in findings))

    def test_blocked_mergeability_includes_specific_failed_log_detail(self):
        review_input = {
            "pr": {
                "title": "test",
                "body": "",
                "state": "open",
                "mergeable": True,
                "mergeable_state": "blocked",
            },
            "changed_files": [],
            "full_files": {},
            "comments": {},
            "ci": {
                "check_runs": [],
                "statuses": [],
                "failed_check_logs": [
                    {
                        "name": "pre-commit",
                        "log_excerpt": "hook id: ruff\nconnector.py:1:1: F401 `os` imported but unused",
                    }
                ],
            },
        }

        findings = run_deterministic_checks(review_input)
        merge_findings = [finding for finding in findings if finding["category"] == "merge_conflict"]

        self.assertEqual(len(merge_findings), 1)
        self.assertIn("connector.py:1:1", merge_findings[0]["evidence"])
        self.assertEqual(merge_findings[0]["confidence"], "high")

    def test_tls_verify_false_is_flagged(self):
        review_input = {
            "pr": {"title": "test", "body": ""},
            "changed_files": [{"filename": "connector.py"}],
            "full_files": {
                "connector.py": "def f(self):\n    return requests.get(self._url, verify=False, timeout=30)\n"
            },
            "ci": {"check_runs": [], "statuses": []},
        }

        self.assertIn("TLS certificate verification is explicitly disabled", self.titles(review_input))

    def test_changed_app_json_stale_python_version_is_flagged(self):
        review_input = {
            "pr": {"title": "test", "body": ""},
            "changed_files": [{"filename": "sample.json"}],
            "full_files": {
                "sample.json": '{"type": "sandbox", "python_version": "3.9", "actions": []}'
            },
            "ci": {"check_runs": [], "statuses": []},
        }

        self.assertIn("Changed app JSON python_version may be stale", self.titles(review_input))

    def test_product_version_copied_from_app_version_is_flagged(self):
        review_input = {
            "pr": {"title": "test", "body": ""},
            "changed_files": [{"filename": "sample.json"}],
            "full_files": {
                "sample.json": (
                    '{"type": "sandbox", "python_version": "3.13", '
                    '"app_version": "1.0.0", "product_version": "1.0.0", "actions": []}'
                )
            },
            "ci": {"check_runs": [], "statuses": []},
        }

        self.assertIn("Product version appears to use the SOAR app version", self.titles(review_input))

    def test_latest_tested_version_copied_from_app_version_is_flagged(self):
        review_input = {
            "pr": {"title": "test", "body": ""},
            "changed_files": [{"filename": "sample.json"}],
            "full_files": {
                "sample.json": (
                    '{"type": "sandbox", "python_version": "3.13", '
                    '"app_version": "1.0.0", '
                    '"latest_tested_versions": ["1.0.0"], "actions": []}'
                )
            },
            "ci": {"check_runs": [], "statuses": []},
        }

        self.assertIn("Latest tested version appears to use the SOAR app version", self.titles(review_input))

    def test_python_syntax_error_is_flagged(self):
        review_input = {
            "pr": {"title": "test", "body": ""},
            "changed_files": [{"filename": "connector.py"}],
            "full_files": {"connector.py": "def broken(:\n    pass\n"},
            "ci": {"check_runs": [], "statuses": []},
        }

        findings = run_deterministic_checks(review_input)
        syntax = [finding for finding in findings if finding["title"] == "Changed Python file does not compile"]

        self.assertEqual(len(syntax), 1)
        self.assertEqual(syntax[0]["category"], "precommit")
        self.assertEqual(syntax[0]["confidence"], "high")

    def test_conflict_markers_are_flagged(self):
        review_input = {
            "pr": {"title": "test", "body": ""},
            "changed_files": [{"filename": "connector.py"}],
            "full_files": {
                "connector.py": (
                    "def f():\n"
                    "<<<<<<< HEAD\n"
                    "    return 1\n"
                    "=======\n"
                    "    return 2\n"
                    ">>>>>>> branch\n"
                )
            },
            "ci": {"check_runs": [], "statuses": []},
        }

        self.assertIn("Unresolved merge conflict marker is present", self.titles(review_input))

    def test_misspelled_debug_print_is_flagged_with_suggestion(self):
        review_input = {
            "pr": {"title": "test", "body": ""},
            "changed_files": [{"filename": "connector.py"}],
            "full_files": {"connector.py": 'def f(self):\n    self.degub_print("hello")\n'},
            "ci": {"check_runs": [], "statuses": []},
        }

        findings = run_deterministic_checks(review_input)
        typo = [finding for finding in findings if finding["title"] == "Connector logging method appears misspelled"]

        self.assertEqual(len(typo), 1)
        self.assertIn("debug_print", typo[0]["suggested_code"])

    def test_missing_version_bump_is_flagged(self):
        review_input = {
            "pr": {"title": "test", "body": ""},
            "changed_files": [{"filename": "connector.py", "patch": "+ token refresh"}],
            "full_files": {
                "sample.json": '{"app_version": "1.0.0", "actions": []}',
                "connector.py": "def f(self):\n    self._refresh_token()\n",
            },
            "base_files": {
                "sample.json": '{"app_version": "1.0.0", "actions": []}'
            },
            "ci": {"check_runs": [], "statuses": []},
        }

        self.assertIn("Connector metadata changed without an app_version bump", self.titles(review_input))

    def test_summary_schema_mismatch_is_flagged(self):
        review_input = {
            "pr": {"title": "test", "body": ""},
            "changed_files": [{"filename": "connector.py"}],
            "full_files": {
                "sample.json": (
                    '{"actions": [{"identifier": "lookup", "action": "lookup", "output": []}]}'
                ),
                "connector.py": (
                    "def _handle_lookup(self, param):\n"
                    "    summary = action_result.update_summary({})\n"
                    "    summary[\"risk_score\"] = 7\n"
                    "    return action_result.set_status(phantom.APP_SUCCESS)\n"
                ),
            },
            "ci": {"check_runs": [], "statuses": []},
        }

        self.assertIn("Code emits summary fields missing from app JSON output", self.titles(review_input))

    def test_polling_timestamp_sdi_is_flagged(self):
        review_input = {
            "pr": {"title": "test", "body": ""},
            "changed_files": [{"filename": "connector.py"}],
            "full_files": {
                "connector.py": (
                    "def _handle_on_poll(self, param):\n"
                    "    artifact = {\"source_data_identifier\": datetime.utcnow().isoformat()}\n"
                )
            },
            "ci": {"check_runs": [], "statuses": []},
        }

        self.assertIn("polling_checkpoint", self.categories(review_input))

    def test_polling_container_lookup_per_group_is_flagged(self):
        review_input = {
            "pr": {"title": "test", "body": ""},
            "changed_files": [{"filename": "connector.py"}],
            "full_files": {
                "connector.py": (
                    "def _handle_on_poll(self, param):\n"
                    "    detections_by_rule = self._group_detections()\n"
                    "    for rule_id, detections in detections_by_rule.items():\n"
                    "        container = self._check_for_existing_container(rule_id)\n"
                    "        artifact = {'source_data_identifier': rule_id}\n"
                    "        self.save_container(container)\n"
                )
            },
            "ci": {"check_runs": [], "statuses": []},
        }

        self.assertIn("Polling container lookup may scale with source group cardinality", self.titles(review_input))

    def test_list_action_without_pagination_is_flagged(self):
        review_input = {
            "pr": {"title": "test", "body": ""},
            "changed_files": [{"filename": "connector.py"}],
            "full_files": {
                "sample.json": (
                    '{"actions": [{"identifier": "list_alerts", "action": "list alerts", "output": []}]}'
                ),
                "connector.py": (
                    "def _handle_list_alerts(self, param):\n"
                    "    response = self._make_rest_call('/alerts')\n"
                    "    action_result.add_data(response)\n"
                ),
            },
            "ci": {"check_runs": [], "statuses": []},
        }

        self.assertIn("pagination", self.categories(review_input))

    def test_unencoded_url_interpolation_is_flagged(self):
        review_input = {
            "pr": {"title": "test", "body": ""},
            "changed_files": [{"filename": "connector.py"}],
            "full_files": {
                "connector.py": (
                    "def _handle_get_user(self, param):\n"
                    "    user_id = param['user_id']\n"
                    "    endpoint = f'/users/{user_id}/details'\n"
                    "    return self._make_rest_call(endpoint)\n"
                )
            },
            "ci": {"check_runs": [], "statuses": []},
        }

        self.assertIn("validation", self.categories(review_input))

    def test_swallowed_broad_exception_is_flagged(self):
        review_input = {
            "pr": {"title": "test", "body": ""},
            "changed_files": [
                {
                    "filename": "connector.py",
                    "patch": (
                        "@@ -1,1 +1,5 @@\n"
                        "+def _handle_lookup(self, param):\n"
                        "+    try:\n"
                        "+        self._make_rest_call('/lookup')\n"
                        "+    except Exception:\n"
                        "+        return phantom.APP_SUCCESS\n"
                    ),
                }
            ],
            "full_files": {
                "connector.py": (
                    "def _handle_lookup(self, param):\n"
                    "    try:\n"
                    "        self._make_rest_call('/lookup')\n"
                    "    except Exception:\n"
                    "        return phantom.APP_SUCCESS\n"
                )
            },
            "ci": {"check_runs": [], "statuses": []},
        }

        self.assertIn("Broad exception handler swallows connector errors", self.titles(review_input))

    def test_unreachable_code_after_return_is_flagged(self):
        review_input = {
            "pr": {"title": "test", "body": ""},
            "changed_files": [
                {
                    "filename": "connector.py",
                    "patch": (
                        "@@ -1,1 +1,3 @@\n"
                        "+def _handle_lookup(self, param):\n"
                        "+    return action_result.set_status(phantom.APP_SUCCESS)\n"
                        "+    action_result.add_data({'id': 'never emitted'})\n"
                    ),
                }
            ],
            "full_files": {
                "connector.py": (
                    "def _handle_lookup(self, param):\n"
                    "    return action_result.set_status(phantom.APP_SUCCESS)\n"
                    "    action_result.add_data({'id': 'never emitted'})\n"
                )
            },
            "ci": {"check_runs": [], "statuses": []},
        }

        self.assertIn("Code after a terminating statement is unreachable", self.titles(review_input))

    def test_risky_change_without_tests_is_flagged(self):
        review_input = {
            "pr": {"title": "test", "body": ""},
            "changed_files": [{"filename": "connector.py", "patch": "+ refresh oauth token"}],
            "full_files": {"connector.py": "def f(self):\n    self._refresh_oauth_token()\n"},
            "ci": {"check_runs": [], "statuses": []},
        }

        self.assertIn("missing_tests", self.categories(review_input))

    def test_declared_on_poll_params_not_used_are_flagged(self):
        review_input = {
            "pr": {"title": "test", "body": ""},
            "changed_files": [{"filename": "connector.py"}],
            "full_files": {
                "sample.json": (
                    '{"actions": [{"identifier": "on_poll", "action": "on poll", '
                    '"parameters": [{"name": "start_time"}], "output": []}]}'
                ),
                "connector.py": "def _handle_on_poll(self, param):\n    return phantom.APP_SUCCESS\n",
            },
            "ci": {"check_runs": [], "statuses": []},
        }

        self.assertIn("on_poll parameters are declared but not visibly used", self.titles(review_input))

    def test_added_app_json_mapping_hint_is_flagged(self):
        review_input = {
            "pr": {"title": "new app", "body": ""},
            "changed_files": [{"filename": "sample.json", "status": "added"}],
            "full_files": {
                "sample.json": (
                    '{"appid": "sample", "package_name": "sample", "name": "Sample", "actions": []}'
                )
            },
            "ci": {"check_runs": [], "statuses": []},
        }

        self.assertIn("New app may need app-id/app-name mapping updates", self.titles(review_input))

    def test_base64_decode_before_try_on_changed_line_is_flagged(self):
        review_input = {
            "pr": {"title": "client cert", "body": ""},
            "changed_files": [
                {
                    "filename": "connector.py",
                    "patch": (
                        "@@ -1,4 +1,8 @@\n"
                        " def _handle_test_connectivity(self, param):\n"
                        "+    cert_bytes = base64.b64decode(param['public_cert'])\n"
                        "+    try:\n"
                        "+        self._write_cert(cert_bytes)\n"
                        "+    except Exception as e:\n"
                        "+        raise ActionFailure(str(e)) from e\n"
                    ),
                }
            ],
            "full_files": {
                "connector.py": (
                    "def _handle_test_connectivity(self, param):\n"
                    "    cert_bytes = base64.b64decode(param['public_cert'])\n"
                    "    try:\n"
                    "        self._write_cert(cert_bytes)\n"
                    "    except Exception as e:\n"
                    "        raise ActionFailure(str(e)) from e\n"
                )
            },
            "ci": {"check_runs": [], "statuses": []},
        }

        self.assertIn("base64.b64decode runs before connector error handling", self.titles(review_input))

    def test_removed_timeout_is_flagged_as_regression_signal(self):
        review_input = {
            "pr": {"title": "refactor request", "body": ""},
            "changed_files": [
                {
                    "filename": "connector.py",
                    "patch": (
                        "@@ -10,3 +10,3 @@\n"
                        " def _get_token(self):\n"
                        "-    return requests.post(self._token_url, data=payload, timeout=30)\n"
                        "+    return requests.post(self._token_url, data=payload)\n"
                    ),
                }
            ],
            "full_files": {
                "connector.py": "def _get_token(self):\n    return requests.post(self._token_url, data=payload)\n"
            },
            "ci": {"check_runs": [], "statuses": []},
        }

        self.assertIn("Diff removes request timeout handling without visible replacement", self.titles(review_input))

    def test_removed_validation_is_flagged_as_whole_diff_signal(self):
        review_input = {
            "pr": {"title": "simplify url handling", "body": ""},
            "changed_files": [
                {
                    "filename": "connector.py",
                    "patch": (
                        "@@ -20,5 +20,3 @@\n"
                        " def _handle_get_user(self, param):\n"
                        "-    user_id = urllib.parse.quote(param['user_id'], safe='')\n"
                        "-    if not user_id:\n"
                        "-        raise ActionFailure('user_id is required')\n"
                        "+    user_id = param['user_id']\n"
                        "     return self._make_rest_call(f'/users/{user_id}')\n"
                    ),
                }
            ],
            "full_files": {
                "connector.py": (
                    "def _handle_get_user(self, param):\n"
                    "    user_id = param['user_id']\n"
                    "    return self._make_rest_call(f'/users/{user_id}')\n"
                )
            },
            "ci": {"check_runs": [], "statuses": []},
        }

        self.assertIn("Diff removes validation or connector-friendly error handling without visible replacement", self.titles(review_input))

    def test_removed_tests_with_risky_code_change_is_flagged(self):
        review_input = {
            "pr": {"title": "change oauth refresh", "body": ""},
            "changed_files": [
                {
                    "filename": "connector.py",
                    "patch": (
                        "@@ -40,2 +40,3 @@\n"
                        "-    self._refresh_oauth_token(old=True)\n"
                        "+    self._refresh_oauth_token(new=True)\n"
                    ),
                },
                {
                    "filename": "tests/test_connector.py",
                    "patch": (
                        "@@ -12,5 +12,0 @@\n"
                        "-def test_refresh_oauth_token_failure():\n"
                        "-    assert connector.refresh() == phantom.APP_ERROR\n"
                    ),
                },
            ],
            "full_files": {"connector.py": "def f(self):\n    self._refresh_oauth_token(new=True)\n"},
            "ci": {"check_runs": [], "statuses": []},
        }

        self.assertIn("Risky connector behavior changed while test coverage was removed", self.titles(review_input))

    def test_removed_release_notes_with_code_change_is_flagged(self):
        review_input = {
            "pr": {"title": "change action", "body": ""},
            "changed_files": [
                {"filename": "connector.py", "patch": "@@ -1,1 +1,1 @@\n+def f():\n+    return 1\n"},
                {"filename": "release_notes/unreleased.md", "status": "removed", "patch": ""},
            ],
            "full_files": {"connector.py": "def f():\n    return 1\n"},
            "ci": {"check_runs": [], "statuses": []},
        }

        self.assertIn("Code or metadata changed while release notes were removed", self.titles(review_input))

    def test_sdk_migration_missing_legacy_asset_field_is_flagged(self):
        review_input = {
            "pr": {"title": "sdk migration", "body": ""},
            "changed_files": [
                {"filename": "legacy.json", "status": "removed", "patch": ""},
                {"filename": "pyproject.toml", "status": "added", "patch": "+[tool.soar.app]\n+main_module = \"src.app:app\""},
                {"filename": "src/app.py", "status": "added", "patch": "+app = App(...)"},
                {"filename": "src/asset.py", "status": "added", "patch": "+class Asset(BaseAsset): pass"},
            ],
            "full_files": {
                "pyproject.toml": "[project]\nname = \"sample\"\n[tool.soar.app]\nmain_module = \"src.app:app\"\n",
                "src/app.py": (
                    "from soar_sdk.app import App\n"
                    "from .asset import Asset\n"
                    "app = App(asset_cls=Asset, name='sample', appid='12345678-1234-5678-9012-123456789012', app_type='sandbox', logo='logo.svg', logo_dark='logo_dark.svg', product_vendor='ExampleVendor', product_name='Sample', publisher='ExampleVendor')\n"
                    "@app.test_connectivity()\n"
                    "def test_connectivity(asset: Asset) -> None:\n"
                    "    pass\n"
                ),
                "src/asset.py": (
                    "from soar_sdk.asset import BaseAsset, AssetField\n"
                    "class Asset(BaseAsset):\n"
                    "    client_id: str = AssetField()\n"
                ),
            },
            "base_files": {
                "legacy.json": (
                    '{"configuration": {"client_id": {"data_type": "string"}, "client_secret": {"data_type": "password"}}, '
                    '"actions": [{"identifier": "lookup", "action": "lookup", "output": []}]}'
                )
            },
            "ci": {"check_runs": [], "statuses": []},
        }

        self.assertIn("Legacy asset configuration fields are missing from the SDK asset model", self.titles(review_input))

    def test_sdk_migration_missing_legacy_actions_are_flagged(self):
        review_input = {
            "pr": {"title": "sdk migration", "body": ""},
            "changed_files": [
                {"filename": "legacy.json", "status": "removed", "patch": ""},
                {"filename": "pyproject.toml", "status": "added", "patch": "+[tool.soar.app]\n+main_module = \"src.app:app\""},
                {"filename": "src/app.py", "status": "added", "patch": "+app = App(...)"},
            ],
            "full_files": {
                "pyproject.toml": "[tool.soar.app]\nmain_module = \"src.app:app\"\n",
                "src/app.py": (
                    "from soar_sdk.app import App\n"
                    "from soar_sdk.asset import BaseAsset\n"
                    "app = App(name='sample', appid='12345678-1234-5678-9012-123456789012', app_type='sandbox', logo='logo.svg', logo_dark='logo_dark.svg', product_vendor='ExampleVendor', product_name='Sample', publisher='ExampleVendor')\n"
                    "@app.test_connectivity()\n"
                    "def test_connectivity(asset: BaseAsset) -> None:\n"
                    "    pass\n"
                    "@app.action(read_only=True)\n"
                    "def lookup(params: Params) -> ActionOutput:\n"
                    "    return ActionOutput()\n"
                ),
            },
            "base_files": {
                "legacy.json": '{"actions": [{"identifier": "lookup", "action": "lookup", "output": []}, {"identifier": "delete_file", "action": "delete file", "output": []}]}'
            },
            "ci": {"check_runs": [], "statuses": []},
        }

        self.assertIn("Legacy actions are not registered in the SDK app", self.titles(review_input))

    def test_sdk_migration_custom_view_without_view_handler_is_flagged(self):
        review_input = {
            "pr": {"title": "sdk migration", "body": ""},
            "changed_files": [
                {"filename": "legacy.json", "status": "removed", "patch": ""},
                {"filename": "pyproject.toml", "status": "added", "patch": "+[tool.soar.app]\n+main_module = \"src.app:app\""},
                {"filename": "src/app.py", "status": "added", "patch": "+app = App(...)"},
            ],
            "full_files": {
                "pyproject.toml": "[tool.soar.app]\nmain_module = \"src.app:app\"\n",
                "src/app.py": (
                    "from soar_sdk.app import App\n"
                    "from soar_sdk.action_results import ActionOutput\n"
                    "from soar_sdk.params import Params\n"
                    "app = App(name='sample', appid='12345678-1234-5678-9012-123456789012', app_type='sandbox', logo='logo.svg', logo_dark='logo_dark.svg', product_vendor='ExampleVendor', product_name='Sample', publisher='ExampleVendor')\n"
                    "@app.test_connectivity()\n"
                    "def test_connectivity() -> None:\n"
                    "    pass\n"
                    "@app.action()\n"
                    "def lookup(params: Params) -> ActionOutput:\n"
                    "    return ActionOutput()\n"
                ),
            },
            "base_files": {
                "legacy.json": '{"actions": [{"identifier": "lookup", "action": "lookup", "render": {"type": "custom"}, "output": []}]}'
            },
            "ci": {"check_runs": [], "statuses": []},
        }

        self.assertIn("Legacy custom views were not visibly migrated to SDK view handlers", self.titles(review_input))

    def test_sdk_migration_custom_view_with_action_view_handler_is_not_flagged(self):
        review_input = {
            "pr": {"title": "sdk migration", "body": ""},
            "changed_files": [
                {"filename": "legacy.json", "status": "removed", "patch": ""},
                {"filename": "pyproject.toml", "status": "added", "patch": "+[tool.soar.app]\n+main_module = \"src.app:app\""},
                {"filename": "src/app.py", "status": "added", "patch": "+app = App(...)"},
            ],
            "full_files": {
                "pyproject.toml": "[tool.soar.app]\nmain_module = \"src.app:app\"\n",
                "src/app.py": (
                    "from soar_sdk.app import App\n"
                    "from soar_sdk.action_results import ActionOutput\n"
                    "from soar_sdk.params import Params\n"
                    "app = App(name='sample', appid='12345678-1234-5678-9012-123456789012', app_type='sandbox', logo='logo.svg', logo_dark='logo_dark.svg', product_vendor='ExampleVendor', product_name='Sample', publisher='ExampleVendor')\n"
                    "def lookup_view(outputs):\n"
                    "    return {}\n"
                    "@app.test_connectivity()\n"
                    "def test_connectivity() -> None:\n"
                    "    pass\n"
                    "@app.action(view_handler=app.view_handler()(lookup_view))\n"
                    "def lookup(params: Params) -> ActionOutput:\n"
                    "    return ActionOutput()\n"
                ),
            },
            "base_files": {
                "legacy.json": '{"actions": [{"identifier": "lookup", "action": "lookup", "render": {"type": "custom"}, "output": []}]}'
            },
            "ci": {"check_runs": [], "statuses": []},
        }

        self.assertNotIn("Legacy custom views were not visibly migrated to SDK view handlers", self.titles(review_input))

    def test_sdk_migration_rest_handler_without_sdk_webhook_is_flagged(self):
        review_input = {
            "pr": {"title": "sdk migration", "body": ""},
            "changed_files": [
                {"filename": "legacy.json", "status": "removed", "patch": ""},
                {"filename": "pyproject.toml", "status": "added", "patch": "+[tool.soar.app]\n+main_module = \"src.app:app\""},
                {"filename": "src/app.py", "status": "added", "patch": "+app = App(...)"},
            ],
            "full_files": {
                "pyproject.toml": "[tool.soar.app]\nmain_module = \"src.app:app\"\n",
                "src/app.py": (
                    "from soar_sdk.app import App\n"
                    "app = App(name='sample', appid='12345678-1234-5678-9012-123456789012', app_type='sandbox', logo='logo.svg', logo_dark='logo_dark.svg', product_vendor='ExampleVendor', product_name='Sample', publisher='ExampleVendor')\n"
                    "@app.test_connectivity()\n"
                    "def test_connectivity() -> None:\n"
                    "    pass\n"
                ),
            },
            "base_files": {
                "legacy.json": '{"rest_handler": "legacy.handle_rest", "actions": [{"identifier": "test_connectivity", "action": "test connectivity", "output": []}]}'
            },
            "ci": {"check_runs": [], "statuses": []},
        }

        self.assertIn("Legacy REST/webhook handling was not visibly migrated to SDK webhooks", self.titles(review_input))

    def test_sdk_mutating_action_default_read_only_is_flagged(self):
        review_input = {
            "pr": {"title": "sdk app", "body": ""},
            "changed_files": [{"filename": "src/app.py", "status": "added"}],
            "full_files": {
                "src/app.py": (
                    "from soar_sdk.app import App\n"
                    "from soar_sdk.action_results import ActionOutput\n"
                    "from soar_sdk.params import Params\n"
                    "app = App(name='sample', appid='12345678-1234-5678-9012-123456789012', app_type='sandbox', logo='logo.svg', logo_dark='logo_dark.svg', product_vendor='ExampleVendor', product_name='Sample', publisher='ExampleVendor')\n"
                    "@app.test_connectivity()\n"
                    "def test_connectivity() -> None:\n"
                    "    pass\n"
                    "@app.action()\n"
                    "def delete_file(params: Params) -> ActionOutput:\n"
                    "    return ActionOutput()\n"
                )
            },
            "ci": {"check_runs": [], "statuses": []},
        }

        self.assertIn("Mutating SDK action relies on the read_only=True default", self.titles(review_input))

    def test_sdk_action_missing_return_annotation_is_flagged(self):
        review_input = {
            "pr": {"title": "sdk app", "body": ""},
            "changed_files": [{"filename": "src/app.py", "status": "added"}],
            "full_files": {
                "src/app.py": (
                    "from soar_sdk.app import App\n"
                    "from soar_sdk.params import Params\n"
                    "app = App(name='sample', appid='12345678-1234-5678-9012-123456789012', app_type='sandbox', logo='logo.svg', logo_dark='logo_dark.svg', product_vendor='ExampleVendor', product_name='Sample', publisher='ExampleVendor')\n"
                    "@app.test_connectivity()\n"
                    "def test_connectivity() -> None:\n"
                    "    pass\n"
                    "@app.action()\n"
                    "def lookup(params: Params):\n"
                    "    return None\n"
                )
            },
            "ci": {"check_runs": [], "statuses": []},
        }

        self.assertIn("SDK action is missing a return type annotation", self.titles(review_input))

    def test_sdk_action_explicit_params_and_output_classes_are_not_flagged_as_missing_annotations(self):
        review_input = {
            "pr": {"title": "sdk app", "body": ""},
            "changed_files": [{"filename": "src/app.py", "status": "added"}],
            "full_files": {
                "src/app.py": (
                    "from soar_sdk.action_results import ActionOutput\n"
                    "from soar_sdk.app import App\n"
                    "from soar_sdk.params import Params\n"
                    "app = App(name='sample', appid='12345678-1234-5678-9012-123456789012', app_type='sandbox', logo='logo.svg', logo_dark='logo_dark.svg', product_vendor='ExampleVendor', product_name='Sample', publisher='ExampleVendor')\n"
                    "class LookupParams(Params):\n"
                    "    pass\n"
                    "class LookupOutput(ActionOutput):\n"
                    "    pass\n"
                    "@app.test_connectivity()\n"
                    "def test_connectivity() -> None:\n"
                    "    pass\n"
                    "@app.action(params_class=LookupParams, output_class=LookupOutput)\n"
                    "def lookup(params):\n"
                    "    return LookupOutput()\n"
                )
            },
            "ci": {"check_runs": [], "statuses": []},
        }

        titles = self.titles(review_input)

        self.assertNotIn("SDK action is missing a return type annotation", titles)
        self.assertNotIn("SDK action params argument is missing a type annotation", titles)

    def test_sdk_register_action_visible_function_missing_types_is_flagged(self):
        review_input = {
            "pr": {"title": "sdk app", "body": ""},
            "changed_files": [
                {"filename": "src/app.py", "status": "added"},
                {"filename": "src/actions.py", "status": "added"},
            ],
            "full_files": {
                "src/app.py": (
                    "from soar_sdk.app import App\n"
                    "app = App(name='sample', appid='12345678-1234-5678-9012-123456789012', app_type='sandbox', logo='logo.svg', logo_dark='logo_dark.svg', product_vendor='ExampleVendor', product_name='Sample', publisher='ExampleVendor')\n"
                    "@app.test_connectivity()\n"
                    "def test_connectivity() -> None:\n"
                    "    pass\n"
                    "app.register_action('actions:lookup')\n"
                ),
                "src/actions.py": (
                    "def lookup(params):\n"
                    "    return None\n"
                ),
            },
            "ci": {"check_runs": [], "statuses": []},
        }

        titles = self.titles(review_input)

        self.assertIn("SDK action is missing a return type annotation", titles)
        self.assertIn("SDK action params argument is missing a type annotation", titles)

    def test_sdk_register_action_explicit_classes_are_not_flagged_as_missing_annotations(self):
        review_input = {
            "pr": {"title": "sdk app", "body": ""},
            "changed_files": [
                {"filename": "src/app.py", "status": "added"},
                {"filename": "src/actions.py", "status": "added"},
            ],
            "full_files": {
                "src/app.py": (
                    "from soar_sdk.action_results import ActionOutput\n"
                    "from soar_sdk.app import App\n"
                    "from soar_sdk.params import Params\n"
                    "class LookupParams(Params):\n"
                    "    pass\n"
                    "class LookupOutput(ActionOutput):\n"
                    "    pass\n"
                    "app = App(name='sample', appid='12345678-1234-5678-9012-123456789012', app_type='sandbox', logo='logo.svg', logo_dark='logo_dark.svg', product_vendor='ExampleVendor', product_name='Sample', publisher='ExampleVendor')\n"
                    "@app.test_connectivity()\n"
                    "def test_connectivity() -> None:\n"
                    "    pass\n"
                    "app.register_action('actions:lookup', params_class=LookupParams, output_class=LookupOutput)\n"
                ),
                "src/actions.py": (
                    "def lookup(params):\n"
                    "    return None\n"
                ),
            },
            "ci": {"check_runs": [], "statuses": []},
        }

        titles = self.titles(review_input)

        self.assertNotIn("SDK action is missing a return type annotation", titles)
        self.assertNotIn("SDK action params argument is missing a type annotation", titles)

    def test_sdk_make_request_requires_make_request_params_as_first_argument(self):
        review_input = {
            "pr": {"title": "sdk app", "body": ""},
            "changed_files": [{"filename": "src/app.py", "status": "added"}],
            "full_files": {
                "src/app.py": (
                    "from soar_sdk.action_results import MakeRequestOutput\n"
                    "from soar_sdk.app import App\n"
                    "from soar_sdk.asset import BaseAsset\n"
                    "from soar_sdk.params import MakeRequestParams\n"
                    "class Asset(BaseAsset):\n"
                    "    pass\n"
                    "app = App(name='sample', appid='12345678-1234-5678-9012-123456789012', app_type='sandbox', logo='logo.svg', logo_dark='logo_dark.svg', product_vendor='ExampleVendor', product_name='Sample', publisher='ExampleVendor', asset_cls=Asset)\n"
                    "@app.test_connectivity()\n"
                    "def test_connectivity() -> None:\n"
                    "    pass\n"
                    "@app.make_request()\n"
                    "def make_request(asset: Asset, params: MakeRequestParams) -> MakeRequestOutput:\n"
                    "    return MakeRequestOutput(status_code=200, response_body='{}')\n"
                )
            },
            "ci": {"check_runs": [], "statuses": []},
        }

        self.assertIn(
            "SDK make_request action does not use MakeRequestParams as the first argument",
            self.titles(review_input),
        )

    def test_sdk_test_connectivity_return_value_is_flagged(self):
        review_input = {
            "pr": {"title": "sdk app", "body": ""},
            "changed_files": [{"filename": "src/app.py", "status": "added"}],
            "full_files": {
                "src/app.py": (
                    "from soar_sdk.app import App\n"
                    "app = App(name='sample', appid='12345678-1234-5678-9012-123456789012', app_type='sandbox', logo='logo.svg', logo_dark='logo_dark.svg', product_vendor='ExampleVendor', product_name='Sample', publisher='ExampleVendor')\n"
                    "@app.test_connectivity()\n"
                    "def test_connectivity() -> bool:\n"
                    "    return True\n"
                )
            },
            "ci": {"check_runs": [], "statuses": []},
        }

        titles = self.titles(review_input)

        self.assertIn("SDK test connectivity has a non-None return annotation", titles)
        self.assertIn("SDK test connectivity returns a value", titles)

    def test_sdk_call_style_test_connectivity_is_recognized(self):
        review_input = {
            "pr": {"title": "sdk migration", "body": ""},
            "changed_files": [
                {"filename": "pyproject.toml", "status": "added", "patch": ""},
                {"filename": "src/app.py", "status": "added", "patch": ""},
                {"filename": "src/actions/connectivity.py", "status": "added", "patch": ""},
            ],
            "full_files": {
                "pyproject.toml": "[tool.soar.app]\nmain_module = \"src.app:app\"\n",
                "src/app.py": (
                    "from soar_sdk.app import App\n"
                    "from soar_sdk.asset import BaseAsset\n"
                    "from src.actions.connectivity import run_test_connectivity\n"
                    "app = App(name='Search API', appid='12345678-1234-5678-9012-123456789012', app_type='information', logo='logo.svg', logo_dark='logo_dark.svg', product_vendor='search_api', product_name='search_api', publisher='ExampleVendor', asset_cls=BaseAsset)\n"
                    "app.test_connectivity()(run_test_connectivity)\n"
                ),
                "src/actions/connectivity.py": (
                    "def run_test_connectivity(asset) -> None:\n"
                    "    pass\n"
                ),
            },
            "base_files": {},
            "ci": {"check_runs": [], "statuses": []},
        }

        self.assertNotIn("SDK app does not register test connectivity", self.titles(review_input))

    def test_sdk_params_list_annotation_is_flagged_but_string_allow_list_is_not(self):
        review_input = {
            "pr": {"title": "sdk app", "body": ""},
            "changed_files": [{"filename": "src/app.py", "status": "added"}],
            "full_files": {
                "src/app.py": (
                    "from soar_sdk.app import App\n"
                    "from soar_sdk.action_results import ActionOutput\n"
                    "from soar_sdk.params import Param, Params\n"
                    "app = App(name='sample', appid='12345678-1234-5678-9012-123456789012', app_type='sandbox', logo='logo.svg', logo_dark='logo_dark.svg', product_vendor='ExampleVendor', product_name='Sample', publisher='ExampleVendor')\n"
                    "class LookupParams(Params):\n"
                    "    good_ids: str = Param(allow_list=True)\n"
                    "    bad_ids: list[str] = Param(allow_list=True)\n"
                    "@app.test_connectivity()\n"
                    "def test_connectivity() -> None:\n"
                    "    pass\n"
                    "@app.action()\n"
                    "def lookup(params: LookupParams) -> ActionOutput:\n"
                    "    return ActionOutput()\n"
                )
            },
            "ci": {"check_runs": [], "statuses": []},
        }

        titles = self.titles(review_input)

        self.assertIn("SDK Params field uses an unsupported list annotation", titles)
        self.assertNotIn("SDK allow_list parameter appears typed as a string", titles)
        self.assertNotIn("SDK allow_list parameter may have a scalar type", titles)

    def test_sdk_serializer_rejected_field_shapes_are_flagged(self):
        review_input = {
            "pr": {"title": "sdk app", "body": ""},
            "changed_files": [{"filename": "src/app.py", "status": "added"}],
            "full_files": {
                "src/app.py": (
                    "from soar_sdk.action_results import ActionOutput\n"
                    "from soar_sdk.app import App\n"
                    "from soar_sdk.asset import AssetField, BaseAsset\n"
                    "from soar_sdk.params import MakeRequestParams, Param, Params\n"
                    "app = App(name='sample', appid='12345678-1234-5678-9012-123456789012', app_type='sandbox', logo='logo.svg', logo_dark='logo_dark.svg', product_vendor='ExampleVendor', product_name='Sample', publisher='ExampleVendor')\n"
                    "class Asset(BaseAsset):\n"
                    "    app_version: str = AssetField()\n"
                    "    secret_enabled: bool = AssetField(sensitive=True)\n"
                    "    certificate: int = AssetField(is_file=True)\n"
                    "class LookupParams(Params):\n"
                    "    secret_enabled: bool = Param(sensitive=True)\n"
                    "class BrokenMakeRequestParams(MakeRequestParams):\n"
                    "    not_allowed: str = Param()\n"
                    "@app.test_connectivity()\n"
                    "def test_connectivity() -> None:\n"
                    "    pass\n"
                    "@app.action()\n"
                    "def lookup(params: LookupParams) -> ActionOutput:\n"
                    "    return ActionOutput()\n"
                )
            },
            "ci": {"check_runs": [], "statuses": []},
        }

        titles = self.titles(review_input)

        self.assertIn("SDK asset model uses a reserved SOAR config field", titles)
        self.assertIn("SDK sensitive asset field is not typed as string", titles)
        self.assertIn("SDK file asset field is not typed as string", titles)
        self.assertIn("SDK sensitive action parameter is not typed as string", titles)
        self.assertIn("SDK MakeRequestParams subclass defines unsupported fields", titles)

    def test_sdk_live_only_tests_are_flagged(self):
        review_input = {
            "pr": {"title": "sdk migration", "body": ""},
            "changed_files": [
                {"filename": "legacy.json", "status": "removed", "patch": ""},
                {"filename": "pyproject.toml", "status": "added", "patch": "+[tool.soar.app]\n+main_module = \"src.app:app\""},
                {"filename": "src/app.py", "status": "added", "patch": "+app = App(...)"},
                {"filename": "tests/conftest.py", "status": "added", "patch": "+def live_asset_config(): pass"},
                {"filename": "tests/test_list_drive_live.py", "status": "added", "patch": "+def test_list_drive_live(): pass"},
            ],
            "full_files": {
                "pyproject.toml": "[tool.soar.app]\nmain_module = \"src.app:app\"\n",
                "src/app.py": "from soar_sdk.app import App\napp = App(...)\n",
                "tests/conftest.py": (
                    "def live_asset_config():\n"
                    "    raise RuntimeError('Missing required live test environment variables: tenant_id, client_secret')\n"
                ),
                "tests/test_list_drive_live.py": (
                    "def test_list_drive_live(live_asset_config):\n"
                    "    assert live_asset_config['tenant_id']\n"
                ),
            },
            "base_files": {"legacy.json": '{"actions": [{"identifier": "lookup"}]}'},
            "ci": {"check_runs": [], "statuses": []},
        }

        self.assertIn("SDK migration tests are live-only and require external credentials", self.titles(review_input))

    def test_sdk_live_tests_with_offline_coverage_are_not_flagged(self):
        review_input = {
            "pr": {"title": "sdk migration", "body": ""},
            "changed_files": [
                {"filename": "legacy.json", "status": "removed", "patch": ""},
                {"filename": "pyproject.toml", "status": "added", "patch": "+[tool.soar.app]\n+main_module = \"src.app:app\""},
                {"filename": "src/app.py", "status": "added", "patch": "+app = App(...)"},
                {"filename": "tests/conftest.py", "status": "added", "patch": "+def live_asset_config(): pass"},
                {"filename": "tests/test_models.py", "status": "added", "patch": "+def test_accepts_sparse(): pass"},
            ],
            "full_files": {
                "pyproject.toml": "[tool.soar.app]\nmain_module = \"src.app:app\"\n",
                "src/app.py": "from soar_sdk.app import App\napp = App(...)\n",
                "tests/conftest.py": (
                    "def live_asset_config():\n"
                    "    raise RuntimeError('Missing required live test environment variables: tenant_id, client_secret')\n"
                ),
                "tests/test_models.py": (
                    "def test_accepts_sparse_graph_payload(monkeypatch):\n"
                    "    assert {'id': 'x'}\n"
                ),
            },
            "base_files": {"legacy.json": '{"actions": [{"identifier": "lookup"}]}'},
            "ci": {"check_runs": [], "statuses": []},
        }

        self.assertNotIn("SDK migration tests are live-only and require external credentials", self.titles(review_input))

    def test_sdk_manifest_project_failure_is_flagged(self):
        review_input = {
            "pr": {"title": "sdk app", "body": ""},
            "changed_files": [{"filename": "src/app.py", "status": "added"}],
            "full_files": {
                "pyproject.toml": "[tool.soar.app]\nmain_module = \"src.app:app\"\n",
                "src/app.py": "from soar_sdk.app import App\napp = App(...)\n",
            },
            "sdk_manifest": {
                "attempted": True,
                "status": "failed",
                "stage": "manifest_create",
                "returncode": 1,
                "stderr": (
                    'Traceback\n  File "<project>/src/app.py", line 9, in <module>\n'
                    "TypeError: Action function must specify a return type via type hint\n"
                ),
            },
            "ci": {"check_runs": [], "statuses": []},
        }

        findings = run_deterministic_checks(review_input)
        manifest = [finding for finding in findings if finding["title"] == "SDK generated manifest build fails"]

        self.assertEqual(len(manifest), 1)
        self.assertEqual(manifest[0]["file"], "src/app.py")
        self.assertEqual(manifest[0]["line"], 9)

    def test_sdk_manifest_local_tool_failure_is_not_flagged(self):
        review_input = {
            "pr": {"title": "sdk app", "body": ""},
            "changed_files": [{"filename": "src/app.py", "status": "added"}],
            "full_files": {
                "pyproject.toml": "[tool.soar.app]\nmain_module = \"src.app:app\"\n",
                "src/app.py": "from soar_sdk.app import App\napp = App(...)\n",
            },
            "sdk_manifest": {
                "attempted": True,
                "status": "failed",
                "stage": "manifest_create",
                "returncode": 1,
                "stderr": "Could not resolve host: pypi.org",
            },
            "ci": {"check_runs": [], "statuses": []},
        }

        self.assertNotIn("SDK generated manifest build fails", self.titles(review_input))

    def test_generated_sdk_manifest_missing_legacy_metadata_is_flagged(self):
        review_input = {
            "pr": {"title": "sdk migration", "body": ""},
            "changed_files": [
                {"filename": "legacy.json", "status": "removed", "patch": ""},
                {"filename": "pyproject.toml", "status": "added", "patch": "+[tool.soar.app]\n+main_module = \"src.app:app\""},
                {"filename": "src/app.py", "status": "added", "patch": "+app = App(...)"},
            ],
            "full_files": {
                "pyproject.toml": "[tool.soar.app]\nmain_module = \"src.app:app\"\n",
                "src/app.py": (
                    "from soar_sdk.app import App\n"
                    "app = App(name='sample', appid='12345678-1234-5678-9012-123456789012', app_type='sandbox', logo='logo.svg', logo_dark='logo_dark.svg', product_vendor='ExampleVendor', product_name='Sample', publisher='ExampleVendor')\n"
                    "@app.test_connectivity()\n"
                    "def test_connectivity() -> None:\n"
                    "    pass\n"
                ),
            },
            "base_files": {
                "legacy.json": (
                    '{"configuration": {"api_key": {"data_type": "password"}, "base_url": {"data_type": "string"}}, '
                    '"actions": [{"identifier": "lookup", "action": "lookup", "output": []}, '
                    '{"identifier": "delete_file", "action": "delete file", "output": []}, '
                    '{"identifier": "upload_file", "action": "upload file", "output": []}]}'
                )
            },
            "sdk_manifest": {
                "attempted": True,
                "status": "success",
                "manifest": {
                    "configuration": {"api_key": {"data_type": "password"}},
                    "actions": [
                        {"identifier": "lookup", "action": "lookup", "read_only": True},
                        {"identifier": "delete_file", "action": "delete file", "read_only": True},
                    ],
                },
            },
            "ci": {"check_runs": [], "statuses": []},
        }

        titles = self.titles(review_input)

        self.assertIn("Generated SDK manifest drops legacy actions", titles)
        self.assertIn("Generated SDK manifest drops legacy asset configuration fields", titles)
        self.assertIn("Generated SDK manifest marks a mutating action read-only", titles)

    def test_sdk_migration_flags_output_summary_validation_and_test_regressions(self):
        review_input = {
            "pr": {"title": "sdk migration", "body": "All existing output fields are preserved."},
            "changed_files": [
                {"filename": "legacy.json", "status": "removed", "patch": ""},
                {"filename": "pyproject.toml", "status": "added", "patch": "+[tool.soar.app]\n+main_module = \"src.app:app\""},
                {"filename": "src/app.py", "status": "added", "patch": "+from soar_sdk.app import App\n"},
                {"filename": "src/client.py", "status": "added", "patch": "+sleep(timeout_seconds - 1)\n"},
                {"filename": "README.md", "status": "modified", "patch": "+action_result.data.*.data.result_sets.*.rows.*.data.*.text\n"},
            ],
            "full_files": {
                "pyproject.toml": "[tool.soar.app]\nmain_module = \"src.app:app\"\n",
                "src/app.py": (
                    "from soar_sdk.abstract import SOARClient\n"
                    "from soar_sdk.action_results import ActionOutput, OutputField\n"
                    "from soar_sdk.app import App\n"
                    "from soar_sdk.asset import BaseAsset\n"
                    "from soar_sdk.params import Param, Params\n"
                    "app = App(name='sample', appid='12345678-1234-5678-9012-123456789012', app_type='sandbox', logo='logo.svg', logo_dark='logo_dark.svg', product_vendor='ExampleVendor', product_name='Sample', publisher='ExampleVendor', asset_cls=BaseAsset)\n"
                    "class RunQueryParams(Params):\n"
                    "    return_when_n_results_available: float | None = Param(required=False)\n"
                    "class RunQuerySummaryOutput(ActionOutput):\n"
                    "    number_of_rows: int = OutputField(example_values=[1])\n"
                    "class RunQueryOutput(ActionOutput):\n"
                    "    data: dict\n"
                    "@app.test_connectivity()\n"
                    "def test_connectivity() -> None:\n"
                    "    pass\n"
                    "@app.action()\n"
                    "def run_query(params: RunQueryParams, soar: SOARClient, asset: BaseAsset) -> RunQueryOutput:\n"
                    "    return_when_n = int(params.return_when_n_results_available) if params.return_when_n_results_available is not None else None\n"
                    "    if return_when_n and return_when_n < 0:\n"
                    "        from soar_sdk.exceptions import ActionFailure\n"
                    "        raise ActionFailure('bad')\n"
                    "    raise ActionFailure('missing saved question')\n"
                    "class GetQuestionResultsCellOutput(ActionOutput):\n"
                    "    text: str = OutputField(example_values=['x'])\n"
                    "class GetQuestionResultsRowsOutput(ActionOutput):\n"
                    "    data: list[GetQuestionResultsCellOutput]\n"
                    "class GetQuestionResultsOutput(ActionOutput):\n"
                    "    data: dict\n"
                    "@app.action()\n"
                    "def get_question_results(params: Params, soar: SOARClient, asset: BaseAsset) -> GetQuestionResultsOutput:\n"
                    "    response = {'data': {}}\n"
                    "    return GetQuestionResultsOutput.model_validate(response)\n"
                ),
                "src/client.py": (
                    "from time import sleep\n"
                    "def poll(timeout_seconds):\n"
                    "    sleep(timeout_seconds - 1)\n"
                ),
                "README.md": "action_result.data.*.data.result_sets.*.rows.*.data.*.text\n",
            },
            "base_files": {
                "legacy.json": (
                    '{"actions": ['
                    '{"identifier": "run_query", "action": "run query", "output": ['
                    '{"data_path": "action_result.summary.number_of_rows"},'
                    '{"data_path": "action_result.summary.timeout_seconds"}'
                    ']},'
                    '{"identifier": "get_question_results", "action": "get question results", "output": ['
                    '{"data_path": "action_result.data.*.data.result_sets.*.rows.*.data.*.*.text"},'
                    '{"data_path": "action_result.summary.number_of_rows"}'
                    ']}'
                    ']}'
                )
            },
            "ci": {"check_runs": [], "statuses": []},
        }

        titles = self.titles(review_input)

        self.assertIn("SDK migration changes a legacy nested output datapath", titles)
        self.assertIn("SDK migration drops legacy action summary fields", titles)
        self.assertIn("Large SDK migration has no checked-in regression tests", titles)
        self.assertIn("Branch-local exception import can raise UnboundLocalError", titles)
        self.assertIn("SDK migration coerces numeric action parameters without validation", titles)

    def test_sdk_pat_only_auth_with_oauth_fields_is_flagged(self):
        review_input = {
            "pr": {"title": "sdk migration", "body": "Auth supports OAuth credentials."},
            "changed_files": [
                {"filename": "github.json", "status": "removed", "patch": ""},
                {"filename": "pyproject.toml", "status": "added", "patch": ""},
                {"filename": "src/app.py", "status": "added", "patch": ""},
                {"filename": "src/client.py", "status": "added", "patch": ""},
                {"filename": "README.md", "status": "modified", "patch": ""},
            ],
            "full_files": {
                "pyproject.toml": "[tool.soar.app]\nmain_module = \"src.app:app\"\n",
                "src/app.py": (
                    "from soar_sdk.abstract import SOARClient\n"
                    "from soar_sdk.app import App\n"
                    "from soar_sdk.asset import AssetField, BaseAsset\n"
                    "from soar_sdk.exceptions import ActionFailure\n"
                    "class Asset(BaseAsset):\n"
                    "    personal_access_token: str | None = AssetField(sensitive=True)\n"
                    "    client_id: str | None = AssetField()\n"
                    "    client_secret: str | None = AssetField(sensitive=True)\n"
                    "app = App(name='GitHub', appid='12345678-1234-5678-9012-123456789012', app_type='information', logo='logo.svg', logo_dark='logo_dark.svg', product_vendor='Microsoft', product_name='GitHub', publisher='ExampleVendor', asset_cls=Asset)\n"
                    "@app.test_connectivity()\n"
                    "def test_connectivity(soar: SOARClient, asset: Asset) -> None:\n"
                    "    if not asset.personal_access_token:\n"
                    "        raise ActionFailure('missing PAT')\n"
                ),
                "src/client.py": (
                    "def resolve_auth(asset):\n"
                    "    if asset.personal_access_token:\n"
                    "        return BearerAuth(asset.personal_access_token)\n"
                    "    raise ActionFailure('missing credentials')\n"
                ),
                "README.md": "Configure either a Personal Access Token or OAuth Flow credentials.\n",
            },
            "base_files": {
                "github.json": '{"configuration": {"client_id": {}, "client_secret": {}, "personal_access_token": {}}, "actions": []}'
            },
            "ci": {"check_runs": [], "statuses": []},
        }

        self.assertIn("SDK auth path exposes non-PAT credential fields but only executes PAT auth", self.titles(review_input))

    def test_sdk_issue_number_int_cast_without_positive_validation_is_flagged(self):
        review_input = {
            "pr": {"title": "sdk migration", "body": ""},
            "changed_files": [
                {"filename": "github.json", "status": "removed", "patch": ""},
                {"filename": "pyproject.toml", "status": "added", "patch": ""},
                {"filename": "src/app.py", "status": "added", "patch": ""},
                {"filename": "src/actions/get_issue.py", "status": "added", "patch": ""},
            ],
            "full_files": {
                "pyproject.toml": "[tool.soar.app]\nmain_module = \"src.app:app\"\n",
                "src/app.py": (
                    "from soar_sdk.app import App\n"
                    "from soar_sdk.asset import BaseAsset\n"
                    "app = App(name='GitHub', appid='12345678-1234-5678-9012-123456789012', app_type='information', logo='logo.svg', logo_dark='logo_dark.svg', product_vendor='Microsoft', product_name='GitHub', publisher='ExampleVendor', asset_cls=BaseAsset)\n"
                    "@app.test_connectivity()\n"
                    "def test_connectivity() -> None:\n"
                    "    pass\n"
                ),
                "src/actions/get_issue.py": (
                    "from soar_sdk.params import Params\n"
                    "class GetIssueParams(Params):\n"
                    "    issue_number: float\n"
                    "def get_issue(params):\n"
                    "    endpoint = f'/issues/{int(params.issue_number)}'\n"
                    "    return endpoint\n"
                ),
            },
            "base_files": {
                "github.json": '{"actions": [{"identifier": "get_issue", "output": []}]}',
                "github_connector.py": "self._validate_integer(param['issue_number'], 'issue_number')\n",
            },
            "ci": {"check_runs": [], "statuses": []},
        }

        self.assertIn("SDK migration drops positive-integer validation for issue_number", self.titles(review_input))

    def test_sdk_make_request_dynamic_setattr_output_is_flagged(self):
        review_input = {
            "pr": {"title": "make request", "body": ""},
            "changed_files": [{"filename": "src/actions/make_req.py", "status": "added", "patch": ""}],
            "full_files": {
                "src/actions/make_req.py": (
                    "from soar_sdk.action_results import MakeRequestOutput\n"
                    "class GitHubMakeRequestOutput(MakeRequestOutput):\n"
                    "    @classmethod\n"
                    "    def from_response(cls, response):\n"
                    "        output = cls(status_code=200, response_body=response.text)\n"
                    "        json_body = response.json()\n"
                    "        for key, value in json_body.items():\n"
                    "            object.__setattr__(output, key, value)\n"
                    "        return output\n"
                    "@app.make_request()\n"
                    "def make_request(params) -> GitHubMakeRequestOutput:\n"
                    "    return GitHubMakeRequestOutput.from_response(call())\n"
                )
            },
            "ci": {"check_runs": [], "statuses": []},
        }

        self.assertIn("make request promotes JSON fields with object.__setattr__ that will not serialize", self.titles(review_input))

    def test_sdk_make_request_is_not_flagged_for_read_only_false(self):
        review_input = {
            "pr": {"title": "sdk migration", "body": ""},
            "changed_files": [
                {"filename": "legacy.json", "status": "removed", "patch": ""},
                {"filename": "pyproject.toml", "status": "added", "patch": ""},
                {"filename": "src/app.py", "status": "added", "patch": ""},
            ],
            "full_files": {
                "pyproject.toml": "[tool.soar.app]\nmain_module = \"src.app:app\"\n",
                "src/app.py": (
                    "from soar_sdk.app import App\n"
                    "from soar_sdk.asset import BaseAsset\n"
                    "from soar_sdk.params import MakeRequestParams\n"
                    "from soar_sdk.action_results import MakeRequestOutput\n"
                    "app = App(name='GitHub', appid='12345678-1234-5678-9012-123456789012', app_type='information', logo='logo.svg', logo_dark='logo_dark.svg', product_vendor='Microsoft', product_name='GitHub', publisher='ExampleVendor', asset_cls=BaseAsset)\n"
                    "@app.test_connectivity()\n"
                    "def test_connectivity() -> None:\n"
                    "    pass\n"
                    "@app.make_request()\n"
                    "def make_request(params: MakeRequestParams) -> MakeRequestOutput:\n"
                    "    return MakeRequestOutput(status_code=200, response_body='{}')\n"
                ),
            },
            "base_files": {
                "legacy.json": '{"actions": [{"identifier": "make_request", "output": []}]}',
            },
            "ci": {"check_runs": [], "statuses": []},
        }

        rendered = "\n".join(
            "\n".join(str(finding.get(key) or "") for key in ("title", "evidence", "suggested_fix"))
            for finding in run_deterministic_checks(review_input)
        )

        self.assertNotIn("read_only=False", rendered)

    def test_sdk_make_request_missing_focused_tests_is_flagged(self):
        review_input = {
            "pr": {"title": "sdk migration", "body": ""},
            "changed_files": [
                {"filename": "pyproject.toml", "status": "added", "patch": ""},
                {"filename": "src/app.py", "status": "added", "patch": ""},
                {"filename": "src/actions/make_req.py", "status": "added", "patch": ""},
                {"filename": "tests/test_actions.py", "status": "added", "patch": ""},
            ],
            "full_files": {
                "pyproject.toml": "[tool.soar.app]\nmain_module = \"src.app:app\"\n",
                "src/app.py": (
                    "from soar_sdk.app import App\n"
                    "from soar_sdk.asset import BaseAsset\n"
                    "app = App(name='GitHub', appid='12345678-1234-5678-9012-123456789012', app_type='information', logo='logo.svg', logo_dark='logo_dark.svg', product_vendor='Microsoft', product_name='GitHub', publisher='ExampleVendor', asset_cls=BaseAsset)\n"
                    "@app.test_connectivity()\n"
                    "def test_connectivity() -> None:\n"
                    "    pass\n"
                ),
                "src/actions/make_req.py": (
                    "from soar_sdk.params import MakeRequestParams\n"
                    "from soar_sdk.action_results import MakeRequestOutput\n"
                    "from src.app import app\n"
                    "@app.make_request()\n"
                    "def make_request(params: MakeRequestParams) -> MakeRequestOutput:\n"
                    "    return MakeRequestOutput(status_code=200, response_body='{}')\n"
                ),
                "tests/test_actions.py": "def test_get_issue():\n    assert True\n",
            },
            "ci": {"check_runs": [], "statuses": []},
        }

        self.assertIn("`make request` action has no focused test coverage", self.titles(review_input))

    def test_sdk_removed_manifest_output_schema_contraction_is_flagged(self):
        legacy_outputs = ",".join(
            f'{{"data_path": "action_result.data.*.data.requests.*.field_{index}"}}'
            for index in range(80)
        )
        review_input = {
            "pr": {"title": "sdk migration", "body": "Preserves existing outputs."},
            "changed_files": [
                {"filename": "search_api.json", "status": "removed", "patch": ""},
                {"filename": "pyproject.toml", "status": "added", "patch": ""},
                {"filename": "src/app.py", "status": "added", "patch": ""},
                {"filename": "src/outputs.py", "status": "added", "patch": ""},
            ],
            "full_files": {
                "search_api.json": (
                    '{"actions": [{"identifier": "get_report", "output": ['
                    + legacy_outputs
                    + "]}]}"
                ),
                "pyproject.toml": "[tool.soar.app]\nmain_module = \"src.app:app\"\n",
                "src/app.py": (
                    "from soar_sdk.app import App\n"
                    "from soar_sdk.asset import BaseAsset\n"
                    "from src.outputs import ReportActionOutput, ReportSummary\n"
                    "app = App(name='Search API', appid='12345678-1234-5678-9012-123456789012', app_type='information', logo='logo.svg', logo_dark='logo_dark.svg', product_vendor='search_api', product_name='search_api', publisher='ExampleVendor', asset_cls=BaseAsset)\n"
                    "@app.test_connectivity()\n"
                    "def test_connectivity() -> None:\n"
                    "    pass\n"
                    "def run_get_report(params, soar, asset) -> ReportActionOutput:\n"
                    "    return ReportActionOutput()\n"
                    "app.register_action(run_get_report, identifier='get_report', output_class=ReportActionOutput, summary_type=ReportSummary)\n"
                ),
                "src/outputs.py": (
                    "from soar_sdk.action_results import ActionOutput, OutputField\n"
                    "class ReportSummary(ActionOutput):\n"
                    "    total: int = OutputField(example_values=[1])\n"
                    "class ReportActionOutput(ActionOutput):\n"
                    "    page: dict | None\n"
                    "    task: dict | None\n"
                    "    stats: dict | None\n"
                ),
            },
            "base_files": {},
            "ci": {"check_runs": [], "statuses": []},
        }

        self.assertIn("SDK migration contracts a large legacy action output schema", self.titles(review_input))

    def test_sdk_output_schema_contraction_counts_inherited_output_fields(self):
        legacy_outputs = ",".join(
            f'{{"data_path": "action_result.data.*.field_{index}"}}'
            for index in range(80)
        )
        review_input = {
            "pr": {"title": "sdk migration", "body": ""},
            "changed_files": [
                {"filename": "legacy.json", "status": "removed", "patch": ""},
                {"filename": "pyproject.toml", "status": "added", "patch": ""},
                {"filename": "src/app.py", "status": "added", "patch": ""},
                {"filename": "src/outputs.py", "status": "added", "patch": ""},
            ],
            "full_files": {
                "legacy.json": (
                    '{"actions": [{"identifier": "detonate_url", "output": ['
                    + legacy_outputs
                    + "]}]}"
                ),
                "pyproject.toml": "[tool.soar.app]\nmain_module = \"src.app:app\"\n",
                "src/app.py": (
                    "from soar_sdk.app import App\n"
                    "from soar_sdk.asset import BaseAsset\n"
                    "from src.outputs import DetonateActionOutput\n"
                    "app = App(name='Search API', appid='12345678-1234-5678-9012-123456789012', app_type='information', logo='logo.svg', logo_dark='logo_dark.svg', product_vendor='search_api', product_name='search_api', publisher='ExampleVendor', asset_cls=BaseAsset)\n"
                    "@app.test_connectivity()\n"
                    "def test_connectivity() -> None:\n"
                    "    pass\n"
                    "def run_detonate_url(params, soar, asset) -> DetonateActionOutput:\n"
                    "    return DetonateActionOutput()\n"
                    "app.register_action(run_detonate_url, identifier='detonate_url', output_class=DetonateActionOutput)\n"
                ),
                "src/outputs.py": (
                    "from soar_sdk.action_results import ActionOutput\n"
                    "class BaseReportOutput(ActionOutput):\n"
                    + "".join(f"    field_{index}: str | None\n" for index in range(80))
                    + "class DetonateActionOutput(BaseReportOutput):\n"
                    "    message: str | None\n"
                ),
            },
            "base_files": {},
            "ci": {"check_runs": [], "statuses": []},
        }

        self.assertNotIn("SDK migration contracts a large legacy action output schema", self.titles(review_input))

    def test_raw_search_query_user_input_interpolation_is_flagged(self):
        review_input = {
            "pr": {"title": "search action", "body": ""},
            "changed_files": [{"filename": "src/actions/lookup.py", "status": "modified", "patch": ""}],
            "full_files": {
                "src/actions/lookup.py": (
                    "def run_hunt_domain(params, client):\n"
                    "    endpoint = f'/api/v1/search/?q=domain:{params.domain}'\n"
                    "    return client.request(endpoint)\n"
                )
            },
            "ci": {"check_runs": [], "statuses": []},
        }

        self.assertIn("Search action interpolates raw user input into query syntax", self.titles(review_input))

    def test_escaped_search_query_params_are_not_flagged(self):
        review_input = {
            "pr": {"title": "search action", "body": ""},
            "changed_files": [{"filename": "src/actions/lookup.py", "status": "modified", "patch": ""}],
            "full_files": {
                "src/actions/lookup.py": (
                    "def _escape_search_value(value):\n"
                    "    return value.replace(':', '\\\\:')\n"
                    "def run_hunt_domain(params, client):\n"
                    "    query = f'domain:{_escape_search_value(params.domain)}'\n"
                    "    return client.request('/api/v1/search/', params={'q': query})\n"
                )
            },
            "ci": {"check_runs": [], "statuses": []},
        }

        self.assertNotIn("Search action interpolates raw user input into query syntax", self.titles(review_input))

    def test_search_output_missing_sort_and_id_is_flagged(self):
        review_input = {
            "pr": {"title": "sdk migration", "body": ""},
            "changed_files": [{"filename": "src/outputs.py", "status": "added", "patch": ""}],
            "full_files": {
                "src/constants.py": "SEARCH_API_SEARCH_ENDPOINT = '/api/v1/search/'\n",
                "src/outputs.py": (
                    "from soar_sdk.action_results import ActionOutput\n"
                    "class SearchResultItemOutput(ActionOutput):\n"
                    "    page: dict | None\n"
                    "    task: dict | None\n"
                    "class LookupActionOutput(ActionOutput):\n"
                    "    results: list[SearchResultItemOutput] | None\n"
                ),
            },
            "ci": {"check_runs": [], "statuses": []},
        }

        self.assertIn("Search result output model omits pagination identity fields", self.titles(review_input))

    def test_search_api_client_without_rate_limit_handling_is_flagged(self):
        review_input = {
            "pr": {"title": "Search API client", "body": ""},
            "changed_files": [{"filename": "src/client.py", "status": "modified", "patch": ""}],
            "full_files": {
                "src/client.py": (
                    "import httpx\n"
                    "SEARCH_API_BASE_URL = 'https://api.example.invalid/search'\n"
                    "def process_response(response):\n"
                    "    if response.status_code >= 400:\n"
                    "        return f'Error from server: {response.text}'\n"
                )
            },
            "ci": {"check_runs": [], "statuses": []},
        }

        self.assertIn("Search API client does not surface rate-limit responses distinctly", self.titles(review_input))

    def test_graph_target_user_id_without_strip_is_flagged(self):
        review_input = {
            "pr": {"title": "sdk migration", "body": ""},
            "changed_files": [{"filename": "src/actions/create_folder.py", "status": "added", "patch": ""}],
            "full_files": {
                "src/actions/create_folder.py": (
                    "def create_folder(asset):\n"
                    "    # Microsoft Graph Client Credentials route\n"
                    "    target_user_id = asset.target_user_id\n"
                    "    endpoint = f'/users/{target_user_id}/drive/root/children'\n"
                    "    return endpoint\n"
                )
            },
            "ci": {"check_runs": [], "statuses": []},
        }

        self.assertIn(
            "Client Credentials target_user_id is used in a Graph URL without normalization",
            self.titles(review_input),
        )

    def test_graph_target_user_id_with_strip_is_not_flagged(self):
        review_input = {
            "pr": {"title": "sdk migration", "body": ""},
            "changed_files": [{"filename": "src/actions/create_folder.py", "status": "added", "patch": ""}],
            "full_files": {
                "src/actions/create_folder.py": (
                    "def create_folder(asset):\n"
                    "    # Microsoft Graph Client Credentials route\n"
                    "    target_user_id = (asset.target_user_id or '').strip()\n"
                    "    endpoint = f'/users/{target_user_id}/drive/root/children'\n"
                    "    return endpoint\n"
                )
            },
            "ci": {"check_runs": [], "statuses": []},
        }

        self.assertNotIn(
            "Client Credentials target_user_id is used in a Graph URL without normalization",
            self.titles(review_input),
        )

    def test_display_name_file_path_cef_type_is_flagged(self):
        review_input = {
            "pr": {"title": "sdk migration", "body": ""},
            "changed_files": [{"filename": "src/actions/create_folder.py", "status": "added", "patch": ""}],
            "full_files": {
                "src/actions/create_folder.py": (
                    "from soar_sdk.action_results import ActionOutput, OutputField\n"
                    "class UserOutput(ActionOutput):\n"
                    "    displayName: str | None = OutputField(cef_types=['file path'])\n"
                )
            },
            "ci": {"check_runs": [], "statuses": []},
        }

        self.assertIn("Output CEF type marks a display name as a file path", self.titles(review_input))

    def test_stale_lookup_helper_comment_is_flagged(self):
        review_input = {
            "pr": {"title": "Search API sdk migration", "body": ""},
            "changed_files": [{"filename": "src/constants.py", "status": "modified", "patch": ""}],
            "full_files": {
                "src/constants.py": (
                    "# URL interpolation is done by behavior._build_lookup_url() and percent-encodes user input.\n"
                    "LOOKUP_DOMAIN_ENDPOINT = '/api/v1/search/'\n"
                ),
                "src/behavior.py": "def lookup_domain():\n    pass\n",
            },
            "ci": {"check_runs": [], "statuses": []},
        }

        self.assertIn("Comment references a missing lookup URL helper", self.titles(review_input))

    def test_sdk_tests_import_package_relative_app_as_top_level_is_flagged(self):
        review_input = {
            "pr": {"title": "sdk migration", "body": ""},
            "changed_files": [
                {"filename": "pyproject.toml", "status": "added", "patch": ""},
                {"filename": "src/app.py", "status": "added", "patch": ""},
                {"filename": "tests/conftest.py", "status": "added", "patch": ""},
                {"filename": "tests/test_actions.py", "status": "added", "patch": ""},
            ],
            "full_files": {
                "pyproject.toml": "[tool.soar.app]\nmain_module = \"src.app:app\"\n",
                "src/app.py": (
                    "from soar_sdk.app import App\n"
                    "from soar_sdk.asset import BaseAsset\n"
                    "from .client import call_github\n"
                    "app = App(name='GitHub', appid='12345678-1234-5678-9012-123456789012', app_type='information', logo='logo.svg', logo_dark='logo_dark.svg', product_vendor='Microsoft', product_name='GitHub', publisher='ExampleVendor', asset_cls=BaseAsset)\n"
                ),
                "tests/conftest.py": (
                    "import sys\n"
                    "from pathlib import Path\n"
                    "sys.path.insert(0, str(Path(__file__).parents[1] / 'src'))\n"
                ),
                "tests/test_actions.py": (
                    "from app import app\n"
                    "def test_app():\n"
                    "    assert app\n"
                ),
            },
            "ci": {"check_runs": [], "statuses": []},
        }

        self.assertIn("SDK tests import src/app.py as a top-level module while the app uses package-relative imports", self.titles(review_input))

    def test_verify_server_cert_read_without_json_config_is_flagged(self):
        review_input = {
            "pr": {"title": "legacy connector", "body": ""},
            "changed_files": [{"filename": "connector.py"}, {"filename": "sample.json"}],
            "full_files": {
                "connector.py": (
                    "def initialize(self):\n"
                    "    config = self.get_config()\n"
                    "    self._verify = config.get('verify_server_cert', False)\n"
                ),
                "sample.json": (
                    '{"configuration": {"api_key": {"data_type": "password"}}, "actions": []}'
                ),
            },
            "ci": {"check_runs": [], "statuses": []},
        }

        self.assertIn(
            "Python reads verify_server_cert but app JSON does not expose the configuration field",
            self.titles(review_input),
        )

    def test_python_config_reads_missing_json_fields_are_flagged(self):
        review_input = {
            "pr": {"title": "legacy connector", "body": ""},
            "changed_files": [{"filename": "connector.py"}, {"filename": "sample.json"}],
            "full_files": {
                "connector.py": (
                    "def initialize(self):\n"
                    "    config = self.get_config()\n"
                    "    self._api_key = config.get('api_key')\n"
                    "    self._container_label = config.get('container_label', 'events')\n"
                    "    self._feed_limit = config['feed_limit']\n"
                ),
                "sample.json": (
                    '{"configuration": {"api_key": {"data_type": "password"}}, "actions": []}'
                ),
            },
            "ci": {"check_runs": [], "statuses": []},
        }

        findings = run_deterministic_checks(review_input)

        self.assertIn("Python reads asset configuration fields missing from app JSON", {item["title"] for item in findings})
        evidence = " ".join(item["evidence"] for item in findings)
        self.assertIn("container_label", evidence)
        self.assertIn("feed_limit", evidence)

    def test_connector_helper_calls_without_timeouts_are_flagged(self):
        review_input = {
            "pr": {"title": "timeouts", "body": ""},
            "changed_files": [{"filename": "connector.py"}],
            "full_files": {
                "connector.py": (
                    "def _make_rest_call(self, endpoint, action_result):\n"
                    "    return request_func(url, headers=headers, verify=self._verify)\n"
                    "def _lookup_container(self, url):\n"
                    "    return self._get_requests_session().get(url, verify=False)\n"
                )
            },
            "ci": {"check_runs": [], "statuses": []},
        }

        titles = self.titles(review_input)

        self.assertIn("Direct requests call does not set a timeout", titles)

    def test_response_text_and_headers_in_debug_data_are_flagged(self):
        review_input = {
            "pr": {"title": "debug", "body": ""},
            "changed_files": [{"filename": "connector.py"}],
            "full_files": {
                "connector.py": (
                    "def _make_rest_call(self, response, action_result):\n"
                    "    action_result.add_debug_data({'r_text': response.text})\n"
                    "    action_result.add_debug_data({'r_headers': response.headers})\n"
                )
            },
            "ci": {"check_runs": [], "statuses": []},
        }

        self.assertIn("Debug data records full response text or headers", self.titles(review_input))

    def test_indicator_params_without_contains_are_flagged_for_dict_parameters(self):
        review_input = {
            "pr": {"title": "metadata", "body": ""},
            "changed_files": [{"filename": "sample.json"}],
            "full_files": {
                "sample.json": (
                    '{"actions": [{"identifier": "enrich_sha256", "action": "enrich sha256", '
                    '"parameters": {"Hash": {"data_type": "string", "required": true}}, "output": []}, '
                    '{"identifier": "enrich_ipv4", "action": "enrich ipv4", '
                    '"parameters": {"IP": {"data_type": "string", "required": true}}, "output": []}]}'
                )
            },
            "ci": {"check_runs": [], "statuses": []},
        }

        self.assertIn("Indicator action parameters are missing contains metadata", self.titles(review_input))

    def test_custom_view_action_result_misuse_is_flagged(self):
        review_input = {
            "pr": {"title": "views", "body": ""},
            "changed_files": [{"filename": "sample_view.py"}],
            "full_files": {
                "sample_view.py": (
                    "import phantom.app as phantom\n"
                    "def _render_enrichment_view(request):\n"
                    "    action_result = connector.add_action_result(phantom.action_result.ActionResult(dict()))\n"
                    "    handler(param)\n"
                    "    return render(request, 'view.html', {'data': action_result.get_data()})\n"
                )
            },
            "ci": {"check_runs": [], "statuses": []},
        }

        titles = self.titles(review_input)

        self.assertIn("Custom view references phantom.action_result without importing it", titles)
        self.assertIn("Custom view renders a different ActionResult than the handler populates", titles)

    def test_custom_view_recalls_api_from_get_parameters_is_flagged(self):
        review_input = {
            "pr": {"title": "views", "body": ""},
            "changed_files": [{"filename": "sample_view.py"}],
            "full_files": {
                "sample_view.py": (
                    "from django.http import HttpResponse\n"
                    "def enrich_domain_view(request):\n"
                    "    domain = request.GET.get('domain')\n"
                    "    connector = CyberintConnector()\n"
                    "    connector._handle_enrich_domain({'domain': domain})\n"
                    "    return HttpResponse('ok')\n"
                )
            },
            "ci": {"check_runs": [], "statuses": []},
        }

        self.assertIn("Custom view re-calls API with unchecked GET parameters", self.titles(review_input))

    def test_polling_container_and_artifact_contract_issues_are_flagged(self):
        review_input = {
            "pr": {"title": "poll", "body": ""},
            "changed_files": [{"filename": "connector.py"}],
            "full_files": {
                "connector.py": (
                    "def _handle_on_poll(self, param):\n"
                    "    container = {'name': 'IOC feed', 'source_data_identifier': today}\n"
                    "    self.save_container(container)\n"
                    "    offset = 0\n"
                    "    limit = 10000\n"
                    "    while True:\n"
                    "        iocs = self._fetch_iocs(offset=offset, limit=limit)\n"
                    "        for ioc in iocs:\n"
                    "            indicator_type = ioc['type']\n"
                    "            indicator_value = ioc['value']\n"
                    "            artifact = {\n"
                    "                'name': indicator_value,\n"
                    "                'cef': {indicator_type: indicator_value},\n"
                    "                'source_data_identifier': indicator_value,\n"
                    "            }\n"
                    "            self.save_artifact(artifact)\n"
                    "        if len(iocs) < limit:\n"
                    "            break\n"
                    "        offset += limit\n"
                )
            },
            "ci": {"check_runs": [], "statuses": []},
        }

        titles = self.titles(review_input)

        self.assertIn("on_poll creates containers without a label", titles)
        self.assertIn("on_poll ignores standard SOAR poll controls", titles)
        self.assertIn("on_poll ignores save_artifact return values", titles)
        self.assertIn("Polling artifacts use dynamic CEF keys", titles)
        self.assertIn("Polling artifacts are missing artifact labels", titles)
        self.assertIn("Polling or dashboard pagination has no maximum page guard", titles)

    def test_polling_loaded_state_not_used_as_checkpoint_is_flagged(self):
        review_input = {
            "pr": {"title": "poll", "body": ""},
            "changed_files": [{"filename": "connector.py"}],
            "full_files": {
                "connector.py": (
                    "def initialize(self):\n"
                    "    self._state = self.load_state()\n"
                    "def finalize(self):\n"
                    "    self.save_state(self._state)\n"
                    "def _handle_on_poll(self, param):\n"
                    "    today = datetime.now(timezone.utc).date()\n"
                    "    offset = 0\n"
                    "    while True:\n"
                    "        iocs = self._fetch_iocs(today, offset=offset)\n"
                    "        self.save_artifact({'cef': {'domain': 'example.com'}})\n"
                    "        if len(iocs) < 100:\n"
                    "            break\n"
                    "        offset += 100\n"
                )
            },
            "ci": {"check_runs": [], "statuses": []},
        }

        self.assertIn("Polling state is loaded but not used as an ingestion checkpoint", self.titles(review_input))

    def test_dashboard_date_filter_not_wired_is_flagged(self):
        review_input = {
            "pr": {"title": "dashboard", "body": "Adds a custom dashboard with date filtering."},
            "changed_files": [
                {
                    "filename": "default/data/ui/dashboards/ioc_view.xml",
                    "status": "added",
                    "patch": (
                        "@@ -0,0 +1,7 @@\n"
                        "+<dashboard>\n"
                        "+  <row><panel><html id=\"ioc_view\">\n"
                        "+\n"
                        "+  </html></panel></row>\n"
                        "+</dashboard>\n"
                    ),
                },
                {
                    "filename": "ioc_view.html",
                    "status": "added",
                    "patch": (
                        "@@ -0,0 +1,2 @@\n"
                        "+<input type=\"date\" name=\"date\" />\n"
                        "+<table></table>\n"
                    ),
                },
                {"filename": "sample_view.py", "status": "added"},
            ],
            "full_files": {
                "sample_view.py": (
                    "def ioc_view(request):\n"
                    "    today = datetime.now(timezone.utc).date()\n"
                    "    start = datetime.combine(today, time.min)\n"
                    "    return HttpResponse(str(start))\n"
                )
            },
            "ci": {"check_runs": [], "statuses": []},
        }

        self.assertIn("Dashboard/date filter UI is not wired to the custom view", self.titles(review_input))

    def test_legacy_connector_runtime_edge_cases_are_flagged(self):
        review_input = {
            "pr": {"title": "runtime", "body": ""},
            "changed_files": [{"filename": "connector.py"}],
            "full_files": {
                "connector.py": (
                    "def _parse_jsonl(self, text):\n"
                    "    out = []\n"
                    "    for line in text.splitlines():\n"
                    "        try:\n"
                    "            out.append(json.loads(line))\n"
                    "        except ValueError:\n"
                    "            continue\n"
                    "    return out\n"
                    "def _handle_on_poll(self, param):\n"
                    "    if data.get('count', 0) > 0:\n"
                    "        container_id = data['data'][0]['id']\n"
                    "def handle_action(self, param):\n"
                    "    ret_val = phantom.APP_SUCCESS\n"
                    "    action_id = self.get_action_identifier()\n"
                    "    if action_id in action_mapping:\n"
                    "        ret_val = action_mapping[action_id](param)\n"
                    "    return ret_val\n"
                )
            },
            "ci": {"check_runs": [], "statuses": []},
        }

        titles = self.titles(review_input)

        self.assertIn("JSONL parser silently drops malformed lines", titles)
        self.assertIn("Container lookup indexes response data after only checking count", titles)
        self.assertIn("Unknown action identifiers return success silently", titles)

    def test_rate_limit_and_quota_connectivity_patterns_are_flagged(self):
        review_input = {
            "pr": {"title": "api", "body": ""},
            "changed_files": [{"filename": "connector.py"}],
            "full_files": {
                "connector.py": (
                    "def _make_rest_call(self, endpoint, action_result):\n"
                    "    response = requests.get(endpoint, timeout=30)\n"
                    "    if response.status_code >= 400:\n"
                    "        return action_result.set_status(phantom.APP_ERROR, response.text)\n"
                    "def _handle_test_connectivity(self, param):\n"
                    "    return self._enrich_indicator(param, 'domain', 'cyberint.com')\n"
                )
            },
            "ci": {"check_runs": [], "statuses": []},
        }

        titles = self.titles(review_input)

        self.assertIn("HTTP 429 rate limits are handled as generic API errors", titles)
        self.assertIn("Test connectivity consumes a real action endpoint with a fixed indicator", titles)


if __name__ == "__main__":
    unittest.main()
