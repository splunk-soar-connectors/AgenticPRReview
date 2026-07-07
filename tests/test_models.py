import unittest

from agentic_pr_review.models import normalize_finding, normalize_review_output


class ModelsTest(unittest.TestCase):
    def test_normalize_finding_preserves_optional_suggested_code(self):
        finding = normalize_finding(
            {
                "id": "f1",
                "title": "Wrong value",
                "suggested_code": "\nnew = validated_value\n",
            },
            default_source="claude",
        )

        self.assertEqual(finding["suggested_code"], "new = validated_value")

    def test_normalize_finding_omits_blank_suggested_code(self):
        finding = normalize_finding(
            {
                "id": "f1",
                "title": "Wrong value",
                "suggested_code": "   ",
            },
            default_source="claude",
        )

        self.assertIsNone(finding["suggested_code"])

    def test_review_output_drops_finding_without_file_evidence(self):
        output = normalize_review_output(
            {
                "overall_status": "needs_review",
                "findings": [
                    {
                        "id": "f1",
                        "title": "Generic missing tests",
                        "category": "missing_tests",
                        "confidence": "high",
                        "file": None,
                        "line": None,
                        "evidence": "Risky behavior changed without tests.",
                        "why_it_matters": "Untested connector changes can regress runtime behavior.",
                        "suggested_fix": "Add tests.",
                    }
                ],
            },
            deterministic_findings=[],
        )

        self.assertEqual(output["findings"], [])
        self.assertEqual(output["overall_status"], "looks_good")

    def test_review_output_keeps_file_backed_code_reference(self):
        output = normalize_review_output(
            {
                "overall_status": "needs_review",
                "findings": [
                    {
                        "id": "f1",
                        "title": "Output schema mismatch",
                        "category": "output_schema_mismatch",
                        "confidence": "high",
                        "file": "sample.json",
                        "line": None,
                        "code_reference": "action_result.summary.risk_score",
                        "evidence": "`risk_score` is emitted by connector.py but sample.json does not declare action_result.summary.risk_score.",
                        "why_it_matters": "SOAR playbooks cannot rely on undeclared output paths.",
                        "suggested_fix": "Declare action_result.summary.risk_score in sample.json or stop emitting it.",
                    }
                ],
            },
            deterministic_findings=[],
        )

        self.assertEqual(len(output["findings"]), 1)
        self.assertEqual(output["findings"][0]["code_reference"], "action_result.summary.risk_score")

    def test_review_output_does_not_block_publish_when_actionable_findings_exist(self):
        output = normalize_review_output(
            {
                "overall_status": "needs_review",
                "safe_to_publish": False,
                "findings": [
                    {
                        "id": "f1",
                        "title": "Output schema mismatch",
                        "category": "output_schema_mismatch",
                        "confidence": "high",
                        "file": "src/app.py",
                        "line": 20,
                        "evidence": "The SDK model drops action_result.summary.timeout_seconds.",
                        "why_it_matters": "Existing playbooks can stop resolving the summary datapath.",
                        "suggested_fix": "Restore the summary field or document the compatibility break.",
                    }
                ],
            },
            deterministic_findings=[],
        )

        self.assertTrue(output["safe_to_publish"])

    def test_review_output_drops_speculative_verify_finding(self):
        output = normalize_review_output(
            {
                "overall_status": "needs_review",
                "findings": [
                    {
                        "id": "f1",
                        "title": "Verify timeout behavior",
                        "category": "api_auth_correctness",
                        "confidence": "high",
                        "file": "connector.py",
                        "line": 10,
                        "evidence": "It is unclear whether the timeout is used.",
                        "why_it_matters": "Missing timeouts can hang actions.",
                        "suggested_fix": "Verify that the timeout is passed.",
                    }
                ],
            },
            deterministic_findings=[],
        )

        self.assertEqual(output["findings"], [])

    def test_review_output_promotes_high_confidence_deterministic_findings(self):
        output = normalize_review_output(
            {
                "overall_status": "looks_good",
                "findings": [],
                "model_notes": "model omitted the static finding",
            },
            deterministic_findings=[
                {
                    "id": "det-1",
                    "title": "TLS certificate verification is explicitly disabled",
                    "category": "api_auth_correctness",
                    "severity": "high",
                    "confidence": "high",
                    "file": "connector.py",
                    "line": 20,
                    "evidence": "`requests.get(url, verify=False)` disables TLS verification.",
                    "why_it_matters": "External API traffic can be intercepted.",
                    "suggested_fix": "Use a configurable verify_server_cert asset setting and default it to true.",
                    "source": "deterministic",
                }
            ],
        )

        self.assertEqual(output["overall_status"], "needs_review")
        self.assertEqual(len(output["findings"]), 1)
        self.assertEqual(output["findings"][0]["source"], "deterministic")

    def test_review_output_does_not_promote_low_confidence_deterministic_findings(self):
        output = normalize_review_output(
            {
                "overall_status": "looks_good",
                "findings": [],
            },
            deterministic_findings=[
                {
                    "id": "det-1",
                    "title": "Possible pagination issue",
                    "category": "pagination",
                    "severity": "medium",
                    "confidence": "low",
                    "file": "connector.py",
                    "line": 20,
                    "evidence": "`list_items` has one request.",
                    "why_it_matters": "List actions can truncate results.",
                    "suggested_fix": "Add pagination if the API supports it.",
                }
            ],
        )

        self.assertEqual(output["findings"], [])


if __name__ == "__main__":
    unittest.main()
