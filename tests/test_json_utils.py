import unittest

from agentic_pr_review.json_utils import extract_json_object


class JsonUtilsTest(unittest.TestCase):
    def test_extract_json_object_recovers_fenced_json_with_trailing_commas(self):
        value = extract_json_object(
            "Here is the result:\n"
            "```json\n"
            "{\n"
            '  "summary": "ok",\n'
            '  "findings": [],\n'
            "}\n"
            "```\n"
        )

        self.assertEqual(value, {"summary": "ok", "findings": []})

    def test_extract_json_object_recovers_prose_wrapped_json(self):
        value = extract_json_object('Review complete: {"summary": "ok", "findings": []} Thanks.')

        self.assertEqual(value["summary"], "ok")


if __name__ == "__main__":
    unittest.main()
