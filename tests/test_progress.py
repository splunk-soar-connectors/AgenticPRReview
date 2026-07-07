from datetime import datetime, timezone
from io import StringIO
import unittest
from unittest.mock import patch

from agentic_pr_review.gateway_claude import GatewayClaudeReviewer
from agentic_pr_review.progress import AdaptiveETA, format_duration, ProgressReporter


class ProgressTest(unittest.TestCase):
    def test_format_duration(self):
        self.assertEqual(format_duration(0), "00:00:00")
        self.assertEqual(format_duration(3661.9), "01:01:01")

    def test_reporter_includes_utc_timestamp_and_total_elapsed_time(self):
        stream = StringIO()
        monotonic_values = iter([100.0, 161.0])
        reporter = ProgressReporter(
            stream=stream,
            wall_clock=lambda: datetime(2026, 6, 26, 18, 30, 45, tzinfo=timezone.utc),
            monotonic=lambda: next(monotonic_values),
        )

        reporter("Deep-review chunk 2/5 started.")

        self.assertEqual(
            stream.getvalue(),
            "[2026-06-26T18:30:45Z] [+00:01:01] Deep-review chunk 2/5 started.\n",
        )

    def test_adaptive_eta_uses_completed_request_average_for_whole_run(self):
        eta = AdaptiveETA(total_requests=4, initial_request_seconds=120, final_processing_seconds=15)

        self.assertEqual(eta.remaining_seconds(), 495)

        eta.complete_request(60)

        self.assertEqual(eta.remaining_seconds(), 195)
        self.assertEqual(eta.remaining_seconds(current_request_elapsed=20), 175)

    def test_gateway_heartbeat_reports_request_elapsed_time(self):
        class StopAfterFirstHeartbeat:
            def __init__(self):
                self.calls = 0

            def wait(self, timeout):
                self.calls += 1
                self.timeout = timeout
                return self.calls > 1

        messages = []
        reviewer = object.__new__(GatewayClaudeReviewer)
        reviewer.progress = messages.append
        reviewer.eta = AdaptiveETA(total_requests=2, initial_request_seconds=120, final_processing_seconds=15)
        stop = StopAfterFirstHeartbeat()

        with patch("agentic_pr_review.gateway_claude.time.monotonic", return_value=161.0):
            reviewer._report_gateway_heartbeat(stop, request_started=100.0)

        self.assertEqual(stop.timeout, 60)
        self.assertEqual(
            messages,
            [
                "Gateway request still running (00:01:01 elapsed for this request); "
                "estimated whole-review time remaining: ~00:03:14."
            ],
        )


if __name__ == "__main__":
    unittest.main()
