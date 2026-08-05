"""Human-readable progress reporting for long-running reviews."""

from __future__ import annotations

from datetime import datetime, timezone
import sys
import time
from typing import Callable, TextIO


def format_duration(seconds: float) -> str:
    """Format a duration as HH:MM:SS, rounding down partial seconds."""
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class AdaptiveETA:
    """Estimate whole-run time remaining from completed model requests."""

    def __init__(
        self,
        total_requests: int,
        *,
        initial_request_seconds: float = 120,
        final_processing_seconds: float = 15,
    ) -> None:
        self.total_requests = max(0, total_requests)
        self.initial_request_seconds = max(1, initial_request_seconds)
        self.final_processing_seconds = max(0, final_processing_seconds)
        self.completed_request_seconds: list[float] = []

    def complete_request(self, duration_seconds: float) -> None:
        self.completed_request_seconds.append(max(0, duration_seconds))

    def add_requests(self, count: int) -> None:
        self.total_requests += max(0, int(count))

    def skip_requests(self, count: int) -> None:
        self.total_requests = max(len(self.completed_request_seconds), self.total_requests - max(0, int(count)))

    @property
    def estimated_request_seconds(self) -> float:
        if not self.completed_request_seconds:
            return self.initial_request_seconds
        return sum(self.completed_request_seconds) / len(self.completed_request_seconds)

    def remaining_seconds(self, *, current_request_elapsed: float | None = None) -> float:
        remaining_requests = max(0, self.total_requests - len(self.completed_request_seconds))
        request_estimate = self.estimated_request_seconds
        if current_request_elapsed is None or remaining_requests == 0:
            model_seconds = remaining_requests * request_estimate
        else:
            minimum_current_remaining = min(30.0, request_estimate * 0.25)
            current_remaining = max(request_estimate - current_request_elapsed, minimum_current_remaining)
            model_seconds = current_remaining + max(0, remaining_requests - 1) * request_estimate
        return model_seconds + self.final_processing_seconds


class ProgressReporter:
    """Emit timestamped, immediately flushed progress messages to stderr."""

    def __init__(
        self,
        *,
        stream: TextIO | None = None,
        wall_clock: Callable[[], datetime] | None = None,
        monotonic: Callable[[], float] | None = None,
    ) -> None:
        self.stream = stream or sys.stderr
        self.wall_clock = wall_clock or (lambda: datetime.now(timezone.utc))
        self.monotonic = monotonic or time.monotonic
        self.started_at = self.monotonic()

    def __call__(self, message: str) -> None:
        now = self.wall_clock().astimezone(timezone.utc)
        timestamp = now.isoformat(timespec="seconds").replace("+00:00", "Z")
        elapsed = format_duration(self.monotonic() - self.started_at)
        print(f"[{timestamp}] [+{elapsed}] {message}", file=self.stream, flush=True)
