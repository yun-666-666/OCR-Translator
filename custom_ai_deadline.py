"""Safe absolute deadline values for Custom AI requests."""

from dataclasses import dataclass
import math
import time


class CustomAIRequestDeadlineExceeded(TimeoutError):
    """Safe terminal error: no Custom AI request budget remains."""


def _finite(value):
    try:
        normalized = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return normalized if math.isfinite(normalized) else None


def _positive_finite(value):
    normalized = _finite(value)
    return normalized if normalized is not None and normalized > 0.0 else None


@dataclass(frozen=True)
class CustomAIRequestDeadline:
    """One immutable monotonic expiry shared by a Custom AI request."""

    deadline_monotonic: float

    def __post_init__(self):
        deadline = _finite(self.deadline_monotonic)
        if deadline is None:
            raise ValueError("Custom AI request deadline must be finite")
        object.__setattr__(self, "deadline_monotonic", deadline)

    @classmethod
    def from_timeout(cls, timeout_seconds, now=None, fallback_seconds=30.0):
        started_at = _finite(now)
        if started_at is None:
            started_at = time.monotonic()
        budget = _positive_finite(timeout_seconds)
        if budget is None:
            budget = _positive_finite(fallback_seconds)
        if budget is None:
            raise ValueError("Custom AI request timeout must be positive")
        return cls(started_at + budget)

    @classmethod
    def from_absolute(cls, deadline_monotonic):
        absolute = _finite(deadline_monotonic)
        if absolute is None:
            raise ValueError("Custom AI request deadline must be finite")
        return cls(absolute)

    def remaining_seconds(self, now=None):
        current = _finite(now)
        if current is None:
            current = time.monotonic()
        return max(0.0, self.deadline_monotonic - current)

    def http_timeout(self, now=None):
        remaining = self.remaining_seconds(now)
        if remaining <= 0.0:
            raise CustomAIRequestDeadlineExceeded(
                "Custom AI request deadline expired"
            )
        return remaining
