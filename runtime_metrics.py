import math
import re
import threading
import time
from collections import defaultdict, deque


_AUTH_HEADER_RE = re.compile(
    r"authorization\s*:\s*(?:bearer|basic)?\s*[^\s,;]+",
    re.IGNORECASE,
)
_BEARER_RE = re.compile(r"\bbearer\s+[A-Za-z0-9._~+/=-]+", re.IGNORECASE)
_COMMON_SECRET_RE = re.compile(
    r"\b(?:sk|rk|pk|api|key|token)-[A-Za-z0-9._-]{6,}\b",
    re.IGNORECASE,
)
_LONG_TOKEN_RE = re.compile(r"\b[A-Za-z0-9_-]{32,}\b")


def sanitize_metric_label(value, max_length=96):
    """Return a short, display-safe label for diagnostics UI."""
    text = str(value or "").replace("\r", " ").replace("\n", " ").strip()
    text = _AUTH_HEADER_RE.sub("[redacted-auth]", text)
    text = _BEARER_RE.sub("Bearer [redacted]", text)
    text = _COMMON_SECRET_RE.sub("[redacted]", text)
    text = _LONG_TOKEN_RE.sub("[redacted]", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_length:
        text = text[: max(0, max_length - 1)].rstrip() + "..."
    return text


class RuntimeMetrics:
    """Thread-safe, bounded runtime metrics for lightweight diagnostics."""

    def __init__(self, max_events=240, max_age_seconds=60.0, clock=None):
        self.max_events = max(1, int(max_events))
        self.max_age_seconds = max(0.0, float(max_age_seconds))
        self._clock = clock or time.monotonic
        self._lock = threading.RLock()
        self._timings = defaultdict(lambda: deque(maxlen=self.max_events))
        self._counters = defaultdict(int)
        self._gauges = {}
        self._labels = {}

    def record_timing(self, name, seconds):
        try:
            value = max(0.0, float(seconds))
        except (TypeError, ValueError):
            return
        metric_name = str(name or "").strip()
        if not metric_name:
            return
        timestamp = float(self._clock())
        with self._lock:
            self._timings[metric_name].append((timestamp, value))

    def increment(self, name, amount=1):
        metric_name = str(name or "").strip()
        if not metric_name:
            return
        try:
            delta = int(amount)
        except (TypeError, ValueError):
            return
        with self._lock:
            self._counters[metric_name] += delta

    def set_gauge(self, name, value):
        metric_name = str(name or "").strip()
        if not metric_name:
            return
        try:
            gauge_value = float(value)
        except (TypeError, ValueError):
            return
        if gauge_value.is_integer():
            gauge_value = int(gauge_value)
        with self._lock:
            self._gauges[metric_name] = gauge_value

    def set_label(self, name, value):
        metric_name = str(name or "").strip()
        if not metric_name:
            return
        with self._lock:
            self._labels[metric_name] = sanitize_metric_label(value)

    def reset(self):
        with self._lock:
            self._timings.clear()
            self._counters.clear()
            self._gauges.clear()
            self._labels.clear()

    def snapshot(self):
        now = float(self._clock())
        with self._lock:
            timings = {}
            for name, events in list(self._timings.items()):
                values = self._fresh_timing_values(events, now)
                if values:
                    timings[name] = {
                        "count": len(values),
                        "latest": values[-1],
                        "p50": self._percentile(values, 50),
                        "p90": self._percentile(values, 90),
                    }
                elif not events:
                    self._timings.pop(name, None)

            return {
                "window_seconds": self.max_age_seconds,
                "timings": timings,
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "labels": dict(self._labels),
            }

    def summary_text(self):
        snapshot = self.snapshot()
        lines = [
            f"Runtime metrics window: {snapshot['window_seconds']:.0f}s",
            "Timings:",
        ]
        if snapshot["timings"]:
            for name in sorted(snapshot["timings"]):
                timing = snapshot["timings"][name]
                lines.append(
                    f"- {name}: latest={timing['latest']:.3f}s "
                    f"p50={timing['p50']:.3f}s p90={timing['p90']:.3f}s "
                    f"count={timing['count']}"
                )
        else:
            lines.append("- none")

        lines.append("Counters:")
        if snapshot["counters"]:
            for name in sorted(snapshot["counters"]):
                lines.append(f"- {name}: {snapshot['counters'][name]}")
        else:
            lines.append("- none")

        lines.append("Gauges:")
        if snapshot["gauges"]:
            for name in sorted(snapshot["gauges"]):
                lines.append(f"- {name}: {snapshot['gauges'][name]}")
        else:
            lines.append("- none")

        lines.append("Labels:")
        if snapshot["labels"]:
            for name in sorted(snapshot["labels"]):
                lines.append(f"- {name}: {snapshot['labels'][name]}")
        else:
            lines.append("- none")
        return "\n".join(lines)

    def _fresh_timing_values(self, events, now):
        if self.max_age_seconds <= 0:
            return [value for _timestamp, value in events]

        cutoff = now - self.max_age_seconds
        while events and events[0][0] < cutoff:
            events.popleft()
        return [value for _timestamp, value in events]

    @staticmethod
    def _percentile(values, percentile):
        if not values:
            return 0.0
        ordered = sorted(values)
        rank = int(math.ceil((percentile / 100.0) * len(ordered)))
        index = min(len(ordered) - 1, max(0, rank - 1))
        return ordered[index]
