import unittest

from runtime_metrics import RuntimeMetrics


class RuntimeMetricsTests(unittest.TestCase):
    def test_timing_snapshot_reports_latest_p50_and_p90(self):
        metrics = RuntimeMetrics(max_events=10, clock=lambda: 100.0)

        for value in [0.10, 0.20, 0.30, 0.40, 0.50]:
            metrics.record_timing("capture_duration", value)

        timing = metrics.snapshot()["timings"]["capture_duration"]
        self.assertEqual(timing["count"], 5)
        self.assertAlmostEqual(timing["latest"], 0.50)
        self.assertAlmostEqual(timing["p50"], 0.30)
        self.assertAlmostEqual(timing["p90"], 0.50)

    def test_counters_gauges_and_reset(self):
        metrics = RuntimeMetrics(max_events=10, clock=lambda: 100.0)

        metrics.increment("instant_cache_hit")
        metrics.increment("instant_cache_hit", 2)
        metrics.set_gauge("active_translation_calls", 2)
        metrics.set_gauge("translation_concurrency_limit", 6)

        snapshot = metrics.snapshot()
        self.assertEqual(snapshot["counters"]["instant_cache_hit"], 3)
        self.assertEqual(snapshot["gauges"]["active_translation_calls"], 2)
        self.assertEqual(snapshot["gauges"]["translation_concurrency_limit"], 6)

        metrics.reset()

        reset_snapshot = metrics.snapshot()
        self.assertEqual(reset_snapshot["timings"], {})
        self.assertEqual(reset_snapshot["counters"], {})
        self.assertEqual(reset_snapshot["gauges"], {})
        self.assertEqual(reset_snapshot["labels"], {})

    def test_event_window_is_bounded_by_count_and_age(self):
        current_time = [100.0]
        metrics = RuntimeMetrics(max_events=3, max_age_seconds=5.0, clock=lambda: current_time[0])

        for index, value in enumerate([1.0, 2.0, 3.0, 4.0]):
            current_time[0] = 100.0 + index
            metrics.record_timing("ocr_duration", value)

        timing = metrics.snapshot()["timings"]["ocr_duration"]
        self.assertEqual(timing["count"], 3)
        self.assertAlmostEqual(timing["latest"], 4.0)
        self.assertAlmostEqual(timing["p50"], 3.0)

        current_time[0] = 110.0

        self.assertEqual(metrics.snapshot()["timings"], {})

    def test_labels_are_redacted_before_snapshot_and_summary(self):
        metrics = RuntimeMetrics(clock=lambda: 100.0)
        metrics.set_label(
            "race_winner",
            "Fast profile sk-test-secret Authorization: Bearer abc123XYZ",
        )

        snapshot_value = metrics.snapshot()["labels"]["race_winner"]
        summary = metrics.summary_text()

        self.assertIn("Fast profile", snapshot_value)
        self.assertNotIn("sk-test-secret", snapshot_value)
        self.assertNotIn("Bearer abc123XYZ", snapshot_value)
        self.assertNotIn("Authorization:", summary)
        self.assertNotIn("sk-test-secret", summary)


if __name__ == "__main__":
    unittest.main()
