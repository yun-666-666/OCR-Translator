import inspect
import queue
import types
import unittest
from unittest.mock import Mock, patch

from PIL import Image

from worker_capture import (
    CaptureUISnapshot,
    get_capture_ui_snapshot,
    publish_capture_ui_snapshot,
    run_capture_thread,
)


class CaptureUISnapshotTests(unittest.TestCase):
    def test_snapshot_is_frozen_and_widget_free(self):
        app = types.SimpleNamespace(
            current_scan_interval=120,
            base_scan_interval=100,
            source_area=[1, 2, 11, 12],
            get_ocr_model_setting=lambda: "custom_ai",
            is_api_based_ocr_model=lambda model=None: True,
            keep_linebreaks_var=types.SimpleNamespace(get=lambda: False),
            source_overlay=None,
        )
        snapshot = publish_capture_ui_snapshot(app, reason="unit")
        self.assertIsInstance(snapshot, CaptureUISnapshot)
        self.assertEqual(snapshot.source_geometry, (1, 2, 11, 12))
        self.assertEqual(snapshot.ocr_model, "custom_ai")
        self.assertTrue(snapshot.is_api_based)
        with self.assertRaises(Exception):
            snapshot.ocr_model = "paddleocr"
        fields = snapshot.__dict__
        self.assertNotIn("source_overlay", fields)
        self.assertTrue(all(not hasattr(value, "winfo_exists") for value in fields.values()))

    def test_worker_reads_snapshot_without_touching_tk_apis(self):
        screenshot = Image.new("RGB", (4, 4), (9, 9, 9))
        forbidden = Mock(side_effect=AssertionError("Tk access from capture worker"))

        app = types.SimpleNamespace(
            is_running=True,
            current_scan_interval=50,
            base_scan_interval=50,
            update_adaptive_scan_interval=lambda: None,
            get_ocr_model_setting=lambda: "custom_ai",
            is_api_based_ocr_model=lambda model=None: True,
            source_area=[10, 20, 14, 24],
            source_overlay=None,
            scan_interval_var=types.SimpleNamespace(get=lambda: 50),
            keep_linebreaks_var=types.SimpleNamespace(get=lambda: False),
            ocr_frame_cache=types.SimpleNamespace(clear=Mock()),
            ocr_stability_gate=types.SimpleNamespace(clear=Mock(return_value=True)),
            ocr_queue=queue.Queue(maxsize=1),
            last_processed_subtitle=None,
            previous_text="",
            text_stability_counter=0,
            active_ocr_calls=set(),
            get_effective_ocr_concurrency_limit=lambda provider=None: 2,
        )
        # Publish while overlay APIs are still healthy, then poison them so the
        # worker loop must rely only on the immutable snapshot.
        publish_capture_ui_snapshot(app, reason="setup")
        app.source_overlay = types.SimpleNamespace(
            winfo_exists=forbidden,
            get_geometry=forbidden,
        )
        app.scan_interval_var = types.SimpleNamespace(get=forbidden)
        app.keep_linebreaks_var = types.SimpleNamespace(get=forbidden)
        captured = []

        def stop_after_put(item):
            captured.append(item)
            app.is_running = False

        app.ocr_queue.put_nowait = stop_after_put

        with patch(
            "worker_threads.capture_screen_region",
            return_value=screenshot,
        ) as capture, patch(
            "worker_capture.time.sleep",
            return_value=None,
        ):
            run_capture_thread(app)

        capture.assert_called_once_with((10, 20, 4, 4))
        forbidden.assert_not_called()
        self.assertEqual(captured, [screenshot])

    def test_generation_change_clears_old_ocr_state(self):
        screenshot_a = Image.new("RGB", (4, 4), (1, 2, 3))
        screenshot_b = Image.new("RGB", (4, 4), (4, 5, 6))
        app = types.SimpleNamespace(
            is_running=True,
            current_scan_interval=50,
            base_scan_interval=50,
            update_adaptive_scan_interval=lambda: None,
            get_ocr_model_setting=lambda: "custom_ai",
            is_api_based_ocr_model=lambda model=None: True,
            source_area=[0, 0, 4, 4],
            source_overlay=None,
            ocr_frame_cache=types.SimpleNamespace(clear=Mock()),
            ocr_stability_gate=types.SimpleNamespace(clear=Mock(return_value=True)),
            ocr_queue=queue.Queue(maxsize=2),
            last_processed_subtitle="old",
            previous_text="old",
            text_stability_counter=5,
            keep_linebreaks_var=types.SimpleNamespace(get=lambda: False),
            active_ocr_calls=set(),
            get_effective_ocr_concurrency_limit=lambda provider=None: 2,
        )
        publish_capture_ui_snapshot(app, reason="first")
        frames = [screenshot_a, screenshot_b]
        captured = []

        def capture_side_effect(region):
            frame = frames.pop(0)
            if not frames:
                publish_capture_ui_snapshot(
                    app,
                    bump_generation=True,
                    reason="geometry changed mid-run",
                )
            return frame

        def stop_after_put(item):
            captured.append(item)
            if len(captured) >= 2:
                app.is_running = False

        app.ocr_queue.put_nowait = stop_after_put

        with patch(
            "worker_threads.capture_screen_region",
            side_effect=capture_side_effect,
        ), patch(
            "worker_capture.time.sleep",
            return_value=None,
        ):
            run_capture_thread(app)

        self.assertIsNone(app.last_processed_subtitle)
        self.assertEqual(app.previous_text, "")
        self.assertEqual(app.text_stability_counter, 0)
        app.ocr_frame_cache.clear.assert_called()
        app.ocr_stability_gate.clear.assert_called()
        self.assertEqual(len(captured), 2)

    def test_capture_hot_path_source_has_no_tk_widget_calls(self):
        source = inspect.getsource(run_capture_thread)
        self.assertNotIn("winfo_exists", source)
        self.assertNotIn("get_geometry", source)
        self.assertNotIn("scan_interval_var", source)
        self.assertNotIn("source_overlay", source)
        self.assertNotIn("tk.Toplevel", source)
        self.assertNotIn("TclError", source)


if __name__ == "__main__":
    unittest.main()
