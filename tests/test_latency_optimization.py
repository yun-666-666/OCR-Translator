import importlib
import importlib.util
import base64
import inspect
import io
import os
import queue
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
from runtime_metrics import RuntimeMetrics


def import_ocr_utils_for_tests():
    if "cv2" not in sys.modules:
        module = types.ModuleType("cv2")
        module.__spec__ = importlib.machinery.ModuleSpec("cv2", loader=None)
        sys.modules["cv2"] = module
    return importlib.import_module("ocr_utils")


def import_translation_handler_for_tests():
    spec = importlib.util.spec_from_file_location(
        "translation_handler_latency_tests",
        Path(__file__).resolve().parents[1] / "handlers" / "translation_handler.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TranslationHandler


def import_worker_threads_for_tests():
    for optional_module in ("cv2", "pyautogui"):
        if optional_module not in sys.modules:
            module = types.ModuleType(optional_module)
            module.__spec__ = importlib.machinery.ModuleSpec(optional_module, loader=None)
            sys.modules[optional_module] = module
    return importlib.import_module("worker_threads")


class LatencyCaptureBackendTests(unittest.TestCase):
    def test_capture_uses_mss(self):
        capture_screen_region = import_ocr_utils_for_tests().capture_screen_region

        class FakeShot:
            size = (1, 1)
            bgra = bytes([30, 20, 10, 255])

        class FakeMss:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def grab(self, monitor):
                self.monitor = monitor
                return FakeShot()

        fake_mss = FakeMss()
        image = capture_screen_region(
            (10, 20, 30, 40),
            mss_factory=lambda: fake_mss,
        )

        self.assertEqual(image.mode, "RGB")
        self.assertEqual(image.size, (1, 1))
        self.assertEqual(image.getpixel((0, 0)), (10, 20, 30))
        self.assertEqual(fake_mss.monitor, {"left": 10, "top": 20, "width": 30, "height": 40})

    def test_capture_reports_mss_failure_without_hidden_backend_switch(self):
        capture_screen_region = import_ocr_utils_for_tests().capture_screen_region

        with self.assertRaisesRegex(RuntimeError, "MSS capture failed"):
            capture_screen_region(
                (5, 6, 7, 8),
                mss_factory=lambda: (_ for _ in ()).throw(
                    RuntimeError("mss unavailable")
                ),
            )


class CaptureOcrHotPathLoggingTests(unittest.TestCase):
    def test_capture_success_uses_coalesced_backend_geometry_key(self):
        ocr_utils = import_ocr_utils_for_tests()

        class FakeShot:
            size = (1, 1)
            bgra = bytes([30, 20, 10, 255])

        class FakeMss:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def grab(self, _monitor):
                return FakeShot()

        with patch.object(
            ocr_utils,
            "log_debug_coalesced",
        ) as log_coalesced:
            image = ocr_utils.capture_screen_region(
                (10, 20, 30, 40),
                mss_factory=FakeMss,
            )

        self.assertEqual(image.size, (1, 1))
        log_coalesced.assert_called_once_with(
            ("capture-success", "mss", 30, 40),
            "CAPTURE: mss captured 30x40",
            interval_seconds=5.0,
        )

    def test_capture_failure_is_logged_immediately(self):
        ocr_utils = import_ocr_utils_for_tests()

        with patch.object(ocr_utils, "log_debug") as immediate_log:
            with patch.object(
                ocr_utils,
                "log_debug_coalesced",
            ) as log_coalesced:
                with self.assertRaisesRegex(RuntimeError, "MSS capture failed"):
                    ocr_utils.capture_screen_region(
                        (5, 6, 7, 8),
                        mss_factory=lambda: (_ for _ in ()).throw(
                            RuntimeError("mss unavailable")
                        ),
                    )

        self.assertTrue(
            any(
                "mss backend failed" in call.args[0]
                for call in immediate_log.call_args_list
            )
        )
        log_coalesced.assert_not_called()

    def test_ocr_frame_cache_hit_uses_coalesced_log_without_changing_value(self):
        ocr_utils = import_ocr_utils_for_tests()
        cache = ocr_utils.OCRFrameCache(max_size=2)
        key = ("frame", "paddleocr", "en")
        cache.put(key, "recognized subtitle")

        with patch.object(
            ocr_utils,
            "log_debug_coalesced",
        ) as log_coalesced:
            result = cache.get(key)

        self.assertEqual(result, "recognized subtitle")
        log_coalesced.assert_called_once_with(
            "ocr-frame-cache-hit",
            "OCR CACHE: frame hit",
            interval_seconds=5.0,
        )

    def test_worker_timing_routes_normal_and_slow_events_independently(self):
        worker_threads = import_worker_threads_for_tests()

        with patch.object(
            worker_threads,
            "log_debug_coalesced",
        ) as log_coalesced:
            worker_threads._log_hot_path_timing(
                "capture-mss",
                "capture took 0.010s",
                duration_seconds=0.010,
                slow_threshold_seconds=0.050,
            )
            worker_threads._log_hot_path_timing(
                "capture-mss",
                "capture took 0.050s",
                duration_seconds=0.050,
                slow_threshold_seconds=0.050,
            )

        self.assertEqual(
            log_coalesced.call_args_list,
            [
                unittest.mock.call(
                    ("capture-mss", "normal"),
                    "capture took 0.010s",
                    interval_seconds=5.0,
                ),
                unittest.mock.call(
                    ("capture-mss", "slow"),
                    "SLOW: capture took 0.050s",
                    interval_seconds=1.0,
                ),
            ],
        )

    def test_paddle_routing_uses_stable_coalesced_log(self):
        worker_threads = import_worker_threads_for_tests()

        with patch.object(
            worker_threads,
            "log_debug_coalesced",
        ) as log_coalesced:
            worker_threads._log_paddle_ocr_route()

        log_coalesced.assert_called_once_with(
            "ocr-routing-paddle",
            "WT: OCR routing to PaddleOCR PP-OCRv6",
            interval_seconds=5.0,
        )


class LatencyCaptureThreadBackendSelectionTests(unittest.TestCase):
    def test_local_capture_signature_allows_one_duplicate_before_skipping(self):
        worker_threads = import_worker_threads_for_tests()
        first_signature = ("frame-a", 10, 20, 300, 80, "mss")
        changed_signature = ("frame-b", 10, 20, 300, 80, "mss")

        last, repeats, enqueue = worker_threads._advance_local_capture_signature(
            None,
            0,
            first_signature,
        )
        self.assertEqual((last, repeats, enqueue), (first_signature, 0, True))

        last, repeats, enqueue = worker_threads._advance_local_capture_signature(
            last,
            repeats,
            first_signature,
        )
        self.assertEqual((repeats, enqueue), (1, True))

        last, repeats, enqueue = worker_threads._advance_local_capture_signature(
            last,
            repeats,
            first_signature,
        )
        self.assertEqual((repeats, enqueue), (2, False))

        last, repeats, enqueue = worker_threads._advance_local_capture_signature(
            last,
            repeats,
            changed_signature,
        )
        self.assertEqual((last, repeats, enqueue), (changed_signature, 0, True))

    def test_api_capture_saturation_uses_effective_limit(self):
        worker_threads = import_worker_threads_for_tests()
        app = types.SimpleNamespace(
            active_ocr_calls={"first", "second"},
            max_concurrent_ocr_calls=8,
            is_api_based_ocr_model=lambda model: model == "custom_ai",
            get_effective_ocr_concurrency_limit=lambda provider: 2,
        )

        self.assertTrue(
            worker_threads._api_ocr_capture_is_saturated(app, "custom_ai")
        )
        self.assertFalse(
            worker_threads._api_ocr_capture_is_saturated(app, "paddleocr")
        )

        app.active_ocr_calls.clear()
        app.get_effective_ocr_concurrency_limit = lambda provider: 0
        self.assertTrue(
            worker_threads._api_ocr_capture_is_saturated(app, "custom_ai")
        )

    def test_saturated_api_capture_waits_then_resumes_with_current_frame(self):
        worker_threads = import_worker_threads_for_tests()
        screenshot = Image.new("RGB", (8, 8), (1, 2, 3))
        metrics = RuntimeMetrics(clock=lambda: 100.0)

        class FakeOverlay:
            def winfo_exists(self):
                return True

            def get_geometry(self):
                return (10, 20, 18, 28)

        app = types.SimpleNamespace(
            is_running=True,
            current_scan_interval=200,
            scan_interval_var=types.SimpleNamespace(get=lambda: 200),
            update_adaptive_scan_interval=lambda: None,
            get_ocr_model_setting=lambda: "custom_ai",
            is_api_based_ocr_model=lambda model=None: model == "custom_ai",
            get_effective_ocr_concurrency_limit=lambda provider=None: 2,
            active_ocr_calls={"first", "second"},
            max_concurrent_ocr_calls=8,
            source_overlay=FakeOverlay(),
            ocr_frame_cache=types.SimpleNamespace(clear=Mock()),
            ocr_stability_gate=types.SimpleNamespace(clear=Mock(return_value=True)),
            ocr_queue=queue.Queue(maxsize=4),
            runtime_metrics=metrics,
            last_processed_subtitle=None,
            previous_text="",
            text_stability_counter=0,
        )
        original_put_nowait = app.ocr_queue.put_nowait

        def stop_after_put(item):
            original_put_nowait(item)
            app.is_running = False

        app.ocr_queue.put_nowait = stop_after_put

        def release_capacity(_seconds):
            app.active_ocr_calls.clear()

        with (
            patch.object(worker_threads.tk, "Toplevel", FakeOverlay),
            patch.object(
                worker_threads,
                "capture_screen_region",
                return_value=screenshot,
            ) as capture,
            patch.object(worker_threads.time, "sleep", side_effect=release_capacity),
        ):
            worker_threads.run_capture_thread(app)

        capture.assert_called_once_with((10, 20, 8, 8))
        self.assertIs(app.ocr_queue.get_nowait(), screenshot)
        self.assertEqual(
            metrics.snapshot()["counters"]["api_ocr_capture_backpressure_skip"],
            1,
        )

    def test_custom_ai_cooldown_bypasses_backpressure_for_paddle_fallback(self):
        worker_threads = import_worker_threads_for_tests()
        screenshot = Image.new("RGB", (8, 8), (4, 5, 6))
        metrics = RuntimeMetrics(clock=lambda: 100.0)

        class FakeOverlay:
            def winfo_exists(self):
                return True

            def get_geometry(self):
                return (10, 20, 18, 28)

        app = types.SimpleNamespace(
            is_running=True,
            current_scan_interval=200,
            scan_interval_var=types.SimpleNamespace(get=lambda: 200),
            update_adaptive_scan_interval=lambda: None,
            get_ocr_model_setting=lambda: "custom_ai",
            is_api_based_ocr_model=lambda model=None: model == "custom_ai",
            get_effective_ocr_concurrency_limit=lambda provider=None: 2,
            active_ocr_calls={"first", "second"},
            max_concurrent_ocr_calls=8,
            translation_handler=types.SimpleNamespace(
                get_active_custom_ai_ocr_cooldown_seconds=lambda: 30.0
            ),
            source_overlay=FakeOverlay(),
            ocr_frame_cache=types.SimpleNamespace(clear=Mock()),
            ocr_stability_gate=types.SimpleNamespace(clear=Mock(return_value=True)),
            ocr_queue=queue.Queue(maxsize=4),
            runtime_metrics=metrics,
            last_processed_subtitle=None,
            previous_text="",
            text_stability_counter=0,
        )
        original_put_nowait = app.ocr_queue.put_nowait

        def stop_after_put(item):
            original_put_nowait(item)
            app.is_running = False

        app.ocr_queue.put_nowait = stop_after_put

        def stop_if_backpressured(_seconds):
            app.is_running = False

        with (
            patch.object(worker_threads.tk, "Toplevel", FakeOverlay),
            patch.object(
                worker_threads,
                "capture_screen_region",
                return_value=screenshot,
            ) as capture,
            patch.object(
                worker_threads.time,
                "sleep",
                side_effect=stop_if_backpressured,
            ),
        ):
            worker_threads.run_capture_thread(app)

        capture.assert_called_once_with((10, 20, 8, 8))
        self.assertIs(app.ocr_queue.get_nowait(), screenshot)
        self.assertEqual(
            metrics.snapshot()["counters"].get(
                "api_ocr_capture_backpressure_skip",
                0,
            ),
            0,
        )

    def test_capture_thread_uses_mss_and_continues_after_capture_error(self):
        worker_threads = import_worker_threads_for_tests()
        screenshot = Image.new("RGB", (8, 8), (1, 2, 3))
        capture_regions = []

        class FakeOverlay:
            def winfo_exists(self):
                return True

            def get_geometry(self):
                return (10, 20, 18, 28)

        app = types.SimpleNamespace(
            is_running=True,
            current_scan_interval=50,
            scan_interval_var=types.SimpleNamespace(get=lambda: 50),
            update_adaptive_scan_interval=lambda: None,
            get_ocr_model_setting=lambda: "custom_ai",
            is_api_based_ocr_model=lambda model=None: True,
            source_overlay=FakeOverlay(),
            ocr_frame_cache=types.SimpleNamespace(clear=Mock()),
            ocr_stability_gate=types.SimpleNamespace(clear=Mock(return_value=True)),
            ocr_queue=queue.Queue(maxsize=4),
            last_processed_subtitle=None,
            previous_text="",
            text_stability_counter=0,
        )

        original_put_nowait = app.ocr_queue.put_nowait

        def stop_after_put(item):
            original_put_nowait(item)
            app.is_running = False

        app.ocr_queue.put_nowait = stop_after_put

        def capture_func(region):
            capture_regions.append(region)
            if len(capture_regions) == 1:
                raise RuntimeError("transient capture failure")
            return screenshot

        with (
            patch.object(worker_threads.tk, "Toplevel", FakeOverlay),
            patch.object(worker_threads, "capture_screen_region", side_effect=capture_func),
            patch.object(worker_threads.time, "sleep", return_value=None),
        ):
            worker_threads.run_capture_thread(app)

        self.assertEqual(capture_regions, [(10, 20, 8, 8), (10, 20, 8, 8)])
        app.ocr_stability_gate.clear.assert_called()
        self.assertIs(app.ocr_queue.get_nowait(), screenshot)


class LatencyOcrCacheTests(unittest.TestCase):
    def test_ocr_frame_cache_returns_cached_text_without_recomputing(self):
        ocr_utils = import_ocr_utils_for_tests()
        OCRFrameCache = ocr_utils.OCRFrameCache
        build_ocr_frame_cache_key = ocr_utils.build_ocr_frame_cache_key

        cache = OCRFrameCache(max_size=2)
        key = build_ocr_frame_cache_key(
            image_hash="abc",
            ocr_model="paddleocr",
            source_lang="en",
            preprocessing_mode="none",
            region_size=(300, 80),
        )

        self.assertIsNone(cache.get(key))
        cache.put(key, "Hello")
        self.assertEqual(cache.get(key), "Hello")

    def test_ocr_frame_cache_ignores_empty_text(self):
        ocr_utils = import_ocr_utils_for_tests()
        cache = ocr_utils.OCRFrameCache(max_size=2)
        key = ocr_utils.build_ocr_frame_cache_key(
            image_hash="empty",
            ocr_model="paddleocr",
            source_lang="en",
            preprocessing_mode="none",
            region_size=(300, 80),
        )

        cache.put(key, "")

        self.assertIsNone(cache.get(key))

    def test_ocr_frame_cache_resize_trims_oldest_entries_and_can_disable_cache(self):
        ocr_utils = import_ocr_utils_for_tests()
        cache = ocr_utils.OCRFrameCache(max_size=3)
        keys = [
            ocr_utils.build_ocr_frame_cache_key(
                image_hash=f"frame-{index}",
                ocr_model="paddleocr",
                source_lang="en",
                preprocessing_mode="none",
                region_size=(300, 80),
            )
            for index in range(3)
        ]
        for index, key in enumerate(keys):
            cache.put(key, f"text-{index}")

        cache.resize(2)

        self.assertIsNone(cache.get(keys[0]))
        self.assertEqual(cache.get(keys[1]), "text-1")
        self.assertEqual(cache.get(keys[2]), "text-2")

        cache.resize(0)

        self.assertIsNone(cache.get(keys[1]))
        self.assertIsNone(cache.get(keys[2]))

    def test_app_resizes_existing_ocr_frame_cache_when_size_setting_changes(self):
        ocr_utils = import_ocr_utils_for_tests()
        import app_logic

        cache = ocr_utils.OCRFrameCache(max_size=3)
        keys = [
            ocr_utils.build_ocr_frame_cache_key(
                image_hash=f"frame-{index}",
                ocr_model="paddleocr",
                source_lang="en",
                preprocessing_mode="none",
                region_size=(300, 80),
            )
            for index in range(3)
        ]
        for index, key in enumerate(keys):
            cache.put(key, f"text-{index}")

        app = object.__new__(app_logic.GameChangingTranslator)
        app.ocr_frame_cache = cache
        app.ocr_frame_cache_size_var = types.SimpleNamespace(get=lambda: 2)

        app.on_ocr_frame_cache_size_change()

        self.assertIsNone(cache.get(keys[0]))
        self.assertEqual(cache.get(keys[1]), "text-1")
        self.assertEqual(cache.get(keys[2]), "text-2")

    def test_app_clear_cache_clears_ocr_frame_cache_and_last_processed_subtitle(self):
        ocr_utils = import_ocr_utils_for_tests()
        import app_logic

        cache = ocr_utils.OCRFrameCache(max_size=2)
        cache_key = ocr_utils.build_ocr_frame_cache_key(
            image_hash="frame",
            ocr_model="custom_ai",
            source_lang="en",
            preprocessing_mode="api",
            region_size=(300, 80),
        )
        cache.put(cache_key, "cached OCR")

        app = object.__new__(app_logic.GameChangingTranslator)
        app.marian_translator = None
        app.translation_handler = types.SimpleNamespace(clear_cache=Mock())
        app.google_file_cache = {"cached": "value"}
        app.deepl_file_cache = {"cached": "value"}
        app.ocr_queue = queue.Queue()
        app.translation_queue = queue.Queue()
        app.active_translation_inflight_keys = {"inflight"}
        app.translation_cache = {"old": "translation"}
        app.text_stability_counter = 3
        app.previous_text = "old text"
        app.ocr_frame_cache = cache
        app.last_processed_subtitle = "cached OCR"
        app.ocr_stability_gate = types.SimpleNamespace(clear=Mock(return_value=True))
        app.status_label = types.SimpleNamespace(
            cget=lambda _name: "Status: Running",
            config=Mock(),
            winfo_exists=lambda: True,
        )
        app.root = types.SimpleNamespace(
            winfo_exists=lambda: True,
            after=lambda *args: None,
        )

        app.clear_cache()

        self.assertIsNone(cache.get(cache_key))
        self.assertIsNone(app.last_processed_subtitle)
        app.ocr_stability_gate.clear.assert_called()

    def test_ocr_cache_key_includes_region_origin(self):
        ocr_utils = import_ocr_utils_for_tests()

        first = ocr_utils.build_ocr_frame_cache_key(
            image_hash="same",
            ocr_model="paddleocr",
            source_lang="en",
            preprocessing_mode="none",
            region_size=(300, 80),
            region_origin=(10, 20),
        )
        second = ocr_utils.build_ocr_frame_cache_key(
            image_hash="same",
            ocr_model="paddleocr",
            source_lang="en",
            preprocessing_mode="none",
            region_size=(300, 80),
            region_origin=(500, 600),
        )

        self.assertNotEqual(first, second)

    def test_capture_signature_includes_region_and_backend(self):
        ocr_utils = import_ocr_utils_for_tests()

        first = ocr_utils.build_capture_signature("same", (10, 20, 300, 80), "auto")
        second = ocr_utils.build_capture_signature("same", (500, 600, 300, 80), "auto")
        third = ocr_utils.build_capture_signature("same", (10, 20, 300, 80), "mss")

        self.assertNotEqual(first, second)
        self.assertNotEqual(first, third)


class LatencyShutdownTests(unittest.TestCase):
    def test_on_closing_stops_running_app_without_user_stop_poll(self):
        import app_logic

        scheduled = []

        class Root:
            def __init__(self):
                self.destroyed = False

            def after(self, *args):
                scheduled.append(args)
                return f"after-{len(scheduled)}"

            def after_cancel(self, _after_id):
                return None

            def winfo_exists(self):
                return not self.destroyed

            def destroy(self):
                self.destroyed = True

        app = object.__new__(app_logic.GameChangingTranslator)
        app.root = Root()
        app.runtime_metrics_refresh_after_id = None
        app.ocr_preview_window = None
        app.is_running = True
        app.toggle_translation = Mock(
            side_effect=AssertionError("app exit must not use the user Stop path")
        )
        app.threads = []
        app.active_ocr_calls = {1}
        app.active_translation_calls = {2}
        app.translation_handler = Mock()
        app.marian_translator = None
        app.ocr_thread_pool = Mock()
        app.translation_thread_pool = Mock()
        app._fully_initialized = True
        app.save_settings = Mock()
        app.KEYBOARD_AVAILABLE = False
        app.source_overlay = None
        app.target_overlay = None
        app.translation_text = object()

        with patch.object(app_logic, "log_debug"):
            app.on_closing()

        self.assertEqual(scheduled, [])
        self.assertFalse(app.is_running)
        app.translation_handler.close.assert_called_once()
        app.ocr_thread_pool.shutdown.assert_called_once_with(
            wait=False,
            cancel_futures=True,
        )
        app.translation_thread_pool.shutdown.assert_called_once_with(
            wait=False,
            cancel_futures=True,
        )
        self.assertTrue(app.root.destroyed)

    def test_finalize_shutdown_is_idempotent_and_skips_dead_widgets(self):
        import app_logic

        class DeadWidget:
            def winfo_exists(self):
                return False

            def config(self, *args, **kwargs):
                raise AssertionError("dead widgets must not be touched")

            def delete(self, *args, **kwargs):
                raise AssertionError("dead widgets must not be touched")

            def winfo_viewable(self):
                raise AssertionError("dead widgets must not be touched")

            def hide(self):
                raise AssertionError("dead widgets must not be touched")

        app = object.__new__(app_logic.GameChangingTranslator)
        app.translation_handler = Mock()
        app.ocr_queue = queue.Queue()
        app.translation_queue = queue.Queue()
        app.clear_ocr_stability_gate = Mock()
        app.translation_text = DeadWidget()
        app.source_overlay = DeadWidget()
        app.target_overlay = DeadWidget()
        app.start_stop_btn = Mock()
        app.status_label = Mock()
        app.ui_lang = types.SimpleNamespace(
            get_label=lambda _key, default=None: default or _key
        )
        app.toggle_in_progress = True
        app.pending_translation_request = {
            "text": "old subtitle",
            "ocr_sequence_number": 0,
            "requested_at_monotonic": 10.0,
        }
        app.pending_translation_flush_scheduled = True
        app.pending_translation_flush_deadline_monotonic = 70.0
        app.pending_translation_flush_generation = 2
        app.latest_translation_candidate = dict(app.pending_translation_request)
        app.translation_profile_refresh_generation = 5
        app.last_translation_submit_monotonic = 12.0
        app.active_translation_inflight_keys = {"old-request"}
        app.active_translation_started_monotonic = {4: 12.0}

        with patch.object(app_logic, "log_debug"):
            app._finalize_shutdown()
            app._finalize_shutdown()

        app.translation_handler.request_end_ocr_session.assert_called_once()
        app.translation_handler.request_end_translation_session.assert_called_once()
        app.clear_ocr_stability_gate.assert_called_once_with(
            "translation stopped"
        )
        self.assertIsNone(app.pending_translation_request)
        self.assertFalse(app.pending_translation_flush_scheduled)
        self.assertIsNone(app.latest_translation_candidate)
        self.assertEqual(app.pending_translation_flush_generation, 3)
        self.assertEqual(app.translation_profile_refresh_generation, 6)
        self.assertEqual(app.last_translation_submit_monotonic, 0.0)
        self.assertFalse(app.toggle_in_progress)

    def test_graceful_shutdown_waits_for_active_custom_ai_sets(self):
        import app_logic

        scheduled = []
        app = object.__new__(app_logic.GameChangingTranslator)
        app.translation_handler = types.SimpleNamespace(
            ocr_providers={},
            providers={},
        )
        app.active_ocr_calls = {7}
        app.active_translation_calls = set()
        app._shutdown_start_time = 100.0
        app.root = types.SimpleNamespace(after=lambda *args: scheduled.append(args))
        app._finalize_shutdown = Mock()

        with patch.object(app_logic.time, "monotonic", return_value=105.0):
            app._graceful_shutdown_poll()

        app._finalize_shutdown.assert_not_called()
        self.assertEqual(len(scheduled), 1)
        self.assertEqual(scheduled[0][0], 100)

    def test_graceful_shutdown_waits_for_active_translation_calls(self):
        import app_logic

        scheduled = []
        app = object.__new__(app_logic.GameChangingTranslator)
        app.translation_handler = types.SimpleNamespace(
            ocr_providers={},
            providers={},
        )
        app.active_ocr_calls = set()
        app.active_translation_calls = {11}
        app._shutdown_start_time = 100.0
        app.root = types.SimpleNamespace(after=lambda *args: scheduled.append(args))
        app._finalize_shutdown = Mock()

        with patch.object(app_logic.time, "monotonic", return_value=105.0):
            app._graceful_shutdown_poll()

        app._finalize_shutdown.assert_not_called()
        self.assertEqual(len(scheduled), 1)
        self.assertEqual(scheduled[0][0], 100)


class TranslationSessionBoundaryTests(unittest.TestCase):
    def test_scheduler_session_reset_invalidates_previous_session_state(self):
        worker_threads = import_worker_threads_for_tests()
        app = types.SimpleNamespace(
            pending_translation_request={
                "text": "old subtitle",
                "ocr_sequence_number": 0,
                "requested_at_monotonic": 10.0,
            },
            pending_translation_flush_scheduled=True,
            pending_translation_flush_deadline_monotonic=70.0,
            pending_translation_flush_generation=7,
            latest_translation_candidate={
                "text": "old subtitle",
                "ocr_sequence_number": 0,
                "requested_at_monotonic": 10.0,
            },
            translation_profile_refresh_generation=3,
            last_translation_submit_monotonic=12.0,
            active_translation_inflight_keys={"old-request"},
            active_translation_started_monotonic={4: 12.0},
            translation_sequence_counter=9,
            latest_translation_sequence_started=8,
            last_displayed_translation_sequence=4,
            update_translation_text=Mock(),
        )

        worker_threads.reset_translation_scheduler_session_state(
            app,
            "new session",
        )

        self.assertIsNone(app.pending_translation_request)
        self.assertFalse(app.pending_translation_flush_scheduled)
        self.assertEqual(app.pending_translation_flush_deadline_monotonic, 0.0)
        self.assertEqual(app.pending_translation_flush_generation, 8)
        self.assertIsNone(app.latest_translation_candidate)
        self.assertEqual(app.translation_profile_refresh_generation, 4)
        self.assertEqual(app.last_translation_submit_monotonic, 0.0)
        self.assertEqual(app.active_translation_inflight_keys, set())
        self.assertEqual(app.active_translation_started_monotonic, {})
        self.assertEqual(app.last_displayed_translation_sequence, 9)

        worker_threads.process_translation_response(
            app,
            "old session result",
            translation_sequence=9,
            original_text="old subtitle",
            ocr_sequence_number=0,
        )
        app.update_translation_text.assert_not_called()

    def test_scheduler_session_reset_preserves_timed_out_active_request_ownership(self):
        worker_threads = import_worker_threads_for_tests()
        inflight_key = ("custom_ai", "still running", "request-snapshot")
        app = types.SimpleNamespace(
            pending_translation_request=None,
            pending_translation_flush_scheduled=False,
            pending_translation_flush_deadline_monotonic=0.0,
            pending_translation_flush_generation=2,
            latest_translation_candidate=None,
            translation_profile_refresh_generation=4,
            last_translation_submit_monotonic=22.0,
            active_translation_calls={7},
            active_translation_inflight_keys={inflight_key},
            active_translation_started_monotonic={7: 20.0, 6: 10.0},
            translation_sequence_counter=7,
            latest_translation_sequence_started=7,
            last_displayed_translation_sequence=5,
        )

        worker_threads.reset_translation_scheduler_session_state(
            app,
            "shutdown timeout",
        )

        self.assertEqual(app.active_translation_calls, {7})
        self.assertEqual(app.active_translation_inflight_keys, {inflight_key})
        self.assertEqual(app.active_translation_started_monotonic, {7: 20.0})
        self.assertEqual(app.last_displayed_translation_sequence, 7)

    def test_app_delegates_translation_scheduler_session_reset(self):
        import app_logic

        worker_threads = import_worker_threads_for_tests()
        app = object.__new__(app_logic.GameChangingTranslator)

        with patch.object(
            worker_threads,
            "reset_translation_scheduler_session_state",
        ) as reset_state:
            app._reset_translation_scheduler_session_state(
                "translation starting"
            )

        reset_state.assert_called_once_with(app, "translation starting")


class ApiOcrImagePayloadEncodingTests(unittest.TestCase):
    def test_lossless_webp_matches_current_api_encoding(self):
        ocr_utils = import_ocr_utils_for_tests()
        image = Image.new("RGBA", (16, 8), (10, 20, 30, 128))
        expected_image = Image.new("RGB", image.size, (255, 255, 255))
        expected_image.paste(image, mask=image.split()[-1])
        expected_buffer = io.BytesIO()
        expected_image.save(
            expected_buffer,
            format="WebP",
            lossless=True,
            method=0,
            exact=True,
        )

        encoded = ocr_utils.encode_image_for_api_ocr(
            image,
            mode="lossless_webp",
            quality=85,
        )

        self.assertEqual(encoded, expected_buffer.getvalue())

    def test_balanced_webp_is_smaller_than_lossless_for_noisy_frame(self):
        ocr_utils = import_ocr_utils_for_tests()
        rng = np.random.default_rng(1234)
        image = Image.fromarray(
            rng.integers(0, 256, size=(96, 160, 3), dtype=np.uint8)
        )

        lossless = ocr_utils.encode_image_for_api_ocr(
            image,
            mode="lossless_webp",
            quality=85,
        )
        balanced = ocr_utils.encode_image_for_api_ocr(
            image,
            mode="balanced_webp",
            quality=85,
        )

        self.assertLess(len(balanced), len(lossless))

    def test_small_grayscale_webp_handles_rgba_images(self):
        ocr_utils = import_ocr_utils_for_tests()
        image = Image.new("RGBA", (32, 24), (20, 40, 220, 180))
        image.putpixel((5, 5), (255, 32, 16, 255))

        encoded = ocr_utils.encode_image_for_api_ocr(
            image,
            mode="small_grayscale_webp",
            quality=80,
        )
        decoded = Image.open(io.BytesIO(encoded))
        sample = decoded.convert("RGB").getpixel((5, 5))

        self.assertEqual(decoded.format, "WEBP")
        self.assertEqual(decoded.size, image.size)
        self.assertLessEqual(max(sample) - min(sample), 2)

    def test_encode_image_for_api_ocr_rejects_empty_payload(self):
        ocr_utils = import_ocr_utils_for_tests()

        class EmptySavingImage:
            mode = "RGB"
            size = (10, 10)

            def save(self, _buffer, **_kwargs):
                return None

        with self.assertRaises(ValueError):
            ocr_utils.encode_image_for_api_ocr(
                EmptySavingImage(),
                mode="balanced_webp",
                quality=85,
            )

    def test_encode_api_ocr_image_payload_supports_png_and_jpeg_metadata(self):
        ocr_utils = import_ocr_utils_for_tests()
        image = Image.new("RGBA", (16, 8), (10, 20, 30, 128))

        png_payload = ocr_utils.encode_image_for_api_ocr_payload(
            image,
            mode="balanced_webp",
            quality=85,
            image_format="png",
        )
        jpeg_payload = ocr_utils.encode_image_for_api_ocr_payload(
            image,
            mode="balanced_webp",
            quality=85,
            image_format="jpeg",
        )

        self.assertEqual(png_payload.image_format, "png")
        self.assertEqual(png_payload.mime_type, "image/png")
        self.assertEqual(Image.open(io.BytesIO(png_payload.data)).format, "PNG")
        self.assertEqual(jpeg_payload.image_format, "jpeg")
        self.assertEqual(jpeg_payload.mime_type, "image/jpeg")
        self.assertEqual(Image.open(io.BytesIO(jpeg_payload.data)).format, "JPEG")

    def test_invalid_api_ocr_image_format_normalizes_to_webp(self):
        ocr_utils = import_ocr_utils_for_tests()

        self.assertEqual(ocr_utils.normalize_api_ocr_image_format("png"), "png")
        self.assertEqual(ocr_utils.normalize_api_ocr_image_format("jpg"), "jpeg")
        self.assertEqual(ocr_utils.normalize_api_ocr_image_format("bad"), "webp")


class TranslationInactivityClearTests(unittest.TestCase):
    def _make_app(self):
        scheduled = []
        displayed = []
        metrics = RuntimeMetrics(clock=lambda: 100.0)
        app = types.SimpleNamespace(
            is_running=True,
            previous_text="",
            active_translation_calls=set(),
            active_translation_inflight_keys=set(),
            pending_translation_request=None,
            pending_translation_flush_scheduled=False,
            translation_queue=queue.Queue(),
            ocr_stability_gate=types.SimpleNamespace(
                has_pending=lambda: False,
            ),
            last_successful_translation_time=95.0,
            last_displayed_translation_sequence=7,
            last_local_ocr_submitted_text="Same subtitle",
            last_local_ocr_submitted_norm="same subtitle",
            last_local_ocr_submitted_scope=("scope",),
            root=types.SimpleNamespace(
                after=lambda delay, callback, *args: scheduled.append(
                    (delay, callback, args)
                ),
            ),
            display_manager=types.SimpleNamespace(
                _update_translation_text_on_main_thread=displayed.append,
            ),
            update_translation_text=displayed.append,
            runtime_metrics=metrics,
        )
        return app, scheduled, displayed, metrics

    def test_active_and_pending_work_block_inactivity_clear_scheduling(self):
        worker_threads = import_worker_threads_for_tests()
        cases = {
            "stopped app": lambda app: setattr(app, "is_running", False),
            "source text": lambda app: setattr(app, "previous_text", "Text"),
            "active call": lambda app: app.active_translation_calls.add(1),
            "inflight identity": lambda app: app.active_translation_inflight_keys.add(
                ("custom_ai", "Text")
            ),
            "pending request": lambda app: setattr(
                app,
                "pending_translation_request",
                {"text": "Text"},
            ),
            "pending flush": lambda app: setattr(
                app,
                "pending_translation_flush_scheduled",
                True,
            ),
            "legacy queue": lambda app: app.translation_queue.put("Text"),
            "OCR stability": lambda app: setattr(
                app,
                "ocr_stability_gate",
                types.SimpleNamespace(has_pending=lambda: True),
            ),
        }

        for name, configure in cases.items():
            with self.subTest(name=name):
                app, scheduled, displayed, _metrics = self._make_app()
                configure(app)

                result = worker_threads._schedule_inactive_translation_clear(
                    app,
                    inactive_duration=5.0,
                    timeout_seconds=2.0,
                )

                self.assertFalse(result)
                self.assertEqual(scheduled, [])
                self.assertEqual(displayed, [])

    def test_inactivity_clear_runs_once_and_resets_local_resubmit_state(self):
        worker_threads = import_worker_threads_for_tests()
        app, scheduled, displayed, metrics = self._make_app()
        expected_epoch = (95.0, 7)

        self.assertTrue(
            worker_threads._should_skip_local_ocr_resubmit(
                app,
                "Same subtitle",
            )
        )
        self.assertTrue(
            worker_threads._schedule_inactive_translation_clear(
                app,
                inactive_duration=5.0,
                timeout_seconds=2.0,
            )
        )
        self.assertFalse(
            worker_threads._schedule_inactive_translation_clear(
                app,
                inactive_duration=5.1,
                timeout_seconds=2.0,
            )
        )
        self.assertEqual(len(scheduled), 1)

        _delay, callback, args = scheduled[0]
        callback(*args)

        self.assertEqual(displayed, [""])
        self.assertEqual(
            app.translation_inactivity_cleared_epoch,
            expected_epoch,
        )
        self.assertIsNone(app.last_local_ocr_submitted_text)
        self.assertIsNone(app.last_local_ocr_submitted_norm)
        self.assertIsNone(app.last_local_ocr_submitted_scope)
        self.assertFalse(
            worker_threads._should_skip_local_ocr_resubmit(
                app,
                "Same subtitle",
            )
        )
        self.assertEqual(
            metrics.snapshot()["counters"]["translation_inactivity_clear"],
            1,
        )
        self.assertFalse(
            worker_threads._schedule_inactive_translation_clear(
                app,
                inactive_duration=6.0,
                timeout_seconds=2.0,
            )
        )
        self.assertEqual(len(scheduled), 1)

    def test_display_activity_observation_advances_for_success_and_errors(self):
        worker_threads = import_worker_threads_for_tests()
        app, _scheduled, _displayed, _metrics = self._make_app()

        last_time, last_sequence = (
            worker_threads._advance_translation_display_activity(
                app,
                now=100.0,
                last_display_time=90.0,
                last_display_sequence=7,
            )
        )
        self.assertEqual((last_time, last_sequence), (95.0, 7))

        app.last_displayed_translation_sequence = 8
        last_time, last_sequence = (
            worker_threads._advance_translation_display_activity(
                app,
                now=101.0,
                last_display_time=last_time,
                last_display_sequence=last_sequence,
            )
        )
        self.assertEqual((last_time, last_sequence), (101.0, 8))

    def test_newer_display_epoch_invalidates_scheduled_clear(self):
        worker_threads = import_worker_threads_for_tests()
        mutations = {
            "success time": lambda app: setattr(
                app,
                "last_successful_translation_time",
                96.0,
            ),
            "display sequence": lambda app: setattr(
                app,
                "last_displayed_translation_sequence",
                8,
            ),
        }

        for name, mutate in mutations.items():
            with self.subTest(name=name):
                app, scheduled, displayed, metrics = self._make_app()
                self.assertTrue(
                    worker_threads._schedule_inactive_translation_clear(
                        app,
                        inactive_duration=5.0,
                        timeout_seconds=2.0,
                    )
                )
                mutate(app)

                _delay, callback, args = scheduled[0]
                callback(*args)

                self.assertEqual(displayed, [])
                self.assertIsNone(
                    getattr(
                        app,
                        "translation_inactivity_clear_scheduled_epoch",
                        None,
                    )
                )
                self.assertIsNone(
                    getattr(
                        app,
                        "translation_inactivity_cleared_epoch",
                        None,
                    )
                )
                self.assertEqual(
                    metrics.snapshot()["counters"].get(
                        "translation_inactivity_clear",
                        0,
                    ),
                    0,
                )

    def test_pending_work_appearing_before_ui_callback_cancels_clear(self):
        worker_threads = import_worker_threads_for_tests()
        app, scheduled, displayed, metrics = self._make_app()
        self.assertTrue(
            worker_threads._schedule_inactive_translation_clear(
                app,
                inactive_duration=5.0,
                timeout_seconds=2.0,
            )
        )
        app.active_translation_calls.add(42)

        _delay, callback, args = scheduled[0]
        callback(*args)

        self.assertEqual(displayed, [])
        self.assertIsNone(
            getattr(
                app,
                "translation_inactivity_clear_scheduled_epoch",
                None,
            )
        )
        self.assertIsNone(
            getattr(app, "translation_inactivity_cleared_epoch", None)
        )
        self.assertEqual(
            metrics.snapshot()["counters"].get(
                "translation_inactivity_clear",
                0,
            ),
            0,
        )


class LatencyTranslationCacheTests(unittest.TestCase):
    def test_api_ocr_cache_hit_reuses_cached_text_without_webp_or_submit(self):
        worker_threads = import_worker_threads_for_tests()
        ocr_utils = import_ocr_utils_for_tests()

        class Pool:
            def submit(self, *args):
                raise AssertionError("API OCR should not be submitted for cached frames")

        convert_calls = []
        translation_calls = []

        cache = ocr_utils.OCRFrameCache(max_size=4)
        cache_key = ocr_utils.build_ocr_frame_cache_key(
            "repeat-hash",
            "custom_ai",
            "en",
            "api",
            (320, 120),
            region_origin=(10, 20),
        )
        cache.put(cache_key, "Cached OCR text")

        screenshot = Image.new("RGB", (320, 120), (1, 2, 3))
        screenshot._gct_frame_hash = "repeat-hash"
        screenshot._gct_region_origin = (10, 20)

        app = types.SimpleNamespace(
            get_ocr_model_setting=lambda: "custom_ai",
            batch_sequence_counter=0,
            active_ocr_calls=set(),
            max_concurrent_ocr_calls=2,
            convert_to_webp_for_api=lambda _image: convert_calls.append("called") or b"webp",
            translation_model_var=types.SimpleNamespace(get=lambda: "custom_ai"),
            is_gemini_model=lambda model: False,
            is_openai_model=lambda model: False,
            source_lang_var=types.SimpleNamespace(get=lambda: "en"),
            ocr_thread_pool=Pool(),
            ocr_frame_cache=cache,
            last_displayed_batch_sequence=0,
            last_processed_subtitle=None,
            reset_clear_timeout=Mock(),
        )

        with patch.object(worker_threads, "start_async_translation", side_effect=lambda *args: translation_calls.append(args)):
            worker_threads.run_api_ocr(app, screenshot)

        self.assertEqual(convert_calls, [])
        self.assertEqual(translation_calls, [(app, "Cached OCR text", 1)])
        self.assertEqual(app.batch_sequence_counter, 1)
        self.assertEqual(app.active_ocr_calls, set())

    def test_api_ocr_cache_hit_is_used_even_when_concurrency_is_full(self):
        worker_threads = import_worker_threads_for_tests()
        ocr_utils = import_ocr_utils_for_tests()

        class Pool:
            def submit(self, *args):
                raise AssertionError("API OCR should not be submitted for cached frames")

        translation_calls = []
        cache = ocr_utils.OCRFrameCache(max_size=4)
        cache_key = ocr_utils.build_ocr_frame_cache_key(
            "repeat-hash",
            "custom_ai",
            "en",
            "api",
            (320, 120),
            region_origin=(10, 20),
        )
        cache.put(cache_key, "Cached OCR text under load")

        screenshot = Image.new("RGB", (320, 120), (1, 2, 3))
        screenshot._gct_frame_hash = "repeat-hash"
        screenshot._gct_region_origin = (10, 20)

        app = types.SimpleNamespace(
            get_ocr_model_setting=lambda: "custom_ai",
            batch_sequence_counter=0,
            active_ocr_calls={99},
            max_concurrent_ocr_calls=1,
            convert_to_webp_for_api=lambda _image: (_ for _ in ()).throw(
                AssertionError("Cached frames should not be converted")
            ),
            translation_model_var=types.SimpleNamespace(get=lambda: "custom_ai"),
            is_gemini_model=lambda model: False,
            is_openai_model=lambda model: False,
            source_lang_var=types.SimpleNamespace(get=lambda: "en"),
            ocr_thread_pool=Pool(),
            ocr_frame_cache=cache,
            last_displayed_batch_sequence=0,
            last_processed_subtitle=None,
            reset_clear_timeout=Mock(),
        )

        with patch.object(worker_threads, "start_async_translation", side_effect=lambda *args: translation_calls.append(args)):
            worker_threads.run_api_ocr(app, screenshot)

        self.assertEqual(translation_calls, [(app, "Cached OCR text under load", 1)])
        self.assertEqual(app.active_ocr_calls, {99})

    def test_api_ocr_cache_does_not_cross_custom_ai_ocr_profiles(self):
        worker_threads = import_worker_threads_for_tests()
        ocr_utils = import_ocr_utils_for_tests()

        profile_a = {
            "id": "profile-a",
            "base_url": "https://a.example/v1",
            "model": "vision-a",
        }
        profile_b = {
            "id": "profile-b",
            "base_url": "https://b.example/v1",
            "model": "vision-b",
        }

        def expected_model_key(profile):
            return (
                "custom_ai|ocr_profile="
                f"{profile['id']}|{profile['base_url']}|{profile['model']}"
            )

        class Pool:
            def __init__(self):
                self.submissions = []

            def submit(self, fn, *args):
                self.submissions.append((fn, args))
                return object()

        class Profiles:
            def __init__(self):
                self.active_profile = profile_b
                self.requested_kind = None

            def get_active_profile(self, kind):
                self.requested_kind = kind
                return self.active_profile

        profiles = Profiles()
        cache = ocr_utils.OCRFrameCache(max_size=4)
        stale_generic_cache_key = ocr_utils.build_ocr_frame_cache_key(
            "repeat-hash",
            "custom_ai",
            "en",
            "api",
            (320, 120),
            region_origin=(10, 20),
        )
        cache.put(stale_generic_cache_key, "Profile A cached OCR")

        screenshot = Image.new("RGB", (320, 120), (1, 2, 3))
        screenshot._gct_frame_hash = "repeat-hash"
        screenshot._gct_region_origin = (10, 20)

        convert_calls = []
        pool = Pool()
        app = types.SimpleNamespace(
            get_ocr_model_setting=lambda: "custom_ai",
            batch_sequence_counter=0,
            active_ocr_calls=set(),
            max_concurrent_ocr_calls=2,
            convert_to_webp_for_api=lambda _image: convert_calls.append("called") or b"profile-b-webp",
            translation_model_var=types.SimpleNamespace(get=lambda: "custom_ai"),
            is_gemini_model=lambda model: False,
            is_openai_model=lambda model: False,
            source_lang_var=types.SimpleNamespace(get=lambda: "en"),
            ocr_thread_pool=pool,
            ocr_frame_cache=cache,
            custom_ai_profiles=profiles,
            last_displayed_batch_sequence=0,
            last_processed_subtitle=None,
            reset_clear_timeout=Mock(),
        )

        with patch.object(worker_threads, "start_async_translation"):
            worker_threads.run_api_ocr(app, screenshot)

        self.assertEqual(profiles.requested_kind, "ocr")
        self.assertEqual(convert_calls, ["called"])
        self.assertEqual(len(pool.submissions), 1)
        self.assertEqual(app.active_ocr_calls, {1})

    def test_api_ocr_cache_does_not_cross_keep_linebreaks_mode(self):
        worker_threads = import_worker_threads_for_tests()
        ocr_utils = import_ocr_utils_for_tests()

        class Pool:
            def __init__(self):
                self.submissions = []

            def submit(self, fn, *args):
                self.submissions.append((fn, args))
                return object()

        cache = ocr_utils.OCRFrameCache(max_size=4)
        stale_single_line_key = ocr_utils.build_ocr_frame_cache_key(
            "repeat-hash",
            "custom_ai",
            "en",
            "api",
            (320, 120),
            region_origin=(10, 20),
        )
        cache.put(stale_single_line_key, "Cached without line breaks")

        screenshot = Image.new("RGB", (320, 120), (1, 2, 3))
        screenshot._gct_frame_hash = "repeat-hash"
        screenshot._gct_region_origin = (10, 20)

        convert_calls = []
        pool = Pool()
        app = types.SimpleNamespace(
            get_ocr_model_setting=lambda: "custom_ai",
            batch_sequence_counter=0,
            active_ocr_calls=set(),
            max_concurrent_ocr_calls=2,
            convert_to_webp_for_api=lambda _image: convert_calls.append("called") or b"linebreak-webp",
            translation_model_var=types.SimpleNamespace(get=lambda: "custom_ai"),
            is_gemini_model=lambda model: False,
            is_openai_model=lambda model: False,
            source_lang_var=types.SimpleNamespace(get=lambda: "en"),
            keep_linebreaks_var=types.SimpleNamespace(get=lambda: True),
            ocr_thread_pool=pool,
            ocr_frame_cache=cache,
        )

        worker_threads.run_api_ocr(app, screenshot)

        self.assertEqual(convert_calls, ["called"])
        self.assertEqual(len(pool.submissions), 1)
        self.assertEqual(app.active_ocr_calls, {1})

    def test_api_ocr_cache_does_not_cross_image_payload_settings(self):
        worker_threads = import_worker_threads_for_tests()
        ocr_utils = import_ocr_utils_for_tests()

        class Pool:
            def __init__(self):
                self.submissions = []

            def submit(self, fn, *args):
                self.submissions.append((fn, args))
                return object()

        cache = ocr_utils.OCRFrameCache(max_size=4)
        stale_default_payload_key = ocr_utils.build_ocr_frame_cache_key(
            "repeat-hash",
            "custom_ai",
            "en",
            "api|keep_linebreaks=False",
            (320, 120),
            region_origin=(10, 20),
        )
        cache.put(stale_default_payload_key, "Cached with different payload settings")

        screenshot = Image.new("RGB", (320, 120), (1, 2, 3))
        screenshot._gct_frame_hash = "repeat-hash"
        screenshot._gct_region_origin = (10, 20)

        convert_calls = []
        pool = Pool()
        app = types.SimpleNamespace(
            get_ocr_model_setting=lambda: "custom_ai",
            batch_sequence_counter=0,
            active_ocr_calls=set(),
            max_concurrent_ocr_calls=2,
            convert_to_webp_for_api=lambda _image: convert_calls.append("called") or b"balanced-webp",
            translation_model_var=types.SimpleNamespace(get=lambda: "custom_ai"),
            is_gemini_model=lambda model: False,
            is_openai_model=lambda model: False,
            source_lang_var=types.SimpleNamespace(get=lambda: "en"),
            keep_linebreaks_var=types.SimpleNamespace(get=lambda: False),
            get_ai_ocr_image_decision=lambda image_size=None: types.SimpleNamespace(
                contract_key="webp|balanced_webp|85|auto"
            ),
            ocr_thread_pool=pool,
            ocr_frame_cache=cache,
        )

        with patch.object(worker_threads, "start_async_translation"):
            worker_threads.run_api_ocr(app, screenshot)

        self.assertEqual(convert_calls, ["called"])
        self.assertEqual(len(pool.submissions), 1)
        self.assertEqual(app.active_ocr_calls, {1})

    def test_api_ocr_cache_mode_key_includes_image_payload_settings(self):
        worker_threads = import_worker_threads_for_tests()
        app = types.SimpleNamespace(
            keep_linebreaks_var=types.SimpleNamespace(get=lambda: True),
            get_ai_ocr_image_decision=lambda image_size=None: types.SimpleNamespace(
                contract_key="png|lossless_webp|100|high"
            ),
        )

        cache_mode_key = worker_threads._get_api_ocr_cache_mode_key(
            app,
            image_size=(320, 120),
        )

        self.assertIn("keep_linebreaks=True", cache_mode_key)
        self.assertIn(
            "image_contract=png|lossless_webp|100|high",
            cache_mode_key,
        )

    def test_api_ocr_cache_mode_key_includes_effective_reasoning_contract(self):
        worker_threads = import_worker_threads_for_tests()
        profile = {
            "id": "ocr-profile",
            "base_url": "https://provider.example/v1",
            "model": "vision",
            "reasoning_effort": "low",
        }

        class Profiles:
            def get_active_profile(self, kind):
                return profile

        class Provider:
            def reasoning_effort_request_contract(self, active_profile, request_kind):
                self.request_kind = request_kind
                return active_profile["reasoning_effort"]

        provider = Provider()
        app = types.SimpleNamespace(
            custom_ai_profiles=Profiles(),
            translation_handler=types.SimpleNamespace(custom_ai_provider=provider),
            keep_linebreaks_var=types.SimpleNamespace(get=lambda: False),
        )

        low_key = worker_threads._get_api_ocr_cache_mode_key(app)
        profile["reasoning_effort"] = "none"
        none_key = worker_threads._get_api_ocr_cache_mode_key(app)

        self.assertIn("reasoning_effort=low", low_key)
        self.assertIn("reasoning_effort=none", none_key)
        self.assertNotEqual(low_key, none_key)
        self.assertEqual(provider.request_kind, "ocr")

    def test_convert_to_webp_for_api_logs_payload_metadata_without_base64(self):
        import app_logic

        log_messages = []
        image = Image.new("RGB", (48, 24), (250, 250, 250))
        app = types.SimpleNamespace(
            get_ai_ocr_image_decision=lambda image_size=None: types.SimpleNamespace(
                image_format="webp",
                image_mode="small_grayscale_webp",
                image_quality=72,
                image_detail="low",
                reason="speed_policy",
                contract_key="webp|small_grayscale_webp|72|low",
            ),
        )

        with patch.object(app_logic, "log_debug", side_effect=log_messages.append):
            encoded = app_logic.GameChangingTranslator.convert_to_webp_for_api(app, image)

        encoded_base64 = base64.b64encode(encoded).decode("ascii")
        log_text = "\n".join(log_messages)
        self.assertIn("mode=small_grayscale_webp", log_text)
        self.assertIn("bytes=", log_text)
        self.assertIn("detail=low", log_text)
        self.assertIn("policy_reason=speed_policy", log_text)
        self.assertIn("duration=", log_text)
        self.assertNotIn(encoded_base64, log_text)
        self.assertNotIn("data:image", log_text)

    def test_ai_ocr_image_decision_uses_recent_route_latency_and_active_profile(self):
        import app_logic
        from ai_optimization import AiOcrImageCapabilityMemory

        profile = {
            "base_url": "https://relay.example/v1",
            "model": "vision-model",
        }
        app = types.SimpleNamespace(
            ai_optimization_mode_var=types.SimpleNamespace(get=lambda: "auto"),
            custom_ai_profiles=types.SimpleNamespace(
                get_active_profile=lambda kind: profile
            ),
            runtime_metrics=types.SimpleNamespace(
                snapshot=lambda: {
                    "timings": {
                        "ocr_duration": {
                            "count": 8,
                            "p90": 4.2,
                        }
                    }
                }
            ),
            ai_ocr_image_capability_memory=AiOcrImageCapabilityMemory(),
        )

        decision = app_logic.GameChangingTranslator.get_ai_ocr_image_decision(
            app,
            image_size=(800, 300),
        )

        self.assertEqual(decision.image_format, "webp")
        self.assertEqual(decision.image_mode, "small_grayscale_webp")
        self.assertEqual(decision.image_quality, 75)
        self.assertEqual(decision.image_detail, "low")
        self.assertEqual(decision.reason, "auto_slow_route")

    def test_custom_ai_api_ocr_uses_custom_source_language(self):
        worker_threads = import_worker_threads_for_tests()

        class Pool:
            def __init__(self):
                self.submissions = []

            def submit(self, fn, *args):
                self.submissions.append((fn, args))
                return object()

        screenshot = Image.new("RGB", (320, 120), (1, 2, 3))
        pool = Pool()
        app = types.SimpleNamespace(
            get_ocr_model_setting=lambda: "custom_ai",
            batch_sequence_counter=0,
            active_ocr_calls=set(),
            max_concurrent_ocr_calls=2,
            convert_to_webp_for_api=lambda _image: b"webp",
            translation_model_var=types.SimpleNamespace(get=lambda: "gemini_api"),
            is_gemini_model=lambda model: model == "gemini_api",
            is_openai_model=lambda model: False,
            gemini_source_lang="de",
            source_lang_var=types.SimpleNamespace(get=lambda: "en"),
            custom_source_lang="ja",
            ocr_thread_pool=pool,
        )

        worker_threads.run_api_ocr(app, screenshot)

        self.assertEqual(len(pool.submissions), 1)
        _fn, args = pool.submissions[0]
        self.assertEqual(args[2], "ja")

    def test_api_ocr_submission_carries_encoded_image_mime_type(self):
        worker_threads = import_worker_threads_for_tests()

        class Pool:
            def __init__(self):
                self.submissions = []

            def submit(self, fn, *args):
                self.submissions.append((fn, args))
                return object()

        encoded_image = types.SimpleNamespace(
            data=b"png-bytes",
            mime_type="image/png",
            image_format="png",
        )
        pool = Pool()
        app = types.SimpleNamespace(
            get_ocr_model_setting=lambda: "custom_ai",
            batch_sequence_counter=0,
            active_ocr_calls=set(),
            max_concurrent_ocr_calls=2,
            convert_to_api_ocr_image=lambda _image: encoded_image,
            convert_to_webp_for_api=lambda _image: (_ for _ in ()).throw(
                AssertionError("metadata-aware encoder should be preferred")
            ),
            translation_model_var=types.SimpleNamespace(get=lambda: "custom_ai"),
            is_gemini_model=lambda model: False,
            is_openai_model=lambda model: False,
            source_lang_var=types.SimpleNamespace(get=lambda: "en"),
            custom_source_lang="en",
            ocr_thread_pool=pool,
        )

        worker_threads.run_api_ocr(app, Image.new("RGB", (24, 12), (1, 2, 3)))

        self.assertEqual(len(pool.submissions), 1)
        _fn, args = pool.submissions[0]
        self.assertEqual(args[1], b"png-bytes")
        self.assertEqual(args[6], "image/png")

    def test_api_ocr_skips_webp_conversion_when_concurrency_limit_is_full(self):
        worker_threads = import_worker_threads_for_tests()

        class Pool:
            def submit(self, *args):
                raise AssertionError("API OCR should not be submitted when concurrency is full")

        convert_calls = []

        def convert_to_webp_for_api(_image):
            convert_calls.append("called")
            return b"webp"

        app = types.SimpleNamespace(
            get_ocr_model_setting=lambda: "custom_ai",
            batch_sequence_counter=7,
            active_ocr_calls={1},
            max_concurrent_ocr_calls=1,
            convert_to_webp_for_api=convert_to_webp_for_api,
            translation_model_var=types.SimpleNamespace(get=lambda: "custom_ai"),
            is_gemini_model=lambda model: False,
            is_openai_model=lambda model: False,
            source_lang_var=types.SimpleNamespace(get=lambda: "en"),
            ocr_thread_pool=Pool(),
        )

        worker_threads.run_api_ocr(app, object())

        self.assertEqual(convert_calls, [])
        self.assertEqual(app.batch_sequence_counter, 7)

    def test_custom_ai_ocr_cooldown_uses_paddleocr_for_current_frame(self):
        worker_threads = import_worker_threads_for_tests()
        app = types.SimpleNamespace(
            translation_handler=types.SimpleNamespace(
                get_active_custom_ai_ocr_cooldown_seconds=lambda: 42.0
            )
        )

        self.assertEqual(
            worker_threads._effective_ocr_model_for_frame(app, "custom_ai"),
            "paddleocr",
        )

    def test_custom_ai_ocr_recovers_automatically_after_cooldown(self):
        worker_threads = import_worker_threads_for_tests()
        app = types.SimpleNamespace(
            translation_handler=types.SimpleNamespace(
                get_active_custom_ai_ocr_cooldown_seconds=lambda: 0.0
            )
        )

        self.assertEqual(
            worker_threads._effective_ocr_model_for_frame(app, "custom_ai"),
            "custom_ai",
        )

    def test_custom_ai_ocr_uses_provider_specific_concurrency_limit(self):
        worker_threads = import_worker_threads_for_tests()

        class Pool:
            def submit(self, *args):
                raise AssertionError("Custom AI OCR should stop at its own limit")

        convert_calls = []
        app = types.SimpleNamespace(
            get_ocr_model_setting=lambda: "custom_ai",
            batch_sequence_counter=7,
            active_ocr_calls={1, 2},
            max_concurrent_ocr_calls=8,
            convert_to_webp_for_api=lambda _image: convert_calls.append("called"),
            translation_model_var=types.SimpleNamespace(get=lambda: "custom_ai"),
            is_gemini_model=lambda model: False,
            is_openai_model=lambda model: False,
            source_lang_var=types.SimpleNamespace(get=lambda: "en"),
            ocr_thread_pool=Pool(),
        )

        worker_threads.run_api_ocr(app, object())

        self.assertEqual(convert_calls, [])
        self.assertEqual(app.batch_sequence_counter, 7)

    def test_other_api_ocr_keeps_generic_concurrency_limit(self):
        worker_threads = import_worker_threads_for_tests()

        class Pool:
            def __init__(self):
                self.submissions = []

            def submit(self, fn, *args):
                self.submissions.append((fn, args))
                return object()

        pool = Pool()
        app = types.SimpleNamespace(
            get_ocr_model_setting=lambda: "gemini_api",
            batch_sequence_counter=7,
            active_ocr_calls={1, 2},
            max_concurrent_ocr_calls=8,
            convert_to_webp_for_api=lambda _image: b"webp",
            translation_model_var=types.SimpleNamespace(get=lambda: "gemini_api"),
            is_gemini_model=lambda model: model == "gemini_api",
            is_openai_model=lambda model: False,
            gemini_source_lang="de",
            source_lang_var=types.SimpleNamespace(get=lambda: "en"),
            ocr_thread_pool=pool,
        )

        worker_threads.run_api_ocr(app, object())

        self.assertEqual(len(pool.submissions), 1)
        self.assertEqual(app.batch_sequence_counter, 8)

    def test_cooling_custom_ai_ocr_error_preserves_current_translation(self):
        worker_threads = import_worker_threads_for_tests()
        app = types.SimpleNamespace(
            last_displayed_batch_sequence=0,
            update_translation_text=Mock(),
            translation_handler=types.SimpleNamespace(
                get_active_custom_ai_ocr_cooldown_seconds=lambda: 30.0
            ),
        )

        worker_threads.process_api_ocr_response(
            app,
            "<e>: provider cooling down",
            3,
            "en",
            "custom_ai",
        )

        app.update_translation_text.assert_not_called()
        self.assertEqual(app.last_displayed_batch_sequence, 3)

    def test_non_cooling_custom_ai_ocr_error_remains_visible(self):
        worker_threads = import_worker_threads_for_tests()
        app = types.SimpleNamespace(
            last_displayed_batch_sequence=0,
            update_translation_text=Mock(),
            translation_handler=types.SimpleNamespace(
                get_active_custom_ai_ocr_cooldown_seconds=lambda: 0.0
            ),
        )

        worker_threads.process_api_ocr_response(
            app,
            "<e>: provider failed",
            3,
            "en",
            "custom_ai",
        )

        app.update_translation_text.assert_called_once_with(
            "OCR Error:\nprovider failed"
        )
        self.assertEqual(app.last_displayed_batch_sequence, 3)

    def test_api_ocr_submit_failure_releases_active_call_slot(self):
        worker_threads = import_worker_threads_for_tests()

        class Pool:
            def submit(self, *args):
                raise RuntimeError("thread pool stopped")

        screenshot = Image.new("RGB", (320, 120), (1, 2, 3))
        app = types.SimpleNamespace(
            get_ocr_model_setting=lambda: "custom_ai",
            batch_sequence_counter=0,
            active_ocr_calls=set(),
            max_concurrent_ocr_calls=2,
            convert_to_webp_for_api=lambda _image: b"webp",
            translation_model_var=types.SimpleNamespace(get=lambda: "custom_ai"),
            is_gemini_model=lambda model: False,
            is_openai_model=lambda model: False,
            source_lang_var=types.SimpleNamespace(get=lambda: "en"),
            ocr_thread_pool=Pool(),
        )

        worker_threads.run_api_ocr(app, screenshot)

        self.assertEqual(app.active_ocr_calls, set())

    def test_stale_api_ocr_request_skips_provider_call(self):
        worker_threads = import_worker_threads_for_tests()
        perform_calls = []
        scheduled = []

        class Handler:
            def perform_ocr(self, image_data, source_lang):
                perform_calls.append((image_data, source_lang))
                return "older text"

        app = types.SimpleNamespace(
            is_running=True,
            batch_sequence_counter=5,
            active_ocr_calls={4},
            translation_handler=Handler(),
            root=types.SimpleNamespace(after=lambda *args: scheduled.append(args)),
        )

        worker_threads.process_api_ocr_async(app, b"older-webp", "en", 4, "custom_ai")

        self.assertEqual(perform_calls, [])
        self.assertEqual(scheduled, [])
        self.assertEqual(app.active_ocr_calls, set())

    def test_successful_api_ocr_response_is_cached_for_frame_reuse(self):
        worker_threads = import_worker_threads_for_tests()
        ocr_utils = import_ocr_utils_for_tests()
        translation_calls = []

        cache = ocr_utils.OCRFrameCache(max_size=4)
        cache_key = ocr_utils.build_ocr_frame_cache_key(
            "fresh-hash",
            "custom_ai",
            "en",
            "api",
            (320, 120),
            region_origin=(10, 20),
        )

        app = types.SimpleNamespace(
            ocr_frame_cache=cache,
            last_displayed_batch_sequence=0,
            last_processed_subtitle=None,
            reset_clear_timeout=Mock(),
        )

        with patch.object(worker_threads, "start_async_translation", side_effect=lambda *args: translation_calls.append(args)):
            worker_threads.process_api_ocr_response(
                app,
                "Fresh OCR text",
                3,
                "en",
                "custom_ai",
                ocr_cache_key=cache_key,
            )

        self.assertEqual(cache.get(cache_key), "Fresh OCR text")
        self.assertEqual(translation_calls, [(app, "Fresh OCR text", 3)])

    def test_custom_ai_cache_hit_can_be_displayed_without_thread_pool(self):
        TranslationHandler = import_translation_handler_for_tests()

        profile = {
            "id": "profile-1",
            "base_url": "https://example.test/v1",
            "model": "fast-model",
        }
        app = types.SimpleNamespace(
            translation_model_var=types.SimpleNamespace(get=lambda: "custom_ai"),
            custom_ai_profiles=types.SimpleNamespace(get_active_profile=lambda kind: profile),
            custom_source_lang="ja",
            custom_target_lang="en",
            source_lang_var=types.SimpleNamespace(get=lambda: "ja"),
            target_lang_var=types.SimpleNamespace(get=lambda: "en"),
            keep_linebreaks_var=types.SimpleNamespace(get=lambda: False),
        )
        handler = TranslationHandler(app)
        cache_params = handler._cache_params_for_profile(profile)
        handler.unified_cache.store(
            "こんにちは",
            "ja",
            "en",
            "custom_ai",
            "Hello",
            **cache_params,
        )

        self.assertEqual(handler.get_cached_translation_for_display("こんにちは"), "Hello")

    def test_instant_cache_hit_invalidates_older_pending_translation(self):
        worker_threads = import_worker_threads_for_tests()
        scheduled = []
        displayed = []

        class Handler:
            def get_cached_translation_for_display(self, text):
                return "Cached latest"

        app = types.SimpleNamespace(
            translation_sequence_counter=4,
            latest_translation_sequence_started=3,
            last_displayed_translation_sequence=3,
            active_translation_calls=set(),
            active_translation_inflight_keys=set(),
            active_translation_started_monotonic={},
            translation_thread_pool=Mock(),
            translation_handler=Handler(),
            enable_instant_cache_display_var=types.SimpleNamespace(
                get=lambda: True
            ),
            root=types.SimpleNamespace(
                after=lambda delay, callback, *args: scheduled.append(
                    (delay, callback, args)
                )
            ),
            initialize_async_translation_infrastructure=lambda: None,
            update_translation_text=displayed.append,
            pending_translation_request=None,
            pending_translation_flush_scheduled=False,
            pending_translation_flush_deadline_monotonic=0.0,
            pending_translation_flush_generation=0,
            last_successful_translation_time=0.0,
            is_running=True,
        )

        with patch.object(worker_threads.time, "monotonic", return_value=100.0):
            worker_threads._queue_pending_translation_request(
                app,
                "Older uncached text",
                1,
                5.0,
                "active translation",
            )

        old_generation = app.pending_translation_flush_generation
        old_timer = scheduled[0]

        with patch.object(worker_threads.time, "monotonic", return_value=100.1):
            worker_threads.start_async_translation(
                app,
                "Newest cached text",
                2,
            )

        self.assertIsNone(app.pending_translation_request)
        self.assertFalse(app.pending_translation_flush_scheduled)
        self.assertEqual(app.pending_translation_flush_deadline_monotonic, 0.0)
        self.assertEqual(
            app.pending_translation_flush_generation,
            old_generation + 1,
        )
        self.assertEqual(displayed, ["Cached latest"])
        self.assertEqual(app.translation_sequence_counter, 5)
        self.assertEqual(app.last_displayed_translation_sequence, 5)
        self.assertEqual(app.latest_translation_sequence_started, 5)
        app.translation_thread_pool.submit.assert_not_called()

        with patch.object(
            worker_threads,
            "start_async_translation",
        ) as restart_translation:
            old_timer[1](*old_timer[2])

        restart_translation.assert_not_called()
        self.assertIsNone(app.pending_translation_request)

        worker_threads.process_translation_response(
            app,
            "Obsolete network result",
            4,
            "Older uncached text",
            1,
        )
        self.assertEqual(displayed, ["Cached latest"])

    def test_duplicate_inflight_translation_is_not_submitted_twice(self):
        worker_threads = import_worker_threads_for_tests()

        class Pool:
            def __init__(self):
                self.submissions = []

            def submit(self, fn, *args):
                self.submissions.append((fn, args))
                return object()

        class Handler:
            def get_cached_translation_for_display(self, text):
                return None

            def get_inflight_translation_key(self, text):
                return ("custom_ai", text, "same-context")

            def translate_text_with_timeout(self, text, timeout_seconds=10.0, ocr_batch_number=None):
                return "translated"

        pool = Pool()
        app = types.SimpleNamespace(
            translation_sequence_counter=0,
            active_translation_calls=set(),
            max_concurrent_translation_calls=6,
            translation_thread_pool=pool,
            translation_handler=Handler(),
            enable_instant_cache_display_var=types.SimpleNamespace(get=lambda: False),
            root=types.SimpleNamespace(after=lambda *args, **kwargs: None),
            initialize_async_translation_infrastructure=lambda: None,
        )

        worker_threads.start_async_translation(app, "Hello", 1)
        worker_threads.start_async_translation(app, "Hello", 2)

        self.assertEqual(len(pool.submissions), 1)
        self.assertEqual(app.active_translation_inflight_keys, {("custom_ai", "Hello", "same-context")})

        fn, args = pool.submissions[0]
        fn(*args)

        self.assertEqual(app.active_translation_inflight_keys, set())

    def test_duplicate_inflight_translation_is_not_submitted_twice_when_latency_mode_is_none(self):
        worker_threads = import_worker_threads_for_tests()

        class Pool:
            def __init__(self):
                self.submissions = []

            def submit(self, fn, *args):
                self.submissions.append((fn, args))
                return object()

        class Handler:
            def get_cached_translation_for_display(self, text):
                return None

            def get_inflight_translation_key(self, text):
                return ("custom_ai", text, "same-context")

            def translate_text_with_timeout(self, text, timeout_seconds=10.0, ocr_batch_number=None):
                return "translated"

        pool = Pool()
        app = types.SimpleNamespace(
            translation_sequence_counter=0,
            active_translation_calls=set(),
            active_translation_inflight_keys=set(),
            max_concurrent_translation_calls=6,
            translation_thread_pool=pool,
            translation_handler=Handler(),
            enable_instant_cache_display_var=types.SimpleNamespace(get=lambda: False),
            custom_ai_latency_mode_var=types.SimpleNamespace(get=lambda: "none"),
            root=types.SimpleNamespace(after=lambda *args, **kwargs: None),
            initialize_async_translation_infrastructure=lambda: None,
        )

        worker_threads.start_async_translation(app, "Hello", 1)
        worker_threads.start_async_translation(app, "Hello", 2)

        self.assertEqual(len(pool.submissions), 1)
        self.assertEqual(app.active_translation_inflight_keys, {("custom_ai", "Hello", "same-context")})

    def test_start_async_translation_passes_resolved_latency_snapshot_to_worker(self):
        worker_threads = import_worker_threads_for_tests()

        class Pool:
            def __init__(self):
                self.submissions = []

            def submit(self, fn, *args):
                self.submissions.append((fn, args))
                return object()

        class ModeVar:
            value = "adaptive"

            def get(self):
                return self.value

        class Handler:
            def __init__(self):
                self.snapshot_requests = []
                self.committed_modes = []
                self.cooldown_modes = []
                self.translate_modes = []
                self.translate_timeouts = []
                self.translate_snapshots = []

            def get_cached_translation_for_display(self, text):
                return None

            def get_inflight_translation_key(self, text):
                return ("custom_ai", text, "configured-adaptive")

            def get_custom_ai_translation_request_snapshot(
                self,
                text,
                commit=False,
            ):
                self.snapshot_requests.append((text, commit))
                return {
                    "inflight_key": ("custom_ai", text, "resolved-stream"),
                    "latency_mode": "stream",
                    "reason": "p90_high",
                    "timeout_seconds": 4.0,
                    "timeout_reason": "fast_route_tail_guard",
                    "profile": {"id": "relay", "model": "original"},
                    "cache_params": {"model": "original"},
                }

            def commit_custom_ai_latency_mode_snapshot(self, snapshot):
                self.committed_modes.append(snapshot["latency_mode"])

            def get_translation_submit_interval_seconds(self, text_content=None):
                return 0.0

            def get_translation_provider_cooldown_seconds(self, latency_mode=None):
                self.cooldown_modes.append(latency_mode)
                return 0.0

            def translate_text_with_timeout(
                self,
                text,
                timeout_seconds=10.0,
                ocr_batch_number=None,
                stream_callback=None,
                translation_sequence=None,
                latency_mode=None,
                request_snapshot=None,
            ):
                self.translate_modes.append(latency_mode)
                self.translate_timeouts.append(timeout_seconds)
                self.translate_snapshots.append(request_snapshot)
                if stream_callback:
                    stream_callback("translated")
                return "translated"

        pool = Pool()
        mode_var = ModeVar()
        handler = Handler()
        scheduled = []
        displayed = []
        app = types.SimpleNamespace(
            translation_sequence_counter=0,
            active_translation_calls=set(),
            active_translation_inflight_keys=set(),
            active_translation_started_monotonic={},
            max_concurrent_translation_calls=6,
            translation_thread_pool=pool,
            translation_handler=handler,
            enable_instant_cache_display_var=types.SimpleNamespace(get=lambda: False),
            custom_ai_latency_mode_var=mode_var,
            root=types.SimpleNamespace(
                after=lambda delay, callback, *args: scheduled.append(
                    (delay, callback, args)
                )
            ),
            initialize_async_translation_infrastructure=lambda: None,
            last_translation_submit_monotonic=0.0,
            pending_translation_request=None,
            is_running=True,
            latest_translation_sequence_started=1,
            last_displayed_translation_sequence=0,
            update_translation_text=lambda text: displayed.append(text),
            last_successful_translation_time=0,
        )

        worker_threads.start_async_translation(app, "Hello", 1)
        mode_var.value = "safe"

        self.assertEqual(handler.snapshot_requests, [("Hello", False)])
        self.assertEqual(handler.cooldown_modes, ["stream"])
        self.assertEqual(handler.committed_modes, ["stream"])
        self.assertEqual(len(pool.submissions), 1)

        fn, args = pool.submissions[0]
        fn(*args)

        self.assertEqual(handler.translate_modes, ["stream"])
        self.assertEqual(
            handler.translate_timeouts,
            [10.0],
            "streaming requests must keep the full socket read deadline",
        )
        self.assertIsNotNone(
            handler.translate_snapshots[0],
            "the resolved request snapshot must reach the worker handler",
        )
        self.assertEqual(
            handler.translate_snapshots[0]["profile"]["model"],
            "original",
        )
        self.assertEqual(app.active_translation_inflight_keys, set())

    def test_request_snapshot_timeout_is_validated_by_worker(self):
        worker_threads = import_worker_threads_for_tests()
        resolver = getattr(
            worker_threads,
            "_request_snapshot_timeout_seconds",
            None,
        )
        self.assertTrue(
            callable(resolver),
            "the worker needs one boundary for snapshot timeout validation",
        )

        cases = (
            ("safe", {"timeout_seconds": 4.0}, 4.0),
            ("safe", {"timeout_seconds": "invalid"}, 10.0),
            ("safe", {}, 10.0),
            ("stream", {"timeout_seconds": 4.0}, 10.0),
            ("race", {"timeout_seconds": 4.0}, 10.0),
        )
        for mode, snapshot, expected in cases:
            with self.subTest(mode=mode, snapshot=snapshot):
                self.assertEqual(resolver(snapshot, mode), expected)

    def test_process_translation_async_passes_safe_snapshot_timeout(self):
        worker_threads = import_worker_threads_for_tests()
        received_timeouts = []
        scheduled = []

        class Handler:
            def translate_text_with_timeout(
                self,
                _text,
                timeout_seconds=10.0,
                **_kwargs,
            ):
                received_timeouts.append(timeout_seconds)
                return "translated"

        app = types.SimpleNamespace(
            root=types.SimpleNamespace(
                after=lambda delay, callback, *args: scheduled.append(
                    (delay, callback, args)
                )
            ),
            translation_handler=Handler(),
            active_translation_calls={1},
            active_translation_inflight_keys={("request",)},
            active_translation_started_monotonic={1: time.monotonic()},
            pending_translation_request=None,
        )

        worker_threads.process_translation_async(
            app,
            "Hello",
            1,
            0,
            inflight_key=("request",),
            latency_mode="safe",
            request_snapshot={"timeout_seconds": 4.0},
        )

        self.assertEqual(received_timeouts, [4.0])
        self.assertEqual(len(scheduled), 1)

    def test_start_async_translation_queues_latest_request_during_submit_cooldown(self):
        worker_threads = import_worker_threads_for_tests()
        scheduled = []

        class Pool:
            def __init__(self):
                self.submissions = []

            def submit(self, fn, *args):
                self.submissions.append((fn, args))
                return object()

        class Handler:
            def get_cached_translation_for_display(self, text):
                return None

            def get_inflight_translation_key(self, text):
                return ("custom_ai", text, "same-context")

            def get_translation_submit_interval_seconds(self, text_content=None):
                return 1.0

            def get_translation_provider_cooldown_seconds(self):
                return 0.0

        pool = Pool()
        app = types.SimpleNamespace(
            translation_sequence_counter=0,
            active_translation_calls=set(),
            active_translation_inflight_keys=set(),
            max_concurrent_translation_calls=6,
            translation_thread_pool=pool,
            translation_handler=Handler(),
            enable_instant_cache_display_var=types.SimpleNamespace(get=lambda: False),
            custom_ai_latency_mode_var=types.SimpleNamespace(get=lambda: "none"),
            root=types.SimpleNamespace(after=lambda delay, callback, *args: scheduled.append((delay, callback, args))),
            initialize_async_translation_infrastructure=lambda: None,
            last_translation_submit_monotonic=100.0,
            pending_translation_request=None,
            pending_translation_flush_scheduled=False,
        )

        with patch.object(worker_threads.time, "monotonic", return_value=100.2):
            worker_threads.start_async_translation(app, "First", 1)
            worker_threads.start_async_translation(app, "Second", 2)

        self.assertEqual(len(pool.submissions), 0)
        self.assertEqual(app.pending_translation_request["text"], "Second")
        self.assertEqual(len(scheduled), 1)

        delay, callback, args = scheduled[0]
        self.assertGreaterEqual(delay, 1)

        with patch.object(worker_threads.time, "monotonic", return_value=101.5):
            callback(*args)

        self.assertEqual(len(pool.submissions), 1)
        self.assertEqual(pool.submissions[0][1][1], "Second")

    def test_stale_active_translation_can_be_superseded_once(self):
        worker_threads = import_worker_threads_for_tests()

        class Pool:
            def __init__(self):
                self.submissions = []

            def submit(self, fn, *args):
                self.submissions.append((fn, args))
                return object()

        class Handler:
            def get_cached_translation_for_display(self, text):
                return None

            def get_inflight_translation_key(self, text):
                return ("custom_ai", text, "scope")

            def get_translation_submit_interval_seconds(self, text_content=None):
                return 0.0

            def get_translation_provider_cooldown_seconds(self):
                return 0.0

            def get_translation_concurrency_limit(self):
                return 1

        pool = Pool()
        app = types.SimpleNamespace(
            translation_sequence_counter=1,
            active_translation_calls={1},
            active_translation_inflight_keys={("custom_ai", "Old", "scope")},
            active_translation_started_monotonic={1: 100.0},
            translation_thread_pool=pool,
            translation_handler=Handler(),
            enable_instant_cache_display_var=types.SimpleNamespace(get=lambda: False),
            root=types.SimpleNamespace(after=lambda *args, **kwargs: None),
            initialize_async_translation_infrastructure=lambda: None,
            last_translation_submit_monotonic=100.0,
        )

        with patch.object(worker_threads.time, "monotonic", return_value=101.6):
            worker_threads.start_async_translation(app, "Latest", 2)

        self.assertEqual(len(pool.submissions), 1)
        self.assertEqual(app.active_translation_calls, {1, 2})
        self.assertEqual(
            app.active_translation_started_monotonic,
            {1: 100.0, 2: 101.6},
        )

    def test_translation_supersede_threshold_uses_route_p90(self):
        worker_threads = import_worker_threads_for_tests()
        app = types.SimpleNamespace()
        resolver = worker_threads._get_translation_supersede_after_seconds
        snapshot = {
            "route_p90_seconds": 8.0,
            "route_sample_count": 8,
        }
        threshold = (
            resolver(app, snapshot)
            if "request_snapshot" in inspect.signature(resolver).parameters
            else resolver(app)
        )

        self.assertEqual(threshold, 4.0)

    def test_translation_supersede_threshold_caps_slow_route_p90(self):
        worker_threads = import_worker_threads_for_tests()
        app = types.SimpleNamespace()
        resolver = worker_threads._get_translation_supersede_after_seconds
        snapshot = {
            "route_p90_seconds": 20.0,
            "route_sample_count": 8,
        }
        threshold = (
            resolver(app, snapshot)
            if "request_snapshot" in inspect.signature(resolver).parameters
            else resolver(app)
        )

        self.assertEqual(threshold, 4.0)

    def test_translation_supersede_threshold_requires_enough_samples(self):
        worker_threads = import_worker_threads_for_tests()
        app = types.SimpleNamespace()
        resolver = worker_threads._get_translation_supersede_after_seconds
        snapshot = {
            "route_p90_seconds": 4.0,
            "route_sample_count": 7,
        }
        threshold = (
            resolver(app, snapshot)
            if "request_snapshot" in inspect.signature(resolver).parameters
            else resolver(app)
        )

        self.assertEqual(threshold, 1.5)

    def test_route_p90_keeps_young_active_translation_queued(self):
        worker_threads = import_worker_threads_for_tests()
        scheduled = []

        class Handler:
            def get_cached_translation_for_display(self, text):
                return None

            def get_custom_ai_translation_request_snapshot(
                self,
                text,
                commit=False,
            ):
                return {
                    "inflight_key": ("custom_ai", text, "scope"),
                    "latency_mode": "safe",
                    "route_p90_seconds": 8.0,
                    "route_sample_count": 8,
                }

            def get_translation_submit_interval_seconds(self, text_content=None):
                return 0.0

            def get_translation_provider_cooldown_seconds(self, **_kwargs):
                return 0.0

            def get_translation_concurrency_limit(self):
                return 1

        app = types.SimpleNamespace(
            translation_sequence_counter=1,
            active_translation_calls={1},
            active_translation_inflight_keys={
                ("custom_ai", "Old", "scope")
            },
            active_translation_started_monotonic={1: 100.0},
            translation_thread_pool=Mock(),
            translation_handler=Handler(),
            enable_instant_cache_display_var=types.SimpleNamespace(
                get=lambda: False
            ),
            root=types.SimpleNamespace(
                after=lambda delay, callback, *args: scheduled.append(
                    (delay, callback, args)
                )
            ),
            initialize_async_translation_infrastructure=lambda: None,
            last_translation_submit_monotonic=100.0,
            pending_translation_request=None,
            pending_translation_flush_scheduled=False,
            pending_translation_flush_deadline_monotonic=0.0,
            pending_translation_flush_generation=0,
        )

        with patch.object(worker_threads.time, "monotonic", return_value=102.0):
            worker_threads.start_async_translation(app, "Latest", 2)

        app.translation_thread_pool.submit.assert_not_called()
        self.assertEqual(app.pending_translation_request["text"], "Latest")
        self.assertEqual(len(scheduled), 1)
        self.assertGreaterEqual(scheduled[0][0], 1999)
        self.assertLessEqual(scheduled[0][0], 2001)

    def test_stale_translation_supersession_never_exceeds_two_active_calls(self):
        worker_threads = import_worker_threads_for_tests()
        scheduled = []

        class Pool:
            def __init__(self):
                self.submissions = []

            def submit(self, fn, *args):
                self.submissions.append((fn, args))
                return object()

        class Handler:
            def get_cached_translation_for_display(self, text):
                return None

            def get_inflight_translation_key(self, text):
                return ("custom_ai", text, "scope")

            def get_translation_submit_interval_seconds(self, text_content=None):
                return 0.0

            def get_translation_provider_cooldown_seconds(self):
                return 0.0

            def get_translation_concurrency_limit(self):
                return 1

        pool = Pool()
        app = types.SimpleNamespace(
            translation_sequence_counter=2,
            active_translation_calls={1, 2},
            active_translation_inflight_keys={
                ("custom_ai", "Old", "scope"),
                ("custom_ai", "Newer", "scope"),
            },
            active_translation_started_monotonic={1: 100.0, 2: 101.0},
            translation_thread_pool=pool,
            translation_handler=Handler(),
            enable_instant_cache_display_var=types.SimpleNamespace(get=lambda: False),
            root=types.SimpleNamespace(
                after=lambda delay, callback, *args: scheduled.append(
                    (delay, callback, args)
                )
            ),
            initialize_async_translation_infrastructure=lambda: None,
            last_translation_submit_monotonic=101.0,
            pending_translation_request=None,
            pending_translation_flush_scheduled=False,
            pending_translation_flush_deadline_monotonic=0.0,
            pending_translation_flush_generation=0,
        )

        with patch.object(worker_threads.time, "monotonic", return_value=103.0):
            worker_threads.start_async_translation(app, "Latest", 3)

        self.assertEqual(pool.submissions, [])
        self.assertEqual(app.pending_translation_request["text"], "Latest")
        self.assertEqual(len(scheduled), 1)

    def test_race_mode_does_not_stack_stale_translation_supersession(self):
        worker_threads = import_worker_threads_for_tests()
        scheduled = []

        class Handler:
            def get_cached_translation_for_display(self, text):
                return None

            def get_inflight_translation_key(self, text):
                return ("custom_ai", text, "scope")

            def get_translation_submit_interval_seconds(self, text_content=None):
                return 0.0

            def get_translation_provider_cooldown_seconds(self):
                return 0.0

            def get_translation_concurrency_limit(self):
                return 1

        app = types.SimpleNamespace(
            translation_sequence_counter=1,
            active_translation_calls={1},
            active_translation_inflight_keys={("custom_ai", "Old", "scope")},
            active_translation_started_monotonic={1: 100.0},
            translation_thread_pool=Mock(),
            translation_handler=Handler(),
            enable_instant_cache_display_var=types.SimpleNamespace(get=lambda: False),
            custom_ai_latency_mode_var=types.SimpleNamespace(get=lambda: "race"),
            root=types.SimpleNamespace(
                after=lambda delay, callback, *args: scheduled.append(
                    (delay, callback, args)
                )
            ),
            initialize_async_translation_infrastructure=lambda: None,
            last_translation_submit_monotonic=100.0,
            pending_translation_request=None,
            pending_translation_flush_scheduled=False,
            pending_translation_flush_deadline_monotonic=0.0,
            pending_translation_flush_generation=0,
        )

        with patch.object(worker_threads.time, "monotonic", return_value=103.0):
            worker_threads.start_async_translation(app, "Latest", 2)

        app.translation_thread_pool.submit.assert_not_called()
        self.assertEqual(app.pending_translation_request["text"], "Latest")
        self.assertEqual(len(scheduled), 1)

    def test_young_active_translation_queues_until_supersede_deadline(self):
        worker_threads = import_worker_threads_for_tests()
        scheduled = []

        class Handler:
            def get_cached_translation_for_display(self, text):
                return None

            def get_inflight_translation_key(self, text):
                return ("custom_ai", text, "scope")

            def get_translation_submit_interval_seconds(self, text_content=None):
                return 0.0

            def get_translation_provider_cooldown_seconds(self):
                return 0.0

            def get_translation_concurrency_limit(self):
                return 1

        app = types.SimpleNamespace(
            translation_sequence_counter=1,
            active_translation_calls={1},
            active_translation_inflight_keys={("custom_ai", "Old", "scope")},
            active_translation_started_monotonic={1: 100.0},
            translation_thread_pool=Mock(),
            translation_handler=Handler(),
            enable_instant_cache_display_var=types.SimpleNamespace(get=lambda: False),
            root=types.SimpleNamespace(
                after=lambda delay, callback, *args: scheduled.append(
                    (delay, callback, args)
                )
            ),
            initialize_async_translation_infrastructure=lambda: None,
            last_translation_submit_monotonic=100.0,
            pending_translation_request=None,
            pending_translation_flush_scheduled=False,
            pending_translation_flush_deadline_monotonic=0.0,
            pending_translation_flush_generation=0,
        )

        with patch.object(worker_threads.time, "monotonic", return_value=100.4):
            worker_threads.start_async_translation(app, "Latest", 2)

        self.assertEqual(len(scheduled), 1)
        self.assertGreaterEqual(scheduled[0][0], 1099)
        self.assertLessEqual(scheduled[0][0], 1101)
        self.assertEqual(app.pending_translation_request["text"], "Latest")

    def test_translation_completion_clears_active_start_time(self):
        worker_threads = import_worker_threads_for_tests()
        inflight_key = ("custom_ai", "Current", "scope")

        class Handler:
            def translate_text_with_timeout(self, text, **kwargs):
                return "done"

        app = types.SimpleNamespace(
            root=types.SimpleNamespace(after=lambda *args, **kwargs: None),
            translation_handler=Handler(),
            active_translation_calls={7},
            active_translation_inflight_keys={inflight_key},
            active_translation_started_monotonic={7: 100.0},
            pending_translation_request=None,
            is_running=True,
        )

        worker_threads.process_translation_async(
            app,
            "Current",
            translation_sequence=7,
            ocr_sequence_number=6,
            inflight_key=inflight_key,
        )

        self.assertEqual(app.active_translation_started_monotonic, {})

    def test_pending_translation_reschedules_when_latest_request_has_earlier_deadline(self):
        worker_threads = import_worker_threads_for_tests()
        scheduled = []

        class Root:
            def after(self, delay, callback, *args):
                timer_id = f"timer-{len(scheduled) + 1}"
                scheduled.append((timer_id, delay, callback, args))
                return timer_id

        app = types.SimpleNamespace(
            root=Root(),
            pending_translation_request=None,
            pending_translation_flush_scheduled=False,
            pending_translation_flush_deadline_monotonic=0.0,
            pending_translation_flush_generation=0,
        )

        with patch.object(worker_threads.time, "monotonic", return_value=100.0):
            worker_threads._queue_pending_translation_request(app, "Old", 1, 5.0, "first")
        with patch.object(worker_threads.time, "monotonic", return_value=101.0):
            worker_threads._queue_pending_translation_request(app, "Latest", 2, 1.0, "earlier")

        self.assertEqual(len(scheduled), 2)
        self.assertEqual(app.pending_translation_request["text"], "Latest")
        self.assertEqual(app.pending_translation_flush_generation, 2)
        self.assertEqual(app.pending_translation_flush_deadline_monotonic, 102.0)

    def test_matching_pending_translation_is_coalesced_before_cache_lookup(self):
        worker_threads = import_worker_threads_for_tests()
        scheduled = []
        metrics = RuntimeMetrics(clock=lambda: 100.0)
        cache_lookup = Mock(return_value=None)

        app = types.SimpleNamespace(
            root=types.SimpleNamespace(
                after=lambda *args: scheduled.append(args),
            ),
            initialize_async_translation_infrastructure=lambda: None,
            translation_handler=types.SimpleNamespace(
                get_cached_translation_for_display=cache_lookup,
                get_translation_provider_cooldown_seconds=lambda **kwargs: 10.0,
                get_translation_submit_interval_seconds=lambda text: 0.3,
                get_translation_concurrency_limit=lambda: 1,
            ),
            pending_translation_request={
                "text": "Same",
                "ocr_sequence_number": 8,
                "requested_at_monotonic": 99.5,
            },
            pending_translation_flush_scheduled=True,
            pending_translation_flush_deadline_monotonic=110.0,
            pending_translation_flush_generation=4,
            active_translation_calls=set(),
            active_translation_inflight_keys=set(),
            last_translation_submit_monotonic=0.0,
            runtime_metrics=metrics,
            is_running=True,
        )

        with patch.object(worker_threads.time, "monotonic", return_value=100.0):
            worker_threads.start_async_translation(app, "Same", 9)

        self.assertEqual(app.pending_translation_request["ocr_sequence_number"], 9)
        self.assertEqual(
            app.pending_translation_request["requested_at_monotonic"],
            99.5,
        )
        cache_lookup.assert_not_called()
        self.assertEqual(scheduled, [])
        self.assertEqual(
            metrics.snapshot()["counters"]["pending_translation_coalesced"],
            1,
        )

    def test_unscheduled_matching_pending_translation_reenters_queue_path(self):
        worker_threads = import_worker_threads_for_tests()
        scheduled = []
        metrics = RuntimeMetrics(clock=lambda: 100.0)
        cache_lookup = Mock(return_value=None)

        app = types.SimpleNamespace(
            root=types.SimpleNamespace(
                after=lambda *args: scheduled.append(args),
            ),
            initialize_async_translation_infrastructure=lambda: None,
            translation_handler=types.SimpleNamespace(
                get_cached_translation_for_display=cache_lookup,
                get_translation_provider_cooldown_seconds=lambda **kwargs: 10.0,
                get_translation_submit_interval_seconds=lambda text: 0.3,
                get_translation_concurrency_limit=lambda: 1,
            ),
            pending_translation_request={
                "text": "Same",
                "ocr_sequence_number": 8,
                "requested_at_monotonic": 99.5,
            },
            pending_translation_flush_scheduled=False,
            pending_translation_flush_deadline_monotonic=0.0,
            pending_translation_flush_generation=4,
            active_translation_calls=set(),
            active_translation_inflight_keys=set(),
            last_translation_submit_monotonic=0.0,
            runtime_metrics=metrics,
            is_running=True,
        )

        with patch.object(worker_threads.time, "monotonic", return_value=100.0):
            worker_threads.start_async_translation(app, "Same", 9)

        cache_lookup.assert_called_once_with("Same")
        self.assertEqual(len(scheduled), 1)
        self.assertEqual(app.pending_translation_request["ocr_sequence_number"], 9)
        self.assertEqual(
            metrics.snapshot()["counters"].get(
                "pending_translation_coalesced",
                0,
            ),
            0,
        )

    def test_pending_translation_preserves_request_arrival_time_until_submit(self):
        worker_threads = import_worker_threads_for_tests()
        scheduled = []

        app = types.SimpleNamespace(
            root=types.SimpleNamespace(
                after=lambda delay, callback, *args: scheduled.append(
                    (delay, callback, args)
                )
            ),
            pending_translation_request=None,
            pending_translation_flush_scheduled=False,
            pending_translation_flush_deadline_monotonic=0.0,
            pending_translation_flush_generation=0,
            is_running=True,
        )

        with patch.object(worker_threads.time, "monotonic", return_value=100.0):
            worker_threads._queue_pending_translation_request(
                app,
                "Latest",
                2,
                0.5,
                "submit interval",
                requested_at_monotonic=99.75,
            )

        self.assertEqual(
            app.pending_translation_request["requested_at_monotonic"],
            99.75,
        )

        with patch.object(worker_threads, "start_async_translation") as start_translation:
            with patch.object(worker_threads.time, "monotonic", return_value=100.5):
                worker_threads._flush_pending_translation_request(
                    app,
                    app.pending_translation_flush_generation,
                )

        start_translation.assert_called_once_with(
            app,
            "Latest",
            2,
            requested_at_monotonic=99.75,
        )

    def test_stale_pending_translation_timer_cannot_consume_latest_request(self):
        worker_threads = import_worker_threads_for_tests()
        scheduled = []

        class Root:
            def after(self, delay, callback, *args):
                timer_id = f"timer-{len(scheduled) + 1}"
                scheduled.append((timer_id, delay, callback, args))
                return timer_id

        app = types.SimpleNamespace(
            root=Root(),
            pending_translation_request=None,
            pending_translation_flush_scheduled=False,
            pending_translation_flush_deadline_monotonic=0.0,
            pending_translation_flush_generation=0,
        )

        with patch.object(worker_threads.time, "monotonic", return_value=100.0):
            worker_threads._queue_pending_translation_request(app, "Old", 1, 5.0, "first")
        with patch.object(worker_threads.time, "monotonic", return_value=101.0):
            worker_threads._queue_pending_translation_request(app, "Latest", 2, 1.0, "earlier")

        first_timer = scheduled[0]
        with patch.object(worker_threads, "start_async_translation") as start_translation:
            first_timer[2](*first_timer[3])

        start_translation.assert_not_called()
        self.assertEqual(app.pending_translation_request["text"], "Latest")
        self.assertTrue(app.pending_translation_flush_scheduled)

    def test_translation_completion_expedites_pending_latest_request(self):
        worker_threads = import_worker_threads_for_tests()
        scheduled = []
        inflight_key = ("custom_ai", "Current", "scope")

        class Root:
            def after(self, delay, callback, *args):
                timer_id = f"timer-{len(scheduled) + 1}"
                scheduled.append((delay, callback, args))
                return timer_id

        class Handler:
            def translate_text_with_timeout(self, text, **kwargs):
                return "done"

        app = types.SimpleNamespace(
            root=Root(),
            translation_handler=Handler(),
            active_translation_calls={7},
            active_translation_inflight_keys={inflight_key},
            pending_translation_request={
                "text": "Latest",
                "ocr_sequence_number": 8,
            },
            pending_translation_flush_scheduled=True,
            pending_translation_flush_deadline_monotonic=110.0,
            pending_translation_flush_generation=1,
            is_running=True,
        )

        with patch.object(worker_threads.time, "monotonic", return_value=105.0):
            worker_threads.process_translation_async(
                app,
                "Current",
                translation_sequence=7,
                ocr_sequence_number=6,
                inflight_key=inflight_key,
            )

        self.assertEqual(app.active_translation_calls, set())
        self.assertEqual(app.active_translation_inflight_keys, set())
        self.assertEqual(app.pending_translation_flush_generation, 2)
        flush_calls = [
            item
            for item in scheduled
            if item[1] is worker_threads._flush_pending_translation_request
        ]
        self.assertEqual(len(flush_calls), 1)
        self.assertLessEqual(flush_calls[0][0], 1)

    def test_translation_timing_summary_is_numeric_and_content_free(self):
        worker_threads = import_worker_threads_for_tests()

        class Handler:
            def translate_text_with_timeout(self, text, **kwargs):
                return "translated-secret"

        app = types.SimpleNamespace(
            root=types.SimpleNamespace(after=lambda *args: None),
            translation_handler=Handler(),
            active_translation_calls={7},
            active_translation_inflight_keys=set(),
            pending_translation_request=None,
            custom_ai_latency_mode_var=types.SimpleNamespace(get=lambda: "safe"),
        )

        with patch.object(
            worker_threads.time,
            "monotonic",
            side_effect=[100.0, 102.0],
        ):
            with patch.object(worker_threads, "log_debug") as debug_log:
                worker_threads.process_translation_async(
                    app,
                    "source-secret",
                    translation_sequence=7,
                    ocr_sequence_number=6,
                    requested_at_monotonic=99.5,
                )

        timing_messages = [
            call.args[0]
            for call in debug_log.call_args_list
            if call.args and call.args[0].startswith("LATENCY: translation timing ")
        ]
        self.assertEqual(len(timing_messages), 1)
        timing_message = timing_messages[0]
        self.assertIn("sequence=7", timing_message)
        self.assertIn("queue=0.500s", timing_message)
        self.assertIn("worker=2.000s", timing_message)
        self.assertIn("total=2.500s", timing_message)
        self.assertNotIn("source-secret", timing_message)
        self.assertNotIn("translated-secret", timing_message)

    def test_translation_timing_is_recorded_in_runtime_metrics(self):
        worker_threads = import_worker_threads_for_tests()
        metrics = RuntimeMetrics(clock=lambda: 200.0)

        class Handler:
            def translate_text_with_timeout(self, text, **kwargs):
                return "translated"

        app = types.SimpleNamespace(
            root=types.SimpleNamespace(after=lambda *args: None),
            translation_handler=Handler(),
            active_translation_calls={7},
            active_translation_inflight_keys=set(),
            active_translation_started_monotonic={7: 100.0},
            pending_translation_request=None,
            custom_ai_latency_mode_var=types.SimpleNamespace(get=lambda: "safe"),
            runtime_metrics=metrics,
            is_running=True,
        )

        with patch.object(
            worker_threads.time,
            "monotonic",
            side_effect=[100.0, 102.0],
        ):
            worker_threads.process_translation_async(
                app,
                "source",
                translation_sequence=7,
                ocr_sequence_number=6,
                requested_at_monotonic=99.5,
            )

        snapshot = metrics.snapshot()
        self.assertAlmostEqual(
            snapshot["timings"]["translation_queue_time"]["latest"],
            0.5,
        )
        self.assertAlmostEqual(
            snapshot["timings"]["translation_worker_time"]["latest"],
            2.0,
        )
        self.assertAlmostEqual(
            snapshot["timings"]["translation_total_latency"]["latest"],
            2.5,
        )
        self.assertEqual(snapshot["gauges"]["active_translation_calls"], 0)

    def test_translation_runtime_metrics_record_cache_duplicate_queue_and_cooldown(self):
        worker_threads = import_worker_threads_for_tests()
        metrics = RuntimeMetrics(clock=lambda: 200.0)
        displayed = []
        scheduled = []

        class Handler:
            def get_cached_translation_for_display(self, text):
                if text == "Cached":
                    return "Cached result"
                return None

            def get_inflight_translation_key(self, text):
                return ("custom_ai", text, "scope")

            def get_translation_submit_interval_seconds(self, text_content=None):
                return 0.0

            def get_translation_provider_cooldown_seconds(self):
                return 2.5

            def get_translation_concurrency_limit(self):
                return 1

        app = types.SimpleNamespace(
            translation_sequence_counter=0,
            latest_translation_sequence_started=0,
            last_displayed_translation_sequence=0,
            active_translation_calls=set(),
            active_translation_inflight_keys={("custom_ai", "Duplicate", "scope")},
            active_translation_started_monotonic={},
            translation_thread_pool=Mock(),
            translation_handler=Handler(),
            enable_instant_cache_display_var=types.SimpleNamespace(get=lambda: True),
            custom_ai_latency_mode_var=types.SimpleNamespace(get=lambda: "safe"),
            root=types.SimpleNamespace(
                after=lambda delay, callback, *args: scheduled.append(
                    (delay, callback, args)
                )
            ),
            initialize_async_translation_infrastructure=lambda: None,
            update_translation_text=displayed.append,
            pending_translation_request=None,
            pending_translation_flush_scheduled=False,
            pending_translation_flush_deadline_monotonic=0.0,
            pending_translation_flush_generation=0,
            last_successful_translation_time=0.0,
            last_translation_submit_monotonic=0.0,
            runtime_metrics=metrics,
            is_running=True,
        )

        with patch.object(worker_threads.time, "monotonic", return_value=100.0):
            worker_threads.start_async_translation(app, "Cached", 1)
            worker_threads.start_async_translation(app, "Duplicate", 2)
            worker_threads.start_async_translation(app, "Queued", 3)

        snapshot = metrics.snapshot()
        self.assertEqual(snapshot["counters"]["instant_cache_hit"], 1)
        self.assertEqual(snapshot["counters"]["duplicate_inflight_skip"], 1)
        self.assertEqual(snapshot["counters"]["pending_translation_queued"], 1)
        self.assertEqual(snapshot["gauges"]["provider_cooldown_seconds"], 2.5)
        self.assertEqual(snapshot["gauges"]["translation_concurrency_limit"], 1)
        self.assertEqual(snapshot["gauges"]["active_translation_calls"], 0)
        self.assertEqual(displayed, ["Cached result"])
        self.assertEqual(app.pending_translation_request["text"], "Queued")

    def test_stale_translation_response_increments_runtime_metric(self):
        worker_threads = import_worker_threads_for_tests()
        metrics = RuntimeMetrics(clock=lambda: 200.0)
        app = types.SimpleNamespace(
            last_displayed_translation_sequence=5,
            runtime_metrics=metrics,
        )

        worker_threads.process_translation_response(
            app,
            "Obsolete",
            4,
            "source",
            3,
        )

        self.assertEqual(
            metrics.snapshot()["counters"]["stale_response_discarded"],
            1,
        )

    def test_streaming_partial_display_increments_runtime_metric(self):
        worker_threads = import_worker_threads_for_tests()
        metrics = RuntimeMetrics(clock=lambda: 200.0)
        scheduled = []
        displayed = []
        app = types.SimpleNamespace(
            is_running=True,
            latest_translation_sequence_started=1,
            last_displayed_translation_sequence=0,
            root=types.SimpleNamespace(
                after=lambda delay, callback, *args: scheduled.append(
                    (delay, callback, args)
                )
            ),
            update_translation_text=displayed.append,
            last_successful_translation_time=0.0,
            runtime_metrics=metrics,
        )

        stream_callback = worker_threads._build_streaming_display_callback(app, 1)
        stream_callback("Partial")

        scheduled[0][1](*scheduled[0][2])

        self.assertEqual(displayed, ["Partial"])
        self.assertEqual(
            metrics.snapshot()["counters"]["stream_partial_display"],
            1,
        )

    def test_api_ocr_cache_hit_increments_runtime_metric(self):
        worker_threads = import_worker_threads_for_tests()
        ocr_utils = import_ocr_utils_for_tests()
        metrics = RuntimeMetrics(clock=lambda: 200.0)
        screenshot = Image.new("RGB", (8, 8), (1, 2, 3))
        screenshot._gct_frame_hash = "frame-hash"
        screenshot._gct_region_origin = (10, 20)
        profile = {
            "id": "ocr-profile",
            "name": "OCR",
            "base_url": "https://provider.example/v1",
            "model": "vision",
        }
        cache = ocr_utils.OCRFrameCache(max_size=4)
        key = ocr_utils.build_ocr_frame_cache_key(
            "frame-hash",
            "custom_ai|ocr_profile=ocr-profile|https://provider.example/v1|vision",
            "en",
            "api|keep_linebreaks=False|reasoning_effort=low",
            screenshot.size,
            region_origin=(10, 20),
        )
        cache.put(key, "Cached OCR")

        app = types.SimpleNamespace(
            get_ocr_model_setting=lambda: "custom_ai",
            custom_source_lang="en",
            source_lang_var=types.SimpleNamespace(get=lambda: "en"),
            custom_ai_profiles=types.SimpleNamespace(
                get_active_profile=lambda kind: profile
            ),
            keep_linebreaks_var=types.SimpleNamespace(get=lambda: False),
            ocr_frame_cache=cache,
            batch_sequence_counter=0,
            active_ocr_calls=set(),
            max_concurrent_ocr_calls=8,
            last_displayed_batch_sequence=0,
            last_processed_subtitle="Cached OCR",
            reset_clear_timeout=Mock(),
            runtime_metrics=metrics,
        )

        worker_threads.run_api_ocr(app, screenshot)

        self.assertEqual(
            metrics.snapshot()["counters"]["ocr_frame_cache_hit"],
            1,
        )

    def test_custom_ai_race_winner_runtime_label_is_redacted(self):
        TranslationHandler = import_translation_handler_for_tests()
        metrics = RuntimeMetrics(clock=lambda: 200.0)
        profile = {
            "id": "fast",
            "name": "Fast profile sk-test-secret",
            "base_url": "https://provider.example/v1",
            "api_key": "secret",
            "model": "model-a",
            "wire_api": "chat_completions",
            "enabled": True,
        }
        app = types.SimpleNamespace(
            runtime_metrics=metrics,
            custom_ai_profiles=types.SimpleNamespace(
                list_profiles=lambda *args, **kwargs: [profile]
            ),
        )
        handler = TranslationHandler(app)
        handler.custom_ai_provider.translate = Mock(
            return_value=("Translated", {}, 0.25)
        )

        handler._custom_ai_translate_race(
            profile,
            "Hello",
            "en",
            "pl",
            [],
            False,
            custom_prompt="",
        )

        race_winner = metrics.snapshot()["labels"]["race_winner"]
        self.assertIn("Fast profile", race_winner)
        self.assertNotIn("sk-test-secret", race_winner)

    def test_pending_translation_is_dropped_when_app_is_stopped(self):
        worker_threads = import_worker_threads_for_tests()
        app = types.SimpleNamespace(
            pending_translation_request={
                "text": "Do not submit",
                "ocr_sequence_number": 3,
            },
            pending_translation_flush_scheduled=True,
            pending_translation_flush_deadline_monotonic=110.0,
            pending_translation_flush_generation=4,
            is_running=False,
        )

        with patch.object(worker_threads, "start_async_translation") as start_translation:
            worker_threads._flush_pending_translation_request(app, 4)

        start_translation.assert_not_called()
        self.assertIsNone(app.pending_translation_request)
        self.assertFalse(app.pending_translation_flush_scheduled)
        self.assertEqual(app.pending_translation_flush_deadline_monotonic, 0.0)

    def test_custom_ai_submit_interval_expands_for_large_ocr_payloads(self):
        TranslationHandler = import_translation_handler_for_tests()

        app = types.SimpleNamespace(
            min_translation_interval=0.3,
            translation_model_var=types.SimpleNamespace(get=lambda: "custom_ai"),
        )
        handler = TranslationHandler(app)

        short_interval = handler.get_translation_submit_interval_seconds("Hello")
        long_interval = handler.get_translation_submit_interval_seconds("Line " * 140)

        self.assertEqual(short_interval, 0.3)
        self.assertGreater(long_interval, short_interval)
        self.assertGreaterEqual(long_interval, 2.0)

    def test_custom_ai_submit_interval_uses_clamped_user_milliseconds(self):
        TranslationHandler = import_translation_handler_for_tests()

        for milliseconds, expected_seconds in [
            (0, 0.0),
            (650, 0.65),
            (-10, 0.0),
            (6000, 5.0),
        ]:
            with self.subTest(milliseconds=milliseconds):
                app = types.SimpleNamespace(
                    custom_ai_submit_interval_ms_var=types.SimpleNamespace(
                        get=lambda value=milliseconds: value
                    ),
                    min_translation_interval=0.3,
                    translation_model_var=types.SimpleNamespace(get=lambda: "custom_ai"),
                )
                handler = TranslationHandler(app)

                self.assertEqual(
                    handler.get_translation_submit_interval_seconds("Hello"),
                    expected_seconds,
                )

    def test_local_ocr_translation_gate_uses_provider_submit_interval_without_blocking_pending_refreshes(self):
        worker_threads = import_worker_threads_for_tests()

        class CooldownHandler:
            def get_translation_submit_interval_seconds(self, text_content=None):
                return 2.5

            def get_translation_provider_cooldown_seconds(self):
                return 7.0

        class SubmitIntervalHandler:
            def get_translation_submit_interval_seconds(self, text_content=None):
                return 2.5

            def get_translation_provider_cooldown_seconds(self):
                return 0.0

        cooldown_app = types.SimpleNamespace(translation_handler=CooldownHandler())
        submit_interval_app = types.SimpleNamespace(translation_handler=SubmitIntervalHandler())

        gate_seconds = worker_threads._get_local_ocr_translation_gate_seconds(
            cooldown_app,
            "Long OCR payload",
            0.3,
        )
        submit_interval_gate = worker_threads._get_local_ocr_translation_gate_seconds(
            submit_interval_app,
            "Long OCR payload",
            0.3,
        )

        self.assertEqual(gate_seconds, 2.5)
        self.assertEqual(submit_interval_gate, 2.5)

    def test_local_ocr_elapsed_since_last_submit_prefers_submit_timestamp(self):
        worker_threads = import_worker_threads_for_tests()
        app = types.SimpleNamespace(
            last_translation_submit_monotonic=98.0,
            last_successful_translation_time=40.0,
        )

        elapsed = worker_threads._get_local_ocr_elapsed_since_last_submit(app, 100.0)

        self.assertEqual(elapsed, 2.0)

    def test_local_ocr_elapsed_since_last_submit_falls_back_to_success_time(self):
        worker_threads = import_worker_threads_for_tests()
        app = types.SimpleNamespace(last_successful_translation_time=95.0)

        elapsed = worker_threads._get_local_ocr_elapsed_since_last_submit(app, 100.0)

        self.assertEqual(elapsed, 5.0)

    def test_local_ocr_submit_dedup_skips_normalized_repeat_after_gate(self):
        worker_threads = import_worker_threads_for_tests()
        app = types.SimpleNamespace(
            last_local_ocr_submitted_text="The treasure door is open.",
            last_local_ocr_submitted_norm="the treasure door is open",
        )

        self.assertTrue(
            worker_threads._should_skip_local_ocr_resubmit(
                app,
                "  The treasure door is open!  ",
            )
        )

    def test_local_ocr_submit_dedup_allows_meaningful_change(self):
        worker_threads = import_worker_threads_for_tests()
        app = types.SimpleNamespace(
            last_local_ocr_submitted_text="The treasure door is open.",
            last_local_ocr_submitted_norm="the treasure door is open",
        )

        self.assertFalse(
            worker_threads._should_skip_local_ocr_resubmit(
                app,
                "The treasure door is closed.",
            )
        )

    def test_local_ocr_submit_dedup_allows_resubmit_when_translation_scope_changes(self):
        worker_threads = import_worker_threads_for_tests()

        class Handler:
            def get_inflight_translation_key(self, text):
                return ("custom_ai", text, "en", "zh-CN", "profile-b", "https://b.example", "model-b")

        app = types.SimpleNamespace(
            translation_handler=Handler(),
            last_local_ocr_submitted_text="The treasure door is open.",
            last_local_ocr_submitted_norm="the treasure door is open",
            last_local_ocr_submitted_scope=("en", "zh-CN", "profile-a", "https://a.example", "model-a"),
        )

        self.assertFalse(
            worker_threads._should_skip_local_ocr_resubmit(
                app,
                "The treasure door is open!",
            )
        )

    def test_translate_text_with_timeout_waits_for_real_translation_completion(self):
        TranslationHandler = import_translation_handler_for_tests()

        app = types.SimpleNamespace(translation_model_var=types.SimpleNamespace(get=lambda: "custom_ai"))
        handler = TranslationHandler(app)
        started = threading.Event()
        release = threading.Event()

        def blocking_translate(*args, **kwargs):
            started.set()
            release.wait(timeout=1.0)
            return "translated"

        handler.translate_text = blocking_translate

        def delayed_release():
            started.wait(timeout=1.0)
            time.sleep(0.05)
            release.set()

        releaser = threading.Thread(target=delayed_release, daemon=True)
        releaser.start()

        start = time.monotonic()
        result = handler.translate_text_with_timeout("Hello", timeout_seconds=0.01)
        elapsed = time.monotonic() - start

        self.assertEqual(result, "translated")
        self.assertGreaterEqual(elapsed, 0.04)

    def test_custom_ai_errors_do_not_enter_success_state(self):
        worker_threads = import_worker_threads_for_tests()

        for error_result in [
            "Custom AI translation error: ValueError - upstream busy",
            "AI model profile for translation is missing.",
        ]:
            with self.subTest(error_result=error_result):
                displayed = []
                app = types.SimpleNamespace(
                    last_displayed_translation_sequence=0,
                    last_successful_translation_time=123.0,
                    last_local_ocr_submitted_text="Hello",
                    last_local_ocr_submitted_norm="hello",
                    last_local_ocr_submitted_scope=("scope",),
                    update_translation_text=displayed.append,
                )

                worker_threads.process_translation_response(
                    app,
                    error_result,
                    1,
                    "Hello",
                    0,
                )

                self.assertEqual(
                    displayed,
                    [f"Translation Error:\n{error_result}"],
                )
                self.assertEqual(
                    app.last_successful_translation_time,
                    123.0,
                )
                self.assertEqual(
                    app.last_displayed_translation_sequence,
                    1,
                )
                self.assertIsNone(app.last_local_ocr_submitted_text)
                self.assertIsNone(app.last_local_ocr_submitted_norm)
                self.assertIsNone(app.last_local_ocr_submitted_scope)

    def test_transient_custom_ai_provider_errors_do_not_cover_current_subtitle(self):
        worker_threads = import_worker_threads_for_tests()
        cases = (
            (
                "Custom AI translation error: ValueError - "
                "https://api.x.ai/v1/chat/completions: "
                "Streaming API response did not contain message content"
            ),
            (
                "Custom AI translation error: ValueError - "
                "Structured translation response contained empty translation"
            ),
            (
                "Custom AI translation error: SSLEOFError - "
                "TLS/SSL connection was closed "
                "(unexpected_eof_while_reading)"
            ),
        )

        for error_text in cases:
            with self.subTest(error_text=error_text):
                displayed = []
                app = types.SimpleNamespace(
                    last_displayed_translation_sequence=0,
                    last_successful_translation_time=123.0,
                    last_local_ocr_submitted_text="NMMNm",
                    last_local_ocr_submitted_norm="nmmnm",
                    last_local_ocr_submitted_scope=("scope",),
                    update_translation_text=displayed.append,
                )

                worker_threads.process_translation_response(
                    app,
                    error_text,
                    26,
                    "NMMNm",
                    0,
                )

                self.assertEqual(displayed, [])
                self.assertEqual(app.last_displayed_translation_sequence, 0)
                self.assertEqual(app.last_successful_translation_time, 123.0)
                self.assertIsNone(app.last_local_ocr_submitted_text)
                self.assertIsNone(app.last_local_ocr_submitted_norm)
                self.assertIsNone(app.last_local_ocr_submitted_scope)

    def test_streaming_translation_partial_updates_are_scheduled_on_ui_thread(self):
        worker_threads = import_worker_threads_for_tests()
        scheduled = []
        displayed = []

        class Handler:
            def translate_text_with_timeout(
                self,
                text,
                timeout_seconds=10.0,
                ocr_batch_number=None,
                stream_callback=None,
                translation_sequence=None,
                latency_mode=None,
            ):
                stream_callback("Hel")
                stream_callback("Hello")
                return "Hello"

        app = types.SimpleNamespace(
            is_running=True,
            latest_translation_sequence_started=1,
            last_displayed_translation_sequence=0,
            active_translation_calls={1},
            active_translation_inflight_keys=set(),
            translation_handler=Handler(),
            custom_ai_latency_mode_var=types.SimpleNamespace(get=lambda: "stream"),
            root=types.SimpleNamespace(after=lambda delay, callback, *args: scheduled.append((delay, callback, args))),
            update_translation_text=lambda text: displayed.append(text),
            last_successful_translation_time=0,
        )

        worker_threads.process_translation_async(app, "Hello", 1, 7, ("custom_ai", "Hello"))

        self.assertEqual(displayed, [])
        for _delay, callback, args in scheduled:
            callback(*args)

        self.assertTrue(displayed)
        self.assertTrue(all(text == "Hello" for text in displayed))

    def test_streaming_translation_coalesces_pending_ui_updates(self):
        worker_threads = import_worker_threads_for_tests()
        scheduled = []
        displayed = []

        class Handler:
            def translate_text_with_timeout(
                self,
                text,
                timeout_seconds=10.0,
                ocr_batch_number=None,
                stream_callback=None,
                translation_sequence=None,
                latency_mode=None,
            ):
                stream_callback("H")
                stream_callback("He")
                stream_callback("Hello")
                return "Hello"

        app = types.SimpleNamespace(
            is_running=True,
            latest_translation_sequence_started=1,
            last_displayed_translation_sequence=0,
            active_translation_calls={1},
            active_translation_inflight_keys=set(),
            translation_handler=Handler(),
            custom_ai_latency_mode_var=types.SimpleNamespace(get=lambda: "stream"),
            root=types.SimpleNamespace(
                after=lambda delay, callback, *args: scheduled.append(
                    (delay, callback, args)
                )
            ),
            update_translation_text=lambda text: displayed.append(text),
            last_successful_translation_time=0,
        )

        worker_threads.process_translation_async(
            app,
            "Hello",
            1,
            7,
            ("custom_ai", "Hello"),
        )

        self.assertEqual(len(scheduled), 2)
        _delay, partial_callback, partial_args = scheduled[0]
        partial_callback(*partial_args)
        self.assertEqual(displayed, ["Hello"])
        _delay, final_callback, final_args = scheduled[1]
        final_callback(*final_args)
        self.assertEqual(displayed, ["Hello"])


class LatencyOcrStabilityGateTests(unittest.TestCase):
    @staticmethod
    def _make_root(scheduled):
        return types.SimpleNamespace(
            after=lambda delay, callback, *args: scheduled.append(
                (delay, callback, args)
            )
        )

    @staticmethod
    def _make_app(scheduled, handler=None):
        return types.SimpleNamespace(
            root=LatencyOcrStabilityGateTests._make_root(scheduled),
            translation_handler=handler,
            enable_instant_cache_display_var=types.SimpleNamespace(get=lambda: True),
            reset_clear_timeout=Mock(),
            is_running=True,
        )

    def test_unstable_short_candidate_is_replaced_by_more_complete_text_before_translation(self):
        worker_threads = import_worker_threads_for_tests()
        scheduled = []
        submitted = []
        app = self._make_app(scheduled)

        def submit(_app, text, sequence, requested_at_monotonic=None):
            submitted.append((text, sequence, requested_at_monotonic))

        with patch.object(worker_threads, "start_async_translation", side_effect=submit):
            first_result = worker_threads._route_local_ocr_candidate_for_translation(
                app,
                "The treas",
                now=100.0,
            )
            second_result = worker_threads._route_local_ocr_candidate_for_translation(
                app,
                "The treasure door is open.",
                now=100.1,
            )

            for _delay, callback, args in list(scheduled):
                callback(*args)

        self.assertEqual(first_result, "pending")
        self.assertEqual(second_result, "submitted")
        self.assertEqual(
            submitted,
            [("The treasure door is open.", 0, 100.1)],
        )

    def test_clear_stable_candidate_submits_without_ocr_gate_delay(self):
        worker_threads = import_worker_threads_for_tests()
        scheduled = []
        submitted = []
        app = self._make_app(scheduled)

        with patch.object(
            worker_threads,
            "start_async_translation",
            side_effect=lambda _app, text, sequence, requested_at_monotonic=None: submitted.append(text),
        ):
            result = worker_threads._route_local_ocr_candidate_for_translation(
                app,
                "The treasure door is open.",
                now=200.0,
            )

        self.assertEqual(result, "submitted")
        self.assertEqual(submitted, ["The treasure door is open."])
        self.assertEqual(scheduled, [])

    def test_pending_local_ocr_candidate_does_not_probe_translation_cache(self):
        worker_threads = import_worker_threads_for_tests()
        scheduled = []

        class Handler:
            get_cached_translation_for_display = Mock(return_value=None)

        app = self._make_app(scheduled, handler=Handler())
        with patch.object(worker_threads, "start_async_translation") as start_translation:
            result = worker_threads._route_local_ocr_candidate_for_translation(
                app,
                "The treas",
                now=100.0,
            )

        self.assertEqual(result, "pending")
        app.translation_handler.get_cached_translation_for_display.assert_not_called()
        start_translation.assert_not_called()

    def test_truncated_ocr_candidate_waits_for_more_complete_text(self):
        worker_threads = import_worker_threads_for_tests()
        scheduled = []
        submitted = []
        app = self._make_app(scheduled)

        def submit(_app, text, sequence, requested_at_monotonic=None):
            submitted.append((text, sequence, requested_at_monotonic))

        with patch.object(worker_threads, "start_async_translation", side_effect=submit):
            first_result = worker_threads._route_local_ocr_candidate_for_translation(
                app,
                "If I'm going to lis",
                now=700.0,
            )
            second_result = worker_threads._route_local_ocr_candidate_for_translation(
                app,
                "If I'm going to live under Lois's rule,",
                now=700.2,
            )

            for _delay, callback, args in list(scheduled):
                callback(*args)

        self.assertEqual(first_result, "pending")
        self.assertEqual(second_result, "submitted")
        self.assertEqual(
            submitted,
            [("If I'm going to live under Lois's rule,", 0, 700.2)],
        )

    def test_pending_candidate_flushes_after_max_wait(self):
        worker_threads = import_worker_threads_for_tests()
        scheduled = []
        submitted = []
        app = self._make_app(scheduled)

        with patch.object(
            worker_threads,
            "start_async_translation",
            side_effect=lambda _app, text, sequence, requested_at_monotonic=None: submitted.append(text),
        ):
            result = worker_threads._route_local_ocr_candidate_for_translation(
                app,
                "The treas",
                now=300.0,
            )

            self.assertEqual(result, "pending")
            self.assertEqual(submitted, [])
            self.assertEqual(len(scheduled), 1)
            self.assertGreaterEqual(scheduled[0][0], 100)
            self.assertLessEqual(scheduled[0][0], 250)

            with patch.object(worker_threads.time, "monotonic", return_value=300.3):
                scheduled[0][1](*scheduled[0][2])

        self.assertEqual(submitted, ["The treas"])
        self.assertIsNone(app.ocr_stability_gate.pending_text)

    def test_ocr_stability_flush_drops_pending_when_app_is_stopped(self):
        worker_threads = import_worker_threads_for_tests()
        scheduled = []
        submitted = []
        app = self._make_app(scheduled)

        with patch.object(
            worker_threads,
            "start_async_translation",
            side_effect=lambda _app, text, sequence, requested_at_monotonic=None: submitted.append(text),
        ):
            worker_threads._route_local_ocr_candidate_for_translation(
                app,
                "The treas",
                now=400.0,
            )
            app.is_running = False
            with patch.object(worker_threads.time, "monotonic", return_value=400.3):
                scheduled[0][1](*scheduled[0][2])

        self.assertEqual(submitted, [])
        self.assertIsNone(app.ocr_stability_gate.pending_text)

    def test_instant_cache_hit_clears_older_pending_gate(self):
        worker_threads = import_worker_threads_for_tests()
        scheduled = []
        submitted = []

        class Handler:
            def get_cached_translation_for_display(self, text):
                if text == "The treasure door is open.":
                    return "Cached translation"
                return None

        app = self._make_app(scheduled, handler=Handler())

        with patch.object(
            worker_threads,
            "start_async_translation",
            side_effect=lambda _app, text, sequence, requested_at_monotonic=None: submitted.append(text),
        ):
            worker_threads._route_local_ocr_candidate_for_translation(
                app,
                "The treas",
                now=500.0,
            )
            result = worker_threads._route_local_ocr_candidate_for_translation(
                app,
                "The treasure door is open.",
                now=500.05,
            )

            for _delay, callback, args in list(scheduled):
                callback(*args)

        self.assertEqual(result, "submitted")
        self.assertEqual(submitted, ["The treasure door is open."])
        self.assertIsNone(app.ocr_stability_gate.pending_text)

    def test_near_duplicate_skip_still_happens_before_gate(self):
        worker_threads = import_worker_threads_for_tests()
        scheduled = []
        app = self._make_app(scheduled)
        app.last_local_ocr_submitted_text = "The treasure door is open."
        app.last_local_ocr_submitted_norm = "the treasure door is open"

        with patch.object(worker_threads, "start_async_translation") as start_translation:
            result = worker_threads._route_local_ocr_candidate_for_translation(
                app,
                "The treasure door is open!",
                now=600.0,
            )

        self.assertEqual(result, "skipped")
        start_translation.assert_not_called()
        self.assertEqual(scheduled, [])

    def test_active_inflight_dedup_remains_in_start_async_path(self):
        worker_threads = import_worker_threads_for_tests()
        scheduled = []

        class Pool:
            def __init__(self):
                self.submissions = []

            def submit(self, fn, *args):
                self.submissions.append((fn, args))
                return object()

        class Handler:
            def get_cached_translation_for_display(self, text):
                return None

            def get_inflight_translation_key(self, text):
                return ("custom_ai", text, "same-context")

        pool = Pool()
        app = self._make_app(scheduled, handler=Handler())
        app.translation_sequence_counter = 0
        app.active_translation_calls = set()
        app.active_translation_inflight_keys = {
            ("custom_ai", "The treasure door is open.", "same-context")
        }
        app.active_translation_started_monotonic = {}
        app.max_concurrent_translation_calls = 6
        app.translation_thread_pool = pool
        app.initialize_async_translation_infrastructure = lambda: None

        result = worker_threads._route_local_ocr_candidate_for_translation(
            app,
            "The treasure door is open.",
            now=700.0,
        )

        self.assertEqual(result, "submitted")
        self.assertEqual(pool.submissions, [])
        self.assertEqual(
            app.active_translation_inflight_keys,
            {("custom_ai", "The treasure door is open.", "same-context")},
        )

    def test_ocr_model_change_clears_stability_gate(self):
        import app_logic

        app = object.__new__(app_logic.GameChangingTranslator)
        app.is_running = False
        app.is_api_based_ocr_model = lambda: False
        app.ocr_preview_window = None
        app.ocr_model_var = types.SimpleNamespace(get=lambda: "custom_ai")
        app.ocr_stability_gate = types.SimpleNamespace(clear=Mock(return_value=True))

        app_logic.GameChangingTranslator.on_ocr_model_change(app)

        app.ocr_stability_gate.clear.assert_called()


class RuntimeLogCoalescingTests(unittest.TestCase):
    def test_pending_queue_uses_content_free_coalesced_log(self):
        worker_threads = import_worker_threads_for_tests()
        scheduled = []
        metrics = RuntimeMetrics(clock=lambda: 100.0)
        app = types.SimpleNamespace(
            root=types.SimpleNamespace(
                after=lambda delay, callback, *args: scheduled.append(
                    (delay, callback, args)
                )
            ),
            pending_translation_request=None,
            pending_translation_flush_scheduled=False,
            pending_translation_flush_deadline_monotonic=0.0,
            pending_translation_flush_generation=0,
            runtime_metrics=metrics,
        )

        with patch.object(worker_threads.time, "monotonic", return_value=100.0):
            with patch.object(
                worker_threads,
                "log_debug_coalesced",
            ) as log_coalesced:
                worker_threads._queue_pending_translation_request(
                    app,
                    "secret queued subtitle",
                    7,
                    0.5,
                    "active calls 2/2",
                )

        log_coalesced.assert_called_once_with(
            "translation-pending-queue",
            "LATENCY: queued latest translation request for OCR batch 7 "
            "delay=0.500s reason=active calls 2/2",
            interval_seconds=5.0,
        )
        self.assertEqual(
            app.pending_translation_request["text"],
            "secret queued subtitle",
        )
        self.assertEqual(
            metrics.snapshot()["counters"]["pending_translation_queued"],
            1,
        )
        self.assertEqual(len(scheduled), 1)

    def test_duplicate_inflight_skip_uses_content_free_coalesced_log(self):
        worker_threads = import_worker_threads_for_tests()
        metrics = RuntimeMetrics(clock=lambda: 100.0)
        inflight_key = ("custom_ai", "secret duplicate subtitle", "scope")
        app = types.SimpleNamespace(
            initialize_async_translation_infrastructure=lambda: None,
            translation_handler=types.SimpleNamespace(
                get_cached_translation_for_display=lambda text: None,
                get_inflight_translation_key=lambda text: inflight_key,
            ),
            enable_instant_cache_display_var=types.SimpleNamespace(
                get=lambda: False
            ),
            active_translation_calls={1},
            active_translation_inflight_keys={inflight_key},
            translation_thread_pool=Mock(),
            runtime_metrics=metrics,
        )

        with patch.object(
            worker_threads,
            "log_debug_coalesced",
        ) as log_coalesced:
            worker_threads.start_async_translation(
                app,
                "secret duplicate subtitle",
                9,
            )

        log_coalesced.assert_called_once_with(
            "translation-duplicate-inflight",
            "LATENCY: duplicate in-flight translation skipped for OCR batch 9",
            interval_seconds=5.0,
        )
        self.assertEqual(
            metrics.snapshot()["counters"]["duplicate_inflight_skip"],
            1,
        )
        app.translation_thread_pool.submit.assert_not_called()

    def test_stale_pending_timer_uses_coalesced_log_without_consuming_request(self):
        worker_threads = import_worker_threads_for_tests()
        pending = {
            "text": "latest subtitle",
            "ocr_sequence_number": 4,
            "requested_at_monotonic": 99.0,
        }
        app = types.SimpleNamespace(
            pending_translation_flush_generation=6,
            pending_translation_flush_scheduled=True,
            pending_translation_flush_deadline_monotonic=105.0,
            pending_translation_request=pending,
            is_running=True,
        )

        with patch.object(
            worker_threads,
            "log_debug_coalesced",
        ) as log_coalesced:
            worker_threads._flush_pending_translation_request(
                app,
                flush_generation=5,
            )

        log_coalesced.assert_called_once_with(
            "translation-stale-pending-timer",
            "LATENCY: ignored stale pending translation timer "
            "generation=5 current=6",
            interval_seconds=5.0,
        )
        self.assertIs(app.pending_translation_request, pending)
        self.assertTrue(app.pending_translation_flush_scheduled)


class RuntimeContentFreeTranslationLogTests(unittest.TestCase):
    def test_translation_submission_log_excludes_source_content(self):
        worker_threads = import_worker_threads_for_tests()
        app = types.SimpleNamespace(
            translation_sequence_counter=0,
            active_translation_calls=set(),
            active_translation_inflight_keys=set(),
            active_translation_started_monotonic={},
            translation_thread_pool=types.SimpleNamespace(submit=Mock()),
        )

        with patch.object(worker_threads, "log_debug") as debug_log:
            worker_threads._submit_async_translation_request(
                app,
                "source-secret",
                7,
                ("request",),
                requested_at_monotonic=100.0,
            )

        messages = "\n".join(
            str(call.args[0]) for call in debug_log.call_args_list
        )
        self.assertNotIn("source-secret", messages)
        self.assertIn("chars=13 lines=1", messages)
        self.assertIn("translation 1", messages.lower())

    def test_translation_completion_response_and_display_logs_exclude_content(self):
        worker_threads = import_worker_threads_for_tests()
        scheduled = []

        class Handler:
            def translate_text_with_timeout(self, _text, **_kwargs):
                return "result-secret"

        app = types.SimpleNamespace(
            root=types.SimpleNamespace(
                after=lambda delay, callback, *args: scheduled.append(
                    (delay, callback, args)
                )
            ),
            translation_handler=Handler(),
            active_translation_calls={3},
            active_translation_inflight_keys={("request",)},
            active_translation_started_monotonic={3: 100.0},
            pending_translation_request=None,
            max_concurrent_translation_calls=1,
            is_running=True,
            last_displayed_translation_sequence=0,
            latest_translation_sequence_started=3,
            update_translation_text=Mock(),
            last_successful_translation_time=0.0,
        )

        with patch.object(worker_threads, "log_debug") as debug_log:
            worker_threads.process_translation_async(
                app,
                "source-secret",
                translation_sequence=3,
                ocr_sequence_number=0,
                inflight_key=("request",),
            )
            _delay, callback, args = scheduled.pop(0)
            callback(*args)

        messages = "\n".join(
            str(call.args[0]) for call in debug_log.call_args_list
        )
        self.assertNotIn("source-secret", messages)
        self.assertNotIn("result-secret", messages)
        self.assertIn("chars=13 lines=1", messages)
        self.assertIn("sequence 3", messages.lower())
        app.update_translation_text.assert_called_once_with("result-secret")

    def test_api_ocr_to_translation_logs_exclude_recognized_content(self):
        worker_threads = import_worker_threads_for_tests()
        scheduled = []

        class Handler:
            def perform_ocr(self, _image, _source_lang, **_kwargs):
                return "recognized-secret"

        app = types.SimpleNamespace(
            batch_sequence_counter=3,
            active_ocr_calls={3},
            translation_handler=Handler(),
            root=types.SimpleNamespace(
                after=lambda delay, callback, *args: scheduled.append(
                    (delay, callback, args)
                )
            ),
            last_displayed_batch_sequence=0,
            last_processed_subtitle=None,
            reset_clear_timeout=Mock(),
        )

        with patch.object(worker_threads, "log_debug") as debug_log:
            with patch.object(worker_threads, "start_async_translation") as start:
                worker_threads.process_api_ocr_async(
                    app,
                    b"image",
                    "en",
                    3,
                    "custom_ai",
                )
                _delay, callback, args = scheduled.pop(0)
                callback(*args)

        messages = "\n".join(
            str(call.args[0]) for call in debug_log.call_args_list
        )
        self.assertNotIn("recognized-secret", messages)
        self.assertIn("chars=17 lines=1", messages)
        start.assert_called_once_with(app, "recognized-secret", 3)


class LatencyLegacyOcrRemovalTests(unittest.TestCase):
    def test_platform_ocr_support_is_removed_from_project_files(self):
        files_to_check = [
            "app_logic.py",
            "gui_builder.py",
            "worker_threads.py",
            "ocr_utils.py",
            "requirements.txt",
            "ocr_translator_config.example.ini",
            "resources/gui_eng.csv",
            "resources/gui_pol.csv",
            "resources/gui_zh.csv",
        ]

        forbidden_tokens = (
            "windows" + "_ocr",
            "windows" + "_ocr_available",
            "is_" + "windows" + "_ocr_available",
            "recognize_" + "windows" + "_ocr",
            "win" + "sdk",
            "win" + "rt",
            "windows " + "ocr",
        )

        for relative_path in files_to_check:
            source = Path(relative_path).read_text(encoding="utf-8-sig").lower()
            for token in forbidden_tokens:
                self.assertNotIn(token, source, msg=f"{relative_path}: {token}")

    def test_removed_local_ocr_backend_tokens_are_absent(self):
        files_to_check = [
            "app_logic.py",
            "worker_threads.py",
            "ocr_utils.py",
            "handlers/configuration_handler.py",
            "handlers/ui_interaction_handler.py",
            "requirements.txt",
            "setup.py",
        ]

        removed_backend = "tess" + "eract"
        removed_wrapper = "tess" + "erocr"
        old_python_wrapper = "py" + removed_backend

        for relative_path in files_to_check:
            source = Path(relative_path).read_text(encoding="utf-8-sig").lower()
            self.assertNotIn(removed_backend, source, msg=relative_path)
            self.assertNotIn(removed_wrapper, source, msg=relative_path)
            self.assertNotIn(old_python_wrapper, source, msg=relative_path)


class AdaptiveScanLoggingTests(unittest.TestCase):
    @staticmethod
    def _make_app(
        active_count=0,
        ocr_model="custom_ai",
        local_ocr_timing=None,
    ):
        import app_logic

        app = object.__new__(app_logic.GameChangingTranslator)
        app.load_check_timer = 0.0
        app.active_ocr_calls = set(range(active_count))
        app.max_concurrent_ocr_calls = 8
        app.current_scan_interval = 200
        app.base_scan_interval = 200
        app.overload_detected = False
        app.scan_interval_var = types.SimpleNamespace(get=lambda: 200)
        app.ocr_model_var = types.SimpleNamespace(get=lambda: ocr_model)
        app.runtime_metrics = types.SimpleNamespace(
            snapshot=lambda: {
                "timings": (
                    {"local_ocr_duration": local_ocr_timing}
                    if local_ocr_timing is not None
                    else {}
                )
            }
        )
        app._last_adaptive_log_state = None
        app._last_adaptive_log_time = 0.0
        return app

    def test_completed_local_ocr_work_records_dedicated_duration_metric(self):
        worker_threads = import_worker_threads_for_tests()
        image = Image.new("RGB", (16, 10), "white")
        metrics = RuntimeMetrics()
        app = types.SimpleNamespace(
            is_running=True,
            ocr_queue=queue.Queue(),
            get_ocr_model_setting=lambda: "paddleocr",
            is_api_based_ocr_model=lambda _model: False,
            ocr_debugging_var=types.SimpleNamespace(get=lambda: False),
            previous_text="",
            text_stability_counter=0,
            stable_threshold=2,
            is_placeholder_text=lambda _text: False,
            calculate_text_similarity=lambda _current, _previous: 0.0,
            reset_clear_timeout=Mock(),
            runtime_metrics=metrics,
        )
        app.ocr_queue.put_nowait(image)

        def stop_after_submit(
            _app,
            _text,
            _ocr_sequence_number,
            requested_at_monotonic=None,
        ):
            app.is_running = False

        with (
            patch.object(
                worker_threads,
                "process_local_ocr_frame",
                return_value=("Clear subtitle text.", None, "PaddleOCR"),
            ),
            patch.object(
                worker_threads,
                "start_async_translation",
                side_effect=stop_after_submit,
            ),
            patch.object(
                worker_threads.time,
                "sleep",
                side_effect=lambda _seconds: setattr(
                    app,
                    "is_running",
                    False,
                ),
            ),
        ):
            worker_threads.run_ocr_thread(app)

        self.assertIn(
            "local_ocr_duration",
            metrics.snapshot()["timings"],
        )

    def test_local_ocr_p50_raises_scan_interval_after_enough_samples(self):
        import app_logic

        app = self._make_app(
            ocr_model="paddleocr",
            local_ocr_timing={
                "count": 8,
                "latest": 0.219,
                "p50": 0.219,
                "p90": 0.359,
            },
        )
        with (
            patch.object(app_logic.time, "monotonic", return_value=2.1),
            patch.object(app_logic, "log_debug") as debug_log,
        ):
            app.update_adaptive_scan_interval()

        self.assertTrue(app.overload_detected)
        self.assertEqual(app.current_scan_interval, 250)
        self.assertIn("local ocr", debug_log.call_args.args[0].lower())

    def test_local_ocr_p50_requires_enough_samples(self):
        import app_logic

        app = self._make_app(
            ocr_model="paddleocr",
            local_ocr_timing={
                "count": 7,
                "latest": 0.219,
                "p50": 0.219,
                "p90": 0.359,
            },
        )
        with (
            patch.object(app_logic.time, "monotonic", return_value=2.1),
            patch.object(app_logic, "log_debug"),
        ):
            app.update_adaptive_scan_interval()

        self.assertFalse(app.overload_detected)
        self.assertEqual(app.current_scan_interval, 200)

    def test_switch_from_local_pacing_recomputes_api_overload_interval(self):
        import app_logic

        selected_model = types.SimpleNamespace(value="paddleocr")
        app = self._make_app(
            ocr_model="paddleocr",
            local_ocr_timing={
                "count": 8,
                "latest": 0.219,
                "p50": 0.219,
                "p90": 0.359,
            },
        )
        app.ocr_model_var = types.SimpleNamespace(
            get=lambda: selected_model.value
        )

        with (
            patch.object(
                app_logic.time,
                "monotonic",
                side_effect=[2.1, 4.2],
            ),
            patch.object(app_logic, "log_debug"),
        ):
            app.update_adaptive_scan_interval()
            selected_model.value = "custom_ai"
            app.active_ocr_calls = {1, 2}
            app.update_adaptive_scan_interval()

        self.assertTrue(app.overload_detected)
        self.assertEqual(app.current_scan_interval, 300)

    def test_switch_from_local_pacing_resets_api_moderate_interval(self):
        import app_logic

        selected_model = types.SimpleNamespace(value="paddleocr")
        app = self._make_app(
            ocr_model="paddleocr",
            local_ocr_timing={
                "count": 8,
                "latest": 0.5,
                "p50": 0.5,
                "p90": 0.6,
            },
        )
        app.ocr_model_var = types.SimpleNamespace(
            get=lambda: selected_model.value
        )

        with (
            patch.object(
                app_logic.time,
                "monotonic",
                side_effect=[2.1, 4.2],
            ),
            patch.object(app_logic, "log_debug"),
        ):
            app.update_adaptive_scan_interval()
            selected_model.value = "custom_ai"
            app.active_ocr_calls = {1}
            app.update_adaptive_scan_interval()

        self.assertFalse(app.overload_detected)
        self.assertEqual(app.current_scan_interval, 200)

    def test_malformed_local_ocr_timing_keeps_base_interval(self):
        import app_logic

        app = self._make_app(
            ocr_model="paddleocr",
            local_ocr_timing={"count": 8, "p50": "invalid"},
        )
        with (
            patch.object(app_logic.time, "monotonic", return_value=2.1),
            patch.object(app_logic, "log_debug"),
        ):
            app.update_adaptive_scan_interval()

        self.assertFalse(app.overload_detected)
        self.assertEqual(app.current_scan_interval, 200)

    def test_effective_limit_caps_custom_ai_and_preserves_generic_capacity(self):
        app = self._make_app()

        self.assertEqual(app.get_effective_ocr_concurrency_limit("custom_ai"), 2)
        self.assertEqual(app.get_effective_ocr_concurrency_limit("other_api"), 8)

    def test_effective_limit_preserves_zero_and_recovers_invalid_config(self):
        app = self._make_app()
        app.max_concurrent_ocr_calls = 0
        self.assertEqual(app.get_effective_ocr_concurrency_limit("custom_ai"), 0)

        app.max_concurrent_ocr_calls = "invalid"
        self.assertEqual(app.get_effective_ocr_concurrency_limit("custom_ai"), 1)

        app.max_concurrent_ocr_calls = float("inf")
        self.assertEqual(app.get_effective_ocr_concurrency_limit("custom_ai"), 1)

    def test_worker_limit_recovers_from_overflowing_compatibility_value(self):
        worker_threads = import_worker_threads_for_tests()
        app = types.SimpleNamespace(
            max_concurrent_ocr_calls=float("inf"),
            get_effective_ocr_concurrency_limit=lambda provider: float("inf"),
        )

        self.assertEqual(
            worker_threads._api_ocr_concurrency_limit(app, "custom_ai"),
            1,
        )

    def test_two_active_custom_ai_calls_trigger_adaptive_overload(self):
        import app_logic

        app = self._make_app(active_count=2, ocr_model="custom_ai")
        with (
            patch.object(app_logic.time, "monotonic", return_value=2.1),
            patch.object(app_logic, "log_debug") as debug_log,
        ):
            app.update_adaptive_scan_interval()

        self.assertTrue(app.overload_detected)
        self.assertEqual(app.current_scan_interval, 300)
        self.assertIn("overload detected", debug_log.call_args.args[0].lower())

    def test_six_active_generic_calls_keep_historical_overload_threshold(self):
        import app_logic

        app = self._make_app(active_count=6, ocr_model="other_api")
        with (
            patch.object(app_logic.time, "monotonic", return_value=2.1),
            patch.object(app_logic, "log_debug"),
        ):
            app.update_adaptive_scan_interval()

        self.assertTrue(app.overload_detected)
        self.assertEqual(app.current_scan_interval, 300)

    def test_unchanged_adaptive_state_logs_once_inside_heartbeat_window(self):
        import app_logic

        app = self._make_app(active_count=0)
        with (
            patch.object(
                app_logic.time,
                "monotonic",
                side_effect=[2.1, 4.2, 6.3],
            ),
            patch.object(app_logic, "log_debug") as debug_log,
        ):
            app.update_adaptive_scan_interval()
            app.update_adaptive_scan_interval()
            app.update_adaptive_scan_interval()

        self.assertEqual(debug_log.call_count, 1)

    def test_adaptive_state_transition_logs_immediately(self):
        import app_logic

        app = self._make_app(active_count=0)
        with (
            patch.object(
                app_logic.time,
                "monotonic",
                side_effect=[2.1, 4.2],
            ),
            patch.object(app_logic, "log_debug") as debug_log,
        ):
            app.update_adaptive_scan_interval()
            app.active_ocr_calls = set(range(6))
            app.update_adaptive_scan_interval()

        self.assertEqual(debug_log.call_count, 2)
        self.assertIn("overload detected", debug_log.call_args.args[0].lower())


class LiveCustomAIModelSwitchTests(unittest.TestCase):
    def test_profile_model_selection_reports_persistence_error(self):
        import gui_builder

        profile = {"id": "relay", "name": "Relay", "model": "old"}

        class Profiles:
            def get_profile(self, profile_id):
                return profile

            def update_profile(self, profile_id, **updates):
                raise RuntimeError("disk unavailable")

        app = types.SimpleNamespace(
            ai_profile_selected_id="relay",
            ai_profile_model_var=types.SimpleNamespace(get=lambda: "new"),
            custom_ai_profiles=Profiles(),
            ui_lang=types.SimpleNamespace(
                get_label=lambda key, fallback: fallback
            ),
            root=object(),
        )

        with patch.object(gui_builder.messagebox, "showerror") as showerror:
            try:
                changed = gui_builder.apply_custom_ai_profile_model_selection(
                    app
                )
            except RuntimeError as error:
                self.fail(f"Tk callback leaked persistence error: {error}")

        self.assertFalse(changed)
        showerror.assert_called_once()

    def test_active_profile_model_selection_persists_and_refreshes_translation(self):
        import gui_builder

        self.assertTrue(
            hasattr(gui_builder, "apply_custom_ai_profile_model_selection"),
            "profile model selections need an immediate apply helper",
        )

        profile = {
            "id": "relay",
            "name": "Relay",
            "model": "gpt-5.6-sol",
        }

        class Profiles:
            def get_profile(self, profile_id):
                return profile if profile_id == "relay" else None

            def get_active_profile(self, kind):
                return profile if kind == "translation" else None

            def update_profile(self, profile_id, **updates):
                self.updated = (profile_id, updates)
                profile.update(updates)
                return profile

        profiles = Profiles()
        app = types.SimpleNamespace(
            ai_profile_selected_id="relay",
            ai_profile_model_var=types.SimpleNamespace(
                get=lambda: "gpt-5.5-sol"
            ),
            custom_ai_profiles=profiles,
            translation_handler=types.SimpleNamespace(
                _clear_active_context=Mock()
            ),
            is_running=True,
        )

        with patch(
            "worker_threads.refresh_translation_after_profile_change",
            create=True,
        ) as refresh:
            changed = gui_builder.apply_custom_ai_profile_model_selection(app)

        self.assertTrue(changed)
        self.assertEqual(
            profiles.updated,
            ("relay", {"model": "gpt-5.5-sol"}),
        )
        app.translation_handler._clear_active_context.assert_called_once_with()
        refresh.assert_called_once_with(
            app,
            reason="active profile model changed",
        )

    def test_inactive_profile_model_selection_does_not_refresh_translation(self):
        import gui_builder

        self.assertTrue(
            hasattr(gui_builder, "apply_custom_ai_profile_model_selection"),
            "profile model selections need an immediate apply helper",
        )

        edited = {"id": "other", "name": "Other", "model": "old"}
        active = {"id": "active", "name": "Active", "model": "live"}

        class Profiles:
            def get_profile(self, profile_id):
                return edited if profile_id == "other" else active

            def get_active_profile(self, kind):
                return active

            def update_profile(self, profile_id, **updates):
                edited.update(updates)
                return edited

        app = types.SimpleNamespace(
            ai_profile_selected_id="other",
            ai_profile_model_var=types.SimpleNamespace(get=lambda: "new"),
            custom_ai_profiles=Profiles(),
            translation_handler=types.SimpleNamespace(
                _clear_active_context=Mock()
            ),
            is_running=True,
        )

        with patch(
            "worker_threads.refresh_translation_after_profile_change",
            create=True,
        ) as refresh:
            changed = gui_builder.apply_custom_ai_profile_model_selection(app)

        self.assertTrue(changed)
        self.assertEqual(edited["model"], "new")
        app.translation_handler._clear_active_context.assert_not_called()
        refresh.assert_not_called()

    def test_profile_change_invalidates_cooldown_timer_and_wakes_latest_subtitle(self):
        worker_threads = import_worker_threads_for_tests()
        self.assertTrue(
            hasattr(
                worker_threads,
                "refresh_translation_after_profile_change",
            ),
            "profile changes need a scheduler refresh entry point",
        )
        scheduled = []
        pending = {
            "text": "latest subtitle",
            "ocr_sequence_number": 8,
            "requested_at_monotonic": 99.0,
        }
        app = types.SimpleNamespace(
            is_running=True,
            root=types.SimpleNamespace(
                after=lambda delay, callback, *args: scheduled.append(
                    (delay, callback, args)
                )
            ),
            pending_translation_request=pending,
            pending_translation_flush_scheduled=True,
            pending_translation_flush_deadline_monotonic=160.0,
            pending_translation_flush_generation=4,
        )

        refreshed = worker_threads.refresh_translation_after_profile_change(
            app,
            reason="active profile model changed",
        )

        self.assertTrue(refreshed)
        self.assertIsNone(app.pending_translation_request)
        self.assertFalse(app.pending_translation_flush_scheduled)
        self.assertEqual(app.pending_translation_flush_generation, 5)
        self.assertEqual(len(scheduled), 1)
        delay, callback, args = scheduled[0]
        self.assertEqual(delay, 0)
        self.assertIs(
            callback,
            worker_threads._apply_translation_profile_refresh,
        )
        self.assertIs(args[0], app)
        self.assertEqual(args[1], 1)
        self.assertEqual(args[2], pending)
        self.assertEqual(args[3], "active profile model changed")

    def test_profile_refresh_schedule_failure_preserves_pending_request(self):
        worker_threads = import_worker_threads_for_tests()
        pending = {
            "text": "latest subtitle",
            "ocr_sequence_number": 8,
            "requested_at_monotonic": 99.0,
        }
        app = types.SimpleNamespace(
            is_running=True,
            root=types.SimpleNamespace(
                after=Mock(side_effect=RuntimeError("Tk closing"))
            ),
            pending_translation_request=pending,
            pending_translation_flush_scheduled=True,
            pending_translation_flush_deadline_monotonic=160.0,
            pending_translation_flush_generation=4,
        )

        with self.assertRaisesRegex(RuntimeError, "Tk closing"):
            worker_threads.refresh_translation_after_profile_change(app)

        self.assertIs(app.pending_translation_request, pending)
        self.assertTrue(app.pending_translation_flush_scheduled)
        self.assertEqual(app.pending_translation_flush_deadline_monotonic, 160.0)
        self.assertEqual(app.pending_translation_flush_generation, 4)

    def test_profile_refresh_callback_uses_newer_translation_candidate(self):
        worker_threads = import_worker_threads_for_tests()
        self.assertTrue(
            hasattr(
                worker_threads,
                "_apply_translation_profile_refresh",
            ),
            "profile refresh needs a generation-checked callback",
        )
        scheduled = []
        app = types.SimpleNamespace(
            is_running=True,
            root=types.SimpleNamespace(
                after=lambda delay, callback, *args: scheduled.append(
                    (delay, callback, args)
                )
            ),
            latest_translation_candidate={
                "text": "older subtitle",
                "ocr_sequence_number": 3,
                "requested_at_monotonic": 50.0,
            },
            pending_translation_request=None,
            pending_translation_flush_scheduled=False,
            pending_translation_flush_deadline_monotonic=0.0,
            pending_translation_flush_generation=2,
        )

        self.assertTrue(
            worker_threads.refresh_translation_after_profile_change(app)
        )
        app.latest_translation_candidate = {
            "text": "newer subtitle",
            "ocr_sequence_number": 4,
            "requested_at_monotonic": 51.0,
        }
        _delay, callback, args = scheduled[0]

        with patch.object(
            worker_threads,
            "start_async_translation",
        ) as start_translation:
            callback(*args)

        start_translation.assert_called_once_with(
            app,
            "newer subtitle",
            4,
            requested_at_monotonic=51.0,
            configuration_refresh=True,
        )

    def test_profile_change_can_use_bounded_overflow_without_age_wait(self):
        import inspect

        worker_threads = import_worker_threads_for_tests()
        self.assertIn(
            "configuration_refresh",
            inspect.signature(worker_threads.start_async_translation).parameters,
            "configuration refreshes need an explicit bounded-priority flag",
        )

        class Handler:
            def get_cached_translation_for_display(self, text):
                return None

            def get_custom_ai_translation_request_snapshot(
                self,
                text,
                commit=False,
            ):
                return {
                    "inflight_key": ("custom_ai", text, "new-model"),
                    "latency_mode": "safe",
                }

            def get_translation_submit_interval_seconds(self, text_content=None):
                return 5.0

            def get_translation_provider_cooldown_seconds(self, latency_mode=None):
                return 0.0

            def get_translation_concurrency_limit(self):
                return 1

        app = types.SimpleNamespace(
            translation_sequence_counter=1,
            active_translation_calls={1},
            active_translation_inflight_keys={
                ("custom_ai", "old subtitle", "old-model")
            },
            active_translation_started_monotonic={1: 100.0},
            translation_thread_pool=Mock(),
            translation_handler=Handler(),
            enable_instant_cache_display_var=types.SimpleNamespace(
                get=lambda: False
            ),
            root=types.SimpleNamespace(after=Mock()),
            initialize_async_translation_infrastructure=lambda: None,
            last_translation_submit_monotonic=100.0,
            pending_translation_request=None,
            pending_translation_flush_scheduled=False,
            pending_translation_flush_deadline_monotonic=0.0,
            pending_translation_flush_generation=0,
        )

        with patch.object(worker_threads.time, "monotonic", return_value=100.1):
            worker_threads.start_async_translation(
                app,
                "latest subtitle",
                9,
                requested_at_monotonic=100.0,
                configuration_refresh=True,
            )

        app.translation_thread_pool.submit.assert_called_once()
        self.assertEqual(app.translation_sequence_counter, 2)
        self.assertEqual(app.active_translation_calls, {1, 2})
        self.assertIsNone(app.pending_translation_request)

    def test_profile_change_overflow_handles_missing_active_start_time(self):
        worker_threads = import_worker_threads_for_tests()

        class Handler:
            def get_cached_translation_for_display(self, text):
                return None

            def get_custom_ai_translation_request_snapshot(
                self,
                text,
                commit=False,
            ):
                return {
                    "inflight_key": ("custom_ai", text, "new-model"),
                    "latency_mode": "safe",
                }

            def get_translation_submit_interval_seconds(self, text_content=None):
                return 0.0

            def get_translation_provider_cooldown_seconds(self, latency_mode=None):
                return 0.0

            def get_translation_concurrency_limit(self):
                return 1

        app = types.SimpleNamespace(
            translation_sequence_counter=1,
            active_translation_calls={1},
            active_translation_inflight_keys={
                ("custom_ai", "old subtitle", "old-model")
            },
            active_translation_started_monotonic={},
            translation_thread_pool=Mock(),
            translation_handler=Handler(),
            enable_instant_cache_display_var=types.SimpleNamespace(
                get=lambda: False
            ),
            root=types.SimpleNamespace(after=Mock()),
            initialize_async_translation_infrastructure=lambda: None,
            last_translation_submit_monotonic=100.0,
            pending_translation_request=None,
            pending_translation_flush_scheduled=False,
            pending_translation_flush_deadline_monotonic=0.0,
            pending_translation_flush_generation=0,
        )

        with patch.object(worker_threads.time, "monotonic", return_value=100.1):
            worker_threads.start_async_translation(
                app,
                "latest subtitle",
                9,
                requested_at_monotonic=100.0,
                configuration_refresh=True,
            )

        app.translation_thread_pool.submit.assert_called_once()


class LiveCustomAIProfileSwitchTests(unittest.TestCase):
    class Value:
        def __init__(self, value):
            self.value = value

        def get(self):
            return self.value

        def set(self, value):
            self.value = value

    class Profiles:
        def __init__(self, active_id="failed", fail=False):
            self.profiles = [
                {"id": "failed", "name": "Failed", "model": "gpt-5.6"},
                {"id": "working", "name": "Working", "model": "grok-fast"},
            ]
            self.active_id = active_id
            self.fail = fail
            self.set_calls = []

        def list_profiles(self, enabled_only=False):
            return list(self.profiles)

        def get_active_profile(self, kind):
            if kind != "translation":
                return None
            return next(
                profile
                for profile in self.profiles
                if profile["id"] == self.active_id
            )

        def set_active_profile(self, kind, profile_id):
            self.set_calls.append((kind, profile_id))
            if self.fail:
                raise RuntimeError("profile store unavailable")
            self.active_id = profile_id
            return self.get_active_profile(kind)

    @classmethod
    def _make_app(cls, profiles, running=True):
        return types.SimpleNamespace(
            custom_ai_profiles=profiles,
            translation_model_var=cls.Value("custom_ai"),
            translation_handler=types.SimpleNamespace(
                _clear_active_context=Mock()
            ),
            is_running=running,
            ui_lang=types.SimpleNamespace(
                get_label=lambda _key, default=None: default or _key
            ),
            root=object(),
        )

    def test_running_profile_switch_clears_context_and_wakes_latest_subtitle(self):
        import gui_builder

        profiles = self.Profiles()
        app = self._make_app(profiles, running=True)

        with patch(
            "worker_threads.refresh_translation_after_profile_change",
        ) as refresh:
            changed = (
                gui_builder.apply_custom_ai_translation_profile_selection(
                    app,
                    "Working",
                )
            )

        self.assertTrue(changed)
        self.assertEqual(profiles.active_id, "working")
        self.assertEqual(app.translation_model_var.value, "custom_ai")
        app.translation_handler._clear_active_context.assert_called_once_with()
        refresh.assert_called_once_with(
            app,
            reason="active translation profile changed",
        )

    def test_same_profile_selection_is_a_noop(self):
        import gui_builder

        profiles = self.Profiles(active_id="working")
        app = self._make_app(profiles, running=True)

        with patch(
            "worker_threads.refresh_translation_after_profile_change",
        ) as refresh:
            changed = (
                gui_builder.apply_custom_ai_translation_profile_selection(
                    app,
                    "Working",
                )
            )

        self.assertFalse(changed)
        self.assertEqual(profiles.set_calls, [])
        app.translation_handler._clear_active_context.assert_not_called()
        refresh.assert_not_called()

    def test_stopped_profile_switch_persists_without_scheduling_work(self):
        import gui_builder

        profiles = self.Profiles()
        app = self._make_app(profiles, running=False)

        with patch(
            "worker_threads.refresh_translation_after_profile_change",
        ) as refresh:
            changed = (
                gui_builder.apply_custom_ai_translation_profile_selection(
                    app,
                    "Working",
                )
            )

        self.assertTrue(changed)
        self.assertEqual(profiles.active_id, "working")
        app.translation_handler._clear_active_context.assert_called_once_with()
        refresh.assert_not_called()

    def test_profile_switch_persistence_error_is_reported(self):
        import gui_builder

        profiles = self.Profiles(fail=True)
        app = self._make_app(profiles, running=True)

        with patch.object(gui_builder.messagebox, "showerror") as show_error:
            changed = (
                gui_builder.apply_custom_ai_translation_profile_selection(
                    app,
                    "Working",
                )
            )

        self.assertFalse(changed)
        show_error.assert_called_once()
        app.translation_handler._clear_active_context.assert_not_called()

    def test_primary_profile_selection_handler_uses_live_apply_helper(self):
        import gui_builder

        events = []
        app = types.SimpleNamespace(
            on_translation_model_selection_changed=lambda **kwargs: events.append(
                ("changed", kwargs)
            )
        )

        with patch.object(
            gui_builder,
            "apply_custom_ai_translation_profile_selection",
            create=True,
            side_effect=lambda _app, selected: events.append(
                ("applied", selected)
            ) or True,
        ) as apply_profile:
            changed = gui_builder.handle_translation_profile_selection(
                app,
                "Working",
                event="event",
            )

        self.assertTrue(changed)
        apply_profile.assert_called_once_with(app, "Working")
        self.assertEqual(
            events,
            [
                ("applied", "Working"),
                (
                    "changed",
                    {
                        "event": None,
                        "initial_setup": False,
                        "synchronize_ui_only": True,
                    },
                ),
            ],
        )

    def test_ui_only_profile_sync_skips_session_lifecycle_and_settings_save(self):
        import app_logic

        app = object.__new__(app_logic.GameChangingTranslator)
        app.is_running = True
        app._fully_initialized = True
        app.translation_model_var = self.Value("custom_ai")
        app.translation_handler = types.SimpleNamespace(
            request_end_translation_session=Mock(),
            start_translation_session=Mock(),
        )
        app.ui_interaction_handler = types.SimpleNamespace(
            on_translation_model_selection_changed=Mock()
        )
        app.save_settings = Mock()

        app.on_translation_model_selection_changed(
            event=None,
            initial_setup=False,
            synchronize_ui_only=True,
        )

        app.translation_handler.request_end_translation_session.assert_not_called()
        app.translation_handler.start_translation_session.assert_not_called()
        app.ui_interaction_handler.on_translation_model_selection_changed.assert_called_once_with(
            None,
            False,
        )
        app.save_settings.assert_not_called()


if __name__ == "__main__":
    unittest.main()
