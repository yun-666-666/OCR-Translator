import threading
import time
import types
import unittest
from unittest.mock import Mock, patch

from PIL import Image

from app_capture_ocr import AppCaptureOcrMixin
from paddle_ocr_backend import PADDLEOCR_MODEL_CODE, PaddleOCRSettings
from rapid_ocr_backend import RAPIDOCR_MODEL_CODE, RapidOCRSettings


class _DummyVar:
    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value


class _FakeWidget:
    def __init__(self, exists=True):
        self._exists = exists
        self.image = None
        self.texts = []
        self.configs = []

    def winfo_exists(self):
        return self._exists

    def configure(self, **kwargs):
        self.configs.append(kwargs)
        if "image" in kwargs:
            self.image = kwargs["image"]

    def config(self, **kwargs):
        self.configure(**kwargs)

    def update_idletasks(self):
        return None

    def winfo_reqwidth(self):
        return 120

    def winfo_reqheight(self):
        return 40

    def delete(self, *_args, **_kwargs):
        return None

    def insert(self, _index, text):
        self.texts.append(text)


class _FakeCanvas:
    def __init__(self):
        self.item_configs = []
        self.configs = []

    def configure(self, **kwargs):
        self.configs.append(kwargs)

    def itemconfig(self, *_args, **kwargs):
        self.item_configs.append(kwargs)


class _FakeUILang:
    def get_label(self, _key, default=""):
        return default


class _PreviewHost(AppCaptureOcrMixin):
    def __init__(self):
        self.root = types.SimpleNamespace(after=self._after)
        self.ui_lang = _FakeUILang()
        self.ocr_preview_window = _FakeWidget(exists=True)
        self.preview_image_label = _FakeWidget(exists=True)
        self.preview_text_widget = _FakeWidget(exists=True)
        self.preview_image_canvas = _FakeCanvas()
        self.preview_image_canvas_item = 1
        self.source_overlay = None
        self.last_screenshot = Image.new("RGB", (8, 4), color=(12, 34, 56))
        self.last_processed_subtitle = "main worker text"
        self.keep_linebreaks_var = _DummyVar(False)
        self.md3_palette = None
        self._after_calls = []
        self._ocr_model = PADDLEOCR_MODEL_CODE

    def _after(self, delay_ms, callback):
        self._after_calls.append((delay_ms, callback))
        return len(self._after_calls)

    def get_ocr_model_setting(self):
        return self._ocr_model


class PreviewLatestOnlyOcrTests(unittest.TestCase):
    def test_refresh_does_not_run_paddleocr_on_ui_thread(self):
        host = _PreviewHost()
        settings = PaddleOCRSettings()
        recognize = Mock(side_effect=AssertionError("OCR must not run on UI path"))

        with patch(
            "worker_threads.get_paddleocr_settings_from_app",
            return_value=settings,
        ), patch(
            "app_capture_ocr.recognize_with_paddleocr",
            recognize,
        ), patch.object(
            host,
            "_schedule_preview_ocr",
            return_value=True,
        ) as schedule:
            host.refresh_ocr_preview()

        schedule.assert_called_once()
        recognize.assert_not_called()
        self.assertEqual(host.preview_text_widget.texts[-1], "main worker text")

    def test_refresh_rapidocr_uses_current_minimum_score_settings(self):
        host = _PreviewHost()
        host._ocr_model = RAPIDOCR_MODEL_CODE
        settings = RapidOCRSettings(min_score=0.82)

        with patch(
            "worker_threads.get_rapidocr_settings_from_app",
            return_value=settings,
        ), patch.object(
            host,
            "_schedule_preview_ocr",
            return_value=True,
        ) as schedule:
            host.refresh_ocr_preview()

        schedule.assert_called_once()
        self.assertIs(schedule.call_args.args[1], settings)

    def test_high_frequency_refresh_keeps_one_inflight_and_latest_pending(self):
        host = _PreviewHost()
        host._ensure_preview_ocr_runtime()
        host._apply_preview_image = Mock()
        host._apply_preview_text = Mock()
        started = threading.Event()
        release = threading.Event()
        call_count = {"n": 0}
        seen_frames = []

        def slow_job(screenshot_pil, paddleocr_settings, keep_linebreaks):
            call_count["n"] += 1
            seen_frames.append(screenshot_pil)
            if call_count["n"] == 1:
                started.set()
                self.assertTrue(release.wait(timeout=2.0))
            return screenshot_pil, f"text-{call_count['n']}"

        host._run_preview_ocr_job = slow_job
        settings = PaddleOCRSettings()
        frame_a = Image.new("RGB", (4, 4), color=(1, 0, 0))
        frame_b = Image.new("RGB", (4, 4), color=(0, 1, 0))
        frame_c = Image.new("RGB", (4, 4), color=(0, 0, 1))

        self.assertTrue(host._schedule_preview_ocr(frame_a, settings, False))
        self.assertTrue(started.wait(timeout=1.0))
        self.assertFalse(host._schedule_preview_ocr(frame_b, settings, False))
        self.assertFalse(host._schedule_preview_ocr(frame_c, settings, False))

        with host._preview_ocr_lock:
            self.assertTrue(host._preview_ocr_in_flight)
            pending = host._preview_ocr_pending_frame
        self.assertIsNotNone(pending)
        self.assertIs(pending["screenshot"], frame_c)

        release.set()
        settled = False
        for _ in range(100):
            # Drain UI completion callbacks posted via root.after(0, ...).
            callbacks = list(host._after_calls)
            host._after_calls.clear()
            for _delay, callback in callbacks:
                callback()
            with host._preview_ocr_lock:
                idle = (
                    not host._preview_ocr_in_flight
                    and host._preview_ocr_pending_frame is None
                )
            if idle and call_count["n"] >= 2:
                settled = True
                break
            time.sleep(0.01)
        self.assertTrue(settled)
        self.assertEqual(call_count["n"], 2)
        self.assertIs(seen_frames[0], frame_a)
        self.assertIs(seen_frames[1], frame_c)

    def test_closed_preview_discards_stale_future_without_ui_write(self):
        host = _PreviewHost()
        host._ensure_preview_ocr_runtime()
        generation = host._preview_ocr_generation
        host.close_ocr_preview = types.MethodType(
            lambda self: self._bump_preview_ocr_generation("preview closed"),
            host,
        )
        host.close_ocr_preview()
        host.ocr_preview_window = None
        host.preview_text_widget.texts.clear()

        host._finish_preview_ocr_job(
            generation,
            (Image.new("RGB", (2, 2), color=(9, 9, 9)), "stale text"),
        )

        self.assertEqual(host.preview_text_widget.texts, [])
        self.assertIsNone(host.ocr_preview_window)

    def test_worker_never_falls_back_to_tk_ui_when_after_fails(self):
        host = _PreviewHost()
        host._ensure_preview_ocr_runtime()
        host.root = types.SimpleNamespace(
            after=Mock(side_effect=RuntimeError("Tk is closing"))
        )
        host._run_preview_ocr_job = Mock(
            return_value=(host.last_screenshot, "worker result")
        )

        with patch.object(host, "_finish_preview_ocr_job") as finish:
            self.assertTrue(
                host._schedule_preview_ocr(
                    host.last_screenshot,
                    PaddleOCRSettings(),
                    False,
                )
            )
            for _ in range(100):
                with host._preview_ocr_lock:
                    if not host._preview_ocr_in_flight:
                        break
                time.sleep(0.01)

        finish.assert_not_called()
        with host._preview_ocr_lock:
            self.assertFalse(host._preview_ocr_in_flight)
            self.assertIsNone(host._preview_ocr_pending_frame)
        host.shutdown_preview_ocr_executor()

    def test_shutdown_preview_executor_cancels_pending_work(self):
        host = _PreviewHost()
        host._ensure_preview_ocr_runtime()
        executor = Mock()
        host._preview_ocr_executor = executor
        host._preview_ocr_pending_frame = {"frame": "pending"}
        previous_generation = host._preview_ocr_generation

        self.assertTrue(host.shutdown_preview_ocr_executor())
        executor.shutdown.assert_called_once_with(wait=False, cancel_futures=True)
        self.assertEqual(host._preview_ocr_generation, previous_generation + 1)
        self.assertIsNone(host._preview_ocr_pending_frame)


if __name__ == "__main__":
    unittest.main()
