import inspect
import types
import unittest
from unittest.mock import Mock, patch

import app_configuration
import app_lifecycle
import app_logic
import worker_capture


class _Widget:
    def config(self, **_kwargs):
        return None

    def winfo_exists(self):
        return True


class _Root:
    def update_idletasks(self):
        return None

    def winfo_exists(self):
        return True


class _Overlay:
    def __init__(self):
        self.visible = False

    def get_geometry(self):
        return (0, 0, 100, 100)

    def winfo_exists(self):
        return True

    def winfo_viewable(self):
        return self.visible

    def show(self):
        self.visible = True


class PaddleOCRLazyStartTests(unittest.TestCase):
    def _build_app(self):
        app = object.__new__(app_logic.GameChangingTranslator)
        app.root = _Root()
        app.start_stop_btn = _Widget()
        app.status_label = _Widget()
        app.source_overlay = _Overlay()
        app.target_overlay = _Overlay()
        app.translation_text = _Widget()
        app.threads = []
        app.is_running = False
        app.toggle_in_progress = False
        app._app_is_closing = False
        app._shutdown_finalized = False
        app.KEYBOARD_AVAILABLE = False
        app.ocr_queue = object()
        app.translation_queue = object()
        app.ui_lang = types.SimpleNamespace(
            get_label=lambda _key, fallback=None, **_kwargs: fallback or "label"
        )
        app.custom_ai_profiles = types.SimpleNamespace(
            get_active_profile=lambda role: {"id": "profile"}
            if role == "translation"
            else None
        )
        app.translation_handler = types.SimpleNamespace(
            start_ocr_session=Mock(),
            start_translation_session=Mock(),
        )
        app.get_ocr_model_setting = lambda: "paddleocr"
        app.is_api_based_ocr_model = lambda: False
        app._ensure_overlays_for_start = lambda: True
        app._widget_exists_safely = lambda widget: widget is not None
        app._validate_area_coords = lambda _area, _name: True
        app._reset_gemini_batch_state = Mock()
        app._clear_queue = Mock()
        app._reset_translation_scheduler_session_state = Mock()
        app.clear_ocr_stability_gate = Mock()
        app.publish_capture_ui_snapshot = Mock()
        app.start_capture_ui_snapshot_refresh = Mock()
        app._set_async_submissions_frozen = Mock()
        app.ensure_paddleocr_ready_if_selected = Mock(
            side_effect=AssertionError("PaddleOCR prewarm must not run")
        )
        app.wait_for_paddleocr_prewarm = Mock(
            side_effect=AssertionError("PaddleOCR start must not wait")
        )
        app._start_translation_workers = (
            app_lifecycle.AppLifecycleMixin._start_translation_workers.__get__(
                app, type(app)
            )
        )
        return app

    def test_paddleocr_start_launches_workers_without_prewarm_or_wait(self):
        app = self._build_app()
        started_names = []

        class FakeThread:
            def __init__(self, target=None, args=(), name=None, daemon=None):
                self.name = name

            def start(self):
                started_names.append(self.name)

            def is_alive(self):
                return False

        with patch("app_lifecycle.threading.Thread", side_effect=FakeThread), patch(
            "worker_threads.run_capture_thread", create=True
        ), patch("worker_threads.run_ocr_thread", create=True), patch(
            "worker_threads.run_translation_thread", create=True
        ):
            app_lifecycle.AppLifecycleMixin.toggle_translation(app)

        self.assertTrue(app.is_running)
        self.assertEqual(
            started_names,
            ["CaptureThread", "OCRThread", "TranslationThread"],
        )
        app.ensure_paddleocr_ready_if_selected.assert_not_called()
        app.wait_for_paddleocr_prewarm.assert_not_called()

    def test_runtime_paths_do_not_schedule_or_wait_for_paddleocr(self):
        readiness_source = inspect.getsource(
            app_logic.GameChangingTranslator.schedule_initial_ui_readiness
        )
        save_source = inspect.getsource(app_configuration.AppConfigurationMixin.save_settings)
        capture_source = inspect.getsource(worker_capture.process_local_ocr_frame)
        start_source = inspect.getsource(app_lifecycle.AppLifecycleMixin.toggle_translation)

        self.assertNotIn("schedule_initial_paddleocr_prewarm", readiness_source)
        self.assertNotIn("ensure_paddleocr_ready_if_selected", save_source)
        self.assertNotIn("wait_for_paddleocr_prewarm", capture_source)
        self.assertNotIn("_begin_paddleocr_start_wait", start_source)


if __name__ == "__main__":
    unittest.main()
