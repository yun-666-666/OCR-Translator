import types
import unittest
from unittest.mock import Mock, patch

import app_lifecycle


class _Widget:
    def __init__(self):
        self.text = ""
        self.state = "normal"
        self.configs = []

    def config(self, **kwargs):
        self.configs.append(dict(kwargs))
        if "text" in kwargs:
            self.text = kwargs["text"]
        if "state" in kwargs:
            self.state = kwargs["state"]

    def winfo_exists(self):
        return True


class _Root:
    def __init__(self):
        self.callbacks = []
        self.update_calls = 0

    def after(self, delay_ms, callback):
        self.callbacks.append((int(delay_ms), callback))
        return len(self.callbacks)

    def after_cancel(self, handle):
        return None

    def update_idletasks(self):
        self.update_calls += 1

    def winfo_exists(self):
        return True


class _Overlay:
    def __init__(self, geometry=(0, 0, 100, 100)):
        self.geometry = geometry
        self.visible = False

    def get_geometry(self):
        return self.geometry

    def winfo_exists(self):
        return True

    def winfo_viewable(self):
        return self.visible

    def show(self):
        self.visible = True


class PaddleOCRStartWaitTests(unittest.TestCase):
    def _settings(self, model_size="tiny"):
        return types.SimpleNamespace(
            ocr_version="PP-OCRv6",
            model_size=model_size,
            device="cpu",
            lang="en",
        )

    def _build_app(self, ocr_model="paddleocr", ready=False):
        import app_logic

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
            get_label=lambda key, fallback=None, **kwargs: (
                (fallback or key).format(**kwargs) if kwargs else (fallback or key)
            )
        )
        app.custom_ai_profiles = types.SimpleNamespace(
            get_active_profile=lambda role: (
                {"id": "profile"} if role == "translation" else None
            )
        )
        app.translation_handler = types.SimpleNamespace(
            start_ocr_session=Mock(),
            start_translation_session=Mock(),
        )
        app.get_ocr_model_setting = lambda: ocr_model
        app.is_api_based_ocr_model = lambda: ocr_model == "custom_ai"
        app._ensure_overlays_for_start = lambda: True
        app._widget_exists_safely = lambda widget: widget is not None
        app._validate_area_coords = lambda area, name: True
        app._reset_gemini_batch_state = Mock()
        app._clear_queue = Mock()
        app._reset_translation_scheduler_session_state = Mock()
        app.clear_ocr_stability_gate = Mock()
        app.publish_capture_ui_snapshot = Mock()
        app.start_capture_ui_snapshot_refresh = Mock()
        app.stop_capture_ui_snapshot_refresh = Mock()
        settings = self._settings()
        app._paddleocr_settings = settings
        app._paddleocr_ready = ready
        app._paddleocr_prewarm_generation = 1
        app._paddleocr_prewarm_metrics = {
            "status": "running" if not ready else "completed",
            "ready": bool(ready),
            "generation": 1,
            "error_type": "",
            "active": not ready,
        }
        app._paddleocr_start_wait = None

        def get_settings():
            return app._paddleocr_settings

        def is_ready(candidate=None):
            if candidate is not None and candidate != app._paddleocr_settings:
                return False
            return bool(app._paddleocr_ready)

        def ensure_ready(reason="PaddleOCR selected"):
            app._ensure_calls = getattr(app, "_ensure_calls", 0) + 1
            app._last_ensure_reason = reason
            return True

        def snapshot():
            return dict(app._paddleocr_prewarm_metrics)

        app.get_current_paddleocr_settings = get_settings
        app.is_paddleocr_ready_for_settings = is_ready
        app.ensure_paddleocr_ready_if_selected = ensure_ready
        app.get_paddleocr_prewarm_metrics_snapshot = snapshot
        app.start_paddleocr_prewarm = Mock(return_value=False)
        # Bind lifecycle helpers so object.__new__ instances still work.
        for name in (
            "_get_paddleocr_start_wait_state",
            "_is_waiting_for_paddleocr_start",
            "_format_paddleocr_loading_status",
            "_clear_paddleocr_start_wait",
            "_cancel_paddleocr_start_wait",
            "_fail_paddleocr_start_wait",
            "_begin_paddleocr_start_wait",
            "_schedule_paddleocr_start_wait_poll",
            "_poll_paddleocr_start_wait",
            "_start_translation_workers",
            "toggle_translation",
        ):
            setattr(
                app,
                name,
                getattr(app_lifecycle.AppLifecycleMixin, name).__get__(app, type(app)),
            )
        return app

    def _run_pending_callbacks(self, app, max_steps=20):
        steps = 0
        while app.root.callbacks and steps < max_steps:
            _delay, callback = app.root.callbacks.pop(0)
            callback()
            steps += 1
        return steps

    def test_not_ready_defers_start_and_polls_with_after(self):
        import app_lifecycle

        app = self._build_app(ready=False)
        worker_starts = []

        with patch(
            "app_lifecycle.threading.Thread",
            side_effect=lambda *args, **kwargs: worker_starts.append(kwargs.get("name") or args)
            or types.SimpleNamespace(start=Mock(), name=kwargs.get("name"), is_alive=lambda: False),
        ):
            with patch(
                "worker_threads.run_capture_thread",
                create=True,
            ), patch(
                "worker_threads.run_ocr_thread",
                create=True,
            ), patch(
                "worker_threads.run_translation_thread",
                create=True,
            ):
                app_lifecycle.AppLifecycleMixin.toggle_translation(app)

        self.assertFalse(app.is_running)
        self.assertEqual(worker_starts, [])
        self.assertFalse(app.translation_handler.start_translation_session.called)
        self.assertFalse(app.translation_handler.start_ocr_session.called)
        self.assertIsNotNone(app._paddleocr_start_wait)
        self.assertEqual(app._paddleocr_start_wait["state"], "waiting")
        self.assertTrue(app.root.callbacks)
        self.assertEqual(app.start_stop_btn.text, "Stop")
        self.assertIn("Local OCR loading", app.status_label.text)
        self.assertGreaterEqual(getattr(app, "_ensure_calls", 0), 1)

    def test_completed_different_prewarm_retries_matching_settings_once(self):
        import app_lifecycle

        app = self._build_app(ready=False)
        started_names = []

        class FakeThread:
            def __init__(self, target=None, args=(), name=None, daemon=None):
                self.name = name

            def start(self):
                started_names.append(self.name)

            def is_alive(self):
                return False

        with patch("app_lifecycle.threading.Thread", side_effect=FakeThread):
            app_lifecycle.AppLifecycleMixin.toggle_translation(app)
            self.assertEqual(app._ensure_calls, 1)

            # The prewarm that blocked Start finished, but it was for different
            # settings, so the captured settings are still not ready.
            app._paddleocr_prewarm_metrics = {
                "status": "completed",
                "ready": True,
                "generation": 1,
                "error_type": "",
                "active": False,
            }
            self._run_pending_callbacks(app, max_steps=1)
            self.assertEqual(app._ensure_calls, 2)
            self.assertFalse(app.is_running)
            self.assertIsNotNone(app._paddleocr_start_wait)

            # The matching retry becomes ready and only then starts workers.
            app._paddleocr_ready = True
            app._paddleocr_prewarm_metrics = {
                "status": "completed",
                "ready": True,
                "generation": 2,
                "error_type": "",
                "active": False,
            }
            self._run_pending_callbacks(app, max_steps=2)

        self.assertEqual(app._ensure_calls, 2)
        self.assertEqual(
            started_names,
            ["CaptureThread", "OCRThread", "TranslationThread"],
        )

    def test_settings_capture_failure_does_not_enter_wait_state(self):
        import app_lifecycle

        app = self._build_app(ready=False)
        app.get_current_paddleocr_settings = Mock(side_effect=RuntimeError("bad setting"))

        app_lifecycle.AppLifecycleMixin.toggle_translation(app)

        self.assertFalse(app.is_running)
        self.assertIsNone(app._paddleocr_start_wait)
        self.assertEqual(app.start_stop_btn.text, "Start")
        self.assertIn("failed", app.status_label.text.lower())
        self.assertFalse(hasattr(app, "_ensure_calls"))

    def test_ready_after_wait_starts_workers_once(self):
        import app_lifecycle

        app = self._build_app(ready=False)
        started_names = []

        class FakeThread:
            def __init__(self, target=None, args=(), name=None, daemon=None):
                self.target = target
                self.args = args
                self.name = name
                self.daemon = daemon
                self.started = False

            def start(self):
                self.started = True
                started_names.append(self.name)

            def is_alive(self):
                return False

        with patch("app_lifecycle.threading.Thread", side_effect=FakeThread):
            app_lifecycle.AppLifecycleMixin.toggle_translation(app)
            self.assertFalse(app.is_running)
            self.assertEqual(started_names, [])

            app._paddleocr_ready = True
            app._paddleocr_prewarm_metrics = {
                "status": "completed",
                "ready": True,
                "generation": 1,
                "error_type": "",
                "active": False,
            }
            self._run_pending_callbacks(app, max_steps=5)

        self.assertTrue(app.is_running)
        self.assertEqual(
            started_names,
            ["CaptureThread", "OCRThread", "TranslationThread"],
        )
        self.assertIsNone(app._paddleocr_start_wait)
        self.assertTrue(app.translation_handler.start_translation_session.called)

        # Stale leftover callbacks must not restart workers.
        leftover = list(app.root.callbacks)
        app.root.callbacks.clear()
        for _delay, callback in leftover:
            callback()
        self.assertEqual(
            started_names,
            ["CaptureThread", "OCRThread", "TranslationThread"],
        )

    def test_cancel_during_wait_blocks_later_ready_callback(self):
        import app_lifecycle

        app = self._build_app(ready=False)
        started_names = []

        class FakeThread:
            def __init__(self, target=None, args=(), name=None, daemon=None):
                self.name = name

            def start(self):
                started_names.append(self.name)

            def is_alive(self):
                return False

        with patch("app_lifecycle.threading.Thread", side_effect=FakeThread):
            app_lifecycle.AppLifecycleMixin.toggle_translation(app)
            token = app._paddleocr_start_wait["token"]
            self.assertEqual(app._paddleocr_start_wait["state"], "waiting")

            # Cancel via toggle while waiting.
            app_lifecycle.AppLifecycleMixin.toggle_translation(app)
            self.assertIsNone(app._paddleocr_start_wait)
            self.assertFalse(app.is_running)
            self.assertEqual(started_names, [])
            self.assertIn("cancelled", app.status_label.text.lower())

            app._paddleocr_ready = True
            app._paddleocr_prewarm_metrics = {
                "status": "completed",
                "ready": True,
                "generation": 1,
                "error_type": "",
                "active": False,
            }
            # Drain any scheduled callbacks; cancelled token must not start workers.
            self._run_pending_callbacks(app, max_steps=10)

        self.assertEqual(started_names, [])
        self.assertFalse(app.is_running)
        self.assertNotEqual(token, None)

    def test_prewarm_failure_during_wait_does_not_start_workers(self):
        import app_lifecycle

        app = self._build_app(ready=False)
        started_names = []

        class FakeThread:
            def __init__(self, target=None, args=(), name=None, daemon=None):
                self.name = name

            def start(self):
                started_names.append(self.name)

            def is_alive(self):
                return False

        with patch("app_lifecycle.threading.Thread", side_effect=FakeThread):
            app_lifecycle.AppLifecycleMixin.toggle_translation(app)
            app._paddleocr_prewarm_metrics = {
                "status": "failed",
                "ready": False,
                "generation": 1,
                "error_type": "RuntimeError",
                "active": False,
            }
            self._run_pending_callbacks(app, max_steps=5)

        self.assertEqual(started_names, [])
        self.assertFalse(app.is_running)
        self.assertIsNone(app._paddleocr_start_wait)
        self.assertIn("failed", app.status_label.text.lower())

    def test_already_ready_starts_immediately_without_wait_state(self):
        import app_lifecycle

        app = self._build_app(ready=True)
        started_names = []

        class FakeThread:
            def __init__(self, target=None, args=(), name=None, daemon=None):
                self.name = name

            def start(self):
                started_names.append(self.name)

            def is_alive(self):
                return False

        with patch("app_lifecycle.threading.Thread", side_effect=FakeThread):
            app_lifecycle.AppLifecycleMixin.toggle_translation(app)

        self.assertTrue(app.is_running)
        self.assertIsNone(app._paddleocr_start_wait)
        self.assertEqual(
            started_names,
            ["CaptureThread", "OCRThread", "TranslationThread"],
        )
        self.assertEqual(app.root.callbacks, [])

    def test_custom_ai_ocr_does_not_enter_local_wait(self):
        import app_lifecycle

        app = self._build_app(ocr_model="custom_ai", ready=False)
        app.custom_ai_profiles = types.SimpleNamespace(
            get_active_profile=lambda role: {"id": "profile"}
        )
        started_names = []

        class FakeThread:
            def __init__(self, target=None, args=(), name=None, daemon=None):
                self.name = name

            def start(self):
                started_names.append(self.name)

            def is_alive(self):
                return False

        with patch("app_lifecycle.threading.Thread", side_effect=FakeThread):
            app_lifecycle.AppLifecycleMixin.toggle_translation(app)

        self.assertTrue(app.is_running)
        self.assertIsNone(getattr(app, "_paddleocr_start_wait", None))
        self.assertEqual(
            started_names,
            ["CaptureThread", "OCRThread", "TranslationThread"],
        )
        self.assertTrue(app.translation_handler.start_ocr_session.called)

    def test_settings_change_during_wait_does_not_start_workers(self):
        import app_lifecycle

        app = self._build_app(ready=False)
        started_names = []

        class FakeThread:
            def __init__(self, target=None, args=(), name=None, daemon=None):
                self.name = name

            def start(self):
                started_names.append(self.name)

            def is_alive(self):
                return False

        with patch("app_lifecycle.threading.Thread", side_effect=FakeThread):
            app_lifecycle.AppLifecycleMixin.toggle_translation(app)
            # Settings generation changed / no longer matching.
            app._paddleocr_settings = self._settings(model_size="small")
            app._paddleocr_ready = True
            app._paddleocr_prewarm_metrics = {
                "status": "completed",
                "ready": True,
                "generation": 2,
                "error_type": "",
                "active": False,
            }
            self._run_pending_callbacks(app, max_steps=5)

        self.assertEqual(started_names, [])
        self.assertFalse(app.is_running)
        self.assertIsNone(app._paddleocr_start_wait)


if __name__ == "__main__":
    unittest.main()
