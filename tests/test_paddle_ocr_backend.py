import queue
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image, ImageDraw


class PaddleOCRBackendTests(unittest.TestCase):
    def test_resolve_small_ppocrv6_model_names(self):
        from paddle_ocr_backend import resolve_ppocrv6_model_names

        self.assertEqual(
            resolve_ppocrv6_model_names("small"),
            ("PP-OCRv6_small_det", "PP-OCRv6_small_rec"),
        )

    def test_paddleocr_settings_default_to_fast_subtitle_values(self):
        from paddle_ocr_backend import PaddleOCRSettings

        settings = PaddleOCRSettings()

        self.assertEqual(settings.model_size, "tiny")
        self.assertEqual(settings.upscale, 1.0)

    def test_flatten_filters_low_scores_and_preserves_linebreaks(self):
        from paddle_ocr_backend import flatten_paddleocr_result

        result = [{
            "rec_texts": [" HELLO ", "noise", "WORLD"],
            "rec_scores": [0.91, 0.12, 0.72],
            "rec_boxes": [],
        }]

        text, lines = flatten_paddleocr_result(
            result,
            min_score=0.35,
            keep_linebreaks=True,
        )

        self.assertEqual(text, "HELLO\nWORLD")
        self.assertEqual([line.text for line in lines], ["HELLO", "WORLD"])
        self.assertEqual([line.confidence for line in lines], [0.91, 0.72])

    def test_flatten_joins_lines_with_spaces_when_linebreaks_disabled(self):
        from paddle_ocr_backend import flatten_paddleocr_result

        result = [{
            "rec_texts": ["The door", "is open"],
            "rec_scores": [0.8, 0.9],
        }]

        text, _lines = flatten_paddleocr_result(
            result,
            min_score=0.35,
            keep_linebreaks=False,
        )

        self.assertEqual(text, "The door is open")

    def test_flatten_accepts_numpy_rec_boxes(self):
        from paddle_ocr_backend import flatten_paddleocr_result

        result = [{
            "rec_texts": ["HELLO"],
            "rec_scores": [0.99],
            "rec_boxes": np.array([[1, 2, 30, 12]], dtype=np.int16),
        }]

        text, lines = flatten_paddleocr_result(result, min_score=0.35)

        self.assertEqual(text, "HELLO")
        self.assertEqual(lines[0].bbox, [1, 2, 30, 12])

    def test_recognize_uses_cached_engine_and_predicts_prepared_image(self):
        from paddle_ocr_backend import PaddleOCRSettings, recognize_with_paddleocr

        fake_engine = Mock()
        fake_engine.predict.return_value = [{
            "rec_texts": ["Hello"],
            "rec_scores": [0.88],
        }]
        settings = PaddleOCRSettings(model_size="small", min_score=0.35, upscale=1.0)
        image = Image.new("RGB", (12, 8), "white")

        with patch("paddle_ocr_backend.get_paddleocr_engine", return_value=fake_engine) as get_engine:
            text, lines = recognize_with_paddleocr(image, settings, keep_linebreaks=False)

        self.assertEqual(text, "Hello")
        self.assertEqual(lines[0].text, "Hello")
        get_engine.assert_called_once_with(settings)
        fake_engine.predict.assert_called_once()

    def test_build_cpu_engine_disables_mkldnn(self):
        from paddle_ocr_backend import PaddleOCRSettings, _build_paddleocr_engine

        captured_kwargs = {}

        class FakePaddleOCR:
            def __init__(self, **kwargs):
                captured_kwargs.update(kwargs)

        settings = PaddleOCRSettings(device="cpu")

        with patch("paddle_ocr_backend._import_paddleocr", return_value=FakePaddleOCR):
            _build_paddleocr_engine(settings)

        self.assertIs(captured_kwargs["enable_mkldnn"], False)

    def test_subtitle_line_extraction_crops_bright_text_band(self):
        from paddle_ocr_backend import (
            PaddleOCRSettings,
            prepare_paddleocr_subtitle_line_images,
        )

        image = Image.new("RGB", (720, 96), "black")
        draw = ImageDraw.Draw(image)
        draw.rectangle((230, 32, 490, 60), fill="white")

        line_images = prepare_paddleocr_subtitle_line_images(
            image,
            PaddleOCRSettings(upscale=1.0),
        )

        self.assertEqual(len(line_images), 1)
        self.assertLess(line_images[0].width, 360)
        self.assertLess(line_images[0].height, 70)

    def test_subtitle_line_extraction_ignores_tall_bright_background_regions(self):
        from paddle_ocr_backend import (
            PaddleOCRSettings,
            prepare_paddleocr_subtitle_line_images,
        )

        image = Image.new("RGB", (600, 120), "black")
        draw = ImageDraw.Draw(image)
        draw.rectangle((420, 0, 520, 119), fill=(210, 210, 210))
        draw.rectangle((140, 72, 360, 94), fill="white")

        line_images = prepare_paddleocr_subtitle_line_images(
            image,
            PaddleOCRSettings(upscale=1.0),
        )

        self.assertEqual(len(line_images), 1)
        self.assertLess(line_images[0].height, 60)
        self.assertLess(line_images[0].width, 300)

    def test_subtitle_line_extraction_uses_edges_when_bright_background_distracts_threshold(self):
        from paddle_ocr_backend import (
            PaddleOCRSettings,
            prepare_paddleocr_subtitle_line_images,
        )

        image = Image.new("RGB", (640, 128), (245, 245, 245))
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 72, 639, 127), fill=(18, 22, 28))
        draw.rectangle((150, 91, 490, 109), fill=(165, 165, 165))

        line_images = prepare_paddleocr_subtitle_line_images(
            image,
            PaddleOCRSettings(upscale=1.0),
        )

        self.assertEqual(len(line_images), 1)
        self.assertLess(line_images[0].height, 50)

    def test_subtitle_line_extraction_uses_edges_when_text_is_not_bright(self):
        from paddle_ocr_backend import (
            PaddleOCRSettings,
            prepare_paddleocr_subtitle_line_images,
        )

        image = Image.new("RGB", (640, 128), "black")
        draw = ImageDraw.Draw(image)
        draw.rectangle((150, 88, 490, 108), fill=(120, 120, 120))

        line_images = prepare_paddleocr_subtitle_line_images(
            image,
            PaddleOCRSettings(upscale=1.0),
        )

        self.assertEqual(len(line_images), 1)
        self.assertLess(line_images[0].height, 50)

    def test_subtitle_text_recognition_filters_symbol_noise_lines(self):
        from paddle_ocr_backend import flatten_paddleocr_text_recognition_result

        result = [
            {"rec_text": "And now it's all breaking loose.", "rec_score": 0.95},
            {"rec_text": "7/#7$%8：：-+", "rec_score": 0.99},
        ]

        text, lines = flatten_paddleocr_text_recognition_result(result, min_score=0.35)

        self.assertEqual(text, "And now it's all breaking loose.")
        self.assertEqual([line.text for line in lines], ["And now it's all breaking loose."])

    def test_subtitle_text_recognition_filters_short_alnum_noise_lines(self):
        from paddle_ocr_backend import flatten_paddleocr_text_recognition_result

        result = [
            {"rec_text": "O0", "rec_score": 0.91},
            {"rec_text": "Besides, he-he's gonna be dead", "rec_score": 0.96},
        ]

        text, lines = flatten_paddleocr_text_recognition_result(result, min_score=0.35)

        self.assertEqual(text, "Besides, he-he's gonna be dead")
        self.assertEqual([line.text for line in lines], ["Besides, he-he's gonna be dead"])

    def test_subtitle_text_recognition_repairs_common_english_spacing_merges(self):
        from paddle_ocr_backend import flatten_paddleocr_text_recognition_result

        result = [
            {"rec_text": "Imean,weirdly,Ican see it.", "rec_score": 0.97},
            {"rec_text": "We got to goto IHOP,right?", "rec_score": 0.96},
        ]

        text, lines = flatten_paddleocr_text_recognition_result(result, min_score=0.35)

        self.assertEqual(
            text,
            "I mean, weirdly, I can see it. We got to goto IHOP, right?",
        )
        self.assertEqual(
            [line.text for line in lines],
            [
                "I mean, weirdly, I can see it.",
                "We got to goto IHOP, right?",
            ],
        )

    def test_subtitle_recognition_uses_text_recognition_fast_path(self):
        from paddle_ocr_backend import PaddleOCRSettings, recognize_subtitle_with_paddleocr

        image = Image.new("RGB", (720, 96), "black")
        draw = ImageDraw.Draw(image)
        draw.rectangle((230, 32, 490, 60), fill="white")
        fake_engine = Mock()
        fake_engine.predict.return_value = [{
            "rec_text": "Here, kitty kitty!",
            "rec_score": 0.99,
        }]

        with patch(
            "paddle_ocr_backend.get_paddleocr_text_recognition_engine",
            return_value=fake_engine,
        ):
            with patch(
                "paddle_ocr_backend.get_paddleocr_engine",
                side_effect=AssertionError("full detector should not run"),
            ):
                text, lines = recognize_subtitle_with_paddleocr(
                    image,
                    PaddleOCRSettings(upscale=1.0, min_score=0.35),
                    keep_linebreaks=False,
                )

        self.assertEqual(text, "Here, kitty kitty!")
        self.assertEqual(lines[0].text, "Here, kitty kitty!")
        fake_engine.predict.assert_called_once()

    def test_subtitle_recognition_batches_multiple_line_crops(self):
        from paddle_ocr_backend import PaddleOCRSettings, recognize_subtitle_with_paddleocr

        image = Image.new("RGB", (720, 120), "black")
        draw = ImageDraw.Draw(image)
        draw.rectangle((180, 36, 540, 54), fill="white")
        draw.rectangle((170, 76, 550, 94), fill="white")
        fake_engine = Mock()
        fake_engine.predict.return_value = [
            {"rec_text": "First line", "rec_score": 0.97},
            {"rec_text": "Second line", "rec_score": 0.96},
        ]

        with patch(
            "paddle_ocr_backend.get_paddleocr_text_recognition_engine",
            return_value=fake_engine,
        ):
            text, lines = recognize_subtitle_with_paddleocr(
                image,
                PaddleOCRSettings(upscale=1.0, min_score=0.35),
                keep_linebreaks=True,
            )

        self.assertEqual(text, "First line\nSecond line")
        self.assertEqual([line.text for line in lines], ["First line", "Second line"])
        fake_engine.predict.assert_called_once()
        call_kwargs = fake_engine.predict.call_args.kwargs
        self.assertIsInstance(call_kwargs["input"], list)
        self.assertEqual(len(call_kwargs["input"]), 2)

    def test_subtitle_recognition_falls_back_to_full_ocr_without_line_crop(self):
        from paddle_ocr_backend import PaddleOCRSettings, recognize_subtitle_with_paddleocr

        image = Image.new("RGB", (320, 80), "black")
        fake_full_engine = Mock()
        fake_full_engine.predict.return_value = [{
            "rec_texts": ["Fallback text"],
            "rec_scores": [0.88],
        }]

        with patch(
            "paddle_ocr_backend.get_paddleocr_text_recognition_engine",
            side_effect=AssertionError("text recognition should not run"),
        ):
            with patch(
                "paddle_ocr_backend.get_paddleocr_engine",
                return_value=fake_full_engine,
            ):
                text, _lines = recognize_subtitle_with_paddleocr(
                    image,
                    PaddleOCRSettings(upscale=1.0, min_score=0.35),
                    keep_linebreaks=False,
                )

        self.assertEqual(text, "Fallback text")


class PaddleOCRConfigAndUITests(unittest.TestCase):
    def test_paddleocr_defaults_exist(self):
        from config_manager import DEFAULT_CONFIG_SETTINGS

        self.assertEqual(DEFAULT_CONFIG_SETTINGS["ocr_model"], "tesseract")
        self.assertEqual(DEFAULT_CONFIG_SETTINGS["paddleocr_source_dir"], "PaddleOCR-3.7.0")
        self.assertEqual(DEFAULT_CONFIG_SETTINGS["paddleocr_ocr_version"], "PP-OCRv6")
        self.assertEqual(DEFAULT_CONFIG_SETTINGS["paddleocr_model_size"], "tiny")
        self.assertEqual(DEFAULT_CONFIG_SETTINGS["paddleocr_min_score"], "0.35")
        self.assertEqual(DEFAULT_CONFIG_SETTINGS["paddleocr_upscale"], "1.0")

    def test_paddleocr_display_option_is_before_custom_ai_profiles(self):
        import gui_builder

        app = types.SimpleNamespace(
            ui_lang=types.SimpleNamespace(get_label=lambda _key, fallback=None: fallback),
            custom_ai_profiles=types.SimpleNamespace(
                list_profiles=lambda enabled_only=True: [{"name": "Vision API"}]
            ),
        )

        self.assertEqual(
            gui_builder.build_ocr_model_display_options(app),
            [
                "Tesseract (offline)",
                "PaddleOCR PP-OCRv6 (offline)",
                "Vision API",
            ],
        )

    def test_resolve_paddleocr_display_selection(self):
        import gui_builder

        app = types.SimpleNamespace(
            ui_lang=types.SimpleNamespace(get_label=lambda _key, fallback=None: fallback),
            custom_ai_profiles=types.SimpleNamespace(list_profiles=lambda enabled_only=True: []),
        )

        model_code, profile_id = gui_builder.resolve_ocr_model_display_selection(
            app,
            "PaddleOCR PP-OCRv6 (offline)",
        )

        self.assertEqual(model_code, "paddleocr")
        self.assertIsNone(profile_id)

    def test_paddleocr_is_local_not_api_based(self):
        import app_logic

        app = object.__new__(app_logic.GameChangingTranslator)
        app.ocr_model_var = types.SimpleNamespace(get=lambda: "paddleocr")

        self.assertFalse(app_logic.GameChangingTranslator.is_api_based_ocr_model(app))


class PaddleOCRPrewarmTests(unittest.TestCase):
    def _paddle_app(self):
        import app_logic

        app = object.__new__(app_logic.GameChangingTranslator)
        app.ocr_model_var = types.SimpleNamespace(get=lambda: "paddleocr")
        app.keep_linebreaks_var = types.SimpleNamespace(get=lambda: False)
        app.paddleocr_source_dir_var = types.SimpleNamespace(get=lambda: "PaddleOCR-3.7.0")
        app.paddleocr_lang_var = types.SimpleNamespace(get=lambda: "en")
        app.paddleocr_ocr_version_var = types.SimpleNamespace(get=lambda: "PP-OCRv6")
        app.paddleocr_model_size_var = types.SimpleNamespace(get=lambda: "tiny")
        app.paddleocr_device_var = types.SimpleNamespace(get=lambda: "cpu")
        app.paddleocr_min_score_var = types.SimpleNamespace(get=lambda: "0.35")
        app.paddleocr_upscale_var = types.SimpleNamespace(get=lambda: "1.0")
        app.paddleocr_text_det_limit_side_len_var = types.SimpleNamespace(get=lambda: "960")
        app.paddleocr_text_det_limit_type_var = types.SimpleNamespace(get=lambda: "max")
        app.paddleocr_use_textline_orientation_var = types.SimpleNamespace(get=lambda: False)
        app.get_ocr_model_setting = lambda: app.ocr_model_var.get()
        return app

    def test_initial_startup_schedules_paddleocr_prewarm_only_for_paddleocr(self):
        import app_logic

        paddle_app = object.__new__(app_logic.GameChangingTranslator)
        scheduled_callbacks = []
        paddle_app.root = types.SimpleNamespace(
            after=lambda delay, callback: scheduled_callbacks.append((delay, callback))
        )
        paddle_app.get_ocr_model_setting = lambda: "paddleocr"
        paddle_app.ensure_paddleocr_ready_if_selected = Mock()

        scheduled = app_logic.GameChangingTranslator.schedule_initial_paddleocr_prewarm(paddle_app)

        self.assertTrue(scheduled)
        self.assertEqual(scheduled_callbacks[0][0], 0)
        scheduled_callbacks[0][1]()
        paddle_app.ensure_paddleocr_ready_if_selected.assert_called_once_with("application startup")

        ai_app = object.__new__(app_logic.GameChangingTranslator)
        ai_app.root = types.SimpleNamespace(after=Mock())
        ai_app.get_ocr_model_setting = lambda: "custom_ai"
        ai_app.ensure_paddleocr_ready_if_selected = Mock()

        scheduled = app_logic.GameChangingTranslator.schedule_initial_paddleocr_prewarm(ai_app)

        self.assertFalse(scheduled)
        ai_app.root.after.assert_not_called()
        ai_app.ensure_paddleocr_ready_if_selected.assert_not_called()

    def test_save_settings_starts_paddleocr_prewarm_after_successful_switch(self):
        import app_logic

        app = object.__new__(app_logic.GameChangingTranslator)
        app._fully_initialized = True
        app._app_is_closing = False
        app.ui_interaction_handler = types.SimpleNamespace(save_settings=Mock(return_value=True))
        app.get_ocr_model_setting = lambda: "paddleocr"
        app.ensure_paddleocr_ready_if_selected = Mock()

        self.assertTrue(app_logic.GameChangingTranslator.save_settings(app))

        app.ensure_paddleocr_ready_if_selected.assert_called_once_with("settings saved")

    def test_save_settings_does_not_prewarm_for_custom_ai_ocr(self):
        import app_logic

        app = object.__new__(app_logic.GameChangingTranslator)
        app._fully_initialized = True
        app._app_is_closing = False
        app.ui_interaction_handler = types.SimpleNamespace(save_settings=Mock(return_value=True))
        app.get_ocr_model_setting = lambda: "custom_ai"
        app.ensure_paddleocr_ready_if_selected = Mock()

        self.assertTrue(app_logic.GameChangingTranslator.save_settings(app))

        app.ensure_paddleocr_ready_if_selected.assert_not_called()

    def test_paddleocr_prewarm_initializes_fast_path_and_fallback_engines(self):
        import app_logic
        from paddle_ocr_backend import PaddleOCRSettings

        app = self._paddle_app()
        created_threads = []

        class InlineThread:
            def __init__(self, target, args=(), name=None, daemon=None):
                self.target = target
                self.args = args
                self.name = name
                self.daemon = daemon
                self.started = False

            def start(self):
                self.started = True
                self.target(*self.args)

            def is_alive(self):
                return False

        def make_thread(*args, **kwargs):
            thread = InlineThread(*args, **kwargs)
            created_threads.append(thread)
            return thread

        with patch.object(app_logic.threading, "Thread", side_effect=make_thread):
            with patch.object(app_logic, "get_paddleocr_text_recognition_engine") as text_engine:
                with patch.object(app_logic, "get_paddleocr_engine") as full_engine:
                    started = app_logic.GameChangingTranslator.ensure_paddleocr_ready_if_selected(
                        app,
                        "settings saved",
                    )

        self.assertTrue(started)
        self.assertEqual(len(created_threads), 1)
        self.assertEqual(created_threads[0].name, "PaddleOCRPrewarm")
        self.assertIsInstance(text_engine.call_args.args[0], PaddleOCRSettings)
        self.assertEqual(text_engine.call_args.args[0].ocr_version, "PP-OCRv6")
        full_engine.assert_called_once_with(text_engine.call_args.args[0])


class PaddleOCRWorkerRoutingTests(unittest.TestCase):
    def _var(self, value):
        return types.SimpleNamespace(get=lambda: value)

    def test_worker_builds_paddleocr_settings_from_app(self):
        import worker_threads
        from paddle_ocr_backend import PaddleOCRSettings

        app = types.SimpleNamespace(
            paddleocr_source_dir_var=types.SimpleNamespace(get=lambda: "PaddleOCR-3.7.0"),
            paddleocr_lang_var=types.SimpleNamespace(get=lambda: "en"),
            paddleocr_ocr_version_var=types.SimpleNamespace(get=lambda: "PP-OCRv6"),
            paddleocr_model_size_var=types.SimpleNamespace(get=lambda: "small"),
            paddleocr_device_var=types.SimpleNamespace(get=lambda: "cpu"),
            paddleocr_min_score_var=types.SimpleNamespace(get=lambda: "0.35"),
            paddleocr_upscale_var=types.SimpleNamespace(get=lambda: "2.0"),
            paddleocr_text_det_limit_side_len_var=types.SimpleNamespace(get=lambda: "960"),
            paddleocr_text_det_limit_type_var=types.SimpleNamespace(get=lambda: "max"),
            paddleocr_use_textline_orientation_var=types.SimpleNamespace(get=lambda: False),
        )

        settings = worker_threads.get_paddleocr_settings_from_app(app)

        self.assertIsInstance(settings, PaddleOCRSettings)
        self.assertEqual(settings.model_size, "small")
        self.assertEqual(settings.min_score, 0.35)
        self.assertEqual(settings.upscale, 2.0)

    def test_paddleocr_queue_keeps_latest_frame_only(self):
        import worker_threads

        app = types.SimpleNamespace(ocr_queue=queue.Queue(maxsize=8))
        old_frame = object()
        newer_frame = object()
        latest_frame = object()
        app.ocr_queue.put_nowait(old_frame)
        app.ocr_queue.put_nowait(newer_frame)

        worker_threads.enqueue_ocr_frame_for_model(app, latest_frame, "paddleocr")

        self.assertEqual(app.ocr_queue.qsize(), 1)
        self.assertIs(app.ocr_queue.get_nowait(), latest_frame)

    def test_paddleocr_first_clear_text_submits_without_legacy_stability_threshold(self):
        import worker_threads

        submitted = []
        image = Image.new("RGB", (16, 10), "white")
        app = types.SimpleNamespace(
            is_running=True,
            ocr_queue=queue.Queue(),
            get_ocr_model_setting=lambda: "paddleocr",
            confidence_threshold=35,
            confidence_var=self._var(35),
            is_api_based_ocr_model=lambda _model: False,
            preprocessing_mode_var=self._var("binary"),
            adaptive_block_size_var=self._var("41"),
            adaptive_c_var=self._var("-60"),
            ocr_debugging_var=self._var(False),
            previous_text="",
            text_stability_counter=0,
            stable_threshold=2,
            is_placeholder_text=lambda _text: False,
            calculate_text_similarity=lambda _current, _previous: 0.0,
            reset_clear_timeout=Mock(),
        )
        app.ocr_queue.put_nowait(image)

        def fake_start_async_translation(_app, text, _ocr_sequence_number, requested_at_monotonic=None):
            submitted.append(text)
            app.is_running = False

        with patch.object(
            worker_threads,
            "process_local_ocr_frame",
            return_value=("Clear subtitle text.", None, "PaddleOCR"),
        ):
            with patch.object(
                worker_threads,
                "start_async_translation",
                side_effect=fake_start_async_translation,
            ):
                with patch.object(worker_threads.time, "sleep", side_effect=lambda _seconds: setattr(app, "is_running", False)):
                    worker_threads.run_ocr_thread(app)

        self.assertEqual(submitted, ["Clear subtitle text."])

    def test_process_local_ocr_frame_routes_paddleocr_without_tesseract(self):
        import worker_threads

        image = Image.new("RGB", (16, 10), "white")
        app = types.SimpleNamespace(
            keep_linebreaks_var=types.SimpleNamespace(get=lambda: False),
            paddleocr_source_dir_var=types.SimpleNamespace(get=lambda: "PaddleOCR-3.7.0"),
            paddleocr_lang_var=types.SimpleNamespace(get=lambda: "en"),
            paddleocr_ocr_version_var=types.SimpleNamespace(get=lambda: "PP-OCRv6"),
            paddleocr_model_size_var=types.SimpleNamespace(get=lambda: "small"),
            paddleocr_device_var=types.SimpleNamespace(get=lambda: "cpu"),
            paddleocr_min_score_var=types.SimpleNamespace(get=lambda: "0.35"),
            paddleocr_upscale_var=types.SimpleNamespace(get=lambda: "1.0"),
            paddleocr_text_det_limit_side_len_var=types.SimpleNamespace(get=lambda: "960"),
            paddleocr_text_det_limit_type_var=types.SimpleNamespace(get=lambda: "max"),
            paddleocr_use_textline_orientation_var=types.SimpleNamespace(get=lambda: False),
        )

        with patch.object(
            worker_threads,
            "recognize_subtitle_with_paddleocr",
            return_value=("Hello world", []),
        ) as recognize:
            with patch.object(
                worker_threads,
                "ocr_region_with_confidence",
                side_effect=AssertionError("Tesseract should not run"),
            ):
                text, processed_cv_img, engine_label = worker_threads.process_local_ocr_frame(
                    app,
                    image,
                    "paddleocr",
                    tess_langs=None,
                    tessdata_dir=None,
                    current_conf_thresh=50,
                    prep_mode="binary",
                    block_size=41,
                    c_value=-60,
                )

        self.assertEqual(text, "Hello world")
        self.assertEqual(engine_label, "PaddleOCR")
        self.assertEqual(processed_cv_img.shape[0:2], (10, 16))
        recognize.assert_called_once()

    def test_process_local_ocr_frame_uses_paddleocr_subtitle_fast_path(self):
        import worker_threads

        image = Image.new("RGB", (16, 10), "white")
        app = types.SimpleNamespace(
            keep_linebreaks_var=types.SimpleNamespace(get=lambda: False),
            paddleocr_source_dir_var=types.SimpleNamespace(get=lambda: "PaddleOCR-3.7.0"),
            paddleocr_lang_var=types.SimpleNamespace(get=lambda: "en"),
            paddleocr_ocr_version_var=types.SimpleNamespace(get=lambda: "PP-OCRv6"),
            paddleocr_model_size_var=types.SimpleNamespace(get=lambda: "tiny"),
            paddleocr_device_var=types.SimpleNamespace(get=lambda: "cpu"),
            paddleocr_min_score_var=types.SimpleNamespace(get=lambda: "0.35"),
            paddleocr_upscale_var=types.SimpleNamespace(get=lambda: "1.0"),
            paddleocr_text_det_limit_side_len_var=types.SimpleNamespace(get=lambda: "960"),
            paddleocr_text_det_limit_type_var=types.SimpleNamespace(get=lambda: "max"),
            paddleocr_use_textline_orientation_var=types.SimpleNamespace(get=lambda: False),
        )

        with patch.object(
            worker_threads,
            "recognize_subtitle_with_paddleocr",
            return_value=("Fast subtitle", []),
        ) as recognize:
            text, _processed_cv_img, engine_label = worker_threads.process_local_ocr_frame(
                app,
                image,
                "paddleocr",
                tess_langs=None,
                tessdata_dir=None,
                current_conf_thresh=50,
                prep_mode="binary",
                block_size=41,
                c_value=-60,
            )

        self.assertEqual(text, "Fast subtitle")
        self.assertEqual(engine_label, "PaddleOCR")
        recognize.assert_called_once()

    def test_paddleocr_cache_key_includes_engine_settings(self):
        import worker_threads

        app = types.SimpleNamespace(
            keep_linebreaks_var=types.SimpleNamespace(get=lambda: True),
            paddleocr_source_dir_var=types.SimpleNamespace(get=lambda: "PaddleOCR-3.7.0"),
            paddleocr_lang_var=types.SimpleNamespace(get=lambda: "en"),
            paddleocr_ocr_version_var=types.SimpleNamespace(get=lambda: "PP-OCRv6"),
            paddleocr_model_size_var=types.SimpleNamespace(get=lambda: "small"),
            paddleocr_device_var=types.SimpleNamespace(get=lambda: "cpu"),
            paddleocr_min_score_var=types.SimpleNamespace(get=lambda: "0.35"),
            paddleocr_upscale_var=types.SimpleNamespace(get=lambda: "2.0"),
            paddleocr_text_det_limit_side_len_var=types.SimpleNamespace(get=lambda: "960"),
            paddleocr_text_det_limit_type_var=types.SimpleNamespace(get=lambda: "max"),
            paddleocr_use_textline_orientation_var=types.SimpleNamespace(get=lambda: False),
        )

        key = worker_threads.get_paddleocr_ocr_cache_mode_key(app)

        self.assertIn("paddleocr", key)
        self.assertIn("version=PP-OCRv6", key)
        self.assertIn("size=small", key)
        self.assertIn("min_score=0.35", key)
        self.assertIn("upscale=2.0", key)
        self.assertIn("keep_linebreaks=True", key)


if __name__ == "__main__":
    unittest.main()
