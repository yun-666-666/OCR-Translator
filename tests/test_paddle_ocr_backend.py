import queue
import threading
import time
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

    def test_coerce_float_and_int_share_backend_contract(self):
        from paddle_ocr_backend import _coerce_float, _coerce_int

        self.assertEqual(_coerce_float("0.35", 0.45, 0.0, 1.0), 0.35)
        self.assertEqual(_coerce_float("bad", 0.45, 0.0, 1.0), 0.45)
        self.assertEqual(_coerce_float(None, 0.45, 0.0, 1.0), 0.45)
        self.assertEqual(_coerce_float("2.5", 1.0, 1.0, 2.0), 2.0)
        self.assertEqual(_coerce_float("0.1", 1.0, 1.0, 4.0), 1.0)

        self.assertEqual(_coerce_int("960", 960, 128, 4096), 960)
        self.assertEqual(_coerce_int("12.9", 960, 128, 4096), 960)
        self.assertEqual(_coerce_int(None, 960, 128, 4096), 960)
        self.assertEqual(_coerce_int("64", 960, 128, 4096), 128)
        self.assertEqual(_coerce_int("9999", 960, 128, 4096), 4096)

    def test_normalize_paddleocr_settings_coerces_numeric_fields(self):
        from paddle_ocr_backend import PaddleOCRSettings, normalize_paddleocr_settings

        settings = normalize_paddleocr_settings(
            PaddleOCRSettings(
                min_score="1.5",
                upscale="0.25",
                text_det_limit_side_len="50",
            )
        )

        self.assertEqual(settings.min_score, 1.0)
        self.assertEqual(settings.upscale, 1.0)
        self.assertEqual(settings.text_det_limit_side_len, 128)

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

    def test_low_confidence_fast_path_lines_still_reject_and_fallback(self):
        from paddle_ocr_backend import PaddleOCRSettings, recognize_subtitle_with_paddleocr

        image = Image.new("RGB", (720, 96), "black")
        draw = ImageDraw.Draw(image)
        draw.rectangle((230, 32, 490, 60), fill="white")
        fake_text_engine = Mock()
        fake_text_engine.predict.return_value = [{
            "rec_text": "xx",
            "rec_score": 0.20,
        }]
        full_engine = Mock()
        full_engine.predict.return_value = [{
            "rec_texts": ["safe fallback"],
            "rec_scores": [0.93],
        }]

        with patch(
            "paddle_ocr_backend.get_paddleocr_text_recognition_engine",
            return_value=fake_text_engine,
        ):
            with patch(
                "paddle_ocr_backend.get_paddleocr_engine",
                return_value=full_engine,
            ):
                text, lines = recognize_subtitle_with_paddleocr(
                    image,
                    PaddleOCRSettings(upscale=1.0, min_score=0.45),
                    keep_linebreaks=False,
                )

        self.assertEqual(text, "safe fallback")
        self.assertEqual(lines[0].text, "safe fallback")
        fake_text_engine.predict.assert_called_once()
        full_engine.predict.assert_called_once()

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

    def test_subtitle_fast_path_success_log_is_coalesced_and_content_free(self):
        import paddle_ocr_backend

        image = Image.new("RGB", (720, 120), "black")
        draw = ImageDraw.Draw(image)
        draw.rectangle((180, 36, 540, 54), fill="white")
        draw.rectangle((170, 76, 550, 94), fill="white")
        fake_engine = Mock()
        fake_engine.predict.return_value = [
            {"rec_text": "First line", "rec_score": 0.97},
            {"rec_text": "Second line", "rec_score": 0.96},
        ]

        line_images = [
            Image.new("RGB", (100, 20), "white"),
            Image.new("RGB", (100, 20), "white"),
        ]
        with patch.object(
            paddle_ocr_backend,
            "prepare_paddleocr_subtitle_line_images",
            return_value=line_images,
        ):
            with patch.object(
                paddle_ocr_backend,
                "get_paddleocr_text_recognition_engine",
                return_value=fake_engine,
            ):
                with patch.object(
                    paddle_ocr_backend,
                    "log_debug_coalesced",
                ) as log_coalesced:
                    text, lines = (
                        paddle_ocr_backend.recognize_subtitle_with_paddleocr(
                            image,
                            paddle_ocr_backend.PaddleOCRSettings(
                                upscale=1.0,
                                min_score=0.35,
                            ),
                            keep_linebreaks=True,
                        )
                    )

        self.assertEqual(text, "First line\nSecond line")
        self.assertEqual(len(lines), 2)
        self.assertEqual(log_coalesced.call_args.args[0], "paddle-subtitle-fast-path-success")
        logged_message = log_coalesced.call_args.args[1]
        self.assertIn(
            "PaddleOCR subtitle fast path recognized lines=2 chars=22",
            logged_message,
        )
        self.assertEqual(log_coalesced.call_args.kwargs.get("interval_seconds"), 5.0)
        self.assertNotIn("First line", logged_message)
        self.assertNotIn("Second line", logged_message)

    def test_subtitle_filter_log_is_coalesced_and_content_free(self):
        import paddle_ocr_backend

        with patch.object(paddle_ocr_backend, "log_debug") as direct_log:
            with patch.object(
                paddle_ocr_backend,
                "log_debug_coalesced",
            ) as log_coalesced:
                text, lines = (
                    paddle_ocr_backend.flatten_paddleocr_text_recognition_result(
                        [{
                            "rec_text": "subtitle-secret",
                            "rec_score": 0.1,
                        }],
                        min_score=0.35,
                    )
                )

        self.assertEqual(text, "")
        self.assertEqual(lines, [])
        direct_log.assert_not_called()
        log_coalesced.assert_called_once_with(
            "paddle-subtitle-fast-path-low-confidence",
            "PaddleOCR subtitle fast path filtered low-confidence line "
            "chars=15 lines=1 confidence=0.100",
            interval_seconds=5.0,
        )
        self.assertNotIn(
            "subtitle-secret",
            log_coalesced.call_args.args[1],
        )

    def test_subtitle_no_line_crop_log_is_coalesced(self):
        import paddle_ocr_backend

        image = Image.new("RGB", (320, 80), "black")
        with patch.object(
            paddle_ocr_backend,
            "prepare_paddleocr_subtitle_line_images",
            return_value=[],
        ):
            with patch.object(
                paddle_ocr_backend,
                "recognize_with_paddleocr",
                return_value=("Fallback text", []),
            ):
                with patch.object(
                    paddle_ocr_backend,
                    "log_debug",
                ) as direct_log:
                    with patch.object(
                        paddle_ocr_backend,
                        "log_debug_coalesced",
                    ) as log_coalesced:
                        text, _lines = (
                            paddle_ocr_backend.recognize_subtitle_with_paddleocr(
                                image,
                                paddle_ocr_backend.PaddleOCRSettings(),
                            )
                        )

        self.assertEqual(text, "Fallback text")
        direct_log.assert_not_called()
        self.assertEqual(
            log_coalesced.call_args_list[0].args[0],
            "paddle-subtitle-fast-path-no-line-crop",
        )
        self.assertIn(
            "PaddleOCR subtitle fast path found no line crop; falling back to full OCR",
            log_coalesced.call_args_list[0].args[1],
        )
        # Full fallback result is also classified content-free.
        event_names = [call.args[0] for call in log_coalesced.call_args_list]
        self.assertIn("paddle-subtitle-full-fallback-text", event_names)
        full_fallback_log = next(
            call.args[1]
            for call in log_coalesced.call_args_list
            if call.args[0] == "paddle-subtitle-full-fallback-text"
        )
        self.assertIn("outcome=full_fallback_text", full_fallback_log)
        for call in log_coalesced.call_args_list:
            self.assertNotIn("Fallback text", call.args[1])

    def test_subtitle_diagnostics_classifies_fullwidth_short_reject(self):
        import paddle_ocr_backend
        from paddle_ocr_backend import (
            PaddleOCRSettings,
            prepare_paddleocr_subtitle_line_images,
            reset_subtitle_fastpath_diagnostics,
        )

        reset_subtitle_fastpath_diagnostics()
        # Nearly full-width short band under the 14% height rule.
        # Edge fallback may still crop; diagnostics must still record that the
        # bright path rejected via fullwidth_short.
        image = Image.new("RGB", (1658, 188), "black")
        draw = ImageDraw.Draw(image)
        draw.rectangle((20, 84, 1638, 104), fill="white")  # h=20 / 188 ≈ 10.6%
        diagnostics = {}
        prepare_paddleocr_subtitle_line_images(
            image,
            PaddleOCRSettings(upscale=1.0),
            diagnostics=diagnostics,
        )
        fullwidth_hits = int(diagnostics.get("bright_reject_fullwidth_short", 0) or 0)
        fullwidth_hits += sum(
            int(v or 0)
            for k, v in diagnostics.items()
            if str(k).startswith("bright_reject_fullwidth_short_")
        )
        self.assertGreaterEqual(fullwidth_hits, 1)
        self.assertTrue(
            any(
                str(k).startswith("no_crop_reason_bright_reject_fullwidth_short")
                for k in diagnostics
            )
            or int(diagnostics.get("bright_no_crop", 0) or 0) >= 1
        )

    def test_subtitle_diagnostics_classifies_low_confidence_then_full_fallback(self):
        import paddle_ocr_backend
        from paddle_ocr_backend import (
            PaddleOCRSettings,
            recognize_subtitle_with_paddleocr,
            reset_subtitle_fastpath_diagnostics,
            get_subtitle_fastpath_diagnostics_snapshot,
        )

        reset_subtitle_fastpath_diagnostics()
        image = Image.new("RGB", (720, 96), "black")
        draw = ImageDraw.Draw(image)
        draw.rectangle((230, 32, 490, 60), fill="white")
        fake_text = Mock()
        fake_text.predict.return_value = [{
            "rec_text": "secret-subtitle",
            "rec_score": 0.10,
        }]
        with patch.object(
            paddle_ocr_backend,
            "get_paddleocr_text_recognition_engine",
            return_value=fake_text,
        ):
            with patch.object(
                paddle_ocr_backend,
                "recognize_with_paddleocr",
                return_value=("", []),
            ):
                with patch.object(
                    paddle_ocr_backend,
                    "log_debug_coalesced",
                ) as log_coalesced:
                    text, lines = recognize_subtitle_with_paddleocr(
                        image,
                        PaddleOCRSettings(upscale=1.0, min_score=0.45),
                    )
        self.assertEqual(text, "")
        self.assertEqual(lines, [])
        snap = get_subtitle_fastpath_diagnostics_snapshot()
        self.assertEqual(int(snap.get("rec_low_confidence", 0) or 0), 1)
        self.assertEqual(int(snap.get("fast_path_no_usable_text", 0) or 0), 1)
        self.assertEqual(int(snap.get("full_fallback_empty", 0) or 0), 1)
        for call in log_coalesced.call_args_list:
            self.assertNotIn("secret-subtitle", str(call.args[1]))

    def test_subtitle_diagnostics_edge_fallback_success_is_counted(self):
        import paddle_ocr_backend
        from paddle_ocr_backend import (
            PaddleOCRSettings,
            prepare_paddleocr_subtitle_line_images,
            reset_subtitle_fastpath_diagnostics,
        )

        reset_subtitle_fastpath_diagnostics()
        # Dark background + mid-gray band: bright mask fails, edge fallback crops.
        image = Image.new("RGB", (640, 128), "black")
        draw = ImageDraw.Draw(image)
        draw.rectangle((150, 88, 490, 108), fill=(120, 120, 120))
        diagnostics = {}
        crops = prepare_paddleocr_subtitle_line_images(
            image,
            PaddleOCRSettings(upscale=1.0),
            diagnostics=diagnostics,
        )
        self.assertEqual(len(crops), 1)
        self.assertGreaterEqual(int(diagnostics.get("edge_fallback_attempted", 0) or 0), 1)
        self.assertGreaterEqual(int(diagnostics.get("edge_fallback_crop_success", 0) or 0), 1)

    def test_subtitle_diagnostics_empty_frame_has_no_crop_reason(self):
        import paddle_ocr_backend
        from paddle_ocr_backend import (
            PaddleOCRSettings,
            prepare_paddleocr_subtitle_line_images,
            reset_subtitle_fastpath_diagnostics,
        )

        reset_subtitle_fastpath_diagnostics()
        image = Image.new("RGB", (640, 128), "black")
        diagnostics = {}
        crops = prepare_paddleocr_subtitle_line_images(
            image,
            PaddleOCRSettings(upscale=1.0),
            diagnostics=diagnostics,
        )
        self.assertEqual(crops, [])
        # Empty/near-empty frames should land in bright/edge no-candidate path,
        # not invent OCR text.
        self.assertTrue(
            int(diagnostics.get("bright_mask_empty", 0) or 0) >= 1
            or int(diagnostics.get("edge_fallback_no_crop", 0) or 0) >= 1
            or any(str(k).startswith("no_crop_reason_") for k in diagnostics)
        )

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

    def test_subtitle_fast_path_error_log_has_sanitized_bounded_reason(self):
        import paddle_ocr_backend

        image = Image.new("RGB", (720, 96), "black")
        draw = ImageDraw.Draw(image)
        draw.rectangle((230, 32, 490, 60), fill="white")
        private_reason = (
            "CUDA kernel unavailable for 'subtitle-secret'\n"
            + ("diagnostic " * 40)
        )

        with patch.object(
            paddle_ocr_backend,
            "get_paddleocr_text_recognition_engine",
            side_effect=RuntimeError(private_reason),
        ):
            with patch.object(
                paddle_ocr_backend,
                "recognize_with_paddleocr",
                return_value=("Fallback text", []),
            ):
                with patch.object(
                    paddle_ocr_backend,
                    "log_debug_coalesced",
                ) as log_coalesced:
                    text, _lines = (
                        paddle_ocr_backend.recognize_subtitle_with_paddleocr(
                            image,
                            paddle_ocr_backend.PaddleOCRSettings(),
                        )
                    )

        self.assertEqual(text, "Fallback text")
        error_calls = [
            call
            for call in log_coalesced.call_args_list
            if call.args[0] == "paddle-subtitle-fast-path-error"
        ]
        self.assertEqual(len(error_calls), 1)
        message = error_calls[0].args[1]
        self.assertIn("CUDA kernel unavailable", message)
        self.assertIn("<redacted>", message)
        self.assertNotIn("subtitle-secret", message)
        self.assertNotIn("\n", message)
        self.assertLessEqual(len(message), 220)


class PaddleOCRConfigAndUITests(unittest.TestCase):
    def test_paddleocr_defaults_exist(self):
        from config_manager import DEFAULT_CONFIG_SETTINGS

        self.assertEqual(DEFAULT_CONFIG_SETTINGS["ocr_model"], "paddleocr")
        self.assertEqual(DEFAULT_CONFIG_SETTINGS["paddleocr_source_dir"], "PaddleOCR-3.7.0")
        self.assertEqual(DEFAULT_CONFIG_SETTINGS["paddleocr_ocr_version"], "PP-OCRv6")
        self.assertEqual(DEFAULT_CONFIG_SETTINGS["paddleocr_model_size"], "tiny")
        self.assertEqual(DEFAULT_CONFIG_SETTINGS["paddleocr_min_score"], "0.45")
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
                    with patch.object(app_logic, "log_debug") as log_debug:
                        started = app_logic.GameChangingTranslator.ensure_paddleocr_ready_if_selected(
                            app,
                            "settings saved",
                        )

        self.assertTrue(started)
        self.assertEqual(len(created_threads), 1)
        self.assertEqual(created_threads[0].name, "PaddleOCRPrewarm")
        self.assertIsInstance(text_engine.call_args.args[0], PaddleOCRSettings)
        self.assertEqual(text_engine.call_args.args[0].ocr_version, "PP-OCRv6")
        full_engine.assert_called_once()
        self.assertEqual(full_engine.call_args.args[0], text_engine.call_args.args[0])
        self.assertIn("phase_metrics", full_engine.call_args.kwargs)
        messages = [call.args[0] for call in log_debug.call_args_list]
        self.assertTrue(any("phase=text_recognition" in message for message in messages))
        self.assertTrue(any("phase=full_engine" in message for message in messages))

    def test_worker_waits_for_matching_paddleocr_prewarm(self):
        import app_logic
        import worker_threads

        app = self._paddle_app()
        settings = worker_threads.get_paddleocr_settings_from_app(app)
        app._paddleocr_prewarm_lock = threading.RLock()
        app._paddleocr_prewarm_thread = types.SimpleNamespace(
            is_alive=lambda: True
        )
        app._paddleocr_prewarm_settings = settings
        app._paddleocr_prewarmed_settings = None
        app._paddleocr_prewarm_generation = 1
        app._paddleocr_prewarm_event = threading.Event()
        result = []
        finished = threading.Event()

        def wait_for_ready():
            result.append(
                app_logic.GameChangingTranslator.wait_for_paddleocr_prewarm(
                    app,
                    settings,
                    timeout=1.0,
                )
            )
            finished.set()

        waiter = threading.Thread(target=wait_for_ready)
        waiter.start()
        self.assertFalse(finished.wait(0.05))

        with app._paddleocr_prewarm_lock:
            app._paddleocr_prewarmed_settings = settings
            app._paddleocr_prewarm_event.set()

        waiter.join(timeout=1.0)
        self.assertEqual(result, [True])

    def test_paddleocr_prewarm_starts_thread_while_state_lock_is_owned(self):
        import app_logic
        import worker_threads

        app = self._paddle_app()
        settings = worker_threads.get_paddleocr_settings_from_app(app)
        lock_owned_at_start = []

        class RecordingThread:
            def __init__(self, *args, **kwargs):
                self.started = False

            def start(self):
                lock_owned_at_start.append(
                    app._paddleocr_prewarm_lock._is_owned()
                )
                self.started = True

            def is_alive(self):
                return self.started

        with patch.object(
            app_logic.threading,
            "Thread",
            side_effect=RecordingThread,
        ):
            self.assertTrue(
                app_logic.GameChangingTranslator.start_paddleocr_prewarm(
                    app,
                    settings,
                    "test",
                )
            )

        self.assertEqual(lock_owned_at_start, [True])

    def test_different_paddleocr_prewarm_settings_do_not_start_concurrently(self):
        import app_logic
        import worker_threads
        from dataclasses import replace

        app = self._paddle_app()
        first_settings = worker_threads.get_paddleocr_settings_from_app(app)
        second_settings = replace(first_settings, model_size="small")
        app._paddleocr_prewarm_lock = threading.RLock()
        app._paddleocr_prewarm_thread = types.SimpleNamespace(
            is_alive=lambda: True
        )
        app._paddleocr_prewarm_settings = first_settings
        app._paddleocr_prewarmed_settings = None
        app._paddleocr_prewarm_generation = 1
        app._paddleocr_prewarm_event = threading.Event()

        with patch.object(app_logic.threading, "Thread") as thread_factory:
            started = (
                app_logic.GameChangingTranslator.start_paddleocr_prewarm(
                    app,
                    second_settings,
                    "settings changed",
                )
            )

        self.assertFalse(started)
        thread_factory.assert_not_called()

    def test_invalidated_paddleocr_wait_tracks_the_still_active_operation(self):
        import app_logic
        import worker_threads
        from dataclasses import replace

        app = self._paddle_app()
        first_settings = worker_threads.get_paddleocr_settings_from_app(app)
        second_settings = replace(first_settings, model_size="small")
        app._paddleocr_prewarm_lock = threading.RLock()
        app._paddleocr_prewarm_thread = types.SimpleNamespace(
            is_alive=lambda: True
        )
        app._paddleocr_prewarm_settings = first_settings
        app._paddleocr_prewarmed_settings = None
        app._paddleocr_prewarm_generation = 1
        active_event = threading.Event()
        app._paddleocr_prewarm_event = active_event
        result = []
        finished = threading.Event()

        app_logic.GameChangingTranslator._invalidate_paddleocr_prewarm_state(
            app
        )

        waiter = threading.Thread(
            target=lambda: (
                result.append(
                    app_logic.GameChangingTranslator.wait_for_paddleocr_prewarm(
                        app,
                        second_settings,
                        timeout=1.0,
                    )
                ),
                finished.set(),
            )
        )
        waiter.start()
        self.assertFalse(finished.wait(0.05))

        active_event.set()
        self.assertTrue(finished.wait(0.2))
        waiter.join(timeout=1.0)

        self.assertFalse(waiter.is_alive())
        self.assertEqual(result, [False])

    def test_paddleocr_worker_wait_default_is_twenty_seconds(self):
        import app_logic
        import inspect

        self.assertEqual(
            inspect.signature(
                app_logic.GameChangingTranslator.wait_for_paddleocr_prewarm
            ).parameters["timeout"].default,
            20.0,
        )

    def test_worker_stops_waiting_for_paddleocr_when_app_stops(self):
        import app_logic
        import worker_threads

        app = self._paddle_app()
        settings = worker_threads.get_paddleocr_settings_from_app(app)
        app.is_running = True
        app._paddleocr_prewarm_lock = threading.RLock()
        app._paddleocr_prewarm_thread = types.SimpleNamespace(
            is_alive=lambda: True
        )
        app._paddleocr_prewarm_settings = settings
        app._paddleocr_prewarmed_settings = None
        app._paddleocr_prewarm_generation = 1
        app._paddleocr_prewarm_event = threading.Event()
        result = []

        waiter = threading.Thread(
            target=lambda: result.append(
                app_logic.GameChangingTranslator.wait_for_paddleocr_prewarm(
                    app,
                    settings,
                    timeout=1.0,
                )
            )
        )
        waiter.start()
        app.is_running = False
        waiter.join(timeout=0.5)

        self.assertFalse(waiter.is_alive())
        self.assertEqual(result, [False])

    def _inline_thread_factory(self, created_threads=None):
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
            if created_threads is not None:
                created_threads.append(thread)
            return thread

        return make_thread

    def test_settings_summary_omits_user_paths_and_secrets(self):
        from paddle_ocr_backend import PaddleOCRSettings, summarize_paddleocr_settings

        summary = summarize_paddleocr_settings(
            PaddleOCRSettings(source_dir=r"C:\Users\secret\models\billing-client-secret")
        )

        self.assertNotIn("source_dir_name", summary)
        self.assertTrue(summary["source_dir_configured"])
        self.assertNotIn("Users", str(summary))
        self.assertNotIn("secret", str(summary))
        self.assertEqual(summary["model_size"], "tiny")
        self.assertEqual(summary["device"], "cpu")

    def test_engine_phase_metrics_distinguish_construct_and_cache_hit(self):
        from paddle_ocr_backend import (
            PaddleOCRSettings,
            clear_paddleocr_engines,
            get_paddleocr_text_recognition_engine,
        )

        clear_paddleocr_engines()
        settings = PaddleOCRSettings()
        cold = {}
        hot = {}

        class FakeEngine:
            def __init__(self, **kwargs):
                pass

        with patch(
            "paddle_ocr_backend._import_text_recognition",
            return_value=FakeEngine,
        ):
            with patch(
                "paddle_ocr_backend._probe_ppocrv6_model_files",
                side_effect=["absent", "present", "present", "present"],
            ):
                get_paddleocr_text_recognition_engine(settings, phase_metrics=cold)
                get_paddleocr_text_recognition_engine(settings, phase_metrics=hot)

        self.assertFalse(cold["cache_hit"])
        self.assertEqual(cold["build_kind"], "model_download_and_build")
        self.assertEqual(cold["engine_kind"], "text_recognition")
        self.assertIn("import_s", cold)
        self.assertIn("model_files_probe_s", cold)
        self.assertIn("construct_s", cold)
        self.assertIn("total_s", cold)
        self.assertTrue(hot["cache_hit"])
        self.assertEqual(hot["build_kind"], "cache_hit")
        self.assertEqual(hot["import_s"], 0.0)
        self.assertEqual(hot["model_files_probe_s"], 0.0)
        self.assertEqual(hot["construct_s"], 0.0)
        clear_paddleocr_engines()

    def test_prewarm_publishes_text_recognition_breakdown_metrics_and_log(self):
        import app_logic
        import worker_threads
        from runtime_metrics import RuntimeMetrics

        app = self._paddle_app()
        app.runtime_metrics = RuntimeMetrics(max_age_seconds=0)
        settings = worker_threads.get_paddleocr_settings_from_app(app)
        logs = []

        def text_engine(settings_arg, phase_metrics=None, clock=None):
            if phase_metrics is not None:
                phase_metrics.update(
                    {
                        "engine_kind": "text_recognition",
                        "cache_hit": False,
                        "build_kind": "model_build_cached_files",
                        "import_s": 12.5,
                        "model_files_probe_s": 0.01,
                        "construct_s": 3.25,
                        "total_s": 15.76,
                        "model_files_before": "present",
                        "model_files_after": "present",
                    }
                )

        def full_engine(settings_arg, phase_metrics=None, clock=None):
            if phase_metrics is not None:
                phase_metrics.update(
                    {
                        "engine_kind": "full",
                        "cache_hit": True,
                        "build_kind": "cache_hit",
                        "import_s": 0.0,
                        "model_files_probe_s": 0.0,
                        "construct_s": 0.0,
                        "total_s": 0.0,
                        "model_files_before": "present",
                        "model_files_after": "present",
                    }
                )

        with patch.object(
            app_logic.threading,
            "Thread",
            side_effect=self._inline_thread_factory(),
        ):
            with patch.object(
                app_logic,
                "get_paddleocr_text_recognition_engine",
                side_effect=text_engine,
            ):
                with patch.object(
                    app_logic,
                    "get_paddleocr_engine",
                    side_effect=full_engine,
                ):
                    with patch.object(app_logic, "log_debug", side_effect=logs.append):
                        app_logic.GameChangingTranslator.start_paddleocr_prewarm(
                            app,
                            settings,
                            "application startup",
                        )

        snapshot = app.get_paddleocr_prewarm_metrics_snapshot()
        text_phase = snapshot["text_recognition"]
        self.assertEqual(text_phase["import_s"], 12.5)
        self.assertEqual(text_phase["model_files_probe_s"], 0.01)
        self.assertEqual(text_phase["construct_s"], 3.25)
        self.assertEqual(text_phase["total_s"], 15.76)
        gauges = app.runtime_metrics.snapshot()["gauges"]
        self.assertEqual(
            gauges["paddleocr_prewarm_text_recognition_import_s"],
            12.5,
        )
        self.assertEqual(
            gauges["paddleocr_prewarm_text_recognition_model_files_probe_s"],
            0.01,
        )
        self.assertEqual(
            gauges["paddleocr_prewarm_text_recognition_construct_s"],
            3.25,
        )
        breakdown_logs = [
            message
            for message in logs
            if "phase=text_recognition_breakdown" in str(message)
        ]
        self.assertTrue(breakdown_logs)
        self.assertIn("import_s=12.5", breakdown_logs[-1])
        self.assertIn("probe_s=0.01", breakdown_logs[-1])
        self.assertIn("construct_s=3.25", breakdown_logs[-1])
        self.assertIn("files_before=present", breakdown_logs[-1])

    def test_prewarm_cache_hit_breakdown_metrics_are_zeroed(self):
        import app_logic
        import worker_threads
        from runtime_metrics import RuntimeMetrics

        app = self._paddle_app()
        app.runtime_metrics = RuntimeMetrics(max_age_seconds=0)
        settings = worker_threads.get_paddleocr_settings_from_app(app)

        def text_engine(settings_arg, phase_metrics=None, clock=None):
            if phase_metrics is not None:
                phase_metrics.update(
                    {
                        "engine_kind": "text_recognition",
                        "cache_hit": True,
                        "build_kind": "cache_hit",
                        "import_s": 0.0,
                        "model_files_probe_s": 0.0,
                        "construct_s": 0.0,
                        "total_s": 0.0,
                        "model_files_before": "present",
                        "model_files_after": "present",
                    }
                )

        def full_engine(settings_arg, phase_metrics=None, clock=None):
            if phase_metrics is not None:
                phase_metrics.update(
                    {
                        "engine_kind": "full",
                        "cache_hit": True,
                        "build_kind": "cache_hit",
                        "import_s": 0.0,
                        "model_files_probe_s": 0.0,
                        "construct_s": 0.0,
                        "total_s": 0.0,
                    }
                )

        with patch.object(
            app_logic.threading,
            "Thread",
            side_effect=self._inline_thread_factory(),
        ):
            with patch.object(
                app_logic,
                "get_paddleocr_text_recognition_engine",
                side_effect=text_engine,
            ):
                with patch.object(
                    app_logic,
                    "get_paddleocr_engine",
                    side_effect=full_engine,
                ):
                    app_logic.GameChangingTranslator.start_paddleocr_prewarm(
                        app,
                        settings,
                        "application startup",
                    )

        gauges = app.runtime_metrics.snapshot()["gauges"]
        self.assertEqual(gauges["paddleocr_prewarm_text_recognition_import_s"], 0.0)
        self.assertEqual(
            gauges["paddleocr_prewarm_text_recognition_model_files_probe_s"],
            0.0,
        )
        self.assertEqual(
            gauges["paddleocr_prewarm_text_recognition_construct_s"],
            0.0,
        )

    def test_prewarm_metrics_cover_cold_hot_ready_settings_change_and_failure(self):
        import app_logic
        import worker_threads
        from dataclasses import replace
        from runtime_metrics import RuntimeMetrics

        app = self._paddle_app()
        app.runtime_metrics = RuntimeMetrics(max_age_seconds=0)
        settings = worker_threads.get_paddleocr_settings_from_app(app)
        changed_settings = replace(settings, model_size="small")

        clock = {"value": 100.0}

        def monotonic():
            return clock["value"]

        def advance(seconds):
            clock["value"] += float(seconds)

        def text_engine(settings_arg, phase_metrics=None, clock=None):
            advance(1.5)
            if phase_metrics is not None:
                phase_metrics.update(
                    {
                        "engine_kind": "text_recognition",
                        "cache_hit": False,
                        "build_kind": "model_build_cached_files",
                        "total_s": 1.5,
                    }
                )

        def full_engine(settings_arg, phase_metrics=None, clock=None):
            advance(2.5)
            if phase_metrics is not None:
                phase_metrics.update(
                    {
                        "engine_kind": "full",
                        "cache_hit": False,
                        "build_kind": "model_build_cached_files",
                        "total_s": 2.5,
                    }
                )

        with patch.object(app_logic.time, "monotonic", side_effect=monotonic):
            with patch.object(
                app_logic.threading,
                "Thread",
                side_effect=self._inline_thread_factory(),
            ):
                with patch.object(
                    app_logic,
                    "get_paddleocr_text_recognition_engine",
                    side_effect=text_engine,
                ):
                    with patch.object(
                        app_logic,
                        "get_paddleocr_engine",
                        side_effect=full_engine,
                    ):
                        started = app_logic.GameChangingTranslator.start_paddleocr_prewarm(
                            app,
                            settings,
                            "application startup",
                        )

        self.assertTrue(started)
        cold = app.get_paddleocr_prewarm_metrics_snapshot()
        self.assertEqual(cold["status"], "completed")
        self.assertEqual(cold["outcome"], "completed")
        self.assertEqual(cold["trigger"], "startup")
        self.assertEqual(cold["start_path"], "start_thread")
        self.assertEqual(cold["generation"], 1)
        self.assertTrue(cold["ready"])
        self.assertFalse(cold["active"])
        self.assertEqual(cold["phase_text_recognition_s"], 1.5)
        self.assertEqual(cold["phase_full_engine_s"], 2.5)
        self.assertEqual(cold["total_s"], 4.0)
        self.assertEqual(cold["settings_summary"]["model_size"], "tiny")
        self.assertIn("cpu_count", cold["host"])
        self.assertNotIn("Users", str(cold))

        skipped_ready = app_logic.GameChangingTranslator.start_paddleocr_prewarm(
            app,
            settings,
            "settings saved",
        )
        ready_snapshot = app.get_paddleocr_prewarm_metrics_snapshot()
        self.assertFalse(skipped_ready)
        self.assertEqual(ready_snapshot["outcome"], "already_ready")
        self.assertTrue(ready_snapshot["ready"])

        # Settings change after ready: invalidate then start a new generation.
        app_logic.GameChangingTranslator.clear_paddleocr_runtime_cache(
            app,
            "settings changed",
        )
        with patch.object(app_logic.time, "monotonic", side_effect=monotonic):
            with patch.object(
                app_logic.threading,
                "Thread",
                side_effect=self._inline_thread_factory(),
            ):
                with patch.object(
                    app_logic,
                    "get_paddleocr_text_recognition_engine",
                    side_effect=text_engine,
                ):
                    with patch.object(
                        app_logic,
                        "get_paddleocr_engine",
                        side_effect=full_engine,
                    ):
                        changed = app_logic.GameChangingTranslator.start_paddleocr_prewarm(
                            app,
                            changed_settings,
                            "settings saved",
                        )
        changed_snapshot = app.get_paddleocr_prewarm_metrics_snapshot()
        self.assertTrue(changed)
        self.assertGreaterEqual(changed_snapshot["generation"], 2)
        self.assertEqual(changed_snapshot["settings_summary"]["model_size"], "small")
        self.assertEqual(changed_snapshot["status"], "completed")

        fail_app = self._paddle_app()
        fail_app.runtime_metrics = RuntimeMetrics(max_age_seconds=0)
        with patch.object(
            app_logic.threading,
            "Thread",
            side_effect=self._inline_thread_factory(),
        ):
            with patch.object(
                app_logic,
                "get_paddleocr_text_recognition_engine",
                side_effect=RuntimeError('failed "OCR SECRET TEXT"'),
            ):
                app_logic.GameChangingTranslator.start_paddleocr_prewarm(
                    fail_app,
                    settings,
                    "settings saved",
                )
        failed = fail_app.get_paddleocr_prewarm_metrics_snapshot()
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["error_type"], "RuntimeError")
        self.assertFalse(failed["ready"])
        self.assertNotIn("OCR SECRET TEXT", str(failed))
        self.assertNotIn("OCR SECRET TEXT", fail_app.runtime_metrics.summary_text())

    def test_worker_wait_and_first_ocr_wait_metrics_are_distinct(self):
        import app_logic
        import worker_threads
        from runtime_metrics import RuntimeMetrics

        app = self._paddle_app()
        app.runtime_metrics = RuntimeMetrics(max_age_seconds=0)
        settings = worker_threads.get_paddleocr_settings_from_app(app)
        app._paddleocr_prewarm_lock = threading.RLock()
        app._paddleocr_prewarm_thread = types.SimpleNamespace(is_alive=lambda: True)
        app._paddleocr_prewarm_settings = settings
        app._paddleocr_prewarmed_settings = None
        app._paddleocr_prewarm_generation = 3
        app._paddleocr_prewarm_event = threading.Event()
        app._paddleocr_prewarm_metrics = (
            app_logic.GameChangingTranslator._new_paddleocr_prewarm_metrics()
        )
        app._paddleocr_first_ocr_wait_generation = 0

        def release_after_wait():
            time.sleep(0.05)
            with app._paddleocr_prewarm_lock:
                app._paddleocr_prewarmed_settings = settings
                app._paddleocr_prewarm_event.set()

        releaser = threading.Thread(target=release_after_wait)
        releaser.start()
        ready = app_logic.GameChangingTranslator.wait_for_paddleocr_prewarm(
            app,
            settings,
            timeout=1.0,
        )
        releaser.join(timeout=1.0)
        wait_snapshot = app.get_paddleocr_prewarm_metrics_snapshot()
        self.assertTrue(ready)
        self.assertEqual(wait_snapshot["wait_path"], "worker_wait")
        self.assertTrue(wait_snapshot["waited_for_ready"])
        self.assertGreaterEqual(wait_snapshot["wait_s"], 0.0)
        self.assertNotEqual(wait_snapshot["wait_path"], "worker_first_ocr")

        app_logic.GameChangingTranslator.note_paddleocr_first_ocr_wait(
            app,
            settings,
            waited_s=0.37,
            ready=True,
            generation=3,
        )
        first_snapshot = app.get_paddleocr_prewarm_metrics_snapshot()
        self.assertEqual(first_snapshot["wait_path"], "worker_first_ocr")
        self.assertEqual(first_snapshot["first_ocr_wait_s"], 0.37)
        self.assertEqual(first_snapshot["first_ocr_wait_generation"], 3)

        # First-OCR wait is one-shot per generation.
        app_logic.GameChangingTranslator.note_paddleocr_first_ocr_wait(
            app,
            settings,
            waited_s=9.0,
            ready=False,
            generation=3,
        )
        self.assertEqual(
            app.get_paddleocr_prewarm_metrics_snapshot()["first_ocr_wait_s"],
            0.37,
        )

        runtime = app.runtime_metrics.snapshot()
        self.assertIn("paddleocr_prewarm_wait_duration", runtime["timings"])
        self.assertIn("paddleocr_first_ocr_wait_duration", runtime["timings"])
        self.assertEqual(
            runtime["timings"]["paddleocr_first_ocr_wait_duration"]["latest"],
            0.37,
        )
        self.assertEqual(runtime["labels"]["paddleocr_prewarm_wait_path"], "worker_first_ocr")

    def test_worker_wait_does_not_duplicate_terminal_prewarm_timings(self):
        import app_logic
        import worker_threads
        from runtime_metrics import RuntimeMetrics

        app = self._paddle_app()
        app.runtime_metrics = RuntimeMetrics(max_age_seconds=0)
        settings = worker_threads.get_paddleocr_settings_from_app(app)
        app._paddleocr_prewarm_lock = threading.RLock()
        app._paddleocr_prewarm_thread = None
        app._paddleocr_prewarm_settings = None
        app._paddleocr_prewarmed_settings = settings
        app._paddleocr_prewarm_generation = 5
        app._paddleocr_prewarm_event = threading.Event()
        app._paddleocr_prewarm_metrics = (
            app_logic.GameChangingTranslator._new_paddleocr_prewarm_metrics()
        )

        app_logic.GameChangingTranslator._update_paddleocr_prewarm_metrics(
            app,
            generation=5,
            status="completed",
            outcome="completed",
            phase_text_recognition_s=1.0,
            phase_full_engine_s=2.0,
            total_s=3.0,
            ready=True,
        )
        app_logic.GameChangingTranslator.wait_for_paddleocr_prewarm(
            app,
            settings,
            timeout=20.0,
        )
        app_logic.GameChangingTranslator.wait_for_paddleocr_prewarm(
            app,
            settings,
            timeout=20.0,
        )

        timings = app.runtime_metrics.snapshot()["timings"]
        self.assertEqual(
            timings["paddleocr_prewarm_total_duration"]["count"],
            1,
        )
        self.assertEqual(
            timings["paddleocr_prewarm_text_recognition_duration"]["count"],
            1,
        )
        self.assertEqual(
            timings["paddleocr_prewarm_full_engine_duration"]["count"],
            1,
        )

    def test_process_local_ocr_frame_records_first_ocr_wait_metric(self):
        import worker_threads
        from runtime_metrics import RuntimeMetrics

        image = Image.new("RGB", (16, 10), "white")
        noted = []
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
            runtime_metrics=RuntimeMetrics(max_age_seconds=0),
            wait_for_paddleocr_prewarm=Mock(return_value=True),
            note_paddleocr_first_ocr_wait=lambda settings, waited_s, ready: noted.append(
                (settings.model_size, round(float(waited_s), 4), bool(ready))
            ),
        )

        with patch.object(
            worker_threads,
            "recognize_subtitle_with_paddleocr",
            return_value=("Hello world", []),
        ):
            text, _img, engine_label = worker_threads.process_local_ocr_frame(
                app,
                image,
                "paddleocr",
            )

        self.assertEqual(text, "Hello world")
        self.assertEqual(engine_label, "PaddleOCR")
        self.assertEqual(len(noted), 1)
        self.assertEqual(noted[0][0], "tiny")
        self.assertTrue(noted[0][2])
        app.wait_for_paddleocr_prewarm.assert_called_once_with(
            worker_threads.get_paddleocr_settings_from_app(app),
            timeout=20.0,
        )


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

    def test_worker_paddleocr_settings_use_shared_numeric_coercion(self):
        import worker_capture
        import worker_threads
        from paddle_ocr_backend import _coerce_float, _coerce_int

        app = types.SimpleNamespace(
            paddleocr_source_dir_var=types.SimpleNamespace(get=lambda: "PaddleOCR-3.7.0"),
            paddleocr_lang_var=types.SimpleNamespace(get=lambda: "en"),
            paddleocr_ocr_version_var=types.SimpleNamespace(get=lambda: "PP-OCRv6"),
            paddleocr_model_size_var=types.SimpleNamespace(get=lambda: "tiny"),
            paddleocr_device_var=types.SimpleNamespace(get=lambda: "cpu"),
            paddleocr_min_score_var=types.SimpleNamespace(get=lambda: "not-a-number"),
            paddleocr_upscale_var=types.SimpleNamespace(get=lambda: "9.0"),
            paddleocr_text_det_limit_side_len_var=types.SimpleNamespace(get=lambda: "12"),
            paddleocr_text_det_limit_type_var=types.SimpleNamespace(get=lambda: "max"),
            paddleocr_use_textline_orientation_var=types.SimpleNamespace(get=lambda: "yes"),
        )

        settings = worker_threads.get_paddleocr_settings_from_app(app)

        # Characterisation: invalid / out-of-range UI values must keep the same
        # clamp-and-default contract as paddle_ocr_backend helpers.
        self.assertEqual(
            settings.min_score,
            _coerce_float("not-a-number", 0.45, 0.0, 1.0),
        )
        self.assertEqual(
            settings.upscale,
            _coerce_float("9.0", 1.0, 1.0, 4.0),
        )
        self.assertEqual(
            settings.text_det_limit_side_len,
            _coerce_int("12", 960, 128, 4096),
        )
        self.assertEqual(settings.min_score, 0.45)
        self.assertEqual(settings.upscale, 4.0)
        self.assertEqual(settings.text_det_limit_side_len, 128)
        self.assertTrue(settings.use_textline_orientation)
        # Dedup invariant: worker modules re-export the backend helpers.
        self.assertIs(worker_capture._coerce_float, _coerce_float)
        self.assertIs(worker_capture._coerce_int, _coerce_int)
        self.assertIs(worker_threads._coerce_float, _coerce_float)
        self.assertIs(worker_threads._coerce_int, _coerce_int)

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

    def test_custom_ai_ocr_queue_keeps_latest_frame_only(self):
        import worker_threads

        increments = []
        app = types.SimpleNamespace(
            ocr_queue=queue.Queue(maxsize=8),
            runtime_metrics=types.SimpleNamespace(
                increment=lambda name, amount=1: increments.append((name, amount))
            ),
        )
        old_frame = object()
        newer_frame = object()
        latest_frame = object()
        app.ocr_queue.put_nowait(old_frame)
        app.ocr_queue.put_nowait(newer_frame)

        enqueued = worker_threads.enqueue_ocr_frame_for_model(
            app,
            latest_frame,
            "custom_ai",
        )

        self.assertTrue(enqueued)
        self.assertEqual(app.ocr_queue.qsize(), 1)
        self.assertIs(app.ocr_queue.get_nowait(), latest_frame)
        self.assertIn(("ocr_queue_stale_frame_drop", 2), increments)

    def test_custom_ai_ocr_queue_full_replaces_with_latest_frame(self):
        import worker_threads

        increments = []
        app = types.SimpleNamespace(
            ocr_queue=queue.Queue(maxsize=1),
            runtime_metrics=types.SimpleNamespace(
                increment=lambda name, amount=1: increments.append((name, amount))
            ),
        )
        old_frame = object()
        latest_frame = object()
        app.ocr_queue.put_nowait(old_frame)

        enqueued = worker_threads.enqueue_ocr_frame_for_model(
            app,
            latest_frame,
            "custom_ai",
        )

        self.assertTrue(enqueued)
        self.assertEqual(app.ocr_queue.qsize(), 1)
        self.assertIs(app.ocr_queue.get_nowait(), latest_frame)
        self.assertTrue(
            any(name == "ocr_queue_stale_frame_drop" for name, _amount in increments)
        )

    def test_paddleocr_first_clear_text_submits_without_legacy_stability_threshold(self):
        import worker_threads
        from paddle_ocr_backend import PaddleOCRSettings
        from worker_capture import CaptureUISnapshot

        submitted = []
        scheduled = []
        image = Image.new("RGB", (16, 10), "white")
        snapshot = CaptureUISnapshot(
            generation=1,
            source_geometry=None,
            ocr_model="paddleocr",
            scan_interval_ms=100,
            base_scan_interval_ms=100,
            keep_linebreaks=False,
            is_api_based=False,
            paddleocr_settings=PaddleOCRSettings(),
        )
        image._gct_capture_snapshot = snapshot
        app = types.SimpleNamespace(
            is_running=True,
            ocr_queue=queue.Queue(),
            capture_ui_snapshot=snapshot,
            previous_text="",
            text_stability_counter=0,
            stable_threshold=2,
            is_placeholder_text=lambda _text: False,
            calculate_text_similarity=lambda _current, _previous: 0.0,
            reset_clear_timeout=Mock(),
            root=types.SimpleNamespace(
                after=lambda delay, callback, *args: scheduled.append(
                    (delay, callback, args)
                ),
                winfo_exists=lambda: True,
            ),
            _app_is_closing=False,
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
                for _delay, callback, args in list(scheduled):
                    callback(*args)

        self.assertEqual(submitted, ["Clear subtitle text."])
        self.assertEqual(len(scheduled), 1)
        self.assertEqual(scheduled[0][0], 0)

    def test_process_local_ocr_frame_routes_paddleocr(self):
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
            ocr_debugging_var=types.SimpleNamespace(get=lambda: True),
            wait_for_paddleocr_prewarm=Mock(return_value=True),
        )

        with patch.object(
            worker_threads,
            "recognize_subtitle_with_paddleocr",
            return_value=("Hello world", []),
        ) as recognize:
            text, processed_cv_img, engine_label = worker_threads.process_local_ocr_frame(
                app,
                image,
                "paddleocr",
            )

        self.assertEqual(text, "Hello world")
        self.assertEqual(engine_label, "PaddleOCR")
        self.assertEqual(processed_cv_img.shape[0:2], (10, 16))
        recognize.assert_called_once()
        app.wait_for_paddleocr_prewarm.assert_called_once_with(
            recognize.call_args.args[1],
            timeout=20.0,
        )

    def test_process_local_ocr_frame_skips_debug_image_when_debugging_off(self):
        import worker_threads
        import worker_capture

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
            ocr_debugging_var=types.SimpleNamespace(get=lambda: False),
            wait_for_paddleocr_prewarm=Mock(return_value=True),
        )

        with patch.object(
            worker_threads,
            "recognize_subtitle_with_paddleocr",
            return_value=("Hello world", []),
        ), patch.object(
            worker_capture,
            "_prepare_paddleocr_image",
        ) as prepare_preview:
            text, processed_cv_img, engine_label = worker_threads.process_local_ocr_frame(
                app,
                image,
                "paddleocr",
            )

        # Debugging off: no preview image built and the expensive prep is skipped.
        self.assertEqual(text, "Hello world")
        self.assertEqual(engine_label, "PaddleOCR")
        self.assertIsNone(processed_cv_img)
        prepare_preview.assert_not_called()

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
