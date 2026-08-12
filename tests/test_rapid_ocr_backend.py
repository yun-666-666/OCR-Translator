import configparser
import io
import logging
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

import rapid_ocr_backend
import worker_capture
from config_manager import DEFAULT_CONFIG_SETTINGS, load_app_config
from rapid_ocr_backend import (
    RAPIDOCR_MODEL_CODE,
    RapidOCRSettings,
    flatten_rapidocr_result,
)


class RapidOCRBackendTests(unittest.TestCase):
    def tearDown(self):
        rapid_ocr_backend.clear_rapidocr_engine()

    def test_default_configuration_selects_rapidocr(self):
        self.assertEqual(DEFAULT_CONFIG_SETTINGS["ocr_model"], RAPIDOCR_MODEL_CODE)

    def test_removed_legacy_ocr_model_migrates_to_rapidocr(self):
        import config_manager

        with tempfile.TemporaryDirectory() as temp_dir:
            previous = os.environ.get(config_manager.CONFIG_DIR_ENV)
            os.environ[config_manager.CONFIG_DIR_ENV] = temp_dir
            config_path = Path(temp_dir) / "ocr_translator_config.ini"
            try:
                config_path.write_text(
                    "[Settings]\nocr_model = tesseract\n",
                    encoding="utf-8",
                )
                loaded = load_app_config()
            finally:
                if previous is None:
                    os.environ.pop(config_manager.CONFIG_DIR_ENV, None)
                else:
                    os.environ[config_manager.CONFIG_DIR_ENV] = previous

            self.assertEqual(loaded["Settings"]["ocr_model"], RAPIDOCR_MODEL_CODE)
            persisted = configparser.ConfigParser()
            persisted.read(config_path, encoding="utf-8")
            self.assertEqual(persisted["Settings"]["ocr_model"], RAPIDOCR_MODEL_CODE)

    def test_build_engine_uses_benchmarked_onnx_configuration(self):
        fake_engine = object()
        with patch("rapidocr.RapidOCR", return_value=fake_engine) as constructor, patch(
            "rapid_ocr_backend._model_root_dir",
            return_value=Path("C:/rapid-models"),
        ):
            built = rapid_ocr_backend._build_engine()

        self.assertIs(built, fake_engine)
        params = constructor.call_args.kwargs["params"]
        self.assertEqual(params["Global.text_score"], 0.45)
        self.assertFalse(params["Global.use_cls"])
        self.assertEqual(params["EngineConfig.onnxruntime.intra_op_num_threads"], 2)
        self.assertEqual(params["EngineConfig.onnxruntime.inter_op_num_threads"], 1)
        self.assertFalse(params["EngineConfig.onnxruntime.use_cuda"])
        self.assertEqual(params["Det.ocr_version"].value, "PP-OCRv6")
        self.assertEqual(params["Det.model_type"].value, "tiny")
        self.assertEqual(params["Det.limit_side_len"], 960)
        self.assertEqual(params["Det.limit_type"], "max")
        self.assertEqual(params["Rec.ocr_version"].value, "PP-OCRv6")
        self.assertEqual(params["Rec.model_type"].value, "tiny")

    def test_rapidocr_logger_suppresses_only_empty_detection_warning(self):
        rapidocr_logger = logging.getLogger("RapidOCR")
        original_filters = list(rapidocr_logger.filters)
        original_handlers = list(rapidocr_logger.handlers)
        original_level = rapidocr_logger.level
        original_propagate = rapidocr_logger.propagate
        output = io.StringIO()
        rapidocr_logger.filters = []
        rapidocr_logger.handlers = [logging.StreamHandler(output)]
        rapidocr_logger.setLevel(logging.WARNING)
        rapidocr_logger.propagate = False

        try:
            rapid_ocr_backend._install_rapidocr_warning_filter()
            rapid_ocr_backend._install_rapidocr_warning_filter()
            rapidocr_logger.warning(
                rapid_ocr_backend.RAPIDOCR_EMPTY_DETECTION_WARNING
            )
            rapidocr_logger.warning("A useful RapidOCR warning")
        finally:
            installed_filter_count = sum(
                isinstance(
                    existing_filter,
                    rapid_ocr_backend._SuppressEmptyDetectionWarning,
                )
                for existing_filter in rapidocr_logger.filters
            )
            rapidocr_logger.filters = original_filters
            rapidocr_logger.handlers = original_handlers
            rapidocr_logger.setLevel(original_level)
            rapidocr_logger.propagate = original_propagate

        self.assertEqual(installed_filter_count, 1)
        self.assertNotIn(
            rapid_ocr_backend.RAPIDOCR_EMPTY_DETECTION_WARNING,
            output.getvalue(),
        )
        self.assertIn("A useful RapidOCR warning", output.getvalue())

    def test_flatten_filters_low_scores_and_orders_lines(self):
        result = types.SimpleNamespace(
            txts=("second", "ignored", "first"),
            scores=(0.91, 0.20, 0.97),
            boxes=np.asarray(
                [
                    [[10, 40], [80, 40], [80, 60], [10, 60]],
                    [[0, 20], [20, 20], [20, 30], [0, 30]],
                    [[5, 5], [60, 5], [60, 20], [5, 20]],
                ]
            ),
        )

        text, lines = flatten_rapidocr_result(result, keep_linebreaks=True)

        self.assertEqual(text, "first\nsecond")
        self.assertEqual([line.text for line in lines], ["first", "second"])

    def test_worker_routes_rapidocr_and_keeps_paddle_fallback(self):
        app = types.SimpleNamespace(
            keep_linebreaks_var=types.SimpleNamespace(get=lambda: False),
            ocr_debugging_var=types.SimpleNamespace(get=lambda: False),
        )
        image = Image.new("RGB", (32, 16), color="black")
        snapshot = worker_capture.CaptureUISnapshot(
            generation=1,
            source_geometry=(0, 0, 32, 16),
            ocr_model=RAPIDOCR_MODEL_CODE,
            scan_interval_ms=100,
            base_scan_interval_ms=100,
            keep_linebreaks=False,
            is_api_based=False,
            rapidocr_settings=RapidOCRSettings(),
        )

        with patch(
            "worker_capture._recognize_with_rapidocr",
            return_value=("rapid text", []),
        ) as recognize:
            text, debug_image, label = worker_capture.process_local_ocr_frame(
                app,
                image,
                RAPIDOCR_MODEL_CODE,
                capture_snapshot=snapshot,
            )

        self.assertEqual((text, debug_image, label), ("rapid text", None, "RapidOCR"))
        recognize.assert_called_once()

        snapshot = worker_capture.CaptureUISnapshot(
            **{
                **snapshot.__dict__,
                "paddleocr_settings": worker_capture.PaddleOCRSettings(),
            }
        )
        with patch(
            "worker_capture._recognize_with_rapidocr",
            side_effect=RuntimeError("rapid unavailable"),
        ), patch(
            "worker_capture._recognize_subtitle_with_paddleocr",
            return_value=("fallback text", []),
        ) as fallback:
            text, debug_image, label = worker_capture.process_local_ocr_frame(
                app,
                image,
                RAPIDOCR_MODEL_CODE,
                capture_snapshot=snapshot,
            )

        self.assertEqual(
            (text, debug_image, label),
            ("fallback text", None, "PaddleOCR fallback"),
        )
        fallback.assert_called_once()


if __name__ == "__main__":
    unittest.main()
