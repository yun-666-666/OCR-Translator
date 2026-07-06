import importlib
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


def import_ocr_utils_for_tests():
    if "cv2" not in sys.modules:
        module = types.ModuleType("cv2")
        module.__spec__ = importlib.machinery.ModuleSpec("cv2", loader=None)
        sys.modules["cv2"] = module
    if "tesserocr" not in sys.modules:
        module = types.ModuleType("tesserocr")
        module.__spec__ = importlib.machinery.ModuleSpec("tesserocr", loader=None)
        module.PyTessBaseAPI = object
        module.RIL = types.SimpleNamespace(WORD=0)
        module.get_languages = lambda path=None: (path, ["eng"])
        sys.modules["tesserocr"] = module
    return importlib.import_module("ocr_utils")


def import_worker_threads_for_tests():
    for optional_module in ("cv2", "pyautogui"):
        if optional_module not in sys.modules:
            module = types.ModuleType(optional_module)
            module.__spec__ = importlib.machinery.ModuleSpec(optional_module, loader=None)
            sys.modules[optional_module] = module
    return importlib.import_module("worker_threads")


class AdaptiveBlockSizeNormalizationTests(unittest.TestCase):
    def test_normalize_adaptive_block_size_returns_valid_odd_integer(self):
        ocr_utils = import_ocr_utils_for_tests()

        cases = [
            (41, 41),
            ("41", 41),
            (40, 41),
            (2, 3),
            (1, 3),
            (0, 3),
            (-5, 3),
            ("abc", 41),
        ]

        for raw_value, expected in cases:
            with self.subTest(raw_value=raw_value):
                self.assertEqual(
                    ocr_utils.normalize_adaptive_block_size(raw_value),
                    expected,
                )
                self.assertIsInstance(
                    ocr_utils.normalize_adaptive_block_size(raw_value),
                    int,
                )

    def test_preprocess_for_ocr_normalizes_even_adaptive_block_size_before_thresholding(self):
        ocr_utils = import_ocr_utils_for_tests()
        image = np.zeros((4, 4, 3), dtype=np.uint8)
        gray = np.zeros((4, 4), dtype=np.uint8)
        thresholded = np.full((4, 4), 255, dtype=np.uint8)
        calls = []

        class FakeCv2:
            COLOR_BGR2GRAY = 1
            ADAPTIVE_THRESH_GAUSSIAN_C = 2
            THRESH_BINARY_INV = 3

            def cvtColor(self, img, code):
                return gray

            def adaptiveThreshold(self, src, max_value, method, threshold_type, block_size, c_value):
                calls.append((block_size, c_value))
                if block_size <= 1 or block_size % 2 == 0:
                    raise AssertionError("invalid block size reached adaptiveThreshold")
                return thresholded

        with patch.object(ocr_utils, "_cv2", return_value=FakeCv2()):
            processed = ocr_utils.preprocess_for_ocr(
                image,
                mode="adaptive",
                block_size=40,
                c_value=-60,
            )

        self.assertEqual(calls, [(41, -60)])
        np.testing.assert_array_equal(processed, thresholded)


class AdaptiveBlockSizeConfigTests(unittest.TestCase):
    def test_load_app_config_normalizes_invalid_adaptive_block_size_to_string(self):
        cases = [
            ("40", "41"),
            ("abc", "41"),
        ]

        for raw_value, expected in cases:
            with self.subTest(raw_value=raw_value):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    original_cwd = os.getcwd()
                    Path(tmp_dir, "ocr_translator_config.ini").write_text(
                        f"[Settings]\nadaptive_block_size = {raw_value}\n",
                        encoding="utf-8",
                    )
                    try:
                        os.chdir(tmp_dir)
                        import config_manager

                        with patch("config_manager.log_debug") as log_debug:
                            loaded = config_manager.load_app_config()

                        persisted = Path("ocr_translator_config.ini").read_text(encoding="utf-8")
                    finally:
                        os.chdir(original_cwd)

                self.assertEqual(loaded["Settings"].get("adaptive_block_size"), expected)
                self.assertIn(f"adaptive_block_size = {expected}", persisted)
                messages = [str(call.args[0]) for call in log_debug.call_args_list if call.args]
                self.assertIn(
                    f"Config: Invalid adaptive_block_size '{raw_value}' normalized to '{expected}'",
                    messages,
                )


class AdaptiveBlockSizeRuntimeTests(unittest.TestCase):
    def test_tesseract_cache_mode_key_normalizes_invalid_gui_block_size(self):
        worker_threads = import_worker_threads_for_tests()

        app = types.SimpleNamespace(
            preprocessing_mode_var=types.SimpleNamespace(get=lambda: "adaptive"),
            adaptive_block_size_var=types.SimpleNamespace(get=lambda: 40),
            adaptive_c_var=types.SimpleNamespace(get=lambda: -60),
            confidence_threshold=60,
            keep_linebreaks_var=types.SimpleNamespace(get=lambda: False),
            remove_trailing_garbage_var=types.SimpleNamespace(get=lambda: False),
        )

        cache_key = worker_threads._get_tesseract_ocr_cache_mode_key(app)

        self.assertIn("|block=41|", cache_key)


if __name__ == "__main__":
    unittest.main()
