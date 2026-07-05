import importlib
import importlib.util
import base64
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
    if "tesserocr" not in sys.modules:
        module = types.ModuleType("tesserocr")
        module.__spec__ = importlib.machinery.ModuleSpec("tesserocr", loader=None)
        module.PyTessBaseAPI = object
        module.RIL = types.SimpleNamespace(WORD=0)
        module.get_languages = lambda path=None: (path, ["eng"])
        sys.modules["tesserocr"] = module
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
    def test_auto_capture_uses_mss_when_available(self):
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
            backend="auto",
            mss_factory=lambda: fake_mss,
            pyautogui_module=Mock(),
        )

        self.assertEqual(image.mode, "RGB")
        self.assertEqual(image.size, (1, 1))
        self.assertEqual(image.getpixel((0, 0)), (10, 20, 30))
        self.assertEqual(fake_mss.monitor, {"left": 10, "top": 20, "width": 30, "height": 40})

    def test_auto_capture_falls_back_to_pyautogui_when_mss_fails(self):
        capture_screen_region = import_ocr_utils_for_tests().capture_screen_region

        fallback_image = Image.new("RGB", (2, 2), (1, 2, 3))
        pyautogui_module = types.SimpleNamespace(screenshot=Mock(return_value=fallback_image))

        image = capture_screen_region(
            (5, 6, 7, 8),
            backend="auto",
            mss_factory=lambda: (_ for _ in ()).throw(RuntimeError("mss unavailable")),
            pyautogui_module=pyautogui_module,
        )

        self.assertIs(image, fallback_image)
        pyautogui_module.screenshot.assert_called_once_with(region=(5, 6, 7, 8))


class LatencyCaptureBackendSelectorTests(unittest.TestCase):
    def _make_selector(self, capture_func, monotonic_times=None, sample_count=2):
        ocr_utils = import_ocr_utils_for_tests()
        monotonic_values = list(monotonic_times or [100.0, 101.0, 102.0, 103.0])

        def monotonic_clock():
            if monotonic_values:
                return monotonic_values.pop(0)
            return 999.0

        return ocr_utils.CaptureBackendSelector(
            sample_count=sample_count,
            min_recheck_interval_seconds=0.0,
            capture_func=capture_func,
            monotonic_clock=monotonic_clock,
        )

    def test_auto_selector_chooses_fastest_available_backend(self):
        elapsed = {
            "mss": [0.005, 0.006],
            "pyautogui": [0.022, 0.020],
        }
        current_time = [0.0]
        calls = []

        def perf_counter():
            return current_time[0]

        def capture_func(region, backend="auto", **_kwargs):
            calls.append((backend, region))
            current_time[0] += elapsed[backend].pop(0)
            return Image.new("RGB", (2, 2), (1, 2, 3))

        selector = self._make_selector(capture_func)
        selector.perf_counter = perf_counter

        selected = selector.resolve_backend("auto", (10, 20, 300, 120))

        self.assertEqual(selected, "mss")
        self.assertEqual(
            [backend for backend, _region in calls],
            ["mss", "mss", "pyautogui", "pyautogui"],
        )

    def test_auto_selector_excludes_failing_mss_backend(self):
        current_time = [0.0]
        calls = []

        def perf_counter():
            return current_time[0]

        def capture_func(region, backend="auto", **_kwargs):
            calls.append(backend)
            current_time[0] += 0.01
            if backend == "mss":
                raise RuntimeError("mss unavailable")
            return Image.new("RGB", (2, 2), (1, 2, 3))

        selector = self._make_selector(capture_func)
        selector.perf_counter = perf_counter

        selected = selector.resolve_backend("auto", (10, 20, 300, 120))

        self.assertEqual(selected, "pyautogui")
        self.assertIn("mss", calls)
        self.assertIn("pyautogui", calls)

    def test_explicit_pyautogui_is_not_replaced_by_benchmark_result(self):
        calls = []

        def capture_func(region, backend="auto", **_kwargs):
            calls.append((backend, region))
            return Image.new("RGB", (2, 2), (1, 2, 3))

        selector = self._make_selector(capture_func)

        selected = selector.resolve_backend("pyautogui", (10, 20, 300, 120))

        self.assertEqual(selected, "pyautogui")
        self.assertEqual(calls, [])

    def test_auto_selector_uses_cached_benchmark_for_same_context(self):
        current_time = [0.0]
        calls = []

        def perf_counter():
            return current_time[0]

        def capture_func(region, backend="auto", **_kwargs):
            calls.append((backend, region))
            current_time[0] += 0.01 if backend == "mss" else 0.02
            return Image.new("RGB", (2, 2), (1, 2, 3))

        selector = self._make_selector(capture_func)
        selector.perf_counter = perf_counter

        first = selector.resolve_backend("auto", (10, 20, 300, 120))
        second = selector.resolve_backend("auto", (10, 20, 300, 120))

        self.assertEqual(first, "mss")
        self.assertEqual(second, "mss")
        self.assertEqual(len(calls), 4)

    def test_auto_selector_rechecks_when_geometry_changes(self):
        elapsed = {
            ((10, 20, 300, 120), "mss"): [0.005, 0.006],
            ((10, 20, 300, 120), "pyautogui"): [0.025, 0.023],
            ((50, 60, 200, 80), "mss"): [0.040, 0.042],
            ((50, 60, 200, 80), "pyautogui"): [0.010, 0.011],
        }
        current_time = [0.0]
        calls = []

        def perf_counter():
            return current_time[0]

        def capture_func(region, backend="auto", **_kwargs):
            normalized_region = tuple(region)
            calls.append((backend, normalized_region))
            current_time[0] += elapsed[(normalized_region, backend)].pop(0)
            return Image.new("RGB", (2, 2), (1, 2, 3))

        selector = self._make_selector(capture_func)
        selector.perf_counter = perf_counter

        first = selector.resolve_backend("auto", (10, 20, 300, 120))
        second = selector.resolve_backend("auto", (50, 60, 200, 80))

        self.assertEqual(first, "mss")
        self.assertEqual(second, "pyautogui")
        self.assertEqual(len(calls), 8)


class LatencyCaptureThreadBackendSelectionTests(unittest.TestCase):
    def test_unknown_capture_backend_config_uses_auto_selector(self):
        worker_threads = import_worker_threads_for_tests()

        class FakeSelector:
            def __init__(self):
                self.calls = []

            def resolve_backend(self, configured_backend, region):
                self.calls.append((configured_backend, region))
                return "mss"

        app = types.SimpleNamespace(capture_backend_selector=FakeSelector())

        selected = worker_threads._resolve_capture_backend(app, "legacy_slow", (1, 2, 3, 4))

        self.assertEqual(selected, "mss")
        self.assertEqual(app.capture_backend_selector.calls, [("auto", (1, 2, 3, 4))])

    def test_capture_thread_uses_resolved_auto_backend_and_continues_after_capture_error(self):
        worker_threads = import_worker_threads_for_tests()
        screenshot = Image.new("RGB", (8, 8), (1, 2, 3))
        capture_backends = []

        class FakeOverlay:
            def winfo_exists(self):
                return True

            def get_geometry(self):
                return (10, 20, 18, 28)

        class FakeSelector:
            def __init__(self):
                self.calls = []

            def resolve_backend(self, configured_backend, region):
                self.calls.append((configured_backend, region))
                return "mss"

        app = types.SimpleNamespace(
            is_running=True,
            current_scan_interval=50,
            scan_interval_var=types.SimpleNamespace(get=lambda: 50),
            update_adaptive_scan_interval=lambda: None,
            get_ocr_model_setting=lambda: "custom_ai",
            is_api_based_ocr_model=lambda model=None: True,
            source_overlay=FakeOverlay(),
            capture_backend_var=types.SimpleNamespace(get=lambda: "auto"),
            capture_backend_selector=FakeSelector(),
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

        def capture_func(region, backend="auto"):
            capture_backends.append(backend)
            if len(capture_backends) == 1:
                raise RuntimeError("transient capture failure")
            return screenshot

        with (
            patch.object(worker_threads.tk, "Toplevel", FakeOverlay),
            patch.object(worker_threads, "capture_screen_region", side_effect=capture_func),
            patch.object(worker_threads.time, "sleep", return_value=None),
        ):
            worker_threads.run_capture_thread(app)

        self.assertEqual(capture_backends, ["mss", "mss"])
        self.assertEqual(app.capture_backend_selector.calls[0], ("auto", (10, 20, 8, 8)))
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
            ocr_model="tesseract",
            source_lang="eng",
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
            ocr_model="tesseract",
            source_lang="eng",
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
                ocr_model="tesseract",
                source_lang="eng",
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
                ocr_model="tesseract",
                source_lang="eng",
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
            ocr_model="tesseract",
            source_lang="eng",
            preprocessing_mode="none",
            region_size=(300, 80),
            region_origin=(10, 20),
        )
        second = ocr_utils.build_ocr_frame_cache_key(
            image_hash="same",
            ocr_model="tesseract",
            source_lang="eng",
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


class LatencyTesserocrTests(unittest.TestCase):
    def test_tesseract_language_lookup_is_cached_until_cleared(self):
        ocr_utils = import_ocr_utils_for_tests()

        class FakeTesserocr:
            calls = 0

            @staticmethod
            def get_languages(path=None):
                FakeTesserocr.calls += 1
                return path, ["eng", "chi_sim"]

        with patch.object(ocr_utils, "_tesserocr", return_value=FakeTesserocr):
            ocr_utils.clear_tesseract_languages_cache()
            first = ocr_utils.get_tesseract_languages(r"C:\Program Files\Tesseract-OCR\tessdata")
            second = ocr_utils.get_tesseract_languages(r"C:\Program Files\Tesseract-OCR\tessdata")

            self.assertEqual(first[1], ("chi_sim", "eng"))
            self.assertEqual(second[1], ("chi_sim", "eng"))
            self.assertEqual(FakeTesserocr.calls, 1)

            ocr_utils.clear_tesseract_languages_cache()
            third = ocr_utils.get_tesseract_languages(r"C:\Program Files\Tesseract-OCR\tessdata")

            self.assertEqual(third[1], ("chi_sim", "eng"))
            self.assertEqual(FakeTesserocr.calls, 2)

    def test_tesseract_cache_mode_key_changes_when_ocr_settings_change(self):
        worker_threads = import_worker_threads_for_tests()

        app = types.SimpleNamespace(
            preprocessing_mode_var=types.SimpleNamespace(get=lambda: "none"),
            adaptive_block_size_var=types.SimpleNamespace(get=lambda: 41),
            adaptive_c_var=types.SimpleNamespace(get=lambda: -60),
            confidence_threshold=60,
            keep_linebreaks_var=types.SimpleNamespace(get=lambda: False),
            remove_trailing_garbage_var=types.SimpleNamespace(get=lambda: False),
        )

        cache_key_builder = getattr(worker_threads, "_get_tesseract_ocr_cache_mode_key", None)
        self.assertIsNotNone(cache_key_builder)

        base_key = cache_key_builder(app)

        app.keep_linebreaks_var = types.SimpleNamespace(get=lambda: True)
        keep_linebreaks_key = cache_key_builder(app)

        app.keep_linebreaks_var = types.SimpleNamespace(get=lambda: False)
        app.confidence_threshold = 75
        confidence_key = cache_key_builder(app)

        app.confidence_threshold = 60
        app.adaptive_block_size_var = types.SimpleNamespace(get=lambda: 51)
        block_key = cache_key_builder(app)

        self.assertNotEqual(base_key, keep_linebreaks_key)
        self.assertNotEqual(base_key, confidence_key)
        self.assertNotEqual(base_key, block_key)

    def test_ocr_region_uses_injected_reusable_tesserocr_engine(self):
        ocr_utils = import_ocr_utils_for_tests()
        img = np.zeros((320, 320), dtype=np.uint8)

        class FakeEngine:
            def __init__(self):
                self.calls = []

            def recognize(self, pil_image, lang_code, ocr_config, confidence_threshold):
                self.calls.append((pil_image.size, lang_code, ocr_config, confidence_threshold))
                return "Reusable OCR text"

        engine = FakeEngine()
        config = {"psm": 6, "oem": 3, "variables": {}}

        result = ocr_utils.ocr_region_with_confidence(
            img,
            (0, 0, 320, 320),
            "eng",
            config,
            55,
            ocr_engine=engine,
        )

        self.assertEqual(result, "Reusable OCR text")
        self.assertEqual(len(engine.calls), 1)
        self.assertEqual(engine.calls[0][1:], ("eng", config, 55))

    def test_tesserocr_engine_fails_fast_when_requested_language_is_missing(self):
        ocr_utils = import_ocr_utils_for_tests()
        img = Image.new("L", (320, 320), 255)
        created_apis = []

        class FakeApi:
            def __init__(self, **kwargs):
                created_apis.append(kwargs)

        fake_tesserocr = types.SimpleNamespace(
            PyTessBaseAPI=FakeApi,
            RIL=types.SimpleNamespace(WORD=1),
            get_languages=lambda path=None: (path, ["eng"]),
        )

        engine = ocr_utils.TesseractOcrEngine(
            tessdata_dir=r"C:\Program Files\Tesseract-OCR\tessdata",
            tesserocr_module=fake_tesserocr,
        )
        config = {"psm": 6, "oem": 3, "variables": {}}

        with self.assertRaises(ocr_utils.TesseractOcrUnavailableError):
            engine.recognize(img, "fra", config, 50)

        self.assertEqual(created_apis, [])

    def test_tesseract_path_can_resolve_neighbor_tessdata_directory(self):
        ocr_utils = import_ocr_utils_for_tests()

        resolved = ocr_utils.resolve_tessdata_dir_from_tesseract_path(
            r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        )

        self.assertEqual(resolved, r"C:\Program Files\Tesseract-OCR\tessdata")

    def test_tesseract_path_uses_tessdata_prefix_when_path_setting_is_missing(self):
        ocr_utils = import_ocr_utils_for_tests()

        with tempfile.TemporaryDirectory() as temp_dir:
            tessdata_dir = Path(temp_dir) / "tessdata"
            tessdata_dir.mkdir()

            with patch.dict(os.environ, {"TESSDATA_PREFIX": temp_dir}, clear=False):
                resolved = ocr_utils.resolve_tessdata_dir_from_tesseract_path("")

        self.assertEqual(resolved, str(tessdata_dir))

    def test_tesseract_path_can_resolve_default_windows_install_when_setting_is_missing(self):
        ocr_utils = import_ocr_utils_for_tests()
        default_tessdata = Path(r"C:\Program Files\Tesseract-OCR\tessdata")
        if not default_tessdata.is_dir():
            self.skipTest("Default Windows Tesseract tessdata directory is not installed")

        resolved = ocr_utils.resolve_tessdata_dir_from_tesseract_path(None)

        self.assertEqual(resolved, str(default_tessdata))

    def test_tesseract_path_resolution_uses_windows_registry_install_dir(self):
        ocr_utils = import_ocr_utils_for_tests()

        with tempfile.TemporaryDirectory() as temp_dir:
            install_dir = Path(temp_dir) / "Tesseract-OCR"
            tessdata_dir = install_dir / "tessdata"
            tessdata_dir.mkdir(parents=True)

            ocr_utils.clear_tessdata_dir_cache()
            with patch.object(
                ocr_utils,
                "_get_windows_tesseract_install_dirs",
                return_value=(str(install_dir),),
                create=True,
            ):
                resolved = ocr_utils.resolve_tessdata_dir_from_tesseract_path("")

        self.assertEqual(resolved, str(tessdata_dir))

    def test_tesseract_path_resolution_uses_pyinstaller_meipass_tessdata(self):
        ocr_utils = import_ocr_utils_for_tests()

        with tempfile.TemporaryDirectory() as temp_dir:
            tessdata_dir = Path(temp_dir) / "tessdata"
            tessdata_dir.mkdir()

            ocr_utils.clear_tessdata_dir_cache()
            fake_sys = types.SimpleNamespace(frozen=True, _MEIPASS=temp_dir)
            with patch.object(ocr_utils, "sys", fake_sys, create=True):
                resolved = ocr_utils.resolve_tessdata_dir_from_tesseract_path("")

        self.assertEqual(resolved, str(tessdata_dir))

    def test_tesseract_path_resolution_uses_tesseract_executable_from_path(self):
        ocr_utils = import_ocr_utils_for_tests()

        with tempfile.TemporaryDirectory() as temp_dir:
            install_dir = Path(temp_dir) / "CustomTesseract"
            tessdata_dir = install_dir / "tessdata"
            tessdata_dir.mkdir(parents=True)
            tesseract_exe = install_dir / "tesseract.exe"
            tesseract_exe.write_text("", encoding="utf-8")

            ocr_utils.clear_tessdata_dir_cache()
            fake_shutil = types.SimpleNamespace(which=lambda _name: str(tesseract_exe))
            with patch.object(ocr_utils, "shutil", fake_shutil, create=True):
                resolved = ocr_utils.resolve_tessdata_dir_from_tesseract_path("")

        self.assertEqual(resolved, str(tessdata_dir))

    def test_tesseract_path_resolution_uses_tesserocr_package_adjacent_tessdata(self):
        ocr_utils = import_ocr_utils_for_tests()

        with tempfile.TemporaryDirectory() as temp_dir:
            package_dir = Path(temp_dir) / "site-packages" / "tesserocr"
            tessdata_dir = package_dir / "tessdata"
            tessdata_dir.mkdir(parents=True)

            ocr_utils.clear_tessdata_dir_cache()
            with patch.object(ocr_utils, "_get_tesserocr_package_root", return_value=str(package_dir)):
                resolved = ocr_utils.resolve_tessdata_dir_from_tesseract_path("")

        self.assertEqual(resolved, str(tessdata_dir))

    def test_tesseract_path_resolution_is_cached_until_cleared(self):
        ocr_utils = import_ocr_utils_for_tests()

        with tempfile.TemporaryDirectory() as temp_dir:
            install_dir = Path(temp_dir) / "Tesseract-OCR"
            tessdata_dir = install_dir / "tessdata"
            tessdata_dir.mkdir(parents=True)
            tesseract_exe = install_dir / "tesseract.exe"

            ocr_utils.clear_tessdata_dir_cache()
            with patch.object(ocr_utils.os.path, "isdir", wraps=ocr_utils.os.path.isdir) as isdir_mock:
                first = ocr_utils.resolve_tessdata_dir_from_tesseract_path(str(tesseract_exe))
                second = ocr_utils.resolve_tessdata_dir_from_tesseract_path(str(tesseract_exe))

            self.assertEqual(first, str(tessdata_dir))
            self.assertEqual(second, str(tessdata_dir))
            self.assertEqual(isdir_mock.call_count, 2)

            ocr_utils.clear_tessdata_dir_cache()
            with patch.object(ocr_utils.os.path, "isdir", wraps=ocr_utils.os.path.isdir) as isdir_after_clear:
                refreshed = ocr_utils.resolve_tessdata_dir_from_tesseract_path(str(tesseract_exe))

            self.assertEqual(refreshed, str(tessdata_dir))
            self.assertEqual(isdir_after_clear.call_count, 2)

    def test_tesseract_path_resolution_cache_key_includes_tessdata_prefix(self):
        ocr_utils = import_ocr_utils_for_tests()

        with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
            first_tessdata = Path(first_dir) / "tessdata"
            second_tessdata = Path(second_dir) / "tessdata"
            first_tessdata.mkdir()
            second_tessdata.mkdir()

            ocr_utils.clear_tessdata_dir_cache()
            with patch.dict(os.environ, {"TESSDATA_PREFIX": first_dir}, clear=False):
                first = ocr_utils.resolve_tessdata_dir_from_tesseract_path("")
            with patch.dict(os.environ, {"TESSDATA_PREFIX": second_dir}, clear=False):
                second = ocr_utils.resolve_tessdata_dir_from_tesseract_path("")

        self.assertEqual(first, str(first_tessdata))
        self.assertEqual(second, str(second_tessdata))

    def test_tesseract_startup_does_not_require_tesseract_executable_path(self):
        source = Path("app_logic.py").read_text(encoding="utf-8-sig").lower()

        self.assertNotIn("tesseract path invalid", source)
        self.assertNotIn("os.path.isfile(tesseract_exe_path)", source)

    def test_tesserocr_engine_reuses_api_and_filters_low_confidence_words(self):
        ocr_utils = import_ocr_utils_for_tests()

        class FakeWord:
            def __init__(self, text, confidence):
                self.text = text
                self.confidence = confidence

            def GetUTF8Text(self, level):
                return self.text

            def Confidence(self, level):
                return self.confidence

        class FakeApi:
            instances = []

            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.variables = {}
                self.images = []
                FakeApi.instances.append(self)

            def SetVariable(self, name, value):
                self.variables[name] = value

            def SetImage(self, image):
                self.images.append(image)

            def Recognize(self):
                return True

            def GetIterator(self):
                return object()

        fake_tesserocr = types.SimpleNamespace(
            PyTessBaseAPI=FakeApi,
            RIL=types.SimpleNamespace(WORD=1),
            get_languages=lambda path=None: (path, ["eng"]),
            iterate_level=lambda iterator, level: [
                FakeWord("keep", 92),
                FakeWord("drop", 12),
                FakeWord("also", 88),
            ],
        )

        engine = ocr_utils.TesseractOcrEngine(
            tessdata_dir=r"C:\Program Files\Tesseract-OCR\tessdata",
            tesserocr_module=fake_tesserocr,
        )
        config = {
            "psm": 6,
            "oem": 3,
            "variables": {"preserve_interword_spaces": "1"},
        }
        image = Image.new("L", (320, 320), 255)

        first = engine.recognize(image, "eng", config, 50)
        second = engine.recognize(image, "eng", config, 50)

        self.assertEqual(first, "keep also")
        self.assertEqual(second, "keep also")
        self.assertEqual(len(FakeApi.instances), 1)
        self.assertEqual(FakeApi.instances[0].kwargs["lang"], "eng")
        self.assertEqual(FakeApi.instances[0].kwargs["path"], r"C:\Program Files\Tesseract-OCR\tessdata")
        self.assertEqual(FakeApi.instances[0].variables, {"preserve_interword_spaces": "1"})


class LatencyOcrRuntimeRefreshTests(unittest.TestCase):
    def test_tesseract_runtime_settings_refresh_after_interval_in_fast_loop(self):
        worker_threads = import_worker_threads_for_tests()

        class EmptyQueue:
            maxsize = 1

            def qsize(self):
                return 0

            def get(self, timeout=None):
                raise worker_threads.queue.Empty

        lang_calls = []

        def get_tesseract_lang_code():
            lang_calls.append("called")
            return "eng"

        app = types.SimpleNamespace(
            is_running=True,
            get_ocr_model_setting=lambda: "tesseract",
            get_tesseract_lang_code=get_tesseract_lang_code,
            tesseract_path_var=types.SimpleNamespace(get=lambda: ""),
            confidence_threshold=50,
            confidence_var=types.SimpleNamespace(get=lambda: 50),
            is_api_based_ocr_model=lambda model: False,
            ocr_queue=EmptyQueue(),
        )

        sleep_calls = []

        def fake_sleep(_duration):
            sleep_calls.append(_duration)
            if len(sleep_calls) >= 3:
                app.is_running = False

        with patch.object(worker_threads.time, "monotonic", side_effect=[0.0, 1.0, 2.0, 6.1]):
            with patch.object(worker_threads.time, "sleep", side_effect=fake_sleep):
                with patch.object(
                    worker_threads,
                    "resolve_tessdata_dir_from_tesseract_path",
                    return_value=r"C:\Program Files\Tesseract-OCR\tessdata",
                ) as resolve_tessdata:
                    worker_threads.run_ocr_thread(app)

        self.assertEqual(len(lang_calls), 2)
        self.assertEqual(resolve_tessdata.call_count, 2)


class LatencyShutdownTests(unittest.TestCase):
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
            custom_ai_ocr_image_mode_var=types.SimpleNamespace(get=lambda: "balanced_webp"),
            custom_ai_ocr_image_quality_var=types.SimpleNamespace(get=lambda: 85),
            custom_ai_ocr_image_detail_var=types.SimpleNamespace(get=lambda: "low"),
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
            custom_ai_ocr_image_mode_var=types.SimpleNamespace(get=lambda: "small_grayscale_webp"),
            custom_ai_ocr_image_quality_var=types.SimpleNamespace(get=lambda: 80),
            custom_ai_ocr_image_detail_var=types.SimpleNamespace(get=lambda: "high"),
        )

        cache_mode_key = worker_threads._get_api_ocr_cache_mode_key(app)

        self.assertIn("keep_linebreaks=True", cache_mode_key)
        self.assertIn("image_mode=small_grayscale_webp", cache_mode_key)
        self.assertIn("image_quality=80", cache_mode_key)
        self.assertIn("image_detail=high", cache_mode_key)

    def test_convert_to_webp_for_api_logs_payload_metadata_without_base64(self):
        import app_logic

        log_messages = []
        image = Image.new("RGB", (48, 24), (250, 250, 250))
        app = types.SimpleNamespace(
            custom_ai_ocr_image_mode_var=types.SimpleNamespace(get=lambda: "small_grayscale_webp"),
            custom_ai_ocr_image_quality_var=types.SimpleNamespace(get=lambda: 80),
            custom_ai_ocr_image_detail_var=types.SimpleNamespace(get=lambda: "low"),
        )

        with patch.object(app_logic, "log_debug", side_effect=log_messages.append):
            encoded = app_logic.GameChangingTranslator.convert_to_webp_for_api(app, image)

        encoded_base64 = base64.b64encode(encoded).decode("ascii")
        log_text = "\n".join(log_messages)
        self.assertIn("mode=small_grayscale_webp", log_text)
        self.assertIn("bytes=", log_text)
        self.assertIn("detail=low", log_text)
        self.assertIn("duration=", log_text)
        self.assertNotIn(encoded_base64, log_text)
        self.assertNotIn("data:image", log_text)

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
            "api|keep_linebreaks=False",
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
        app.clear_tesseract_runtime_cache = Mock()
        app.is_running = False
        app.is_api_based_ocr_model = lambda: False
        app.ocr_preview_window = None
        app.ocr_model_var = types.SimpleNamespace(get=lambda: "custom_ai")
        app.ocr_stability_gate = types.SimpleNamespace(clear=Mock(return_value=True))

        app_logic.GameChangingTranslator.on_ocr_model_change(app)

        app.ocr_stability_gate.clear.assert_called()


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

    def test_local_tesseract_path_uses_tesserocr_not_pytesseract(self):
        files_to_check = [
            "app_logic.py",
            "worker_threads.py",
            "ocr_utils.py",
            "handlers/configuration_handler.py",
            "handlers/ui_interaction_handler.py",
            "requirements.txt",
            "setup.py",
        ]

        for relative_path in files_to_check:
            source = Path(relative_path).read_text(encoding="utf-8-sig").lower()
            self.assertNotIn("pytesseract", source, msg=relative_path)

        self.assertIn("tesserocr", Path("requirements.txt").read_text(encoding="utf-8-sig").lower())
        self.assertIn("tesserocr", Path("setup.py").read_text(encoding="utf-8-sig").lower())


class AdaptiveScanLoggingTests(unittest.TestCase):
    @staticmethod
    def _make_app(active_count=0):
        import app_logic

        app = object.__new__(app_logic.GameChangingTranslator)
        app.load_check_timer = 0.0
        app.active_ocr_calls = set(range(active_count))
        app.max_concurrent_ocr_calls = 8
        app.current_scan_interval = 200
        app.base_scan_interval = 200
        app.overload_detected = False
        app.scan_interval_var = types.SimpleNamespace(get=lambda: 200)
        app._last_adaptive_log_state = None
        app._last_adaptive_log_time = 0.0
        return app

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


if __name__ == "__main__":
    unittest.main()
