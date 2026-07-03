import importlib
import importlib.util
import types
import sys
import unittest
import csv
import tempfile
from pathlib import Path
from unittest.mock import patch

from language_manager import LanguageManager
from language_ui import UILanguageManager


class CustomAILanguageTests(unittest.TestCase):
    def test_custom_ai_uses_generic_google_language_lookup(self):
        manager = LanguageManager()

        self.assertEqual(
            manager.get_code_from_name("English", "custom_ai", "target"),
            "en",
        )
        self.assertEqual(
            manager.get_name_from_code("en", "custom_ai", "target"),
            "English",
        )
        self.assertEqual(
            manager.get_code_from_localized_name("English", "custom_ai", "english"),
            "en",
        )


class StartupOptimizationTests(unittest.TestCase):
    def test_default_config_includes_custom_ai_latency_mode(self):
        from config_manager import DEFAULT_CONFIG_SETTINGS

        self.assertEqual(DEFAULT_CONFIG_SETTINGS["custom_ai_latency_mode"], "safe")

    def test_default_config_includes_custom_ai_submit_interval(self):
        from config_manager import DEFAULT_CONFIG_SETTINGS

        self.assertEqual(DEFAULT_CONFIG_SETTINGS["custom_ai_submit_interval_ms"], "300")

    def test_custom_ai_submit_interval_is_wired_to_settings_and_localizations(self):
        gui_builder_source = Path("gui_builder.py").read_text(encoding="utf-8-sig")
        save_source = Path("handlers/ui_interaction_handler.py").read_text(encoding="utf-8-sig")

        self.assertIn("custom_ai_submit_interval_ms_var", gui_builder_source)
        self.assertIn("custom_ai_submit_interval_label", gui_builder_source)
        self.assertIn("custom_ai_submit_interval_ms", save_source)

        for path in ("resources/gui_eng.csv", "resources/gui_zh.csv"):
            with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
                labels = {
                    row[0]: row[1]
                    for row in csv.reader(f)
                    if len(row) >= 2 and row[0]
                }
            self.assertIn("custom_ai_submit_interval_label", labels, msg=path)
            self.assertTrue(labels["custom_ai_submit_interval_label"].strip(), msg=path)

    def test_app_logic_import_does_not_import_removed_provider_sdks(self):
        for optional_module in ("cv2", "pyautogui", "tesserocr"):
            if importlib.util.find_spec(optional_module) is None:
                module = types.ModuleType(optional_module)
                sys.modules[optional_module] = module

        heavy_modules = {
            "torch",
            "transformers",
            "deepl",
            "google.generativeai",
            "openai",
        }
        for module_name in list(sys.modules):
            if module_name in heavy_modules or any(module_name.startswith(f"{name}.") for name in heavy_modules):
                del sys.modules[module_name]

        importlib.import_module("app_logic")
        importlib.import_module("main")

        loaded = {
            module_name
            for module_name in sys.modules
            if module_name in heavy_modules or any(module_name.startswith(f"{name}.") for name in heavy_modules)
        }
        self.assertEqual(loaded, set())

    def test_startup_does_not_schedule_usage_log_parsing(self):
        app_logic_source = Path("app_logic.py").read_text(encoding="utf-8-sig")

        self.assertNotIn("after_idle(lambda: self._delayed_deepl_usage_update())", app_logic_source)
        self.assertNotIn("after_idle(lambda: self._delayed_api_stats_refresh())", app_logic_source)

    def test_custom_ai_ocr_does_not_force_500ms_scan_interval(self):
        app_logic_source = Path("app_logic.py").read_text(encoding="utf-8-sig")

        self.assertNotIn("too low for API OCR, setting to 500ms minimum", app_logic_source)
        self.assertNotIn("updating scan interval from {current_value}ms to 500ms minimum", app_logic_source)

    def test_removed_about_and_api_usage_code_is_not_wired_at_startup(self):
        app_logic_source = Path("app_logic.py").read_text(encoding="utf-8-sig")
        gui_builder_source = Path("gui_builder.py").read_text(encoding="utf-8-sig")

        self.assertNotIn("create_api_usage_tab", app_logic_source)
        self.assertNotIn("StatisticsHandler", app_logic_source)
        self.assertNotIn("UpdateChecker", app_logic_source)
        self.assertNotIn("def create_about_tab", app_logic_source)
        self.assertNotIn("def create_api_usage_tab", gui_builder_source)

    def test_rapidocr_support_is_removed_from_startup_files(self):
        files_to_check = [
            "app_logic.py",
            "config_manager.py",
            "gui_builder.py",
            "ocr_utils.py",
            "worker_threads.py",
            "handlers/configuration_handler.py",
            "handlers/ui_interaction_handler.py",
            "requirements.txt",
            "resources/gui_eng.csv",
            "resources/gui_pol.csv",
            "resources/gui_zh.csv",
            "ocr_translator_config.ini",
        ]

        for relative_path in files_to_check:
            source = Path(relative_path).read_text(encoding="utf-8-sig")
            lowered = source.lower()
            self.assertNotIn("rapidocr", lowered, msg=relative_path)
            self.assertNotIn("onnxruntime", lowered, msg=relative_path)

    def test_missing_custom_prompt_loads_optimized_default_prompt(self):
        import app_logic

        expected_prompt = (
            "Use context to resolve ambiguity. Translate naturally and concisely while preserving meaning, "
            "tone, and character voice. Keep names and game terms consistent."
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            prompt_path = Path(tmp_dir) / "custom_prompt.txt"
            dummy_app = types.SimpleNamespace(custom_prompt_file=str(prompt_path), custom_prompt_text="")

            app_logic.GameChangingTranslator.load_custom_prompt(dummy_app)

            self.assertTrue(prompt_path.exists())
            self.assertEqual(dummy_app.custom_prompt_text, expected_prompt)
            self.assertLess(len(dummy_app.custom_prompt_text), 180)
            self.assertEqual(prompt_path.read_text(encoding="utf-8-sig"), expected_prompt)

    def test_shipped_custom_prompt_matches_runtime_default(self):
        import app_logic

        expected_prompt = (
            "Use context to resolve ambiguity. Translate naturally and concisely while preserving meaning, "
            "tone, and character voice. Keep names and game terms consistent."
        )
        shipped_prompt = Path("custom_prompt.txt").read_text(encoding="utf-8-sig").strip()

        self.assertEqual(app_logic.DEFAULT_CUSTOM_PROMPT, expected_prompt)
        self.assertEqual(shipped_prompt, expected_prompt)

    def test_tesseract_runtime_cache_is_cleared_when_ocr_parameters_change(self):
        import app_logic

        dummy_app = object.__new__(app_logic.GameChangingTranslator)
        dummy_app.ocr_preview_window = None
        dummy_app.get_ocr_model_setting = lambda: "tesseract"

        with patch.object(app_logic, "clear_tessdata_dir_cache") as clear_tessdata:
            with patch.object(app_logic, "clear_tesseract_languages_cache") as clear_languages:
                with patch.object(app_logic, "clear_tesseract_ocr_engines") as clear_engines:
                    dummy_app.on_ocr_parameter_change()

        clear_tessdata.assert_called_once()
        clear_languages.assert_called_once()
        clear_engines.assert_called_once()

    def test_tesseract_runtime_cache_is_cleared_when_ocr_model_changes(self):
        import app_logic

        dummy_app = object.__new__(app_logic.GameChangingTranslator)
        dummy_app.is_running = False
        dummy_app.is_api_based_ocr_model = lambda *args, **kwargs: False
        dummy_app.ocr_preview_window = None
        dummy_app.ui_interaction_handler = None
        dummy_app.update_adaptive_fields_visibility = lambda: None
        dummy_app.ocr_model_var = types.SimpleNamespace(get=lambda: "tesseract")

        with patch.object(app_logic, "clear_tessdata_dir_cache") as clear_tessdata:
            with patch.object(app_logic, "clear_tesseract_languages_cache") as clear_languages:
                with patch.object(app_logic, "clear_tesseract_ocr_engines") as clear_engines:
                    dummy_app.on_ocr_model_change()

        clear_tessdata.assert_called_once()
        clear_languages.assert_called_once()
        clear_engines.assert_called_once()

    def test_tesseract_runtime_cache_clear_is_wired_into_start_and_stop_paths(self):
        app_logic_source = Path("app_logic.py").read_text(encoding="utf-8-sig")

        self.assertIn('clear_tesseract_runtime_cache("translation starting")', app_logic_source)
        self.assertIn('clear_tesseract_runtime_cache("translation stopped")', app_logic_source)
        self.assertIn('clear_tesseract_languages_cache()', app_logic_source)


class ChineseUILanguageTests(unittest.TestCase):
    def test_chinese_language_is_available(self):
        manager = UILanguageManager()

        self.assertIn("zh", manager.get_available_languages())
        self.assertEqual(manager.get_available_languages()["zh"], "中文")

    def test_chinese_gui_csv_covers_english_keys(self):
        def keys(path):
            with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
                reader = csv.reader(f)
                next(reader, None)
                return {row[0] for row in reader if len(row) >= 2 and row[0]}

        english_keys = keys("resources/gui_eng.csv")
        chinese_keys = keys("resources/gui_zh.csv")

        self.assertEqual(english_keys - chinese_keys, set())


if __name__ == "__main__":
    unittest.main()
