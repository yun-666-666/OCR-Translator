import importlib
import importlib.util
import types
import sys
import unittest
import csv
import tempfile
from pathlib import Path

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
    def test_app_logic_import_does_not_import_removed_provider_sdks(self):
        for optional_module in ("cv2", "pyautogui"):
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
            "Translate naturally and concisely. Preserve meaning, tone, names, and terminology; "
            "use context only when needed."
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
            "Translate naturally and concisely. Preserve meaning, tone, names, and terminology; "
            "use context only when needed."
        )
        shipped_prompt = Path("custom_prompt.txt").read_text(encoding="utf-8-sig").strip()

        self.assertEqual(app_logic.DEFAULT_CUSTOM_PROMPT, expected_prompt)
        self.assertEqual(shipped_prompt, expected_prompt)


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
