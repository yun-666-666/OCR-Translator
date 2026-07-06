import importlib
import importlib.util
import types
import sys
import unittest
import csv
import configparser
import os
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

    def test_example_config_matches_custom_ai_latency_default(self):
        from config_manager import DEFAULT_CONFIG_SETTINGS

        example_config = configparser.ConfigParser()
        example_config.read("ocr_translator_config.example.ini", encoding="utf-8-sig")

        self.assertEqual(
            example_config["Settings"]["custom_ai_latency_mode"],
            DEFAULT_CONFIG_SETTINGS["custom_ai_latency_mode"],
        )

    def test_legacy_removed_ocr_model_config_migrates_to_paddleocr(self):
        import config_manager

        legacy_model = "tes" + "seract"
        with tempfile.TemporaryDirectory() as tmp_dir:
            previous_cwd = os.getcwd()
            os.chdir(tmp_dir)
            try:
                Path("ocr_translator_config.ini").write_text(
                    f"[Settings]\nocr_model = {legacy_model}\n",
                    encoding="utf-8",
                )

                loaded_config = config_manager.load_app_config()
                persisted_config = Path("ocr_translator_config.ini").read_text(encoding="utf-8")
            finally:
                os.chdir(previous_cwd)

        self.assertEqual(loaded_config["Settings"]["ocr_model"], "paddleocr")
        self.assertIn("ocr_model = paddleocr", persisted_config)

    def test_default_config_includes_custom_ai_submit_interval(self):
        from config_manager import DEFAULT_CONFIG_SETTINGS

        self.assertEqual(DEFAULT_CONFIG_SETTINGS["custom_ai_submit_interval_ms"], "300")

    def test_default_config_includes_custom_ai_ocr_image_payload_settings(self):
        from config_manager import DEFAULT_CONFIG_SETTINGS

        self.assertEqual(DEFAULT_CONFIG_SETTINGS["custom_ai_ocr_image_format"], "webp")
        self.assertEqual(DEFAULT_CONFIG_SETTINGS["custom_ai_ocr_image_mode"], "balanced_webp")
        self.assertEqual(DEFAULT_CONFIG_SETTINGS["custom_ai_ocr_image_quality"], "85")
        self.assertEqual(DEFAULT_CONFIG_SETTINGS["custom_ai_ocr_image_detail"], "auto")

    def test_custom_ai_ocr_image_payload_controls_are_wired_to_settings(self):
        gui_builder_source = Path("gui_builder.py").read_text(encoding="utf-8-sig")
        save_source = Path("handlers/ui_interaction_handler.py").read_text(encoding="utf-8-sig")

        self.assertIn("custom_ai_ocr_image_format_var", gui_builder_source)
        self.assertIn("custom_ai_ocr_image_mode_var", gui_builder_source)
        self.assertIn("custom_ai_ocr_image_quality_var", gui_builder_source)
        self.assertIn("custom_ai_ocr_image_detail_var", gui_builder_source)
        self.assertIn("custom_ai_ocr_image_format", save_source)
        self.assertIn("custom_ai_ocr_image_mode", save_source)
        self.assertIn("custom_ai_ocr_image_quality", save_source)
        self.assertIn("custom_ai_ocr_image_detail", save_source)

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

    def test_paddleocr_score_control_is_visible_and_legacy_threshold_controls_are_removed(self):
        gui_builder_source = Path("gui_builder.py").read_text(encoding="utf-8-sig")
        ui_handler_source = Path("handlers/ui_interaction_handler.py").read_text(encoding="utf-8-sig")
        legacy_confidence_key = "confidence" + "_threshold_label"

        self.assertIn("paddleocr_min_score_label", gui_builder_source)
        self.assertIn("paddleocr_min_score_spinbox", gui_builder_source)
        self.assertIn("paddleocr_min_score_var", gui_builder_source)
        self.assertIn("paddleocr_min_score_label, show=is_paddleocr", ui_handler_source)
        self.assertIn("paddleocr_min_score_spinbox, show=is_paddleocr", ui_handler_source)
        self.assertNotIn("confidence_label", ui_handler_source)
        self.assertNotIn(legacy_confidence_key, gui_builder_source)

        for path in ("resources/gui_eng.csv", "resources/gui_zh.csv", "resources/gui_pol.csv"):
            with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
                labels = {
                    row[0]: row[1]
                    for row in csv.reader(f)
                    if len(row) >= 2 and row[0]
                }
            self.assertIn("paddleocr_min_score_label", labels, msg=path)
            self.assertTrue(labels["paddleocr_min_score_label"].strip(), msg=path)
            self.assertNotIn(legacy_confidence_key, labels, msg=path)

    def test_new_custom_ai_controls_have_all_language_labels(self):
        required_keys = {
            "ai_profile_structured_output_label",
            "ai_profile_reasoning_effort_label",
            "custom_ai_reasoning_effort_low",
            "custom_ai_reasoning_effort_medium",
            "custom_ai_reasoning_effort_high",
            "custom_ai_reasoning_effort_ultra",
            "custom_ai_latency_mode_adaptive",
            "custom_ai_submit_interval_label",
            "custom_ai_ocr_image_format_label",
            "custom_ai_ocr_image_format_webp",
            "custom_ai_ocr_image_format_png",
            "custom_ai_ocr_image_format_jpeg",
            "custom_ai_ocr_image_mode_label",
            "custom_ai_ocr_image_mode_lossless",
            "custom_ai_ocr_image_mode_balanced",
            "custom_ai_ocr_image_mode_grayscale",
            "custom_ai_ocr_image_quality_label",
            "custom_ai_ocr_image_detail_label",
            "custom_ai_ocr_image_detail_auto",
            "custom_ai_ocr_image_detail_low",
            "custom_ai_ocr_image_detail_high",
        }

        for path in ("resources/gui_eng.csv", "resources/gui_zh.csv", "resources/gui_pol.csv"):
            with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
                labels = {
                    row[0]: row[1]
                    for row in csv.reader(f)
                    if len(row) >= 2 and row[0]
                }
            missing_keys = required_keys - labels.keys()
            empty_keys = {key for key in required_keys & labels.keys() if not labels[key].strip()}
            self.assertEqual(missing_keys, set(), msg=path)
            self.assertEqual(empty_keys, set(), msg=path)

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

    def test_removed_ocr_backend_tokens_are_absent_from_active_runtime_files(self):
        files_to_check = [
            "app_logic.py",
            "config_manager.py",
            "gui_builder.py",
            "language_manager.py",
            "ocr_utils.py",
            "worker_threads.py",
            "handlers/configuration_handler.py",
            "handlers/display_manager.py",
            "handlers/ui_interaction_handler.py",
            "resources/gui_eng.csv",
            "resources/gui_pol.csv",
            "resources/gui_zh.csv",
            "requirements.txt",
            "setup.py",
            "GameChangingTranslator.spec",
            "GameChangingTranslator_GPU.spec",
        ]
        forbidden_tokens = (
            "tes" + "seract",
            "pytes" + "seract",
            "tess" + "erocr",
            "confidence" + "_threshold",
            "image" + "_preprocessing_mode",
            "adaptive" + "_block_size",
            "adaptive" + "_c",
            "remove" + "_trailing_garbage",
        )

        for relative_path in files_to_check:
            source = Path(relative_path).read_text(encoding="utf-8-sig").lower()
            for token in forbidden_tokens:
                self.assertNotIn(token.lower(), source, msg=f"{relative_path}: {token}")

    def test_settings_save_no_longer_writes_removed_ocr_keys(self):
        save_source = Path("handlers/ui_interaction_handler.py").read_text(encoding="utf-8-sig")
        removed_keys = (
            "tes" + "seract_path",
            "confidence" + "_threshold",
            "image" + "_preprocessing_mode",
            "adaptive" + "_block_size",
            "adaptive" + "_c",
            "remove" + "_trailing_garbage",
        )

        for key in removed_keys:
            self.assertNotIn(key, save_source)


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
