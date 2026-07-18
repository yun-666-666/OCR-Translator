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

    def test_language_manager_exposes_only_live_custom_ai_lists(self):
        manager = LanguageManager()

        self.assertTrue(hasattr(manager, "get_language_lists"))
        self.assertIn(("English", "en"), manager.get_language_lists("custom_ai", "source"))
        self.assertIn(("English", "en"), manager.get_language_lists("custom_ai", "target"))
        self.assertEqual(manager.get_language_lists("deepl_api", "source"), [])
        self.assertFalse(hasattr(manager, "google_source_languages"))
        self.assertFalse(hasattr(manager, "deepl_source_languages"))
        self.assertFalse(hasattr(manager, "gemini_source_languages"))
        self.assertFalse(hasattr(manager, "openai_source_languages"))


class StartupOptimizationTests(unittest.TestCase):
    def test_default_config_uses_unified_ai_optimization_mode(self):
        from config_manager import DEFAULT_CONFIG_SETTINGS

        self.assertEqual(DEFAULT_CONFIG_SETTINGS["ai_optimization_mode"], "auto")
        self.assertNotIn("custom_ai_latency_mode", DEFAULT_CONFIG_SETTINGS)

    def test_example_config_matches_ai_optimization_default(self):
        from config_manager import DEFAULT_CONFIG_SETTINGS

        example_config = configparser.ConfigParser()
        example_config.read("ocr_translator_config.example.ini", encoding="utf-8-sig")

        self.assertEqual(
            example_config["Settings"]["ai_optimization_mode"],
            DEFAULT_CONFIG_SETTINGS["ai_optimization_mode"],
        )

    def test_legacy_removed_ocr_model_config_migrates_to_paddleocr(self):
        import config_manager

        legacy_model = "tes" + "seract"
        with tempfile.TemporaryDirectory() as tmp_dir:
            previous = os.environ.get(config_manager.CONFIG_DIR_ENV)
            os.environ[config_manager.CONFIG_DIR_ENV] = tmp_dir
            try:
                config_path = Path(tmp_dir) / "ocr_translator_config.ini"
                config_path.write_text(
                    f"[Settings]\nocr_model = {legacy_model}\n",
                    encoding="utf-8",
                )

                loaded_config = config_manager.load_app_config()
                persisted_config = config_path.read_text(encoding="utf-8")
            finally:
                if previous is None:
                    os.environ.pop(config_manager.CONFIG_DIR_ENV, None)
                else:
                    os.environ[config_manager.CONFIG_DIR_ENV] = previous

        self.assertEqual(loaded_config["Settings"]["ocr_model"], "paddleocr")
        self.assertIn("ocr_model = paddleocr", persisted_config)

    def test_default_config_includes_custom_ai_submit_interval(self):
        from config_manager import DEFAULT_CONFIG_SETTINGS

        self.assertEqual(DEFAULT_CONFIG_SETTINGS["custom_ai_submit_interval_ms"], "300")

    def test_log_tuned_ocr_defaults_are_shipped(self):
        from config_manager import DEFAULT_CONFIG_SETTINGS

        self.assertEqual(DEFAULT_CONFIG_SETTINGS["stability_threshold"], "0")
        self.assertEqual(DEFAULT_CONFIG_SETTINGS["paddleocr_min_score"], "0.45")

    def test_default_config_removes_manual_ai_image_payload_settings(self):
        from config_manager import DEFAULT_CONFIG_SETTINGS

        for key in (
            "custom_ai_ocr_image_format",
            "custom_ai_ocr_image_mode",
            "custom_ai_ocr_image_quality",
            "custom_ai_ocr_image_detail",
        ):
            self.assertNotIn(key, DEFAULT_CONFIG_SETTINGS)

    def test_ai_optimization_control_replaces_legacy_response_and_image_controls(self):
        gui_builder_source = Path("gui_settings_builder.py").read_text(encoding="utf-8-sig")
        save_source = Path("handlers/ui_interaction_handler.py").read_text(encoding="utf-8-sig")

        self.assertIn("ai_optimization_mode_var", gui_builder_source)
        self.assertIn("ai_optimization_mode_combobox", gui_builder_source)
        self.assertIn("ai_optimization_mode", save_source)
        self.assertIn(
            "show_ai_optimization = is_custom or is_custom_ai_ocr",
            save_source,
        )
        self.assertIn(
            "show_ai_optimization = is_custom_translation or is_custom_ai_ocr",
            save_source,
        )
        for legacy_key in (
            "custom_ai_latency_mode_combobox",
            "custom_ai_ocr_image_format_var",
            "custom_ai_ocr_image_mode_var",
            "custom_ai_ocr_image_quality_var",
            "custom_ai_ocr_image_detail_var",
        ):
            self.assertNotIn(legacy_key, gui_builder_source)
        for legacy_key in (
            "cfg['custom_ai_latency_mode']",
            "cfg['custom_ai_ocr_image_format']",
            "cfg['custom_ai_ocr_image_mode']",
            "cfg['custom_ai_ocr_image_quality']",
            "cfg['custom_ai_ocr_image_detail']",
        ):
            self.assertNotIn(legacy_key, save_source)

    def test_hidden_legacy_provider_controls_and_settings_are_removed(self):
        sources = {
            path: Path(path).read_text(encoding="utf-8-sig")
            for path in (
                "app_logic.py",
                "app_lifecycle.py",
                "gui_settings_builder.py",
                "handlers/ui_interaction_handler.py",
                "app_configuration.py",
                "handlers/configuration_handler.py",
                "config_manager.py",
                "ocr_translator_config.example.ini",
                "resources/gui_eng.csv",
                "resources/gui_pol.csv",
                "resources/gui_zh.csv",
            )
        }
        forbidden_by_path = {
            "gui_settings_builder.py": (
                "google_api_key_label",
                "deepl_api_key_label",
                "gemini_api_key_label",
                "openai_api_key_label",
                "marian_model_combobox",
                "beam_spinbox",
                "marian_explanation_labels",
            ),
            "handlers/ui_interaction_handler.py": (
                "toggle_api_key_visibility",
                "google_api_key_label",
                "deepl_api_key_label",
                "gemini_api_key_label",
                "openai_api_key_label",
                "marian_model_combobox",
                "num_beams_var",
                "marian_explanation_labels",
                "cfg['google_translate_api_key']",
                "cfg['deepl_api_key']",
                "cfg['gemini_api_key']",
                "cfg['openai_api_key']",
                "cfg['marian_model']",
            ),
            "app_logic.py": (
                "CacheManager",
                "gemini_models_manager",
                "openai_models_manager",
                "google_api_key_var",
                "deepl_api_key_var",
                "gemini_api_key_var",
                "openai_api_key_var",
                "marian_model_var",
                "num_beams_var",
            ),
            "app_lifecycle.py": (
                "cache_manager",
                "google_file_cache",
                "deepl_file_cache",
                "notify_cache_cleared",
            ),
            "app_configuration.py": (
                "toggle_api_key_visibility",
                "browse_marian_models_file",
                "on_marian_model_selection_changed",
                "reset_gemini_api_log",
                "update_openai_stats",
                "_get_cumulative_openai_totals",
                "reset_openai_api_log",
                "get_current_gemini_model_for_translation",
                "get_current_openai_model_for_translation",
                "update_deepl_model_type_for_language",
                "gemini_client",
            ),
            "handlers/configuration_handler.py": (
                "load_marian_models",
                "browse_marian_models_file",
            ),
            "ocr_translator_config.example.ini": (
                "translation_model = marianmt",
                "google_translate_api_key =",
                "deepl_api_key =",
                "gemini_api_key =",
                "openai_api_key =",
                "marian_model =",
            ),
        }

        for path, forbidden_tokens in forbidden_by_path.items():
            for token in forbidden_tokens:
                self.assertNotIn(token, sources[path], msg=f"{path}: {token}")

        self.assertFalse(Path("handlers/cache_manager.py").exists())

        removed_label_keys = (
            "translation_model_marianmt_offline",
            "ocr_model_gemini",
            "marian_model_label",
            "google_api_key_label",
            "deepl_api_key_label",
            "gemini_api_key_label",
            "openai_api_key_label",
            "deepl_model_type_label",
            "gemini_context_window_label",
            "openai_context_window_label",
            "deepl_context_window_label",
            "models_file_label",
            "beam_size_label",
            "google_cache_checkbox",
            "deepl_cache_checkbox",
            "gemini_file_cache_checkbox",
            "openai_file_cache_checkbox",
            "marian_beam_explanation",
            "browse_marian_models_title",
            "gemini_total_words_label",
            "gemini_enable_api_log_checkbox",
            "deepl_usage_label",
            "gemini_reset_success_title",
        )
        for path in (
            "resources/gui_eng.csv",
            "resources/gui_pol.csv",
            "resources/gui_zh.csv",
        ):
            for key in removed_label_keys:
                self.assertNotIn(f"{key},", sources[path], msg=f"{path}: {key}")

        self.assertIn("custom_ai_profiles.list_profiles", sources["gui_settings_builder.py"])
        self.assertIn("build_ocr_model_display_options", sources["gui_settings_builder.py"])
        self.assertIn("def clear_file_caches(self):\n        return self.clear_cache()", sources["app_lifecycle.py"])
        self.assertIn("'translation_model': 'custom_ai'", sources["config_manager.py"])
        self.assertIn("'ocr_model': 'paddleocr'", sources["config_manager.py"])

        from config_manager import DEFAULT_CONFIG_SETTINGS

        for key in (
            "google_translate_api_key",
            "deepl_api_key",
            "gemini_api_key",
            "openai_api_key",
            "marian_model",
            "num_beams",
        ):
            self.assertNotIn(key, DEFAULT_CONFIG_SETTINGS)

    def test_live_runtime_state_survives_legacy_ui_cleanup(self):
        app_logic_source = Path("app_logic.py").read_text(encoding="utf-8-sig")
        lifecycle_source = Path("app_lifecycle.py").read_text(encoding="utf-8-sig")
        results_source = Path("handlers/translation_results.py").read_text(
            encoding="utf-8-sig"
        )

        self.assertIn(
            'self.custom_prompt_file = os.path.join(self.base_dir, "custom_prompt.txt")',
            app_logic_source,
        )
        self.assertIn("self.load_custom_prompt()", app_logic_source)
        self.assertIn(
            "self.ocr_frame_cache = OCRFrameCache(self.ocr_frame_cache_size_var.get())",
            app_logic_source,
        )
        self.assertIn(
            "set_debug_logging_enabled(self.debug_logging_enabled_var.get())",
            app_logic_source,
        )
        self.assertNotIn("self.translation_cache.clear()", lifecycle_source)
        self.assertNotIn("self.app.cache_manager", results_source)

    def test_capture_backend_setting_and_pyautogui_dependency_are_removed(self):
        gui_builder_source = Path("gui_settings_builder.py").read_text(encoding="utf-8-sig")
        save_source = Path("handlers/ui_interaction_handler.py").read_text(encoding="utf-8-sig")

        self.assertNotIn("capture_backend_combobox", gui_builder_source)
        self.assertNotIn("cfg['capture_backend']", save_source)
        for path in (
            "requirements.txt",
            "setup.py",
            "GameChangingTranslator.spec",
            "GameChangingTranslator_GPU.spec",
            "LICENSE",
        ):
            self.assertNotIn(
                "pyautogui",
                Path(path).read_text(encoding="utf-8-sig").lower(),
                msg=path,
            )

    def test_custom_ai_submit_interval_is_wired_to_settings_and_localizations(self):
        gui_builder_source = Path("gui_settings_builder.py").read_text(encoding="utf-8-sig")
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
        gui_builder_source = Path("gui_settings_builder.py").read_text(encoding="utf-8-sig")
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
            "custom_ai_reasoning_effort_none",
            "custom_ai_reasoning_effort_low",
            "custom_ai_reasoning_effort_medium",
            "custom_ai_reasoning_effort_high",
            "custom_ai_reasoning_effort_ultra",
            "ai_optimization_mode_label",
            "ai_optimization_mode_auto",
            "ai_optimization_mode_stream",
            "ai_optimization_mode_speed",
            "ai_optimization_mode_quality",
            "custom_ai_submit_interval_label",
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
