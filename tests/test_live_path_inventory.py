import ast
import importlib
import unittest
from pathlib import Path
from unittest import mock

import compile_app


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class LivePathInventoryTests(unittest.TestCase):
    def test_unwired_rust_overlay_experiment_is_removed(self):
        self.assertFalse((PROJECT_ROOT / "rust_overlay_manager.py").exists())

    def test_legacy_provider_source_inventory_is_removed(self):
        legacy_paths = (
            "marian_mt_translator.py",
            "convert_marian.py",
            "handlers/gemini_provider.py",
            "handlers/gemini_ocr_provider.py",
            "handlers/gemini_models_manager.py",
            "handlers/openai_provider.py",
            "handlers/openai_ocr_provider.py",
            "handlers/openai_models_manager.py",
            "handlers/ocr_provider_base.py",
            "handlers/llm_provider_base.py",
            "handlers/cache_manager.py",
            "handlers/statistics_handler.py",
        )
        remaining = [path for path in legacy_paths if (PROJECT_ROOT / path).exists()]
        self.assertEqual([], remaining)

    def test_provider_only_language_resources_are_removed(self):
        legacy_resources = (
            "resources/deepl_trans_source.csv",
            "resources/deepl_trans_target.csv",
            "resources/gemini_models.csv",
            "resources/gemini_trans_source.csv",
            "resources/gemini_trans_target.csv",
            "resources/openai_models.csv",
            "resources/openai_trans_source.csv",
            "resources/openai_trans_target.csv",
            "resources/MarianMT_models_short_list.csv",
            "resources/MarianMT_select_models.csv",
        )
        remaining = [path for path in legacy_resources if (PROJECT_ROOT / path).exists()]
        self.assertEqual([], remaining)

    def test_active_custom_ai_language_resources_are_preserved(self):
        for path in (
            "resources/google_trans_source.csv",
            "resources/google_trans_target.csv",
        ):
            self.assertTrue((PROJECT_ROOT / path).is_file(), path)

    def test_live_path_product_surface_modules_exist(self):
        live_modules = (
            "paddle_ocr_backend.py",
            "custom_ai.py",
            "custom_ai_profiles.py",
            "custom_ai_requests.py",
            "unified_translation_cache.py",
            "pyside_overlay.py",
            "handlers/translation_handler.py",
            "handlers/translation_requests.py",
            "handlers/translation_results.py",
            "handlers/display_manager.py",
            "worker_threads.py",
            "worker_ocr.py",
            "worker_translation.py",
            "worker_capture.py",
        )
        missing = [path for path in live_modules if not (PROJECT_ROOT / path).is_file()]
        self.assertEqual([], missing)

    def test_pyinstaller_specs_exclude_legacy_provider_dependencies(self):
        forbidden_tokens = (
            "convert_marian.py",
            "marian_mt_translator",
            "handlers.cache_manager",
            "handlers.gemini_models_manager",
            "handlers.statistics_handler",
            "google.cloud.translate_v2",
            "google.genai",
            "google.generativeai",
            "deepl",
            "torch",
            "transformers",
            "sentencepiece",
            "nvidia_ml_py3",
            "custom_ai_diagnostics",
            "custom_ai_metrics",
            "diagnostic_sanitizer",
            "update_applier",
            "update_checker",
        )
        for relative_path in (
            "GameChangingTranslator.spec",
            "GameChangingTranslator_GPU.spec",
        ):
            content = (PROJECT_ROOT / relative_path).read_text(encoding="utf-8").lower()
            inclusion_content = content.split("excludes=[", 1)[0]
            with self.subTest(spec=relative_path):
                present = [
                    token
                    for token in forbidden_tokens
                    if token.lower() in inclusion_content
                ]
                self.assertEqual([], present)

    def test_legacy_translation_adapters_are_inert(self):
        source = (
            PROJECT_ROOT / "handlers" / "translation_results.py"
        ).read_text(encoding="utf-8")
        tree = ast.parse(source)
        adapter_names = {
            "_google_translate",
            "_deepl_translate",
            "get_deepl_usage",
            "_legacy_translate_disabled",
            "_initialize_deepl_log_file",
            "_log_deepl_translation_call",
            "_is_deepl_logging_enabled",
        }
        found = set()
        for node in tree.body:
            if not isinstance(node, ast.ClassDef) or node.name != "TranslationResultsMixin":
                continue
            for item in node.body:
                if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if item.name not in adapter_names:
                    continue
                found.add(item.name)
                body_src = ast.get_source_segment(source, item) or ""
                for forbidden in (
                    "google_api_key_var",
                    "deepl_api_key_var",
                    "deepl_api_client",
                    "requests.",
                    "translate_text",
                    "translation.googleapis.com",
                    "api-free.deepl.com",
                    "api.deepl.com",
                ):
                    self.assertNotIn(
                        forbidden,
                        body_src,
                        msg=f"{item.name} still contains live legacy side effect: {forbidden}",
                    )
        self.assertEqual(adapter_names, found)

    def test_legacy_translation_adapters_do_not_read_keys_or_call_network(self):
        results = importlib.import_module("handlers.translation_results")

        class _Boom:
            def __getattr__(self, name):
                raise AssertionError(f"legacy adapter accessed app.{name}")

        class _Probe(results.TranslationResultsMixin):
            def __init__(self):
                self.app = _Boom()
                self.deepl_context_window = ["should-not-be-used"]
                self.deepl_current_source_lang = "en"
                self.deepl_current_target_lang = "zh"
                self.deepl_log_file = "should-not-be-used"
                self.deepl_log_lock = object()

            def _clear_deepl_context(self):
                raise AssertionError("legacy adapter should not clear DeepL context")

            def _build_deepl_context(self, *_args, **_kwargs):
                raise AssertionError("legacy adapter should not build DeepL context")

            def _update_deepl_context(self, *_args, **_kwargs):
                raise AssertionError("legacy adapter should not update DeepL context")

        probe = _Probe()
        self.assertIsNone(probe._legacy_translate_disabled())
        self.assertIsNone(probe._initialize_deepl_log_file())
        self.assertFalse(probe._is_deepl_logging_enabled())
        self.assertIsNone(
            probe._log_deepl_translation_call(
                "text", "en", "zh", "ctx", 1, "out", "quality_optimized", None, 0.1
            )
        )
        self.assertEqual(
            probe._google_translate("hello", "en", "zh"),
            "Legacy Google Translate path is archived",
        )
        self.assertEqual(
            probe._deepl_translate("hello", "en", "zh"),
            "Legacy DeepL path is archived",
        )
        self.assertIsNone(probe.get_deepl_usage())

    def test_app_logic_marks_legacy_provider_availability_false(self):
        app_logic_source = (PROJECT_ROOT / "app_logic.py").read_text(encoding="utf-8")
        for token in (
            "GOOGLE_TRANSLATE_API_AVAILABLE = False",
            "DEEPL_API_AVAILABLE = False",
            "GEMINI_API_AVAILABLE = False",
            "OPENAI_API_AVAILABLE = False",
            "MARIANMT_AVAILABLE = False",
        ):
            self.assertIn(token, app_logic_source)
        for token in (
            "import deepl",
            "import google.generativeai",
            "from google import genai",
            "import marian_mt_translator",
        ):
            self.assertNotIn(token, app_logic_source)

    def test_compile_script_validates_paddle_without_mutating_dependencies(self):
        content = (PROJECT_ROOT / "compile_app.py").read_text(encoding="utf-8").lower()
        for forbidden_token in (
            "pip uninstall",
            "pip install",
            "pytorch",
            "import torch",
        ):
            with self.subTest(token=forbidden_token):
                self.assertNotIn(forbidden_token, content)
        self.assertIn("verify_paddle_installation", content)

    def test_cpu_and_gpu_builds_validate_matching_paddle_runtime(self):
        self.assertTrue(hasattr(compile_app, "verify_paddle_installation"))
        with (
            mock.patch.object(compile_app, "verify_paddle_installation", return_value=True) as verify,
            mock.patch.object(compile_app, "run_command", return_value=True) as run,
        ):
            self.assertTrue(compile_app.compile_cpu_version())
            verify.assert_called_once_with(require_cuda=False)
            self.assertIn("GameChangingTranslator.spec", run.call_args.args[0])

        with (
            mock.patch.object(compile_app, "verify_paddle_installation", return_value=True) as verify,
            mock.patch.object(compile_app, "run_command", return_value=True) as run,
        ):
            self.assertTrue(compile_app.compile_gpu_version())
            verify.assert_called_once_with(require_cuda=True)
            self.assertIn("GameChangingTranslator_GPU.spec", run.call_args.args[0])

    def test_paddle_verifier_ignores_optional_torch_discovery(self):
        completed = compile_app.subprocess.CompletedProcess([], 0, stdout="ok", stderr="")
        with mock.patch.object(compile_app.subprocess, "run", return_value=completed) as run:
            self.assertTrue(compile_app.verify_paddle_installation(require_cuda=False))
        verification_code = run.call_args.args[0][2]
        self.assertIn("_real_find_spec", verification_code)
        self.assertIn('name == "torch"', verification_code)


if __name__ == "__main__":
    unittest.main()
