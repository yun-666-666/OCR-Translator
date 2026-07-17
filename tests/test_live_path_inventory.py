import unittest
from pathlib import Path
from unittest import mock

import compile_app


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class LivePathInventoryTests(unittest.TestCase):
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
