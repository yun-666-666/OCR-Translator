# Live Path Inventory Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove verified dead provider inventory and stale packaging dependencies while preserving the PaddleOCR and Custom AI live path.

**Architecture:** Treat the supported repository inventory as an explicit contract. Delete only the closed legacy provider/resource cluster, remove its Marian facade methods, and narrow the existing PyInstaller declarations without changing live runtime data flow.

**Tech Stack:** Python 3.12, `unittest`, PyInstaller 6.20, PowerShell, Git

---

### Task 1: Add the failing live-path inventory contract

**Files:**
- Create: `tests/test_live_path_inventory.py`

- [ ] **Step 1: Write the failing repository contract**

```python
import unittest
from pathlib import Path


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
            "google.cloud.translate_v2",
            "google.genai",
            "google.generativeai",
            "deepl",
            "transformers",
            "sentencepiece",
            "nvidia_ml_py3",
        )
        for relative_path in (
            "GameChangingTranslator.spec",
            "GameChangingTranslator_GPU.spec",
        ):
            content = (PROJECT_ROOT / relative_path).read_text(encoding="utf-8").lower()
            with self.subTest(spec=relative_path):
                present = [token for token in forbidden_tokens if token.lower() in content]
                self.assertEqual([], present)
```

- [ ] **Step 2: Run the test and verify RED**

Run: `python -m unittest tests.test_live_path_inventory -v`

Expected: FAIL because the legacy source/resources still exist and both specs still name legacy dependencies.

### Task 2: Back up and remove the closed legacy inventory

**Files:**
- Delete: `marian_mt_translator.py`
- Delete: `convert_marian.py`
- Delete: `handlers/gemini_provider.py`
- Delete: `handlers/gemini_ocr_provider.py`
- Delete: `handlers/gemini_models_manager.py`
- Delete: `handlers/openai_provider.py`
- Delete: `handlers/openai_ocr_provider.py`
- Delete: `handlers/openai_models_manager.py`
- Delete: `handlers/ocr_provider_base.py`
- Delete: `handlers/llm_provider_base.py`
- Delete: `resources/deepl_trans_source.csv`
- Delete: `resources/deepl_trans_target.csv`
- Delete: `resources/gemini_models.csv`
- Delete: `resources/gemini_trans_source.csv`
- Delete: `resources/gemini_trans_target.csv`
- Delete: `resources/openai_models.csv`
- Delete: `resources/openai_trans_source.csv`
- Delete: `resources/openai_trans_target.csv`
- Delete: `resources/MarianMT_models_short_list.csv`
- Delete: `resources/MarianMT_select_models.csv`

- [ ] **Step 1: Copy every existing target into one timestamped `.codex/backups/` tree**

Use workspace-relative paths and verify file counts before deletion.

- [ ] **Step 2: Delete only the listed tracked files**

Use `apply_patch` deletion patches; do not touch the active Google language resources.

- [ ] **Step 3: Re-run the inventory test**

Expected: source/resource tests pass; spec test still fails until Task 4.

### Task 3: Remove dead Marian facade methods and obsolete test scan entries

**Files:**
- Modify: `handlers/translation_results.py`
- Modify: `tests/test_runtime_logging.py`

- [ ] **Step 1: Remove Marian-only methods from `TranslationResultsMixin`**

Delete `_marian_translate`, `update_marian_active_model`, `update_marian_beam_value`, and `initialize_marian_translator`. Preserve `clear_cache`, error formatting, and unrelated result handling. Remove imports only when a reference scan proves they became unused.

- [ ] **Step 2: Remove deleted-file entries from the runtime logging source scan**

Keep the sanitizer assertions for all live Custom AI, OCR, and worker modules.

- [ ] **Step 3: Run focused runtime tests**

Run: `python -m unittest tests.test_runtime_logging tests.test_custom_ai_startup -v`

Expected: PASS.

### Task 4: Narrow both PyInstaller specs to the live dependency set

**Files:**
- Modify: `GameChangingTranslator.spec`
- Modify: `GameChangingTranslator_GPU.spec`

- [ ] **Step 1: Remove deleted data and hidden-import entries**

Remove `convert_marian.py`, `marian_mt_translator`, `handlers.cache_manager`, and `handlers.gemini_models_manager`.

- [ ] **Step 2: Remove retired provider SDK dependency families**

Remove Google Translate, Gemini SDK, DeepL, Marian/Torch/Transformers/tokenizer, and legacy GPU-monitoring entries. Retain application, Tk, PySide6, image, HTTP, OCR, RTL, update, and standard-library declarations.

- [ ] **Step 3: Verify GREEN**

Run: `python -m unittest tests.test_live_path_inventory -v`

Expected: all four inventory tests pass.

- [ ] **Step 4: Validate spec syntax and execute a build when practical**

Run: `python -m py_compile GameChangingTranslator.spec GameChangingTranslator_GPU.spec`

Run: `python -m PyInstaller --clean --noconfirm GameChangingTranslator.spec`

Expected: spec compilation succeeds; the CPU build exits 0. If environment-specific Paddle collection prevents the build, capture the exact failure and do not claim packaged-runtime verification.

### Task 5: Full verification and handoff

**Files:**
- Create: `.codex/handoffs/YYYY-MM-DD_HH-mm-ss.md`

- [ ] **Step 1: Run the full suite**

Run: `python -m unittest discover -s tests -q`

Expected: all tests pass.

- [ ] **Step 2: Compile touched Python files and scan references**

Run: `python -m py_compile handlers/translation_results.py tests/test_runtime_logging.py tests/test_live_path_inventory.py`

Run repository scans for deleted module names and resource paths outside docs, backups, generated manifests, and historical changelogs.

- [ ] **Step 3: Check repository hygiene**

Run: `git diff --check`

Review: `git status --short` and the scoped diff, preserving unrelated untracked files.

- [ ] **Step 4: Write the required handoff**

Record the task summary, exact files, backup path, commands and results, design decisions, rejected Grok advice, and remaining packaging/runtime verification.
