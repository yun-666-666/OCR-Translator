# Live Path Inventory Cleanup Design

## Goal

Make the tracked source and PyInstaller inventory match the supported runtime path: PaddleOCR or Custom AI OCR, followed by Custom AI translation and overlay display.

## Evidence and scope

The July 17 Grok review correctly identifies dedicated Gemini, OpenAI, and MarianMT implementations as dead inventory. Repository-wide reference scans show that those modules are referenced only by one another, stale packaging declarations, or tests that scan their text. The current 615-test baseline passes without importing them.

This increment will:

- remove the dedicated Gemini/OpenAI translation and OCR providers, their model managers, and their now-private base classes;
- remove the MarianMT runtime and conversion script;
- remove provider-only DeepL, Gemini, OpenAI, and Marian CSV resources;
- remove dead Marian adapter methods from the translation-result mixin;
- remove corresponding legacy-only entries and dependency families from both PyInstaller specs;
- add a repository contract test that prevents those files and packaging references from returning.

This increment will not:

- delete `resources/google_trans_source.csv` or `resources/google_trans_target.csv`, because `LanguageManager` still uses them as the Custom AI language list;
- remove secure legacy-key migration/stripping in `config_manager.py`;
- rename active Custom AI OCR concurrency fields that still carry Gemini-era names;
- add UI modes, diagnostics panels, or speculative cache changes;
- split large live-path modules or tests.

## Design

### Inventory contract

A focused test will define the files that must not exist and the legacy tokens that must not appear in either PyInstaller spec. It will also assert that the two Google-named language CSVs remain present while they are an active dependency. The test is intentionally a repository contract: the behavior under protection is the supported product inventory and build declaration, not an internal implementation detail.

### Source cleanup

The unused provider modules can be deleted as a closed dependency cluster. The Marian helper methods in `handlers/translation_results.py` will be removed so the facade no longer advertises a provider whose implementation is gone. Imports that become unused after that removal will be cleaned only in the touched file.

### Resource cleanup

Only CSV files with no live reader will be deleted. Canonical UI/localization resources and the Google-named language lists used by Custom AI remain unchanged.

### Packaging cleanup

Both specs will retain their current application structure and data layout, while removing explicit inclusion of deleted source, deleted handler shims, Google/DeepL/Gemini SDKs, and Marian/Torch/Transformers dependencies. PaddleOCR continues to be discovered from the live imports and installed environment; this increment does not invent a new GPU packaging architecture without a verified build need.

## Safety and verification

Existing files will be backed up with workspace-relative paths before edits or deletions. TDD will be used: the inventory contract must fail against the current tree, then pass after cleanup. Verification will include the full unit suite, Python compilation for touched Python files, static spec execution/compilation checks, stale-reference scans, `git diff --check`, and a PyInstaller build when practical in the current environment. Any packaging behavior that cannot be exercised will be reported as unverified rather than inferred.

## Rejected or deferred Grok suggestions

- Deleting the Google language CSVs is rejected for now because it would break the active Custom AI language loader.
- Performance tuning, diagnostics UI, and cache-key changes are deferred until runtime logs or benchmarks identify a concrete bottleneck.
- Large module/test splits are deferred because they create broad churn without a demonstrated runtime gain.
- Automatic cleanup of `.codex`, handoffs, backups, or user runtime files is rejected as destructive workspace hygiene outside this product increment.
