# SQLite Unified Translation Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace JSON whole-file cache persistence with SQLite-backed incremental persistence while preserving cache semantics and migrating existing legacy JSON cache data.

**Architecture:** Keep `UnifiedTranslationCache` as the in-memory cache façade and swap only its persistence backend. SQLite stores cache rows keyed by the existing five-part cache key; the cache continues to batch ordinary stores behind the current delayed timer and uses incremental row updates/deletes on flush.

**Tech Stack:** Python 3, `sqlite3`, `unittest`, existing app cache and handler code

---

### Task 1: Lock the new persistence behavior with failing tests

**Files:**
- Modify: `C:\Users\wordy\Desktop\OCR-Translator-3.9.6-experiment\tests\test_custom_ai.py`
- Test: `C:\Users\wordy\Desktop\OCR-Translator-3.9.6-experiment\tests\test_custom_ai.py`

- [ ] **Step 1: Add failing tests for SQLite persistence and migration**

Add tests that prove:

- the persisted cache file is a SQLite database after flush
- schema-versioned legacy JSON migrates into SQLite
- invalid legacy JSON without schema is ignored

- [ ] **Step 2: Run the focused tests and verify they fail for the expected reason**

Run:

```powershell
python -m unittest `
  tests.test_custom_ai.CustomAIProviderTests.test_persistent_cache_flush_creates_sqlite_database `
  tests.test_custom_ai.CustomAIProviderTests.test_persistent_cache_migrates_legacy_json_to_sqlite `
  tests.test_custom_ai.CustomAIProviderTests.test_persistent_cache_ignores_invalid_legacy_json `
  -v
```

Expected: failures because the current implementation still writes/reads JSON snapshots rather than SQLite.

### Task 2: Implement SQLite persistence inside `UnifiedTranslationCache`

**Files:**
- Modify: `C:\Users\wordy\Desktop\OCR-Translator-3.9.6-experiment\unified_translation_cache.py`
- Test: `C:\Users\wordy\Desktop\OCR-Translator-3.9.6-experiment\tests\test_custom_ai.py`

- [ ] **Step 1: Add SQLite persistence helpers and format detection**

Implement:

- lazy SQLite connection creation
- schema initialization
- SQLite row load
- legacy JSON detection and migration helpers

- [ ] **Step 2: Replace snapshot-file persistence with delta-aware SQLite writes**

Implement:

- dirty upsert tracking
- provider-clear persistence tracking
- full-resync path for `clear_all()` and migration
- file removal when the persisted cache becomes empty after `clear_all()`

- [ ] **Step 3: Run the focused persistence tests and make them pass**

Run:

```powershell
python -m unittest `
  tests.test_custom_ai.CustomAIProviderTests.test_persistent_cache_flush_creates_sqlite_database `
  tests.test_custom_ai.CustomAIProviderTests.test_persistent_cache_migrates_legacy_json_to_sqlite `
  tests.test_custom_ai.CustomAIProviderTests.test_persistent_cache_ignores_invalid_legacy_json `
  tests.test_custom_ai.CustomAIProviderTests.test_persistent_cache_store_is_deferred_until_flush `
  tests.test_custom_ai.CustomAIProviderTests.test_persistent_cache_clear_missing_provider_is_noop `
  tests.test_custom_ai.CustomAIProviderTests.test_persistent_cache_close_flushes_pending_entries `
  tests.test_custom_ai.CustomAIProviderTests.test_persistent_cache_clear_cannot_be_resurrected `
  -v
```

Expected: all pass.

### Task 3: Switch the app default cache path and verify integration

**Files:**
- Modify: `C:\Users\wordy\Desktop\OCR-Translator-3.9.6-experiment\app_logic.py`
- Modify if needed: `C:\Users\wordy\Desktop\OCR-Translator-3.9.6-experiment\handlers\translation_handler.py`
- Test: `C:\Users\wordy\Desktop\OCR-Translator-3.9.6-experiment\tests\test_custom_ai.py`

- [ ] **Step 1: Change the production cache filename to a SQLite path**

Update the app default from `custom_ai_translation_cache.json` to `custom_ai_translation_cache.sqlite3`.

- [ ] **Step 2: Adjust handler-level persistence tests if the path assumption changes**

Keep the handler contract the same: it should still reuse persisted translations between handler instances and should still avoid persisting failed translations.

- [ ] **Step 3: Run handler-level regression tests**

Run:

```powershell
python -m unittest `
  tests.test_custom_ai.TranslationHandlerCustomAITests.test_custom_ai_translation_uses_persistent_cache_between_handler_instances `
  tests.test_custom_ai.TranslationHandlerCustomAITests.test_custom_ai_translation_errors_are_not_persisted `
  tests.test_custom_ai.TranslationHandlerCustomAITests.test_close_flushes_cache_and_closes_provider `
  -v
```

Expected: all pass.

### Task 4: Full verification and delivery

**Files:**
- Modify: `C:\Users\wordy\Desktop\OCR-Translator-3.9.6-experiment\.codex\handoffs\<timestamp>.md`

- [ ] **Step 1: Run full fresh verification**

Run:

```powershell
python -m unittest tests.test_custom_ai -v
python -m unittest discover -s tests
python -m unittest discover
python -m compileall -q unified_translation_cache.py handlers\translation_handler.py app_logic.py tests\test_custom_ai.py
git diff --check
git diff --cached --check
```

Expected:

- all tests pass
- compile succeeds
- diff checks produce no output

- [ ] **Step 2: Write the handoff document**

Record:

- task summary
- changed files
- backup location
- tests/commands run and results
- migration decisions
- remaining issues or follow-ups

- [ ] **Step 3: Commit and push**

Run:

```powershell
git add unified_translation_cache.py tests\test_custom_ai.py app_logic.py docs\superpowers\specs\2026-07-03-sqlite-unified-translation-cache-design.md docs\superpowers\plans\2026-07-03-sqlite-unified-translation-cache.md .codex\handoffs\<timestamp>.md
git commit -m "perf: migrate unified translation cache persistence to sqlite"
git push origin codex/optimized-sync-2026-07-02
```

Expected:

- commit succeeds
- remote branch updates to the new commit
