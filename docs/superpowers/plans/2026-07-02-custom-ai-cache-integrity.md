# Custom AI Cache Integrity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Custom AI cache identity complete and move whole-file persistence out of the synchronous translation completion path without losing shutdown durability.

**Architecture:** `TranslationHandler` will build one semantic identity payload containing the profile transport and reasoning settings, and both cache and in-flight keys will consume it. `UnifiedTranslationCache` will keep synchronous in-memory LRU updates but coalesce disk writes behind one daemon timer, with versioned atomic snapshots and explicit `flush()`/`close()` lifecycle methods.

**Tech Stack:** Python 3, `threading.RLock`, `threading.Lock`, `threading.Timer`, JSON, `pathlib`, `unittest`

---

### Task 1: Complete Custom AI semantic identity

**Files:**
- Modify: `tests/test_custom_ai.py`
- Modify: `handlers/translation_handler.py`
- Modify: `unified_translation_cache.py`

- [x] **Step 1: Write failing cache and in-flight identity tests**

Add a cache test that stores under `wire_api="responses", reasoning_effort="xhigh"` and asserts misses for `chat_completions` and a different reasoning effort. Add a handler test that mutates the active profile and confirms `get_inflight_translation_key()` changes:

```python
def test_cache_key_is_isolated_by_wire_api_and_reasoning_effort(self):
    cache = UnifiedTranslationCache(max_size=10)
    params = {
        "profile_id": "profile-1",
        "base_url": "https://host.example/v1/",
        "model": "gpt-5.5",
        "wire_api": "responses",
        "reasoning_effort": "xhigh",
    }
    cache.store("Hello", "en", "zh-CN", "custom_ai", "你好", **params)
    self.assertIsNone(cache.get(
        "Hello", "en", "zh-CN", "custom_ai",
        **{**params, "wire_api": "chat_completions"},
    ))
    self.assertIsNone(cache.get(
        "Hello", "en", "zh-CN", "custom_ai",
        **{**params, "reasoning_effort": "low"},
    ))
```

- [x] **Step 2: Run the focused tests and verify RED**

Run:

```powershell
python -m unittest `
  tests.test_custom_ai.CustomAIProviderTests.test_cache_key_is_isolated_by_wire_api_and_reasoning_effort `
  tests.test_custom_ai.TranslationHandlerCustomAITests.test_inflight_key_isolated_by_wire_api_and_reasoning_effort -v
```

Expected: failures because the two identity fields are currently absent.

- [x] **Step 3: Add the identity fields once at the handler boundary**

Extend `_cache_params_for_profile()`:

```python
return {
    "profile_id": profile.get("id", ""),
    "base_url": str(profile.get("base_url", "")).strip().rstrip("/"),
    "model": profile.get("model", ""),
    "wire_api": str(profile.get("wire_api") or "chat_completions").strip().lower(),
    "reasoning_effort": str(
        profile.get("reasoning_effort")
        or profile.get("model_reasoning_effort")
        or ""
    ).strip().lower(),
    "custom_prompt": getattr(self.app, "custom_prompt_text", ""),
    "keep_linebreaks": keep_linebreaks,
    "context": tuple(self._get_custom_context_for_request()),
}
```

Make `get_inflight_translation_key()` consume the complete parameter payload, and include the two fields in `UnifiedTranslationCache._generate_cache_key()`.

- [x] **Step 4: Run focused tests and verify GREEN**

Run the Step 2 command. Expected: both tests pass.

### Task 2: Add versioned deferred persistence

**Files:**
- Modify: `tests/test_custom_ai.py`
- Modify: `unified_translation_cache.py`

- [x] **Step 1: Write failing persistence lifecycle tests**

Cover deferred store, explicit flush, coalescing, close durability, a late store racing after close, clear safety, and legacy schema invalidation. Use a long delay so tests are deterministic:

```python
cache = UnifiedTranslationCache(
    max_size=10,
    persistence_path=cache_path,
    persistence_delay_seconds=60.0,
)
cache.store("Bonjour", "fr", "en", "custom_ai", "Hello")
self.assertFalse(cache_path.exists())
self.assertTrue(cache.flush())
self.assertTrue(cache_path.exists())
cache.close()
```

Update the existing restore test to call `flush()` before constructing the second cache instance.

- [x] **Step 2: Run persistence tests and verify RED**

Run:

```powershell
python -m unittest `
  tests.test_custom_ai.CustomAIProviderTests.test_persistent_cache_store_is_deferred_until_flush `
  tests.test_custom_ai.CustomAIProviderTests.test_persistent_cache_coalesces_multiple_stores `
  tests.test_custom_ai.CustomAIProviderTests.test_persistent_cache_close_flushes_pending_entries `
  tests.test_custom_ai.CustomAIProviderTests.test_persistent_cache_store_after_close_is_synchronously_persisted `
  tests.test_custom_ai.CustomAIProviderTests.test_persistent_cache_clear_cannot_be_resurrected `
  tests.test_custom_ai.CustomAIProviderTests.test_persistent_cache_ignores_legacy_schema -v
```

Expected: errors or failures because delayed persistence, `flush()`, `close()`, and schema checks do not exist.

- [x] **Step 3: Implement the timer and versioned snapshot lifecycle**

Add:

```python
CACHE_SCHEMA_VERSION = 2

def __init__(self, max_size=1000, persistence_path=None, persistence_delay_seconds=0.25):
    self.persistence_delay_seconds = max(0.0, float(persistence_delay_seconds))
    self._persistence_lock = threading.Lock()
    self._persistence_timer = None
    self._persistence_generation = 0
    self._persisted_generation = 0
    self._closed = False
```

`store()` increments the generation and calls `_schedule_persistence_locked()`. The scheduled callback snapshots under `self.lock`, writes outside it under `_persistence_lock`, and skips a snapshot older than `_persisted_generation`. `flush()` cancels a pending timer and writes the latest generation synchronously. `close()` marks the cache closed and calls `flush()`. A translation that finishes after close stores synchronously because no future timer can be scheduled during shutdown.

Persist this envelope:

```python
payload = {
    "schema_version": CACHE_SCHEMA_VERSION,
    "entries": snapshot,
}
```

Only load payloads with the current schema version. Keep the `.tmp` plus `os.replace()` atomic write.

- [x] **Step 4: Make clear operations synchronously durable**

`clear_all()` and `clear_provider()` must increment the generation under the cache lock, cancel a pending timer, take the cleared snapshot, and write that generation synchronously. A concurrent newer store remains protected because an older generation cannot overwrite a newer persisted generation.

- [x] **Step 5: Run focused and existing cache tests**

Run:

```powershell
python -m unittest tests.test_custom_ai.CustomAIProviderTests -v
```

Expected: all provider and cache tests pass.

### Task 3: Flush cache during application shutdown

**Files:**
- Modify: `tests/test_custom_ai.py`
- Modify: `handlers/translation_handler.py`

- [x] **Step 1: Write a failing handler close test**

Replace `handler.unified_cache` with a mock and assert `handler.close()` calls cache close before provider close:

```python
handler.unified_cache = Mock()
handler.custom_ai_provider = Mock()
handler.close()
handler.unified_cache.close.assert_called_once_with()
handler.custom_ai_provider.close.assert_called_once_with()
```

- [x] **Step 2: Run the close test and verify RED**

Run:

```powershell
python -m unittest tests.test_custom_ai.TranslationHandlerCustomAITests.test_close_flushes_cache_and_closes_provider -v
```

Expected: failure because the cache is not closed.

- [x] **Step 3: Close the cache before the provider**

Update `TranslationHandler.close()` to call `self.unified_cache.close()` with sanitized debug logging, then close the provider as it does today.

- [x] **Step 4: Run focused tests and verify GREEN**

Run the Step 2 command. Expected: pass.

### Task 4: Verify, commit, and document

**Files:**
- Create: `.codex/handoffs/YYYY-MM-DD_HH-mm-ss.md`

- [x] **Step 1: Run full automated verification**

```powershell
python -m unittest tests.test_custom_ai -v
python -m unittest discover -s tests
python -m unittest discover
python -m compileall -q unified_translation_cache.py handlers\translation_handler.py tests\test_custom_ai.py
```

Expected: all tests pass and compileall exits 0.

- [x] **Step 2: Audit Git and ignored secrets**

```powershell
git diff --check
git status --short
git check-ignore ocr_translator_config.ini custom_ai_profiles.json custom_ai_translation_cache.json translator_debug.log
```

Expected: only intended source, test, plan, and handoff changes are visible; every runtime/sensitive path is ignored.

- [x] **Step 3: Run a source-mode startup/shutdown smoke test**

Launch `python main.py`, confirm the main window renders, close it normally, and confirm the process exits and the cache file remains valid JSON with schema version 2 after a cache write/flush probe.

- [x] **Step 4: Write the required handoff**

Record task summary, changed files, backup path, design decisions, red/green evidence, full verification, Git commits, current status, and remaining follow-up work.

- [x] **Step 5: Commit the implementation**

```powershell
git add unified_translation_cache.py handlers/translation_handler.py tests/test_custom_ai.py docs/superpowers/plans/2026-07-02-custom-ai-cache-integrity.md .codex/handoffs/<timestamp>.md
git commit -m "perf: harden custom AI translation cache"
```
