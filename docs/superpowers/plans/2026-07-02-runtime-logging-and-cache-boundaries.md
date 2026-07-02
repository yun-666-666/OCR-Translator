# Runtime Logging and Cache Boundaries Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bound and accelerate runtime logging, isolate automated-test logs, throttle repetitive adaptive OCR diagnostics, and correct translation-cache capacity behavior.

**Architecture:** `logger.py` becomes the single owner of thread-safe rotating text writers and runtime/test path resolution. Existing call sites keep `log_debug()`, while Custom AI raw logs and UI clearing use new focused logger APIs. Cache and adaptive changes remain local to their existing components.

**Tech Stack:** Python 3, `threading.RLock`, line-buffered UTF-8-SIG files, `pathlib`, `unittest`, Tk application state

---

### Task 1: Correct cache capacity boundaries

**Files:**
- Modify: `tests/test_custom_ai.py`
- Modify: `unified_translation_cache.py`

- [x] **Step 1: Write failing capacity tests**

```python
def test_full_cache_update_does_not_evict_another_entry(self):
    cache = UnifiedTranslationCache(max_size=3)
    for key in ("one", "two", "three"):
        cache.store(key, "en", "zh-CN", "custom_ai", key)
    cache.store("three", "en", "zh-CN", "custom_ai", "updated")
    self.assertEqual(cache.get_stats()["total_entries"], 3)
    self.assertEqual(cache.get("one", "en", "zh-CN", "custom_ai"), "one")

def test_persistent_load_trims_exactly_to_max_size(self):
    # Write a schema-2 payload with 12 valid entries and increasing access times.
    cache = UnifiedTranslationCache(max_size=3, persistence_path=cache_path)
    self.assertEqual(cache.get_stats()["total_entries"], 3)
```

- [x] **Step 2: Run tests and verify RED**

Run:

```powershell
python -m unittest `
  tests.test_custom_ai.CustomAIProviderTests.test_full_cache_update_does_not_evict_another_entry `
  tests.test_custom_ai.CustomAIProviderTests.test_persistent_load_trims_exactly_to_max_size -v
```

Expected: the update leaves only two entries, and oversized loading remains above three.

- [x] **Step 3: Implement exact eviction semantics**

```python
if cache_key not in self._cache and len(self._cache) >= self.max_size:
    self._evict_lru_entries()
```

Extend `_evict_lru_entries(entries_to_evict=None)` so persisted loading passes `len(self._cache) - self.max_size`, while normal insertion keeps the 10% batch.

- [x] **Step 4: Run focused tests and verify GREEN**

Run the Step 2 command. Expected: both tests pass.

### Task 2: Implement reusable bounded log writers

**Files:**
- Create: `tests/test_runtime_logging.py`
- Modify: `logger.py`
- Modify: `.gitignore`

- [x] **Step 1: Write failing writer tests**

Cover stream reuse, bounded rotation, concurrent writes, safe clear/reuse, and test-path resolution:

```python
writer = logger._RotatingTextWriter(path, max_bytes=256, backup_count=2)
writer.write("first\n")
stream = writer._stream
writer.write("second\n")
self.assertIs(writer._stream, stream)
```

Use multiple threads and assert every unique line is present exactly once across the active log and backups.

- [x] **Step 2: Run tests and verify RED**

Run:

```powershell
python -m unittest tests.test_runtime_logging -v
```

Expected: import errors because the writer and resolver do not exist.

- [x] **Step 3: Implement the writer registry and path resolver**

`logger.py` will define:

```python
class _RotatingTextWriter:
    def write(self, text): ...
    def clear(self, marker): ...
    def close(self): ...

def resolve_runtime_log_path(filename, test_process=None): ...
def append_rotating_text(filename, text, max_bytes, backup_count): ...
def clear_debug_log(): ...
```

`resolve_runtime_log_path()` uses `OCR_TRANSLATOR_LOG_DIR` when set, otherwise redirects unittest/pytest processes to a PID-specific system-temp directory. Register `close_log_writers()` with `atexit`.

- [x] **Step 4: Ignore numbered rotated files**

Add:

```gitignore
*.log.*
*_Log.txt.*
*_log.txt.*
```

- [x] **Step 5: Run logging tests and verify GREEN**

Run the Step 2 command. Expected: all logging tests pass.

### Task 3: Route application logging and throttle adaptive diagnostics

**Files:**
- Modify: `tests/test_runtime_logging.py`
- Modify: `tests/test_latency_optimization.py`
- Modify: `handlers/translation_handler.py`
- Modify: `handlers/ui_interaction_handler.py`
- Modify: `app_logic.py`

- [x] **Step 1: Write failing integration and throttling tests**

Assert Custom AI short calls use `append_rotating_text()`, UI clear calls `logger.clear_debug_log()`, repeated normal adaptive checks emit one message within 30 seconds, and a normal-to-overload transition logs immediately.

- [x] **Step 2: Run focused tests and verify RED**

Run:

```powershell
python -m unittest `
  tests.test_runtime_logging `
  tests.test_latency_optimization.AdaptiveScanLoggingTests -v
```

Expected: failures because direct file opens and per-check logging remain.

- [x] **Step 3: Route raw and clear operations through logger APIs**

Build one Custom AI short-log block and call:

```python
append_rotating_text(log_file, block, max_bytes=2 * 1024 * 1024, backup_count=2)
```

Replace direct truncation in `UIInteractionHandler.clear_debug_log()` with the logger-level clear API.

- [x] **Step 4: Add transition plus heartbeat adaptive logging**

Initialize:

```python
self._last_adaptive_log_state = None
self._last_adaptive_log_time = 0.0
```

Log only when the state bucket changes or 30 seconds elapsed, without changing interval decisions.

- [x] **Step 5: Run focused tests and verify GREEN**

Run the Step 2 command. Expected: all integration and adaptive tests pass.

### Task 4: Verify, desktop-debug, review, and integrate

**Files:**
- Create: `.codex/handoffs/<timestamp>.md`

- [x] **Step 1: Run full automated verification**

```powershell
python -m unittest tests.test_runtime_logging -v
python -m unittest discover -s tests
python -m unittest discover
python -m compileall -q logger.py unified_translation_cache.py handlers\translation_handler.py handlers\ui_interaction_handler.py app_logic.py tests
```

- [x] **Step 2: Run performance and isolation probes**

Compare 5,000 debug writes against the backed-up logger using temporary files. Hash the real runtime logs before and after full unit tests and assert they do not change.

- [x] **Step 3: Use Computer Use for visible desktop validation**

Launch the source application, inspect the Home and Debugging states, invoke the non-destructive refresh control, then close the app normally. The destructive clear control is covered by an automated test and is not clicked without action-time confirmation.

- [x] **Step 4: Review remaining hotspots**

Reinspect fresh runtime logs and changed diffs. If another bounded, independently testable high-value issue is found, add its RED-GREEN increment before completion.

- [ ] **Step 5: Write handoff and commit**

Record backups, design decisions, TDD evidence, performance numbers, desktop observations, remaining issues, and Git state. Commit the implementation on the feature branch, fast-forward `main`, rerun full tests, and delete the merged feature branch.
