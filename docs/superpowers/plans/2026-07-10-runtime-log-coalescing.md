# Runtime Log Coalescing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce repeated cache and translation-state log lines by about 90 percent without losing first occurrences or aggregate counts.

**Architecture:** Add one thread-safe message-free coalescing gate in `logger.py`. Opt only the four evidence-backed noisy call sites into a five-second window; do not change cache, queue, timer, translation, or metric state.

**Tech Stack:** Python 3.12, `threading.RLock`, monotonic time, `unittest`, existing rotating debug logger.

---

### Task 1: Thread-Safe Coalescing Primitive

**Files:**
- Modify: `logger.py`
- Test: `tests/test_runtime_logging.py`

- [x] **Step 1: Write failing gate tests**

Test a fake clock at 100, 101, and 105 seconds. Require the first message,
`None` inside the window, then a current message suffixed with one suppressed
event. Add independent-key, clear, and concurrent-call tests.

- [x] **Step 2: Run tests and verify RED**

```powershell
py -m unittest tests.test_runtime_logging.LogCoalescingGateTests -v
```

Expected: import or attribute failure because `_LogCoalescingGate` does not
exist.

- [x] **Step 3: Implement the minimal gate and wrapper**

Add `_LogCoalescingGate.prepare(key, message, interval_seconds=5.0)`, storing
only `(last_logged_monotonic, suppressed_count)` under an `RLock`. Add
`log_debug_coalesced(event_key, message, interval_seconds=5.0)` that calls
`log_debug` only for non-`None` prepared messages and returns a boolean. Add a
`clear()` method and call it from `clear_debug_log`.

- [x] **Step 4: Run tests and verify GREEN**

Use the Task 1 command. Expected: all gate tests pass.

### Task 2: Coalesce Unified Cache Misses

**Files:**
- Modify: `unified_translation_cache.py`
- Test: `tests/test_custom_ai.py`

- [x] **Step 1: Write a failing cache log test**

Patch `unified_translation_cache.log_debug_coalesced`, make two misses for the
same provider/language route, and assert both cache calls still return `None`
while the stable event key and content-free message are sent to the coalescer.

- [x] **Step 2: Run test and verify RED**

```powershell
py -m unittest tests.test_custom_ai.CustomAIProviderTests.test_unified_cache_miss_uses_coalesced_content_free_log -v
```

Expected: patch target missing or zero calls.

- [x] **Step 3: Route only MISS through the coalescer**

Import `log_debug_coalesced`, create a key from normalized provider/source/
target/model type, and send the existing MISS message through a five-second
window. Leave HIT, STORE, persistence errors, evictions, and clears unchanged.

- [x] **Step 4: Run the focused test and cache regressions**

Run the Task 2 command and `py -m unittest tests.test_custom_ai -v`.

### Task 3: Coalesce Worker Queue State

**Files:**
- Modify: `worker_threads.py`
- Test: `tests/test_latency_optimization.py`

- [x] **Step 1: Write failing worker log tests**

Patch `worker_threads.log_debug_coalesced`. Exercise queueing, duplicate
in-flight skip, and stale timer branches. Assert stable keys, five-second
intervals, diagnostic numeric state, and absence of source text. Also assert
pending request and runtime metric state is unchanged.

- [x] **Step 2: Run tests and verify RED**

```powershell
py -m unittest tests.test_latency_optimization.RuntimeLogCoalescingTests -v
```

Expected: patch target missing or direct `log_debug` calls remain.

- [x] **Step 3: Route the three call sites through the coalescer**

Import `log_debug_coalesced`, replace only the selected direct calls, remove
source text from queue/duplicate messages, and preserve existing state updates.

- [x] **Step 4: Run tests and verify GREEN**

Use the Task 3 command, then run
`py -m unittest tests.test_latency_optimization -v`.

### Task 4: Full Verification And Handoff

**Files:**
- Create: `.codex/handoffs/2026-07-10_14-35-30.md`

- [x] **Step 1: Run targeted suites**

```powershell
py -m unittest tests.test_runtime_logging -v
py -m unittest tests.test_latency_optimization -v
py -m unittest tests.test_custom_ai -v
```

- [x] **Step 2: Run full checks**

```powershell
py -m unittest discover -s tests
py -m unittest discover
py -B -m py_compile logger.py unified_translation_cache.py worker_threads.py tests\test_runtime_logging.py tests\test_custom_ai.py tests\test_latency_optimization.py
git diff --check
```

- [x] **Step 3: Compare against the backup**

Review only changes relative to `.codex/backups/2026-07-10_14-35-18/` and
confirm all non-selected logging and all functional state remain unchanged.

- [x] **Step 4: Write the handoff**

Record evidence, files, backup path, RED/GREEN results, final test counts,
decisions, and the remaining live-session validation gap in the exact handoff
path above.
