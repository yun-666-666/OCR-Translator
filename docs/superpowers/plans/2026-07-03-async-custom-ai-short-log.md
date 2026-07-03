# Async Custom AI Short Log Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Custom AI short-log writes non-blocking for translation/OCR calls while flushing accepted writes on close.

**Architecture:** `TranslationHandler` will submit preformatted log blocks to one private single-worker executor. A state lock will serialize submission against executor detachment during close.

**Tech Stack:** Python, `concurrent.futures.ThreadPoolExecutor`, `threading`, `unittest`

---

### Task 1: Non-blocking write and shutdown barrier

**Files:**
- Modify: `handlers/translation_handler.py`
- Modify: `tests/test_custom_ai.py`

- [x] **Step 1: Write the failing blocked-writer test**

```python
def test_custom_ai_short_log_does_not_block_translation_and_close_flushes(self):
    handler = TranslationHandler(object())
    write_started = threading.Event()
    release_write = threading.Event()
    log_returned = threading.Event()
    close_returned = threading.Event()

    def blocked_append(*args, **kwargs):
        write_started.set()
        release_write.wait(timeout=2.0)

    def log_call():
        handler._log_custom_short_call(
            "translation",
            {"name": "Translator", "model": "demo"},
            "translated",
            {"prompt_tokens": 1, "completion_tokens": 1},
            0.01,
        )
        log_returned.set()

    with patch.object(
        translation_handler_module,
        "append_rotating_text",
        side_effect=blocked_append,
    ):
        caller = threading.Thread(target=log_call)
        caller.start()
        self.assertTrue(write_started.wait(timeout=1.0))
        returned_while_blocked = log_returned.wait(timeout=0.05)

        closer = threading.Thread(
            target=lambda: (handler.close(), close_returned.set())
        )
        closer.start()
        close_waited_for_write = not close_returned.wait(timeout=0.05)

        release_write.set()
        caller.join(timeout=1.0)
        closer.join(timeout=1.0)

    self.assertTrue(returned_while_blocked)
    self.assertTrue(close_waited_for_write)
    self.assertTrue(close_returned.is_set())
```

- [x] **Step 2: Verify RED**

Run:
`python -m unittest tests.test_custom_ai.TranslationHandlerCustomAITests.test_custom_ai_short_log_does_not_block_translation_and_close_flushes -v`

Expected: FAIL because the logging call blocks and `close()` does not wait for it.

- [x] **Step 3: Initialize async log state**

In `TranslationHandler.__init__()` create:

```python
self._custom_log_state_lock = threading.Lock()
self._custom_log_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="CustomAIShortLog",
)
self._custom_session_started = set()
```

- [x] **Step 4: Submit writes asynchronously**

Move `append_rotating_text(...)` into:

```python
def _write_custom_short_log(self, log_file, block):
    try:
        append_rotating_text(
            log_file,
            block,
            max_bytes=2 * 1024 * 1024,
            backup_count=2,
        )
    except Exception as error:
        log_debug(f"Custom AI short log write failed: {error}")
```

Build the block under `_custom_log_state_lock`, add the call type to the session
set, and submit `_write_custom_short_log` while holding the same lock.

- [x] **Step 5: Add the close barrier**

Under `_custom_log_state_lock`, move the executor to a local variable and set the
instance attribute to `None`. Outside the lock, call
`executor.shutdown(wait=True, cancel_futures=False)` and log shutdown exceptions.

- [x] **Step 6: Verify GREEN**

Run the Step 2 command and expect PASS.

### Task 2: Existing log-content compatibility

**Files:**
- Modify: `tests/test_custom_ai.py`

- [x] **Step 1: Update the cached-token log test**

Keep the `append_rotating_text` patch active, call
`handler._log_custom_short_call(...)`, then call `handler.close()` before
inspecting `append_text.call_args`.

- [x] **Step 2: Run focused log tests**

Run the blocked-writer and cached-token log tests. Expect both PASS and no live
logging thread after handler close.

### Task 3: Full verification and handoff

**Files:**
- Create: `.codex/handoffs/2026-07-03_18-15-43.md`

- [x] Run `python -m unittest tests.test_custom_ai -q`.
- [x] Run `python -m unittest tests.test_latency_optimization -q`.
- [x] Run `python -m unittest discover -s tests -q`.
- [x] Run `python -m unittest discover -q`.
- [x] Run `python -m compileall -q handlers\translation_handler.py tests`.
- [x] Run `git diff --check`.
- [x] Repeat the blocked-write probe and record critical-path return and close
  flush behavior in the handoff.
