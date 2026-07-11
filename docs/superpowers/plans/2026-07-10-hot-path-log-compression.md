# Hot-Path Log Compression Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce synchronous runtime log writes and remove subtitle bodies from the main debug log without weakening relay/cooldown diagnostics.

**Architecture:** Add one shared content-free text summary, then replace per-frame OCR noise with stable-key coalesced events and replace translation/display bodies with shape metadata. Keep errors, sequence IDs, timings, status codes, cooldowns, and the dedicated Custom AI short log unchanged.

**Tech Stack:** Python 3, rotating UTF-8 logs, existing `_LogCoalescingGate`, `unittest`, `unittest.mock`.

---

### Task 1: Add the shared content-free text summary

**Files:**
- Modify: `tests/test_runtime_logging.py`
- Modify: `logger.py`

- [x] **Step 1: Write the failing summary test**

```python
summary = logger.summarize_text_for_log("source-secret\nsecond line")
self.assertEqual(summary, "chars=25 lines=2")
self.assertNotIn("source-secret", summary)
self.assertEqual(logger.summarize_text_for_log(None), "chars=0 lines=0")
```

- [x] **Step 2: Run RED**

```powershell
py -m unittest tests.test_runtime_logging.RuntimeTextSummaryTests -v
```

Expected: `AttributeError` because the helper does not exist.

- [x] **Step 3: Implement the minimal helper**

```python
def summarize_text_for_log(value):
    text = value if isinstance(value, str) else "" if value is None else str(value)
    line_count = text.count("\n") + 1 if text else 0
    return f"chars={len(text)} lines={line_count}"
```

- [x] **Step 4: Run GREEN**

Run the same command. Expected: all summary tests pass.

### Task 2: Coalesce PaddleOCR failure/filter events

**Files:**
- Modify: `tests/test_paddle_ocr_backend.py`
- Modify: `paddle_ocr_backend.py`

- [x] **Step 1: Write failing behavioral tests**

Add tests that feed a low-confidence secret line and a no-line-crop image. Patch
both log functions and assert:

```python
log_debug.assert_not_called()
log_debug_coalesced.assert_called_with(
    "paddle-subtitle-fast-path-no-line-crop",
    "PaddleOCR subtitle fast path found no line crop; falling back to full OCR",
    interval_seconds=5.0,
)
```

For filter events, assert the coalesced message includes `chars=` and
`confidence=` but not the secret OCR text.

- [x] **Step 2: Run RED**

```powershell
py -m unittest tests.test_paddle_ocr_backend.PaddleOCRBackendTests.test_subtitle_filter_log_is_coalesced_and_content_free tests.test_paddle_ocr_backend.PaddleOCRBackendTests.test_subtitle_no_line_crop_log_is_coalesced -v
```

Expected: failures because both paths still call `log_debug` per event.

- [x] **Step 3: Replace per-frame writes with stable coalesced events**

Use separate event keys for full low confidence, subtitle low confidence,
subtitle symbol noise, no usable text, no line crop, and fast-path exception.
Build filter messages with `summarize_text_for_log(clean_text)`.

- [x] **Step 4: Run GREEN and the PaddleOCR suite**

```powershell
py -m unittest tests.test_paddle_ocr_backend -q
```

Expected: all PaddleOCR tests pass.

### Task 3: Make translation and display diagnostics content-free

**Files:**
- Modify: `tests/test_latency_optimization.py`
- Modify: `tests/test_runtime_logging.py`
- Modify: `tests/test_pyside_text_render.py`
- Modify: `worker_threads.py`
- Modify: `handlers/translation_handler.py`
- Modify: `handlers/display_manager.py`

- [x] **Step 1: Write failing worker pipeline tests**

Exercise `_submit_async_translation_request`, `process_translation_async`, and
`process_translation_response` with `source-secret` and `result-secret`.
Assert normal debug messages contain sequence IDs and `chars=/lines=` but never
either secret.

- [x] **Step 2: Write failing handler/display tests**

Call `_format_dialog_text()` with secret dialog text and the PySide display path
with secret text. Assert dialog no-op formatting writes nothing, applied
formatting emits one content-free coalesced event, and display logs contain only
shape plus coalesced language/backend route messages.

- [x] **Step 3: Run RED**

```powershell
py -m unittest tests.test_latency_optimization.RuntimeContentFreeTranslationLogTests tests.test_runtime_logging.RuntimeContentFreeHandlerLogTests tests.test_pyside_text_render.DisplayManagerPySideRenderTests.test_pyside_logging_is_content_free_and_routes_are_coalesced -v
```

Expected: failures because current messages include source/result/display text
and dialog formatting emits multiple direct logs.

- [x] **Step 4: Implement content-free/coalesced logging**

Import `summarize_text_for_log` at the three runtime boundaries. Keep detailed
error messages, but use summaries for successful source/result/display events.
Delete unconditional dialog debug writes and emit one
`translation-dialog-format-applied` coalesced event only when output changes.
Coalesce display language and widget-route messages with five-second windows.

- [x] **Step 5: Run GREEN and focused suites**

```powershell
py -m unittest tests.test_runtime_logging tests.test_paddle_ocr_backend tests.test_latency_optimization tests.test_pyside_text_render -q
```

Expected: all focused tests pass.

### Task 4: Verify the complete increment

**Files:**
- Modify: `docs/superpowers/plans/2026-07-10-hot-path-log-compression.md`

- [x] **Step 1: Run the full suites**

```powershell
py -m unittest discover -s tests
py -m unittest discover
```

Expected: all tests pass.

- [x] **Step 2: Run compile and patch checks**

```powershell
py -B -m py_compile logger.py paddle_ocr_backend.py worker_threads.py handlers\translation_handler.py handlers\display_manager.py tests\test_runtime_logging.py tests\test_paddle_ocr_backend.py tests\test_latency_optimization.py tests\test_pyside_text_render.py
py -m compileall -q logger.py paddle_ocr_backend.py worker_threads.py handlers tests
git diff --check
```

Expected: all commands exit 0.

- [x] **Step 3: Run an A/B logging-volume probe**

Replay representative repeated OCR failure/filter, translation, dialog, and
display events against the backed-up and current modules in isolated temporary
log directories. Report write count and bytes; do not make network calls.

- [x] **Step 4: Mark this plan complete**

Change checkboxes to `[x]` only after their evidence has been recorded.

### Review follow-up: Complete the main-log content boundary

- [x] Add an API OCR-to-translation regression proving recognized subtitle
  text reaches the scheduler but never the main debug log.
- [x] Replace remaining cache, legacy provider, unified translation, API OCR,
  and translation-error body logs with `chars=/lines=` summaries.
- [x] Add an AST-backed invariant test that rejects direct interpolation of
  runtime OCR/translation text in main debug log calls.
- [x] Make the shared summary helper fail closed when arbitrary `__str__`
  conversion raises.

### Final review follow-up: Legacy provider and bounded fallback reason

- [x] Extend the AST-backed invariant across Marian translation methods.
- [x] Replace Marian source, sentence, fallback, and result bodies with shared
  content-free shape summaries.
- [x] Add a Paddle fast-path fallback regression and emit a single-line,
  quote/path/URL-redacted, 120-character exception reason.
