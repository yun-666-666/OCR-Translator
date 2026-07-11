# Capture And OCR Hot-Path Logging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove about 95 percent of successful per-frame capture/OCR log writes while keeping failures, state changes, slow events, and periodic health summaries visible.

**Architecture:** Reuse `logger.log_debug_coalesced`. Route utility-level capture/cache success and Paddle recognition success through stable five-second keys. Add a worker-only timing router that separates normal five-second summaries from slow one-second summaries. Do not change capture, OCR, caching, queue, or metric state.

**Tech Stack:** Python 3.12, `unittest`, `unittest.mock`, existing rotating debug logger and coalescing gate.

---

### Task 1: Lock Utility Success Logging

**Files:**
- Modify: `tests/test_latency_optimization.py`
- Modify: `ocr_utils.py`

- [x] **Step 1: Add failing tests**

Add tests proving successful `capture_screen_region` calls use a stable
backend/geometry coalescing key and `OCRFrameCache.get` uses one stable cache
hit key. Assert capture/cache results and cache LRU behavior are unchanged.

- [x] **Step 2: Run focused tests and verify RED**

```powershell
py -m unittest tests.test_latency_optimization.CaptureOcrHotPathLoggingTests -v
```

Expected: import/patch/call failures because utility successes still call
`log_debug` directly.

- [x] **Step 3: Route only successful utility events through the coalescer**

Import `log_debug_coalesced` in `ocr_utils.py`. Use a five-second key containing
capture backend/width/height and one stable OCR frame-hit key. Leave unknown
backend, backend failure, fallback, and terminal failure logs immediate.

- [x] **Step 4: Run focused tests and verify GREEN**

Use the Task 1 command.

### Task 2: Add Normal/Slow Worker Timing Channels

**Files:**
- Modify: `tests/test_latency_optimization.py`
- Modify: `worker_threads.py`

- [x] **Step 1: Add failing timing/router tests**

Require a normal event to use `(key, "normal")` with a five-second interval and
a threshold event to use `(key, "slow")`, a `SLOW:` prefix, and a one-second
interval. Add a Paddle-routing assertion using one stable key.

- [x] **Step 2: Run focused tests and verify RED**

Expected: helper attribute/call failures.

- [x] **Step 3: Implement the timing helper and update worker call sites**

Add safe numeric normalization. Route capture timing with 50 ms MSS / 250 ms
pyautogui thresholds, OCR cache-hit timing with 50 ms, PaddleOCR timing with
500 ms, and Paddle routing with a five-second stable key. Preserve metric
recording and all functional branches.

- [x] **Step 4: Run focused tests and verify GREEN**

Run the Task 1 class plus existing capture/cache regression classes.

### Task 3: Remove Repeated Paddle OCR Text From Success Logs

**Files:**
- Modify: `tests/test_paddle_ocr_backend.py`
- Modify: `paddle_ocr_backend.py`

- [x] **Step 1: Add a failing fast-path diagnostic test**

Exercise a successful multi-line subtitle result, patch
`log_debug_coalesced`, and require one stable key plus a message containing
line and character counts but not the recognized text.

- [x] **Step 2: Run the focused test and verify RED**

```powershell
py -m unittest tests.test_paddle_ocr_backend.PaddleOCRBackendTests.test_subtitle_fast_path_success_log_is_coalesced_and_content_free -v
```

Expected: coalesced logger patch/call failure.

- [x] **Step 3: Coalesce successful recognition only**

Import `log_debug_coalesced`, log line/character counts on a five-second key,
and keep all filter/fallback/error diagnostics immediate.

- [x] **Step 4: Run PaddleOCR tests and verify GREEN**

```powershell
py -m unittest tests.test_paddle_ocr_backend -v
```

### Task 4: Regression, Review, And Handoff

**Files:**
- Create: `.codex/handoffs/2026-07-10_18-05-00.md`

- [x] **Step 1: Run targeted modules**

```powershell
py -m unittest tests.test_latency_optimization tests.test_paddle_ocr_backend tests.test_runtime_logging
```

- [x] **Step 2: Run full verification**

```powershell
py -m unittest discover -s tests
py -m unittest discover
py -B -m py_compile ocr_utils.py paddle_ocr_backend.py worker_threads.py tests\test_latency_optimization.py tests\test_paddle_ocr_backend.py
py -m compileall -q ocr_utils.py paddle_ocr_backend.py worker_threads.py tests
git diff --check
```

- [x] **Step 3: Review only the backup-relative diff**

Confirm success logs alone moved to coalescing, slow keys remain separately
visible, Paddle success text is absent, error/state logs remain immediate, and
no capture/OCR/cache/metric behavior changed.

- [x] **Step 4: Write the handoff**

Record the 858-line evidence, backup path, RED/GREEN results, final counts,
design decisions, live-validation status, and next optimization candidate.
