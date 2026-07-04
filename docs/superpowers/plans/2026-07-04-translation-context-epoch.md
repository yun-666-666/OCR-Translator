# Translation Context Epoch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans
> to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Prevent provider results and cache hits started before a context clear
from repopulating the new translation session.

**Architecture:** Increment a locked context generation on clear, capture it
before slow work, and validate it at context writeback.

**Tech Stack:** Python, threading.RLock, unittest

---

### Task 1: Reproduce stale context writeback

**Files:**
- Modify: `tests/test_custom_ai.py`

- [x] Add direct stale-generation invalidation test.
- [x] Add blocked provider completion after clear test.
- [x] Add cache lookup concurrent-clear test.
- [x] Run focused tests and confirm RED.

### Task 2: Implement context epoch validation

**Files:**
- Modify: `handlers/translation_handler.py`

- [x] Initialize locked context generation.
- [x] Increment generation on every active-context clear.
- [x] Capture generation before provider/cache work.
- [x] Pass generation to provider and cache-hit context updates.
- [x] Ignore mismatched writebacks under the lock.
- [x] Preserve legacy direct updates without an expected generation.
- [x] Run focused tests and confirm GREEN.

### Task 3: Regression verification

- [x] Run context, cache-hit, session-clear, sequence, and concurrency tests.
- [x] Run `python -m unittest tests.test_custom_ai -q`.
- [x] Run `python -m unittest tests.test_latency_optimization -q`.
- [x] Run `python -m unittest discover -s tests -q`.
- [x] Run `python -m unittest discover -q`.
- [x] Run `python -m compileall -q handlers/translation_handler.py tests`.
- [x] Run `git diff --check`.

### Task 4: Delivery

- [x] Create the required handoff.
- [ ] Stage only this context-epoch optimization.
- [ ] Commit and push `codex/optimized-sync-2026-07-02`.
