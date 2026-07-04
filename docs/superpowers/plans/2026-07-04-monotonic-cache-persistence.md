# Monotonic Cache Persistence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans
> to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Prevent stale cache persistence operations and completion callbacks
from rolling back or re-dirtying a newer committed generation.

**Architecture:** Add a successful-write generation watermark protected by the
SQLite lock, and ignore already-finalized generations in cache-state result
handling.

**Tech Stack:** Python, sqlite3, threading, unittest

---

### Task 1: Reproduce persistence generation rollback

**Files:**
- Modify: `tests/test_custom_ai.py`

- [x] Add real SQLite newer-then-older operation test.
- [x] Add stale completion after newer finalization test.
- [x] Add failed-newer then older-retry watermark test.
- [x] Run focused tests and confirm RED.

### Task 2: Enforce monotonic persistence commits

**Files:**
- Modify: `unified_translation_cache.py`

- [x] Add last-successfully-applied generation state.
- [x] Skip older operations under the SQLite lock.
- [x] Advance the watermark only after successful writes.
- [x] Ignore already-finalized success callbacks.
- [x] Preserve retry reset and pending-newer scheduling.
- [x] Run focused tests and confirm GREEN.

### Task 3: Regression verification

- [x] Run persistence, retry, close, migration, and eviction tests.
- [x] Run `python -m unittest tests.test_custom_ai -q`.
- [x] Run `python -m unittest tests.test_latency_optimization -q`.
- [x] Run `python -m unittest discover -s tests -q`.
- [x] Run `python -m unittest discover -q`.
- [x] Run `python -m compileall -q unified_translation_cache.py tests`.
- [x] Run `git diff --check`.

### Task 4: Delivery

- [x] Create the required handoff.
- [ ] Stage only this persistence-ordering optimization.
- [ ] Commit and push `codex/optimized-sync-2026-07-02`.
