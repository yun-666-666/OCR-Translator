# Persistent Cache Retry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans
> to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Automatically recover dirty translation-cache persistence after
transient SQLite failures without busy-looping.

**Architecture:** Route scheduled and explicit persistence results through one
locked handler. Failed open-cache writes reuse the single timer with capped
exponential backoff; successful writes reset recovery state.

**Tech Stack:** Python, sqlite3, threading.Timer, unittest

---

### Task 1: Reproduce the persistence recovery gap

**Files:**
- Modify: `tests/test_custom_ai.py`

- [x] Add a scheduled failure then success recovery test.
- [x] Add a capped exponential backoff test.
- [x] Add an explicit flush failure retry test.
- [x] Add a close failure no-retry test.
- [x] Run focused tests and confirm RED.

### Task 2: Implement bounded retry recovery

**Files:**
- Modify: `unified_translation_cache.py`

- [x] Add retry delay constants and consecutive-failure state.
- [x] Allow the existing single-timer scheduler to accept a recovery delay.
- [x] Centralize success/failure result handling under the cache lock.
- [x] Reset backoff after success.
- [x] Preserve dirty state and retry only while open.
- [x] Run focused tests and confirm GREEN.

### Task 3: Regression verification

- [x] Run persistence, close, clear, migration, and eviction tests.
- [x] Run `python -m unittest tests.test_custom_ai -q`.
- [x] Run `python -m unittest tests.test_latency_optimization -q`.
- [x] Run `python -m unittest discover -s tests -q`.
- [x] Run `python -m unittest discover -q`.
- [x] Run `python -m compileall -q unified_translation_cache.py tests`.
- [x] Run `git diff --check`.

### Task 4: Delivery

- [x] Create the required handoff.
- [ ] Stage only this retry optimization.
- [ ] Commit and push `codex/optimized-sync-2026-07-02`.
