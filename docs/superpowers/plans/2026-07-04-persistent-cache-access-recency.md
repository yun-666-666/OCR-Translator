# Persistent Cache Access Recency Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans
> to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Persist throttled cache-hit recency so hot translations survive
restart eviction while keeping hits non-blocking.

**Architecture:** Cache hits reuse the existing dirty-upsert and delayed SQLite
writer, gated by a per-key 60-second request timestamp. Close batches remaining
recency changes into its normal synchronous flush.

**Tech Stack:** Python, sqlite3, unittest, UnifiedTranslationCache

---

### Task 1: Reproduce stale persisted LRU state

**Files:**
- Modify: `tests/test_custom_ai.py`

- [x] Add restart/max-size test where an older stored entry is hit most recently.
- [x] Add throttle test proving hits inside 60 seconds schedule no persistence.
- [x] Add interval test proving the first later hit schedules one delta.
- [x] Add close-time test proving an unqueued recent hit is persisted.
- [x] Run focused tests and confirm RED.

### Task 2: Implement throttled access persistence

**Files:**
- Modify: `unified_translation_cache.py`

- [x] Add interval constant and last-request state.
- [x] Seed/clear state during load, restore, store, and removal.
- [x] Mark throttled hit upserts without blocking on SQLite.
- [x] Reuse delayed/coalesced persistence scheduling.
- [x] Batch newer unqueued access times before close flush.
- [x] Run focused tests and confirm GREEN.

### Task 3: Regression verification

- [x] Run focused persistence, clear, migration, eviction, and close tests.
- [x] Run `python -m unittest tests.test_custom_ai -q`.
- [x] Run `python -m unittest tests.test_latency_optimization -q`.
- [x] Run `python -m unittest discover -s tests -q`.
- [x] Run `python -m unittest discover -q`.
- [x] Run `python -m compileall -q unified_translation_cache.py tests`.
- [x] Run `git diff --check`.

### Task 4: Delivery

- [x] Create the required handoff.
- [ ] Stage only this persistent-recency optimization.
- [ ] Commit and push `codex/optimized-sync-2026-07-02`.
