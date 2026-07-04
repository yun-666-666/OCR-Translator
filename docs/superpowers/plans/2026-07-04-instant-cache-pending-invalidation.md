# Instant Cache Pending Invalidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans
> to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Invalidate older queued translation work whenever the newest subtitle
is satisfied by the instant translation cache.

**Architecture:** Add one state helper in `worker_threads.py`. The cache-hit path
uses it before advancing/displaying the cached sequence; generation checks make
already-scheduled callbacks harmless.

**Tech Stack:** Python, unittest, existing async translation scheduler

---

### Task 1: Reproduce stale pending work

**Files:**
- Modify: `tests/test_latency_optimization.py`

- [x] Add a cache-hit test with an older pending request and generation.
- [x] Assert all pending state is cleared and generation advances.
- [x] Invoke the captured stale callback and prove it submits nothing.
- [x] Assert no thread-pool work is submitted.
- [x] Assert cache sequence becomes latest started/displayed.
- [x] Run focused test and confirm RED.

### Task 2: Implement authoritative cache-hit state

**Files:**
- Modify: `worker_threads.py`

- [x] Add `_invalidate_pending_translation_request(app, reason)`.
- [x] Lazily initialize and advance generation.
- [x] Clear request, scheduled flag, and deadline together.
- [x] Call it on valid instant cache hits.
- [x] Advance `latest_translation_sequence_started` with the cache sequence.
- [x] Run focused tests and confirm GREEN.

### Task 3: Regression verification

- [x] Run pending-timer, cache-display, duplicate, and stale-response tests.
- [x] Run `python -m unittest tests.test_latency_optimization -q`.
- [x] Run `python -m unittest tests.test_custom_ai -q`.
- [x] Run `python -m unittest discover -s tests -q`.
- [x] Run `python -m unittest discover -q`.
- [x] Run `python -m compileall -q worker_threads.py tests`.
- [x] Run `git diff --check`.

### Task 4: Delivery

- [x] Create the required handoff.
- [ ] Stage only this scheduler optimization.
- [ ] Commit and push `codex/optimized-sync-2026-07-02`.
