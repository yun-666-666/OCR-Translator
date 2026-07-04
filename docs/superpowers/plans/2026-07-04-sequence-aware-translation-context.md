# Sequence-Aware Translation Context Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans
> to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Keep concurrent translation context thread-safe and chronologically
ordered by subtitle request sequence.

**Architecture:** Preserve the existing context entry shape while maintaining
locked internal source-order metadata. Propagate the scheduler's existing
translation sequence into provider and cache-hit updates.

**Tech Stack:** Python, threading.RLock, unittest

---

### Task 1: Reproduce out-of-order context

**Files:**
- Modify: `tests/test_custom_ai.py`

- [x] Add late older-result ordering test.
- [x] Add one-item window stale-displacement test.
- [x] Add stale duplicate-source test.
- [x] Add provider sequence propagation test.
- [x] Add instant-cache planned-sequence test.
- [x] Run focused tests and confirm RED.

### Task 2: Implement sequence-aware context

**Files:**
- Modify: `handlers/translation_handler.py`

- [x] Add context lock, order metadata, and fallback counter.
- [x] Make clear, read, and update operations locked.
- [x] Sort and trim by sequence order.
- [x] Ignore stale same-source results.
- [x] Pass provider translation sequence into cache and success updates.
- [x] Infer instant-cache planned sequence from the app counter.
- [x] Run focused tests and confirm GREEN.

### Task 3: Regression verification

- [x] Run context, cache-hit, race, and concurrency tests.
- [x] Run `python -m unittest tests.test_custom_ai -q`.
- [x] Run `python -m unittest tests.test_latency_optimization -q`.
- [x] Run `python -m unittest discover -s tests -q`.
- [x] Run `python -m unittest discover -q`.
- [x] Run `python -m compileall -q handlers/translation_handler.py tests`.
- [x] Run `git diff --check`.

### Task 4: Delivery

- [x] Create the required handoff.
- [ ] Stage only this context-order optimization.
- [ ] Commit and push `codex/optimized-sync-2026-07-02`.
