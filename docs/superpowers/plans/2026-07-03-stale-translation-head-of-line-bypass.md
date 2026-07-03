# Stale Translation Head-of-Line Bypass Implementation Plan

**Goal:** Let the newest subtitle bypass one stale slow translation without
allowing unbounded Custom AI concurrency.

### Task 1: Lock scheduling behavior with tests

**Files:** `tests/test_latency_optimization.py`

- [x] Add a failing test proving one request older than 1.5 seconds can be
  superseded.
- [x] Add a failing test proving two active requests remain the hard cap.
- [x] Add a failing test proving `race` mode cannot stack outer supersession on
  top of its internal endpoint fan-out.
- [x] Add a failing test proving a younger request is queued until its exact
  supersede deadline.
- [x] Add a failing test proving worker cleanup removes start-time state.
- [x] Run the focused tests and confirm RED for missing behavior.

### Task 2: Implement bounded supersession

**Files:** `worker_threads.py`, `app_logic.py`

- [x] Initialize `active_translation_started_monotonic`.
- [x] Record submission start time and roll it back on submit failure.
- [x] Add staleness-threshold and active-age helpers.
- [x] Permit one overflow submission only when all normal gates pass.
- [x] Queue below-threshold work to the precise threshold deadline.
- [x] Remove start-time state in worker cleanup.
- [x] Run focused tests and confirm GREEN.

### Task 3: Regression verification

- [x] Run `python -m unittest tests.test_latency_optimization -q`.
- [x] Run `python -m unittest tests.test_custom_ai -q`.
- [x] Run `python -m unittest discover -s tests -q`.
- [x] Run `python -m unittest discover -q`.
- [x] Run `python -m compileall -q worker_threads.py app_logic.py tests`.
- [x] Run `git diff --check`.

### Task 4: Delivery

- [x] Create the required `.codex/handoffs/<timestamp>.md`.
- [ ] Review and stage only this optimization.
- [ ] Commit with a focused performance message.
- [ ] Push `codex/optimized-sync-2026-07-02` to GitHub.
