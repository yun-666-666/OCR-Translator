# Translation Terminal Integrity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans
> to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Prevent explicitly failed or incomplete translation responses from
being returned and cached as successful text.

**Architecture:** Preserve Responses failure terminal metadata, then run both
wire formats through one final translation-state validator after the existing
single truncation-repair attempt.

**Tech Stack:** Python, SSE parsing, unittest

---

### Task 1: Reproduce terminal integrity failures

**Files:**
- Modify: `tests/test_custom_ai.py`

- [x] Add persistent Chat and Responses truncation tests.
- [x] Add Chat content-filter termination test.
- [x] Add Responses non-token incomplete-reason test.
- [x] Add streaming `response.failed` partial-output test.
- [x] Run focused tests and confirm RED.

### Task 2: Enforce successful translation terminal state

**Files:**
- Modify: `custom_ai.py`

- [x] Add translation terminal-state validator.
- [x] Validate after the optional repair and before text normalization.
- [x] Preserve `response.failed` status and structured error from SSE.
- [x] Infer failed status from an explicit failed event when omitted.
- [x] Keep metadata-free compatibility responses accepted.
- [x] Run focused tests and confirm GREEN.

### Task 3: Regression verification

- [x] Run truncation, stream, error, callback, and relay-compatibility tests.
- [x] Run `python -m unittest tests.test_custom_ai -q`.
- [x] Run `python -m unittest tests.test_latency_optimization -q`.
- [x] Run `python -m unittest discover -s tests -q`.
- [x] Run `python -m unittest discover -q`.
- [x] Run `python -m compileall -q custom_ai.py tests`.
- [x] Run `git diff --check`.

### Task 4: Delivery

- [x] Create the required handoff.
- [ ] Stage only this terminal-integrity optimization.
- [ ] Commit and push `codex/optimized-sync-2026-07-02`.
