# Streaming Truncation Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans
> to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Detect explicitly truncated streamed translations and automatically
retry once without the output limit.

**Architecture:** Preserve terminal completion metadata in the existing stream
parsers so the shared non-streaming truncation detector and retry path work for
Chat Completions and Responses streams too.

**Tech Stack:** Python, SSE parsing, unittest

---

### Task 1: Reproduce missed stream truncation

**Files:**
- Modify: `tests/test_custom_ai.py`

- [x] Add Chat stream length-finish integration test.
- [x] Add Responses incomplete-event integration test.
- [x] Assert the repair request omits the output-limit field.
- [x] Run focused tests and confirm RED.

### Task 2: Preserve terminal stream metadata

**Files:**
- Modify: `custom_ai.py`

- [x] Preserve Chat `finish_reason`.
- [x] Recognize Responses completed and incomplete terminal events.
- [x] Preserve Responses status, incomplete details, and usage.
- [x] Infer incomplete status when a relay omits it.
- [x] Reuse the existing one-retry repair path.
- [x] Run focused tests and confirm GREEN.

### Task 3: Regression verification

- [x] Run stream parsing, callback, and truncation tests.
- [x] Run `python -m unittest tests.test_custom_ai -q`.
- [x] Run `python -m unittest tests.test_latency_optimization -q`.
- [x] Run `python -m unittest discover -s tests -q`.
- [x] Run `python -m unittest discover -q`.
- [x] Run `python -m compileall -q custom_ai.py tests`.
- [x] Run `git diff --check`.

### Task 4: Delivery

- [x] Create the required handoff.
- [ ] Stage only this streaming-quality optimization.
- [ ] Commit and push `codex/optimized-sync-2026-07-02`.
