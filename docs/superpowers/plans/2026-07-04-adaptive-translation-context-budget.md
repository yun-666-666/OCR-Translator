# Adaptive Translation Context Budget Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans
> to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Reduce translation prompt size by adapting historical context budget
to the current source length while preserving recent paired context.

**Architecture:** `TranslationHandler` keeps its existing context window and
cache API. A new pure budget helper computes the allowance, and the existing
newest-first selector consumes that request-specific budget.

**Tech Stack:** Python, unittest, existing Custom AI handler/cache

---

### Task 1: Lock the adaptive budget contract

**Files:**
- Modify: `tests/test_custom_ai.py`

- [x] Add formula tests for empty, short, medium, and long current source.
- [x] Add short-source test proving several recent pairs remain.
- [x] Add long-source test proving context shrinks to 600 characters.
- [x] Add oversized-newest-entry test proving balanced truncation.
- [x] Run focused tests and confirm RED.

### Task 2: Implement adaptive selection

**Files:**
- Modify: `handlers/translation_handler.py`

- [x] Replace the fixed 4000-character constant with max/min/penalty constants.
- [x] Add `_get_custom_context_char_budget(current_source=None)`.
- [x] Use the computed budget in `_get_custom_context_for_request()`.
- [x] Keep current-source exclusion, recent-first selection, pair truncation,
  and chronological return order unchanged.
- [x] Run focused tests and confirm GREEN.

### Task 3: Regression verification

- [x] Run immediate-repeat cache and current-source exclusion tests.
- [x] Run `python -m unittest tests.test_custom_ai -q`.
- [x] Run `python -m unittest tests.test_latency_optimization -q`.
- [x] Run `python -m unittest discover -s tests -q`.
- [x] Run `python -m unittest discover -q`.
- [x] Run `python -m compileall -q handlers tests`.
- [x] Run `git diff --check`.

### Task 4: Delivery

- [x] Create the required handoff.
- [ ] Stage only this adaptive-context optimization.
- [ ] Commit and push `codex/optimized-sync-2026-07-02`.
