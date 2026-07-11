# Live Profile Model Switch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a selected relay model take effect immediately and prevent old model cooldown/request state from stalling the latest subtitle.

**Architecture:** Persist explicit model selections in the existing profile manager, then invalidate and re-submit the scheduler's latest candidate with a bounded configuration-change priority. Freeze active profile dictionaries at request construction so in-flight work retains its original cache identity.

**Tech Stack:** Python 3, Tkinter/ttk, existing Custom AI provider/scheduler, `unittest`.

---

### Task 1: Profile model selection contract

**Files:**
- Modify: `gui_builder.py`
- Test: `tests/test_latency_optimization.py`

- [x] Write tests proving an explicit selection persists the selected profile and refreshes only when that profile is active for translation.
- [x] Run the focused tests and confirm they fail because no apply helper or model event binding exists.
- [x] Add a testable apply helper and bind it to `<<ComboboxSelected>>` and Enter.
- [x] Run the focused tests and confirm they pass.

### Task 2: Scheduler wakeup after configuration change

**Files:**
- Modify: `worker_threads.py`
- Test: `tests/test_latency_optimization.py`

- [x] Write tests for latest-candidate tracking, stale pending timer invalidation, zero-delay refresh, immediate bounded overflow, and the hard two-call ceiling.
- [x] Run the focused tests and confirm the new behavior is absent.
- [x] Add the configuration-change refresh path while reusing the existing pending generation and overflow limit.
- [x] Run the focused tests and confirm they pass.

### Task 3: Immutable provider request identity

**Files:**
- Modify: `handlers/translation_handler.py`
- Test: `tests/test_custom_ai.py`

- [x] Write a regression test that mutates the live manager profile during a provider call and expects the original model/cache identity.
- [x] Run it and confirm the live profile reference causes the expected failure.
- [x] Copy the active profile at request/cache snapshot construction.
- [x] Run the focused and full Custom AI tests.

### Task 4: Verification and handoff

**Files:**
- Create: `.codex/handoffs/2026-07-10_22-38-11.md`

- [x] Run targeted suites, full discovery, compile checks, and `git diff --check`.
- [x] Review the diff against the runtime evidence and cost/concurrency limits.
- [x] Record backups, RED/GREEN evidence, commands, decisions, and remaining upstream latency risk in the handoff.
