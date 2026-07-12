# Route-Adaptive Request Deadline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Release stale tail requests earlier on historically fast Custom AI routes without retrying, duplicating paid calls, or shortening slow and unproven routes.

**Architecture:** Extend the existing route-local latency advisor with a conservative timeout decision. Freeze that decision in the Custom AI request snapshot and make the worker consume the frozen value.

**Tech Stack:** Python 3, Requests, existing Custom AI route state, `unittest`.

---

### Task 1: Prove the deadline policy RED

**Files:**
- Modify: `tests/test_custom_ai.py`

- [ ] Add tests that call `resolve_request_timeout(10.0, latency_mode=...)`.
- [ ] Assert eight 0.6--0.9s observations resolve to 4.0s.
- [ ] Assert fewer than eight samples, P90 >= 2.5s, a recent error, `stream`,
  and `race` all retain 10.0s.
- [ ] Run `py -m unittest tests.test_custom_ai.CustomAILatencyModeAdvisorTests -v`.
  Expected: fail because the deadline API does not exist.

### Task 2: Implement the policy GREEN

**Files:**
- Modify: `custom_ai.py`
- Test: `tests/test_custom_ai.py`

- [ ] Add `CustomAIRequestTimeoutDecision(seconds, reason, p90_seconds,
  sample_count)`.
- [ ] Add `resolve_request_timeout()` under the advisor lock with these exact
  guards: valid configured timeout, non-stream/race mode, at least 8 samples,
  no current consecutive error, then `min(configured, max(4.0, p90 * 4.0))`.
- [ ] Run the focused advisor tests. Expected: pass.

### Task 3: Freeze a route-specific deadline in the snapshot

**Files:**
- Modify: `tests/test_custom_ai.py`
- Modify: `handlers/translation_handler.py`

- [ ] Add a failing route-isolation test: train only the active fast route,
  obtain a snapshot, and assert `timeout_seconds == 4.0`; an untrained or slow
  route must remain 10.0.
- [ ] Resolve the timeout from the active route advisor after latency mode is
  resolved and add `timeout_seconds`, `timeout_reason`, `timeout_p90_seconds`,
  and `timeout_sample_count` to the snapshot.
- [ ] Publish gauges/labels in `commit_custom_ai_latency_mode_snapshot()`.
- [ ] Run the new snapshot test and the full `tests.test_custom_ai` suite.

### Task 4: Consume the immutable deadline in the worker

**Files:**
- Modify: `tests/test_latency_optimization.py`
- Modify: `worker_threads.py`

- [ ] Extend the existing resolved-snapshot worker test with
  `timeout_seconds: 4.0` and record the timeout received by
  `translate_text_with_timeout()`.
- [ ] Run the exact test. Expected: fail because the worker still passes 10.0.
- [ ] Add a helper that accepts only finite positive snapshot values and falls
  back to 10.0; force 10.0 for stream and race modes.
- [ ] Pass the helper result through `translation_kwargs`.
- [ ] Add invalid, missing, stream, and race fallback assertions.
- [ ] Run `py -m unittest tests.test_latency_optimization -v`.

### Task 5: Verify and document

**Files:**
- Create: `.codex/handoffs/YYYY-MM-DD_HH-mm-ss.md`

- [ ] Run `py -m unittest tests.test_custom_ai tests.test_latency_optimization -q`.
- [ ] Run `py -m unittest discover -s tests` and `py -m unittest discover`.
- [ ] Run `py -B -m py_compile custom_ai.py handlers/translation_handler.py worker_threads.py tests/test_custom_ai.py tests/test_latency_optimization.py`.
- [ ] Run `git diff --check` and inspect the focused diff against
  `.codex/backups/2026-07-11_14-52-18/`.
- [ ] Record commands, results, design decisions, backup path, and remaining
  real-network validation in the handoff.
