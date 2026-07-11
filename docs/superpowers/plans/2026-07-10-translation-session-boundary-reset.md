# Translation Session Boundary Reset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent a pending translation candidate or timer from a stopped session from entering a newly started session.

**Architecture:** Centralize scheduler lifecycle cleanup in `worker_threads.reset_translation_scheduler_session_state`. Reuse existing pending/profile generations so already-scheduled Tk callbacks become stale, and invoke the reset at both stop completion and start preparation without resetting response-order sequence counters.

**Tech Stack:** Python 3, Tkinter lifecycle methods, `unittest`, `unittest.mock`.

---

### Task 1: Lock the scheduler reset contract with a failing test

**Files:**
- Modify: `tests/test_latency_optimization.py`
- Modify: `worker_threads.py`

- [x] **Step 1: Write the failing scheduler-state test**

Add a test that constructs an app with old pending/latest candidates, pending
and profile generations, a submit timestamp, and completed-session in-flight
state, then calls the wished-for API:

```python
worker_threads.reset_translation_scheduler_session_state(app, "new session")

self.assertIsNone(app.pending_translation_request)
self.assertFalse(app.pending_translation_flush_scheduled)
self.assertEqual(app.pending_translation_flush_deadline_monotonic, 0.0)
self.assertEqual(app.pending_translation_flush_generation, 8)
self.assertIsNone(app.latest_translation_candidate)
self.assertEqual(app.translation_profile_refresh_generation, 4)
self.assertEqual(app.last_translation_submit_monotonic, 0.0)
self.assertEqual(app.active_translation_inflight_keys, set())
self.assertEqual(app.active_translation_started_monotonic, {})
```

- [x] **Step 2: Run the test to verify RED**

Run:

```powershell
py -m unittest tests.test_latency_optimization.TranslationSessionBoundaryTests.test_scheduler_session_reset_invalidates_previous_session_state -v
```

Expected: `ERROR` because
`reset_translation_scheduler_session_state` does not exist.

- [x] **Step 3: Implement the minimal reset helper**

Add a public helper next to `_invalidate_pending_translation_request`:

```python
def reset_translation_scheduler_session_state(app, reason):
    _invalidate_pending_translation_request(app, reason)
    app.latest_translation_candidate = None
    app.translation_profile_refresh_generation = int(
        getattr(app, "translation_profile_refresh_generation", 0) or 0
    ) + 1
    app.last_translation_submit_monotonic = 0.0
    for name in (
        "active_translation_inflight_keys",
        "active_translation_started_monotonic",
    ):
        state = getattr(app, name, None)
        if hasattr(state, "clear"):
            state.clear()
```

- [x] **Step 4: Run the test to verify GREEN**

Run the same targeted command. Expected: one passing test.

### Task 2: Wire both application lifecycle boundaries

**Files:**
- Modify: `tests/test_latency_optimization.py`
- Modify: `app_logic.py`

- [x] **Step 1: Write failing stop/start boundary tests**

Extend the shutdown fixture with old scheduler state and assert it is reset.
Add a focused app-method test that invokes a new private lifecycle wrapper and
asserts it delegates to the worker helper:

```python
with patch.object(
    worker_threads,
    "reset_translation_scheduler_session_state",
) as reset_state:
    app._reset_translation_scheduler_session_state("translation starting")
reset_state.assert_called_once_with(app, "translation starting")
```

- [x] **Step 2: Run both tests to verify RED**

Run:

```powershell
py -m unittest tests.test_latency_optimization.LatencyShutdownTests.test_finalize_shutdown_is_idempotent_and_skips_dead_widgets tests.test_latency_optimization.TranslationSessionBoundaryTests.test_app_delegates_translation_scheduler_session_reset -v
```

Expected: failure because app lifecycle code does not call/delegate the reset.

- [x] **Step 3: Add and invoke the lifecycle wrapper**

Add this lazy-import wrapper to `GameChangingTranslator`:

```python
def _reset_translation_scheduler_session_state(self, reason):
    from worker_threads import reset_translation_scheduler_session_state

    reset_translation_scheduler_session_state(self, reason)
```

Call it in `_finalize_shutdown()` after clearing queues and in the successful
start path after clearing queues but before setting `is_running = True`.

- [x] **Step 4: Run both tests to verify GREEN**

Run the same two-test command. Expected: two passing tests.

### Task 3: Verify the complete increment

**Files:**
- Modify: `docs/superpowers/plans/2026-07-10-translation-session-boundary-reset.md`

- [x] **Step 1: Run focused regression suites**

```powershell
py -m unittest tests.test_latency_optimization tests.test_custom_ai -q
```

Expected: all tests pass.

- [x] **Step 2: Run the full suite and compile checks**

```powershell
py -m unittest discover -s tests
py -m unittest discover
py -B -m py_compile app_logic.py worker_threads.py tests\test_latency_optimization.py
py -m compileall -q app_logic.py worker_threads.py tests
```

Expected: all commands exit 0.

- [x] **Step 3: Check the patch**

```powershell
git diff --check
```

Expected: no output and exit 0.

- [x] **Step 4: Mark this plan complete**

Change each checkbox from `[ ]` to `[x]` only after its command has produced
the expected evidence.

### Review follow-up: Reject already-started old-session responses

- [x] Add a failing regression showing the boundary reset must advance
  `last_displayed_translation_sequence` to the highest sequence that started in
  the old session.
- [x] Implement the sequence floor without resetting the monotonic translation
  counter.
- [x] Prove `process_translation_response()` discards that old sequence and
  does not update the display.

### Review follow-up: Preserve timed-out request ownership

- [x] Add a failing regression for a provider request that remains active after
  the shutdown wait timeout.
- [x] Preserve its in-flight deduplication identity across restart and prune
  only stale sequence timestamps that no longer belong to an active call.
- [x] Prove completed-session state still clears while timed-out active state is
  retained until the worker's normal cleanup runs.
