# Translation Inactivity Continuity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve the last good subtitle across short OCR gaps while translation work is active or pending, then apply one generation-safe clear after sustained true inactivity.

**Architecture:** `worker_threads.py` owns a small clear-policy boundary: blocker detection, display-epoch identity, deduplicated Tk scheduling, and UI-time revalidation. The existing display manager performs the actual main-thread clear, and local OCR submit identity is reset only after a real clear.

**Tech Stack:** Python 3, Tkinter event loop, existing async translation scheduler, `unittest`, `RuntimeMetrics`.

---

## File structure

- Modify `worker_threads.py`: pending-aware, generation-safe one-shot inactivity clear.
- Modify `tests/test_latency_optimization.py`: policy, race, dedupe, and resubmission regressions.
- Create `.codex/handoffs/2026-07-10_13-45-40.md`: second-increment evidence and remaining work.

### Task 1: Establish baseline and backups

**Files:**
- Back up: `worker_threads.py`
- Back up: `tests/test_latency_optimization.py`

- [ ] **Step 1: Run the focused baseline**

Run: `py -m unittest tests.test_latency_optimization -v`

Expected: exit 0 with the current latency suite count recorded.

- [ ] **Step 2: Back up both existing files**

Copy each file to `.codex/backups/YYYY-MM-DD_HH-mm-ss/` while preserving its workspace-relative path.

- [ ] **Step 3: Verify SHA-256 pairs**

Run `Get-FileHash` for both source/backup pairs. Expected: 2/2 matches.

### Task 2: Add failing inactivity-clear regressions

**Files:**
- Modify: `tests/test_latency_optimization.py`

- [ ] **Step 1: Add a focused test app factory**

```python
def _make_inactivity_clear_app(self):
    scheduled = []
    displayed = []
    metrics = RuntimeMetrics(clock=lambda: 100.0)
    app = types.SimpleNamespace(
        is_running=True,
        previous_text="",
        active_translation_calls=set(),
        active_translation_inflight_keys=set(),
        pending_translation_request=None,
        pending_translation_flush_scheduled=False,
        translation_queue=queue.Queue(),
        ocr_stability_gate=types.SimpleNamespace(has_pending=lambda: False),
        last_successful_translation_time=95.0,
        last_displayed_translation_sequence=7,
        last_local_ocr_submitted_text="Same subtitle",
        last_local_ocr_submitted_norm="same subtitle",
        last_local_ocr_submitted_scope=("scope",),
        root=types.SimpleNamespace(
            after=lambda delay, callback, *args: scheduled.append(
                (delay, callback, args)
            )
        ),
        display_manager=types.SimpleNamespace(
            _update_translation_text_on_main_thread=displayed.append
        ),
        update_translation_text=displayed.append,
        runtime_metrics=metrics,
    )
    return app, scheduled, displayed, metrics
```

- [ ] **Step 2: Add active/pending blocker coverage**

Write subtests proving `_schedule_inactive_translation_clear` returns false and schedules nothing for active calls, inflight identities, pending requests, pending flushes, non-empty legacy queue, pending OCR stability, source text, and a stopped app.

- [ ] **Step 3: Add one-shot and generation-race coverage**

Write tests that prove:

- two schedule attempts for the same epoch create one Tk callback;
- executing it clears exactly once and resets all three local OCR submit fields;
- the `translation_inactivity_clear` metric increments once;
- changing the successful timestamp or displayed sequence before the callback makes it stale and prevents a clear;
- the same displayed epoch cannot be cleared twice.

- [ ] **Step 4: Run the new tests and verify RED**

Run the new test class with `py -m unittest tests.test_latency_optimization.TranslationInactivityClearTests -v`.

Expected: failure because the new scheduling and blocker helpers do not exist.

### Task 3: Implement the generation-safe clear boundary

**Files:**
- Modify: `worker_threads.py:619-670`
- Modify: `worker_threads.py:1391-1413`
- Test: `tests/test_latency_optimization.py`

- [ ] **Step 1: Add display epoch and blocker helpers**

```python
def _translation_display_epoch(app):
    return (
        float(getattr(app, "last_successful_translation_time", 0.0) or 0.0),
        int(getattr(app, "last_displayed_translation_sequence", 0) or 0),
    )


def _translation_clear_blocker(app):
    if getattr(app, "is_running", True) is False:
        return "app stopped"
    if str(getattr(app, "previous_text", "") or "").strip():
        return "source text present"
    if getattr(app, "active_translation_calls", None):
        return "active translation calls"
    if getattr(app, "active_translation_inflight_keys", None):
        return "active translation identities"
    if getattr(app, "pending_translation_request", None) is not None:
        return "pending translation request"
    if getattr(app, "pending_translation_flush_scheduled", False):
        return "pending translation flush"
    if _has_pending_ocr_stability_candidate(app):
        return "pending OCR stability candidate"
    translation_queue = getattr(app, "translation_queue", None)
    if translation_queue is not None:
        try:
            if not translation_queue.empty():
                return "legacy translation queue"
        except Exception:
            pass
    return None
```

- [ ] **Step 2: Add UI-time apply and deduplicated schedule helpers**

The scheduler must reject the initial `(0.0, 0)` epoch, blocker states, an already scheduled epoch, and an already cleared epoch. It stores `translation_inactivity_clear_scheduled_epoch`, calls `root.after(0, ...)`, and rolls the marker back if scheduling fails.

The UI callback clears the scheduled marker, rechecks epoch and blocker, invokes the existing display manager main-thread updater, calls `_clear_local_ocr_submit_state`, stores `translation_inactivity_cleared_epoch`, increments `translation_inactivity_clear`, and logs one applied clear.

- [ ] **Step 3: Reorder and replace the translation-thread clear block**

Observe `last_successful_translation_time` before computing inactivity. For local OCR, call `_schedule_inactive_translation_clear` after the configured timeout. Remove the direct background-thread clear, repeated timer reset, and repeated source-present log.

- [ ] **Step 4: Run the focused new test class and verify GREEN**

Run: `py -m unittest tests.test_latency_optimization.TranslationInactivityClearTests -v`

Expected: all new tests pass.

### Task 4: Verify same-subtitle recovery

**Files:**
- Modify: `tests/test_latency_optimization.py`

- [ ] **Step 1: Add a resubmission regression**

After applying a real inactivity clear, call `_should_skip_local_ocr_resubmit(app, "Same subtitle")` and assert false. This proves the local OCR dedupe identity cannot strand the overlay blank.

- [ ] **Step 2: Run the regression**

Run the exact resubmission test. Expected: pass after Task 3 and fail against the pre-change clear behavior.

### Task 5: Full verification and handoff

**Files:**
- Create: `.codex/handoffs/2026-07-10_13-45-40.md`

- [ ] **Step 1: Run focused and full suites**

```powershell
py -m unittest tests.test_latency_optimization -v
py -m unittest discover -s tests
py -m unittest discover
```

- [ ] **Step 2: Run syntax and whitespace checks**

```powershell
py -B -m py_compile worker_threads.py tests\test_latency_optimization.py
py -m compileall -q worker_threads.py tests\test_latency_optimization.py
git diff --check
```

- [ ] **Step 3: Review requirements against the backup diff**

Confirm that local OCR only is changed, active/pending work blocks clearing, UI-time epoch revalidation exists, a display epoch clears once, local dedupe resets only after a real clear, and no cache/network policy is mixed into this increment.

- [ ] **Step 4: Write the handoff**

Record the 47/47 active-clear evidence, backup path, RED/GREEN results, verification counts, design decisions, live-validation limits, and the next log-volume optimization candidate.
