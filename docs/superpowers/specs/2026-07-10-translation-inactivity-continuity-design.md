# Translation Inactivity Continuity Design

## Scope

This is the second large optimization increment requested on 2026-07-10. It
changes only the local-OCR translation overlay clear policy. Cache-MISS logging
and network timeout recovery remain separate later increments.

## Runtime evidence

During the working xAI session from 13:09 through 13:11, the log contained 43
displayed translations and 47 timeout clears. Replaying the logged active-call
counts shows that all 47 clears occurred while one or two translation calls were
still active.

The current translation thread:

1. computes inactivity from its last observed successful display;
2. clears whenever 'previous_text' is empty after the configured timeout;
3. checks for a newer successful display only after the clear decision;
4. resets its local timer after every clear, allowing another clear every
   timeout period.

It does not inspect active translation calls, pending scheduler work, queued
legacy work, or a pending OCR stability candidate. It also does not clear the
local OCR resubmission identity, unlike the separate API-OCR empty-result path.
A subtitle that disappears, is cleared, and later returns unchanged can
therefore be suppressed as an already submitted local OCR result.

## Considered approaches

### 1. Increase the default clear timeout

This reduces frequency but does not prevent clearing while a replacement is
already in flight, repeated clear calls, or same-subtitle resubmission
suppression. It is rejected.

### 2. Pending-aware, generation-safe one-shot clear

Before scheduling a clear, require a sustained source absence and no active or
pending translation work. Schedule the clear on the Tk event loop and re-check
both the display epoch and blockers immediately before applying it. Record the
cleared epoch so one displayed result can be cleared only once. Reset local OCR
resubmission identity when the clear is applied. This is the selected approach.

### 3. Move all clearing into a new OCR lifecycle controller

A dedicated controller could unify local and API OCR clearing, but it would
touch both pipelines and settings persistence without evidence that the API OCR
path caused the current symptom. It is deferred.

## State and interfaces

'worker_threads.py' adds focused helpers:

- '_translation_display_epoch(app)' returns the last successful display
  timestamp plus last displayed sequence. The pair also changes for visible
  error results whose successful timestamp intentionally does not advance.
- '_translation_clear_blocker(app)' reports whether the app is stopped, source
  text is present, a translation call/inflight identity is active, a pending
  request/flush exists, the legacy translation queue is non-empty, or an OCR
  stability candidate is pending.
- '_schedule_inactive_translation_clear(...)' deduplicates scheduled and already
  applied clears for one display epoch.
- '_apply_inactive_translation_clear(...)' runs on the Tk event loop, re-checks
  epoch and blockers, clears the display immediately on the UI thread, resets
  local OCR submit identity, records a runtime metric, and marks the epoch
  cleared.

The direct UI-thread update uses the existing display manager main-thread
method. A compatibility fallback calls 'app.update_translation_text("")' for
minimal test doubles or legacy app shells without the display manager method.

## Translation-thread order

Each loop first observes a newer 'last_successful_translation_time', then
computes inactivity. This prevents a newly displayed result from being cleared
using the previous loop's stale time.

For local OCR, an inactivity timeout attempts to schedule one guarded clear.
Blockers produce no repeated high-frequency log message. The actual UI callback
logs either one applied clear or a concise stale/deferred diagnostic. Existing
queue consumption and API-OCR behavior are unchanged.

## Concurrency and safety

The scheduled epoch attribute prevents the worker loop from queuing duplicate
Tk callbacks. The UI callback clears that scheduled marker before deciding
whether to apply.

The callback re-reads:

- 'last_successful_translation_time';
- 'last_displayed_translation_sequence';
- source presence;
- all active and pending work indicators.

If any value changed, it does not clear. A translation response callback queued
before the clear therefore updates the epoch first, making the clear stale. A
new request appearing after the worker's initial decision is caught by the
second blocker check.

No lock is added: the state consists of atomic Python attribute/set/dict
observations and the decisive display update occurs on Tk's UI thread.

## Test design

Regression tests in 'tests/test_latency_optimization.py' cover:

1. active, inflight, pending, queued, and OCR-stability work block scheduling;
2. one epoch schedules only one callback and clears only once;
3. a newer successful display or displayed sequence invalidates a scheduled
   clear;
4. an applied clear resets all local OCR resubmission fields;
5. an unchanged subtitle may be submitted again after a real clear;
6. runtime metrics count only applied inactivity clears.

The focused latency suite, full 'tests/' discovery, root discovery, syntax
compilation, and whitespace checks must remain green.

## Out of scope and next review

After this increment, the next evidence-backed candidate is log-volume control:
782 unified-cache MISS lines and 149 duplicate-inflight lines appeared in about
two minutes. Sampling or transition-only logging should be designed without
removing counters needed for performance diagnosis.
