# Translation Session Boundary Reset Design

## Problem

The 2026-07-10 23:06 runtime session exposed a translation candidate from the
previous stopped session. A relay request entered a 60-second model-scoped
cooldown, the user stopped translation, selected a working profile, and started
again. The first current subtitle translated normally, but completion then
expedited the old pending request. Its timing record showed a 34.203-second
queue age and it briefly replaced the current subtitle.

The scheduler already invalidates superseded timers by generation, but the app
only clears the thread queues and OCR stability state at stop/start boundaries.
It does not invalidate `pending_translation_request`,
`latest_translation_candidate`, the pending timer generation, or profile
refresh callbacks.

## Selected approach

Add one scheduler-state reset operation and call it at both lifecycle
boundaries:

- after graceful stop has completed;
- before a new translation session accepts OCR work.

The operation invalidates the pending timer generation, clears the pending and
latest candidates, invalidates profile-refresh callbacks, resets the submit
interval clock, and clears completed-session in-flight timing identities. It
does not reset monotonically increasing translation sequence counters, because
those counters remain the final defense against late response ordering.

This is preferred over merely clamping queue timing because the stale request
would still be sent and displayed. A new session epoch on every request would
also work, but it would duplicate the generation protections already present
and require a much larger change across request snapshots and callbacks.

## Components

- `worker_threads.reset_translation_scheduler_session_state(app, reason)` owns
  scheduler invalidation and is safe when optional attributes are absent.
- `GameChangingTranslator._finalize_shutdown()` calls the reset after active
  provider calls have drained.
- `GameChangingTranslator.toggle_translation()` calls the reset immediately
  before `is_running` becomes true and worker threads start.

## Error handling and compatibility

The reset is synchronous, local, and performs no network or UI work. It uses
defensive attribute access because tests and older integrations construct
partial app objects. Existing Tk callbacks are not cancelled by identifier;
incrementing their generations makes them harmless when they eventually run.

## Tests

1. A scheduler-unit regression starts with an old pending request, latest
   candidate, timer generation, profile refresh generation, submit timestamp,
   and completed-session timing identity. The reset must clear state and advance
   both generations.
2. A shutdown regression proves `_finalize_shutdown()` invokes the reset.
3. A start-boundary regression exercises the helper through the app method so a
   previous-session pending request cannot be expedited after restart.
4. Run the focused latency suite, the full test suite, compilation checks, and
   `git diff --check`.

## Success criteria

- No stopped-session candidate can be submitted or displayed in a new session.
- Old Tk pending/profile callbacks are generation-stale.
- New-session throttling starts from a clean submit timestamp.
- Existing ordering, cooldown, profile-switch, and concurrency tests remain
  green.
