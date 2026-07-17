# Superseded Custom AI Failover Design

## Goal

Prevent an older live-subtitle translation from starting another Custom AI failover candidate after a newer translation request has already been submitted.

## Evidence

The latest runtime session contains sequence 878, which timed out on xai, continued to the fallback provider, took 9.687 seconds end to end, and was discarded after sequence 880 displayed. Earlier sequence 281 followed a longer stream/retry/failover path, took 49.344 seconds, and was also discarded as stale. The current failover abort hook only treats a request as obsolete after another sequence reaches display, even though the scheduler already records the latest submitted sequence.

## Options

### A. Display-only obsolescence

Keep the existing behavior. This preserves every possible fallback result but allows already-superseded requests to spend another provider call while a newer subtitle is in flight.

### B. Submission-boundary obsolescence

At existing failover boundaries, treat a positive request sequence as obsolete when it is lower than either the latest submitted sequence or the last displayed sequence. Do not interrupt an HTTP request already in progress. This reuses existing state and matches the streaming partial-display policy.

### C. Cooperative cancellation and scheduler redesign

Propagate cancellation through workers and HTTP transports. This has a much larger regression surface and is not justified by the observed failure.

## Decision

Use option B. It is the smallest change that addresses both the current 9.687-second outlier and the historical stale failover chains. Option C and module splitting are explicitly out of scope.

## Behavior

- A request with no usable positive sequence keeps current failover behavior.
- A request remains eligible for fallback when no newer request has started and no equal-or-newer request has displayed.
- Before each candidate and immediately after a provider error, a request stops when `request_sequence < latest_translation_sequence_started`.
- The existing display check remains: `request_sequence <= last_displayed_translation_sequence` is obsolete.
- An in-progress provider HTTP call is never forcibly cancelled by this change.
- Successful stale results may still be cached; display ordering remains enforced by the worker response path.

## Files

- Modify `handlers/translation_requests.py`: extend the existing obsolescence predicate and its diagnostic log.
- Modify `tests/test_custom_ai.py`: add scheduler state to the failover test fixture and add one regression test for a newer started-but-not-displayed sequence.
- Create `docs/cleanup/grok-4.5-log-review-2026-07-17.md`: retain the full evidence and Grok 4.5 review disposition.

## Test Strategy

1. Add a test where sequence 7 fails on the primary provider while `latest_translation_sequence_started` advances to 8 and `last_displayed_translation_sequence` remains 0. Expect no fallback call and `None`.
2. Keep the existing test that allows normal primary-to-fallback behavior when no sequence supersedes the request.
3. Keep the existing displayed-sequence and app-stopped abort tests.
4. Run the focused failover test class, the full Custom AI test module, the latency tests, and then the repository suite.

## Non-goals

- Changing model profiles, user configuration, timeouts, or failover ordering.
- Cancelling in-flight HTTP requests.
- Changing cache retention semantics.
- Refactoring or splitting the large translation handler module.
