# Stale Translation Head-of-Line Bypass Design

## Goal

Keep a slow translation for an obsolete subtitle from blocking the newest
subtitle for its full network duration.

## Current bottleneck

Custom AI translation deliberately has a concurrency limit of one. This protects
relays from request storms, but it also creates head-of-line blocking:

1. subtitle A starts a slow request;
2. OCR advances to subtitle B;
3. B is retained as the latest pending request;
4. B cannot start until A finishes, even though A's eventual result will be
   rejected by sequence ordering.

Historical short-log samples for real profiles have multi-second median latency
and much larger tails, so this wait can dominate end-to-end subtitle latency.

## Chosen approach

Allow one controlled superseding request when the sole active Custom AI request
has exceeded a staleness threshold.

- normal concurrency remains one;
- after 1.5 seconds, the latest subtitle may use one overflow slot;
- hard maximum concurrency remains two;
- `race` mode does not use the overflow slot because one logical translation
  already fans out across multiple same-model profiles;
- while two requests are active, only the newest pending subtitle is retained;
- existing sequence checks continue to prevent obsolete results from replacing
  newer text;
- completed obsolete translations may still populate the cache for later reuse;
- cooldown and minimum submit-interval gates still take precedence.

## State

Add `active_translation_started_monotonic`, keyed by translation sequence.
Submission records the start time and cleanup removes it in the worker's
`finally` block.

The gate computes the oldest active age:

- one active call below threshold: queue until the exact threshold deadline;
- one active call at/above threshold: submit into the overflow slot;
- two or more active calls: keep only the latest pending request and wait.

## Files

- `worker_threads.py`
- `app_logic.py`
- `tests/test_latency_optimization.py`

## Safety constraints

- Never exceed two concurrent Custom AI translations through this bypass.
- Never bypass provider cooldown.
- Never bypass the normal minimum submit interval.
- Never stack the overflow mechanism on top of `race` mode.
- Preserve duplicate in-flight suppression.
- Preserve latest-only pending request semantics.
- Lazily initialize the new state for compatibility with tests and older app
  objects.

## Verification

- red/green tests for threshold bypass, hard cap, precise delayed scheduling,
  and cleanup;
- existing pending-request, duplicate-inflight, sequence-order, and latency
  tests;
- full project discovery, compile, and diff checks.
