# Instant Cache Pending Invalidation Design

## Goal

Make an instant translation-cache hit authoritative over older queued work so a
stale timer cannot later send an unnecessary request or display obsolete text.

## Current problem

`start_async_translation()` checks the cache before duplicate, cooldown,
interval, and concurrency gates. On a hit it displays immediately and returns.

However, an older request may already be stored in:

- `pending_translation_request`;
- `pending_translation_flush_scheduled`;
- `pending_translation_flush_deadline_monotonic`;
- a scheduled callback carrying
  `pending_translation_flush_generation`.

The cache-hit path currently leaves that state untouched. The old callback can
later call `start_async_translation()` for obsolete text, consuming network and
potentially competing with newer work.

## Chosen behavior

Add one scheduler helper that atomically invalidates pending translation state:

1. increment `pending_translation_flush_generation`;
2. set `pending_translation_request` to `None`;
3. clear the scheduled flag;
4. clear the deadline;
5. log the reason only when pending/scheduled state existed.

The old GUI timer does not need to be canceled by ID. Its captured generation
will no longer match, so `_flush_pending_translation_request()` will ignore it.

Call this helper immediately after a valid instant cache hit and before
displaying/returning.

## Sequence behavior

The cache hit continues to:

- increment `translation_sequence_counter`;
- set `last_displayed_translation_sequence`;
- update the successful translation timestamp.

It will also set `latest_translation_sequence_started` to the cache-hit sequence
so any active stream partial sees the cache display as newer work. Existing
response ordering already prevents an older final response from overwriting it.

## Compatibility

- No pending state: generation still advances only when the helper is explicitly
  called for a cache hit; fields are initialized lazily.
- Stale timer callbacks return without consuming the current pending request.
- Cache misses, cooldown scheduling, and active-call cleanup keep existing
  behavior.
- The current cache-hit request performs no thread-pool submission.

## Files

- `worker_threads.py`
- `tests/test_latency_optimization.py`

## Verification

- cache hit clears all older pending fields;
- generation advances and invalidates the captured old callback;
- no network/thread-pool submission occurs;
- displayed sequence is also the latest started sequence;
- old active final/partial ordering remains protected;
- existing pending timer and cache-display tests remain green.
