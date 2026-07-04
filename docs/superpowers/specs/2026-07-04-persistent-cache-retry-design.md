# Persistent Cache Retry Design

## Problem

The delayed cache writer preserves dirty state when a SQLite operation fails,
but it only schedules another write after a successful operation or a later
cache mutation. A transient lock or file-system interruption can therefore
leave translations memory-only for the rest of a quiet session.

## Goals

- Retry failed background and explicit flushes automatically while the cache is
  open.
- Avoid tight retry loops and sustained disk pressure during a longer outage.
- Preserve the existing single-timer coalescing behavior.
- Reset retry delay immediately after any successful persistence.
- Never create a background retry after cache close begins.

## Design

Track consecutive persistence failures on the cache instance. The first failed
operation schedules a retry after one second. Later failures use exponential
backoff capped at 30 seconds:

`1, 2, 4, 8, 16, 30, 30, ...`

`_schedule_persistence_locked()` accepts an optional delay. Normal mutations
continue to use the configured coalescing delay; recovery uses the computed
retry delay. Its existing timer guard ensures that a concurrent mutation and a
failed write still leave only one pending timer.

Both the scheduled writer and `flush()` pass their results through the same
locked result handler. On success it resets the failure count and finalizes the
persisted generation. On failure it keeps all dirty state, increments the
failure count, and schedules a retry only when the cache remains open.

`close()` marks the cache closed before calling `flush()`, so its final
synchronous failure remains observable to the caller without creating a timer
that could outlive shutdown.

## Verification

- A scheduled failure requests a one-second retry and a later success persists
  the original generation.
- Repeated failures follow the capped backoff sequence.
- A failed explicit flush retries while open.
- A failed close never schedules background work.
- Existing persistence, migration, eviction, and full translation suites remain
  green.
