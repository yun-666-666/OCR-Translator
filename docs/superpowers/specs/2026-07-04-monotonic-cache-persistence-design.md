# Monotonic Cache Persistence Design

## Problem

Persistence operations are captured under the cache-state lock and later
serialized by a separate SQLite lock. A timer, explicit flush, close, or clear
can therefore capture different generations and race to acquire the SQLite
lock.

Serialization alone does not guarantee generation order. If generation 2
writes first and generation 1 writes second, the database is rolled back to
older translations. A stale success callback that runs after generation 2 was
already finalized can also mark the cache dirty again and schedule an
unnecessary full rewrite.

## Goals

- Make successful on-disk generations monotonically non-decreasing.
- Skip stale writes before they touch SQLite.
- Do not let stale completion callbacks re-dirty a newer finalized state.
- Do not advance the write watermark when a newer operation fails.
- Preserve retry, close, clear, migration, and coalescing behavior.

## Design

Track `_last_applied_persistence_generation` under `_persistence_lock`.
Initialize it to `-1` so generation-zero migration and trimming writes remain
eligible.

Before applying an operation:

1. Acquire `_persistence_lock`.
2. If its generation is lower than the last successfully applied generation,
   treat it as an idempotent success without writing.
3. Otherwise execute the full snapshot or delta.
4. Advance the applied watermark only when the write succeeds.

In locked result handling, a successful operation whose generation is already
at or below `_persisted_generation` is stale completion bookkeeping. Reset the
failure backoff but do not finalize it again or mark a full resync.

Equal generations remain eligible at the SQLite layer because migration and
idempotent duplicate operations are safe. Their result callbacks are still
deduplicated once the generation has been finalized.

## Verification

- Apply a newer real SQLite delta followed by an older delta and verify restart
  restores the newer value.
- Finalize an older success after a newer generation and verify it does not
  request a full resync.
- Fail a newer write, then apply an older retry and verify failure did not
  advance the watermark.
- Run all existing persistence and translation suites.
