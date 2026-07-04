# Persistent Cache Access Recency Design

## Goal

Preserve hot translation-cache entries across application restarts without
turning every cache hit into a disk write.

## Current problem

`UnifiedTranslationCache.get()` updates `_access_times` only in memory. SQLite
continues storing the entry's old access time, usually its store/load time.

After restart, loading with a smaller/full cache or later eviction can therefore
discard a frequently used translation while retaining a colder entry whose
persisted timestamp happens to be newer. This lowers future cache hit rate and
adds avoidable provider calls.

## Options considered

### 1. Persist every hit

Accurate but creates high-frequency SQLite writes during repeated subtitles.
Rejected.

### 2. Persist access times only at close

No runtime write overhead, but abnormal exits lose all recency updates.

### 3. Throttled incremental persistence plus close-time flush

Recommended. Each key requests at most one access-time persistence update per
60 seconds; updates share the existing delayed/coalesced SQLite delta writer.
Normal close batches any newer access times not yet requested.

## State

Add:

- `ACCESS_TIME_PERSIST_INTERVAL_SECONDS = 60.0`;
- `_last_access_persist_request_times`, keyed by cache key.

The map records the access timestamp most recently queued for persistence, not
necessarily the last successful database transaction. If a write fails, dirty
upsert state remains and a later interval can schedule another attempt.

## Hit behavior

On a cache hit:

1. update `_access_times` immediately;
2. if at least 60 seconds passed since this key's last persistence request:
   - increment persistence generation;
   - mark the existing translation/access pair as a dirty upsert;
   - update the last-request map;
   - schedule the existing coalesced persistence timer;
3. otherwise perform no persistence scheduling.

The returned translation remains immediate; no SQLite operation occurs on the
calling thread.

## Close behavior

Before setting `_closed`, compare each live `_access_times` value with its last
requested value. Add newer entries to the dirty upsert batch, advance generation
once for the batch, then use the existing synchronous close flush.

At most `max_size` rows are upserted in one close transaction.

## Lifecycle integration

- Loaded legacy/SQLite access times seed the last-request map.
- Changed stores seed it because the normal dirty upsert already includes the
  current access time.
- Entry removal/eviction clears it.
- Full restore clears and rebuilds it.
- Provider/all clears keep existing persistence behavior.

## Files

- `unified_translation_cache.py`
- `tests/test_custom_ai.py`

## Verification

- a recently hit older entry survives restart/max-size trimming over a colder
  newer entry;
- repeated hits inside 60 seconds do not advance generation or schedule a timer;
- the first hit after 60 seconds schedules one delta;
- close persists a recent hit even when it was inside the throttle interval;
- clear, eviction, store-after-close, migration, and deferred persistence tests
  remain green.
