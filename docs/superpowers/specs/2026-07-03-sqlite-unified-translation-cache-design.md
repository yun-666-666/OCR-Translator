# SQLite Unified Translation Cache Design

## Goal

Upgrade `UnifiedTranslationCache` persistence from whole-file JSON snapshots to an incremental SQLite-backed store while keeping the existing in-memory cache behavior, public API, cache-key isolation rules, and `TranslationHandler` call sites stable.

## Why this change

The recent optimization work already reduced in-memory cache overhead. The remaining heavy path is persistence:

- JSON persistence still rebuilds a full snapshot and rewrites the whole cache file.
- Provider clears and large cache flushes pay serialization cost proportional to the entire cache size.
- Future cache growth is still constrained by whole-file rewrite behavior rather than the in-memory cache itself.

SQLite lets us preserve the fast in-memory cache while making persistence incremental, indexed, and safer for larger caches.

## Options considered

### 1. Recommended: SQLite persistence behind the existing cache API

- Keep `UnifiedTranslationCache.get/store/flush/close/clear_*` as the public surface.
- Replace JSON persistence internals with SQLite row storage.
- Continue using delayed/coalesced persistence for ordinary `store()` bursts.
- Use SQL deletes/upserts for provider clears and normal flushes.

Pros:

- Meaningful structural optimization instead of another micro-tweak
- No new runtime dependency; uses Python built-in `sqlite3`
- Natural fit for provider clears, incremental updates, and future cache growth

Cons:

- More code than the current JSON file approach
- Requires migration handling from the old on-disk cache

### 2. JSON journal plus periodic compaction

Pros:

- Keeps JSON on disk
- Smaller conceptual change than SQLite

Cons:

- Recreates transaction, corruption-recovery, and compaction logic manually
- Harder to keep correct than using a real embedded database

### 3. Leave persistence alone and focus on worker-thread scheduling

Pros:

- Could reduce wasted calls in some OCR/translation timing scenarios

Cons:

- Higher behavior risk in the real-time pipeline
- Does not address the remaining cache persistence bottleneck

## Approved direction

Proceed with option 1: SQLite-backed persistence behind the existing `UnifiedTranslationCache` API.

## Scope

### In scope

- `UnifiedTranslationCache` persistence backend rewrite to SQLite
- Lazy database creation
- Incremental persistence for changed rows instead of whole-cache JSON rewrites
- Legacy JSON cache migration into SQLite
- Preserving cache-key isolation behavior, LRU behavior, and delayed persistence semantics
- Updating app default cache path to a SQLite filename
- Regression and migration tests

### Out of scope

- Rewriting translation scheduling or OCR pipelines
- Removing the in-memory cache layer
- Changing translation provider semantics
- UI changes

## Design

### Public behavior that must stay the same

- `get()` still returns cached translations and refreshes in-memory access time.
- `store()` still updates the in-memory cache immediately.
- Repeated `store()` of the same translation remains a persistence no-op except for LRU/access-time refresh.
- `flush()` still forces persistence.
- `close()` still flushes pending persistence and stops future timer scheduling.
- `clear_provider()` and `clear_all()` still update the in-memory cache immediately.
- Cache-key isolation by provider/profile/url/model/wire API/reasoning/custom prompt/context/linebreak mode must remain intact.

### New persistence model

`UnifiedTranslationCache` remains the owner of the in-memory dictionaries:

- `_cache`
- `_access_times`
- `_provider_keys`

SQLite becomes the persistence backend for those entries.

Each cache row stores:

- `text_hash`
- `source_lang`
- `target_lang`
- `provider`
- `params_hash`
- `translation`
- `access_time`

Composite primary key:

- `(text_hash, source_lang, target_lang, provider, params_hash)`

This preserves current key semantics without changing call sites.

### Dirty-state tracking

Instead of snapshotting the whole cache on every flush, the cache tracks persistence deltas:

- dirty upserts for changed/new entries
- dirty provider clears
- dirty per-key deletes when needed
- a full-resync flag for `clear_all()` / migration / rare full rebuild paths

Normal `store()` bursts continue to coalesce behind the existing delayed timer, but the eventual flush writes only changed rows.

### Clear behavior

- `clear_provider(provider)` deletes matching in-memory entries immediately and persists via SQL delete for that provider.
- `clear_all()` clears memory immediately and removes the SQLite file when the resulting persisted cache is empty.

### Database lifecycle

- Do not create the SQLite file until the first real persistence write is needed.
- On first DB creation, initialize schema and set pragmatic defaults appropriate for a local embedded cache.
- Use explicit transactions for batched flushes.

### Migration

Production startup path will move from:

- `custom_ai_translation_cache.json`

to:

- `custom_ai_translation_cache.sqlite3`

Migration rules:

1. If SQLite DB exists and is valid, load it.
2. Else if a sibling legacy JSON cache exists with the old schema payload, load it into memory, trim to `max_size`, write it into SQLite, and remove the legacy JSON file after successful migration.
3. Invalid/legacy-without-schema JSON should still be ignored rather than crashing startup.

## Files expected to change

- `unified_translation_cache.py`
- `tests/test_custom_ai.py`
- `app_logic.py`

Possible touch only if needed:

- `handlers/translation_handler.py`

## Testing strategy

- Red/green tests for SQLite persistence creation and reload
- Red/green tests for JSON-to-SQLite migration
- Existing regression coverage for:
  - cache-key isolation
  - deferred persistence
  - repeated same-value stores
  - close flush
  - clear-all non-resurrection
  - handler-level persistent cache reuse

Full verification must include:

- focused cache tests
- full `tests` discovery
- root discovery
- `compileall`
- git diff whitespace checks

## Risks and mitigations

### Risk: behavior drift during persistence rewrite

Mitigation:

- Keep public API unchanged
- Preserve existing in-memory structures and most high-level control flow
- Add migration and persistence regression tests first

### Risk: corrupted or unexpected old cache files

Mitigation:

- Detect format before loading
- Ignore invalid legacy payloads without crashing
- Only delete legacy JSON after successful SQLite migration

### Risk: shutdown or clear semantics regress

Mitigation:

- Keep explicit `flush()` / `close()` contract
- Add tests for close-flush, post-close store, provider clear, and clear-all file removal
