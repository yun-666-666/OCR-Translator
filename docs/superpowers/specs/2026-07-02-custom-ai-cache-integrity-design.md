# Custom AI Cache Integrity and Deferred Persistence Design

## Context

The Custom AI translation path already has an in-memory LRU cache with JSON persistence. Its identity currently includes profile, URL, model, prompt, line-break mode, and context, but omits two request-shaping profile fields: `wire_api` and `reasoning_effort`. Changing either field can therefore reuse a translation or an in-flight request created under different semantics.

Every successful cache store also serializes and atomically replaces the entire persistence file while the cache lock is held. The current file is small, but this work grows linearly with cache size and delays return of a completed translation before it can be displayed.

## Goals

- Isolate cached and in-flight translations by every supported Custom AI profile field that changes request semantics.
- Remove persistence file I/O from the synchronous translation completion path.
- Preserve atomic JSON replacement, bounded LRU behavior, thread safety, explicit clearing, and shutdown durability.
- Invalidate legacy persisted cache entries once because their keys cannot prove the new identity contract.
- Keep API keys and other secrets out of cache keys, logs, and Git.

## Non-goals

- No UI or configuration controls.
- No changes to OCR scheduling, translation rate limits, provider payloads, prompts, or context-window behavior.
- No database dependency or background worker pool.
- No attempt to migrate legacy persisted entries whose missing identity fields cannot be reconstructed safely.

## Cache Identity

The Custom AI parameter payload used by `UnifiedTranslationCache` will include:

- profile ID;
- normalized base URL;
- model;
- normalized wire API (`chat_completions` or `responses`);
- normalized reasoning effort;
- custom prompt;
- line-break mode;
- current context tuple.

`TranslationHandler.get_inflight_translation_key()` will derive its identity from the same parameter payload. The API key is deliberately excluded: rotating a credential without changing request semantics must not invalidate correct cached translations, and secrets must never be persisted.

## Deferred Persistence

`UnifiedTranslationCache.store()` will update memory immediately, mark persistence dirty, and schedule one daemon `threading.Timer`. Repeated stores before the delay expires will reuse that scheduled flush rather than create more timers.

The flush operation will:

1. take a consistent cache snapshot under the cache lock;
2. release the cache lock before JSON serialization and file I/O;
3. serialize writes through a dedicated persistence lock;
4. write to the existing `.tmp` sibling and atomically replace the destination;
5. retain the dirty state and schedule another flush if a store occurred while the snapshot was being written.

The default delay will be short enough to coalesce subtitle bursts without materially increasing crash-loss exposure. `flush()` will synchronously persist the newest snapshot. `close()` will cancel a pending timer, flush synchronously, and reject future scheduling. `TranslationHandler.close()` will close the cache before closing the HTTP provider.

`clear_all()` and `clear_provider()` will cancel or supersede pending stale writes and synchronously persist the cleared state so a delayed timer cannot resurrect deleted entries.

## Persistence Format

The JSON payload will gain a cache schema version. Files without the current version will be ignored and replaced on the next successful store or explicit flush. Corrupt or unreadable files will continue to fail open: the application starts with an empty memory cache and records a sanitized debug message.

## Git Initialization

The repository will use branch `main` and the already configured global Git identity. Before the initial snapshot:

- tests and `custom_prompt.txt` must no longer be excluded by broad ignore patterns;
- `custom_ai_profiles.json`, runtime configuration, logs, translation cache files, and `.codex/backups/` must be ignored;
- ignored sensitive/runtime paths will be checked explicitly before staging;
- the initial commit will capture the current source baseline and the approved design, giving the cache implementation a clean diff.

## Testing

TDD coverage will prove:

- cache misses across wire API changes;
- cache misses across reasoning-effort changes;
- in-flight keys change for the same two fields;
- `store()` does not synchronously create or rewrite the persistence file;
- multiple stores are coalesced and a forced flush contains the newest entries;
- `close()` flushes pending data;
- clear operations cannot be undone by an older delayed write;
- legacy schema entries are ignored;
- existing persistence restore, cache clearing, prompt/context isolation, and handler behavior remain intact.

Final verification will run focused cache tests, the complete `tests/` discovery suite, root discovery, compile checks, a Git ignored-file audit, and a source-mode startup/shutdown smoke test.
