# Translation Context Epoch Design

## Problem

Language, model, settings, and session changes clear approved translation
context. Provider requests that started before the clear can still finish
afterward and call `_update_custom_context()`, repopulating the new session with
old-language or old-model translations.

The same race can occur during a cache lookup if a clear happens between
identity construction and the hit update.

## Goals

- Make every context clear invalidate all already-started context writes.
- Cover provider results and cache hits.
- Keep translation results available for normal display/cache behavior; only
  reject their obsolete context side effect.
- Preserve sequence ordering inside the active context generation.
- Avoid holding the context lock during network or SQLite work.

## Design

Maintain `_custom_context_generation` under the existing reentrant context
lock. Every clear increments it while resetting entries and order metadata.

At the start of `_custom_ai_translate()`, capture the current generation. Pass
that value to both its cache-hit path and final provider-result context update.

`_get_custom_ai_cached_translation()` captures the current generation itself
when no caller-provided value exists, before performing the cache lookup.

`_update_custom_context()` accepts an optional expected generation. Under the
lock, it ignores the update if the expected value differs from the current
generation. Direct legacy callers without an expected generation retain their
existing behavior.

## Verification

- A direct stale-generation update after clear is ignored while a current one
  succeeds.
- A blocked provider result released after clear does not reinsert context.
- A cache hit whose lookup triggers a concurrent clear returns its translation
  but does not restore old context.
- Existing sequence ordering, cache reuse, context budgets, and concurrency
  tests remain green.
