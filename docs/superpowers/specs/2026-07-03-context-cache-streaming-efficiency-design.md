# Context Cache and Streaming Efficiency Design

## Goal

Reduce avoidable Custom AI requests and UI-thread work while keeping translation
context useful and bounded.

## Chosen approach

Use the current source text when selecting request context. Any prior context entry
with that same source is excluded from the request and cache key. This prevents a
subtitle from being supplied as its own history and lets an immediately repeated
subtitle reuse the cache entry produced before it was added to history.

Keep recent source/translation pairs within both the configured item count and a
4,000-character budget. Selection works newest-first and preserves chronological
order in the final prompt. The newest pair is retained with deterministic
truncation if it alone exceeds the budget.

Coalesce streaming partial translations per request. At most one Tk callback is
pending; it displays the newest accumulated partial rather than every token-sized
intermediate value. Existing sequence checks continue to reject stale streams.
When that streamed value already equals the final processed result, keep the final
sequence bookkeeping but skip the duplicate widget update.

## Alternatives considered

- Remove context from cache keys entirely. This maximizes cache hits but may reuse
  a translation produced under materially different context, so it was rejected.
- Disable streaming updates. This reduces UI work but worsens perceived latency,
  so coalescing is preferable.
- Add a tokenizer dependency for exact token budgeting. Character budgeting is
  deterministic, dependency-free, and accurate enough for a defensive ceiling.

## Components

- `handlers/translation_handler.py`: context filtering, budgeting, cache-key
  consistency, and duplicate history refresh.
- `worker_threads.py`: one-pending-callback streaming display coalescer.
- `custom_ai.py`: preserve provider-reported cached input token counts.
- `tests/test_custom_ai.py`: cache reuse and context-budget behavior.
- `tests/test_latency_optimization.py`: streaming coalescing and stale-sequence
  behavior.

Short Custom AI logs include only the numeric cached-input-token count. This makes
prompt-cache effectiveness measurable without logging prompts, keys, or URLs.

## Error handling and compatibility

Legacy string-only context entries remain supported. Invalid context-window values
retain the existing fallback. Streaming callback failures remain diagnostic-only
and cannot terminate the translation request.

## Verification

Each behavior is introduced with a failing regression test. Focused suites run
after each implementation, followed by full discovery, compile checks, and diff
hygiene.
