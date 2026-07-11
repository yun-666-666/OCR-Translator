# Route-scoped Custom AI adaptation design

## Goal

Prevent latency and prompt-cache observations from one Custom AI route from
changing adaptive behavior for another route after a live profile or model
switch.

## Evidence

The latest runtime session switched from `测试3` to `测试2`. The first route
returned fast HTTP 502 responses while the second completed 26 translations
successfully. `TranslationHandler` currently owns one global latency advisor
and one global prompt-cache EMA, so those unrelated routes share error, p90,
cached-token, and input-token history.

OpenAI's prompt-caching guidance recommends stable cache keys for requests that
share prefixes and monitoring cached-token metrics. That makes route-local
measurement the correct boundary for this application's endpoint and model
switching behavior.

## Design

Each route is identified by a tuple containing:

1. canonical request endpoint;
2. credential fingerprint;
3. normalized wire API;
4. model name.

`TranslationHandler` will keep two LRU-ordered, lock-protected maps keyed by
that identity: one map of `CustomAILatencyModeAdvisor` instances and one map of
prompt-cache EMA records. Each map is capped at 32 entries. Sparse test or
legacy profiles without a usable base URL share a compatibility fallback
bucket.

Adaptive-mode resolution and commit use the request profile's route. Request
snapshots retain the frozen profile identity. Successful race calls record the
winning profile; failures record the frozen attempted profile. Context-budget
calculation reads only the active request profile's cache metrics.

## Compatibility and failure handling

- Explicit `safe`, `stream`, and `race` modes remain unchanged.
- Existing profiles, cache keys, persisted files, and credentials are not
  migrated or rewritten.
- Route-key calculation falls back to a non-sensitive compatibility bucket if
  profile normalization fails.
- Metrics and advisor lookup failures continue to fail open without blocking a
  translation.
- Runtime metrics still show the latest request, but their adaptive state is
  route-local.

## Verification

- A slow route must resolve independently from a fresh healthy route.
- Error hold state must not cross routes.
- Low cache-hit observations must shrink only that route's context budget.
- A result from a frozen request snapshot must update the original route after
  the active profile changes.
- More than 32 route identities must evict least-recently-used state.
- Existing Custom AI, latency, and full test suites must remain green.
