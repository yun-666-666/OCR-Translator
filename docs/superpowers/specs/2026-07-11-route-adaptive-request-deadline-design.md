# Route-Adaptive Request Deadline Design

## Scope

This optimization changes only Custom AI translation request deadlines and the
background worker that consumes an immutable request snapshot. It does not
change the UI, OCR behavior, provider selection, retry count, request payload,
or the configured ten-second upper bound.

## Runtime evidence

The 2026-07-11 14:37:52--14:41:12 application session separated into two
routes:

- relay profile `测试2`: 20 successful calls, P50 5.531s, P90 7.031s, max
  8.515s;
- xAI: 76 successful calls, P50 0.656s, P90 0.844s, max 1.765s, plus two
  read timeouts at 10.0s.

Both timed-out xAI calls completed after six or more newer subtitles had
already displayed. The existing bounded overflow slot preserved visible
progress, but each tail request occupied worker capacity for roughly six
seconds longer than the successful route history justified.

Requests documents that its scalar timeout is a socket inactivity deadline,
not a total response deadline. The implementation therefore treats the new
value as a conservative read deadline and never claims that it caps total wall
time.

## Considered approaches

### Retry read timeouts

Rejected. Retrying after a full ten-second wait increases cost and provider
load while the subtitle is already stale. The existing recovery remains
limited to connection-reset failures and bounded 502/504 recovery.

### Hedge every slow request

Rejected. The codebase already provides explicit `race` mode. Applying hidden
hedging in `safe` mode would duplicate paid requests and could amplify an
upstream incident.

### Route-adaptive conservative deadline

Selected. Reuse the existing route-isolated latency advisor. After at least
eight observations, a single-route non-streaming request may use
`max(4.0s, P90 * 4)`, capped by the existing configured timeout. Slow routes,
sample-poor routes, stream/race requests, and the first request after a route
error keep the full configured timeout.

## State and data flow

`CustomAILatencyModeAdvisor.resolve_request_timeout()` returns a small immutable
decision containing the chosen seconds, reason, P90, and sample count. It reads
the same bounded route-local history used by adaptive latency mode.

`TranslationHandler.get_custom_ai_translation_request_snapshot()` resolves the
deadline after resolving the route and mode, then freezes it in the existing
request snapshot. `commit_custom_ai_latency_mode_snapshot()` publishes the
chosen value and reason through runtime metrics without adding a hot-path log.

`worker_threads.process_translation_async()` validates the frozen value and
passes it to `translate_text_with_timeout()`. Missing, invalid, non-positive,
stream, and race values retain the existing ten-second behavior.

## Safety invariants

- The adaptive deadline never exceeds or raises the existing configured limit.
- Fewer than eight route samples cannot shorten it.
- P90 values at or above 2.5 seconds keep the full ten seconds.
- Any most-recent route error grants the next request a full-time probe.
- Stream and race modes are excluded because their transport and multi-route
  semantics differ.
- Route identity remains secret-safe and bounded by the existing LRU map.
- No retry, duplicate request, provider switch, or UI change is introduced.

## Verification

Tests cover fast, slow, sparse, recent-error, stream, and race decisions;
route-isolated snapshot resolution; immutable deadline propagation into the
worker; invalid snapshot fallback; and existing translation/provider suites.
Full test discovery, syntax compilation, whitespace checks, and backup hash
verification remain required.
