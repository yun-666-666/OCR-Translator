# Log-Driven Translation Freshness and Local OCR Design

## Context

The July 16 live logs show that the latest meaningful PaddleOCR session ran from 00:39:22 to 00:42:40. It started 81 translations, displayed 68, discarded 13 stale responses, used the bounded overflow slot 76 times, queued the latest candidate 586 times, and suppressed 192 duplicate in-flight requests. Translation worker latency was 3.391 seconds at p50 and 9.328 seconds at p95, while the overflow threshold remained fixed at 1.5 seconds.

The same session also showed one request that spent 27.532 seconds traversing failover profiles and was discarded as stale when it finally returned. Local PaddleOCR ran at 0.219 seconds p50 while capture continued from a 200 ms base interval. Exact frame-cache hits were rare, but changing cache identity to a perceptual heuristic would create a subtitle correctness risk.

## Goals

1. Stop an already-obsolete translation from spending more time and money on additional failover profiles.
2. Make the bounded overflow threshold respond conservatively to established route latency instead of treating 1.5 seconds as slow for every route.
3. Prevent local PaddleOCR from being driven faster than its recent sustained processing rate.
4. Preserve chronological display, cache identity, context ordering, cooldown recovery, race mode, and existing profile failover behavior for current requests.
5. Add focused regression tests and keep each optimization independently reviewable.

## Non-goals

- No UI or config-schema changes.
- No provider/network calls during verification.
- No change to translation cache keys or context-window semantics.
- No cancellation of a currently executing HTTP request.
- No perceptual frame-cache reuse until a separate benchmark proves that subtitle changes cannot be mistaken for duplicates.
- No MSS handle-reuse work; the July 15 benchmark did not clear the existing benefit gate.

## Considered approaches

### A. Failover abort only

This is the smallest change and removes the clearest 27-second waste, but it leaves the fixed overflow policy and local OCR saturation unchanged.

### B. Staged freshness and pacing changes (selected)

Implement three bounded increments: stale failover abort, route-aware overflow timing, and local OCR duration-aware pacing. Each increment receives a failing regression test, a minimal implementation, focused verification, and a separate commit. This addresses the strongest evidence without introducing fuzzy image identity.

### C. Add perceptual frame caching now

This may reduce OCR work substantially, but a false match can preserve an old subtitle after the pixels have changed. The current logs do not provide a correctness benchmark for choosing a safe threshold, so this approach is deferred.

## Design

### 1. Freshness-aware failover abort

`TranslationHandler` will expose a private helper that treats a translation sequence as obsolete only when it is no newer than `app.last_displayed_translation_sequence`. Missing, invalid, or test-only sequence values remain current so direct handler calls preserve existing behavior.

The failover loop will check this helper before starting each candidate and immediately after a candidate failure. If the request became obsolete while the failed candidate was running, the handler returns `None`, logs a content-free freshness event, and does not contact another profile. The worker already suppresses `None` responses and still performs its normal active-call cleanup and pending-request expedite path.

Using the last displayed sequence, rather than merely the newest started sequence, preserves an older in-flight request as a reliability fallback until a newer translation has actually been shown.

### 2. Route-aware bounded overflow threshold

The existing 1.5-second threshold remains the fallback and minimum. When the immutable request snapshot contains at least eight route observations and a positive p90, the effective threshold becomes:

`max(configured_threshold, min(4.0 seconds, route_p90 * 0.5))`

This produces approximately 4.0 seconds for the recent luna route and about 2.3 seconds for the recent terra route. The cap still allows recovery before the 10-second request deadline. Race mode continues to prohibit the extra overflow slot, configuration-refresh requests preserve their immediate bounded overflow behavior, and snapshots without sufficient evidence keep the current 1.5-second behavior.

### 3. Local OCR duration-aware pacing

Actual non-cache local OCR work will record a dedicated `local_ocr_duration` timing. When local OCR is selected and the 60-second runtime window contains at least eight samples, adaptive scan control computes a target from recent p50 plus a 25 ms scheduling margin, rounded up to 25 ms. The target is bounded between the user's configured base interval and twice that base interval.

For the observed 0.219-second p50 and 200 ms base, the target becomes 250 ms. API OCR keeps its existing active-call capacity policy. When local measurements are absent, insufficient, or faster than the configured base, the scan interval remains at the user's setting.

This change uses the already thread-safe runtime metrics surface and does not add a user-visible setting.

## Error handling and observability

- Stale failover abort logs sequence and last displayed sequence, never subtitle text or provider secrets.
- Invalid route-latency snapshot values fall back to the configured 1.5-second threshold.
- Missing or malformed runtime metrics leave local OCR pacing unchanged.
- Existing provider cooldowns, sanitized exceptions, and request snapshots remain authoritative.

## Testing

1. Add a failover regression where the primary profile fails after a newer sequence is displayed; assert that no secondary provider is called and the result is `None`.
2. Add a control proving failover still reaches the next provider while the original sequence remains current.
3. Add overflow tests for sufficient route history, insufficient history, the 4-second cap, race mode, and configuration refresh.
4. Add adaptive-scan tests proving 0.219-second local OCR selects 250 ms, insufficient samples preserve 200 ms, API OCR behavior is unchanged, and malformed metrics are ignored.
5. Run focused test classes, the full offline unittest surfaces available in this checkout, Python compilation, and `git diff --check`.

## Success criteria

- A stale request cannot start another failover profile after a newer translation is displayed.
- Established slow routes no longer use the overflow slot at 1.5 seconds solely because of the global default.
- The observed local PaddleOCR timing produces a 250 ms adaptive interval without changing the saved 200 ms preference.
- Existing focused and full offline tests pass with no new secret-bearing logs or UI changes.
