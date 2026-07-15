# AI OCR Saturation Backpressure Design

## Context

The first live run after commit `6653d02` confirms that the previous 60-second cooldown defect no longer reproduces. Custom AI OCR is correctly limited to two concurrent calls, and no cooldown or PaddleOCR fallback was needed during the main 22:51:20-22:53:54 session.

The new bottleneck is local scheduling around a slow upstream OCR route:

- 77 Custom AI OCR requests were started.
- 587 capture cycles reached the two-call limit and were discarded before image conversion.
- 16 OCR responses arrived too late and were discarded.
- OCR latency was 2.672 seconds at p50, 6.281 seconds at p90, and 11.328 seconds at maximum.
- The capture loop nevertheless remained at 200 ms and logged two active calls as "normal".

The mismatch exists because adaptive load detection still uses the historical hard-coded threshold of more than five calls, while Custom AI now has an effective concurrency ceiling of two. The capture thread also discovers saturation only after capturing and queueing a frame, so it repeatedly performs work that the OCR worker immediately rejects.

## Goals

- Stop capturing and queueing API OCR frames while all effective provider slots are occupied.
- Resume from the current screen within one adaptive scan interval after a slot is released.
- Make adaptive load thresholds use the effective provider-specific concurrency limit.
- Keep Custom AI OCR at two concurrent calls and preserve configured limits for other API OCR providers.
- Leave local PaddleOCR, cooldown fallback, OCR result ordering, translation scheduling, UI, and user settings unchanged.

## Non-goals

- Do not change the user's 200 ms base scan setting.
- Do not reduce Custom AI OCR concurrency to one.
- Do not add perceptual image-difference heuristics.
- Do not redesign the capture or OCR queues.
- Do not call real providers during verification.

## Considered approaches

### 1. Coalesce the saturation log only

This would make the debug log smaller but would retain unnecessary capture, hashing, queueing, and worker wakeups. It does not address the scheduling defect.

### 2. Increase the configured scan interval

This is simple but applies a static delay to unrelated OCR providers and local OCR. It also changes user intent and cannot react when the provider recovers.

### 3. Provider-aware source backpressure (selected)

Expose one authoritative effective OCR concurrency calculation. The adaptive interval and API submission path both use it. Before capturing an API OCR frame, the capture thread checks whether that provider is saturated. If saturated, it records a coalesced diagnostic/metric and waits for the adaptive interval without taking a screenshot. Once a slot is free, the next iteration captures the current screen, naturally dropping obsolete intermediate video frames.

This approach removes work at its source while preserving responsiveness and existing queue semantics.

## Architecture

### Effective concurrency ownership

`AppCaptureOcrMixin` will expose `get_effective_ocr_concurrency_limit(provider_name=None)`. It returns the configured generic limit for ordinary API OCR and clamps Custom AI to two. `worker_threads._api_ocr_concurrency_limit()` remains the worker-facing compatibility seam and delegates to the app method when available.

This avoids separate hard-coded interpretations between adaptive scanning and API submission.

### Adaptive scan state

`update_adaptive_scan_interval()` will calculate load thresholds from the effective limit:

- overloaded: active calls are at or above 75% of the effective limit, rounded up;
- moderate: one slot below the overload threshold;
- normal: below the moderate threshold.

For the existing generic limit of eight, overload still begins at six calls. For Custom AI's limit of two, two active calls are correctly treated as overloaded and the existing 150% interval behavior applies.

### Capture backpressure

After resolving the OCR model and scan interval, `run_capture_thread()` will check saturation before overlay geometry, screen capture, resizing, hashing, or queue insertion. When full, it will:

- increment an `api_ocr_capture_backpressure_skip` runtime metric;
- emit a coalesced log containing provider, active count, effective limit, and current interval;
- sleep for the current adaptive interval and retry.

The check is limited to API OCR models. PaddleOCR continues through its existing local queue-pressure and duplicate-signature paths.

### Race tolerance

The existing submission-time concurrency check remains mandatory. A slot can fill between capture and worker submission, so the new source check is an optimization, not the only correctness guard.

## Error handling

- Missing or invalid configured limits fall back to one, matching current defensive behavior.
- A zero configured limit remains an explicit disabled state and causes API capture backpressure rather than a busy loop.
- Missing app-level effective-limit support falls back to the existing worker helper behavior for test doubles and compatibility.
- Backpressure diagnostics are coalesced so long upstream requests do not recreate log flooding.

## Testing

Tests will prove:

- Custom AI effective limit is two while another API provider keeps eight.
- Adaptive scanning treats two active Custom AI calls as overloaded and preserves the historical six-of-eight generic threshold.
- A saturated API provider prevents screen capture and queue insertion.
- Releasing a slot allows the next capture iteration to proceed.
- Local PaddleOCR does not use the API saturation guard.
- Existing API submission-time concurrency, cooldown fallback, caching, capture backend, and full offline suites remain green.

## Expected impact

For a session shaped like the latest live run, most of the 587 rejected capture cycles should disappear. OCR request latency remains provider-controlled, but the app should use substantially less local capture/queue work, produce fewer stale frames and saturation logs, and submit a fresher screen as soon as capacity becomes available.
