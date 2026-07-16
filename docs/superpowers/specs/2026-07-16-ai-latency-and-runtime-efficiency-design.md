# AI Latency and Runtime Efficiency Design

## Goal

Reduce time-to-first-translation, wasted Custom AI requests, reasoning-token
overhead, first-use PaddleOCR stalls, shutdown errors, and redundant settings
writes without changing translation identity, provider credentials, cache
isolation, or subtitle rendering behavior.

## Current Evidence

- Automatic AI optimization resolves to the internal `adaptive` mode, but the
  adaptive advisor never returns `stream`.
- The latest translation session submitted nine requests; five were displayed,
  three completed stale, and one timed out.
- Eight successful Grok calls used 2,476 output tokens for short translations.
  The active profile uses `reasoning_effort=low`, while the runtime already
  understands an internal `none` contract that is not exposed in the profile UI.
- PaddleOCR startup prewarm takes about 16 seconds on this machine. Starting
  translation before it completes can cause the worker and prewarm paths to
  contend for first engine initialization.
- A slow request can complete after Tk has been destroyed and call
  `root.after(...)`, producing `RuntimeError: main thread is not in main loop`.
- Tk variable traces call the full settings writer repeatedly during clustered
  UI changes.

## Selected Approach

### 1. Explicit and adaptive streaming

Add `stream` as a user-facing AI optimization choice. Keep `auto`, `speed`, and
`quality` compatible:

- `stream` maps directly to the existing internal streaming transport.
- `auto` maps to `adaptive`.
- `speed` maps to streaming for translation and applies the speed request
  policy described below.
- `quality` maps to safe non-streaming.

The adaptive advisor may select `stream` only when:

- the active profile reports streaming support;
- at least three route observations exist;
- there are no consecutive route errors or active cooldown; and
- the route is slow enough for partial display to be useful.

If the stream is empty, truncated, reset, or times out, the existing bounded
non-stream fallback remains authoritative.

### 2. Translation request policy

Expose a `None` reasoning choice in the Custom AI profile UI and preserve it
through profile serialization. For translation requests:

- speed policy forces the reasoning contract to `none`;
- speed policy limits approved context to at most one entry;
- other policies preserve the profile's configured reasoning effort and
  configured context window.

Usage normalization records provider-reported reasoning tokens separately.
Short logs add a `Reasoning Tokens` line when the provider reports the value.
The value is observational and does not affect cache identity; the effective
reasoning contract already remains part of cache and in-flight identity.

### 3. Slow-route scheduling

Keep one normal Custom AI request plus at most one bounded overflow request.
Change only the overflow threshold:

- during the first route samples, use a conservative three-second threshold;
- after eight samples, use the existing route-aware threshold derived from P90,
  capped at four seconds;
- stream and race requests continue using their existing transport semantics.

The pending queue remains latest-only. No request cancellation is attempted
because the relay may not support cancellation and completed results remain
useful for cache storage.

### 4. PaddleOCR readiness

Represent the current prewarm operation with a thread-safe completion event.
When local PaddleOCR is selected and translation starts while matching prewarm
is active, the OCR worker waits for that event instead of initializing the same
engines concurrently. Waiting is bounded and exits when the application stops
or the prewarm generation changes. AI OCR and other OCR engines never wait.

The UI may show a temporary localized “Preparing local OCR” status while the
wait is active, but the change does not add a new settings control.

### 5. Shutdown-safe UI dispatch

Add one small helper that schedules Tk callbacks only while the application is
running and the root window still exists. Translation success, translation
error, API OCR completion, and streaming partial callbacks use the helper.
Shutdown races are logged at most coalesced/debug level and never become visible
translation errors.

### 6. Debounced settings persistence

Tk variable traces schedule one settings save 400 milliseconds after the latest
change. Explicit Save, model/profile actions that require immediate persistence,
and shutdown continue writing synchronously. A pending delayed save is cancelled
before an immediate save and during shutdown.

## Compatibility Boundaries

- Do not change API keys, profile IDs, endpoint normalization, failover order,
  cache credential scoping, structured-output fallback, or translation result
  normalization.
- Do not change subtitle appearance or overlay layout.
- Do not widen Custom AI concurrency beyond the existing bounded two-request
  maximum.
- Existing persisted `auto`, `speed`, and `quality` values remain valid.
- Existing profile values remain valid; missing or invalid reasoning effort
  still normalizes to `low`.

## Testing

Use focused red-green tests for:

- optimization-mode normalization and response-mode mapping;
- adaptive stream selection and safe fallbacks;
- speed policy reasoning/context behavior;
- reasoning-token extraction and short-log output;
- conservative initial supersede threshold;
- PaddleOCR wait/reuse behavior;
- shutdown-safe callback scheduling;
- settings-save debounce and immediate flush.

Then run the complete `tests/` suite, root compatibility suite, Python
compilation for touched modules, and `git diff --check`.
