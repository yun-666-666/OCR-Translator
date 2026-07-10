# Hot-Path Log Compression Design

## Problem

The post-fix 23:06 runtime session produced about 322 KB and 3,913 log lines in
127 seconds. Two measured groups dominated avoidable synchronous logging:

- 588 PaddleOCR fast-path failure/filter lines, about 55 KB;
- 1,362 content-bearing translation pipeline lines, about 114 KB.

`logger._RotatingTextWriter.write()` flushes every message, so repeated debug
events add file-system work directly to capture, OCR, translation, and UI
display paths. The same subtitle text is also copied through several main-log
boundaries even though the dedicated Custom AI short log already preserves the
provider result for quality diagnosis.

## Selected approach

Keep the synchronous writer and diagnostic event order, but reduce the number
and size of writes at their sources:

1. Add a shared content-free text summary (`chars=N lines=M`) in `logger.py`.
2. Replace repeated PaddleOCR filter/fallback writes with stable-key
   `log_debug_coalesced()` events. Include counts/confidence, never OCR text.
3. Keep per-request sequence/timing/display events, but replace source and
   result bodies with the shared summary.
4. Remove the unconditional multi-line `DIALOG_FORMAT_DEBUG` chatter. Emit one
   coalesced, content-free event only when formatting actually changes text.
5. Coalesce repeated display language/backend route events while retaining one
   immediate and periodic diagnostic.

## Alternatives rejected

- An asynchronous writer could reduce caller latency further, but adds shutdown
  ordering, queue bounds, and crash-tail loss risks to the diagnostic system.
- Disabling debug logging or buffering without flush would hide the exact
  upstream/cooldown evidence that diagnosed the relay incident.
- Broader fuzzy OCR deduplication was rejected from this increment: historical
  replay showed different translations even for adjacent inputs above 0.97
  similarity, so a looser threshold could suppress real subtitle changes.

## Components

- `logger.summarize_text_for_log(value)` is the only shared representation of
  runtime text in the main log.
- `paddle_ocr_backend.py` coalesces low-confidence, symbol-noise, no-line-crop,
  no-usable-text, and fast-path-exception events.
- `worker_threads.py` retains sequence IDs, OCR batch IDs, timings, active-call
  counts, and content-free text shape.
- `handlers/translation_handler.py` retains provider selection and exceptional
  errors, but removes redundant source/result bodies and dialog no-op logs.
- `handlers/display_manager.py` retains display text shape and periodically
  reports the selected language/backend route.

## Error handling and compatibility

The summary helper accepts `None`, strings, and arbitrary values without
raising. Coalescing already falls back to direct logging if its gate fails.
No short-log schema, cache key, provider request, UI content, or translation
result changes.

## Tests

- Summary helper reports character/line counts and never returns content.
- PaddleOCR noisy/fallback events use stable coalesced keys and contain no OCR
  text.
- Worker request/completion/response/display diagnostics contain sequence and
  shape but not source/result secrets.
- Dialog formatting logs only a content-free applied event; no-op formatting
  produces no debug writes.
- Display manager reports content-free shape and coalesced route diagnostics.
- Focused logging/OCR/latency suites, the full suite, compile checks, and diff
  checks remain green.

## Success criteria

- The measured 588-line OCR failure/filter group becomes first-and-periodic
  summaries rather than per-frame writes.
- Main-log translation events contain no subtitle bodies.
- Relay HTTP status, cooldown, timing, request sequence, active-call, and UI
  route evidence remains available.
- Application behavior and dedicated short-log output remain unchanged.
