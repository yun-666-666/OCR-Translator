# Capture And OCR Hot-Path Logging Design

## Evidence

The real 2026-07-10 15:28-15:29 UI session emitted 858 capture/OCR hot-path
lines in roughly 30 seconds:

- 346 successful capture lines (344 MSS plus 2 pyautogui benchmark lines);
- 342 per-frame capture timing lines;
- 44 PaddleOCR routing lines;
- 44 PaddleOCR recognized-text lines;
- 44 PaddleOCR timing lines;
- 19 OCR frame-cache hit lines;
- 19 OCR cache-hit timing lines.

These lines now dwarf provider, queue, display, and error evidence. They also
perform avoidable synchronized file writes on the capture/OCR critical path.
The full recognized source text is repeated even though translation lifecycle
logs already provide accepted-request evidence and the OCR debug window is the
right place for frame-by-frame text inspection.

## Alternatives

1. Disable all successful capture/OCR logs. This maximizes I/O reduction but
   removes confirmation that the pipeline is alive and hides timing drift.
2. Coalesce every line on one fixed five-second interval. This is simple but a
   slow capture/OCR event can be hidden behind a recent normal sample.
3. Coalesce normal events every five seconds and use a separate one-second
   slow-event channel. Keep state changes, fallbacks, and errors immediate.

Use option 3.

## Behavior

### Successful captures

`capture_screen_region` sends success through the existing coalescer, keyed by
backend and region size. Backend failures and fallback decisions continue to
use immediate `log_debug` calls.

The worker's detailed capture timing uses a small hot-path timing helper:

- normal MSS captures are coalesced in a five-second channel;
- MSS captures at or above 50 ms use a separate one-second slow channel;
- normal pyautogui captures are allowed up to 250 ms before the slow channel;
- configured/resolved/actual backend and region dimensions remain in emitted
  messages;
- source-context changes remain immediate.

### OCR frame-cache hits

The cache's generic hit event is coalesced every five seconds. Worker timing is
coalesced normally, with a separate slow channel at or above 50 ms. Cache
return values, LRU recency, metrics, and debug-display behavior do not change.

### PaddleOCR routing and recognition

Repeated PaddleOCR routing uses one stable five-second key. Unknown-model
fallbacks remain immediate.

Successful subtitle fast-path recognition uses one stable five-second key and
logs only line count and character count. It does not log recognized text.
Low-confidence/noise filters, no-text fallback, no-crop fallback, and engine
errors retain their existing immediate diagnostics because they are uncommon
and explain quality failures.

PaddleOCR processing timings use the worker timing helper, with a 500 ms slow
threshold. The first slow event is immediate, and sustained slow events are
summarized at most once per second.

## Timing Helper Contract

The worker helper receives a stable event key, message, measured duration,
slow threshold, normal interval, and slow interval. It:

- converts numeric inputs safely;
- uses `(event_key, "normal")` for values below the threshold;
- uses `(event_key, "slow")` and prefixes `SLOW:` for values at or above the
  threshold;
- delegates counts/thread safety to `log_debug_coalesced`;
- never changes capture/OCR control flow or metrics.

## Expected Effect

An offline replay of the exact 858-event session produces 49 emitted lines and
suppresses 809 repeated events, a 94.3 percent reduction. Slow or changing
conditions remain more visible by design.

## Verification

Tests must prove:

- capture success and frame-cache hits opt into stable coalescing keys;
- capture failures/fallbacks remain immediate;
- normal and slow timings use independent keys and intervals;
- Paddle routing is coalesced;
- successful Paddle recognition logs counts but no OCR text;
- OCR/capture return values, cache recency, metrics, and routing state are
  unchanged;
- latency, PaddleOCR, runtime logging, full discovery, compilation, and
  whitespace checks pass.
