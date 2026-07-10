# Runtime Log Coalescing Design

## Evidence

The 2026-07-10 14:01-14:03 application window contained only 34 real async
translation starts but emitted these high-frequency state messages:

- 483 `Unified cache MISS` lines;
- 361 `queued latest translation request` lines;
- 80 `duplicate in-flight translation skipped` lines;
- 25 `ignored stale pending translation timer` lines.

Those 949 lines obscure the four Responses requests, provider errors, model
switching evidence, and UI display decisions that are needed for diagnosis.
The cache and queue behavior itself is already covered by runtime metrics and
focused lifecycle logs; the problem is repeated file I/O and presentation, not
missing state transitions.

## Alternatives

1. Delete the repeated messages and depend entirely on metrics. This gives the
   largest reduction but removes timing and first-occurrence evidence.
2. Add independent counters and timers in the cache and translation worker.
   This avoids a logger change but duplicates locking and summary behavior.
3. Add a small thread-safe coalescing gate to `logger.py` and opt in only the
   four proven noisy events. This centralizes behavior while leaving all other
   logs unchanged.

Use option 3.

## Coalescing Contract

For each stable, hashable event key:

- the first event is logged immediately;
- events inside the next 5 seconds are suppressed and counted;
- the first event at or after 5 seconds is logged with the current message and
  `suppressed=N similar events since previous log`;
- different keys never suppress each other;
- calls are safe from concurrent OCR and translation threads;
- the gate stores only timestamp and count, never the message body;
- clearing the debug log also clears gate state, so the next event is visible.

The public wrapper returns whether it wrote a line. Logging failures retain the
existing `log_debug` behavior and cannot affect application control flow.

## Event Keys And Messages

### Unified cache miss

Key by provider, source language, target language, and DeepL model type. Keep
the existing content-free message. HIT and STORE remain uncoalesced because
they were low-volume and confirm successful cache behavior.

### Pending translation queue

Use one stable queue key. Keep OCR batch, delay, and current reason, but remove
the OCR text from this high-frequency line. The translation submission and
provider logs already record the accepted request.

### Duplicate in-flight skip

Use one stable duplicate key. Keep OCR batch but remove source text. The
existing `duplicate_inflight_skip` runtime counter remains authoritative.

### Stale pending timer

Use one stable timer key. Keep current and stale generation numbers.

## Expected Effect

At a five-second interval, a two-minute session can emit at most about 24
periodic lines per stable event key, plus first occurrences. For the observed
four categories this reduces roughly 949 lines to about 100 while preserving
the current state and the exact number of suppressed repetitions in summaries.

## Verification

Tests must prove first-event visibility, per-key isolation, interval summaries,
thread-safe counts, reset behavior, cache-miss opt-in, content-free worker
messages, and unchanged functional state. The runtime logging, latency,
Custom AI, full tests directory, root discovery, compilation, and whitespace
checks must pass.
