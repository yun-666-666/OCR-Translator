# Adaptive Translation Context Budget Design

## Goal

Reduce model input size and long-tail translation latency while preserving the
most relevant approved subtitle context.

## Evidence

Historical real-profile short logs contain 242 translation records. Profiles
with token telemetry show average input sizes from roughly 288 to 5435 tokens,
with a maximum of 5678 tokens. The existing fixed 4000-character context budget
can amplify already-long OCR payloads by appending several old source and
translation blocks.

## Options considered

### 1. Fixed smaller context cap

Simple, but unnecessarily removes useful history from short dialogue.

### 2. Adaptive context cap based on current source length

Recommended. Short subtitles retain richer history; long OCR text gets a much
smaller context allowance because it already contains substantial local
information and model work.

### 3. Summarize old context with another model call

Adds latency, cost, failure modes, and cache complexity. Rejected.

## Budget formula

Use three constants:

- maximum context budget: 2400 characters;
- minimum context budget: 600 characters;
- source penalty cap: 1800 characters.

For each request:

```text
budget = max(600, 2400 - min(1800, len(current_source)))
```

Examples:

- 40-character subtitle: 2360 context characters;
- 600-character OCR block: 1800 context characters;
- 1200-character OCR block: 1200 context characters;
- 1800+ character source: 600 context characters.

When no current source is supplied by a compatibility call, use the 2400
maximum.

## Selection behavior

- Continue honoring the configured context entry count.
- Continue excluding the current source from its own request/cache identity.
- Walk history newest-first.
- Include complete recent source/translation pairs while they fit.
- If the newest entry alone exceeds the budget, truncate that one pair using
  the existing balanced source/translation truncation.
- Stop before older entries once the next one does not fit.
- Return selected entries in chronological order.

The current source is never truncated.

## Cache behavior

The exact adaptively selected context remains in the Custom AI cache key.
Requests with different effective context therefore remain isolated. Repeated
current subtitles still exclude themselves and reuse the exact cache entry.

## Files

- `handlers/translation_handler.py`
- `tests/test_custom_ai.py`

## Verification

- formula boundary tests;
- short source retains multiple recent pairs;
- long source shrinks context to the minimum budget;
- oversized newest pair is truncated, not dropped;
- recent-first selection and chronological output order;
- current-source exclusion and immediate-repeat cache behavior remain green;
- full project tests, compilation, and diff checks.
