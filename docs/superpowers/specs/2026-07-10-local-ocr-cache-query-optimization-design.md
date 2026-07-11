# Local OCR and Cache Query Optimization Design

## Goal

Reduce unnecessary local PaddleOCR work and translation-cache lookups without
dropping valid subtitles or adding a user-facing setting.

## Evidence

The 2026-07-09 application session recorded 1,858 captures and 1,085 local
OCR operations in 368 seconds. The capture loop already detects exactly equal
capture signatures, but randomly lets many duplicate frames through. The same
session recorded 1,507 translation-cache misses versus 171 started translation
requests, because the local OCR routing path probes the cache before the OCR
stability gate has decided whether the candidate will be submitted.

## Design

### Deterministic duplicate-frame admission

For local OCR only, the first observation of a capture signature is enqueued.
One exact duplicate is also enqueued so the OCR stability gate can confirm a
short or low-quality subtitle. Further exact duplicates are skipped until the
signature changes. A changed signature resets the duplicate count immediately.

The capture loop records `capture_exact_duplicate_skip` in the existing runtime
metrics. It does not use fuzzy visual similarity, so a changed frame is never
suppressed by this optimization.

### Cache lookup after stability admission

The local OCR router evaluates the existing OCR stability gate before checking
the instant translation cache. Candidates that are pending, dropped, or already
deduplicated therefore make no cache call. Candidates that are admitted still
call `start_async_translation`, which retains the existing instant-cache display
and in-flight translation safeguards.

This keeps immediate display for clear cached subtitles. A cached low-quality
candidate can wait for the existing 120-250ms confirmation window before it is
displayed; this is an intentional accuracy trade-off.

## Scope

- Modify `worker_threads.py`.
- Extend `tests/test_latency_optimization.py`.
- Do not change configuration, UI, model selection, cache persistence, or
  subtitle text filtering.

## Verification

Tests must prove that the third identical local frame is skipped, a changed
frame is admitted, pending OCR candidates do not probe the translation cache,
and an admitted candidate still reaches the existing async translation path.
Run the targeted latency suite and the broader project regression suite after
the implementation.
