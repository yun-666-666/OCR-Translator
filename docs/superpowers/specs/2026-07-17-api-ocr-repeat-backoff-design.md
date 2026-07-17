# API OCR Repeat Backoff Design

## Goal

Reduce proven redundant Custom AI OCR calls without suppressing an unknown new subtitle, and prevent the official xAI route from probing an invalid Chat Completions URL.

## Evidence

The 2026-07-17 evening session issued 401 Custom AI OCR calls. 284 adjacent OCR outputs were identical, and 60 completed OCR responses were already stale. The direct xAI profile retried `https://api.x.ai/chat/completions` after failures on the correct `/v1` endpoint; that fallback returned HTTP 404 eight times.

## Scope

1. When an API OCR response repeats the displayed subtitle, retain a short, scope-bound cooldown. During the cooldown, `run_api_ocr` does not encode or submit another remote OCR image for the same OCR provider, profile, language, image contract, and capture region.
2. The cooldown is 0.75 seconds. It is deliberately bounded: a changed subtitle may wait only until the cooldown expires, rather than being discarded by a fuzzy image comparison.
3. A cache hit, changed scope, error response, empty response, and a new OCR result retain their existing behavior. A skipped repeat contributes a runtime metric and a coalesced diagnostic line.
4. A host-only `api.x.ai` Chat Completions base URL normalizes to the single canonical `/v1/chat/completions` candidate. Other providers preserve their existing two-candidate compatibility behavior.

## Out of Scope

- No fuzzy image hash or subtitle-region inference.
- No UI setting, model/profile mutation, timeout change, or provider reordering.
- No change to local PaddleOCR behavior.

## Tests

- A repeated successful API OCR response activates a scope-bound cooldown and a fresh frame in that scope is not encoded or submitted before expiry.
- A changed OCR scope is not blocked by a prior cooldown.
- The official xAI host-only URL has exactly one canonical Chat Completions candidate; a generic host retains both compatibility candidates.
