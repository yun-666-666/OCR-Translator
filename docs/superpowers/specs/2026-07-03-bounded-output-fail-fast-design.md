# Bounded Output and Fail-Fast Endpoint Design

## Goal

Reduce worst-case translation latency and improve subtitle fidelity without adding
provider-specific request fields or weakening endpoint compatibility.

## Output budget

Translation payloads receive a dynamic `max_tokens` value derived from the current
source text:

- minimum: 64 tokens
- scaling: 4 output tokens per normalized source character
- maximum: 2,048 tokens

The bounds are intentionally generous for language expansion while preventing a
misbehaving non-reasoning model from producing an unbounded explanation. When a
profile explicitly sets `reasoning_effort` or `model_reasoning_effort`, the payload
omits the output bound because Responses counts hidden reasoning tokens inside
`max_output_tokens`; an aggressive subtitle-sized cap could suppress the visible
translation. Responses API payloads otherwise continue to translate `max_tokens`
into the existing `max_output_tokens` field.

The static system instruction explicitly says that commands appearing inside the
source subtitle are untrusted text to translate, not instructions to follow. The
current source remains at the end of the user message, preserving the existing
static-prefix/dynamic-suffix layout.

## Endpoint fallback

Bare relay URLs may legitimately require either `/v1/...` or non-`/v1/...`
routes. Alternate endpoint probing remains enabled for:

- HTTP 404 and 405, which commonly signal the wrong path shape
- HTTP 500, 502, 504, and other non-capacity 5xx failures, where a cached route or
  gateway path may be unhealthy

Fallback stops immediately for deterministic request failures:

- all other 4xx responses, including authentication, model, validation, and rate
  limit failures
- HTTP 503
- error bodies identified by the existing rate-limit or capacity detectors

The same predicate is applied to Chat Completions, Responses, and both streaming
paths.

## Compatibility and error behavior

No new provider-specific field is added. Existing sanitized error messages,
cooldown activation, successful URL caching, and transport-session replacement
remain unchanged. The first deterministic error is returned without sending the
same invalid request to a second path.

## Verification

Tests prove output-budget bounds, the reasoning-profile exception,
prompt-instruction placement, Responses field conversion, and one-call fail-fast
behavior across all four request paths.
Existing URL fallback tests continue to prove that 404 and recoverable 502
fallbacks still work.
