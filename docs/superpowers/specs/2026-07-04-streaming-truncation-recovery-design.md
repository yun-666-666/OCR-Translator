# Streaming Truncation Recovery Design

## Problem

Translation already retries without its output-token limit when a non-streaming
response explicitly reports truncation. The streaming parsers aggregate text
and usage but discard terminal completion metadata:

- Chat Completions emits `finish_reason: "length"` in a final chunk.
- Responses emits `response.incomplete` with
  `incomplete_details.reason: "max_output_tokens"`.

Because the synthesized response lacks those fields, the existing truncation
detector treats partial streamed translations as complete. They can then be
displayed and stored in the translation cache.

## Goals

- Preserve terminal truncation metadata from both supported streaming wire
  formats.
- Reuse the proven output-limit retry path rather than duplicating retry logic.
- Keep partial display callbacks and normal completed streams unchanged.
- Preserve usage extraction.
- Stay compatible with relays that emit no explicit terminal metadata.

## Design

### Chat Completions

Track the latest non-null `finish_reason` while reading chunks. Add it to the
single synthesized choice alongside the aggregated message. The existing
`_response_was_output_limited()` function will then recognize `length`.

### Responses

Treat both `response.completed` and `response.incomplete` as terminal events.
Capture the embedded response object, including `status`,
`incomplete_details`, and usage. Merge those fields into the synthesized
response while keeping the already accumulated `output_text`.

If a relay omits `status` on a `response.incomplete` event, synthesize
`status: "incomplete"` from the event type.

### Retry behavior

No new retry mechanism is needed. `translate()` already removes the output
limit and makes one more request when `_response_was_output_limited()` returns
true. Once stream metadata is retained, both wire formats enter that path.

## Verification

- A Chat stream ending with `finish_reason: "length"` retries once without
  `max_tokens` and returns the complete second result.
- A Responses stream ending with `response.incomplete` retries once without
  `max_output_tokens` and returns the complete second result.
- Normal completed streams, UTF-8 decoding, partial callbacks, usage, and
  non-streaming retries remain unchanged.
