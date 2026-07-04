# Translation Terminal Integrity Design

## Problem

The translation provider repairs one explicit output-limit truncation, but it
does not validate the terminal state after that repair. A second truncated
response is normalized and returned as if complete. Other explicit non-success
states can also carry partial text and currently pass through:

- Chat `finish_reason: "content_filter"` or another non-stop reason.
- Responses `status: "incomplete"` for a non-token-limit reason.
- Streaming `response.failed` events after one or more text deltas.

Returning these partials allows them to be displayed as successful and stored
in the translation cache.

## Goals

- Validate the final response after any repair attempt.
- Never return or cache text from an explicitly incomplete or failed terminal
  state.
- Preserve useful upstream failure details.
- Accept compatible relay responses that omit terminal metadata.
- Keep the current single output-limit repair attempt.

## Design

Add a translation-specific terminal-state validator:

- Chat: accept absent/empty metadata and `stop`; reject `length` as truncated
  and reject every other explicit reason with its name in the error.
- Responses: accept absent/empty metadata and `completed`; reject `incomplete`
  with `incomplete_details.reason`; reject `failed` with the structured error
  message when available; reject any other explicit non-success status.

Call this validator after the optional output-limit repair and before response
text parsing/normalization.

Extend the Responses SSE aggregator to recognize `response.failed`, retain
status and error metadata, and return that terminal envelope even when partial
text was already emitted. Translation validation then turns it into a visible
error instead of a cacheable result.

## Verification

- Chat and Responses responses that remain token-limited after repair raise.
- Chat content-filter termination raises.
- Responses non-token incomplete termination raises with the reason.
- A streamed `response.failed` event rejects already accumulated partial text
  and preserves the upstream message.
- Successful, metadata-free relay responses remain accepted.
