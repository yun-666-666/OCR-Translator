# Streaming Response Lifetime Design

## Goal

Prevent Custom AI compatibility retries and streaming request paths from retaining HTTP responses or pooled connections after the response is no longer usable.

## Scope

This increment changes only `custom_ai_transport.py` and its focused tests. It does not change request payloads, endpoint selection, retry counts, provider capability memory, timeouts, parsing, UI behavior, or public APIs.

## Ownership rule

The transport layer follows one explicit rule:

> A response returned by `_post_with_output_limit_fallback` remains open for its caller. Every response replaced during compatibility fallback, or fully handled by a streaming wrapper, is closed by the layer that stops using it.

## Design

### Compatibility fallback

`_post_with_output_limit_fallback` closes the current response before sending a replacement for an unsupported prompt-cache key, reasoning effort, structured-output contract, or output-limit field. The final response is still returned open. If compatibility inspection or a replacement send raises, the currently owned response is closed before the exception escapes.

### Streaming wrappers

`_stream_post` and `_stream_responses_post` initialize a per-endpoint `response` variable and close it in `finally`. This covers successful parsing and return, HTTP-error fallback, cooldown stop, parser/callback exceptions, and an early terminal Responses event. The existing `_close_response_quietly` helper keeps cleanup failures from replacing the real request result or error.

### Error behavior

Cleanup is best-effort and must not mask parsing, HTTP, or transport exceptions. Existing sanitized error aggregation remains unchanged.

## Testing

Focused tests use fake responses with a `close_calls` counter and no real network traffic.

- A successful Chat Completions stream is closed once.
- A successful Responses API stream, including a terminal event, is closed once.
- Every abandoned compatibility response is closed once while the final returned response remains open.
- A compatibility-inspection exception closes the current owned response.

Then run the Custom AI provider suite, the full `tests/` discovery suite, Python compilation for touched files, and `git diff --check`.

## Risk assessment

Risk is low because cleanup happens only after the code has finished reading or rejecting a response. No giant-module split is part of this increment. Extracting transport classes from `custom_ai_transport.py` would mix structural change with a resource-lifetime fix and is therefore explicitly deferred.
