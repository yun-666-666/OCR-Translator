# Output-Limit Capability Fallback Design

## Goal

Keep dynamic translation output bounds on compatible relays while recovering
automatically when an OpenAI-compatible endpoint rejects the output-limit field.

## Capability identity

Unsupported-field capability is remembered for the lifetime of
`CustomAIProvider`, keyed by:

- normalized wire API
- normalized base URL
- model name

Chat Completions and Responses use separate keys because a relay may accept
`max_tokens` on one wire API and reject `max_output_tokens` on the other.

## Detection

Retry is allowed only when:

- the response status is 400 or 422
- the rejected field name appears in the sanitized error
- the error explicitly identifies the field as unsupported, unknown,
  unrecognized, not permitted, not allowed, or an extra input

Numeric validation such as “max_tokens must be below 1000” is not a capability
failure and continues through normal fail-fast handling.

## Request flow

One shared send helper:

1. Removes the output-limit field before sending when capability memory says it
   is unsupported.
2. Sends the request once.
3. On an explicit unsupported-field response, records capability, removes only
   that field, and retries the same URL once.
4. Returns the retry response to existing status, cooldown, URL-cache, parsing,
   and error logic.

The helper preserves `stream=True` for streaming requests and never mutates the
caller’s payload.

## Truncation recovery

A supported output limit can still be too small for an unusually expansive
translation. After a successful HTTP response, `translate()` checks:

- Chat Completions: `choices[0].finish_reason == "length"`
- Responses: `status == "incomplete"` with
  `incomplete_details.reason == "max_output_tokens"`

When the original translation payload contains `max_tokens`, the provider retries
once without that field and returns only the complete retry. It does not mark the
field unsupported, because truncation proves the field was accepted. A response
that is truncated without an application-supplied limit is not retried.

## Alternatives rejected

- Removing output limits globally would discard useful worst-case latency
  protection.
- A user-facing compatibility switch would require users to diagnose provider
  protocol details.
- Persisting inferred capability into profile configuration would create
  surprising automatic config writes and stale state after relay upgrades.

## Verification

Chat Completions tests prove first-call retry and later omission without another
failure. Streaming Responses tests prove `max_output_tokens` fallback while
retaining stream behavior. A negative test proves numeric limit validation does
not trigger capability memory. Chat and Responses tests prove explicitly
limit-truncated translations retry once without caching or returning the partial
result.
