# Responses Empty-Output Recovery Design

## Context

The 2026-07-10 14:01 application session sent translation requests to the
configured relay through the Responses API. Several requests returned HTTP
200, but the returned JSON contained no extractable `output_text`, so
`parse_responses_response` raised `Responses API response did not contain
output text` and the overlay showed an error instead of a translation.

Live probes established the boundary of the problem:

- `gpt-5.4` can return a standard completed Responses object with
  `output[0].content[0].type=output_text`.
- Two concurrent `gpt-5.4` probes both returned valid output, so the local
  `2/2` counter is not the cause.
- The relay intermittently returns HTTP 502 for `gpt-5.4` and `gpt-5.5`;
  existing transport recovery already handles that separately.
- One `gpt-5.6` probe was rejected with HTTP 404 because that request's
  selected account group had no supporting account. The user confirms that
  the relay normally supports `gpt-5.6`, so this single routing result must
  not be treated as model unavailability or used to hide/remove the model.

The remaining client defect is that an HTTP 200 response with no text is
treated as a final parsing failure, even though translation requests are
idempotent and one bounded retry can recover an intermittent empty response.

## Design

Add one recovery step inside `CustomAIProvider.translate` after terminal
status validation and before returning a parsing error:

1. Keep the existing capability-aware parser for the initial response so an
   internal endpoint-compatibility fallback remains authoritative.
2. If a Responses API response has no output text, log a content-free shape
   summary and retry exactly once.
3. In `auto` structured-output mode, remove the JSON Schema for the retry and
   explicitly parse that retry as plain text. This covers relays that accept
   the schema parameter but intermittently fail to produce structured output.
4. In `strict` or `off` mode, preserve the original request contract for the
   retry.
5. Apply the existing terminal-status checks to the retry response. If the
   second response is also empty, return the existing error without another
   request.

The recovery remains inside the provider layer because that layer owns the
wire contract, response parsing, compatibility fallbacks, and request
duration accounting. The translation worker and display layer should not
need to know whether a provider response required recovery.

## Safe Diagnostics

The empty-response log records only:

- normalized wire API and response status;
- counts of output, reasoning, message, content, output-text, and refusal
  items;
- output-token and reasoning-token counts;
- whether the retry keeps or removes structured output.

It does not record the response body, source text, translated text, API key,
instructions, endpoint URL, response ID, or provider metadata.

## Testing

Deterministic provider tests will prove that:

- an empty completed Responses object in `auto` mode triggers one plain-text
  retry and returns the recovered translation;
- whitespace-only top-level output does not hide valid nested output or bypass
  empty-output recovery;
- diagnostics describe only structure and do not leak a credential or body;
- malformed/non-finite usage counts cannot interrupt recovery diagnostics;
- two consecutive empty Responses objects make exactly two calls and retain
  the existing parsing error;
- `strict` retries preserve the schema and retry terminal errors remain
  visible;
- an ordinary successful Responses object still makes one call.

The full Custom AI suite and repository test discovery will be run after the
focused RED/GREEN cycle.
