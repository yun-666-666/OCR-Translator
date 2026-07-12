# Cost-Protected Custom AI Profile Failover Design

## Goal

Keep subtitle translation available when one Custom AI relay profile fails, while
limiting each subtitle to one sequential request per configured profile and
preventing repeated paid requests when no profile is usable.

## Evidence

The 2026-07-11 22:34 session made 33 calls through profile `234` in about
16 seconds: 15 HTTP 200 responses contained forced SSE `data:` frames with no
completion tokens and no usable translation, followed by 18 HTTP 403 responses
reporting failed prepaid-credit reservation.  The existing scheduler only
deduplicated identical in-flight OCR text, so small OCR changes continued to
submit requests after the first endpoint failure.

## Alternatives Considered

1. Stop after the selected profile fails.  This limits cost but loses the
   user's requested automatic fallback.
2. Send the same translation to all profiles concurrently.  This improves tail
   latency but intentionally spends on multiple providers for one subtitle.
3. Use ordered, sequential failover.  Try the selected profile first, then
   enabled profiles in their saved order, stopping at the first cache hit or
   successful translation.  This retains availability without intentional
   duplicate billing.  This is the selected design.

## Design

### Profile circuit state

`CustomAIProvider` will maintain an in-memory, profile-scoped unavailable
cooldown in addition to its existing rate-limit cooldown.  A profile becomes
unavailable for 60 seconds after either a terminal endpoint failure (including
HTTP 401/403, capacity/rate errors, or a transport failure) or a success-status
response that cannot produce a translation.  A valid translation clears that
profile's unavailable state.  The state is never persisted and contains no
secret material.

The non-stream Chat Completions path will also accept a relay that incorrectly
returns a valid SSE chat stream for a non-stream request.  If that stream has a
real text delta, it is parsed as the translation.  A stream with no text remains
a terminal profile failure and enters the same cooldown.

### Ordered translation path

For ordinary Custom AI translation modes (`safe`, `none`, and `stream`), the
handler constructs an ordered candidate list: selected translation profile
first, then all other enabled profiles in saved order.  Candidates currently in
cooldown are skipped.  For every eligible candidate it checks that candidate's
normal translation cache before contacting the provider.  It performs one
provider call only when the cache misses.

On a terminal failure the handler records the profile cooldown and advances to
the next candidate.  On the first success it stores the result only under the
winning profile's normal cache identity and returns immediately.  It neither
changes the user's saved active profile nor retries a failed candidate in the
same subtitle request.  The explicit `race` mode remains an intentional
parallel-cost mode and retains its current contract.

### Scheduler behavior when every profile is unavailable

The worker's existing provider-cooldown query will consider the same sequential
candidate list.  If any fallback is healthy it returns zero so that the handler
can use it immediately.  If every configured profile is cooling down, it
returns the earliest expiry.  The scheduler then defers subtitle work instead
of emitting a new network request every submit interval.

### Failure reporting and privacy

The UI receives one concise all-profiles-unavailable result after the candidate
list is exhausted.  Detailed per-profile reasons remain in the debug log and
continue through the existing secret-redaction helper.  Neither API keys nor
full provider response bodies are retained in cooldown state or displayed.

## Tests

Focused regression tests will prove that a forced SSE response with text is
accepted, a forced SSE response without text cools the profile, an active
profile failure falls through to the next enabled profile exactly once, cached
fallback output avoids a network call, and all cooling profiles make the worker
wait for the earliest expiry.  Existing Custom AI tests remain the regression
suite for rate-limit and race behavior.
