# Transient Translation Request Recovery Design

## Goal

Recover a translation from a brief relay or TLS failure without turning
permanent errors into repeated slow requests.

## Problem

The provider already reuses pooled HTTP connections and discards an owned
session after TLS/reset failures. However, a non-streaming request using an
explicit `/v1` endpoint has only one URL candidate. A single 502, 504, or
connection-reset exception therefore ends the subtitle even though an immediate
second attempt often succeeds.

Trying alternate path shapes is not equivalent to retrying the known endpoint:
the alternate path may be invalid, and explicit `/v1` URLs have no alternate
candidate at all.

## Chosen behavior

For non-streaming Chat Completions and Responses requests:

- retry the same URL at most once for HTTP 502 or 504;
- retry the same URL at most once for a recognized TLS EOF / connection-reset
  exception;
- rebuild the owned pooled session before retrying a transport-reset failure;
- use the same injected client when the provider does not own it;
- include both attempts in measured duration;
- after the one retry, continue the existing endpoint-candidate logic.

## Explicit no-retry boundaries

- latency mode `none`;
- streaming requests, because partial output may already have reached the UI;
- 400/401/403/404/405/422 permanent request errors;
- 429 rate limits;
- 503 capacity/unavailable responses;
- any response carrying `Retry-After`;
- output-limit compatibility retries remain a separate one-time mechanism.

## Files

- `custom_ai.py`
- `tests/test_custom_ai.py`

## Verification

- recovery from first-attempt 502 and 504;
- recovery from a TLS reset with owned-session replacement;
- no retry in `none` mode;
- no retry for rate limit, authentication, capacity, and stream paths;
- cached URL fallback still works after the same-URL retry is exhausted;
- full project tests, compilation, and diff checks.
