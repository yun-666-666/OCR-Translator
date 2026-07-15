# AI OCR Cooldown Recovery and PaddleOCR Fallback Design

## Runtime evidence

The 2026-07-15 application log shows that Custom AI OCR reached eight active
requests within roughly three seconds. The first translated subtitle was
displayed at 21:25:41. At 21:25:43 one OCR request returned HTTP 502 with a
60-second `Retry-After`, activating a request-scoped cooldown. Other requests
already in flight for the same profile returned HTTP 200 after that failure,
but the success path cleared only the exponential-backoff counter and left the
active cooldown deadline in place.

The OCR loop continued creating work every 200 ms while that cooldown was
active. The session recorded 78 Custom AI OCR batches, 52 cooldown
short-circuits, and 13 stale OCR results. This converted a single gateway error
into a visible 60-second interruption and repeatedly replaced the useful
subtitle with cooldown errors.

The same session also showed translation failover repeatedly probing profiles
that returned HTTP 402 insufficient-balance responses, HTTP 403 precharge
failures, empty response content, timeouts, and endpoint errors. Every failure
currently receives the same 60-second profile cooldown even though these
failure classes need different recovery policies.

## Goals

1. Keep subtitles translating by temporarily routing Custom AI OCR frames
   through the existing PaddleOCR pipeline whenever the selected AI profile is
   cooling down.
2. Return automatically to the selected AI OCR profile as soon as it becomes
   healthy, without changing the saved OCR selection.
3. Let a later successful request clear a request-scoped 5xx cooldown while
   preserving transport-scoped 429 cooldowns.
4. Prevent the Custom AI OCR path from producing an eight-request burst for
   rapidly changing frames.
5. Stop transient AI OCR errors from covering the last useful translation when
   PaddleOCR fallback is available.
6. Apply bounded, error-aware profile cooldowns so permanent balance/auth
   failures are not retried every minute while transient failures recover
   sooner.

## Non-goals

- No changes to the user's selected OCR model, profiles, API keys, prompts, or
  language settings.
- No paid live-network verification.
- No new UI setting or redesign.
- No automatic switching between Custom AI profiles.
- No profile-role schema migration in this increment. Explicit OCR-only versus
  translation-only profile roles remain a later isolated improvement.

## Design

### Request-scoped cooldown recovery

`CustomAITransportMixin._note_rate_limit_success()` will continue resetting
backoff counters after a successful response. It will additionally remove only
the request-scoped cooldown key for the successful profile/model/wire-API
combination. It will not remove the transport-scoped key used for HTTP 429 and
explicit rate-limit responses.

This matches the observed case: a concurrent HTTP 200 proves that one
request-scoped 502 did not represent a continuing model-route outage. A 429
still protects the shared provider/credential transport for the complete
server-directed interval.

### Error-aware profile health

`TranslationRequestsMixin` will own one pure classifier that maps sanitized
failure text to a bounded profile cooldown:

- 300 seconds for deterministic account/configuration failures such as HTTP
  401/402/403, insufficient balance, precharge failure, or a pure invalid
  endpoint/model response;
- 10 seconds for empty or invalid model output;
- 15 seconds for timeouts, non-JSON gateway responses, connection failures,
  and HTTP 5xx responses;
- 30 seconds for unclassified failures.

When OCR or translation succeeds, the existing profile-unavailable marker is
cleared. OCR will now use the same profile health contract as translation.
Transport cooldowns remain independently owned by `CustomAITransportMixin`.

### Automatic PaddleOCR fallback

Before the OCR loop chooses its execution path, it will ask the translation
handler for the selected Custom AI OCR profile's remaining cooldown. If the
saved selection is `custom_ai` and the remaining cooldown is positive, the
current frame is routed through the normal PaddleOCR local path. The saved
selection and active profile are left untouched.

The check runs for every frame, so the first frame after cooldown recovery uses
Custom AI automatically. Transition logs are coalesced and contain the
remaining duration but no OCR text or credentials.

If PaddleOCR itself is unavailable, its existing exception handling remains in
control; the change does not hide a local OCR installation failure.

### Custom AI OCR backpressure

The generic API OCR limit remains unchanged for other providers. Custom AI OCR
will use a provider-specific maximum of two active calls. Frames that arrive
while both calls are active continue to be discarded before image conversion;
the capture queue already supplies a fresh frame on the next loop, giving the
path latest-frame behavior without adding a second queue.

This reduces the observed maximum from eight paid in-flight requests to two,
while retaining overlap for relays that take several seconds to respond.

### Preserve the last useful subtitle

When a Custom AI OCR error arrives and the selected profile is now cooling,
`process_api_ocr_response()` will advance chronological state and log the
failure, but it will not replace the current subtitle with the transient error.
The next OCR-loop frame is handled by PaddleOCR. Non-cooldown OCR errors retain
the existing visible-error behavior.

## Tests

- A successful response clears an active request-scoped 502 cooldown.
- A successful response does not clear a transport-scoped 429 cooldown.
- OCR and translation failures receive the correct 10/15/30/300-second
  profile cooldown, and success clears the profile marker.
- The OCR loop selects PaddleOCR while Custom AI is cooling and returns to
  Custom AI when the remaining cooldown reaches zero.
- Custom AI submits at most two concurrent OCR calls while another API OCR
  provider retains the configured generic limit.
- A cooling Custom AI OCR error does not replace the current subtitle; a
  non-cooling error remains visible.

## Success criteria

- Replaying the observed 502-followed-by-200 sequence clears the false
  request-scoped 60-second pause.
- A genuine 429 remains protected for its complete cooldown interval.
- During a genuine AI cooldown, PaddleOCR continues feeding the translation
  pipeline and the saved AI OCR choice remains unchanged.
- No more than two Custom AI OCR requests are active at once.
- Cooldown errors do not cover the last useful subtitle.
- Known balance/auth failures are not retried every 60 seconds.
- Focused tests, the complete offline suite, compilation, and
  `git diff --check` pass without live API calls.
