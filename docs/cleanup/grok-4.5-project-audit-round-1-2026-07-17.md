# Grok 4.5 Project Audit — Round 1

## Review boundary

- Baseline commit: `42cc05b`
- Model: `grok/grok-4.5` (the working alias exposed by the configured endpoint)
- Access: read-only source review; Grok did not execute commands or modify files
- Excluded from prompts: local configuration, credentials, runtime logs, caches, build output, and release artifacts
- Successful review chunks: capture/local OCR and Custom AI HTTP transport
- Unsuccessful review chunks: two larger cache/orchestration requests ended at the relay with HTTP 504 and produced no findings; they are not treated as evidence

## Ranked findings with proof

### G1 — Streamed HTTP responses are not closed

**Grok verdict:** high confidence, high impact.

**Proof cited by Grok:**

- `custom_ai_transport.py:1025` — `_stream_post`
- `custom_ai_transport.py:1133` — `_stream_responses_post`
- `custom_ai_transport.py:1223` — `_parse_streaming_responses_response`
- `custom_ai_transport.py:1318` — `_parse_streaming_chat_response`

Both streaming request methods obtain a response with `stream=True` and then return parsed content without a `response.close()` or `finally` guard:

```python
response = self._post_with_output_limit_fallback(..., stream=True, ...)
response_json = self._parse_streaming_chat_response(response, stream_callback)
return response_json, duration
```

The Responses parser can also stop at a terminal server-sent event before consuming any trailing bytes:

```python
terminal_event_type = chunk_type
break
```

**Mechanism:** a `requests.Session` streaming response retains its pool connection until it is consumed or closed. Success returns, HTTP-error fallback, parser exceptions, and early terminal events can therefore keep connections checked out.

**Smallest safe change proposed by Grok:** close each streamed response in `finally`, using the existing `_close_response_quietly` helper.

**Proving test proposed by Grok:** fake streaming responses that count `close()` calls across success, HTTP error, parser failure, and early terminal-event paths.

**Local adjudication:** **accepted for round 1.** Current code confirms the missing lifetime guard. A regression test can prove the invariant without real network traffic.

### G2 — Compatibility retries abandon prior responses

**Grok verdict:** high confidence, amplified when `stream=True`.

**Proof cited by Grok:** `custom_ai_transport.py:81-231`, `_post_with_output_limit_fallback`.

The compatibility loop replaces `response` up to four times:

```python
response = send(request_payload)
...
response = send(request_payload)
continue
```

The same replacement appears for unsupported `prompt_cache_key`, reasoning effort, structured output, and output-limit fields. None closes the previous response before reassignment.

**Mechanism:** rejected streaming responses may retain unread bodies and pool connections; several capability fallbacks can multiply the leak within one logical request.

**Smallest safe change proposed by Grok:** close the response immediately before every compatibility resend while leaving the final returned response open for the caller.

**Proving test proposed by Grok:** make the existing multi-fallback fake responses count `close()` calls and assert every abandoned response is closed while the final response remains available.

**Local adjudication:** **accepted for round 1.** This is the same response-ownership invariant as G1 and can be fixed in the same focused commit.

### G3 — Local capture pacing can decay below the adaptive/user baseline

**Grok verdict:** high confidence, potentially high CPU/queue impact.

**Proof cited by Grok:** `worker_capture.py:391-465`, `run_capture_thread`.

```python
base_scan_interval = max(min_interval, scan_interval_ms / 1000.0)
...
else:
    current_scan_interval_sec = max(
        min_interval,
        current_scan_interval_sec * 0.95,
    )
```

`current_scan_interval_sec` starts at the fixed 50 ms floor. With a local OCR queue at or below 40% fullness, it is not clamped to `base_scan_interval`, so a configured/adaptive interval such as 200 ms can still capture at roughly 50 ms.

**Smallest safe change proposed by Grok:** never decay below `base_scan_interval`; preserve higher intervals only while queue pressure is elevated.

**Proving test proposed by Grok:** drive the capture loop with a 500 ms base and an empty queue, then assert the sleep/capture cadence never drops below 500 ms; also cover pressure and recovery.

**Local adjudication:** **accepted as a candidate for a later round.** The code supports the claim, but it is intentionally excluded from round 1 to keep one behavior change per commit.

### G4 — OCR preview performs synchronous PaddleOCR on the Tk thread

**Grok verdict:** high confidence that the work is synchronous; high UI-jank risk when preview is open.

**Proof cited by Grok:** `app_capture_ocr.py`, `preview_realtime_update` and `refresh_ocr_preview`.

The `root.after(500, ...)` callback captures, preprocesses, and calls `recognize_with_paddleocr(...)` before scheduling the next update. Tk callbacks execute on the UI thread.

**Local adjudication:** **deferred pending a focused UI-thread timing test.** The call path exists, but moving it across threads has wider lifecycle and model-serialization risk than G1/G2.

### G5 — Preview can duplicate OCR already performed by the worker path

**Grok verdict:** medium-high confidence in duplicate calls, medium-high resource risk.

**Proof cited by Grok:** `app_capture_ocr.py:refresh_ocr_preview` independently invokes PaddleOCR while `worker_capture.py:process_local_ocr_frame` also performs local OCR for the same region.

**Local adjudication:** **deferred pending measurement.** Duplicate entry points are present, but the exact overlap depends on whether translation and preview are active together. A call-count/CPU benchmark is required before changing ownership.

### G6 — A shared owned Session may be closed while another request is active

**Grok verdict:** high confidence in unsynchronized close, high concurrency risk.

**Proof cited by Grok:** `custom_ai_transport.py:close`, `_get_http_client`, and `_discard_owned_http_client_for_transport_error`.

Transport-reset recovery calls `self.close()` while other threads may still hold the shared Session. The current lock protects creation, not the whole in-flight lifetime.

**Local adjudication:** **deferred.** The structural race is plausible, but a deterministic two-request reproduction is required before introducing reference counting, deferred close, or generation tracking.

## Rejected or insufficient-evidence suggestions

Grok explicitly rejected style-only cleanups, dependency upgrades, unproven PaddleOCR engine thread-safety claims, a speculative stale-geometry result race, and streaming retry-policy changes whose intended semantics were not present in the reviewed files.

## Round 1 implementation decision

Implement only G1 and G2 as one bounded resource-lifetime invariant:

> Every response that is not returned to a caller must be closed exactly once by the layer that owns it.

The capture pacing issue G3 is the leading candidate for the next review/optimization round after this invariant is tested, committed, and re-reviewed by Grok 4.5.
