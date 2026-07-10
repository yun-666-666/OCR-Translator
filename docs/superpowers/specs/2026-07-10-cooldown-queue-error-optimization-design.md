# Cooldown Queue and Gateway Error Optimization Design

## Runtime evidence

The latest application session ran from 2026-07-10 12:24:03 to 12:25:56.
It recorded 1,117 captures and 578 PaddleOCR runs, so the existing exact-frame
suppression is working. The remaining translation path produced 570 pending
request updates. Reconstructing complete multiline source texts showed that 468
of those updates were adjacent exact duplicates, so 82.1 percent can be
coalesced without changing which subtitle is eventually translated.

Two already in-flight gateway requests then returned the same Cloudflare HTTP
502 outage one second apart. The first response started a 60-second cooldown;
the second response counted as another sequential failure and expanded the
same outage to 120 seconds. The raw HTML error page was also copied into debug
logs and the subtitle display.

The latest successful Responses calls reported a 0.90 cached-input ratio. This
design therefore leaves prompt-cache routing and context budgeting unchanged.

## Goals

1. Avoid repeated cache probes, pending-request writes, timer bookkeeping, and
   logs when the newest OCR text exactly matches the request already waiting.
2. Treat late failures from requests that were concurrently in flight as one
   cooldown event while preserving exponential backoff across failures that
   occur after a cooldown expires.
3. Replace upstream HTML error bodies with a short diagnostic that remains
   useful in logs and safe to show in the translation overlay.

## Non-goals

- Automatic provider switching or profile selection.
- Changes to model prompts, context-window budgets, or cache-key routing.
- Fuzzy OCR-text matching.
- Live paid API requests during automated verification.

## Design

### Exact pending-request coalescing

At the start of `start_async_translation`, before the instant translation-cache
lookup, inspect `app.pending_translation_request`. If it is a dictionary and its
text exactly matches the new OCR text:

- update only `ocr_sequence_number` to the newest sequence;
- preserve the original `requested_at_monotonic`, timer deadline, scheduled
  flag, and timer generation;
- increment a `pending_translation_coalesced` runtime counter;
- return without another cache lookup, queue log, or timer operation.

A different text continues through the existing cache, in-flight, cooldown,
and latest-request queue logic. This preserves instant cache display for every
new subtitle and preserves latest-wins behavior.

### Concurrent cooldown coalescing

Inside the existing rate-limit lock, inspect the current cooldown deadline
before incrementing the backoff counter. When a cooldown is already active:

- do not increment the sequential-failure counter;
- retain the later of the existing deadline and `now + Retry-After` (or the
  default cooldown);
- return the actual remaining cooldown duration.

When no cooldown is active, increment the counter and calculate the existing
bounded exponential backoff. Therefore two responses from the same in-flight
outage share one cooldown, while a new request that fails after that cooldown
expires advances from 60 to 120 seconds as intended.

### Compact HTML gateway errors

Before `_response_error_message` and `_non_json_response_message` include a
response body, detect an HTML document from its content type or leading
`<!doctype html` / `<html` markup. Extract and whitespace-normalize the document
title when present, cap it to a short diagnostic length, and otherwise use
`Upstream returned an HTML error page`. Never include the HTML body or markup.

JSON API errors and plain-text errors retain their current handling and secret
redaction.

## Tests

- A matching pending subtitle updates its sequence but performs no translation
  cache lookup and schedules no additional timer.
- A different pending subtitle still performs the normal cache/queue path.
- Two cooldown activations before the first deadline keep one backoff step.
- A failure after the cooldown deadline advances the backoff step.
- A longer concurrent `Retry-After` may extend the active deadline without
  increasing the backoff counter.
- HTML 502 responses contain the normalized title but no `<!DOCTYPE`, `<html>`,
  or body content in the resulting error.
- Plain-text and JSON error diagnostics remain unchanged.

## Success criteria

- Replaying the observed pending-request sequence would coalesce 468 of 570
  updates while preserving the final text and newest OCR sequence.
- Concurrent failures from the observed outage produce one 60-second cooldown,
  not 60 then 120 seconds.
- A genuinely later failed attempt after cooldown expiry produces the next
  120-second backoff.
- Upstream HTML markup cannot reach debug error messages or the subtitle
  display.
- Targeted tests, the full `tests` suite, root test discovery, compilation, and
  `git diff --check` all pass.
