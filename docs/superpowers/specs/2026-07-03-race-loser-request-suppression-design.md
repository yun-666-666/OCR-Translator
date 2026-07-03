# Race Loser Request Suppression Design

## Goal

Prevent slow losing race requests from receiving another request for the next
subtitle while their previous HTTP call is still running.

## Current problem

`_custom_ai_translate_race()` returns as soon as the first candidate succeeds.
`ThreadPoolExecutor.shutdown(wait=False, cancel_futures=True)` cannot cancel
candidate calls that already started. Those losers continue in background.

The outer translation worker then marks the logical request complete, allowing
the next subtitle to start another race. Without tracking, the same slow profile
can accumulate overlapping calls across subtitles, increasing relay load,
rate-limit risk, API cost, and tail latency.

## Chosen approach

Track active race futures by a non-secret profile identity:

- profile ID when present;
- otherwise normalized base URL, model, wire API, and reasoning effort.

Before submitting a race:

- build semantically compatible and healthy candidates as before;
- remove profiles that still have a future from an older race;
- if at least one profile is idle, use only idle candidates;
- if every candidate is marked busy, keep the original list as a defensive
  fallback rather than failing with no candidate.

On submission:

- mark the profile busy before or atomically with future registration;
- attach a done callback that removes the profile identity;
- roll back the busy mark if submission itself fails.

The winning identity is explicitly cleared before return. Slow losers remain
excluded until their real call finishes, then automatically rejoin later races.

## Scope and safety

- Applies only to handler-level `race` mode.
- Does not cancel or reuse a loser result for a different source subtitle.
- Does not expose or store API keys in tracking identity.
- Does not affect safe, stream, or none modes.
- Keeps existing semantic and cooldown filters.
- Handler close remains non-blocking with respect to already-running provider
  calls, matching existing behavior.

## Files

- `handlers/translation_handler.py`
- `tests/test_custom_ai.py`

## Verification

- first race returns the fast winner while the slow loser remains active;
- a second candidate query excludes only the running loser;
- a second translation uses the idle winner without launching another loser;
- releasing the loser automatically makes it eligible again;
- failed executor submission rolls back the busy marker;
- existing race speed, health, semantic, and cache tests remain green.
