# Race Profile Health and Semantics Design

## Goal

Make `race` mode choose only semantically equivalent, currently usable
translation profiles so speed does not silently reduce requested quality and a
cooling active relay does not block a healthy alternative.

## Current problems

Race candidates currently match only the model string.

That allows:

- a low-reasoning profile to win a race started from a high-reasoning profile;
- Chat Completions and Responses profiles with different request semantics to
  compete;
- the winner's result to be stored under the active profile cache identity even
  when the quality configuration differs;
- a cooling active profile to keep the scheduler gated even if another
  equivalent profile is healthy;
- a single healthy alternative to be discovered but the single-candidate fast
  path to call the active profile anyway.

## Semantic race signature

Profiles are race-compatible only when all fields match after normalization:

- model;
- wire API;
- reasoning effort (`reasoning_effort` or legacy
  `model_reasoning_effort`).

Base URL, API key, profile ID, and display name may differ because those are the
relay alternatives race mode is intended to compare.

## Cooldown behavior

- Build the semantically compatible candidate list with the active profile
  first.
- Read provider cooldown for each candidate.
- If one or more candidates are healthy, race only healthy candidates.
- If all candidates are cooling, keep the compatible list so existing errors
  and cooldown reporting remain available.
- In race mode, the scheduler cooldown is the minimum cooldown across compatible
  candidates; one healthy candidate therefore yields zero delay.
- In non-race modes, keep active-profile cooldown behavior unchanged.

## Single-candidate behavior

If health filtering leaves one candidate, call that candidate, not
unconditionally the active profile. Return its cache identity and profile as the
winner.

## Files

- `handlers/translation_handler.py`
- `tests/test_custom_ai.py`

## Verification

- different model, wire API, and reasoning configurations are excluded;
- a cooling active profile does not gate a healthy equivalent profile;
- a cooling candidate is excluded when healthy candidates exist;
- one healthy alternative is actually called;
- all-cooling candidates report the shortest remaining cooldown;
- safe/stream modes retain active-only cooldown;
- existing fastest-profile race and cache tests remain green.
