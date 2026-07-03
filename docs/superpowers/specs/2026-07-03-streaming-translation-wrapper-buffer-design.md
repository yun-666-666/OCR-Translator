# Streaming Translation Wrapper Buffer Design

## Goal

Prevent model wrapper prefixes and Markdown fences from flashing in live
subtitles while preserving near-immediate streaming for ordinary translations.

## Current behavior

The final translation result uses the structured output contract, but Chat and
Responses stream parsers send each accumulated raw partial to the UI callback.
A model that starts with `Translation:`, `Here is the translation:`, or a code
fence can therefore display that wrapper briefly before the clean final result
replaces it.

## Chosen approach

Wrap the caller's stream callback with a bounded, deterministic partial-output
filter.

The filter receives accumulated text and:

- holds text only while it is still a possible prefix of a known wrapper;
- strips a complete explanatory preamble;
- strips a standalone translation label once its newline is known;
- strips a valid opening Markdown fence and withholds a possible closing fence;
- emits immediately once ordinary text no longer matches a wrapper prefix;
- suppresses duplicate normalized partials.

The buffer is naturally bounded by the longest known prefix or fence header. It
does not wait for a sentence, token count, timer, or complete response.

## Preservation rules

- Inline `Translation: unavailable` is emitted intact once the space after the
  colon proves it is not a standalone wrapper line.
- If the source itself begins with a wrapper preamble or standalone label, that
  form is preserved.
- If the source itself is fenced, streaming output is not unwrapped.
- Dialogue quotes and ordinary whitespace remain untouched.
- Final output still passes through the existing full normalization contract.

## Scope

- Chat Completions streaming partials
- Responses API streaming partials
- callback passed by `translate()`

Out of scope:

- OCR streaming
- retrying interrupted streams
- changing the worker/UI coalescer

## Files

- `custom_ai.py`
- `tests/test_custom_ai.py`

## Verification

- wrapper fragments never reach the callback;
- ordinary `Hel` / `Hello` partials still emit promptly;
- standalone labels and fenced output are cleaned incrementally;
- inline labels and source-owned wrappers are preserved;
- Chat and Responses streams share the same behavior;
- full tests, compilation, and diff checks.
