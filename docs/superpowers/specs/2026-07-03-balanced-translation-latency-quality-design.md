# Balanced Translation Latency and Quality Design

## Goal

Reduce the delay before a new subtitle translation starts, make the base submission
interval user-configurable, and improve terminology and voice consistency by giving
the translation model paired source/translation history.

## Current evidence

- Custom AI translations are single-flight and retain only the latest pending
  subtitle. Those protections prevent stale work, preserve context order, and limit
  provider pressure.
- The current short-text submission interval is hard-coded to at least 1.0 seconds
  even though the application's existing base interval is 0.3 seconds.
- Custom AI context contains only prior source subtitles. The provider cannot see
  the accepted translations whose terminology and voice it should continue.
- Existing debug messages expose queue delay, provider duration, translation total,
  and display scheduling separately, but do not emit one sanitized end-to-end
  timing summary.

## Design

### User-configurable submission interval

Add `custom_ai_submit_interval_ms` with a default of 300 milliseconds. Expose it in
Settings as a spinbox from 0 to 5000 milliseconds in 50-millisecond increments.
Persist it in the INI file and include it in the example configuration.

For subtitle-sized input (80 characters or fewer), use the configured value
directly. For longer OCR text, add up to three seconds based on text length. Provider
rate-limit cooldown remains authoritative, single-flight execution remains enabled,
and the pending slot continues to retain only the newest subtitle.

Invalid or missing configuration values fall back to 300 milliseconds. Runtime
values are clamped to the same UI range.

### Sanitized latency measurement

Carry the time at which a subtitle first enters translation scheduling through a
pending request and into the worker. Emit a single `LATENCY: translation timing`
debug line after completion containing only numeric queue, provider/worker, and
end-to-end durations plus the translation sequence. Do not include subtitle text,
API keys, URLs, prompts, or response bodies in this new metric.

### Paired translation context

Store successful Custom AI context entries as `(source_text, translated_text)`
pairs. Cache hits also add the cached pair. Build provider prompts with explicitly
labeled previous source and translation lines, followed by the current source text.
The instruction states that history is reference material and only the current text
should be translated.

The context-size setting continues to control the number of prior subtitle pairs.
Duplicate source or duplicate translated output is not added twice. Language,
profile, prompt, or context-setting changes continue to clear active context.

Cache and in-flight keys include the paired context, so results produced from
different accepted translation histories cannot collide.

## Compatibility and error handling

- Headless tests and older callers without the new Tk variable use the existing
  `min_translation_interval` fallback.
- The payload builder accepts legacy string-only context items defensively.
- Existing SQLite cache rows remain readable; no database schema change is needed.
- No real API request is required for automated verification.

## Verification

- RED/GREEN tests for default, custom, clamped, and long-text intervals.
- RED/GREEN tests for configuration defaults, bilingual labels, and save wiring.
- RED/GREEN tests for pending timing propagation and sanitized timing output.
- RED/GREEN tests for paired context retention, duplicate suppression, cache-hit
  context, payload formatting, and context-sensitive cache keys.
- Full unit discovery, root discovery, compileall, and `git diff --check`.
