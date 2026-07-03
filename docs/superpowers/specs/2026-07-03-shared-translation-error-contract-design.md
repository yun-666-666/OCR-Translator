# Shared Translation Error Contract Design

## Goal

Prevent provider failures from being treated as successful translations while
allowing legitimate short translations such as “Missing”, “Failed”, or
“Not available” to participate in cache and context.

## Root cause

The translation handler and worker independently classify string results:

- the handler uses broad substring markers such as `missing`, `failed`, and
  `not available`
- the worker uses an older fixed prefix tuple that does not include current
  Custom AI or missing-profile errors

The two contracts drifted. A Custom AI error therefore enters the worker success
path, updates `last_successful_translation_time`, and remembers the failed source
as a successful local OCR submission. Conversely, valid translated words matching
the broad handler substrings are not cached.

## Shared classifier

`translation_utils.py` will expose `is_translation_error_result(value)`. It will:

- treat non-string values as invalid
- strip leading whitespace and compare case-insensitively
- match only explicit error prefixes produced by translation backends
- avoid generic substrings inside legitimate translations

The prefix inventory covers current Custom AI errors, missing-profile errors, and
the legacy provider errors already present in `TranslationHandler`.

## Worker state behavior

For a classified error, the worker will:

- show the existing visible `Translation Error:` message
- advance `last_displayed_translation_sequence` so an older result cannot replace
  the newer error
- leave `last_successful_translation_time` unchanged
- clear local OCR submission deduplication state so the same subtitle can retry

Successful translations retain existing behavior.

## Handler cache behavior

`TranslationHandler._is_error_message()` will delegate to the shared classifier.
Known error strings remain excluded from cache and context. Legitimate output that
merely contains words such as “missing” or “failed” is accepted.

## Alternatives rejected

- Adding only the two currently missing prefixes would leave duplicate contracts
  and guarantee future drift.
- A typed `TranslationOutcome` would be cleaner long-term but would change the
  public string-returning API used by `app_logic.py` and tests; it is too broad for
  this correction.

## Verification

Tests reproduce current Custom AI and missing-profile errors through
`process_translation_response()`, assert error-state invariants, and prove valid
translations named `Missing`, `Failed`, and `Not available` are not classified as
errors.
