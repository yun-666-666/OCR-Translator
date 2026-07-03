# Structured Translation I/O Contract Design

## Goal

Improve translation consistency and prevent model wrapper text from entering
the cache without adding another network call to the normal path.

## Input problem

The current user prompt concatenates context using human-readable labels such as
`Source:`, `Translation:`, and `Current source text:`. A source subtitle can
contain those same labels or instruction-like text, making data boundaries less
clear to smaller OpenAI-compatible models.

## Structured input

Keep stable policy and language instructions in the system message. Encode the
user message as compact UTF-8 JSON:

```json
{
  "previous_approved_translations": [
    {"source": "Salut", "translation": "Hi"}
  ],
  "current_source": "Bonjour"
}
```

The system message explicitly states that the user message is data, that
`previous_approved_translations` is context only, and that only
`current_source` should be translated.

This gives unambiguous escaping for quotes, newlines, label-like content, and
instruction-like subtitles while using fewer repeated label tokens.

## Output contract

After a successful translation response and before returning it to the handler:

- trim BOM and surrounding whitespace;
- unwrap one complete Markdown code fence only when the source was not itself
  fenced;
- remove a standalone wrapper line such as `Translation:` or `译文：` when
  non-empty translated content follows;
- remove a small fixed set of explanatory preambles such as
  `Here is the translation:`;
- reject an output that becomes empty after cleanup;
- log when cleanup changed the output.

## Conservative boundaries

- Do not remove ordinary outer quotation marks; game dialogue may require them.
- Do not strip inline `Translation: unavailable`; it can be legitimate content.
- Do not remove trailing commentary heuristically.
- Do not perform language detection or reject source-identical names.
- Do not unwrap fences when the source itself is a fenced block.
- Apply the contract to Chat Completions, Responses, and final streaming output,
  but not OCR transcription or connection tests.

## Files

- `custom_ai.py`
- `tests/test_custom_ai.py`
- `test_custom_ai.py`

## Verification

- JSON round-trip for malicious label/instruction-like source text;
- paired context remains associated correctly;
- fenced and labeled model output is cleaned;
- Responses and streaming final results use the same contract;
- legitimate inline labels, quotes, and fenced source text remain unchanged;
- full test suites, compile, and diff checks.
