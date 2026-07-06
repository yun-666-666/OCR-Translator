# Custom AI Runtime Contracts - 2026-07-05

This release tightens the current Custom AI runtime path and updates the
project documentation to match the implementation.

## Highlights

- Aligns packaging metadata with Python 3.9-3.12 and the pinned
  `PySide6==6.7.3` runtime.
- Passes request-level translation timeouts through to Custom AI HTTP calls,
  including Chat Completions, Responses API, streaming, and race paths.
- Adds configurable Custom AI OCR image payload formats: WebP, PNG, and JPEG.
  WebP remains the default for compatibility.
- Includes OCR image format in cache mode keys to avoid cross-format cache
  reuse.
- Includes `structured_output_contract` in Custom AI translation cache keys.
- Separates normal Stop from app-exit shutdown so close-time cleanup does not
  schedule callbacks against a destroyed root window.
- Updates README, installation docs, manuals, developer guide, and
  troubleshooting guide to describe Tesseract/Custom AI OCR plus Custom
  AI/OpenAI-compatible translation as the active path.

## Verification

- Full unittest discovery: 361 tests passed.
- Setup entrypoint tests: 2 tests passed.
- Python compile check passed for changed runtime modules.
- `git diff --check` passed.
- Real Custom AI relay smoke test passed against a user-supplied relay using a
  temporary in-process API key only. The key was not written to repository files
  or release artifacts.

## Notes

- Legacy DeepL, Google, Gemini, OpenAI, and MarianMT provider material remains
  in the repository as compatibility/reference content.
- This release targets commit `77ef09c`.
