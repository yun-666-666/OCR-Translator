# Default Custom Prompt Optimization Design

## Goal

Replace the repetitive default custom translation prompt with a shorter instruction that improves natural subtitle phrasing while retaining meaning, tone, character voice, and terminology consistency.

## New Default Prompt

> Use context to resolve ambiguity. Translate naturally and concisely while preserving meaning, tone, and character voice. Keep names and game terms consistent.

## Rationale

The provider already supplies these system-level rules in `custom_ai.py`:

- act as an on-screen subtitle translation engine;
- translate from the selected source language to the selected target language;
- return only the translation;
- preserve or remove line breaks according to the active setting.

Repeating those rules in the editable custom prompt consumes tokens and creates two places where the same behavior can drift. The new prompt therefore contains only the style guidance not fully covered by the provider:

- use supplied subtitle context to resolve ambiguity;
- prefer natural, concise phrasing;
- preserve meaning, tone, and character voice;
- keep names and game terminology consistent.

This follows OpenAI's guidance to use clear, specific, concise instructions and avoid unnecessary statements:

- https://platform.openai.com/docs/guides/prompt-generation
- https://platform.openai.com/docs/guides/reasoning-best-practices

## Files and Behavior

### `app_logic.py`

Update `DEFAULT_CUSTOM_PROMPT` to the exact new text. Missing or empty prompt files will receive this value through the existing `load_custom_prompt` behavior.

### `custom_prompt.txt`

Update the workspace's shipped/current default prompt so this experiment copy uses the new text immediately.

### Startup tests

Update both startup test locations because root discovery and `tests` discovery use separate files:

- `test_custom_ai_startup.py`
- `tests/test_custom_ai_startup.py`

The tests will verify:

1. the exact default prompt text;
2. a missing prompt file is created with that default;
3. the shipped `custom_prompt.txt` matches the application constant;
4. the new prompt is shorter than the previous default.

## Compatibility

- Do not change `custom_ai.py` payload structure or provider behavior.
- Do not alter source/target language selection, context-window size, line-break handling, caching, or rate limiting.
- Do not automatically replace arbitrary user-written prompts.
- Existing users retain their custom text. This workspace's `custom_prompt.txt` is updated because it still contains the previous stock default.

## Verification

- Run the focused default-prompt tests red before implementation and green afterward.
- Run both startup test modules.
- Run full `tests` discovery and root discovery.
- Run Python compilation checks.
- Inspect one generated payload to confirm the new custom instruction appears once and the existing system output constraints remain intact.
