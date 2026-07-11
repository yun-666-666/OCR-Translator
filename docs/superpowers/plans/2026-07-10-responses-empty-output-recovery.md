# Responses Empty-Output Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recover one intermittent HTTP 200 Responses API result that contains no output text without hiding persistent provider failures.

**Architecture:** Keep recovery in `CustomAIProvider.translate`, where request contracts and parsing already live. Preserve the capability-aware initial parser, parse only an explicitly schema-free retry as plain text, retry at most once, and emit a content-free response-shape summary before retrying.

**Tech Stack:** Python 3.12, `unittest`, `unittest.mock`, existing Custom AI provider and debug logger.

---

### Task 1: Lock The Empty-Response Behavior

**Files:**
- Modify: `tests/test_custom_ai.py`

- [x] **Step 1: Add a failing auto-mode recovery test**

Create a fake `_post` that first returns a completed Responses object with a
reasoning item but no output text, then returns a plain `output_text` item.
Assert that the result is returned, exactly two calls occur, and the second
payload omits `response_format`.

- [x] **Step 2: Add a failing bounded-retry test**

Return the same empty completed response twice with structured output off.
Assert that `translate` raises `Responses API response did not contain output
text` and `_post` was called exactly twice.

- [x] **Step 3: Add safe-diagnostic assertions**

Patch `custom_ai.log_debug` and assert that the recovery log includes
`status=completed`, `reasoning_items=1`, and `output_text_items=0`, while
excluding the API key and a sentinel response-body value.

- [x] **Step 4: Add parser and contract-boundary coverage**

Assert that whitespace-only top-level `output_text` falls through to valid
nested output, non-finite token counts cannot interrupt diagnostics, strict
mode keeps its schema on retry, and a failed retry preserves its terminal
error.

- [x] **Step 5: Run the focused tests and verify RED**

Run:

```powershell
py -m unittest tests.test_custom_ai.CustomAIProviderTests.test_responses_empty_output_retries_once_without_auto_schema tests.test_custom_ai.CustomAIProviderTests.test_responses_empty_output_retry_is_bounded -v
```

Expected: both tests fail because no empty-output recovery exists.

### Task 2: Implement Bounded Recovery

**Files:**
- Modify: `custom_ai.py`
- Test: `tests/test_custom_ai.py`

- [x] **Step 1: Add a content-free response-shape helper**

Count known response item types and numeric usage fields without including
arbitrary response strings. Return a stable summary suitable for debug logs.

- [x] **Step 2: Preserve capability-aware initial parsing**

Keep `_parse_translation_response_text` unchanged for the initial response so
an internal compatibility retry remains authoritative. When empty-output
recovery explicitly removes the schema, parse only that retry as plain text.

- [x] **Step 3: Retry one empty Responses result**

In `translate`, catch only the exact missing-output parsing failure. Log the
safe summary, remove structured output only for `auto`, call the active request
one more time, re-run terminal validation, and parse using the retry payload.
Do not add a loop.

- [x] **Step 4: Run focused tests and verify GREEN**

Run the Task 1 command. Expected: both tests pass.

### Task 3: Regression Verification

**Files:**
- Create: `.codex/handoffs/2026-07-10_14-16-40.md`

- [x] **Step 1: Run the provider suite**

```powershell
py -m unittest tests.test_custom_ai -v
```

Expected: all tests pass.

- [x] **Step 2: Run full test discovery and syntax checks**

```powershell
py -m unittest discover -s tests
py -m unittest discover
py -B -m py_compile custom_ai.py tests\test_custom_ai.py
git diff --check
```

Expected: all commands exit zero.

- [x] **Step 3: Review the focused diff**

Compare `custom_ai.py` and `tests/test_custom_ai.py` with
`.codex/backups/2026-07-10_14-14-23/`. Confirm there is one bounded retry, no
response-body logging, and no unrelated changes.

- [x] **Step 4: Write the handoff**

Create `.codex/handoffs/2026-07-10_14-16-40.md` with root-cause evidence, files,
backup path, live probe results, test results, decisions, and remaining relay
limitations.
