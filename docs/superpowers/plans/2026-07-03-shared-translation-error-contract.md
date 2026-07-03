# Shared Translation Error Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Centralize translation error classification so failures cannot enter success state and valid short translations remain cacheable.

**Architecture:** `translation_utils.py` will own one strict prefix classifier. `TranslationHandler` and `worker_threads` will consume it, replacing broad substring checks and the worker’s duplicated tuple.

**Tech Stack:** Python, `unittest`

---

### Task 1: Reproduce worker error-state corruption

**Files:**
- Modify: `tests/test_latency_optimization.py`
- Modify: `worker_threads.py`
- Modify: `translation_utils.py`

- [x] **Step 1: Write the failing worker test**

```python
def test_custom_ai_errors_do_not_enter_success_state(self):
    worker_threads = import_worker_threads_for_tests()
    for error_result in [
        "Custom AI translation error: ValueError - upstream busy",
        "AI model profile for translation is missing.",
    ]:
        with self.subTest(error_result=error_result):
            displayed = []
            app = types.SimpleNamespace(
                last_displayed_translation_sequence=0,
                last_successful_translation_time=123.0,
                last_local_ocr_submitted_text="Hello",
                last_local_ocr_submitted_norm="hello",
                last_local_ocr_submitted_scope=("scope",),
                update_translation_text=displayed.append,
            )
            worker_threads.process_translation_response(
                app, error_result, 1, "Hello", 0
            )
            self.assertEqual(displayed, [f"Translation Error:\n{error_result}"])
            self.assertEqual(app.last_successful_translation_time, 123.0)
            self.assertEqual(app.last_displayed_translation_sequence, 1)
            self.assertIsNone(app.last_local_ocr_submitted_text)
```

- [x] **Step 2: Verify RED**

Run:
`python -m unittest tests.test_latency_optimization.LatencyTranslationCacheTests.test_custom_ai_errors_do_not_enter_success_state -v`

Expected: FAIL because current Custom AI errors are displayed raw and enter success
state.

- [x] **Step 3: Implement the strict shared classifier**

```python
TRANSLATION_ERROR_PREFIXES = (
    "err:",
    "translation error:",
    "custom ai translation error:",
    "ai model profile for translation is missing",
    "error: unknown translation model",
    "no translation for model:",
    "google api error:",
    "google api key missing:",
    "google translate api key missing",
    "google translate api client not initialized",
    "requests library not available for google translate",
    "google translate api returned unexpected result:",
    "google translate api request error:",
    "google translate api error:",
    "google client init error:",
    "deepl api libraries not available",
    "deepl api key missing",
    "deepl client init error:",
    "deepl api client not initialized",
    "deepl api returned empty or invalid result",
    "deepl api fallback returned empty or invalid result",
    "deepl api error:",
    "marianmt error:",
    "marianmt not initialized",
    "marianmt language pair not determined:",
    "marianmt translator not initialized",
    "marianmt translation error:",
)

def is_translation_error_result(value):
    if not isinstance(value, str):
        return True
    normalized = value.lstrip().casefold()
    return normalized.startswith(TRANSLATION_ERROR_PREFIXES)
```

- [x] **Step 4: Use the classifier in the worker**

Import `is_translation_error_result`, delete the local prefix tuple, and classify
with the shared function. Remove the assignment to
`last_successful_translation_time` from the error branch.

- [x] **Step 5: Verify GREEN**

Run the Step 2 command and expect PASS.

### Task 2: Preserve legitimate translated words

**Files:**
- Modify: `handlers/translation_handler.py`
- Modify: `tests/test_custom_ai.py`

- [x] **Step 1: Write the failing handler test**

```python
def test_translation_error_classifier_accepts_legitimate_short_results(self):
    handler = TranslationHandler(object())
    for result in ["Missing", "Failed", "Not available"]:
        with self.subTest(result=result):
            self.assertFalse(handler._is_error_message(result))
    self.assertTrue(
        handler._is_error_message(
            "Custom AI translation error: ValueError - upstream busy"
        )
    )
```

- [x] **Step 2: Verify RED**

Run:
`python -m unittest tests.test_custom_ai.TranslationHandlerCustomAITests.test_translation_error_classifier_accepts_legitimate_short_results -v`

Expected: FAIL because broad substring checks reject all three valid translations.

- [x] **Step 3: Delegate handler classification**

Import `is_translation_error_result` and replace `_is_error_message()` with:

```python
def _is_error_message(self, text):
    return is_translation_error_result(text)
```

- [x] **Step 4: Verify GREEN and focused regressions**

Run both new tests plus existing Custom AI cache/error tests. Expect all PASS.

### Task 3: Full verification and handoff

**Files:**
- Create: `.codex/handoffs/2026-07-03_18-00-19.md`

- [x] Run `python -m unittest tests.test_custom_ai -q`.
- [x] Run `python -m unittest tests.test_latency_optimization -q`.
- [x] Run `python -m unittest discover -s tests -q`.
- [x] Run `python -m unittest discover -q`.
- [x] Run `python -m compileall -q translation_utils.py handlers worker_threads.py tests`.
- [x] Run `git diff --check`.
- [x] Repeat the original probe and record corrected display, success-time, dedup,
  and valid-word classification evidence in the handoff.
