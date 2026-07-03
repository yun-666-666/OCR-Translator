# Context Cache and Streaming Efficiency Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore immediate repeated-subtitle cache hits, bound context payload size, and prevent token-by-token Tk callback buildup.

**Architecture:** The translation handler will derive one current-source-aware context snapshot and use it for both provider requests and cache identity. The worker will expose a small per-request streaming coalescer that retains only the latest partial while one UI callback is pending.

**Tech Stack:** Python, `unittest`, Tk callback scheduling, existing `UnifiedTranslationCache`

---

### Task 1: Current-source-aware cache context

**Files:**
- Modify: `handlers/translation_handler.py`
- Test: `tests/test_custom_ai.py`

- [x] **Step 1: Write the failing cache-reuse test**

Create
`TranslationHandlerCustomAITests.test_immediately_repeated_custom_ai_subtitle_reuses_cache_with_context_enabled`.
Call `_custom_ai_translate("Save", 0.0)` twice and assert
`handler.custom_ai_provider.translate.call_count == 1`.

- [x] **Step 2: Run the focused test to verify RED**

Run: `python -m unittest tests.test_custom_ai.TranslationHandlerCustomAITests.test_immediately_repeated_custom_ai_subtitle_reuses_cache_with_context_enabled -v`

Expected: FAIL because the provider is called twice.

- [x] **Step 3: Implement current-source filtering**

Pass `cleaned_text` through `_get_custom_ai_cache_profile_and_params()` and
`_cache_params_for_profile()`. Extend `_get_custom_context_for_request()` with a
`current_source` argument and exclude matching source entries. Use that same
snapshot for provider payloads, cache reads/writes, in-flight keys, and race-mode
active-profile cache targets.

- [x] **Step 4: Run the focused test to verify GREEN**

Run the Step 2 command.

Expected: PASS with one provider call.

### Task 2: Bounded, refreshed context history

**Files:**
- Modify: `handlers/translation_handler.py`
- Test: `tests/test_custom_ai.py`

- [x] **Step 1: Write failing context tests**

Create `test_custom_ai_context_refreshes_repeated_source` with the sequence
`Save/保存`, `Save/另存`, `Store/保存`, `Quit/退出` and assert the final window is
`[("Save", "另存"), ("Store", "保存"), ("Quit", "退出")]`. Create
`test_custom_ai_request_context_obeys_character_budget` with two large pairs and
assert only the newest pair is returned and its content is at most 4,000
characters.

- [x] **Step 2: Run tests to verify RED**

Run: `python -m unittest tests.test_custom_ai.TranslationHandlerCustomAITests.test_custom_ai_context_refreshes_repeated_source tests.test_custom_ai.TranslationHandlerCustomAITests.test_custom_ai_request_context_obeys_character_budget -v`

Expected: FAIL because repeated sources are discarded and no character budget is
applied.

- [x] **Step 3: Implement refresh and budget selection**

Remove prior entries with the same source before appending the newest pair.
Select context newest-first under `CUSTOM_CONTEXT_CHAR_BUDGET = 4000`, truncate
only when the newest entry alone exceeds the ceiling, then restore chronological
order.

- [x] **Step 4: Run the focused tests to verify GREEN**

Run the Step 2 command.

Expected: both tests PASS.

### Task 3: Coalesced streaming UI updates

**Files:**
- Modify: `worker_threads.py`
- Test: `tests/test_latency_optimization.py`

- [x] **Step 1: Write the failing coalescing test**

Simulate three partials arriving before Tk runs callbacks. Assert that only one UI
callback is scheduled and it displays the third partial.

- [x] **Step 2: Run the test to verify RED**

Run: `python -m unittest tests.test_latency_optimization.LatencyTranslationCacheTests.test_streaming_translation_coalesces_pending_ui_updates -v`

Expected: FAIL because three callbacks are currently scheduled.

- [x] **Step 3: Implement the coalescer**

Add `_build_streaming_display_callback(app, translation_sequence)` with a lock,
`latest_text`, and `scheduled` state. The scheduled callback consumes the newest
text, applies the existing running/sequence guards, and allows a later partial to
schedule the next callback. Record the processed streamed value by sequence and
skip a matching final widget update while preserving final response bookkeeping.

- [x] **Step 4: Run stream tests to verify GREEN**

Run the Step 2 command plus the existing UI-thread streaming test.

Expected: both tests PASS.

### Task 4: Regression and completion verification

**Files:**
- Modify: `custom_ai.py`
- Modify: `handlers/translation_handler.py`
- Test: `tests/test_custom_ai.py`

- [x] Add `test_extract_usage_preserves_cached_input_token_counts` for both
  `prompt_tokens_details.cached_tokens` and
  `input_tokens_details.cached_tokens`.
- [x] Add `test_custom_ai_short_log_records_cached_input_tokens` and assert the
  numeric line `Cached Input Tokens: 1024` is written.
- [x] Preserve `cached_prompt_tokens` in `_extract_usage()` and write it to the
  bounded short log without adding request or response content.
- [x] Run both named tests and expect two passes.

### Task 5: Regression and completion verification

**Files:**
- Test: `tests/test_custom_ai.py`
- Test: `tests/test_latency_optimization.py`

- [x] Run `python -m unittest tests.test_custom_ai tests.test_latency_optimization -q`.
- [x] Run `python -m unittest discover -s tests`.
- [x] Run `python -m unittest discover`.
- [x] Run `python -m compileall -q handlers\translation_handler.py worker_threads.py tests`.
- [x] Run `git diff --check`.
- [x] Record commands, results, backup path, decisions, and remaining live-network limits in the latest `.codex/handoffs/` note.
