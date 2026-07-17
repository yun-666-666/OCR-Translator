# API OCR Repeat Backoff Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use inline execution with test-driven development. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut redundant repeated remote OCR submissions while preserving bounded subtitle freshness, and avoid xAI's invalid endpoint probe.

**Architecture:** `worker_threads.run_api_ocr` already builds an OCR cache key before image encoding. Its setting-derived suffix becomes the repeat-backoff scope. `process_api_ocr_response` activates a short deadline only after the provider repeats the current subtitle. `custom_ai_capabilities` special-cases only the official xAI host.

**Tech Stack:** Python, Pillow-backed capture metadata, unittest.

---

### Task 1: Add repeat-backoff regression tests

**Files:**
- Modify: `tests/test_latency_optimization.py`
- Modify: `tests/test_custom_ai.py`

- [ ] **Step 1: Write a failing API OCR test**

Create an app fixture with a matching `api_ocr_repeat_backoff_scope` and future deadline. Assert `run_api_ocr` does not call the image encoder or thread pool and increments the repeat-backoff metric.

- [ ] **Step 2: Verify RED**

Run: `python -m unittest tests.test_latency_optimization.LatencyTranslationCacheTests.test_api_ocr_repeat_backoff_skips_same_scope_before_encoding -v`

Expected: FAIL because no repeat-backoff gate exists.

- [ ] **Step 3: Write a failing endpoint normalization test**

Assert `normalize_chat_completions_url_candidates("https://api.x.ai")` returns only `https://api.x.ai/v1/chat/completions`.

- [ ] **Step 4: Verify RED**

Run: `python -m unittest tests.test_custom_ai.CustomAITests.test_xai_host_only_url_uses_only_v1_chat_completions -v`

Expected: FAIL because the generic fallback includes the invalid root candidate.

### Task 2: Implement minimal production behavior

**Files:**
- Modify: `worker_threads.py`
- Modify: `custom_ai_capabilities.py`

- [ ] **Step 1: Add scope-bound repeat backoff**

Define a 0.75-second constant. Build the scope from `ocr_cache_key[1:]`; before image conversion, skip only while the matching scope deadline is in the future. After an identical valid API OCR result, set the matching scope and deadline. Record `api_ocr_repeat_backoff_skip` and use a coalesced log message.

- [ ] **Step 2: Canonicalize official xAI**

Parse the base URL hostname in the URL-candidate normalizer. Return only the `/v1/chat/completions` candidate for `api.x.ai`; leave generic hosts unchanged.

- [ ] **Step 3: Verify GREEN**

Run the two focused tests from Task 1. Expected: PASS.

### Task 3: Regression verification

**Files:**
- Modify: `tests/test_latency_optimization.py`
- Modify: `tests/test_custom_ai.py`

- [ ] **Step 1: Add changed-scope coverage**

Assert a different profile/language/region-derived scope still encodes and submits while the prior scope is cooling down.

- [ ] **Step 2: Run focused suites**

Run: `python -m unittest tests.test_latency_optimization tests.test_custom_ai -q`

- [ ] **Step 3: Run full suite and compile checks**

Run: `python -m unittest discover -s tests -q` and `python -m py_compile worker_threads.py custom_ai_capabilities.py tests/test_latency_optimization.py tests/test_custom_ai.py`.
