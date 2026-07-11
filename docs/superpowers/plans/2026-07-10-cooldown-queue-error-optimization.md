# Cooldown Queue and Gateway Error Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Coalesce repeated pending translations, prevent concurrent gateway failures from multiplying one cooldown, and keep HTML gateway pages out of logs and overlays.

**Architecture:** Keep changes at existing ownership boundaries. `worker_threads.py` coalesces exact pending OCR text before cache work; `custom_ai.py` owns provider cooldown state and HTTP error normalization. No provider failover, fuzzy matching, or prompt changes are introduced.

**Tech Stack:** Python 3.12, `unittest`, `unittest.mock`, standard-library `html.parser`, existing runtime metrics.

---

### Task 1: Coalesce identical pending translation requests

**Files:**
- Modify: `tests/test_latency_optimization.py`
- Modify: `worker_threads.py:1730-2052`

- [ ] **Step 1: Write the failing test**

Create an app with a scheduled pending request for `Same`, a mock translation
cache getter, and runtime metrics. Call `start_async_translation(app, "Same",
9)` and assert the pending sequence becomes 9, its original request timestamp
is preserved, the cache getter is not called, no timer is added, and
`pending_translation_coalesced` becomes 1.

- [ ] **Step 2: Run test to verify RED**

```powershell
py -m unittest tests.test_latency_optimization.LatencyTranslationCacheTests.test_matching_pending_translation_is_coalesced_before_cache_lookup -v
```

Expected: FAIL because the cache lookup still runs and no coalescing metric is
recorded.

- [ ] **Step 3: Write minimal implementation**

Add this helper and call it immediately after async infrastructure setup:

```python
def _coalesce_matching_pending_translation_request(app, text, sequence):
    pending = getattr(app, "pending_translation_request", None)
    if not isinstance(pending, dict) or pending.get("text") != text:
        return False
    pending["ocr_sequence_number"] = sequence
    _increment_metric(app, "pending_translation_coalesced")
    return True
```

Return before cache lookup when it returns true. Preserve request time,
deadline, generation, and timer state.

- [ ] **Step 4: Run focused tests to verify GREEN**

```powershell
py -m unittest tests.test_latency_optimization.LatencyTranslationCacheTests.test_matching_pending_translation_is_coalesced_before_cache_lookup tests.test_latency_optimization.LatencyTranslationCacheTests.test_pending_translation_reschedules_when_latest_request_has_earlier_deadline -v
```

### Task 2: Coalesce concurrent cooldown activations

**Files:**
- Modify: `tests/test_custom_ai.py:2435-2552`
- Modify: `custom_ai.py:1636-1677`

- [ ] **Step 1: Write failing tests**

Activate a 60-second cooldown at monotonic time 100 and again at 101. Assert
the second activation leaves the backoff counter at one and reports 59 seconds.
Add a variant with `Retry-After: 90` that extends the deadline to 191 without
growing the counter. Move the existing sequential-failure test's second failure
to time 161 so it still proves growth after expiry.

- [ ] **Step 2: Run tests to verify RED**

```powershell
py -m unittest tests.test_custom_ai.CustomAIProviderTests.test_concurrent_cooldown_failures_share_one_backoff_step tests.test_custom_ai.CustomAIProviderTests.test_concurrent_longer_retry_after_extends_deadline_without_backoff_growth tests.test_custom_ai.CustomAIProviderTests.test_rate_limit_cooldown_grows_after_consecutive_retry_after_failures -v
```

- [ ] **Step 3: Write minimal implementation**

Under the existing rate-limit lock, if `existing_until > now`, keep the later
of `existing_until` and `now + base_cooldown_seconds` without incrementing the
backoff counter. Otherwise retain the existing bounded exponential calculation
and set the new deadline from `now`.

- [ ] **Step 4: Run the three tests to verify GREEN**

Use the command from Step 2; all three tests must pass.

### Task 3: Compact HTML gateway errors

**Files:**
- Modify: `tests/test_custom_ai.py`
- Modify: `custom_ai.py:1-12,3309-3339`

- [ ] **Step 1: Write failing tests**

Build a 502 `text/html` response containing a Cloudflare title and body. Assert
the error includes the normalized title but excludes doctype, HTML tags, and
body content. Add a no-title HTML case for `_non_json_response_message` that
expects `Upstream returned an HTML error page`.

- [ ] **Step 2: Run tests to verify RED**

```powershell
py -m unittest tests.test_custom_ai.CustomAIProviderTests.test_html_error_response_is_compacted_to_title tests.test_custom_ai.CustomAIProviderTests.test_html_non_json_response_without_title_uses_generic_summary -v
```

- [ ] **Step 3: Write minimal implementation**

Import `HTMLParser`, add a private title collector, and add a helper that
recognizes `text/html` or leading HTML markup. Return either
`Upstream HTML error page: <title>` or the generic summary. Use the helper in
both response-error paths before redaction and length caps; leave JSON and plain
text behavior unchanged.

- [ ] **Step 4: Run the two tests to verify GREEN**

Use the command from Step 2; both tests must pass.

### Task 4: Verify and hand off

**Files:**
- Create: `.codex/handoffs/` using the filename returned by
  `Get-Date -Format 'yyyy-MM-dd_HH-mm-ss'`

- [ ] **Step 1: Run targeted suites**

```powershell
py -m unittest tests.test_latency_optimization -v
py -m unittest tests.test_custom_ai -v
```

- [ ] **Step 2: Run full verification**

```powershell
py -m unittest discover -s tests
py -m unittest discover
py -B -m py_compile custom_ai.py worker_threads.py tests\test_custom_ai.py tests\test_latency_optimization.py
git diff --check
```

- [ ] **Step 3: Inspect scope and create handoff**

Record latest-log evidence, backup path, files changed, red/green evidence,
verification results, and the remaining live-network validation gap. Do not
commit implementation files because they contain earlier uncommitted work that
must remain independently reviewable.
