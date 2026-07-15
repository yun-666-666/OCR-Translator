# AI OCR Cooldown Recovery and PaddleOCR Fallback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep translation running through PaddleOCR during Custom AI OCR cooldowns, recover immediately from false request-scoped 5xx cooldowns, bound AI OCR concurrency, and classify profile failures by recovery type.

**Architecture:** Keep transport cooldown ownership in `custom_ai_transport.py`, profile-health policy in `handlers/translation_requests.py`, and OCR execution routing in `worker_threads.py`. The saved OCR selection is never changed: the worker calculates an effective per-frame model and temporarily uses the existing PaddleOCR path while the active Custom AI OCR profile is cooling.

**Tech Stack:** Python 3.12, `unittest`, `unittest.mock`, existing Custom AI transport/profile state, existing PaddleOCR worker pipeline.

---

### Task 0: Protect the current checkout

**Files:**
- Back up: `custom_ai_transport.py`
- Back up: `handlers/translation_requests.py`
- Back up: `worker_threads.py`
- Back up: `tests/test_custom_ai.py`
- Back up: `tests/test_latency_optimization.py`

- [ ] **Step 1: Confirm the worktree is clean except for this plan**

Run:

```powershell
git status --short --branch
```

Expected: the branch is ahead of its remote and only the new plan is untracked.

- [ ] **Step 2: Create one timestamped backup**

Run:

```powershell
$stamp = Get-Date -Format 'yyyy-MM-dd_HH-mm-ss'
$backup = Join-Path '.codex/backups' $stamp
$paths = @(
  'custom_ai_transport.py',
  'handlers/translation_requests.py',
  'worker_threads.py',
  'tests/test_custom_ai.py',
  'tests/test_latency_optimization.py'
)
foreach ($path in $paths) {
  $destination = Join-Path $backup $path
  New-Item -ItemType Directory -Force -Path (Split-Path $destination) | Out-Null
  Copy-Item -LiteralPath $path -Destination $destination
}
$backup
```

Expected: a single `.codex/backups/YYYY-MM-DD_HH-mm-ss/` path containing all five existing files with their relative paths preserved.

### Task 1: Clear only request-scoped cooldowns after success

**Files:**
- Modify: `tests/test_custom_ai.py` in `CustomAIProviderTests`
- Modify: `custom_ai_transport.py:391-395`

- [ ] **Step 1: Write the failing transport tests**

Add two tests to `CustomAIProviderTests`:

```python
def test_success_clears_request_scoped_cooldown(self):
    class Response:
        status_code = 502
        headers = {"Retry-After": "60"}

    provider = CustomAIProvider(http_client=object())
    profile = {
        "base_url": "https://relay.example/v1",
        "api_key": "secret",
        "wire_api": "chat_completions",
        "model": "vision-model",
    }

    with patch("custom_ai_transport.time.monotonic", return_value=100.0):
        provider._activate_rate_limit_cooldown(
            profile, Response(), "Upstream gateway unavailable"
        )
    provider._note_rate_limit_success(profile)

    with patch("custom_ai_transport.time.monotonic", return_value=101.0):
        self.assertEqual(provider.get_cooldown_remaining(profile), 0.0)

def test_success_preserves_transport_scoped_rate_limit_cooldown(self):
    class Response:
        status_code = 429
        headers = {"Retry-After": "60"}

    provider = CustomAIProvider(http_client=object())
    profile = {
        "base_url": "https://relay.example/v1",
        "api_key": "secret",
        "wire_api": "chat_completions",
        "model": "vision-model",
    }

    with patch("custom_ai_transport.time.monotonic", return_value=100.0):
        provider._activate_rate_limit_cooldown(
            profile, Response(), "Rate limit exceeded"
        )
    provider._note_rate_limit_success(profile)

    with patch("custom_ai_transport.time.monotonic", return_value=101.0):
        self.assertGreater(provider.get_cooldown_remaining(profile), 0.0)
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```powershell
py -m unittest tests.test_custom_ai.CustomAIProviderTests.test_success_clears_request_scoped_cooldown tests.test_custom_ai.CustomAIProviderTests.test_success_preserves_transport_scoped_rate_limit_cooldown -v
```

Expected: the request-scoped test fails because `_note_rate_limit_success()` does not remove the active request cooldown; the 429 test passes.

- [ ] **Step 3: Implement scoped recovery**

Replace `_note_rate_limit_success()` with:

```python
def _note_rate_limit_success(self, profile):
    cache_keys = self._cooldown_cache_keys_for_profile(profile)
    request_key = self._request_cooldown_cache_key(profile)
    cleared_request_cooldown = False
    with self._rate_limit_lock:
        for cache_key in cache_keys:
            self._rate_limit_backoff_counts.pop(cache_key, None)
        cleared_request_cooldown = (
            self._rate_limit_cooldowns.pop(request_key, None) is not None
        )
    if cleared_request_cooldown:
        values = profile if isinstance(profile, dict) else {}
        _log_debug(
            "RECOVERY: custom_ai request-scoped cooldown cleared after success "
            f"provider={values.get('name', 'Custom AI')}"
        )
```

- [ ] **Step 4: Verify GREEN and adjacent cooldown tests**

Run:

```powershell
py -m unittest tests.test_custom_ai.CustomAIProviderTests.test_success_clears_request_scoped_cooldown tests.test_custom_ai.CustomAIProviderTests.test_success_preserves_transport_scoped_rate_limit_cooldown tests.test_custom_ai.CustomAIProviderTests.test_successful_response_resets_rate_limit_backoff_counter tests.test_custom_ai.CustomAIProviderTests.test_concurrent_cooldown_failures_share_one_backoff_step -v
```

Expected: all four tests pass.

### Task 2: Classify profile failures and share health with OCR

**Files:**
- Modify: `tests/test_custom_ai.py` in `TranslationHandlerCustomAITests`
- Modify: `handlers/translation_requests.py:20-24,189-219,587-604,1041-1047,1102-1108`

- [ ] **Step 1: Write failing classifier and OCR health tests**

Add tests that instantiate `TranslationHandler` with a fake profile manager and provider. Assert:

```python
self.assertEqual(handler._custom_ai_profile_failure_cooldown_seconds(
    "Chat completions request failed (HTTP 402): INSUFFICIENT_BALANCE"
), 300.0)
self.assertEqual(handler._custom_ai_profile_failure_cooldown_seconds(
    "precharge failed because balance is insufficient"
), 300.0)
self.assertEqual(handler._custom_ai_profile_failure_cooldown_seconds(
    "API response did not contain message content"
), 10.0)
self.assertEqual(handler._custom_ai_profile_failure_cooldown_seconds(
    "HTTPSConnectionPool read timed out"
), 15.0)
self.assertEqual(handler._custom_ai_profile_failure_cooldown_seconds(
    "Chat completions request failed (HTTP 503)"
), 15.0)
self.assertEqual(handler._custom_ai_profile_failure_cooldown_seconds(
    "unclassified provider failure"
), 30.0)
```

Add an OCR failure test whose provider `recognize()` raises HTTP 402 and whose `mark_profile_unavailable()` records its arguments. Assert that `perform_ocr()` returns an `<e>:` error and marks the active OCR profile for 300 seconds. Add an OCR success test whose provider returns `("Recognized", {}, 0.2)` and assert `mark_profile_available(profile)` is called.

- [ ] **Step 2: Run the tests and verify RED**

Run:

```powershell
py -m unittest tests.test_custom_ai.TranslationHandlerCustomAITests.test_profile_failure_cooldown_classifier tests.test_custom_ai.TranslationHandlerCustomAITests.test_custom_ai_ocr_failure_marks_profile_with_classified_cooldown tests.test_custom_ai.TranslationHandlerCustomAITests.test_custom_ai_ocr_success_marks_profile_available -v
```

Expected: failures because the classifier does not exist and OCR does not update profile-unavailable state.

- [ ] **Step 3: Add the pure classifier and marker helper**

Add to `TranslationRequestsMixin`:

```python
def _custom_ai_profile_failure_cooldown_seconds(self, error_text):
    lowered = str(error_text or "").strip().lower()
    permanent_markers = (
        "http 401", "http 402", "http 403", "insufficient_balance",
        "insufficient balance", "precharge", "预扣费", "余额不足",
    )
    if any(marker in lowered for marker in permanent_markers):
        return 300.0
    transient_markers = (
        "timed out", "timeout", "connection", "non-json", "non json",
        "bad gateway", "gateway", "http 500", "http 502", "http 503",
        "http 504",
    )
    if any(marker in lowered for marker in transient_markers):
        return 15.0
    empty_markers = (
        "did not contain message content", "returned an invalid translation",
        "empty response", "empty content",
    )
    if any(marker in lowered for marker in empty_markers):
        return 10.0
    deterministic_markers = (
        "http 404", "invalid model", "model not found", "invalid endpoint",
    )
    if any(marker in lowered for marker in deterministic_markers):
        return 300.0
    return 30.0

def _mark_custom_ai_profile_failure(self, profile, error_text):
    marker = getattr(self.custom_ai_provider, "mark_profile_unavailable", None)
    if callable(marker):
        marker(
            profile,
            error_text,
            seconds=self._custom_ai_profile_failure_cooldown_seconds(error_text),
        )
```

Use `_mark_custom_ai_profile_failure()` in both translation failover failure branches and in `perform_ocr()`'s exception branch. After a successful OCR result, call `mark_profile_available(profile)` when available.

- [ ] **Step 4: Expose the active OCR cooldown**

Add:

```python
def get_active_custom_ai_ocr_cooldown_seconds(self):
    profiles = getattr(self.app, "custom_ai_profiles", None)
    getter = getattr(profiles, "get_active_profile", None)
    if not callable(getter):
        return 0.0
    return self._custom_ai_profile_cooldown_seconds(getter("ocr"))
```

Add a test that returns `12.5` from the provider cooldown getter and asserts this public handler method returns `12.5` for the active OCR profile.

- [ ] **Step 5: Verify GREEN and failover regression coverage**

Run:

```powershell
py -m unittest tests.test_custom_ai.TranslationHandlerCustomAITests tests.test_custom_ai.CostProtectedProfileFailoverHandlerTests -v
```

Expected: all tests pass; existing failover tests continue to observe skipped cooling profiles.

### Task 3: Route cooling AI OCR through PaddleOCR and cap concurrency

**Files:**
- Modify: `tests/test_latency_optimization.py` in `LatencyTranslationCacheTests`
- Modify: `worker_threads.py:40-50,138-240,409-570`

- [ ] **Step 1: Write failing effective-model tests**

Add tests for a new helper:

```python
def test_custom_ai_ocr_cooldown_uses_paddleocr_for_current_frame(self):
    worker_threads = import_worker_threads_for_tests()
    app = types.SimpleNamespace(
        translation_handler=types.SimpleNamespace(
            get_active_custom_ai_ocr_cooldown_seconds=lambda: 42.0
        )
    )
    self.assertEqual(
        worker_threads._effective_ocr_model_for_frame(app, "custom_ai"),
        "paddleocr",
    )

def test_custom_ai_ocr_recovers_automatically_after_cooldown(self):
    worker_threads = import_worker_threads_for_tests()
    app = types.SimpleNamespace(
        translation_handler=types.SimpleNamespace(
            get_active_custom_ai_ocr_cooldown_seconds=lambda: 0.0
        )
    )
    self.assertEqual(
        worker_threads._effective_ocr_model_for_frame(app, "custom_ai"),
        "custom_ai",
    )
```

- [ ] **Step 2: Write failing concurrency and display tests**

Add a test with `active_ocr_calls={1, 2}`, configured generic limit `8`, and provider `custom_ai`. Assert `run_api_ocr()` performs no conversion or submission. Add the same setup for a non-Custom-AI API provider and assert it still submits under the generic limit.

Add a `process_api_ocr_response()` test where the result starts with `<e>:`, the provider is `custom_ai`, and the handler reports an active cooldown. Assert `update_translation_text()` is not called and `last_displayed_batch_sequence` advances. Add a non-cooling error variant and assert the visible error behavior remains.

- [ ] **Step 3: Run the new tests and verify RED**

Run:

```powershell
py -m unittest tests.test_latency_optimization.LatencyTranslationCacheTests.test_custom_ai_ocr_cooldown_uses_paddleocr_for_current_frame tests.test_latency_optimization.LatencyTranslationCacheTests.test_custom_ai_ocr_recovers_automatically_after_cooldown tests.test_latency_optimization.LatencyTranslationCacheTests.test_custom_ai_ocr_uses_provider_specific_concurrency_limit tests.test_latency_optimization.LatencyTranslationCacheTests.test_other_api_ocr_keeps_generic_concurrency_limit tests.test_latency_optimization.LatencyTranslationCacheTests.test_cooling_custom_ai_ocr_error_preserves_current_translation tests.test_latency_optimization.LatencyTranslationCacheTests.test_non_cooling_custom_ai_ocr_error_remains_visible -v
```

Expected: failures because effective fallback, provider-specific concurrency, and cooling-error preservation are absent.

- [ ] **Step 4: Add effective per-frame routing**

Add:

```python
MAX_CUSTOM_AI_OCR_CONCURRENCY = 2

def _custom_ai_ocr_cooldown_seconds(app):
    handler = getattr(app, "translation_handler", None)
    getter = getattr(handler, "get_active_custom_ai_ocr_cooldown_seconds", None)
    if not callable(getter):
        return 0.0
    try:
        return max(0.0, float(getter()))
    except Exception as error:
        log_debug(
            "Custom AI OCR cooldown lookup failed: "
            f"{type(error).__name__}"
        )
        return 0.0

def _effective_ocr_model_for_frame(app, selected_model):
    if selected_model != "custom_ai":
        return selected_model
    remaining = _custom_ai_ocr_cooldown_seconds(app)
    if remaining <= 0.0:
        return selected_model
    log_debug_coalesced(
        "custom-ai-ocr-paddle-fallback",
        "RECOVERY: Custom AI OCR cooling; using PaddleOCR temporarily "
        f"remaining={remaining:.1f}s",
        interval_seconds=5.0,
    )
    return PADDLEOCR_MODEL_CODE
```

In `run_ocr_thread()`, retain `selected_ocr_model = app.get_ocr_model_setting()` for user intent and assign `ocr_model = _effective_ocr_model_for_frame(app, selected_ocr_model)` before cache and routing decisions.

- [ ] **Step 5: Add provider-specific backpressure**

Add:

```python
def _api_ocr_concurrency_limit(app, provider_name):
    configured = max(1, int(getattr(app, "max_concurrent_ocr_calls", 1) or 1))
    if provider_name == "custom_ai":
        return min(configured, MAX_CUSTOM_AI_OCR_CONCURRENCY)
    return configured
```

Use the returned limit in `run_api_ocr()` before image conversion and include that effective limit in the existing skip log.

- [ ] **Step 6: Preserve the current subtitle during cooling errors**

In the `<e>:` branch of `process_api_ocr_response()`, before updating the UI:

```python
if (
    provider_name == "custom_ai"
    and _custom_ai_ocr_cooldown_seconds(app) > 0.0
):
    log_debug(
        "RECOVERY: cooling Custom AI OCR error suppressed; "
        "PaddleOCR fallback will preserve translation continuity"
    )
    app.last_displayed_batch_sequence = sequence_number
    return
```

- [ ] **Step 7: Verify GREEN and the complete latency suite**

Run:

```powershell
py -m unittest tests.test_latency_optimization.LatencyTranslationCacheTests -v
py -m unittest tests.test_latency_optimization -v
```

Expected: all tests pass.

### Task 4: Full verification and handoff

**Files:**
- Create: `.codex/handoffs/YYYY-MM-DD_HH-mm-ss.md`

- [ ] **Step 1: Run focused Custom AI coverage**

```powershell
py -m unittest tests.test_custom_ai -v
py -m unittest test_custom_ai test_custom_ai_startup -q
```

Expected: both the main Custom AI suite and root legacy suite pass.

- [ ] **Step 2: Run the repository offline suite**

```powershell
py scripts/run_offline_tests.py
```

Expected: every offline test passes without network access.

- [ ] **Step 3: Run static checks**

```powershell
py -B -m py_compile custom_ai_transport.py handlers/translation_requests.py worker_threads.py tests/test_custom_ai.py tests/test_latency_optimization.py
git diff --check
```

Expected: both commands exit successfully with no output.

- [ ] **Step 4: Review scope and secret safety**

Run:

```powershell
git diff --stat
git diff -- custom_ai_transport.py handlers/translation_requests.py worker_threads.py tests/test_custom_ai.py tests/test_latency_optimization.py
git status --short
```

Confirm that no API key, profile JSON, runtime config, application log, or unrelated file is staged or modified.

- [ ] **Step 5: Write the handoff**

Create `.codex/handoffs/YYYY-MM-DD_HH-mm-ss.md` containing the task summary, changed files, backup path, RED/GREEN evidence, complete verification results, important cooldown/fallback decisions, no-live-network statement, and remaining profile-role isolation work.

- [ ] **Step 6: Commit the verified increment**

```powershell
git add -- custom_ai_transport.py handlers/translation_requests.py worker_threads.py tests/test_custom_ai.py tests/test_latency_optimization.py docs/superpowers/plans/2026-07-15-ai-ocr-cooldown-fallback.md .codex/handoffs/YYYY-MM-DD_HH-mm-ss.md
git commit -m "fix: keep OCR translating through AI cooldowns"
git status --short --branch
```

Expected: the commit succeeds and the worktree is clean. Do not push unless the user separately requests publishing.
