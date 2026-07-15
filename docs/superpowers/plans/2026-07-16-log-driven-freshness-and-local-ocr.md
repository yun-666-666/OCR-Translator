# Log-Driven Freshness and Local OCR Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Subagent execution is intentionally excluded because the workspace instructions do not authorize delegation.

**Goal:** Reduce obsolete translation work and local OCR queue pressure without changing UI, configuration schema, request semantics, or the exact-frame cache.

**Architecture:** Keep the current single pending-request/coalescing architecture. Add a display-sequence freshness guard inside sequential Custom AI failover, derive the bounded translation overflow age from the immutable request snapshot, and feed measured local OCR duration back into the existing adaptive scan interval. Every new decision is conservative when data is missing or malformed.

**Tech Stack:** Python 3, `unittest`, existing `RuntimeMetrics`, Tk worker scheduling, Git.

---

### Task 1: Establish the rollback boundary

**Files:**

- Create: `.codex/backups/<timestamp>/handlers/translation_requests.py`
- Create: `.codex/backups/<timestamp>/worker_translation.py`
- Create: `.codex/backups/<timestamp>/worker_threads.py`
- Create: `.codex/backups/<timestamp>/app_capture_ocr.py`
- Create: `.codex/backups/<timestamp>/tests/test_custom_ai.py`
- Create: `.codex/backups/<timestamp>/tests/test_latency_optimization.py`

**Step 1: Confirm the isolated branch and unrelated state**

Run: `git status --short --branch`

Expected: branch `codex/log-driven-freshness-20260716`; the pre-existing `.codex/release-artifacts/release-notes-v3.10.3-2026-07-15.md` remains untracked and untouched.

**Step 2: Back up every existing file before its first edit**

Create one timestamped backup directory and copy the six files above while preserving their paths from the repository root.

**Step 3: Verify backup identity**

Run SHA-256 checks for each source/backup pair.

Expected: every pair has an identical hash.

### Task 2: Stop sequential provider failover once a newer translation is displayed

**Files:**

- Modify: `tests/test_custom_ai.py:9569`
- Modify: `handlers/translation_requests.py:1078`

**Step 1: Write the failing freshness test**

Extend `CostProtectedProfileFailoverHandlerTests._make_handler()` with `last_displayed_translation_sequence=0`. Add a test whose primary provider marks sequence 8 displayed and raises, then calls:

```python
result = handler._custom_ai_translate(
    "source",
    time.monotonic(),
    translation_sequence=7,
)
```

Assert `result is None` and that only the primary profile was attempted. Keep the existing no-sequence fallback test as the compatibility control.

**Step 2: Run the test to verify RED**

Run: `py -m unittest tests.test_custom_ai.CostProtectedProfileFailoverHandlerTests.test_custom_ai_translation_stops_failover_after_newer_result_is_displayed -v`

Expected: FAIL because the fallback profile is still called and returned.

**Step 3: Implement the minimal freshness guard**

Add a private helper in `TranslationHandler` that treats a request as obsolete only when both sequence values are positive integers and:

```python
translation_sequence <= app.last_displayed_translation_sequence
```

Missing, malformed, zero, and direct-call sequences remain current. Call the helper immediately before every network candidate and after every candidate failure. On staleness, log the sequence boundary and return `None`; do not write cache/context or continue failover.

**Step 4: Run focused GREEN checks**

Run:

```powershell
py -m unittest tests.test_custom_ai.CostProtectedProfileFailoverHandlerTests.test_custom_ai_translation_stops_failover_after_newer_result_is_displayed -v
py -m unittest tests.test_custom_ai.CostProtectedProfileFailoverHandlerTests -v
```

Expected: PASS.

**Step 5: Commit the increment**

```powershell
git add handlers/translation_requests.py tests/test_custom_ai.py
git commit -m "perf: stop stale custom ai failover"
```

### Task 3: Make bounded translation overflow route-aware

**Files:**

- Modify: `tests/test_latency_optimization.py:2572`
- Modify: `worker_translation.py:8`
- Modify: `worker_translation.py:107`
- Modify: `worker_threads.py:797`

**Step 1: Write failing threshold tests**

Add tests around `_get_translation_supersede_after_seconds(app, request_snapshot)` for:

```python
{"p90_seconds": 8.0, "sample_count": 8}   # 4.0 seconds
{"p90_seconds": 20.0, "sample_count": 8}  # capped at 4.0 seconds
{"p90_seconds": 4.0, "sample_count": 7}   # fixed 1.5-second fallback
```

Also add a scheduling test with one active call aged 2.0 seconds and a valid 8-second p90 snapshot. Assert the newest request is queued instead of using the overflow slot.

**Step 2: Run the tests to verify RED**

Run the newly added test methods individually.

Expected: signature/type failures or incorrect 1.5-second decisions because snapshots are not yet considered.

**Step 3: Implement the minimal route-aware policy**

Define constants for 8 samples, 0.5 p90 fraction, and a 4.0-second cap. Change the helper signature to accept the immutable request snapshot and compute:

```python
max(configured_threshold, min(4.0, p90_seconds * 0.5))
```

only when the snapshot is a dictionary with a finite positive p90 and at least 8 samples. Otherwise preserve the configured/fixed threshold. Pass the already-resolved `request_snapshot` from `start_async_translation`; preserve immediate configuration-refresh overflow and the race-mode exclusion.

**Step 4: Run focused GREEN checks**

Run the new threshold and scheduling tests, then:

```powershell
py -m unittest tests.test_latency_optimization.LatencyTranslationCacheTests -v
```

Expected: PASS, including existing one-slot and race-mode bounds.

**Step 5: Commit the increment**

```powershell
git add worker_translation.py worker_threads.py tests/test_latency_optimization.py
git commit -m "perf: adapt translation overflow to route latency"
```

### Task 4: Pace local OCR from measured processing time

**Files:**

- Modify: `tests/test_latency_optimization.py:4303`
- Modify: `worker_threads.py:308`
- Modify: `app_capture_ocr.py:144`

**Step 1: Write failing adaptive pacing tests**

Add a fake runtime-metrics snapshot to `AdaptiveScanLoggingTests` and prove:

- local PaddleOCR with 8 samples and p50 `0.219` seconds raises a 200 ms base interval to 250 ms;
- 7 samples leave the interval at 200 ms;
- malformed or unavailable metrics leave the interval at 200 ms;
- the existing Custom AI active-call behavior remains unchanged.

**Step 2: Run the tests to verify RED**

Run the new local OCR tests individually.

Expected: FAIL because local OCR currently sees zero API calls and remains at 200 ms.

**Step 3: Record only actual local OCR work**

Immediately after the existing local `ocr_duration` observation, also record `local_ocr_duration`. Do not record cache hits or API OCR calls.

**Step 4: Implement the minimal local pacing calculation**

For non-API OCR models, read `RuntimeMetrics.snapshot()["timings"]["local_ocr_duration"]`. With at least 8 samples, compute p50 plus 25 ms scheduling margin, round up to a 25 ms quantum, and clamp to `[base_interval, 2 * base_interval]`. Use the resulting interval in the existing adaptive state/log heartbeat. If metrics are absent or invalid, retain the base interval. Return before the API active-call logic so API behavior is unchanged.

**Step 5: Run focused GREEN checks**

Run:

```powershell
py -m unittest tests.test_latency_optimization.AdaptiveScanLoggingTests -v
```

Expected: PASS.

**Step 6: Commit the increment**

```powershell
git add app_capture_ocr.py worker_threads.py tests/test_latency_optimization.py
git commit -m "perf: pace local ocr from runtime duration"
```

### Task 5: Verify the integrated change and hand off

**Files:**

- Create: `.codex/handoffs/<timestamp>.md`

**Step 1: Run targeted suites**

```powershell
py -m unittest tests.test_custom_ai.CostProtectedProfileFailoverHandlerTests -v
py -m unittest tests.test_latency_optimization.LatencyTranslationCacheTests -v
py -m unittest tests.test_latency_optimization.AdaptiveScanLoggingTests -v
```

Expected: PASS.

**Step 2: Run the full repository test sets**

```powershell
py -m unittest discover -s tests
py -m unittest test_custom_ai test_custom_ai_startup -q
```

Expected: PASS with no new failures.

**Step 3: Run static and diff checks**

```powershell
py -B -m py_compile handlers/translation_requests.py worker_translation.py worker_threads.py app_capture_ocr.py tests/test_custom_ai.py tests/test_latency_optimization.py
git diff --check HEAD~3..HEAD
git status --short
```

Expected: compilation and diff checks succeed; only the pre-existing release note plus the new handoff remain untracked before the handoff commit.

**Step 4: Review behavior against the design boundary**

Confirm there are no UI/config/schema changes, no perceptual matching, no cache-key changes, no MSS handle reuse, and no edits to the release note.

**Step 5: Write and commit the handoff**

Document task summary, files changed, backup location and hashes, every validation result, the conservative fallbacks, and the need for a future real live session to validate route-specific behavior.

```powershell
git add .codex/handoffs/<timestamp>.md
git commit -m "docs: hand off log-driven freshness optimization"
```
