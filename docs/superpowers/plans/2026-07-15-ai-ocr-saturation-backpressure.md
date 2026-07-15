# AI OCR Saturation Backpressure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent the capture loop from producing obsolete frames while all effective Custom AI OCR slots are occupied, then resume with the freshest frame as soon as capacity returns.

**Architecture:** `AppCaptureOcrMixin` owns the provider-aware effective concurrency limit and derives adaptive load thresholds from it. `worker_threads` delegates its existing submission limit to that app seam, while `worker_capture` applies an API-only saturation guard before overlay inspection and screen capture; the existing submission guard remains for races.

**Tech Stack:** Python 3, Tkinter, `unittest`, `unittest.mock`, existing `RuntimeMetrics` and coalesced logger helpers.

---

## File map

- Modify `app_capture_ocr.py`: own the Custom AI limit and calculate provider-aware adaptive states.
- Modify `worker_threads.py`: reuse the app-owned limit and export the new capture helper through the compatibility facade.
- Modify `worker_capture.py`: detect saturation and pause before taking screenshots.
- Modify `tests/test_latency_optimization.py`: cover limit selection, adaptive thresholds, source backpressure, recovery, metrics, and local OCR exclusion.
- Create `.codex/handoffs/2026-07-15_23-05-22.md`: record the finished implementation and verification evidence.

### Task 1: Make effective OCR capacity authoritative

**Files:**
- Modify: `app_capture_ocr.py:1-210`
- Modify: `worker_threads.py:1-190`
- Test: `tests/test_latency_optimization.py:4143-4190`

- [ ] **Step 1: Add failing provider-capacity and adaptive-threshold tests**

Add an OCR model variable to `AdaptiveScanLoggingTests._make_app()` and add tests that exercise both the two-slot Custom AI path and the historical generic eight-slot path:

```python
    @staticmethod
    def _make_app(active_count=0, ocr_model="custom_ai"):
        import app_logic

        app = object.__new__(app_logic.GameChangingTranslator)
        app.load_check_timer = 0.0
        app.active_ocr_calls = set(range(active_count))
        app.max_concurrent_ocr_calls = 8
        app.current_scan_interval = 200
        app.base_scan_interval = 200
        app.overload_detected = False
        app.scan_interval_var = types.SimpleNamespace(get=lambda: 200)
        app.ocr_model_var = types.SimpleNamespace(get=lambda: ocr_model)
        app._last_adaptive_log_state = None
        app._last_adaptive_log_time = 0.0
        return app

    def test_effective_limit_caps_custom_ai_and_preserves_generic_capacity(self):
        app = self._make_app()

        self.assertEqual(app.get_effective_ocr_concurrency_limit("custom_ai"), 2)
        self.assertEqual(app.get_effective_ocr_concurrency_limit("other_api"), 8)

    def test_effective_limit_preserves_zero_and_recovers_invalid_config(self):
        app = self._make_app()
        app.max_concurrent_ocr_calls = 0
        self.assertEqual(app.get_effective_ocr_concurrency_limit("custom_ai"), 0)

        app.max_concurrent_ocr_calls = "invalid"
        self.assertEqual(app.get_effective_ocr_concurrency_limit("custom_ai"), 1)

    def test_two_active_custom_ai_calls_trigger_adaptive_overload(self):
        import app_logic

        app = self._make_app(active_count=2, ocr_model="custom_ai")
        with (
            patch.object(app_logic.time, "monotonic", return_value=2.1),
            patch.object(app_logic, "log_debug") as debug_log,
        ):
            app.update_adaptive_scan_interval()

        self.assertTrue(app.overload_detected)
        self.assertEqual(app.current_scan_interval, 300)
        self.assertIn("overload detected", debug_log.call_args.args[0].lower())

    def test_six_active_generic_calls_keep_historical_overload_threshold(self):
        import app_logic

        app = self._make_app(active_count=6, ocr_model="other_api")
        with (
            patch.object(app_logic.time, "monotonic", return_value=2.1),
            patch.object(app_logic, "log_debug"),
        ):
            app.update_adaptive_scan_interval()

        self.assertTrue(app.overload_detected)
        self.assertEqual(app.current_scan_interval, 300)
```

- [ ] **Step 2: Run the focused tests and confirm the new contract fails**

Run:

```powershell
py -m unittest tests.test_latency_optimization.AdaptiveScanLoggingTests -v
```

Expected: FAIL because `get_effective_ocr_concurrency_limit` does not exist and two active calls are still logged as normal.

- [ ] **Step 3: Implement the app-owned limit and derived thresholds**

Import `math`, define the clamp beside the mixin, add the effective-limit method, and express adaptive state once:

```python
import math

CUSTOM_AI_OCR_CONCURRENCY_LIMIT = 2


class AppCaptureOcrMixin:
    def get_effective_ocr_concurrency_limit(self, provider_name=None):
        """Return the provider-aware OCR request capacity."""
        if provider_name is None:
            provider_name = self.get_ocr_model_setting()
        try:
            configured_limit = max(0, int(self.max_concurrent_ocr_calls))
        except (AttributeError, TypeError, ValueError):
            configured_limit = 1
        if provider_name == "custom_ai":
            return min(configured_limit, CUSTOM_AI_OCR_CONCURRENCY_LIMIT)
        return configured_limit
```

Replace the hard-coded `> 5`, `< 5`, and `== 5` decisions with:

```python
        max_ocr_calls = self.get_effective_ocr_concurrency_limit()
        overload_threshold = max(1, math.ceil(max_ocr_calls * 0.75))
        moderate_threshold = max(1, overload_threshold - 1)

        if active_ocr_count >= overload_threshold:
            adaptive_state = "overloaded"
        elif active_ocr_count >= moderate_threshold:
            adaptive_state = "moderate"
        else:
            adaptive_state = "normal"
```

Branch the existing interval and logging behavior on `adaptive_state` so overload still applies 150%, normal restores the base interval, and moderate preserves the current interval.

- [ ] **Step 4: Delegate the worker submission limit to the app seam**

In `worker_threads.py`, import `CUSTOM_AI_OCR_CONCURRENCY_LIMIT` from `app_capture_ocr`, remove the local duplicate constant, and update the helper while retaining compatibility for test doubles:

```python
from app_capture_ocr import CUSTOM_AI_OCR_CONCURRENCY_LIMIT


def _api_ocr_concurrency_limit(app, provider_name):
    """Return the same provider-aware limit used by adaptive capture."""
    getter = getattr(app, "get_effective_ocr_concurrency_limit", None)
    if callable(getter):
        try:
            return max(0, int(getter(provider_name)))
        except (TypeError, ValueError):
            pass
    try:
        configured_limit = max(0, int(app.max_concurrent_ocr_calls))
    except (AttributeError, TypeError, ValueError):
        configured_limit = 1
    if provider_name == "custom_ai":
        return min(configured_limit, CUSTOM_AI_OCR_CONCURRENCY_LIMIT)
    return configured_limit
```

- [ ] **Step 5: Run the focused tests and commit the capacity change**

Run:

```powershell
py -m unittest tests.test_latency_optimization.AdaptiveScanLoggingTests -v
```

Expected: all `AdaptiveScanLoggingTests` pass, including two-of-two Custom AI overload and six-of-eight generic overload.

Commit:

```powershell
git add -- app_capture_ocr.py worker_threads.py tests/test_latency_optimization.py
git commit -m "fix: align OCR load thresholds with provider capacity"
```

### Task 2: Apply API OCR source backpressure

**Files:**
- Modify: `worker_capture.py:20-55,364-410`
- Modify: `worker_threads.py:50-90`
- Test: `tests/test_latency_optimization.py:365-520`

- [ ] **Step 1: Add failing saturation helper tests**

Add these tests to `LatencyCaptureThreadBackendSelectionTests`:

```python
    def test_api_capture_saturation_uses_effective_limit(self):
        worker_threads = import_worker_threads_for_tests()
        app = types.SimpleNamespace(
            active_ocr_calls={"first", "second"},
            max_concurrent_ocr_calls=8,
            is_api_based_ocr_model=lambda model: model == "custom_ai",
            get_effective_ocr_concurrency_limit=lambda provider: 2,
        )

        self.assertTrue(
            worker_threads._api_ocr_capture_is_saturated(app, "custom_ai")
        )
        self.assertFalse(
            worker_threads._api_ocr_capture_is_saturated(app, "paddleocr")
        )

        app.active_ocr_calls.clear()
        app.get_effective_ocr_concurrency_limit = lambda provider: 0
        self.assertTrue(
            worker_threads._api_ocr_capture_is_saturated(app, "custom_ai")
        )
```

- [ ] **Step 2: Add a failing capture-loop recovery test**

Add a test proving that saturation skips capture, records the skip, then captures exactly once after the slot is released:

```python
    def test_saturated_api_capture_waits_then_resumes_with_current_frame(self):
        worker_threads = import_worker_threads_for_tests()
        screenshot = Image.new("RGB", (8, 8), (1, 2, 3))
        metrics = RuntimeMetrics(clock=lambda: 100.0)

        class FakeOverlay:
            def winfo_exists(self):
                return True

            def get_geometry(self):
                return (10, 20, 18, 28)

        app = types.SimpleNamespace(
            is_running=True,
            current_scan_interval=200,
            scan_interval_var=types.SimpleNamespace(get=lambda: 200),
            update_adaptive_scan_interval=lambda: None,
            get_ocr_model_setting=lambda: "custom_ai",
            is_api_based_ocr_model=lambda model=None: model == "custom_ai",
            get_effective_ocr_concurrency_limit=lambda provider=None: 2,
            active_ocr_calls={"first", "second"},
            max_concurrent_ocr_calls=8,
            source_overlay=FakeOverlay(),
            capture_backend_var=types.SimpleNamespace(get=lambda: "mss"),
            capture_backend_selector=types.SimpleNamespace(
                resolve_backend=lambda configured, region: "mss"
            ),
            ocr_frame_cache=types.SimpleNamespace(clear=Mock()),
            ocr_stability_gate=types.SimpleNamespace(clear=Mock(return_value=True)),
            ocr_queue=queue.Queue(maxsize=4),
            runtime_metrics=metrics,
            last_processed_subtitle=None,
            previous_text="",
            text_stability_counter=0,
        )
        original_put_nowait = app.ocr_queue.put_nowait

        def stop_after_put(item):
            original_put_nowait(item)
            app.is_running = False

        app.ocr_queue.put_nowait = stop_after_put

        def release_capacity(_seconds):
            app.active_ocr_calls.clear()

        with (
            patch.object(worker_threads.tk, "Toplevel", FakeOverlay),
            patch.object(
                worker_threads,
                "capture_screen_region",
                return_value=screenshot,
            ) as capture,
            patch.object(worker_threads.time, "sleep", side_effect=release_capacity),
        ):
            worker_threads.run_capture_thread(app)

        capture.assert_called_once_with((10, 20, 8, 8), backend="mss")
        self.assertIs(app.ocr_queue.get_nowait(), screenshot)
        self.assertEqual(
            metrics.snapshot()["counters"]["api_ocr_capture_backpressure_skip"],
            1,
        )
```

- [ ] **Step 3: Run both new tests and confirm the helper is missing**

Run:

```powershell
py -m unittest \
  tests.test_latency_optimization.LatencyCaptureThreadBackendSelectionTests.test_api_capture_saturation_uses_effective_limit \
  tests.test_latency_optimization.LatencyCaptureThreadBackendSelectionTests.test_saturated_api_capture_waits_then_resumes_with_current_frame -v
```

Expected: ERROR because `_api_ocr_capture_is_saturated` has not been exported and the capture loop has no source guard.

- [ ] **Step 4: Implement the worker-capture saturation helper**

In `worker_capture.py`, resolve capacity through the worker facade so the helper stays independently testable and uses the same submission limit:

```python
def _api_ocr_capture_is_saturated(app, ocr_model):
    """Return whether API OCR capture should pause before taking a frame."""
    try:
        if not app.is_api_based_ocr_model(ocr_model):
            return False
    except Exception:
        return False

    facade = sys.modules.get("worker_threads")
    limit_getter = getattr(facade, "_api_ocr_concurrency_limit", None)
    if not callable(limit_getter):
        return False
    try:
        limit = max(0, int(limit_getter(app, ocr_model)))
        active_count = len(getattr(app, "active_ocr_calls", ()))
    except (TypeError, ValueError):
        return False
    return limit == 0 or active_count >= limit
```

Import `_api_ocr_capture_is_saturated` in `worker_threads.py` beside the other `worker_capture` helpers.

- [ ] **Step 5: Guard capture before overlay and screenshot work**

In the API OCR branch of `run_capture_thread()`, place this block before the normal `last_cap_time` interval check:

```python
                if _api_ocr_capture_is_saturated(app, ocr_model):
                    active_count = len(getattr(app, "active_ocr_calls", ()))
                    facade = sys.modules.get("worker_threads")
                    limit_getter = getattr(
                        facade,
                        "_api_ocr_concurrency_limit",
                        None,
                    )
                    effective_limit = (
                        max(0, int(limit_getter(app, ocr_model)))
                        if callable(limit_getter)
                        else 0
                    )
                    _increment_metric(app, "api_ocr_capture_backpressure_skip")
                    _log_debug_coalesced(
                        ("api-ocr-capture-backpressure", ocr_model),
                        "CAPTURE: API OCR saturated "
                        f"provider={ocr_model} active={active_count} "
                        f"limit={effective_limit}; waiting "
                        f"{scan_interval_ms}ms before capturing a fresh frame",
                        interval_seconds=5.0,
                    )
                    current_scan_interval_sec = base_scan_interval
                    slept_time = 0.0
                    while slept_time < base_scan_interval and app.is_running:
                        chunk = min(0.05, base_scan_interval - slept_time)
                        time.sleep(chunk)
                        slept_time += chunk
                    continue
```

This block must execute before overlay lookup, geometry resolution, capture backend selection, screenshot creation, resizing, hashing, or queue insertion. Do not remove the existing `run_api_ocr()` concurrency check.

- [ ] **Step 6: Run the capture tests and commit source backpressure**

Run:

```powershell
py -m unittest tests.test_latency_optimization.LatencyCaptureThreadBackendSelectionTests -v
```

Expected: all capture-thread backend and saturation tests pass; the recovery test records one skip and one fresh capture.

Commit:

```powershell
git add -- worker_capture.py worker_threads.py tests/test_latency_optimization.py
git commit -m "perf: pause API OCR capture while saturated"
```

### Task 3: Verify the integrated behavior and document the handoff

**Files:**
- Create: `.codex/handoffs/2026-07-15_23-05-22.md`

- [ ] **Step 1: Run focused regression tests**

Run:

```powershell
py -m unittest tests.test_latency_optimization.AdaptiveScanLoggingTests tests.test_latency_optimization.LatencyCaptureThreadBackendSelectionTests -v
```

Expected: PASS with all adaptive and capture-backpressure tests green.

- [ ] **Step 2: Run the complete offline suites**

Run:

```powershell
py scripts/run_offline_tests.py
```

Expected: both the main `tests/` discovery suite and root legacy suite pass without network access.

- [ ] **Step 3: Compile touched Python modules and inspect the diff**

Run:

```powershell
py -m py_compile app_capture_ocr.py worker_capture.py worker_threads.py tests/test_latency_optimization.py
git diff --check
git status --short
```

Expected: compilation and diff check exit 0; status contains only the intentional implementation, tests, and handoff work.

- [ ] **Step 4: Create the required handoff**

Create `.codex/handoffs/2026-07-15_23-05-22.md` with:

```markdown
# AI OCR Saturation Backpressure Handoff

## Task summary
- Confirmed the earlier 60-second cooldown defect did not recur in the latest live session.
- Added provider-aware adaptive capacity and source backpressure for saturated Custom AI OCR.

## Files changed or created
- `app_capture_ocr.py`
- `worker_capture.py`
- `worker_threads.py`
- `tests/test_latency_optimization.py`
- this handoff

## Backup location
- `.codex/backups/2026-07-15_23-05-22/`

## Tests and commands
- list each exact command and observed result

## Important decisions
- Custom AI remains capped at two concurrent OCR requests.
- Capture pauses only for saturated API OCR; PaddleOCR behavior is unchanged.
- Submission-time concurrency checks remain as race protection.

## Remaining issues or next steps
- Provider latency remains upstream-controlled and should be re-measured from the next live run.
```

- [ ] **Step 5: Commit the handoff and final verification evidence**

Run:

```powershell
git add -- .codex/handoffs/2026-07-15_23-05-22.md
git commit -m "docs: hand off AI OCR saturation backpressure"
git status --short
```

Expected: the worktree is clean and the branch contains the capacity, backpressure, and handoff commits.
