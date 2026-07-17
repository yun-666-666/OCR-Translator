# Local Capture Baseline Floor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent local OCR capture pacing from falling below the current adaptive/user baseline while preserving existing queue-pressure behavior.

**Architecture:** Extract the local queue-fullness calculation into a pure helper in `worker_capture.py`. The capture loop passes its current base, prior interval, and queue fullness to the helper. Focused unit tests prove all three pressure branches without starting Tk or a capture thread.

**Tech Stack:** Python 3, `queue.Queue`, `unittest`, existing latency test module.

---

### Task 1: Prove the baseline-floor contract

**Files:**
- Modify: `tests/test_latency_optimization.py`
- Modify: `worker_capture.py:391-465`

- [ ] **Step 1: Add failing pure interval tests**

Add these tests to `LatencyCaptureThreadBackendSelectionTests`:

```python
def test_local_capture_interval_never_falls_below_base(self):
    import_worker_threads_for_tests()
    worker_capture = importlib.import_module("worker_capture")

    self.assertEqual(
        worker_capture._next_local_capture_interval(0.2, 0.05, 0.0),
        0.2,
    )
    self.assertEqual(
        worker_capture._next_local_capture_interval(0.2, 0.2, 0.4),
        0.2,
    )

def test_local_capture_interval_preserves_pressure_and_decay(self):
    import_worker_threads_for_tests()
    worker_capture = importlib.import_module("worker_capture")

    self.assertAlmostEqual(
        worker_capture._next_local_capture_interval(0.2, 0.5, 0.0),
        0.475,
    )
    self.assertAlmostEqual(
        worker_capture._next_local_capture_interval(0.2, 0.2, 0.5),
        0.25,
    )
    self.assertAlmostEqual(
        worker_capture._next_local_capture_interval(0.2, 0.2, 0.8),
        0.36,
    )
```

- [ ] **Step 2: Run and confirm RED**

```powershell
python -m unittest tests.test_latency_optimization.LatencyCaptureThreadBackendSelectionTests.test_local_capture_interval_never_falls_below_base tests.test_latency_optimization.LatencyCaptureThreadBackendSelectionTests.test_local_capture_interval_preserves_pressure_and_decay -v
```

Expected: both error because `_next_local_capture_interval` does not exist.

- [ ] **Step 3: Implement the pure helper and use it**

Add near the other capture helpers:

```python
def _next_local_capture_interval(
    base_scan_interval,
    current_scan_interval,
    queue_fullness,
):
    if queue_fullness > 0.7:
        return base_scan_interval * (1 + queue_fullness)
    if queue_fullness > 0.4:
        return base_scan_interval * 1.25
    return max(base_scan_interval, current_scan_interval * 0.95)
```

Replace the inline local-OCR branch with:

```python
current_scan_interval_sec = _next_local_capture_interval(
    base_scan_interval,
    current_scan_interval_sec,
    q_fullness,
)
```

- [ ] **Step 4: Run and confirm GREEN**

Run the command from Step 2. Expected: `Ran 2 tests ... OK`.

### Task 2: Validate and commit round 2

**Files:**
- Create: `.codex/handoffs/2026-07-17_22-35-34.md`
- Verify: `worker_capture.py`
- Verify: `tests/test_latency_optimization.py`

- [ ] **Step 1: Run related and full tests**

```powershell
python -m unittest tests.test_latency_optimization -q
python -m unittest discover -s tests -q
```

Expected: both exit `0` without failures.

- [ ] **Step 2: Run compile and diff hygiene**

```powershell
python -m py_compile worker_capture.py tests/test_latency_optimization.py
git diff --check
```

Expected: both exit `0`.

- [ ] **Step 3: Write and commit the handoff**

Record backup paths, RED/GREEN evidence, verification, the rejected Session-race rationale, and the preserved threshold behavior. Stage only the round-2 implementation and handoff, then commit:

```powershell
git add -- worker_capture.py tests/test_latency_optimization.py .codex/handoffs/2026-07-17_22-35-34.md
git diff --cached --check
git commit -m "fix: honor local capture interval baseline"
```

- [ ] **Step 4: Ask Grok 4.5 to review the committed interval diff**

Require exact arithmetic/call-site proof, regression analysis for all three thresholds, and at most one next candidate. Save the read-only result as `docs/cleanup/grok-4.5-project-audit-round-3-2026-07-17.md` before another implementation decision.
