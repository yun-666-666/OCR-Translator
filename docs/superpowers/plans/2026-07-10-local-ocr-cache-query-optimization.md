# Local OCR and Cache Query Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce redundant local OCR and translation-cache misses while preserving subtitle accuracy and the existing instant-cache display behavior.

**Architecture:** `worker_threads.py` will replace random admission of identical local frames with a small deterministic state helper. The local OCR stability router will stop probing the translation cache before a candidate is admitted; `start_async_translation` remains the single place that handles an admitted candidate's instant-cache display.

**Tech Stack:** Python 3, `unittest`, existing runtime metrics, Tk/PaddleOCR worker pipeline.

---

### Task 1: Deterministic Local Frame Admission

**Files:**
- Modify: `worker_threads.py:1041-1187`
- Test: `tests/test_latency_optimization.py:230-313`

- [ ] **Step 1: Write the failing test**

```python
def test_local_capture_signature_allows_one_duplicate_before_skipping(self):
    worker_threads = import_worker_threads_for_tests()

    last, repeats, enqueue = worker_threads._advance_local_capture_signature(
        None, 0, ("frame-a", 10, 20, 300, 80, "mss")
    )
    self.assertEqual((last, repeats, enqueue), (("frame-a", 10, 20, 300, 80, "mss"), 0, True))

    last, repeats, enqueue = worker_threads._advance_local_capture_signature(
        last, repeats, ("frame-a", 10, 20, 300, 80, "mss")
    )
    self.assertEqual((repeats, enqueue), (1, True))

    last, repeats, enqueue = worker_threads._advance_local_capture_signature(
        last, repeats, ("frame-a", 10, 20, 300, 80, "mss")
    )
    self.assertEqual((repeats, enqueue), (2, False))

    last, repeats, enqueue = worker_threads._advance_local_capture_signature(
        last, repeats, ("frame-b", 10, 20, 300, 80, "mss")
    )
    self.assertEqual((last, repeats, enqueue), (("frame-b", 10, 20, 300, 80, "mss"), 0, True))
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `py -m unittest tests.test_latency_optimization.LatencyCaptureThreadBackendSelectionTests.test_local_capture_signature_allows_one_duplicate_before_skipping -v`

Expected: `AttributeError` because `_advance_local_capture_signature` does not exist.

- [ ] **Step 3: Write the minimal implementation**

```python
def _advance_local_capture_signature(last_signature, repeat_count, current_signature):
    if current_signature != last_signature:
        return current_signature, 0, True
    repeat_count = max(0, int(repeat_count or 0)) + 1
    return current_signature, repeat_count, repeat_count <= 1
```

Use the helper in `run_capture_thread` for local OCR, increment
`capture_exact_duplicate_skip` when it returns `False`, and keep the API OCR
exact-signature behavior unchanged.

- [ ] **Step 4: Run the test to verify it passes**

Run: `py -m unittest tests.test_latency_optimization.LatencyCaptureThreadBackendSelectionTests.test_local_capture_signature_allows_one_duplicate_before_skipping -v`

Expected: `OK`.

### Task 2: Probe the Translation Cache Only for Admitted OCR Text

**Files:**
- Modify: `worker_threads.py:688-845`
- Test: `tests/test_latency_optimization.py:2726-2976`

- [ ] **Step 1: Write the failing test**

```python
def test_pending_local_ocr_candidate_does_not_probe_translation_cache(self):
    worker_threads = import_worker_threads_for_tests()
    scheduled = []

    class Handler:
        get_cached_translation_for_display = Mock(return_value="cached")

    app = self._make_app(scheduled, handler=Handler())
    with patch.object(worker_threads, "start_async_translation") as start_translation:
        result = worker_threads._route_local_ocr_candidate_for_translation(
            app, "The treas", now=100.0
        )

    self.assertEqual(result, "pending")
    app.translation_handler.get_cached_translation_for_display.assert_not_called()
    start_translation.assert_not_called()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `py -m unittest tests.test_latency_optimization.LatencyOcrStabilityGateTests.test_pending_local_ocr_candidate_does_not_probe_translation_cache -v`

Expected: assertion failure because the current router probes the instant cache before asking the stability gate to hold the candidate.

- [ ] **Step 3: Write the minimal implementation**

Remove `_local_ocr_candidate_has_instant_cache` and its pre-gate branch from
`_route_local_ocr_candidate_for_translation`. Do not modify
`start_async_translation`; it remains responsible for the cache lookup after a
candidate is admitted.

- [ ] **Step 4: Run the test to verify it passes**

Run: `py -m unittest tests.test_latency_optimization.LatencyOcrStabilityGateTests.test_pending_local_ocr_candidate_does_not_probe_translation_cache -v`

Expected: `OK`.

### Task 3: Verify the Combined Change

**Files:**
- Modify: `worker_threads.py`
- Modify: `tests/test_latency_optimization.py`

- [ ] **Step 1: Run focused regression coverage**

Run: `py -m unittest tests.test_latency_optimization -v`

Expected: all latency tests pass, including existing instant-cache and OCR-stability tests.

- [ ] **Step 2: Run project regression coverage**

Run: `py -m unittest discover -s tests`

Expected: all discovered tests pass.

- [ ] **Step 3: Run syntax and whitespace checks**

Run: `py -B -m py_compile worker_threads.py tests\\test_latency_optimization.py`

Expected: exit code 0.

Run: `git diff --check`

Expected: no output and exit code 0.

- [ ] **Step 4: Preserve the current user worktree**

Do not create a commit: the worktree contains user-owned untracked files.
Record only the changed files, backup path, and verification evidence in the
required `.codex/handoffs/` document at completion.
