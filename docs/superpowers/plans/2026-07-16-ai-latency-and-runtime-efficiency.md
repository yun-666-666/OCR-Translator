# AI Latency and Runtime Efficiency Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve Custom AI time-to-first-text and request efficiency while preventing PaddleOCR startup contention, shutdown UI races, and redundant settings writes.

**Architecture:** Extend the existing optimization policy instead of adding a second configuration system. Keep route learning in `CustomAILatencyModeAdvisor`, request shaping in the Custom AI request/context seams, scheduling in `worker_translation.py`, readiness in the existing PaddleOCR prewarm state, and UI safety/persistence in small shared helpers.

**Tech Stack:** Python 3, Tkinter, threading, requests-compatible streaming, unittest.

---

### Task 1: User-facing and adaptive stream policy

**Files:**
- Modify: `ai_optimization.py`
- Modify: `custom_ai_policy.py`
- Modify: `gui_settings_builder.py`
- Modify: `resources/gui_eng.csv`
- Modify: `resources/gui_zh.csv`
- Modify: `resources/gui_pol.csv`
- Test: `tests/test_ai_optimization.py`
- Test: `tests/test_custom_ai.py`
- Test: `tests/test_custom_ai_startup.py`

- [ ] **Step 1: Write failing policy and UI tests**

Add tests asserting:

```python
self.assertEqual(normalize_ai_optimization_mode("stream"), "stream")
self.assertEqual(resolve_ai_response_mode("stream"), "stream")
self.assertEqual(resolve_ai_response_mode("speed"), "stream")
self.assertEqual(resolve_ai_response_mode("quality"), "safe")
```

Add advisor tests asserting that adaptive mode returns `stream` after at least
three healthy observations when streaming is supported and P90 is at least
1.5 seconds, and returns `safe` for unsupported streams, cooldown, or recent
errors. Add startup source/localization assertions for the new stream option.

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
py -m unittest tests.test_ai_optimization tests.test_custom_ai_startup tests.test_custom_ai.CustomAILatencyModeAdvisorTests -v
```

Expected: failures because `stream` is not a public optimization mode and the
advisor never selects it.

- [ ] **Step 3: Implement the minimal policy changes**

Add `AI_OPTIMIZATION_STREAM = "stream"` to the accepted optimization modes.
Map `stream` and `speed` to internal `stream`, keep `auto` as `adaptive`, and
keep `quality` as `safe`. Add the localized stream option to the combobox.

Update `_choose_adaptive_locked(...)` so the ordering is:

```python
if cooldown_or_error:
    return SAFE
if sample_count < min_samples:
    return SAFE
if p90 >= race_threshold and race_is_available:
    return RACE
if stream_supported and p90 >= stream_threshold:
    return STREAM
return SAFE
```

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the command from Step 2 and confirm all selected tests pass.

### Task 2: No-reasoning translation policy and token visibility

**Files:**
- Modify: `custom_ai_policy.py`
- Modify: `custom_ai_capabilities.py`
- Modify: `custom_ai_requests.py`
- Modify: `gui_profile_controls.py`
- Modify: `gui_builder.py`
- Modify: `resources/gui_eng.csv`
- Modify: `resources/gui_zh.csv`
- Modify: `resources/gui_pol.csv`
- Modify: `handlers/translation_context.py`
- Modify: `handlers/translation_results.py`
- Test: `tests/test_custom_ai.py`
- Test: `tests/test_custom_ai_startup.py`
- Test: `tests/test_ai_optimization.py`

- [ ] **Step 1: Write failing request and usage tests**

Add tests asserting:

```python
normalize_custom_ai_reasoning_effort("none") == "none"
provider.reasoning_effort_request_contract(profile, "translation", optimization_mode="speed") == "none"
```

Add a context test where the configured context window is five but speed policy
returns only the newest approved context entry. Add usage extraction tests for
`output_tokens_details.reasoning_tokens` and short-log tests containing:

```text
Reasoning Tokens: 704
```

Add profile-form/source localization tests for the new `None` option.

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
py -m unittest tests.test_custom_ai tests.test_custom_ai_startup tests.test_ai_optimization -v
```

Expected: failures because `none` is not a profile-selectable reasoning effort,
speed does not shape translation reasoning/context, and reasoning tokens are
discarded.

- [ ] **Step 3: Implement the minimal translation policy**

Include `none` in the public reasoning-effort options while retaining `low` as
the fallback for missing or invalid values. Pass the current optimization mode
into translation reasoning-contract construction; return `none` only for
translation under speed policy. Limit speed-policy context selection to one
entry in `_get_custom_context_for_request(...)`.

Normalize provider reasoning usage:

```python
output_details = usage.get("output_tokens_details") or {}
reasoning_tokens = int(output_details.get("reasoning_tokens") or 0)
```

Store it as `reasoning_tokens` and log the line after output tokens.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the command from Step 2 and confirm all selected tests pass.

### Task 3: Conservative slow-route supersession

**Files:**
- Modify: `worker_translation.py`
- Test: `tests/test_latency_optimization.py`

- [ ] **Step 1: Write failing threshold tests**

Add tests asserting:

```python
_get_translation_supersede_after_seconds(app, {"route_sample_count": 0}) == 3.0
_get_translation_supersede_after_seconds(app, {"route_sample_count": 7, "route_p90_seconds": 8.0}) == 3.0
_get_translation_supersede_after_seconds(app, {"route_sample_count": 8, "route_p90_seconds": 8.0}) == 4.0
```

Preserve the existing behavior for requests without route snapshots and for
explicit configuration-refresh bypass.

- [ ] **Step 2: Run test and verify RED**

Run:

```powershell
py -m unittest tests.test_latency_optimization -v
```

Expected: the learning-period cases return 1.5 seconds.

- [ ] **Step 3: Implement the learning-period threshold**

Add:

```python
ROUTE_SUPERSEDE_LEARNING_SECONDS = 3.0
```

When a valid request snapshot exists but has fewer than eight samples, return
`max(configured, ROUTE_SUPERSEDE_LEARNING_SECONDS)`. Keep the established P90
calculation and four-second cap for mature routes.

- [ ] **Step 4: Run focused test and verify GREEN**

Run the command from Step 2 and confirm it passes.

### Task 4: PaddleOCR readiness coordination

**Files:**
- Modify: `app_logic.py`
- Modify: `worker_ocr.py`
- Modify: `worker_threads.py`
- Test: `tests/test_paddle_ocr_backend.py`
- Test: `tests/test_latency_optimization.py`

- [ ] **Step 1: Write failing readiness tests**

Add tests that create a matching active prewarm thread/event and assert the
worker waits for readiness before calling the recognition function. Add tests
that custom AI OCR, a completed prewarm, a changed prewarm generation, and a
stopped app do not wait.

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
py -m unittest tests.test_paddle_ocr_backend tests.test_latency_optimization -v
```

Expected: no readiness event/helper exists and worker recognition proceeds
immediately.

- [ ] **Step 3: Implement shared readiness state**

Add `_paddleocr_prewarm_event` to the existing prewarm state. Clear it when a
new prewarm starts and set it in both success and failure completion paths.
Expose:

```python
def wait_for_paddleocr_ready_if_selected(self, timeout_seconds=20.0):
    ...
```

The helper waits only for a live matching prewarm, checks `is_running`, and
returns a boolean. Call it immediately before the local PaddleOCR recognition
path; engine cache locking remains the final safety net.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the command from Step 2 and confirm it passes.

### Task 5: Shutdown-safe Tk callback scheduling

**Files:**
- Modify: `worker_translation.py`
- Modify: `worker_threads.py`
- Test: `tests/test_latency_optimization.py`

- [ ] **Step 1: Write failing dispatch tests**

Add tests for a helper with these behaviors:

```python
schedule_ui_callback(app, callback, value) is True   # live root
schedule_ui_callback(stopped_app, callback) is False
schedule_ui_callback(destroyed_root_app, callback) is False
```

Verify that translation success/error and streaming partial completion do not
raise when `root.after` reports `RuntimeError("main thread is not in main loop")`.

- [ ] **Step 2: Run test and verify RED**

Run:

```powershell
py -m unittest tests.test_latency_optimization -v
```

Expected: direct `root.after` calls raise during shutdown.

- [ ] **Step 3: Implement and adopt the helper**

Create a small helper in `worker_translation.py` that checks `is_running`,
checks `root.winfo_exists()` when available, catches `RuntimeError` and
`TclError`, logs a coalesced shutdown-drop message, and returns a boolean.
Replace translation and API OCR response scheduling plus stream-partial
scheduling with the helper.

- [ ] **Step 4: Run focused test and verify GREEN**

Run the command from Step 2 and confirm it passes.

### Task 6: Debounced variable-trace settings saves

**Files:**
- Modify: `app_logic.py`
- Modify: `handlers/ui_interaction_handler.py`
- Test: `tests/test_custom_ai_startup.py`
- Test: `tests/test_ui_elements.py`

- [ ] **Step 1: Write failing debounce tests**

Add tests asserting three rapid calls to the trace callback schedule one
400-millisecond callback, rescheduling cancels the previous callback, and an
explicit `save_settings()` cancels the delayed callback before writing.

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
py -m unittest tests.test_custom_ai_startup tests.test_ui_elements -v
```

Expected: every trace invokes `save_settings()` immediately.

- [ ] **Step 3: Implement debounce and immediate flush**

Use the existing `_save_settings_timer` field. The trace callback cancels the
prior timer and schedules `_flush_debounced_settings_save` after 400ms. The
flush clears the timer before calling the real save method. At the beginning of
explicit `UIInteractionHandler.save_settings()`, cancel and clear a pending
timer unless the call originated from the flush itself.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the command from Step 2 and confirm it passes.

### Task 7: Full verification and handoff

**Files:**
- Create: `.codex/handoffs/YYYY-MM-DD_HH-mm-ss.md`

- [ ] **Step 1: Run focused regression suites**

```powershell
py -m unittest tests.test_ai_optimization tests.test_custom_ai tests.test_custom_ai_startup tests.test_latency_optimization tests.test_paddle_ocr_backend tests.test_ui_elements -q
```

- [ ] **Step 2: Run complete suites**

```powershell
py -m unittest discover -s tests -q
py -m unittest -q
```

- [ ] **Step 3: Compile touched Python modules**

```powershell
py -B -m py_compile ai_optimization.py custom_ai_policy.py custom_ai_capabilities.py custom_ai_requests.py gui_profile_controls.py gui_settings_builder.py handlers\translation_context.py handlers\translation_results.py worker_translation.py worker_threads.py worker_ocr.py app_logic.py handlers\ui_interaction_handler.py
```

- [ ] **Step 4: Check diff hygiene**

```powershell
git diff --check
git status --short
```

- [ ] **Step 5: Write the handoff**

Record task summary, changed files, backup directory, red-green evidence,
complete verification results, compatibility decisions, and any remaining
real-network verification gap.
