# Balanced Translation Latency and Quality Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a user-configurable Custom AI submission interval, sanitized end-to-end timing, and paired source/translation context.

**Architecture:** Keep the existing latest-only single-flight scheduler and SQLite cache. Add one persisted millisecond setting at the application/UI boundary, carry scheduling time through the worker, and change only the Custom AI context representation and prompt formatting.

**Tech Stack:** Python 3, Tkinter/ttk, configparser, unittest

---

### Task 1: Configurable base submission interval

**Files:**
- Modify: `config_manager.py`
- Modify: `app_logic.py`
- Modify: `gui_builder.py`
- Modify: `handlers/ui_interaction_handler.py`
- Modify: `handlers/translation_handler.py`
- Modify: `resources/gui_eng.csv`
- Modify: `resources/gui_zh.csv`
- Modify: `ocr_translator_config.example.ini`
- Test: `tests/test_custom_ai_startup.py`
- Test: `tests/test_latency_optimization.py`

- [ ] **Step 1: Write failing configuration and interval tests**

Add tests asserting:

```python
self.assertEqual(DEFAULT_CONFIG_SETTINGS["custom_ai_submit_interval_ms"], "300")
self.assertEqual(handler.get_translation_submit_interval_seconds("Hello"), 0.3)
app.custom_ai_submit_interval_ms_var = DummyVar(650)
self.assertEqual(handler.get_translation_submit_interval_seconds("Hello"), 0.65)
```

Also assert both localization CSV files contain
`custom_ai_submit_interval_label`.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```powershell
python -m unittest tests.test_custom_ai_startup tests.test_latency_optimization.LatencyTranslationCacheTests -v
```

Expected: failures for the missing default/label/variable behavior and the old
1.0-second short-text interval.

- [ ] **Step 3: Implement config, UI, persistence, and interval calculation**

Add a 300 ms default, a clamped `tk.IntVar`, a 0–5000 ms spinbox, save wiring, and
bilingual labels. Calculate:

```python
base_interval = clamped_milliseconds / 1000.0
extra_for_long_text = min(3.0, max(0, len(text) - 80) / 180.0)
return base_interval + extra_for_long_text
```

Retain the old `min_translation_interval` value as the fallback for callers without
the new variable.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the same command and expect all selected tests to pass.

### Task 2: Sanitized end-to-end translation timing

**Files:**
- Modify: `worker_threads.py`
- Test: `tests/test_latency_optimization.py`

- [ ] **Step 1: Write failing timing propagation tests**

Create tests where `time.monotonic()` is controlled and assert that:

```python
pending_request["requested_at_monotonic"] == 100.0
```

survives replacement of pending text, is passed into
`process_translation_async`, and produces a log containing
`queue=`, `worker=`, and `total=` without the source or translated text.

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```powershell
python -m unittest tests.test_latency_optimization.LatencyTranslationCacheTests -v
```

Expected: timing fields and the sanitized summary are absent.

- [ ] **Step 3: Implement timing propagation and summary**

Add optional `requested_at_monotonic` parameters to submission and worker helpers,
preserve the original arrival time while replacing the pending payload with the
latest subtitle, and log:

```python
log_debug(
    "LATENCY: translation timing "
    f"sequence={translation_sequence} queue={queue_seconds:.3f}s "
    f"worker={worker_seconds:.3f}s total={total_seconds:.3f}s"
)
```

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the same test command and expect it to pass.

### Task 3: Paired source/translation context

**Files:**
- Modify: `custom_ai.py`
- Modify: `handlers/translation_handler.py`
- Test: `tests/test_custom_ai.py`

- [ ] **Step 1: Write failing paired-context tests**

Add tests asserting:

```python
handler._update_custom_context("Save", "保存")
self.assertEqual(handler.custom_context_window, [("Save", "保存")])
```

Assert duplicate source or target output is suppressed, a cache hit records both
source and cached translation, and `build_translation_payload()` renders labeled
source/translation history while returning only the current translation.

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```powershell
python -m unittest tests.test_custom_ai.TranslationHandlerCustomAITests tests.test_custom_ai.CustomAIProviderTests -v
```

Expected: `_update_custom_context` rejects the second argument and payload formatting
does not include paired target history.

- [ ] **Step 3: Implement paired context**

Store immutable `(source, translation)` pairs, pass the accepted translation on
successful requests and cache hits, retain the configured history length, and
format context defensively:

```text
Previous approved subtitle translations:
Source: Save
Translation: 保存
Current source text:
Quit
```

Continue accepting legacy string entries as source-only context.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the same test command and expect it to pass.

### Task 4: Regression verification and handoff

**Files:**
- Create: a completion-timestamped Markdown handoff in `.codex/handoffs/`

- [ ] **Step 1: Run all project tests**

```powershell
python -m unittest discover -s tests
python -m unittest discover
```

Expected: zero failures and zero errors.

- [ ] **Step 2: Run compile and diff checks**

```powershell
python -m compileall -q app_logic.py config_manager.py custom_ai.py gui_builder.py worker_threads.py handlers tests
git diff --check
```

Expected: both commands exit 0.

- [ ] **Step 3: Verify requirements from current state**

Confirm the default, UI label, persistence key, interval calculation, timing log,
paired prompt, context-sensitive cache key, and unchanged single-flight/latest-only
tests all have direct passing evidence.

- [ ] **Step 4: Write the required handoff**

Record the task summary, files changed, backup directory, exact commands/results,
important decisions, and any unverified real-network behavior.
