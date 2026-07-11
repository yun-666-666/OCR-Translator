# Live Translation Profile Switch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the primary Custom AI translation profile dropdown apply immediately to a running task and wake the latest subtitle.

**Architecture:** Extract a testable GUI-boundary helper that persists the active profile, clears context, and reuses the generation-checked scheduler refresh built for live model edits. Keep request snapshots and concurrency limits unchanged.

**Tech Stack:** Python 3, Tkinter/ttk, CustomAIProfileManager, `unittest`, `unittest.mock`.

---

### Task 1: Specify live profile-switch semantics

**Files:**
- Modify: `tests/test_latency_optimization.py`
- Modify: `gui_builder.py`

- [x] **Step 1: Write the failing live-switch test**

Construct two profiles and a running app. Call the wished-for helper and assert:

```python
changed = gui_builder.apply_custom_ai_translation_profile_selection(
    app,
    "Working",
)
self.assertTrue(changed)
self.assertEqual(profiles.active_id, "working")
self.assertEqual(app.translation_model_var.value, "custom_ai")
app.translation_handler._clear_active_context.assert_called_once_with()
refresh.assert_called_once_with(
    app,
    reason="active translation profile changed",
)
```

- [x] **Step 2: Add stopped/same-profile/error tests**

Verify a stopped app persists without refresh, an already-active selection is a
no-op, and a persistence failure is reported through `messagebox.showerror`
without escaping the public helper.

- [x] **Step 3: Run RED**

```powershell
py -m unittest tests.test_latency_optimization.LiveCustomAIProfileSwitchTests -v
```

Expected: errors because the profile-selection helper does not exist.

- [x] **Step 4: Implement the minimal helper**

Add a private implementation plus a public error boundary mirroring the existing
model-selection helper. Resolve enabled profiles, compare IDs, persist with
`set_active_profile`, set the `custom_ai` route, clear context, and refresh only
when running.

- [x] **Step 5: Run GREEN**

Run the same class. Expected: all profile-switch tests pass.

### Task 2: Wire the primary dropdown

**Files:**
- Modify: `tests/test_latency_optimization.py`
- Modify: `gui_builder.py`

- [x] **Step 1: Write a failing source-boundary regression**

Patch the public helper, invoke the dropdown's extracted handler or inspect the
builder callback contract, and prove selection delegates once with the display
name before `on_translation_model_selection_changed` runs.

- [x] **Step 2: Run RED**

```powershell
py -m unittest tests.test_latency_optimization.LiveCustomAIProfileSwitchTests.test_primary_profile_selection_handler_uses_live_apply_helper -v
```

Expected: failure because the nested handler still calls
`set_active_profile()` directly.

- [x] **Step 3: Replace direct mutation with the public helper**

The dropdown callback calls
`apply_custom_ai_translation_profile_selection(app, selected_name)` and then
uses the existing selection-changed hook for UI/settings synchronization.

- [x] **Step 4: Run focused suites**

```powershell
py -m unittest tests.test_latency_optimization tests.test_custom_ai tests.test_ui_elements -q
```

Expected: all live model/profile, snapshot, scheduler, and UI tests pass.

### Task 3: Verify the complete increment

**Files:**
- Modify: `docs/superpowers/plans/2026-07-10-live-translation-profile-switch.md`

- [x] **Step 1: Run full tests and compilation**

```powershell
py -m unittest discover -s tests
py -m unittest discover
py -B -m py_compile gui_builder.py tests\test_latency_optimization.py
py -m compileall -q gui_builder.py tests
git diff --check
```

Expected: every command exits 0.

- [x] **Step 2: Review immutable snapshot and concurrency invariants**

Re-run the existing live model-switch snapshot/overflow tests and inspect the
backup/current diff to confirm no request snapshot or concurrency code changed.

- [x] **Step 3: Mark this plan complete**

Change checkboxes to `[x]` after the commands and invariant review succeed.

### Review follow-up: Single persistence boundary and rollback

- [x] Add a failing full-handler regression proving the legacy UI sync receives
  `event=None` after the new helper has already persisted the selection.
- [x] Remove the second `set_active_profile()` call from the dropdown chain.
- [x] Make `set_active_profile()` restore the prior in-memory active ID and
  raise when persistence returns failure.
- [x] Verify the public GUI error boundary leaves context and scheduler state
  untouched after a failed save.

### Final review follow-up: UI-only synchronization

- [x] Add a failing regression against the real `GameChangingTranslator`
  callback showing that profile-dropdown synchronization must not end/start a
  translation session or save settings again.
- [x] Add an explicit `synchronize_ui_only` path and use it after both
  successful and rolled-back profile selections.
- [x] Verify the remaining UI handler still restores the visible active profile
  without repeating context/session lifecycle work.
