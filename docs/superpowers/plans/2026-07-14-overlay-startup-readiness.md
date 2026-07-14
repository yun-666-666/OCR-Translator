# Overlay Startup Readiness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore saved capture/display areas immediately and make Start plus overlay visibility controls work on the first click without waiting for PaddleOCR prewarm.

**Architecture:** Split cheap coordinate restoration from window creation. Add one idempotent overlay-readiness boundary in `overlay_manager.py`, run hidden overlay preparation before local OCR prewarm, and make Start/toggles repair missing windows from valid saved coordinates.

**Tech Stack:** Python 3, tkinter, PySide6 overlay adapter, `unittest`, `unittest.mock`.

---

### Task 1: Saved-area restoration and hidden readiness

**Files:**
- Modify: `tests/test_overlay_startup.py`
- Modify: `overlay_manager.py`

- [x] **Step 1: Write failing restoration and readiness tests**

Add tests that require hidden startup to restore coordinates and create both
overlay objects without making either visible, plus an idempotency test that
does not recreate live objects:

```python
def test_hidden_startup_restores_and_prepares_both_overlays(self):
    app = self._build_app(target_visible=False, source_visible=False)
    created = []

    def create_source(fake_app, force_hidden=False):
        created.append(("source", force_hidden))
        fake_app.source_overlay = FakeReadyOverlay()

    def create_target(fake_app, skip_preservation=False, force_hidden=False):
        created.append(("target", force_hidden))
        fake_app.target_overlay = FakeReadyOverlay()
        fake_app.translation_text = FakeReadyOverlay()

    with patch.object(overlay_manager, "create_source_overlay_om", side_effect=create_source), patch.object(
        overlay_manager, "create_target_overlay_om", side_effect=create_target
    ):
        ready = overlay_manager.load_areas_from_config_om(app)

    self.assertTrue(ready)
    self.assertEqual([("source", True), ("target", True)], created)
    self.assertEqual([0, 0, 100, 50], app.source_area)
    self.assertEqual([10, 20, 210, 120], app.target_area)

def test_readiness_reuses_live_overlays(self):
    app = self._build_app()
    app.source_area = [0, 0, 100, 50]
    app.target_area = [10, 20, 210, 120]
    app.source_overlay = FakeReadyOverlay()
    app.target_overlay = FakeReadyOverlay()
    app.translation_text = FakeReadyOverlay()

    with patch.object(overlay_manager, "create_source_overlay_om") as create_source, patch.object(
        overlay_manager, "create_target_overlay_om"
    ) as create_target:
        ready = overlay_manager.ensure_overlays_ready_om(app, force_hidden=True)

    self.assertTrue(ready)
    create_source.assert_not_called()
    create_target.assert_not_called()
```

- [x] **Step 2: Run the focused tests and verify RED**

Run:

```powershell
py -m unittest tests.test_overlay_startup.OverlayStartupTests.test_hidden_startup_restores_and_prepares_both_overlays tests.test_overlay_startup.OverlayStartupTests.test_readiness_reuses_live_overlays -v
```

Expected: FAIL because `ensure_overlays_ready_om` and the `force_hidden`
creation contract do not exist and hidden startup does not create overlays.

- [x] **Step 3: Implement restoration and readiness**

In `overlay_manager.py`, add small helpers equivalent to:

```python
def _widget_exists_om(widget):
    if widget is None:
        return False
    try:
        return bool(widget.winfo_exists())
    except Exception:
        return False


def _valid_area_om(area):
    try:
        x1, y1, x2, y2 = map(int, area)
    except (TypeError, ValueError):
        return False
    return x2 > x1 and y2 > y1


def restore_areas_from_config_om(app):
    try:
        settings = app.config["Settings"]
        app.source_area = [int(settings.get(name, default)) for name, default in SOURCE_AREA_DEFAULTS]
        app.target_area = [int(settings.get(name, default)) for name, default in TARGET_AREA_DEFAULTS]
    except (KeyError, TypeError, ValueError) as exc:
        log_debug(f"OverlayManager: Could not restore areas from config: {exc}")
        return False
    return _valid_area_om(app.source_area) and _valid_area_om(app.target_area)


def ensure_overlays_ready_om(app, require_source=True, require_target=True, force_hidden=False):
    if ((require_source and not _valid_area_om(app.source_area)) or
            (require_target and not _valid_area_om(app.target_area))):
        if not restore_areas_from_config_om(app):
            return False
    if require_source and not _widget_exists_om(app.source_overlay):
        create_source_overlay_om(app, force_hidden=force_hidden)
    target_ready = _widget_exists_om(app.target_overlay) and _widget_exists_om(
        getattr(app, "translation_text", None)
    )
    if require_target and not target_ready:
        create_target_overlay_om(app, force_hidden=force_hidden)
    return (
        (not require_source or _widget_exists_om(app.source_overlay))
        and (not require_target or (
            _widget_exists_om(app.target_overlay)
            and _widget_exists_om(getattr(app, "translation_text", None))
        ))
    )
```

Update both overlay creation functions with a `force_hidden=False` keyword and
make their visibility branch explicit: forced hidden, configured visible, or
configured hidden. Make `load_areas_from_config_om()` restore coordinates and
call readiness with `force_hidden=True`, returning the readiness result.

- [x] **Step 4: Run the focused tests and verify GREEN**

Run:

```powershell
py -m unittest tests.test_overlay_startup -v
```

Expected: PASS.

### Task 2: First-click visibility toggles

**Files:**
- Modify: `tests/test_overlay_startup.py`
- Modify: `overlay_manager.py`

- [x] **Step 1: Write failing first-click toggle tests**

Update the target toggle test to assert that readiness creates a hidden target
once and the toggle shows it once. Add the equivalent source test. The tests
must also assert that redundant `update_color()` is skipped for an overlay that
was just created and styled.

```python
with patch.object(overlay_manager, "ensure_overlays_ready_om", side_effect=prepare) as ensure:
    overlay_manager.toggle_target_visibility_om(app)

ensure.assert_called_once_with(
    app, require_source=False, require_target=True, force_hidden=True
)
self.assertEqual(1, app.target_overlay.toggle_count)
self.assertEqual(0, app.target_overlay.color_update_count)
```

- [x] **Step 2: Run toggle tests and verify RED**

Run:

```powershell
py -m unittest tests.test_overlay_startup.OverlayStartupTests.test_toggle_target_visibility_creates_overlay_on_demand tests.test_overlay_startup.OverlayStartupTests.test_toggle_source_visibility_creates_overlay_on_demand -v
```

Expected: FAIL because toggles call creation directly and target styling is
reapplied after creation.

- [x] **Step 3: Route toggles through readiness**

In `overlay_manager.py`, replace direct on-demand creation with:

```python
was_ready = _widget_exists_om(app.target_overlay)
ready = ensure_overlays_ready_om(
    app,
    require_source=False,
    require_target=True,
    force_hidden=True,
)
if ready:
    if was_ready and hasattr(app.target_overlay, "update_color"):
        app.target_overlay.update_color(app.target_colour_var.get())
    app.target_overlay.toggle_visibility()
else:
    messagebox.showwarning(
        "Warning",
        "Target area overlay window does not exist.\nPlease select the target area first.",
        parent=app.root,
    )
```

Use the equivalent source-only readiness call for the source toggle.

- [x] **Step 4: Run overlay tests and verify GREEN**

Run:

```powershell
py -m unittest tests.test_overlay_startup -v
```

Expected: PASS.

### Task 3: Startup scheduling and Start self-repair

**Files:**
- Modify: `tests/test_overlay_startup.py`
- Modify: `app_logic.py`
- Modify: `app_lifecycle.py`

- [x] **Step 1: Write failing startup-order and preflight tests**

Add a test for a new `schedule_initial_ui_readiness()` method using a fake root
that records timer delays. Require hidden overlay preparation to be scheduled
before PaddleOCR prewarm. Add a focused lifecycle helper test requiring missing
overlays to be prepared before Start validation:

```python
def test_start_preflight_repairs_missing_overlays(self):
    app = object.__new__(app_logic.GameChangingTranslator)
    app.source_overlay = None
    app.target_overlay = None
    app.translation_text = None

    with patch.object(app_lifecycle, "ensure_overlays_ready_om", return_value=True) as ensure:
        self.assertTrue(app._ensure_overlays_for_start())

    ensure.assert_called_once_with(app, force_hidden=True)
```

- [x] **Step 2: Run scheduling and preflight tests and verify RED**

Run:

```powershell
py -m unittest tests.test_overlay_startup.OverlayStartupTests.test_startup_schedules_overlay_readiness_before_paddleocr_prewarm tests.test_overlay_startup.OverlayStartupTests.test_start_preflight_repairs_missing_overlays -v
```

Expected: FAIL because the scheduling and lifecycle helper methods do not yet
exist.

- [x] **Step 3: Implement UI-critical startup order**

In `app_logic.py`:

- import `restore_areas_from_config_om`;
- call it immediately after `load_app_config()`;
- replace the direct overlay/prewarm timers with a method that schedules hidden
  overlay preparation at 50 ms and PaddleOCR prewarm at 250 ms;
- keep `ensure_window_visible` at 200 ms or earlier so the shell is painted
  independently of OCR initialization.

The scheduling method should be equivalent to:

```python
def schedule_initial_ui_readiness(self):
    self.root.after(50, self.load_initial_overlay_areas)
    self.root.after(250, self.schedule_initial_paddleocr_prewarm)
```

Do not call `schedule_initial_paddleocr_prewarm()` directly from `__init__`.

- [x] **Step 4: Implement Start self-repair**

Import `ensure_overlays_ready_om` in `app_lifecycle.py` and add:

```python
def _ensure_overlays_for_start(self):
    if ensure_overlays_ready_om(self, force_hidden=True):
        return True
    messagebox.showerror(
        "Start Error",
        "Could not initialize the saved source and target areas. Select the areas again.",
        parent=self.root,
    )
    return False
```

Call this once before the existing widget and geometry checks. Keep the existing
profile validation and coordinate validation unchanged.

- [x] **Step 5: Run focused tests and verify GREEN**

Run:

```powershell
py -m unittest tests.test_overlay_startup -v
py -m unittest tests.test_latency_optimization.LatencyShutdownTests -v
```

Expected: PASS.

### Task 4: Full verification and real startup exercise

**Files:**
- Create: the timestamped handoff path printed by
  `Join-Path '.codex\handoffs' ((Get-Date -Format 'yyyy-MM-dd_HH-mm-ss') + '.md')`

- [x] **Step 1: Run static and full automated verification**

Run:

```powershell
py -m unittest discover -s tests
py -m unittest test_custom_ai test_custom_ai_startup -q
py -m compileall -q -x "(Dango-Translator-Ver6.3.1|dist|build|node_modules)" .
git diff --check
```

Expected: all tests pass, compilation succeeds, and diff check is clean.

- [x] **Step 2: Exercise the real application without paid API calls**

Launch the application locally with the current saved coordinates and both
visibility flags false. During PaddleOCR prewarm:

1. Click Hide/Show Target Window and verify prompt visual response at the saved
   target coordinates.
2. Hide it, restart the application, and press Start immediately.
3. Verify no overlay-missing dialog appears, the source capture geometry is the
   saved source area, and the target appears at the saved target area.
4. Stop and close cleanly.

Do not wait for or trigger any translation network request; stop immediately
after overlay readiness is confirmed if necessary.

- [x] **Step 3: Review scope and write handoff**

Confirm `git status --short` contains only the intended tracked changes and the
pre-existing untracked `Dango-Translator-Ver6.3.1/`. Create the required handoff
with task summary, changed files, backup path, commands/results, decisions, and
remaining issues.
