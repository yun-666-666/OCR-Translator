# Single-pass PySide text rendering implementation plan

> **For Codex:** Follow this plan with test-driven development. Keep the change
> limited to the PySide translation render path and its focused tests.

**Goal:** Eliminate the redundant second HTML render performed for every PySide
translation display while preserving correct font updates and compatibility.

**Architecture:** Move font application into `RTLTextDisplay.set_rtl_text()` so
the family and size are active before the document is rendered. Centralize the
font comparison/update in an idempotent helper shared by the direct render and
tkinter-compatible `config(font=...)` paths. Have `DisplayManager` make one
fully specified render call.

**Tech stack:** Python, unittest, PySide6/Qt offscreen tests.

---

## Task 1: Establish the regression tests

**Files:**

- Create: `tests/test_pyside_text_render.py`
- Read: `handlers/display_manager.py`
- Read: `pyside_overlay.py`

1. Run the existing overlay-focused tests to establish a clean baseline:

   ```powershell
   py -m unittest tests.test_overlay_startup tests.test_overlay_geometry tests.test_modern_ui -v
   ```

2. Add a fake-app `DisplayManager` test that invokes
   `_update_translation_text_on_main_thread()` and asserts:

   - `<br>` is normalized to a newline.
   - `set_rtl_text()` is called exactly once.
   - The requested family is passed as `font_family`.
   - `configure(font=...)` is not called.

3. Add offscreen PySide tests with a counting `RTLTextDisplay` subclass that
   records `setHtml()` calls. Verify:

   - A direct render with an explicit family and size performs one HTML update
     and applies the font first.
   - `config(font=...)` with the active font performs no HTML update.
   - `config(font=...)` with a different size performs exactly one HTML update.

4. Run the new test module and confirm it fails for the expected missing API and
   duplicate compatibility render behavior:

   ```powershell
   py -m unittest tests.test_pyside_text_render -v
   ```

## Task 2: Back up the production files

**Files:**

- Back up: `handlers/display_manager.py`
- Back up: `pyside_overlay.py`

1. Create `.codex/backups/YYYY-MM-DD_HH-mm-ss/`.
2. Copy both files while preserving their workspace-relative paths.
3. Compare SHA-256 hashes of each source and backup before editing.

## Task 3: Implement idempotent font application

**Files:**

- Modify: `pyside_overlay.py`
- Test: `tests/test_pyside_text_render.py`

1. Add `_apply_font_if_changed(font_family, font_size) -> bool` to
   `RTLTextDisplay`.
2. Preserve the current family when the optional family is omitted, normalize
   point size, and compare against the active `QFont`.
3. Only call `setFont()` and log when family or size really changes.
4. Extend `set_rtl_text()` with the backward-compatible optional
   `font_family=None` parameter and invoke the helper before constructing HTML.
5. Change `config(font=...)` to use the helper and rerender stored text once
   only after a real font change.
6. Run the focused tests and keep them green:

   ```powershell
   py -m unittest tests.test_pyside_text_render -v
   ```

## Task 4: Make DisplayManager render once

**Files:**

- Modify: `handlers/display_manager.py`
- Test: `tests/test_pyside_text_render.py`

1. Pass `font_family=font_type` to the existing PySide `set_rtl_text()` call.
2. Remove the immediately following `configure(font=...)` call.
3. Rerun the focused tests.

## Task 5: Verify behavior and regressions

**Files:**

- Verify: `handlers/display_manager.py`
- Verify: `pyside_overlay.py`
- Verify: `tests/test_pyside_text_render.py`

1. Run focused and adjacent overlay tests:

   ```powershell
   py -m unittest tests.test_pyside_text_render tests.test_overlay_startup tests.test_overlay_geometry tests.test_modern_ui -v
   ```

2. Run a deterministic offscreen render probe that sends 100 display updates
   through `DisplayManager` into a counting real `RTLTextDisplay`; require 100,
   not 200, `setHtml()` calls.
3. Run the full suites and static checks:

   ```powershell
   py -m unittest discover -s tests
   py -m unittest discover
   py -m compileall -q handlers/display_manager.py pyside_overlay.py tests/test_pyside_text_render.py
   git diff --check
   ```

4. Review the scoped diff for compatibility, error handling, and unrelated
   edits. Address any review finding with another red/green test.

## Task 6: Record the verified increment

**Files:**

- Create: `.codex/handoffs/YYYY-MM-DD_HH-mm-ss.md`

Document the task, changed files, backup location, red/green evidence, focused
and full verification results, design decisions, measured render reduction, and
remaining project-level optimization candidates.
