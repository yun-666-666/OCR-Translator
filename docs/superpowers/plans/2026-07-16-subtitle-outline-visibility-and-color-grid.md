# Subtitle Outline Visibility and Color Grid Correction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent translated-subtitle outlines from covering the fill color and arrange the four settings color controls as a 2x2 grid.

**Architecture:** Keep persistent outline settings unchanged, but scale the final Qt pen width through one pure conversion helper. Replace horizontal packing of color items with grid placement driven by a pure two-column position helper.

**Tech Stack:** Python 3, PySide6 rich text, tkinter/ttk settings UI, `unittest`.

---

### Task 1: Back up the existing files

**Files:**
- Modify: `pyside_overlay.py`
- Modify: `gui_settings_builder.py`
- Modify: `tests/test_pyside_text_render.py`
- Modify: `tests/test_ui_elements.py`

- [ ] **Step 1: Create a timestamped backup**

Copy the four files to `.codex/backups/YYYY-MM-DD_HH-mm-ss/`, preserving their
relative paths.

- [ ] **Step 2: Verify SHA-256**

Require each source and backup pair to have identical SHA-256 hashes before
editing.

### Task 2: Scale configured width to a thinner Qt pen

**Files:**
- Modify: `tests/test_pyside_text_render.py`
- Modify: `pyside_overlay.py`

- [ ] **Step 1: Write failing tests**

Change the native-outline expectations so configured width `2` expects pen
width `0.5`, and configured width `3` expects `0.75`. Keep the zero-width
`Qt.NoPen` assertion.

- [ ] **Step 2: Run the focused tests and verify RED**

```powershell
py -m unittest tests.test_pyside_text_render.RTLTextDisplayFontTests -v
```

Expected: failures showing actual widths `2.0` and `3.0`.

- [ ] **Step 3: Implement the conversion**

Add:

```python
def _outline_pen_width(configured_width):
    try:
        width = float(configured_width)
    except (TypeError, ValueError):
        width = 2.0
    return max(0.0, min(6.0, width)) * 0.25
```

Keep `_outline_width` as the configured value. Use
`_outline_pen_width(width)` only for `QPen.setWidthF()`.

- [ ] **Step 4: Run the focused tests and verify GREEN**

```powershell
py -m unittest tests.test_pyside_text_render.RTLTextDisplayFontTests -v
```

Expected: PASS.

### Task 3: Change the color controls to a 2x2 grid

**Files:**
- Modify: `tests/test_ui_elements.py`
- Modify: `gui_settings_builder.py`

- [ ] **Step 1: Write failing layout tests**

Import and test:

```python
_settings_color_grid_position(0) == (0, 0)
_settings_color_grid_position(1) == (0, 1)
_settings_color_grid_position(2) == (1, 0)
_settings_color_grid_position(3) == (1, 1)
```

Update the source contract to require `color_item_frame.grid(` and reject
`color_item_frame.pack(`.

- [ ] **Step 2: Run the focused tests and verify RED**

```powershell
py -m unittest tests.test_ui_elements.SettingsLayoutSourceTests -v
```

Expected: import or assertion failure because the helper/grid placement is
missing.

- [ ] **Step 3: Implement the two-column grid**

Add:

```python
def _settings_color_grid_position(index):
    return divmod(int(index), 2)
```

For each color item, calculate `grid_row, grid_column` and call
`color_item_frame.grid(row=grid_row, column=grid_column, ...)`. Preserve
clickable swatches, color order, and `app.color_displays`.

- [ ] **Step 4: Run the focused tests and verify GREEN**

```powershell
py -m unittest tests.test_ui_elements.SettingsLayoutSourceTests -v
```

Expected: PASS.

### Task 4: Verify the reported visual case

**Files:**
- Create: `.codex/visual-tests/subtitle-outline-corrected.png`

- [ ] **Step 1: Render the active appearance**

Use real Windows Qt with red fill, green outline, FangSong 22 bold, and
configured width `2`.

- [ ] **Step 2: Inspect the image**

Confirm the red glyph fill remains plainly visible and the green boundary is
thin but continuous.

### Task 5: Full verification and handoff

**Files:**
- Create: `.codex/handoffs/YYYY-MM-DD_HH-mm-ss.md`

- [ ] **Step 1: Run focused suites**

```powershell
py -m unittest tests.test_pyside_text_render tests.test_ui_elements -v
```

- [ ] **Step 2: Run complete suites**

```powershell
py -m unittest discover -s tests -q
py -m unittest -q
```

- [ ] **Step 3: Run compile and diff checks**

```powershell
py -B -m py_compile pyside_overlay.py gui_settings_builder.py tests\test_pyside_text_render.py tests\test_ui_elements.py
git diff --check
git diff --cached --check
```

- [ ] **Step 4: Create the handoff and commit**

Record the root cause, backup location, RED/GREEN evidence, visual evidence,
tests, and remaining limitations. Stage only the intended tracked files and
commit without staging unrelated release notes or `.superpowers/`.
