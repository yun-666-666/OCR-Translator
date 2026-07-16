# Translated Subtitle Outline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add configurable crisp glyph outlines to translated PySide subtitles and set this machine to yellow text with a black 2 px outline.

**Architecture:** Persist outline color and width beside the existing target text settings, forward them through the display and overlay layers, and apply a `QTextCharFormat` text-outline pen after every HTML render. Preserve all legacy call signatures and leave the tkinter fallback functional without outline support.

**Tech Stack:** Python 3, tkinter settings UI, PySide6 `QTextEdit` rich text, `unittest`, INI configuration.

---

## File structure

- `config_manager.py`: fresh-install defaults and missing-key migration.
- `app_logic.py`: tkinter variables and automatic save traces.
- `app_configuration.py`: public UI update forwarding.
- `handlers/ui_interaction_handler.py`: color chooser, live update, validation, and persistence.
- `gui_settings_builder.py`: outline color swatch and width spinbox.
- `handlers/display_manager.py`: per-subtitle forwarding.
- `overlay_manager.py`: initial overlay construction values.
- `pyside_overlay.py`: native glyph-outline rendering and live rerender API.
- `resources/gui_eng.csv`, `resources/gui_zh.csv`, `resources/gui_pol.csv`: localized labels.
- `ocr_translator_config.example.ini`: documented example defaults.
- `ocr_translator_config.ini`: ignored machine-local yellow/black active settings.
- `tests/test_pyside_text_render.py`: display forwarding and Qt pen behavior.
- `tests/test_overlay_startup.py`: initial overlay forwarding.
- `tests/test_ui_elements.py`: UI and persistence source contracts.
- `tests/test_translation_display_layout.py`: updated display call expectations.

### Task 1: Back up all existing files in scope

**Files:**
- Create: `.codex/backups/YYYY-MM-DD_HH-mm-ss/`
- Copy: every existing file listed in the file structure, excluding the new spec and plan.

- [ ] **Step 1: Create the timestamped backup tree**

Use PowerShell `Copy-Item -LiteralPath` for each file while preserving its
relative path below the backup directory.

- [ ] **Step 2: Verify every backup**

Run `Get-FileHash` for each source and backup pair and require identical
SHA-256 values before editing.

### Task 2: Define configuration and UI state

**Files:**
- Modify: `config_manager.py`
- Modify: `app_logic.py`
- Modify: `ocr_translator_config.example.ini`
- Test: `tests/test_ui_elements.py`

- [ ] **Step 1: Write failing default and state tests**

Add assertions equivalent to:

```python
from config_manager import DEFAULT_CONFIG_SETTINGS

self.assertEqual(DEFAULT_CONFIG_SETTINGS["target_text_colour"], "#FFD54F")
self.assertEqual(DEFAULT_CONFIG_SETTINGS["target_text_outline_colour"], "#000000")
self.assertEqual(DEFAULT_CONFIG_SETTINGS["target_text_outline_width"], "2")
```

Also assert that `app_logic.py` contains both outline variables and trace
registrations.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```powershell
py -m unittest tests.test_ui_elements -v
```

Expected: failures for the missing outline defaults and variables.

- [ ] **Step 3: Add minimal configuration and state**

Add:

```python
'target_text_colour': '#FFD54F',
'target_text_outline_colour': '#000000',
'target_text_outline_width': '2',
```

Initialize `target_text_outline_colour_var` as a `StringVar`. Parse
`target_text_outline_width` to an integer, fall back to `2`, clamp it to
`0..6`, and initialize an `IntVar`. Register both with
`settings_changed_callback`. Mirror the three values in the example INI.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run:

```powershell
py -m unittest tests.test_ui_elements -v
```

Expected: PASS.

### Task 3: Add settings controls, live updates, and persistence

**Files:**
- Modify: `gui_settings_builder.py`
- Modify: `app_configuration.py`
- Modify: `handlers/ui_interaction_handler.py`
- Modify: `resources/gui_eng.csv`
- Modify: `resources/gui_zh.csv`
- Modify: `resources/gui_pol.csv`
- Test: `tests/test_ui_elements.py`

- [ ] **Step 1: Write failing UI and persistence tests**

Assert that the settings builder includes `target_outline` in the color
options, creates a `target_text_outline_width_spinbox` with `from_=0, to=6`,
and that the save path writes:

```python
cfg['target_text_outline_colour']
cfg['target_text_outline_width']
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```powershell
py -m unittest tests.test_ui_elements -v
```

Expected: failures for missing swatch, spinbox, forwarding method, and save
keys.

- [ ] **Step 3: Implement the minimal settings behavior**

Extend the color chooser maps with `target_outline`. On selection, update the
variable and swatch, then call:

```python
self.app.target_overlay.update_text_outline(
    hex_color,
    self.app.target_text_outline_width_var.get(),
)
```

Add `update_target_text_outline()` in the handler and forward it through
`app_configuration.py`. The width spinbox clamps `0..6`, calls the update
method, and saves on focus-out. Persist both values in `save_settings()`.

Add localized keys:

```text
target_text_outline_color_label
choose_target_text_outline_color_title
target_text_outline_width_label
```

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run:

```powershell
py -m unittest tests.test_ui_elements -v
```

Expected: PASS.

### Task 4: Forward outline state through display and overlay startup

**Files:**
- Modify: `handlers/display_manager.py`
- Modify: `overlay_manager.py`
- Test: `tests/test_pyside_text_render.py`
- Test: `tests/test_translation_display_layout.py`
- Test: `tests/test_overlay_startup.py`

- [ ] **Step 1: Write failing forwarding tests**

Give test app doubles:

```python
target_text_outline_colour_var=_Value("#000000"),
target_text_outline_width_var=_Value(2),
```

Expect `set_rtl_text()` to receive:

```python
outline_color="#000000",
outline_width=2,
```

Expect overlay creation to receive:

```python
text_outline_color="#000000",
text_outline_width=2,
```

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```powershell
py -m unittest tests.test_pyside_text_render tests.test_translation_display_layout tests.test_overlay_startup -v
```

Expected: failures because outline arguments are not forwarded.

- [ ] **Step 3: Implement minimal forwarding**

Read both variables with safe fallbacks and pass them into the existing PySide
render call. During overlay creation, forward them as constructor keyword
arguments. Do not alter the tkinter creation path.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run:

```powershell
py -m unittest tests.test_pyside_text_render tests.test_translation_display_layout tests.test_overlay_startup -v
```

Expected: forwarding tests pass while Qt rendering tests may still fail until
Task 5.

### Task 5: Apply the native Qt glyph outline

**Files:**
- Modify: `pyside_overlay.py`
- Test: `tests/test_pyside_text_render.py`

- [ ] **Step 1: Write failing Qt formatting tests**

After rendering, select the document and assert:

```python
outline = cursor.charFormat().textOutline()
self.assertEqual(outline.color().name(), "#000000")
self.assertEqual(outline.widthF(), 2.0)
```

Add a second test with `outline_width=0` and assert
`outline.style() == Qt.NoPen`. Add a rerender test proving foreground and font
changes preserve the stored outline.

- [ ] **Step 2: Run the Qt tests and verify RED**

Run:

```powershell
py -m unittest tests.test_pyside_text_render -v
```

Expected: failures because `set_rtl_text()` does not accept or apply outline
arguments.

- [ ] **Step 3: Implement minimal native outline rendering**

Import `QColor`, `QPen`, and `QTextCharFormat`. Append optional
`outline_color` and `outline_width` parameters to `set_rtl_text()`, store
normalized values, and call `_apply_text_outline()` after `setHtml()`.

The helper selects the full document, builds a solid pen for positive widths or
`QPen(Qt.NoPen)` for zero, calls `setTextOutline()`, merges the character
format, clears selection, and restores the cursor to the start.

Extend `PySideTranslationOverlay` with constructor state and:

```python
def update_text_outline(self, color, width):
    self._text_outline_color = color
    self._text_outline_width = width
    self.text_widget.update_text_outline(color, width)
```

All existing color/font rerenders pass the stored outline values.

- [ ] **Step 4: Run the Qt tests and verify GREEN**

Run:

```powershell
py -m unittest tests.test_pyside_text_render -v
```

Expected: PASS.

### Task 6: Apply the requested local yellow/black appearance

**Files:**
- Modify: `ocr_translator_config.ini`

- [ ] **Step 1: Update only the relevant local keys**

Set:

```ini
target_text_colour = #FFD54F
target_text_outline_colour = #000000
target_text_outline_width = 2
```

- [ ] **Step 2: Verify parsed values**

Use `configparser` to read the ignored INI and print only the three relevant
values. Expected values must match exactly.

### Task 7: Full verification and handoff

**Files:**
- Create: `.codex/handoffs/YYYY-MM-DD_HH-mm-ss.md`

- [ ] **Step 1: Run focused suites**

```powershell
py -m unittest tests.test_pyside_text_render tests.test_translation_display_layout tests.test_overlay_startup tests.test_ui_elements -v
```

- [ ] **Step 2: Run the full main and root suites**

```powershell
py -m unittest discover -s tests -q
py -m unittest -q
```

- [ ] **Step 3: Run compile and diff hygiene**

```powershell
py -B -m py_compile config_manager.py app_logic.py app_configuration.py gui_settings_builder.py overlay_manager.py pyside_overlay.py handlers\display_manager.py handlers\ui_interaction_handler.py
git diff --check
```

- [ ] **Step 4: Review the diff against the design**

Confirm the diff contains no OCR/provider changes, no subtitle-positioning
changes, and no unrelated formatting.

- [ ] **Step 5: Create the handoff**

Record task summary, changed files, backup location, RED/GREEN evidence, all
verification results, design decisions, local ignored-config change, and any
remaining tkinter fallback limitation.

- [ ] **Step 6: Commit the implementation**

Stage only tracked feature, test, resource, plan, and handoff files. Do not
stage ignored local config, visual-companion files, release notes, backups, or
other unrelated artifacts.
