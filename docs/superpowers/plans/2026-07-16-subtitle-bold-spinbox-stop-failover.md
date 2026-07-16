# Subtitle Bold, Larger Spinbox Arrows, and Stop-Aware Failover Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add persistent subtitle bold rendering, enlarge settings spinbox arrows, and prevent another sequential provider attempt after translation stops.

**Architecture:** Extend the existing target-font setting pipeline with one boolean and keep PySide/Tk behavior equivalent. Configure spinbox dimensions at the shared ttk theme boundary. Extend the existing Custom AI failover abort boundary with app-running state while leaving in-flight HTTP timeout behavior unchanged.

**Tech Stack:** Python 3, Tkinter/ttk, PySide6, configparser, unittest

---

### Task 1: Add failing behavior tests

**Files:**
- Modify: `tests/test_translation_display_layout.py`
- Modify: `tests/test_pyside_text_render.py`
- Modify: `tests/test_ui_elements.py`
- Modify: `tests/test_custom_ai.py`

- [ ] **Step 1: Add a display-forwarding assertion for bold**

Give the display fixture `target_font_bold_var=_Value(True)` and require:

```python
translation_text.set_rtl_text.assert_called_once_with(
    "one two",
    "zh-CN",
    "#112233",
    "#ffffff",
    20,
    font_family="Microsoft YaHei",
    font_bold=True,
    preserve_linebreaks=False,
    horizontal_centered=True,
)
```

- [ ] **Step 2: Add PySide font-weight tests**

Require direct rendering to leave `widget.font().bold()` true and the emitted
HTML to contain `font-weight: 700`. Require
`widget.config(font=(family, 18, "bold"))` to rerender once when only the
weight changes.

- [ ] **Step 3: Add a theme-style unit test**

Use a recording style double and call the spinbox style helper:

```python
style = RecordingStyle()
_configure_spinbox_style(style, {"surface": "#fff", "text": "#111",
                                 "outline": "#ccc", "text_muted": "#666"})
self.assertEqual(style.options["arrowsize"], 16)
self.assertEqual(style.options["padding"], (5, 3))
```

- [ ] **Step 4: Add a stopped-failover regression test**

Start with `app.is_running = True`. In the active profile's mocked translate
call, set it to `False` and raise an error. Assert the return is `None` and the
fallback profile was never attempted.

- [ ] **Step 5: Run the tests and verify RED**

Run:

```powershell
py -m unittest tests.test_translation_display_layout tests.test_pyside_text_render tests.test_ui_elements tests.test_custom_ai.CostProtectedProfileFailoverHandlerTests -v
```

Expected: failures for the missing `font_bold` forwarding/API, missing spinbox
style helper, and missing stopped-app failover abort.

### Task 2: Persist and expose subtitle bold

**Files:**
- Modify: `config_manager.py`
- Modify: `app_logic.py`
- Modify: `app_configuration.py`
- Modify: `gui_settings_builder.py`
- Modify: `handlers/ui_interaction_handler.py`
- Modify: `ocr_translator_config.example.ini`
- Modify: `resources/gui_eng.csv`
- Modify: `resources/gui_zh.csv`
- Modify: `resources/gui_pol.csv`

- [ ] **Step 1: Add the configuration value and Tk variable**

Add `target_font_bold = False` to defaults and example configuration. Create:

```python
self.target_font_bold_var = tk.BooleanVar(
    value=self.config.getboolean("Settings", "target_font_bold", fallback=False)
)
```

Register it with the existing settings change trace.

- [ ] **Step 2: Add localized checkbutton labels**

Add the `font_bold_label` key in English, Chinese, and Polish, then create a
checkbutton after the font-family selector. Its command calls
`app.update_target_font_weight()`; the normal variable trace persists it.

- [ ] **Step 3: Apply and save the combined font**

Create one internal handler helper that builds:

```python
(font_family, font_size, "bold") if font_bold else (font_family, font_size)
```

Use it from size, family, and weight update methods. Save
`cfg["target_font_bold"]` as the boolean string.

- [ ] **Step 4: Run focused settings tests**

Run:

```powershell
py -m unittest tests.test_translation_display_layout -v
```

Expected: configuration and forwarding tests pass once renderer changes from
Task 3 are also present.

### Task 3: Render bold in PySide and Tk

**Files:**
- Modify: `overlay_manager.py`
- Modify: `handlers/display_manager.py`
- Modify: `pyside_overlay.py`
- Modify: `tests/test_translation_display_layout.py`
- Modify: `tests/test_pyside_text_render.py`

- [ ] **Step 1: Forward bold when overlays are created and rendered**

Read `target_font_bold_var` with a `False` fallback. Pass `font_bold` to the
PySide overlay constructor and `set_rtl_text`. For Tk, append `"bold"` to the
font tuple only when enabled.

- [ ] **Step 2: Extend the PySide font API**

Extend `_apply_font_if_changed` and `set_rtl_text` with optional
`font_bold`. Apply `QFont.setBold`, store `_font_bold`, and emit:

```python
font_weight_css = "700" if self._font_bold else "400"
```

Include `font-weight` in both RTL and LTR HTML.

- [ ] **Step 3: Preserve weight across compatibility rerenders**

Pass `_font_bold` during foreground-color rerenders. Parse any third and later
font tuple items for `"bold"` in `config(font=...)`, and rerender only when
family, size, or weight actually changes.

- [ ] **Step 4: Run focused rendering tests**

Run:

```powershell
py -m unittest tests.test_translation_display_layout tests.test_pyside_text_render -v
```

Expected: all tests pass.

### Task 4: Enlarge all settings spinbox arrows

**Files:**
- Modify: `modern_ui.py`
- Modify: `tests/test_ui_elements.py`

- [ ] **Step 1: Add the shared style helper**

Create `_configure_spinbox_style(style, palette)` and configure `TSpinbox`
with the existing colors plus:

```python
arrowsize=16,
padding=(5, 3),
```

Call the helper from `apply_white_clean_theme`.

- [ ] **Step 2: Run the style and wheel-input tests**

Run:

```powershell
py -m unittest tests.test_ui_elements -v
```

Expected: the pure style test passes; display-dependent Tk tests either pass or
are explicitly skipped when no display is available.

### Task 5: Abort sequential failover after stop

**Files:**
- Modify: `handlers/translation_requests.py`
- Modify: `tests/test_custom_ai.py`

- [ ] **Step 1: Add a stopped-app abort predicate**

Return false when `is_running` is absent. When present and false, log:

```text
LATENCY: custom_ai failover aborted because app stopped sequence=<value>
```

and return true.

- [ ] **Step 2: Combine it with the existing obsolete-sequence guard**

Use the combined guard before each candidate request and after each candidate
error. Return `None` without marking or attempting another provider.

- [ ] **Step 3: Run the sequential failover tests**

Run:

```powershell
py -m unittest tests.test_custom_ai.CostProtectedProfileFailoverHandlerTests -v
```

Expected: normal fallback still attempts two profiles, obsolete output stops at
one, and stopped-app failover stops at one.

### Task 6: Verify the complete change

**Files:**
- Verify all files listed above.

- [ ] **Step 1: Run focused suites**

```powershell
py -m unittest tests.test_translation_display_layout tests.test_pyside_text_render tests.test_ui_elements tests.test_custom_ai.CostProtectedProfileFailoverHandlerTests -v
```

- [ ] **Step 2: Run the complete tests directory**

```powershell
py -m unittest discover -s tests
```

- [ ] **Step 3: Run root legacy Custom AI tests**

```powershell
py -m unittest test_custom_ai test_custom_ai_startup -q
```

- [ ] **Step 4: Compile touched Python modules and tests**

```powershell
py -B -m py_compile config_manager.py app_logic.py app_configuration.py gui_settings_builder.py modern_ui.py overlay_manager.py pyside_overlay.py handlers/ui_interaction_handler.py handlers/display_manager.py handlers/translation_requests.py tests/test_translation_display_layout.py tests/test_pyside_text_render.py tests/test_ui_elements.py tests/test_custom_ai.py
```

- [ ] **Step 5: Check diff hygiene**

```powershell
git diff --check
git status --short
```

Expected: zero test failures, successful compilation, no whitespace errors, and
the pre-existing untracked release-note file remains untouched.
