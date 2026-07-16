# Translation Display Layout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (\`- [ ]\`) syntax for tracking.

**Goal:** Make translated subtitles prefer a single line until the overlay width requires wrapping, preserve source-line layout when selected, and optionally center translated text horizontally.

**Architecture:** Persist a normalized layout enum and a centering boolean while retaining the legacy \`keep_linebreaks\` value as a compatibility mirror. The display manager passes the selected rendering policy into the PySide text widget, which alone decides whether explicit model breaks are preserved or flattened; Qt continues to own width-based wrapping.

**Tech Stack:** Python 3, Tkinter/ttk, PySide6 \`QTextEdit\`, \`unittest\`, INI configuration.

---

## File structure

- \`config_manager.py\`: defaults, legacy migration, and layout normalization.
- \`app_logic.py\`: Tk variables and trace wiring.
- \`gui_settings_builder.py\`: localized line-layout selector and horizontal-center checkbox.
- \`handlers/ui_interaction_handler.py\`: persistence for both new settings and the legacy mirror.
- \`handlers/display_manager.py\`: forwards the selected rendering policy to PySide.
- \`pyside_overlay.py\`: whitespace normalization, HTML alignment, and \`QTextBlockFormat\` alignment.
- \`resources/gui_eng.csv\`, \`resources/gui_zh.csv\`, \`resources/gui_pol.csv\`: labels for the new settings.
- \`ocr_translator_config.example.ini\`: documented default settings only; do not edit the user's \`ocr_translator_config.ini\`.
- \`tests/test_translation_display_layout.py\`: isolated config, request prompt, display forwarding, and PySide rendering regressions.

### Task 1: Add isolated regression coverage

**Files:**
- Create: \`tests/test_translation_display_layout.py\`

- [ ] **Step 1: Write failing configuration and prompt tests**

\`\`\`python
def test_load_app_config_migrates_legacy_keep_linebreaks_to_preserve_layout():
    config = config_manager.load_app_config()
    self.assertEqual(config['Settings']['translation_line_layout'], 'preserve_source_lines')
    self.assertEqual(config['Settings']['keep_linebreaks'], 'True')

def test_compact_translation_prompt_forbids_explicit_line_breaks():
    payload = provider.build_translation_payload(profile, 'one\\ntwo', 'en', 'zh-CN', keep_linebreaks=False)
    self.assertIn('single line', json.dumps(payload).lower())
    self.assertIn('do not use <br>', json.dumps(payload).lower())
\`\`\`

- [ ] **Step 2: Run test to verify RED**

Run: \`py -m unittest tests.test_translation_display_layout -v\`

Expected: FAIL because layout migration and the stronger compact instruction do not exist.

- [ ] **Step 3: Write failing display tests**

\`\`\`python
def test_display_manager_forwards_compact_layout_and_centering():
    DisplayManager(app)._update_translation_text_on_main_thread('one<br>two')
    translation_text.set_rtl_text.assert_called_once_with(
        'one\\ntwo', 'zh-CN', '#112233', '#ffffff', 20,
        font_family='Microsoft YaHei',
        preserve_linebreaks=False,
        horizontal_centered=True,
    )

def test_compact_pyside_render_flattens_breaks_and_centers():
    widget.set_rtl_text('one<br>two\\nthree', 'en', preserve_linebreaks=False, horizontal_centered=True)
    self.assertIn('one two three', widget.toPlainText())
    self.assertNotIn('<br>', widget.html_calls[-1])
    self.assertIn('text-align: center', widget.html_calls[-1])

def test_preserve_layout_keeps_explicit_breaks_and_default_alignment():
    widget.set_rtl_text('one<br>two\\nthree', 'en', preserve_linebreaks=True)
    self.assertEqual(widget.toPlainText().splitlines(), ['one', 'two', 'three'])
    self.assertIn('text-align: left', widget.html_calls[-1])
\`\`\`

- [ ] **Step 4: Run focused tests to verify RED**

Run: \`py -m unittest tests.test_translation_display_layout -v\`

Expected: FAIL because the PySide entry point does not accept the two new keyword arguments.

### Task 2: Implement persistent layout settings and migration

**Files:**
- Modify: \`config_manager.py:27-105,175-277\`
- Modify: \`app_logic.py:264-273,515-524\`
- Modify: \`handlers/ui_interaction_handler.py:1243-1263\`
- Modify: \`ocr_translator_config.example.ini\`
- Test: \`tests/test_translation_display_layout.py\`

- [ ] **Step 1: Add normalized defaults and legacy migration**

\`\`\`python
TRANSLATION_LINE_LAYOUT_COMPACT = 'compact'
TRANSLATION_LINE_LAYOUT_PRESERVE_SOURCE_LINES = 'preserve_source_lines'

def normalize_translation_line_layout(value):
    normalized = str(value or '').strip().lower()
    if normalized == TRANSLATION_LINE_LAYOUT_PRESERVE_SOURCE_LINES:
        return TRANSLATION_LINE_LAYOUT_PRESERVE_SOURCE_LINES
    return TRANSLATION_LINE_LAYOUT_COMPACT
\`\`\`

Add \`translation_line_layout='compact'\` and \`translation_horizontal_centered='False'\` to \`DEFAULT_CONFIG_SETTINGS\`. Before the missing-default loop, derive a missing new layout key from \`keep_linebreaks\`; after normalization, synchronize \`keep_linebreaks\` to \`str(layout == 'preserve_source_lines')\` and mark the config changed only when a persisted value differs.

- [ ] **Step 2: Add application variables and persistence**

\`\`\`python
self.translation_line_layout_var = tk.StringVar(
    value=normalize_translation_line_layout(self.config['Settings'].get('translation_line_layout'))
)
self.translation_horizontal_centered_var = tk.BooleanVar(
    value=self.config.getboolean('Settings', 'translation_horizontal_centered', fallback=False)
)
self.keep_linebreaks_var = tk.BooleanVar(
    value=self.translation_line_layout_var.get() == 'preserve_source_lines'
)
\`\`\`

Trace both new variables with the existing settings callback. In \`save_settings\`, persist the normalized enum and center boolean, then write the legacy mirror from the enum rather than independently reading the old variable.

- [ ] **Step 3: Update the example configuration**

\`\`\`ini
translation_line_layout = compact
translation_horizontal_centered = False
\`\`\`

Do not change the active \`ocr_translator_config.ini\`, because it is user runtime state.

- [ ] **Step 4: Run the configuration-focused tests to verify GREEN**

Run: \`py -m unittest tests.test_translation_display_layout -v\`

Expected: migration and default tests pass; rendering tests still fail until Task 4.

- [ ] **Step 5: Commit the settings foundation**

\`\`\`powershell
git add config_manager.py app_logic.py handlers/ui_interaction_handler.py ocr_translator_config.example.ini tests/test_translation_display_layout.py
git commit -m "feat: persist translation display layout"
\`\`\`

### Task 3: Expose controls and strengthen compact requests

**Files:**
- Modify: \`gui_settings_builder.py:1340-1350,1486-1554\`
- Modify: \`custom_ai_capabilities.py:928-972\`
- Modify: \`resources/gui_eng.csv\`
- Modify: \`resources/gui_zh.csv\`
- Modify: \`resources/gui_pol.csv\`
- Test: \`tests/test_translation_display_layout.py\`

- [ ] **Step 1: Replace the old checkbox with an explicit selector**

\`\`\`python
layout_display_to_value = {
    app.ui_lang.get_label('translation_line_layout_compact', 'Prefer one line'): 'compact',
    app.ui_lang.get_label('translation_line_layout_preserve', 'Match source subtitle lines'): 'preserve_source_lines',
}
layout_combo = ttk.Combobox(frame, state='readonly', values=list(layout_display_to_value))
\`\`\`

Initialize the displayed value from \`translation_line_layout_var\`; on selection, set the enum and set \`keep_linebreaks_var\` to whether the enum is \`preserve_source_lines\`. Place the separate \`translation_horizontal_centered_var\` checkbutton with the target-output font and opacity controls.

- [ ] **Step 2: Add localization entries in every shipped GUI CSV**

\`\`\`csv
translation_line_layout_label,Translation subtitle layout:
translation_line_layout_compact,Prefer one line
translation_line_layout_preserve,Match source subtitle lines
translation_horizontal_centered,Center translated subtitle horizontally
\`\`\`

Use faithful Chinese and Polish translations in the corresponding files, retaining CSV encoding and one key per row.

- [ ] **Step 3: Make compact Custom AI requests unambiguous**

\`\`\`python
linebreak_instruction = (
    'Preserve line breaks using <br>.'
    if keep_linebreaks
    else 'Return one concise single-line translated text. Do not use <br> or newline characters.'
)
\`\`\`

- [ ] **Step 4: Run prompt and GUI-import tests**

Run: \`py -m unittest tests.test_translation_display_layout tests.test_custom_ai_startup -v\`

Expected: PASS for layout prompt behavior and existing startup coverage.

- [ ] **Step 5: Commit the settings UI and request contract**

\`\`\`powershell
git add gui_settings_builder.py custom_ai_capabilities.py resources/gui_eng.csv resources/gui_zh.csv resources/gui_pol.csv tests/test_translation_display_layout.py
git commit -m "feat: add translation layout controls"
\`\`\`

### Task 4: Enforce layout and horizontal alignment in PySide rendering

**Files:**
- Modify: \`handlers/display_manager.py:223-284\`
- Modify: \`pyside_overlay.py:99-244,299-340,627-641\`
- Test: \`tests/test_translation_display_layout.py\`
- Test: \`tests/test_pyside_text_render.py\`

- [ ] **Step 1: Forward policy from the display manager**

\`\`\`python
preserve_linebreaks = (
    self.app.translation_line_layout_var.get() == 'preserve_source_lines'
)
horizontal_centered = bool(self.app.translation_horizontal_centered_var.get())
self.app.translation_text.set_rtl_text(
    new_text_to_display, target_lang_code, bg_color, text_color, font_size,
    font_family=font_type,
    preserve_linebreaks=preserve_linebreaks,
    horizontal_centered=horizontal_centered,
)
\`\`\`

Keep conversion of incoming \`<br>\` tags to logical newlines in the display manager so the renderer receives a single canonical representation.

- [ ] **Step 2: Add backwards-compatible PySide parameters and state**

\`\`\`python
def set_rtl_text(
    self, text, language_code=None, bg_color="#2c3e50", text_color="#ecf0f1",
    font_size=14, font_family=None, preserve_linebreaks=True,
    horizontal_centered=False,
):
    self._preserve_linebreaks = bool(preserve_linebreaks)
    self._horizontal_centered = bool(horizontal_centered)
\`\`\`

When preserving, normalize line endings and collapse whitespace inside each line. When compacting, replace \`<br>\` (case-insensitively), CRLF, CR, and LF with spaces, then collapse all whitespace to one space. Do not disable \`QTextEdit.WidgetWidth\` wrapping.

- [ ] **Step 3: Apply horizontal center only to the horizontal axis**

\`\`\`python
alignment_css = 'center' if horizontal_centered else ('right' if is_rtl else 'left')
block_alignment = Qt.AlignHCenter if horizontal_centered else (
    Qt.AlignRight | Qt.AlignAbsolute if is_rtl else Qt.AlignLeft | Qt.AlignAbsolute
)
\`\`\`

Use \`alignment_css\` in the HTML \`div\` and \`block_alignment\` in \`QTextBlockFormat\`. Keep existing layouts, margins, geometry, and vertical alignment unchanged. Reuse stored policy when color or font changes rerender current text, and preserve the old five-argument call contract via defaults.

- [ ] **Step 4: Run focused rendering tests to verify GREEN**

Run: \`py -m unittest tests.test_translation_display_layout tests.test_pyside_text_render -v\`

Expected: PASS, with PySide tests skipped only when PySide6 is unavailable.

- [ ] **Step 5: Commit rendering behavior**

\`\`\`powershell
git add handlers/display_manager.py pyside_overlay.py tests/test_translation_display_layout.py tests/test_pyside_text_render.py
git commit -m "feat: control translation line layout and alignment"
\`\`\`

### Task 5: Verify and record handoff

**Files:**
- Create: \`.codex/handoffs/YYYY-MM-DD_HH-mm-ss.md\`

- [ ] **Step 1: Run the full offline suite**

Run: \`py scripts/run_offline_tests.py\`

Expected: all discovered tests pass.

- [ ] **Step 2: Compile every changed Python module**

Run: \`py -B -m py_compile config_manager.py app_logic.py gui_settings_builder.py handlers/ui_interaction_handler.py custom_ai_capabilities.py handlers/display_manager.py pyside_overlay.py tests/test_translation_display_layout.py tests/test_pyside_text_render.py\`

Expected: exit code 0.

- [ ] **Step 3: Check final diff and local state**

Run: \`git diff --check HEAD~3..HEAD\` and \`git status --short\`

Expected: no whitespace errors; only intentionally untracked user files remain.

- [ ] **Step 4: Create the required handoff**

Record task summary, changed files, backup directory, commands and results, legacy migration, the facts that compact mode does not suppress width-based wrapping and centering is horizontal only, commit hashes, and any GUI smoke-test limitation.

- [ ] **Step 5: Commit the plan and handoff documents**

\`\`\`powershell
git add docs/superpowers/plans/2026-07-16-translation-display-layout.md .codex/handoffs/YYYY-MM-DD_HH-mm-ss.md
git commit -m "docs: hand off translation display layout"
\`\`\`
