# Subtitle Bold, Larger Spinbox Arrows, and Stop-Aware Failover Design

## Goal

Add a persistent translated-subtitle bold option, make every settings-page
spinbox arrow easier to click, and stop sequential Custom AI failover from
starting another provider request after translation has been stopped.

## Runtime evidence

The latest meaningful session ran from 2026-07-16 15:24:02 through 15:29:37.
Local capture and PaddleOCR were generally around 0.1 seconds, while translation
had 73 completed timing samples with a 4.562-second median, 11.859-second p95,
and 34.156-second maximum.

After the user stopped translation at 15:29:11, two already-running translation
jobs continued sequential failover. The log emitted 269 pending-call wait
messages and 256 lines before the final obsolete request ended at 15:29:36.
The existing freshness guard eventually stopped more failover after a newer
result displayed, but the failover loop does not currently treat
`app.is_running == False` as an abort condition.

## Chosen design

### Subtitle bold

Use one boolean setting named `target_font_bold`, defaulting to `False` so old
configurations retain their current appearance. Expose it as a checkbutton
beside the existing target font controls. Save it through the normal settings
path and update the active overlay immediately when toggled.

The value travels through the existing font pipeline:

- `config_manager.py` and the example INI define the persistent default.
- `app_logic.py` owns `target_font_bold_var` and its save trace.
- `app_configuration.py` forwards the public font-weight update command.
- `gui_settings_builder.py` creates the localized checkbutton.
- `handlers/ui_interaction_handler.py` applies family, size, and weight as one
  font tuple and persists the setting.
- `overlay_manager.py` applies the weight when either a PySide or Tk fallback
  overlay is created.
- `handlers/display_manager.py` forwards the weight on each PySide render.
- `pyside_overlay.py` applies both `QFont.setBold()` and explicit HTML
  `font-weight`, preserving the setting during color changes, resize rerenders,
  and compatibility `config(font=...)` calls.

### Larger numeric stepper buttons

Configure the existing global `TSpinbox` style once in `modern_ui.py` with an
arrow size of 16 pixels and padding of `(5, 3)`. This enlarges the highlighted
up/down controls consistently without changing each spinbox's numeric range,
increment, width, validation, or mouse-wheel behavior.

### Stop-aware sequential failover

Keep the active HTTP request bounded by its existing timeout. Python cannot
safely terminate a blocking request already executing in another thread.
Immediately after that request returns or fails, however, the failover loop
must check whether the app is still running before it starts the next profile.

Add a stopped-app abort predicate alongside the existing obsolete-sequence
predicate and call the combined abort guard:

- before every candidate network request;
- after a candidate error, before continuing to the next provider.

Apps or unit-test fixtures without an `is_running` attribute retain current
behavior. The log records one concise stop-aware abort message. Cache lookup
and provider cooldown behavior otherwise remain unchanged.

## Error handling and compatibility

- Missing or malformed `target_font_bold` values fall back to `False`.
- Existing font APIs that pass only family and size remain valid and preserve
  the current weight.
- A Tk font tuple may include the third `"bold"` style item; normal weight uses
  the existing two-item tuple.
- Stop-aware failover does not mark untried profiles failed and does not change
  cooldown durations.
- No live `ocr_translator_config.ini` values are edited directly.

## Testing

Use test-first changes to prove:

1. Display forwarding includes `font_bold=True`.
2. PySide applies bold before HTML rendering, includes CSS font weight, and
   rerenders exactly once when compatibility font configuration changes weight.
3. The settings handler applies a Tk-compatible bold font tuple.
4. The global spinbox theme requests 16-pixel arrows and `(5, 3)` padding.
5. Sequential failover attempts only the active profile when that failure also
   changes `app.is_running` to `False`.

Then run the focused suites, all `tests/`, root legacy Custom AI tests,
compilation for touched Python modules, and `git diff --check`.
