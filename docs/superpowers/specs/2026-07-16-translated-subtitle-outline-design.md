# Translated Subtitle Outline Design

## Goal

Make translated subtitles readable over bright, dark, and highly saturated
video by adding a configurable glyph outline. The intended local appearance is
warm yellow text (`#FFD54F`) with a black 2 px outline.

## Selected approach

Apply a native Qt text outline to the complete `QTextDocument` after the
existing HTML text has been rendered. `QTextCharFormat.setTextOutline()` with a
`QPen` outlines each glyph without replacing the existing wrapping, alignment,
font-family, font-size, bold, RTL, or text-opacity behavior.

Two alternatives were rejected:

- CSS or repeated `text-shadow` declarations are not reliably supported by
  Qt rich text and produce a soft rather than crisp edge.
- `QGraphicsDropShadowEffect` operates on the whole text widget, blurs the
  result, and can clip or distort at widget boundaries.

## Configuration

Add two persistent settings:

- `target_text_outline_colour`, default `#000000`;
- `target_text_outline_width`, default `2`.

Outline width is an integer in the inclusive range `0` to `6`. Zero disables
the outline. Missing values are added by the existing default-config migration.
Malformed or out-of-range values fall back to `2` at startup and are clamped by
the settings control.

Change the fresh-install default `target_text_colour` to `#FFD54F`. Existing
users retain their saved text color because default migration only fills
missing keys. The current ignored local config is explicitly changed from red
to `#FFD54F`, with black 2 px outline settings added.

## Settings UI

Extend the existing compact color row with a fourth clickable swatch labelled
`Outline colour`. Clicking it uses the existing color chooser and immediately
rerenders visible PySide subtitle text.

Add an `Outline width` spinbox beside the font controls. Its range is `0` to
`6`; focus-out validation clamps the value and applies it immediately. The
normal settings trace and save path persist both values.

English, Chinese, and Polish resources receive labels for the swatch, color
chooser title, and width control.

## Rendering data flow

1. `app_logic.py` owns `target_text_outline_colour_var` and
   `target_text_outline_width_var`.
2. `handlers/display_manager.py` reads both values for each translated
   subtitle and appends them to the PySide `set_rtl_text()` call.
3. `RTLTextDisplay.set_rtl_text()` stores the outline state, renders the
   existing HTML, then applies one `QTextCharFormat` across the document.
4. Color, font, resize, and compatibility rerenders reuse the stored outline
   state.
5. `PySideTranslationOverlay.update_text_outline()` allows live settings
   changes without recreating the overlay.

New optional arguments are appended to existing methods so all legacy
positional and five-argument calls remain valid.

## Fallback behavior

The native outline is available on the normal PySide overlay path. The legacy
tkinter `Text` fallback continues to display the selected foreground color
without an outline because it has no equivalent glyph-stroke API. The fallback
must remain functional and must not fail when the new settings exist.

## Error handling

- Invalid colors fall back to black when constructing the Qt pen.
- Width parsing failures use `2`; live UI values are clamped to `0` through
  `6`.
- Outline application is failure-safe: a rendering exception is logged and
  leaves ordinary text visible.
- Setting width to zero applies a `NoPen` outline and removes any prior stroke.

## Tests

Add or extend tests to verify:

1. default configuration is yellow text with black 2 px outline;
2. display forwarding includes outline color and width;
3. Qt document character formatting contains the expected pen color and width;
4. width zero removes the outline;
5. foreground and font rerenders preserve the active outline;
6. the settings UI contains the outline swatch and width control;
7. settings saving persists both outline keys;
8. overlay startup forwards the configured outline state;
9. legacy calls and tkinter fallback behavior remain valid.

## Scope boundaries

This change does not alter subtitle positioning, vertical alignment, window
opacity, source-region rendering, OCR, translation providers, or the existing
window border. It does not add blurred glow, multiple outline layers, adaptive
contrast analysis, or per-language outline profiles.
