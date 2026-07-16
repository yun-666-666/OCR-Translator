# Subtitle Outline Visibility and Color Grid Correction Design

## Problem

The active subtitle appearance uses red text, a bright green outline, font size
22, bold FangSong, and outline width 2. The current implementation passes the
configured value directly to `QPen.setWidthF(2)`. Qt draws that outline across
the glyph edge, including inward over narrow Chinese strokes, so most of the
red fill is covered by green. Only thick round punctuation retains an obvious
red centre.

The four clickable color controls are also packed into one horizontal row,
which makes the settings page unnecessarily wide and crowded.

## Evidence

A real Windows Qt render using the active font, colors, size, weight, and sample
text compared actual pen widths 2.0, 1.0, 0.75, and 0.5:

- 2.0 reproduced the reported almost-solid green glyphs;
- 1.0 retained clear red fill and a visible green boundary;
- 0.75 was usable but the boundary became weak;
- 0.5 was too subtle to provide reliable separation.

## Selected rendering correction

Keep the user-facing outline setting range at `0` through `6`, but convert it
to the native Qt pen width with:

```python
pen_width = configured_outline_width * 0.5
```

The configured value remains the persistent state used by settings, overlay
construction, live updates, and rerenders. Only the final `QPen.setWidthF()`
value is scaled. This makes the current setting `2` render as the verified
native width `1.0`.

Alternatives rejected:

- Changing only the default or current config to `1` would leave every other
  configured width twice as aggressive and would not correct the meaning of
  the control.
- Replacing Qt rich text with a custom two-pass painter would provide more
  control but would duplicate wrapping, selection, RTL, and document layout
  behavior for no demonstrated benefit.

## Selected settings layout

Keep the existing color order and clickable swatches, but lay the four controls
out in a two-column grid:

1. row 0, column 0: source area color;
2. row 0, column 1: target area color;
3. row 1, column 0: translated text color;
4. row 1, column 1: translated text outline color.

A small pure position helper returns `divmod(index, 2)` so the intended 2x2
mapping is covered by a real unit test rather than only source-text checks.

## Compatibility and scope

- Saved outline values remain unchanged.
- Width zero still disables the outline.
- Text opacity continues to apply to both fill and outline.
- Color selection, persistence, localization, font controls, overlay
  construction, and tkinter fallback behavior remain unchanged.
- No OCR, translation, window geometry, target background, or provider logic is
  modified.

## Tests

1. A configured outline width of `2` produces a native pen width of `1.0`.
2. A configured width of `3` produces `1.5`; zero still uses `Qt.NoPen`.
3. Font and foreground rerenders preserve the configured width while applying
   the scaled native width.
4. Color indices `0, 1, 2, 3` map to `(0,0), (0,1), (1,0), (1,1)`.
5. The settings builder uses grid placement for color controls and no longer
   packs all four items horizontally.
6. Existing focused and full test suites continue to pass.
