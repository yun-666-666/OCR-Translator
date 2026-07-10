# Single-pass PySide text rendering design

## Problem and evidence

The active PySide translation display renders every completed translation twice.
`DisplayManager._update_translation_text_on_main_thread()` first calls
`RTLTextDisplay.set_rtl_text()`, which builds and installs the HTML document, and
then calls `configure(font=...)`. The compatibility `config()` method always
sets a `QFont` and calls `set_rtl_text()` again, even when the requested family
and size are already active.

In the inspected UI session, 12 final translation displays were paired with 12
identical `PySide RTLTextDisplay: Updated font to @黑体 20` messages. This is a
one-to-one confirmation of the redundant path. Streaming partial updates would
amplify the same work.

## Goals

- Render each PySide translation update exactly once.
- Apply the requested family and size before the HTML is constructed.
- Keep the tkinter-compatible `config(font=...)` entry point working.
- Avoid rerenders and log noise when a compatibility caller repeats the active
  font.
- Preserve text normalization, line breaks, direction, alignment, colors, and
  existing five-argument `set_rtl_text()` callers.

## Options considered

### Remove the `configure(font=...)` call

This removes the second render but can leave the widget on a stale font family.
`set_rtl_text()` currently receives only the size and reads the current family,
so this option is incomplete.

### Make only `config(font=...)` idempotent

This eliminates repeated work when the font is unchanged. On a real font
change, however, the display manager would still render once with the old font
and once with the new font.

### Apply the font inside `set_rtl_text()` before rendering

This is the selected design. Add an optional `font_family` parameter and an
internal helper that changes the widget font only when family or point size is
different. The display manager supplies the family in its single
`set_rtl_text()` call and no longer follows it with `configure(font=...)`.

The compatibility `config(font=...)` path reuses the same helper. An identical
font request becomes a no-op; a real font change rerenders the current text
once so legacy callers continue to see the update.

## Detailed behavior

`RTLTextDisplay` receives a private helper that:

1. Normalizes the requested family and point size against the current `QFont`.
2. Compares them with the widget's active family and point size.
3. Returns `False` without calling `setFont()` when nothing changed.
4. Otherwise copies and updates the current `QFont`, calls `setFont()` once,
   logs one real font change, and returns `True`.

`set_rtl_text()` accepts `font_family=None` after the existing parameters. It
invokes the helper before storing state or constructing HTML. Omitting the new
argument retains the current family while still applying the requested point
size, keeping existing callers source-compatible.

`DisplayManager` passes `font_family=font_type` in the existing PySide call and
removes the subsequent `configure(font=...)` call. The HTML therefore reads the
correct active family during its only render.

`config(font=...)` invokes the helper. It rerenders the stored text once only
when the helper reports a real change. Repeating the active font does not call
`setFont()`, `set_rtl_text()`, or emit an update log.

## Tests

- A display-manager unit test uses a fake PySide widget and verifies one
  `set_rtl_text()` call with the requested family, normalized line breaks, and
  no `configure()` call.
- Offscreen PySide tests verify that `set_rtl_text()` applies family/size before
  one `setHtml()` call.
- Offscreen PySide tests verify that repeating the active font through
  `config()` causes no HTML render and that a real size change causes exactly
  one rerender.
- Existing overlay, focused regression, full test, compile, and diff checks
  remain required.

## Expected result

Normal completed translations and streaming partials go from two HTML document
renders to one. Identical per-update font logs disappear. Explicit font changes
remain immediately visible and rerender the stored content only once.
