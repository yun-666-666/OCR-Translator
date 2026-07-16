# Translation display layout design

## Goal

Add two persistent controls for translated subtitles:

- A line-layout strategy that defaults to compact, one-line-first output.
- An independent horizontal-centering option that never changes vertical alignment.

The change must preserve the existing choice to represent the source subtitle's
line structure and must prevent model-supplied line breaks from overriding the
compact strategy.

## Evidence and root cause

The latest Custom AI translation log contains model results with literal
`<br>` markers.  The active `keep_linebreaks=True` setting instructs both OCR
and translation to preserve line breaks.  The PySide output widget converts
newlines to HTML `<br>` markers, so these model-requested breaks become forced
display lines before Qt has an opportunity to wrap text naturally.

The same runtime session shows that local capture and PaddleOCR are not the
current latency bottleneck.  Upstream provider timeouts, unavailable profiles,
and sequential failover dominate the long tail.  This layout work is therefore
kept separate from transport optimization.

## Options considered

### 1. Display strategy with prompt and renderer enforcement (selected)

Add a two-value setting:

- `compact`: ask the provider for one concise line, normalize provider line
  breaks to spaces at the display boundary, then let Qt wrap only when the
  overlay width is exhausted.
- `preserve_source_lines`: retain source/OCR line boundaries and request the
  corresponding translated layout.

This handles imperfect model compliance and supports both requested behaviors.

### 2. Prompt-only change

This is small but cannot prevent a provider from returning `<br>` or a real
newline.  It does not meet the compact-display guarantee.

### 3. Renderer-only change

This could flatten unwanted breaks but cannot safely preserve the source-line
layout in the alternate mode.  It is insufficient on its own.

## Design

### Configuration and migration

- Add `translation_line_layout` with a default of `compact`.
- Add `translation_horizontal_centered` with a default of `False` so existing
  users retain the current left/right behavior until they opt in.
- Continue reading the legacy `keep_linebreaks` key.  If the new layout key is
  absent, migrate `True` to `preserve_source_lines` and `False` to `compact`.
- Persist the new values while keeping the legacy boolean synchronized for
  existing OCR and translation call sites during the compatibility period.

### Settings and request flow

- Replace the ambiguous line-break checkbox with a labelled two-choice control:
  `Prefer one line` and `Match source subtitle lines`.
- Add a `Center translated subtitle horizontally` checkbox in the output
  display settings.
- The selected line strategy controls whether OCR and Custom AI translation
  requests preserve source breaks.  In compact mode the translation prompt
  explicitly requests no line breaks.

### Rendering

- Extend the PySide text-display entry point with the horizontal-centering
  value and preserve backward-compatible callers with a default.
- In compact mode, normalize CRLF, LF, and `<br>` variants to a single space
  before HTML is built.  Qt's existing widget-width wrapping is then the only
  source of a second line.
- In preserve mode, retain normalized line boundaries and render them as HTML
  line breaks as today.
- Apply `text-align` and `QTextBlockFormat` as horizontal center when enabled;
  otherwise retain direction-aware left/right alignment.  Do not alter layout
  vertical alignment, margins, geometry, or font metrics.

### Tests

- Configuration tests cover new defaults and migration from both legacy
  boolean values.
- Request tests cover compact and preserve prompts.
- Renderer tests cover compact `<br>`/newline flattening, preserve-mode line
  rendering, horizontal center alignment, and unchanged direction-aware
  behavior when centering is disabled.
- Run focused tests first, then the full offline suite and Python compilation
  for every touched module.

## Scope boundaries

No changes are planned to provider routing, request timeouts, overlay geometry,
vertical text placement, font sizing, or unrelated UI modernization.
