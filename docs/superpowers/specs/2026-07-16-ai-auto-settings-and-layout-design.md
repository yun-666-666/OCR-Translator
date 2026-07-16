# AI Auto Settings and Compact Settings Layout Design

## Goal

Replace the separate Custom AI response and OCR image controls with one
user-facing optimization policy that automatically chooses a reliable,
cost-aware request mode and image payload. Remove the screenshot-backend
setting in favor of MSS, tune OCR defaults from the current application logs,
and compact the requested settings rows without changing unrelated UI.

## User-facing behavior

The settings UI exposes one `AI optimization` selector:

- `Auto (recommended)`: balance latency, reliability, OCR quality, and token
  use from real route history and the current screenshot.
- `Speed and fewer tokens`: avoid duplicate requests, use smaller image
  payloads, and prefer low image detail when the text remains readable.
- `Quality first`: use high-detail, high-quality image payloads and conservative
  non-streaming responses.

The old response-mode, OCR image format, image mode, image quality, and image
detail widgets disappear. Their legacy config keys are migrated into the
nearest policy and are no longer saved as independent user settings.

The screenshot-backend selector also disappears. All normal capture and OCR
preview paths use MSS. The PyAutoGUI screenshot implementation and packaging
dependency are removed because current logs show materially better aggregate
and tail performance from MSS. Capture errors remain visible in diagnostics
instead of silently changing to a slower backend.

## Automatic response policy

`Auto` resolves through the existing route-scoped latency advisor. It starts
with safe, non-streaming requests. Route observations remain keyed by the
canonical endpoint and model, so a slow relay cannot contaminate a fast direct
route.

The automatic advisor may choose a bounded endpoint race only when:

- enough successful latency samples exist;
- route P90 is materially high;
- at least two compatible, healthy translation profiles are available; and
- the race cooldown has expired.

Automatic mode does not switch to streaming merely because total latency is
high. The current application logs show streaming was slower and less stable,
while official provider documentation only promises improved perceived
performance, not lower completed-response latency. Streaming remains supported
internally for compatibility with old snapshots and direct tests, but the new
single setting does not select it.

`Speed and fewer tokens` and `Quality first` both use safe response delivery.
The speed policy avoids races because duplicate requests can increase token
cost. The quality policy avoids partial subtitle churn.

Hysteresis stays in the advisor through minimum samples, mode hold time, race
cooldown, recent-error fallback, and provider cooldown checks.

## Automatic OCR image policy

A pure resolver returns an immutable payload decision containing:

- image format;
- image preprocessing mode;
- encoding quality;
- API image detail;
- a short reason suitable for sanitized diagnostics.

The resolver uses the optimization policy, active OCR profile, screenshot
dimensions, and endpoint capability memory.

### Auto baseline

- Color WebP, balanced mode, quality 85, detail `auto`.
- Use `high` detail for thin subtitle regions, large images, or images whose
  estimated text height is small relative to the frame.
- Use `low` detail with quality 75 for small, clear subtitle images on slow
  routes.
- Keep `auto` detail for the normal middle case so provider defaults can make
  their own token-versus-detail tradeoff.

### Provider compatibility

- OpenAI-compatible and Gemini-style routes may use WebP.
- Direct xAI endpoints use JPEG because the current official xAI image
  understanding documentation lists JPEG and PNG inputs.
- A route-scoped capability cache remembers explicit unsupported-image-format
  failures and resolves future payloads to JPEG without repeating the rejected
  format.
- PNG is reserved for quality-first requests where lossless text edges are more
  valuable than upload size.

### Explicit policy presets

- `Speed and fewer tokens`: grayscale WebP quality 72 with low detail; direct
  xAI uses JPEG quality 78.
- `Quality first`: PNG with high detail for normal compatible routes; direct
  xAI uses JPEG quality 95 and high detail.

The resolved payload contract is included in the API OCR cache key. Encoding
logs record the policy, resolved contract, byte count, and reason without
logging image data.

## OCR defaults and migration

- New default `stability_threshold`: `0`.
- New default `paddleocr_min_score`: `0.45`.

PaddleOCR clear candidates already use the newer bounded stability gate and do
not need the legacy repeated-frame counter. A zero legacy threshold avoids
adding another scan delay.

The current logs show filtered low-confidence results clustered below 0.45,
with a filtered-result P90 near 0.414. A 0.45 default removes that noise while
remaining user-editable.

Existing values are preserved unless they match the former defaults:

- legacy stability `2` migrates to `0`;
- legacy PaddleOCR score `0.35` migrates to `0.45`.

## Prompt

The built-in and shipped English prompt become:

`Translate naturally and concisely. Preserve meaning, tone, names, and terminology; use context only when needed.`

The request builder already provides source/target language and an output-only
constraint, so the custom prompt does not repeat those instructions.

## Compact layout

- `Bold translated subtitle text` and `Center translated subtitle
  horizontally` share one row inside a small frame.
- Source-area, target-area, and target-text colors share one row.
- Each color entry contains only its short label and clickable color swatch.
- Clicking a swatch opens the existing color chooser. Separate `Choose color`
  buttons are removed.

The color handler continues to update the live Tk and PySide overlays
immediately.

## Configuration and compatibility

The canonical new key is `ai_optimization_mode` with values `auto`, `speed`,
and `quality`.

On first load without that key, old settings map as follows:

- high detail, lossless/PNG, or quality at least 90 -> `quality`;
- low detail, grayscale mode, or quality at most 75 -> `speed`;
- all other combinations -> `auto`.

Old response/image keys are accepted during migration, then removed from the
saved config. Internal compatibility helpers may still accept old test doubles
or request snapshots, but new runtime reads use the unified policy.

The legacy `capture_backend` key is removed during config migration. Capture
cache signatures continue to identify the actual MSS backend.

## Error handling

- Image encoding failure retries once with a conservative provider-compatible
  format.
- Explicit upstream unsupported-format errors update only route-scoped
  capability memory.
- Automatic response selection falls back to safe mode on insufficient
  samples, recent errors, cooldown, or missing compatible race profiles.
- MSS capture failures are logged with sanitized exception text and do not
  trigger a hidden backend change.

## Verification

Tests must cover:

- unified config defaults and migration from each legacy preset;
- response-policy mapping and adaptive safe/race behavior;
- no automatic streaming selection;
- provider- and image-size-aware payload resolution;
- OCR cache identity using the resolved payload;
- MSS-only capture and OCR preview;
- removal of PyAutoGUI packaging requirements;
- compact bold/center and clickable color-swatch layout;
- prompt equality and length;
- OCR default migration;
- existing Custom AI, OCR, overlay, and full test suites.

Final checks include Python compilation, `git diff --check`, a real Tk widget
construction probe when available, and the repository's full unittest
discovery command.
