# Overlay Startup Readiness Design

## Goal

Make the application ready for immediate interaction after launch without
changing OCR or translation behavior. The saved source and target coordinates
must be usable immediately, the overlays must remain hidden until requested,
and pressing Start must work without requiring the user to reselect an area or
toggle a window first.

## Scope

This increment covers only application startup order, source/target overlay
readiness, the Start preflight, and source/target visibility toggles. It does
not change OCR models, OCR parameters, translation providers, request cadence,
or subtitle processing.

## Confirmed User Experience

- On launch, restore the last saved source and target coordinates.
- Keep both overlay windows hidden before the user takes an action.
- Pressing Start immediately after launch must create any missing overlay
  objects, validate the restored areas, and show the target overlay.
- The source overlay remains hidden while still supplying the capture geometry.
- The first Hide/Show Target Window click must not depend on PaddleOCR prewarm
  completing.
- A valid saved area must not trigger an instruction to select the area again.

## Considered Approaches

### 1. Create every overlay before showing the main window

This makes the first action fast, but Qt/PySide initialization delays the main
window becoming visible. It moves the wait instead of improving the perceived
startup experience.

### 2. Keep all overlay creation lazy

This preserves the fastest main-window appearance, but the first Start or
visibility-toggle action can still pay the full creation cost. It also leaves
more timing paths where restored coordinates and overlay objects disagree.

### 3. Staged readiness with an idempotent fallback

This is the selected approach. Restore coordinates synchronously, show the main
window, prepare hidden overlays during the first UI-idle phase, and start the
PaddleOCR background prewarm only after the UI-critical overlay preparation has
been queued or completed. Start and visibility toggles also call the same
idempotent readiness function, so an unusually fast click or a destroyed
overlay repairs itself instead of showing a misleading selection error.

## Architecture

### Saved-area restoration

`overlay_manager.py` will expose one focused operation that parses and validates
the four saved coordinates for each area and stores them on the application.
This operation must not create or show a window. It is safe to run during
application construction and safe to run again.

An area is valid only when it contains four integer coordinates and has positive
width and height. Existing configuration defaults remain the fallback for a
missing configuration key. Invalid saved data will be logged and will continue
to use the existing area-selection error path.

### Overlay readiness

`overlay_manager.py` will expose an idempotent readiness operation for source
and target overlays. It will:

1. Restore coordinates if the in-memory area is absent.
2. Create only the missing or destroyed overlay.
3. Leave newly prepared overlays hidden.
4. Return a clear success/failure result for preflight handling.

The existing source Tk overlay remains the authoritative live capture geometry.
The existing target PySide overlay remains the translation display. No capture
or rendering implementation is replaced.

### Startup ordering

`app_logic.py` will restore coordinate state before the main window becomes
interactive. After the UI is built, it will schedule hidden overlay preparation
ahead of local PaddleOCR prewarm. Heavy OCR initialization remains in its
background thread; only its scheduling order changes so it cannot win the first
UI-idle slot needed by window readiness.

This ordering keeps the main window quick to appear, avoids showing overlay
frames unexpectedly, and removes the dependency between the first UI action and
OCR prewarm progress.

### Start preflight

`app_lifecycle.py` will replace the current overlay-exists-only check with the
readiness operation. Start will then read geometry from the ready overlays and
run the existing coordinate validation. If readiness succeeds, startup proceeds
and shows the target overlay as it does today.

If configuration coordinates are genuinely invalid or overlay creation fails,
Start will retain a user-facing error, but the message will describe the real
failure. A missing object with valid saved coordinates is recoverable and must
not ask the user to reselect an area.

### Visibility toggles and persistence

Both visibility toggles will use the same readiness operation before toggling.
The target toggle will avoid redundant visual updates when creation already
applied the configured style. Existing full settings persistence remains
functionally unchanged unless profiling proves it is part of the interaction
delay; any persistence adjustment must stay limited to overlay geometry and
visibility and retain the current configuration format.

## Error Handling

- Missing or destroyed overlay plus valid coordinates: recreate silently.
- Invalid saved coordinates: log the invalid values and show the existing
  area-selection guidance.
- Overlay creation failure: return failure, restore the Start button state, and
  show a specific initialization error.
- Repeated readiness calls: reuse live overlays without recreation.
- User closes an overlay manually: the next Start or toggle recreates it at the
  last valid coordinates.

## Testing

`tests/test_overlay_startup.py` will be the primary regression surface.
Test-first coverage will include:

- hidden startup restores both saved areas without showing either overlay;
- startup readiness creates both hidden overlay objects;
- readiness is idempotent and does not recreate live overlays;
- target visibility toggle reuses a newly created overlay and toggles once;
- a destroyed overlay is recreated from saved coordinates;
- invalid coordinates produce a readiness failure;
- startup scheduling places overlay preparation before PaddleOCR prewarm;
- Start repairs missing overlays before running existing geometry validation.

Focused overlay and lifecycle tests will run first, followed by the complete
test discovery suite and Python syntax compilation. Manual verification will
launch the real application with both visibility flags false, press Start
without any prior area action, stop, relaunch, and verify that the target
appears at the last saved coordinates. The target visibility button will also
be exercised immediately after launch while PaddleOCR prewarm is still active.

## Files Expected to Change During Implementation

- `overlay_manager.py`: saved-area restoration and idempotent readiness.
- `app_logic.py`: coordinate restoration and UI-critical startup ordering.
- `app_lifecycle.py`: self-healing Start preflight.
- `tests/test_overlay_startup.py`: startup and readiness regressions.
- A lifecycle-focused existing test file only if the current test seams cannot
  exercise Start preflight without duplicating application setup.

No generated directory, dependency directory, untracked comparison project, or
translation/OCR implementation file is in scope.
