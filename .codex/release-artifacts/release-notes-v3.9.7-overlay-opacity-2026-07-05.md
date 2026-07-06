# OCR Translator 3.9.7 - Overlay Opacity Stability

This patch release stabilizes overlay rendering while scrolling web pages under
the OCR source and translation target windows.

## Fixed

- Fixed source and target overlay colors briefly becoming lighter/flickering
  while scrolling an underlying web page.
- Stabilized overlay color refresh so the Tk source overlay no longer uses
  temporary alpha changes to force redraws.
- Kept the Custom AI/xAI compatibility fixes unchanged.

## Implementation Notes

- The Tk source overlay now updates widget backgrounds and redraws without
  changing `-alpha`.
- The PySide target overlay caches unchanged background/top-bar styles and no
  longer reapplies background styling every time the overlay is shown.
- The default target background opacity is now `0.4`, matching the existing
  example config and reducing visible bleed-through from scrolling page content
  while preserving user-configurable transparency.

## Verification

- `python -m unittest tests.test_overlay_startup`: 7 passed.
- `python -m unittest tests.test_latency_optimization`: 103 passed.
- `python -m unittest tests.test_custom_ai`: 213 passed.
- `python -B -m py_compile ui_elements.py pyside_overlay.py overlay_manager.py config_manager.py`: passed.
- `git diff --check`: passed for changed runtime, test, and changelog files.

## Asset

- `OCR-Translator-overlay-opacity-stability-v3.9.7-2026-07-05.zip`
- SHA256: `CFE18E01AFED6AB70563A373967BA04A09D4E06CF521C51C3C71099E1BE49F4D`

## Target

- Branch: `codex/fix-overlay-scroll-flash`
- Commit: `fb195dc06402c66b6c212c378a0e8a3e8da9c34b`
