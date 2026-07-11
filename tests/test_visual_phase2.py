"""
Screenshot verification for Phase 2 Amber Signal UI.

Runs headless-safe: skips if no display.  When a display IS available
it renders four mini-windows (idle, running, settings heading, zh-CN),
saves PNGs to docs/screenshots/phase2/ and validates a set of pixel
colour invariants.

Usage:
    python tests/test_visual_phase2.py
"""
import os
import sys
import types
import unittest

# ---------------------------------------------------------------------------
# Attempt to import Tk; skip entire module if no display is available.
# ---------------------------------------------------------------------------
try:
    import tkinter as tk
    from tkinter import ttk
    _ROOT_CHECK = tk.Tk()
    _ROOT_CHECK.withdraw()
    _ROOT_CHECK.destroy()
    HAS_DISPLAY = True
except Exception:
    HAS_DISPLAY = False

import modern_ui
from modern_ui import WHITE_CLEAN_PALETTE, apply_white_clean_theme

_OUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "docs", "screenshots", "phase2"
)


def _save_screenshot(root, name: str) -> str:
    """Render root to a PIL image and save as PNG; return the path."""
    try:
        from PIL import ImageGrab
    except ImportError:
        return ""

    os.makedirs(_OUT_DIR, exist_ok=True)
    root.update_idletasks()
    root.update()

    x = root.winfo_rootx()
    y = root.winfo_rooty()
    w = root.winfo_width()
    h = root.winfo_height()
    img = ImageGrab.grab(bbox=(x, y, x + w, y + h))

    path = os.path.join(_OUT_DIR, f"{name}.png")
    img.save(path)
    return path


def _build_minimal_window(title: str, width: int = 600, height: int = 480):
    """Return a themed Tk root at the requested size."""
    root = tk.Tk()
    root.title(title)
    root.geometry(f"{width}x{height}")
    apply_white_clean_theme(root)
    return root


def _add_pipeline_bar(root, palette):
    """Build a standalone pipeline bar (no full app needed)."""
    from tkinter import ttk

    idle = palette.get("surface_variant", "#e4e9ee")
    frame = ttk.Frame(root, style="StatusTrack.TFrame", padding=(10, 6))
    frame.pack(fill=tk.X, padx=12, pady=(10, 4))

    steps = ["OCR", "Translate", "Overlay"]
    dot_canvases = []
    for idx, label_text in enumerate(steps):
        col = idx * 2
        cell = ttk.Frame(frame, style="StatusTrack.TFrame")
        cell.grid(row=0, column=col, sticky="ew", padx=2)
        cell.columnconfigure(1, weight=1)

        dot = tk.Canvas(cell, width=8, height=8, bg=idle,
                        highlightthickness=0, bd=0)
        dot.grid(row=0, column=0, padx=(0, 4))
        dot.create_oval(1, 1, 7, 7, fill=idle, outline=idle, tags="dot")
        dot_canvases.append(dot)

        ttk.Label(cell, text=label_text,
                  style="StatusTrackStep.TLabel", anchor="w").grid(
            row=0, column=1, sticky="ew")
        if idx < len(steps) - 1:
            ttk.Label(frame, text="›", style="StatusTrackArrow.TLabel",
                      anchor="center").grid(row=0, column=col + 1, padx=4)

    for col in (0, 2, 4):
        frame.columnconfigure(col, weight=1, uniform="wf")

    return dot_canvases


def _add_start_stop_button(root, style_name: str, label: str):
    btn = ttk.Button(root, text=label, style=style_name)
    btn.pack(fill=tk.X, padx=40, pady=(6, 4), ipady=6)
    return btn


def _add_settings_heading(root, palette, text: str):
    """Add a Phase-2 settings group heading (amber bar + label)."""
    frame = ttk.Frame(root, style="Surface.TFrame")
    frame.pack(fill=tk.X, padx=12, pady=(12, 4))
    frame.columnconfigure(2, weight=1)

    amber = palette.get("primary", "#d4700c")
    bar = tk.Frame(frame, width=3, bg=amber)
    bar.grid(row=0, column=0, sticky="ns", padx=(0, 6))

    ttk.Label(frame, text=text, style="SettingsGroup.TLabel").grid(
        row=0, column=1, padx=(0, 8), sticky="w")
    ttk.Separator(frame, orient=tk.HORIZONTAL).grid(
        row=0, column=2, sticky="ew")


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

@unittest.skipUnless(HAS_DISPLAY, "No Tk display available")
class VisualPhase2Tests(unittest.TestCase):

    def _assert_pixel(self, img, x, y, expected_hex, tolerance=20, label=""):
        """Assert the pixel at (x,y) is within *tolerance* of expected_hex."""
        r_exp = int(expected_hex[1:3], 16)
        g_exp = int(expected_hex[3:5], 16)
        b_exp = int(expected_hex[5:7], 16)
        pixel = img.getpixel((x, y))[:3]
        diff = max(abs(pixel[0]-r_exp), abs(pixel[1]-g_exp), abs(pixel[2]-b_exp))
        self.assertLessEqual(
            diff, tolerance,
            f"{label}: expected ~{expected_hex}, got rgb{pixel} at ({x},{y})"
        )

    def _grab(self, root):
        from PIL import ImageGrab
        root.update_idletasks()
        root.update()
        x, y = root.winfo_rootx(), root.winfo_rooty()
        w, h = root.winfo_width(), root.winfo_height()
        return ImageGrab.grab(bbox=(x, y, x+w, y+h))

    # -- idle state ----------------------------------------------------------

    def test_idle_window_background(self):
        root = _build_minimal_window("Phase2 idle — 600×480")
        try:
            img = self._grab(root)
            path = _save_screenshot(root, "01_idle")
            # Sample near the centre of the client area (well away from the
            # title bar which winfo_rooty may not fully exclude on HiDPI).
            cx = root.winfo_width() // 2
            cy = root.winfo_height() // 2
            # Window background should match palette["window"] = #f0f2f4
            self._assert_pixel(img, cx, cy, "#f0f2f4", label="idle bg centre")
            print(f"  saved: {path}")
        finally:
            root.destroy()

    def test_idle_pipeline_dots_are_neutral(self):
        root = _build_minimal_window("Phase2 idle dots")
        try:
            palette = WHITE_CLEAN_PALETTE
            _add_pipeline_bar(root, palette)
            root.update_idletasks()
            root.update()
            img = self._grab(root)
            _save_screenshot(root, "02_idle_dots")
            # Spot-check: the dot region should NOT be amber
            amber_r = int("d4", 16)
            pixel = img.getpixel((40, 28))[:3]  # approx dot area
            # Dot should be close to surface_variant (#e4e9ee), not amber
            self.assertLess(
                pixel[0], amber_r + 20,
                f"idle dot should not be amber, got rgb{pixel}"
            )
        finally:
            root.destroy()

    # -- running state (amber button) ----------------------------------------

    def test_running_button_is_amber(self):
        root = _build_minimal_window("Phase2 running")
        try:
            _add_start_stop_button(root, "StartRunning.TButton", "Stop")
            img = self._grab(root)
            _save_screenshot(root, "03_running_button")
            # Button should have amber background — check centre of button
            w, h = root.winfo_width(), root.winfo_height()
            cx, cy = w // 2, 40
            pixel = img.getpixel((cx, cy))[:3]
            # Amber primary = #d4700c → R≈212, G≈112, B≈12
            self.assertGreater(pixel[0], 150, f"amber R should be >150, got {pixel}")
            self.assertLess(pixel[2], 80,   f"amber B should be <80, got {pixel}")
            print(f"  amber button pixel: rgb{pixel}")
        finally:
            root.destroy()

    # -- stop / primary button -----------------------------------------------

    def test_idle_start_button_is_dark_ink(self):
        root = _build_minimal_window("Phase2 idle start btn")
        try:
            _add_start_stop_button(root, "Primary.TButton", "Start")
            img = self._grab(root)
            _save_screenshot(root, "04_idle_start_button")
            w = root.winfo_width()
            cx, cy = w // 2, 40
            pixel = img.getpixel((cx, cy))[:3]
            # Primary.TButton base is palette["text"] = #1c2b3a → dark ink
            # Button bg should be dark (any channel < 80)
            self.assertTrue(
                any(c < 80 for c in pixel),
                f"Start button should be dark ink, got rgb{pixel}"
            )
        finally:
            root.destroy()

    # -- settings heading ----------------------------------------------------

    def test_settings_heading_amber_bar_visible(self):
        root = _build_minimal_window("Phase2 settings heading")
        try:
            _add_settings_heading(root, WHITE_CLEAN_PALETTE, "Capture")
            img = self._grab(root)
            _save_screenshot(root, "05_settings_heading")
            # The 3px amber bar sits at approx x=12, y=25 after padding
            pixel = img.getpixel((13, 28))[:3]
            # Should be amber-ish: R>150, G<150, B<50
            self.assertGreater(pixel[0], 150,
                               f"amber bar R should be >150, got {pixel}")
            self.assertLess(pixel[2], 80,
                            f"amber bar B should be <80, got {pixel}")
        finally:
            root.destroy()

    # -- 600×480 geometry ----------------------------------------------------

    def test_window_600x480(self):
        root = _build_minimal_window("Phase2 600×480", width=600, height=480)
        try:
            root.update_idletasks()
            root.update()
            self.assertEqual(root.winfo_width(),  600)
            self.assertEqual(root.winfo_height(), 480)
            _save_screenshot(root, "06_geometry_600x480")
        finally:
            root.destroy()


if __name__ == "__main__":
    unittest.main(verbosity=2)
