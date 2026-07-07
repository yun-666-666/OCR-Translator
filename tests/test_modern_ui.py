"""Tests for modern_ui styling helpers.

Values reflect the Phase 1.5 Amber Signal palette introduced in the
Phase 2 UI redesign commit.  Old teal-palette assertions have been
removed; this file is the canonical source of truth for the new tokens.
"""
import unittest
import tkinter as tk

import modern_ui


class ModernUiThemeTests(unittest.TestCase):

    # ------------------------------------------------------------------
    # Palette token tests
    # ------------------------------------------------------------------

    def test_white_clean_palette_exports_tool_console_colors(self):
        palette = modern_ui.WHITE_CLEAN_PALETTE

        # Amber Signal tokens (Phase 1.5)
        self.assertEqual(palette["window"],           "#f0f2f4")
        self.assertEqual(palette["surface"],          "#ffffff")
        self.assertEqual(palette["text"],             "#1c2b3a")
        self.assertEqual(palette["primary"],          "#d4700c")   # amber signal
        self.assertEqual(palette["primary_container"],"#fdf2e6")
        self.assertEqual(palette["surface_variant"],  "#e4e9ee")
        self.assertEqual(palette["success"],          "#15803d")
        self.assertEqual(palette["warning"],          "#c2410c")
        self.assertEqual(palette["danger"],           "#c63535")
        # "glass" was never part of the palette
        self.assertNotIn("glass", palette)

    def test_white_clean_palette_has_no_legacy_teal(self):
        """Confirm the old teal primary token is gone."""
        self.assertNotEqual(modern_ui.WHITE_CLEAN_PALETTE["primary"], "#0f766e")

    # ------------------------------------------------------------------
    # API surface tests
    # ------------------------------------------------------------------

    def test_white_clean_theme_api_is_available(self):
        self.assertTrue(callable(modern_ui.apply_white_clean_theme))

    def test_pulse_helpers_are_exported(self):
        self.assertTrue(callable(modern_ui.start_pipeline_pulse))
        self.assertTrue(callable(modern_ui.cancel_pipeline_pulse))

    # ------------------------------------------------------------------
    # Live Tk tests (skipped when no display is available)
    # ------------------------------------------------------------------

    def test_white_clean_theme_applies_to_tk_root(self):
        try:
            root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(f"Tk display unavailable: {exc}")
        try:
            palette = modern_ui.apply_white_clean_theme(root)
            self.assertEqual(root.cget("bg"), palette["window"])
        finally:
            root.destroy()

    def test_white_clean_theme_registers_profile_button_styles(self):
        try:
            root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(f"Tk display unavailable: {exc}")
        try:
            modern_ui.apply_white_clean_theme(root)
            style = modern_ui.ttk.Style(root)

            # Secondary button uses surface_variant as background
            self.assertEqual(
                style.lookup("Secondary.TButton", "background"),
                modern_ui.WHITE_CLEAN_PALETTE["surface_variant"],
            )
            self.assertEqual(style.lookup("Secondary.TButton", "relief"), "solid")

            # Profile action buttons
            self.assertEqual(
                style.lookup("ProfileAdd.TButton", "background"),
                modern_ui.WHITE_CLEAN_PALETTE["success"],
            )
            # ProfileSave uses text (dark ink) as base in Phase 2
            self.assertEqual(
                style.lookup("ProfileSave.TButton", "background"),
                modern_ui.WHITE_CLEAN_PALETTE["text"],
            )
            self.assertEqual(
                style.lookup("ProfileDelete.TButton", "background"),
                modern_ui.WHITE_CLEAN_PALETTE["danger"],
            )
            self.assertEqual(
                style.lookup("ProfileTest.TButton", "background"),
                modern_ui.WHITE_CLEAN_PALETTE["warning"],
            )
            self.assertEqual(style.lookup("TNotebook", "tabmargins"), "0 4 0 0")
        finally:
            root.destroy()

    def test_start_running_button_style_is_registered(self):
        """StartRunning.TButton must be registered (amber, used when running)."""
        try:
            root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(f"Tk display unavailable: {exc}")
        try:
            modern_ui.apply_white_clean_theme(root)
            style = modern_ui.ttk.Style(root)
            bg = style.lookup("StartRunning.TButton", "background")
            self.assertEqual(bg, modern_ui.WHITE_CLEAN_PALETTE["primary"])
        finally:
            root.destroy()

    def test_theme_registers_status_track_and_section_styles(self):
        try:
            root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(f"Tk display unavailable: {exc}")
        try:
            modern_ui.apply_white_clean_theme(root)
            style = modern_ui.ttk.Style(root)

            # Status track background is now surface_variant (neutral strip)
            self.assertEqual(
                style.lookup("StatusTrack.TFrame", "background"),
                modern_ui.WHITE_CLEAN_PALETTE["surface_variant"],
            )
            # Done label foreground is amber primary
            self.assertEqual(
                style.lookup("StatusTrackDone.TLabel", "foreground"),
                modern_ui.WHITE_CLEAN_PALETTE["primary"],
            )
            # Section labelframe border
            self.assertEqual(
                style.lookup("Section.TLabelframe", "bordercolor"),
                "#ccd6e0",
            )
            # Settings group label foreground is amber primary
            self.assertEqual(
                style.lookup("SettingsGroup.TLabel", "foreground"),
                modern_ui.WHITE_CLEAN_PALETTE["primary"],
            )
        finally:
            root.destroy()


if __name__ == "__main__":
    unittest.main()
