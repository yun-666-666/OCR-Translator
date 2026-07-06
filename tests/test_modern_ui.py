import unittest
import tkinter as tk

import modern_ui


class ModernUiThemeTests(unittest.TestCase):
    def test_white_clean_palette_exports_tool_console_colors(self):
        palette = modern_ui.WHITE_CLEAN_PALETTE

        self.assertEqual(palette["window"], "#f6f8fb")
        self.assertEqual(palette["surface"], "#ffffff")
        self.assertEqual(palette["text"], "#18212f")
        self.assertEqual(palette["primary"], "#0f766e")
        self.assertEqual(palette["success"], "#15803d")
        self.assertEqual(palette["warning"], "#c2410c")
        self.assertNotIn("glass", palette)

    def test_white_clean_theme_api_is_available(self):
        self.assertTrue(callable(modern_ui.apply_white_clean_theme))

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
            self.assertEqual(style.lookup("Secondary.TButton", "background"), "#edf2f7")
            self.assertEqual(style.lookup("Secondary.TButton", "relief"), "solid")
            self.assertEqual(style.lookup("ProfileAdd.TButton", "background"), "#15803d")
            self.assertEqual(style.lookup("ProfileSave.TButton", "background"), "#0f766e")
            self.assertEqual(style.lookup("ProfileDelete.TButton", "background"), "#b42318")
            self.assertEqual(style.lookup("ProfileTest.TButton", "background"), "#c2410c")
            self.assertEqual(style.lookup("TNotebook", "tabmargins"), "0 4 0 0")
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
            self.assertEqual(style.lookup("StatusTrack.TFrame", "background"), "#e7f5f2")
            self.assertEqual(style.lookup("StatusTrackDone.TLabel", "foreground"), "#0f766e")
            self.assertEqual(style.lookup("Section.TLabelframe", "bordercolor"), "#ccd6e0")
            self.assertEqual(style.lookup("SettingsGroup.TLabel", "foreground"), "#0f766e")
        finally:
            root.destroy()


if __name__ == "__main__":
    unittest.main()
