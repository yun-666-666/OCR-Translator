import unittest
import tkinter as tk

import modern_ui


class ModernUiThemeTests(unittest.TestCase):
    def test_white_clean_palette_exports_light_neutral_colors(self):
        palette = modern_ui.WHITE_CLEAN_PALETTE

        self.assertEqual(palette["window"], "#f7f8fb")
        self.assertEqual(palette["surface"], "#ffffff")
        self.assertEqual(palette["text"], "#1f2937")
        self.assertEqual(palette["primary"], "#2563eb")
        self.assertEqual(palette["success"], "#16a34a")
        self.assertEqual(palette["warning"], "#d97706")
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
            self.assertEqual(style.lookup("ProfileAdd.TButton", "background"), "#16a34a")
            self.assertEqual(style.lookup("ProfileSave.TButton", "background"), "#2563eb")
            self.assertEqual(style.lookup("ProfileDelete.TButton", "background"), "#dc2626")
            self.assertEqual(style.lookup("ProfileTest.TButton", "background"), "#d97706")
            self.assertEqual(style.lookup("ProfileRefresh.TButton", "background"), "#2563eb")
            self.assertEqual(style.lookup("ProfileRefresh.TButton", "foreground"), "#ffffff")
            self.assertEqual(style.lookup("TNotebook", "tabmargins"), "0 4 0 0")
        finally:
            root.destroy()

    def test_square_icon_button_centers_glyph_and_supports_state(self):
        try:
            root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(f"Tk display unavailable: {exc}")
        try:
            clicked = {"count": 0}
            button = modern_ui.SquareIconButton(
                root,
                text="⟳",
                command=lambda: clicked.__setitem__("count", clicked["count"] + 1),
                size=28,
                text_offset_y=-2,
            )
            button.pack()
            root.update_idletasks()

            self.assertEqual(int(button.cget("width")), 28)
            self.assertEqual(int(button.cget("height")), 28)
            x, y = button.coords(button._icon_id)
            self.assertAlmostEqual(x, 14.0, places=1)
            self.assertAlmostEqual(y, 12.0, places=1)

            button.configure(state="disabled")
            button.event_generate("<ButtonRelease-1>")
            self.assertEqual(clicked["count"], 0)

            button.configure(state="normal")
            button._command()
            self.assertEqual(clicked["count"], 1)
        finally:
            root.destroy()


if __name__ == "__main__":
    unittest.main()
