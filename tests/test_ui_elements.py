import tkinter as tk
from tkinter import ttk
import types
import unittest

import modern_ui
from ui_elements import _mousewheel_scroll_units, create_scrollable_tab


class MouseWheelUnitTests(unittest.TestCase):
    def test_normalizes_windows_and_x11_wheel_events(self):
        self.assertEqual(
            _mousewheel_scroll_units(types.SimpleNamespace(delta=-120, num=None)),
            1,
        )
        self.assertEqual(
            _mousewheel_scroll_units(types.SimpleNamespace(delta=120, num=None)),
            -1,
        )
        self.assertEqual(
            _mousewheel_scroll_units(types.SimpleNamespace(delta=0, num=4)),
            -1,
        )
        self.assertEqual(
            _mousewheel_scroll_units(types.SimpleNamespace(delta=0, num=5)),
            1,
        )


class SpinboxStyleTests(unittest.TestCase):
    def test_spinbox_arrows_and_padding_are_touch_friendly(self):
        class RecordingStyle:
            def __init__(self):
                self.options = {}

            def configure(self, name, **options):
                self.asserted_name = name
                self.options = options

        style = RecordingStyle()
        configure_spinbox_style = getattr(
            modern_ui,
            "_configure_spinbox_style",
            None,
        )
        self.assertIsNotNone(configure_spinbox_style)

        configure_spinbox_style(
            style,
            {
                "surface": "#ffffff",
                "text": "#111111",
                "outline": "#cccccc",
                "text_muted": "#666666",
            },
        )

        self.assertEqual(style.asserted_name, "TSpinbox")
        self.assertEqual(style.options["arrowsize"], 16)
        self.assertEqual(style.options["padding"], (5, 3))


class ScrollableTabInputGuardTests(unittest.TestCase):
    def setUp(self):
        try:
            self.root = tk.Tk()
        except tk.TclError as error:
            self.skipTest(f"Tk display unavailable: {error}")
        # Match the application's practical settings-window width so the
        # conditionally packed scrollbar has room beside the canvas.
        self.root.geometry("800x180")

    def tearDown(self):
        root = getattr(self, "root", None)
        if root is not None:
            root.destroy()

    def _build_scrollable_settings(self):
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True)
        content = create_scrollable_tab(
            notebook,
            "Settings",
            protect_wheel_inputs=True,
        )
        for index in range(40):
            ttk.Label(content, text=f"row {index}").pack()
        self.root.update()
        return content, content.master

    def test_hovered_combobox_keeps_value_while_page_scrolls(self):
        content, canvas = self._build_scrollable_settings()
        combo = ttk.Combobox(
            content,
            values=["one", "two"],
            state="readonly",
        )
        combo.set("one")
        combo.pack(before=content.winfo_children()[0])
        self.root.update()

        before = canvas.yview()[0]
        combo.event_generate("<MouseWheel>", delta=-120)
        self.root.update()

        self.assertEqual(combo.get(), "one")
        self.assertGreater(canvas.yview()[0], before)

    def test_hovered_spinbox_keeps_value_while_page_scrolls(self):
        content, canvas = self._build_scrollable_settings()
        spinbox = ttk.Spinbox(content, from_=1, to=10)
        spinbox.set("5")
        spinbox.pack(before=content.winfo_children()[0])
        self.root.update()

        before = canvas.yview()[0]
        spinbox.event_generate("<MouseWheel>", delta=-120)
        self.root.update()

        self.assertEqual(spinbox.get(), "5")
        self.assertGreater(canvas.yview()[0], before)


if __name__ == "__main__":
    unittest.main()
