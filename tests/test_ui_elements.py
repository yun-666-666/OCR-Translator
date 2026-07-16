import tkinter as tk
from tkinter import ttk
import types
import unittest
from pathlib import Path

import modern_ui
import gui_settings_builder
from config_manager import DEFAULT_CONFIG_SETTINGS
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


class SettingsLayoutSourceTests(unittest.TestCase):
    def test_translated_subtitle_outline_defaults_and_state_are_defined(self):
        app_logic_source = Path("app_logic.py").read_text(encoding="utf-8-sig")

        self.assertEqual(
            DEFAULT_CONFIG_SETTINGS["target_text_colour"],
            "#FFD54F",
        )
        self.assertEqual(
            DEFAULT_CONFIG_SETTINGS["target_text_outline_colour"],
            "#000000",
        )
        self.assertEqual(
            DEFAULT_CONFIG_SETTINGS["target_text_outline_width"],
            "2",
        )
        self.assertIn("self.target_text_outline_colour_var", app_logic_source)
        self.assertIn("self.target_text_outline_width_var", app_logic_source)
        self.assertIn(
            'self.target_text_outline_colour_var.trace_add("write", '
            "self.settings_changed_callback)",
            app_logic_source,
        )
        self.assertIn(
            'self.target_text_outline_width_var.trace_add("write", '
            "self.settings_changed_callback)",
            app_logic_source,
        )

    def test_colors_use_two_by_two_clickable_swatches_without_buttons(self):
        source = Path("gui_settings_builder.py").read_text(encoding="utf-8-sig")

        self.assertIn("app.colors_row_frame", source)
        self.assertIn("color_display.bind(", source)
        self.assertIn('"<Button-1>"', source)
        self.assertIn("app.target_text_outline_colour_var", source)
        self.assertIn("'target_outline'", source)
        self.assertIn("color_item_frame.grid(", source)
        self.assertNotIn("color_item_frame.pack(", source)
        self.assertNotIn('get_label("choose_color_btn")', source)

    def test_color_controls_map_to_two_columns(self):
        position = getattr(
            gui_settings_builder,
            "_settings_color_grid_position",
            None,
        )

        self.assertIsNotNone(position)
        self.assertEqual((0, 0), position(0))
        self.assertEqual((0, 1), position(1))
        self.assertEqual((1, 0), position(2))
        self.assertEqual((1, 1), position(3))

    def test_outline_width_control_and_persistence_are_wired(self):
        builder_source = Path("gui_settings_builder.py").read_text(
            encoding="utf-8-sig"
        )
        configuration_source = Path("app_configuration.py").read_text(
            encoding="utf-8-sig"
        )
        handler_source = Path("handlers/ui_interaction_handler.py").read_text(
            encoding="utf-8-sig"
        )

        self.assertIn(
            "app.target_text_outline_width_spinbox = ttk.Spinbox(",
            builder_source,
        )
        self.assertIn("from_=0, to=6", builder_source)
        self.assertIn("def update_target_text_outline(self):", configuration_source)
        self.assertIn("def update_target_text_outline(self):", handler_source)
        self.assertIn("cfg['target_text_outline_colour']", handler_source)
        self.assertIn("cfg['target_text_outline_width']", handler_source)

    def test_bold_and_horizontal_center_controls_share_one_row(self):
        source = Path("gui_settings_builder.py").read_text(encoding="utf-8-sig")

        self.assertIn("app.translation_text_style_frame", source)
        self.assertIn(
            "app.target_font_bold_checkbox = ttk.Checkbutton(\n"
            "        app.translation_text_style_frame,",
            source,
        )
        self.assertIn(
            "app.translation_horizontal_centered_checkbox = ttk.Checkbutton(\n"
            "        app.translation_text_style_frame,",
            source,
        )


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
