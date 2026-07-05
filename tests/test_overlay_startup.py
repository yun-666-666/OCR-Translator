import unittest
from unittest.mock import patch

import configparser

import pyside_overlay
import overlay_manager
import ui_elements
from config_manager import DEFAULT_CONFIG_SETTINGS


class FakeWidget:
    def __init__(self):
        self.config_calls = []

    def configure(self, **kwargs):
        self.config_calls.append(kwargs)

    config = configure


class FakeResizableOverlay:
    def __init__(self, *args, **kwargs):
        self._alpha = 0.1
        self.alpha_sets = []
        self.after_calls = []
        self.redraws = 0
        self.visible = False
        self.destroyed = False
        self.title_bar = FakeWidget()
        self.content_frame = FakeWidget()
        self.close_button = FakeWidget()
        self.resize_n = FakeWidget()
        self.resize_s = FakeWidget()
        self.resize_w = FakeWidget()
        self.resize_e = FakeWidget()
        self.resize_nw = FakeWidget()
        self.resize_ne = FakeWidget()
        self.resize_sw = FakeWidget()
        self.resize_se = FakeWidget()

    def configure(self, **kwargs):
        self.config_calls = getattr(self, "config_calls", [])
        self.config_calls.append(kwargs)

    def winfo_exists(self):
        return not self.destroyed

    def winfo_viewable(self):
        return self.visible

    def attributes(self, key, value=None):
        if key != "-alpha":
            return None
        if value is None:
            return self._alpha
        self._alpha = value
        self.alpha_sets.append(value)
        return None

    def update_idletasks(self):
        self.redraws += 1

    def after(self, delay, callback):
        self.after_calls.append(delay)
        callback()

    def hide(self):
        self.visible = False

    def show(self):
        self.visible = True

    def destroy(self):
        self.destroyed = True

    update_color = ui_elements.ResizableMovableFrame.update_color


class OverlayStartupTests(unittest.TestCase):
    def _build_app(self, target_visible=False, source_visible=False):
        cfg = configparser.ConfigParser()
        cfg["Settings"] = {
            "source_area_x1": "0",
            "source_area_y1": "0",
            "source_area_x2": "100",
            "source_area_y2": "50",
            "target_area_x1": "10",
            "target_area_y1": "20",
            "target_area_x2": "210",
            "target_area_y2": "120",
            "source_area_visible": "True" if source_visible else "False",
            "target_area_visible": "True" if target_visible else "False",
        }
        return type(
            "App",
            (),
            {
                "config": cfg,
                "source_area": None,
                "target_area": None,
                "source_overlay": None,
                "target_overlay": None,
            },
        )()

    def test_hidden_target_overlay_is_not_created_during_config_load(self):
        app = self._build_app(target_visible=False)
        with patch.object(overlay_manager, "create_source_overlay_om") as create_source, patch.object(
            overlay_manager, "create_target_overlay_om"
        ) as create_target:
            overlay_manager.load_areas_from_config_om(app)

        create_source.assert_not_called()
        create_target.assert_not_called()
        self.assertEqual(app.target_area, [10, 20, 210, 120])

    def test_visible_target_overlay_still_gets_created_during_config_load(self):
        app = self._build_app(target_visible=True)
        with patch.object(overlay_manager, "create_source_overlay_om") as create_source, patch.object(
            overlay_manager, "create_target_overlay_om"
        ) as create_target:
            overlay_manager.load_areas_from_config_om(app)

        create_source.assert_not_called()
        create_target.assert_called_once()

    def test_toggle_target_visibility_creates_overlay_on_demand(self):
        app = self._build_app(target_visible=False)
        app.target_area = [10, 20, 210, 120]
        app.root = None
        app.target_colour_var = type("Var", (), {"get": lambda self: "#000000"})()

        class FakeOverlay:
            def __init__(self):
                self.toggled = False

            def winfo_exists(self):
                return True

            def update_color(self, color):
                self.color = color

            def toggle_visibility(self):
                self.toggled = True

            def winfo_viewable(self):
                return True

        def create_overlay(fake_app):
            fake_app.target_overlay = FakeOverlay()

        with patch.object(overlay_manager, "create_target_overlay_om", side_effect=create_overlay) as create_target:
            overlay_manager.toggle_target_visibility_om(app)

        create_target.assert_called_once_with(app)
        self.assertTrue(app.target_overlay.toggled)

    def test_resizable_frame_update_color_does_not_change_window_alpha(self):
        overlay = FakeResizableOverlay()
        overlay.attributes("-alpha", 0.7)
        overlay.alpha_sets.clear()

        ui_elements.ResizableMovableFrame.update_color(overlay, "#123456")

        self.assertEqual([], overlay.alpha_sets)
        self.assertEqual(1, overlay.redraws)

    def test_source_overlay_creation_color_update_keeps_alpha_stable(self):
        app = self._build_app(source_visible=True)
        app.root = object()
        app.source_colour_var = type("Var", (), {"get": lambda self: "#ffff99"})()

        with patch.object(overlay_manager, "ResizableMovableFrame", FakeResizableOverlay):
            overlay_manager.create_source_overlay_om(app)

        self.assertEqual([0.7], app.source_overlay.alpha_sets)

    def test_default_target_opacity_uses_stable_background_value(self):
        self.assertEqual("0.4", DEFAULT_CONFIG_SETTINGS["target_opacity"])


@unittest.skipUnless(pyside_overlay.PYSIDE6_AVAILABLE, "PySide6 is not installed")
class PySideOverlayStartupTests(unittest.TestCase):
    def _fake_pyside_overlay(self):
        class FakeStyleWidget:
            def __init__(self):
                self.styles = []

            def setStyleSheet(self, style):
                self.styles.append(style)

        class FakeTextWidget:
            def __init__(self):
                self.config_calls = []

            def config(self, **kwargs):
                self.config_calls.append(kwargs)

        fake = type("FakePySideOverlay", (), {})()
        central = FakeStyleWidget()
        fake.bg_color = "#162c43"
        fake._opacity = 0.4
        fake._border_px = 1
        fake._corner_radius = 16
        fake._text_padding = (5, 5)
        fake.top_bar = FakeStyleWidget()
        fake.text_widget = FakeTextWidget()
        fake.centralWidget = lambda: central
        fake._hex_to_rgba = pyside_overlay.PySideTranslationOverlay._hex_to_rgba.__get__(fake, fake.__class__)
        fake._adjust_color_brightness = pyside_overlay.PySideTranslationOverlay._adjust_color_brightness.__get__(
            fake,
            fake.__class__,
        )
        return fake, central

    def test_update_color_skips_reapplying_unchanged_background_style(self):
        overlay, central = self._fake_pyside_overlay()

        pyside_overlay.PySideTranslationOverlay.update_color(overlay, "#162c43", 0.4)
        self.assertGreater(len(central.styles), 0)

        central.styles.clear()
        overlay.top_bar.styles.clear()
        overlay.text_widget.config_calls.clear()

        pyside_overlay.PySideTranslationOverlay.update_color(overlay, "#162c43", 0.4)

        self.assertEqual([], central.styles)
        self.assertEqual([], overlay.top_bar.styles)
        self.assertEqual([], overlay.text_widget.config_calls)


if __name__ == "__main__":
    unittest.main()
