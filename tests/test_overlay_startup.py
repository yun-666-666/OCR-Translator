import unittest
from unittest.mock import patch

import configparser

import overlay_manager


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


if __name__ == "__main__":
    unittest.main()
