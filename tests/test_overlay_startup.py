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


class FakeReadyOverlay:
    def __init__(self, visible=False):
        self.visible = visible
        self.destroyed = False

    def winfo_exists(self):
        return not self.destroyed

    def winfo_viewable(self):
        return self.visible

    def hide(self):
        self.visible = False


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
                "translation_text": None,
            },
        )()

    def test_hidden_startup_restores_and_prepares_both_overlays(self):
        app = self._build_app(target_visible=False)
        created = []

        def create_source(fake_app, force_hidden=False):
            created.append(("source", force_hidden))
            fake_app.source_overlay = FakeReadyOverlay()

        def create_target(fake_app, skip_preservation=False, force_hidden=False):
            created.append(("target", force_hidden))
            fake_app.target_overlay = FakeReadyOverlay()
            fake_app.translation_text = FakeReadyOverlay()

        with patch.object(overlay_manager, "create_source_overlay_om", side_effect=create_source), patch.object(
            overlay_manager, "create_target_overlay_om", side_effect=create_target
        ):
            ready = overlay_manager.load_areas_from_config_om(app)

        self.assertTrue(ready)
        self.assertEqual([("source", True), ("target", True)], created)
        self.assertEqual(app.source_area, [0, 0, 100, 50])
        self.assertEqual(app.target_area, [10, 20, 210, 120])

    def test_readiness_reuses_live_overlays(self):
        app = self._build_app()
        app.source_area = [0, 0, 100, 50]
        app.target_area = [10, 20, 210, 120]
        app.source_overlay = FakeReadyOverlay()
        app.target_overlay = FakeReadyOverlay()
        app.translation_text = FakeReadyOverlay()

        with patch.object(overlay_manager, "create_source_overlay_om") as create_source, patch.object(
            overlay_manager, "create_target_overlay_om"
        ) as create_target:
            ready = overlay_manager.ensure_overlays_ready_om(app, force_hidden=True)

        self.assertTrue(ready)
        create_source.assert_not_called()
        create_target.assert_not_called()

    def test_readiness_recreates_destroyed_overlays_from_saved_areas(self):
        app = self._build_app()
        app.source_area = [0, 0, 100, 50]
        app.target_area = [10, 20, 210, 120]
        app.source_overlay = FakeReadyOverlay()
        app.source_overlay.destroyed = True
        app.target_overlay = FakeReadyOverlay()
        app.target_overlay.destroyed = True
        app.translation_text = FakeReadyOverlay()
        app.translation_text.destroyed = True
        created = []

        def create_source(fake_app, force_hidden=False):
            created.append(("source", force_hidden))
            fake_app.source_overlay = FakeReadyOverlay()

        def create_target(fake_app, skip_preservation=False, force_hidden=False):
            created.append(("target", force_hidden))
            fake_app.target_overlay = FakeReadyOverlay()
            fake_app.translation_text = FakeReadyOverlay()

        with patch.object(
            overlay_manager,
            "create_source_overlay_om",
            side_effect=create_source,
        ), patch.object(
            overlay_manager,
            "create_target_overlay_om",
            side_effect=create_target,
        ):
            ready = overlay_manager.ensure_overlays_ready_om(app, force_hidden=True)

        self.assertTrue(ready)
        self.assertEqual([("source", True), ("target", True)], created)

    def test_readiness_rejects_invalid_saved_coordinates(self):
        app = self._build_app()
        app.config["Settings"]["source_area_x2"] = "0"

        with patch.object(overlay_manager, "create_source_overlay_om") as create_source, patch.object(
            overlay_manager,
            "create_target_overlay_om",
        ) as create_target:
            ready = overlay_manager.ensure_overlays_ready_om(app, force_hidden=True)

        self.assertFalse(ready)
        create_source.assert_not_called()
        create_target.assert_not_called()

    def test_force_hidden_source_overlay_is_withdrawn_during_construction(self):
        app = self._build_app(source_visible=True)
        app.root = object()
        app.source_colour_var = type("Var", (), {"get": lambda self: "#ffff99"})()
        overlay = FakeResizableOverlay()
        overlay.visible = True

        with patch.object(
            overlay_manager,
            "ResizableMovableFrame",
            return_value=overlay,
        ) as overlay_factory:
            overlay_manager.create_source_overlay_om(app, force_hidden=True)

        overlay_factory.assert_called_once_with(
            app.root,
            app.source_area,
            bg_color="#ffff99",
            title="",
            start_hidden=True,
        )
        self.assertFalse(app.source_overlay.winfo_viewable())

    def test_toggle_target_visibility_creates_overlay_on_demand(self):
        app = self._build_app(target_visible=False)
        app.target_area = [10, 20, 210, 120]

        class Root:
            def __init__(self):
                self.idle_updates = 0

            def update_idletasks(self):
                self.idle_updates += 1

        app.root = Root()
        app.target_colour_var = type("Var", (), {"get": lambda self: "#000000"})()

        class FakeQApplication:
            def __init__(self):
                self.process_events_count = 0

            def processEvents(self):
                self.process_events_count += 1

        qapp = FakeQApplication()
        manager = type("Manager", (), {"ensure_qapp": lambda self: qapp})()

        class FakeOverlay:
            def __init__(self):
                self.toggle_count = 0
                self.color_update_count = 0
                self.visible = False

            def winfo_exists(self):
                return True

            def update_color(self, color):
                self.color_update_count += 1

            def toggle_visibility(self):
                self.toggle_count += 1
                self.visible = not self.visible

            def winfo_viewable(self):
                return self.visible

        def prepare_overlay(fake_app, **_kwargs):
            fake_app.target_overlay = FakeOverlay()
            fake_app.translation_text = type(
                "FakePySideText",
                (),
                {
                    "winfo_exists": lambda self: True,
                    "set_rtl_text": lambda self, _text: None,
                },
            )()
            return True

        with patch.object(
            overlay_manager, "ensure_overlays_ready_om", side_effect=prepare_overlay
        ) as ensure_ready, patch.object(
            overlay_manager, "create_target_overlay_om", side_effect=prepare_overlay
        ), patch.object(
            overlay_manager,
            "_get_pyside_api",
            return_value=(lambda: manager, lambda: True),
        ):
            overlay_manager.toggle_target_visibility_om(app)

        ensure_ready.assert_called_once_with(
            app,
            require_source=False,
            require_target=True,
            force_hidden=True,
        )
        self.assertEqual(1, app.target_overlay.toggle_count)
        self.assertEqual(0, app.target_overlay.color_update_count)
        self.assertEqual(1, app.root.idle_updates)
        self.assertEqual(1, qapp.process_events_count)

    def test_toggle_source_visibility_creates_overlay_on_demand(self):
        app = self._build_app(source_visible=False)
        app.source_area = [0, 0, 100, 50]
        app.root = None

        class FakeOverlay(FakeReadyOverlay):
            def __init__(self):
                super().__init__()
                self.toggle_count = 0

            def toggle_visibility(self):
                self.toggle_count += 1
                self.visible = not self.visible

        def prepare_overlay(fake_app, **_kwargs):
            fake_app.source_overlay = FakeOverlay()
            return True

        with patch.object(
            overlay_manager, "ensure_overlays_ready_om", side_effect=prepare_overlay
        ) as ensure_ready, patch.object(
            overlay_manager, "create_source_overlay_om", side_effect=prepare_overlay
        ):
            overlay_manager.toggle_source_visibility_om(app)

        ensure_ready.assert_called_once_with(
            app,
            require_source=True,
            require_target=False,
            force_hidden=True,
        )
        self.assertEqual(1, app.source_overlay.toggle_count)

    def test_startup_schedules_overlay_readiness_before_paddleocr_prewarm(self):
        import app_logic

        scheduled = []

        class Root:
            def after(self, delay_ms, callback):
                scheduled.append((delay_ms, callback))

        app = object.__new__(app_logic.GameChangingTranslator)
        app.root = Root()
        app.load_initial_overlay_areas = lambda: None
        app.schedule_initial_paddleocr_prewarm = lambda: None

        app.schedule_initial_ui_readiness()

        self.assertEqual(
            [
                (50, app.load_initial_overlay_areas),
                (250, app.schedule_initial_paddleocr_prewarm),
            ],
            scheduled,
        )

    def test_start_preflight_repairs_missing_overlays(self):
        import app_lifecycle
        import app_logic

        app = object.__new__(app_logic.GameChangingTranslator)
        app.source_overlay = None
        app.target_overlay = None
        app.translation_text = None

        with patch.object(
            app_lifecycle,
            "ensure_overlays_ready_om",
            return_value=True,
            create=True,
        ) as ensure_ready:
            ready = app._ensure_overlays_for_start()

        self.assertTrue(ready)
        ensure_ready.assert_called_once_with(app, force_hidden=True)

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
