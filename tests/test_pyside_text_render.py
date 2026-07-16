import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from handlers.display_manager import DisplayManager
from pyside_overlay import PYSIDE6_AVAILABLE, RTLTextDisplay


class _Value:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value


class _VisibleOverlay:
    def winfo_exists(self):
        return True

    def winfo_viewable(self):
        return True

    def attributes(self, *_args):
        return None


class DisplayManagerPySideRenderTests(unittest.TestCase):
    def test_pyside_update_passes_font_in_one_render_call(self):
        translation_text = Mock()
        app = SimpleNamespace(
            target_overlay=_VisibleOverlay(),
            translation_text=translation_text,
            is_running=True,
            target_lang_var=_Value("zh-CN"),
            target_text_colour_var=_Value("#ffffff"),
            target_font_size_var=_Value(20),
            target_font_type_var=_Value("Microsoft YaHei"),
            target_font_bold_var=_Value(True),
            target_colour_var=_Value("#112233"),
        )

        DisplayManager(app)._update_translation_text_on_main_thread(
            " 第一行<br>第二行 "
        )

        translation_text.set_rtl_text.assert_called_once_with(
            "第一行\n第二行",
            "zh-CN",
            "#112233",
            "#ffffff",
            20,
            font_family="Microsoft YaHei",
            font_bold=True,
            preserve_linebreaks=True,
            horizontal_centered=False,
        )
        translation_text.configure.assert_not_called()

    def test_pyside_logging_is_content_free_and_routes_are_coalesced(self):
        translation_text = Mock()
        app = SimpleNamespace(
            target_overlay=_VisibleOverlay(),
            translation_text=translation_text,
            is_running=True,
            target_lang_var=_Value("zh-CN"),
            target_text_colour_var=_Value("#ffffff"),
            target_font_size_var=_Value(20),
            target_font_type_var=_Value("Microsoft YaHei"),
            target_font_bold_var=_Value(False),
            target_colour_var=_Value("#112233"),
        )

        with patch(
            "handlers.display_manager.log_debug"
        ) as direct_log:
            with patch(
                "handlers.display_manager.log_debug_coalesced",
                create=True,
            ) as coalesced_log:
                DisplayManager(app)._update_translation_text_on_main_thread(
                    " display-secret "
                )

        direct_messages = "\n".join(
            str(call.args[0]) for call in direct_log.call_args_list
        )
        coalesced_messages = "\n".join(
            str(call.args[1]) for call in coalesced_log.call_args_list
        )
        self.assertNotIn("display-secret", direct_messages)
        self.assertNotIn("display-secret", coalesced_messages)
        self.assertIn("chars=14 lines=1", direct_messages)
        self.assertEqual(coalesced_log.call_count, 2)
        translation_text.set_rtl_text.assert_called_once_with(
            "display-secret",
            "zh-CN",
            "#112233",
            "#ffffff",
            20,
            font_family="Microsoft YaHei",
            font_bold=False,
            preserve_linebreaks=True,
            horizontal_centered=False,
        )


@unittest.skipUnless(PYSIDE6_AVAILABLE, "PySide6 is not available")
class RTLTextDisplayFontTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        class CountingRTLTextDisplay(RTLTextDisplay):
            def __init__(self):
                self.html_calls = []
                self.font_at_html = []
                super().__init__()

            def setHtml(self, html):
                self.html_calls.append(html)
                self.font_at_html.append(
                    (
                        self.font().family(),
                        self.font().pointSize(),
                        self.font().bold(),
                    )
                )
                return super().setHtml(html)

        self.widget = CountingRTLTextDisplay()

    def tearDown(self):
        self.widget.close()
        self.widget.deleteLater()

    def _target_family(self):
        from PySide6.QtGui import QFont

        target = QFont(self.widget.font())
        target.setFamily("Consolas")
        return target.family()

    def test_direct_render_applies_font_before_one_html_update(self):
        family = self._target_family()

        self.widget.set_rtl_text(
            "hello",
            "en",
            "#000000",
            "#ffffff",
            18,
            font_family=family,
            font_bold=True,
        )

        self.assertEqual(self.widget.font().family(), family)
        self.assertEqual(self.widget.font().pointSize(), 18)
        self.assertTrue(self.widget.font().bold())
        self.assertEqual(len(self.widget.html_calls), 1)
        self.assertEqual(self.widget.font_at_html, [(family, 18, True)])
        self.assertIn(f"font-family: '{family}'", self.widget.html_calls[0])
        self.assertIn("font-weight: 700", self.widget.html_calls[0])

    def test_five_argument_render_preserves_existing_font_fallbacks(self):
        from PySide6.QtGui import QFont

        existing_font = QFont(self.widget.font())
        existing_font.setFamilies(["Arial Unicode MS", "Segoe UI", "Tahoma"])
        existing_font.setPointSize(12)
        self.widget.setFont(existing_font)
        original_families = self.widget.font().families()

        self.widget.set_rtl_text(
            "hello", "en", "#000000", "#ffffff", 18
        )

        self.assertEqual(self.widget.font().pointSize(), 18)
        self.assertEqual(self.widget.font().families(), original_families)

    def test_legacy_positional_layout_arguments_remain_compatible(self):
        family = self._target_family()

        self.widget.set_rtl_text(
            "one<br>two",
            "en",
            "#000000",
            "#ffffff",
            18,
            family,
            False,
            True,
        )

        self.assertEqual(self.widget.toPlainText(), "one two")
        self.assertIn("text-align: center", self.widget.html_calls[-1])

    def test_repeating_active_font_through_config_does_not_rerender(self):
        family = self._target_family()
        self.widget.set_rtl_text(
            "hello", "en", "#000000", "#ffffff", 18, font_family=family
        )
        self.widget.html_calls.clear()

        self.widget.config(font=(family, 18))

        self.assertEqual(self.widget.html_calls, [])

    def test_real_font_change_through_config_rerenders_once(self):
        family = self._target_family()
        self.widget.set_rtl_text(
            "hello", "en", "#000000", "#ffffff", 18, font_family=family
        )
        self.widget.html_calls.clear()

        self.widget.config(font=(family, 19))

        self.assertEqual(self.widget.font().pointSize(), 19)
        self.assertEqual(len(self.widget.html_calls), 1)

    def test_weight_change_through_config_rerenders_once(self):
        family = self._target_family()
        self.widget.set_rtl_text(
            "hello",
            "en",
            "#000000",
            "#ffffff",
            18,
            font_family=family,
            font_bold=False,
        )
        self.widget.html_calls.clear()

        self.widget.config(font=(family, 18, "bold"))

        self.assertTrue(self.widget.font().bold())
        self.assertEqual(len(self.widget.html_calls), 1)
        self.assertIn("font-weight: 700", self.widget.html_calls[0])


if __name__ == "__main__":
    unittest.main()
