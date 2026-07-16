import configparser
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import config_manager
from custom_ai import CustomAIProvider
from handlers.display_manager import DisplayManager
from handlers.ui_interaction_handler import UIInteractionHandler
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


class _TkFallbackText:
    def __init__(self):
        self.content = ""
        self.configurations = []
        self.tag_configurations = []

    def config(self, **kwargs):
        self.configurations.append(kwargs)
        return None

    configure = config

    def winfo_exists(self):
        return True

    def delete(self, *_args):
        self.content = ""

    def insert(self, _index, text):
        self.content += text

    def tag_configure(self, name, **kwargs):
        self.tag_configurations.append((name, kwargs))

    def tag_add(self, *_args):
        return None

    def see(self, *_args):
        return None


class TranslationDisplayLayoutConfigTests(unittest.TestCase):
    def test_load_app_config_migrates_legacy_keep_linebreaks(self):
        original_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as temporary_directory:
            try:
                os.chdir(temporary_directory)
                config = configparser.ConfigParser()
                config["Settings"] = {"keep_linebreaks": "True"}
                with Path("ocr_translator_config.ini").open("w", encoding="utf-8") as handle:
                    config.write(handle)

                loaded = config_manager.load_app_config()
            finally:
                os.chdir(original_cwd)

        self.assertEqual(
            loaded["Settings"]["translation_line_layout"],
            "preserve_source_lines",
        )
        self.assertEqual(loaded["Settings"]["keep_linebreaks"], "True")

    def test_compact_prompt_forbids_explicit_line_breaks(self):
        payload = CustomAIProvider().build_translation_payload(
            profile={"model": "layout-test"},
            text="one\ntwo",
            source_lang="en",
            target_lang="zh-CN",
            keep_linebreaks=False,
        )

        prompt = json.dumps(payload, ensure_ascii=False).lower()
        self.assertIn("single-line", prompt)
        self.assertIn("do not use <br>", prompt)


class TranslationDisplayLayoutForwardingTests(unittest.TestCase):
    def test_display_manager_forwards_compact_layout_and_centering(self):
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
            translation_line_layout_var=_Value("compact"),
            translation_horizontal_centered_var=_Value(True),
        )

        DisplayManager(app)._update_translation_text_on_main_thread("one<br>two")

        translation_text.set_rtl_text.assert_called_once_with(
            "one two",
            "zh-CN",
            "#112233",
            "#ffffff",
            20,
            font_family="Microsoft YaHei",
            font_bold=True,
            preserve_linebreaks=False,
            horizontal_centered=True,
        )

    def test_tk_fallback_flattens_compact_text_and_centers_horizontally(self):
        translation_text = _TkFallbackText()
        app = SimpleNamespace(
            target_overlay=_VisibleOverlay(),
            translation_text=translation_text,
            is_running=True,
            target_lang_var=_Value("en"),
            target_text_colour_var=_Value("#ffffff"),
            target_font_size_var=_Value(20),
            target_font_type_var=_Value("Arial"),
            target_font_bold_var=_Value(True),
            target_colour_var=_Value("#112233"),
            translation_line_layout_var=_Value("compact"),
            translation_horizontal_centered_var=_Value(True),
        )

        with patch("handlers.display_manager.RTL_PROCESSOR_AVAILABLE", False):
            DisplayManager(app)._update_translation_text_on_main_thread(
                "one<br>two\nthree"
            )

        self.assertEqual(translation_text.content, "one two three")
        self.assertIn(
            ("translation_alignment", {"justify": "center"}),
            translation_text.tag_configurations,
        )

    def test_settings_handler_applies_bold_font_tuple(self):
        translation_text = _TkFallbackText()
        app = SimpleNamespace(
            translation_text=translation_text,
            target_font_size_var=_Value(20),
            target_font_type_var=_Value("Arial"),
            target_font_bold_var=_Value(True),
        )

        UIInteractionHandler(app).update_target_font_weight()

        self.assertIn(
            {"font": ("Arial", 20, "bold")},
            translation_text.configurations,
        )


@unittest.skipUnless(PYSIDE6_AVAILABLE, "PySide6 is not available")
class TranslationDisplayLayoutPySideTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication

        cls.qt_app = QApplication.instance() or QApplication([])

    def setUp(self):
        class RecordingTextDisplay(RTLTextDisplay):
            def __init__(self):
                self.html_calls = []
                super().__init__()

            def setHtml(self, html):
                self.html_calls.append(html)
                return super().setHtml(html)

        self.widget = RecordingTextDisplay()

    def tearDown(self):
        self.widget.close()
        self.widget.deleteLater()

    def test_compact_render_flattens_breaks_and_centers_horizontally(self):
        self.widget.set_rtl_text(
            "one<br>two\nthree",
            "en",
            preserve_linebreaks=False,
            horizontal_centered=True,
            font_bold=True,
        )

        self.assertEqual(self.widget.toPlainText(), "one two three")
        self.assertIn("text-align: center", self.widget.html_calls[-1])
        self.assertIn("font-weight: 700", self.widget.html_calls[-1])
        self.assertTrue(self.widget.font().bold())

    def test_preserve_layout_keeps_breaks_and_default_left_alignment(self):
        self.widget.set_rtl_text(
            "one<br>two\nthree",
            "en",
            preserve_linebreaks=True,
        )

        self.assertEqual(
            self.widget.toPlainText().splitlines(),
            ["one", "two", "three"],
        )
        self.assertIn("text-align: left", self.widget.html_calls[-1])


if __name__ == "__main__":
    unittest.main()
