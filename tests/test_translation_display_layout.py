import configparser
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import config_manager
from custom_ai import CustomAIProvider
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
            target_colour_var=_Value("#112233"),
            translation_line_layout_var=_Value("compact"),
            translation_horizontal_centered_var=_Value(True),
        )

        DisplayManager(app)._update_translation_text_on_main_thread("one<br>two")

        translation_text.set_rtl_text.assert_called_once_with(
            "one\ntwo",
            "zh-CN",
            "#112233",
            "#ffffff",
            20,
            font_family="Microsoft YaHei",
            preserve_linebreaks=False,
            horizontal_centered=True,
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
        )

        self.assertEqual(self.widget.toPlainText(), "one two three")
        self.assertIn("text-align: center", self.widget.html_calls[-1])

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

