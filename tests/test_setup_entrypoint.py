import importlib
import sys
import unittest
from unittest.mock import patch


class SetupConfigurationTests(unittest.TestCase):
    def test_setup_exposes_the_real_console_entry_point_and_runtime_modules(self):
        sys.modules.pop("setup", None)

        with patch("setuptools.setup") as mock_setup:
            importlib.import_module("setup")

        kwargs = mock_setup.call_args.kwargs
        py_modules = set(kwargs["py_modules"])

        self.assertIn("main", py_modules)
        self.assertIn("app_logic", py_modules)
        self.assertNotIn("test_custom_ai", py_modules)
        self.assertEqual(
            kwargs["entry_points"]["console_scripts"],
            ["ocr_translator=main:main_entry_point"],
        )

