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

    def test_setup_python_range_and_runtime_dependencies_match_requirements(self):
        sys.modules.pop("setup", None)

        with patch("setuptools.setup") as mock_setup:
            importlib.import_module("setup")

        kwargs = mock_setup.call_args.kwargs
        install_requires = set(kwargs["install_requires"])
        classifiers = set(kwargs["classifiers"])

        self.assertEqual(kwargs["python_requires"], ">=3.9,<3.13")
        self.assertIn("Programming Language :: Python :: 3.9", classifiers)
        self.assertIn("Programming Language :: Python :: 3.10", classifiers)
        self.assertIn("Programming Language :: Python :: 3.11", classifiers)
        self.assertIn("Programming Language :: Python :: 3.12", classifiers)
        self.assertNotIn("Programming Language :: Python :: 3.7", classifiers)
        self.assertNotIn("Programming Language :: Python :: 3.8", classifiers)

        for requirement in {
            "mss>=9.0.0",
            "PySide6==6.7.3",
            "keyboard>=0.13.5",
            "python-bidi>=0.4.2",
            "arabic-reshaper>=3.0.0",
        }:
            self.assertIn(requirement, install_requires)
