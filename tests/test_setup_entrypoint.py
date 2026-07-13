import importlib
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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

    def test_setup_declares_csv_resources_as_package_data(self):
        sys.modules.pop("setup", None)

        with patch("setuptools.setup") as mock_setup:
            importlib.import_module("setup")

        kwargs = mock_setup.call_args.kwargs
        self.assertIn("resources", kwargs["packages"])
        self.assertEqual(kwargs["package_data"]["resources"], ["*.csv"])

    def test_isolated_wheel_contains_runtime_modules_and_all_csv_resources(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            source_root = Path(temporary_directory) / "source"
            wheel_output = Path(temporary_directory) / "wheel-output"
            source_root.mkdir()
            wheel_output.mkdir()

            for source_path in PROJECT_ROOT.glob("*.py"):
                shutil.copy2(source_path, source_root / source_path.name)
            shutil.copy2(PROJECT_ROOT / "setup.py", source_root / "setup.py")
            shutil.copy2(PROJECT_ROOT / "README.md", source_root / "README.md")
            shutil.copytree(PROJECT_ROOT / "handlers", source_root / "handlers")
            shutil.copytree(PROJECT_ROOT / "resources", source_root / "resources")

            completed = subprocess.run(
                [
                    sys.executable,
                    "setup.py",
                    "bdist_wheel",
                    "--dist-dir",
                    str(wheel_output),
                ],
                cwd=source_root,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)

            wheel_path = next(wheel_output.glob("*.whl"), None)
            self.assertIsNotNone(wheel_path)
            with zipfile.ZipFile(wheel_path) as wheel_archive:
                wheel_members = set(wheel_archive.namelist())

            sys.modules.pop("setup", None)
            with patch("setuptools.setup"):
                setup_module = importlib.import_module("setup")
            runtime_module_names = {
                f"{module_name}.py" for module_name in setup_module.ROOT_PY_MODULES
            }
            self.assertTrue(runtime_module_names)
            self.assertTrue(runtime_module_names.issubset(wheel_members))
            for resource_path in (PROJECT_ROOT / "resources").glob("*.csv"):
                self.assertIn(f"resources/{resource_path.name}", wheel_members)


class CompileAppSafetyTests(unittest.TestCase):
    def _load_compile_app(self):
        sys.modules.pop("compile_app", None)
        return importlib.import_module("compile_app")

    def test_refuses_pip_environment_mutation_without_explicit_flag(self):
        compile_app = self._load_compile_app()

        with patch.object(compile_app.subprocess, "run") as run:
            success = compile_app.run_command(
                [sys.executable, "-m", "pip", "install", "PyInstaller"],
                "Install PyInstaller",
            )

        self.assertFalse(success)
        run.assert_not_called()

    def test_explicit_mutation_flag_uses_active_interpreter_argument_array(self):
        compile_app = self._load_compile_app()

        with patch.object(
            compile_app.subprocess,
            "run",
            return_value=subprocess.CompletedProcess([], 0, stdout="ok", stderr=""),
        ) as run:
            success = compile_app.run_command(
                [sys.executable, "-m", "pip", "install", "PyInstaller"],
                "Install PyInstaller",
                allow_environment_mutation=True,
            )

        self.assertTrue(success)
        run.assert_called_once_with(
            [sys.executable, "-m", "pip", "install", "PyInstaller"],
            check=True,
            capture_output=True,
            text=True,
        )

    def test_missing_pyinstaller_refuses_installation_without_explicit_flag(self):
        compile_app = self._load_compile_app()

        with patch.object(compile_app.importlib.util, "find_spec", return_value=None), patch.object(
            compile_app.subprocess, "run"
        ) as run:
            available = compile_app.ensure_pyinstaller()

        self.assertFalse(available)
        run.assert_not_called()

    def test_missing_pyinstaller_installs_with_explicit_flag_using_active_interpreter(self):
        compile_app = self._load_compile_app()

        with patch.object(compile_app.importlib.util, "find_spec", return_value=None), patch.object(
            compile_app.subprocess,
            "run",
            return_value=subprocess.CompletedProcess([], 0, stdout="ok", stderr=""),
        ) as run:
            available = compile_app.ensure_pyinstaller(allow_environment_mutation=True)

        self.assertTrue(available)
        run.assert_called_once_with(
            [sys.executable, "-m", "pip", "install", "PyInstaller"],
            check=True,
            capture_output=True,
            text=True,
        )

    def test_cpu_build_uses_active_interpreter_pyinstaller_array_without_torch_mutation(self):
        compile_app = self._load_compile_app()

        with patch.object(compile_app, "ensure_pyinstaller", return_value=True), patch.object(
            compile_app, "verify_pytorch_installation", return_value=True
        ), patch.object(
            compile_app.subprocess,
            "run",
            return_value=subprocess.CompletedProcess([], 0, stdout="ok", stderr=""),
        ) as run:
            success = compile_app.compile_cpu_version()

        self.assertTrue(success)
        run.assert_called_once_with(
            [sys.executable, "-m", "PyInstaller", "GameChangingTranslator.spec"],
            check=True,
            capture_output=True,
            text=True,
        )


class PackagingDocumentationTests(unittest.TestCase):
    def test_install_and_build_guidance_matches_the_supported_workflow(self):
        readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
        developer_guide = (PROJECT_ROOT / "docs" / "developer-guide.md").read_text(
            encoding="utf-8"
        )

        for document in (readme, developer_guide):
            self.assertIn("3.10.2", document)
            self.assertIn("requirements.txt", document)
            self.assertIn("does not install PaddleOCR", document)
            self.assertIn("optional PaddleOCR backend", document)

        self.assertIn("GameChangingTranslator.spec", developer_guide)
        self.assertIn("GameChangingTranslator_GPU.spec", developer_guide)
        self.assertIn("isolated virtual environment", developer_guide)
        self.assertIn("python -m pip install wheel", developer_guide)

    def test_offline_ci_installs_wheel_as_a_test_build_tool(self):
        workflow = (PROJECT_ROOT / ".github" / "workflows" / "offline-tests.yml").read_text(
            encoding="utf-8"
        )

        self.assertIn("python -m pip install wheel", workflow)
