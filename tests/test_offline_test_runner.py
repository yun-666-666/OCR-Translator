import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RUNNER_SOURCE = REPOSITORY_ROOT / "scripts" / "run_offline_tests.py"


class OfflineTestRunnerTests(unittest.TestCase):
    def test_windows_workflow_limits_token_to_read_only_contents(self):
        workflow = (
            REPOSITORY_ROOT / ".github" / "workflows" / "offline-tests.yml"
        ).read_text(encoding="utf-8")

        self.assertRegex(
            workflow,
            r"(?m)^permissions:\s+contents: read\s+jobs:$",
        )

    def test_windows_workflow_does_not_persist_checkout_credentials(self):
        workflow = (
            REPOSITORY_ROOT / ".github" / "workflows" / "offline-tests.yml"
        ).read_text(encoding="utf-8")

        self.assertRegex(
            workflow,
            r"- uses: actions/checkout@v4\s+with:\s+persist-credentials: false",
        )

    def _write_isolated_project(self, project_root, discovery_source):
        scripts_directory = project_root / "scripts"
        tests_directory = project_root / "tests"
        scripts_directory.mkdir()
        tests_directory.mkdir()
        shutil.copy2(RUNNER_SOURCE, scripts_directory / RUNNER_SOURCE.name)

        (tests_directory / "test_discovery.py").write_text(
            textwrap.dedent(discovery_source),
            encoding="utf-8",
        )
        (project_root / "test_custom_ai.py").write_text(
            textwrap.dedent(
                """
                import unittest


                class LegacyCustomAITests(unittest.TestCase):
                    def test_legacy_custom_ai_root_runs(self):
                        print("DIRECT_CUSTOM_AI_RAN")
                """
            ),
            encoding="utf-8",
        )
        (project_root / "test_custom_ai_startup.py").write_text(
            textwrap.dedent(
                """
                import unittest


                class LegacyCustomAIStartupTests(unittest.TestCase):
                    def test_legacy_custom_ai_startup_root_runs(self):
                        print("DIRECT_CUSTOM_AI_STARTUP_RAN")
                """
            ),
            encoding="utf-8",
        )

    def test_runner_executes_both_unittest_roots_in_an_isolated_log_directory(self):
        self.assertTrue(
            RUNNER_SOURCE.is_file(),
            "offline suite runner must exist before it can isolate child tests",
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            project_root = Path(temporary_directory) / "isolated-project"
            project_root.mkdir()
            self._write_isolated_project(
                project_root,
                """
                import os
                from pathlib import Path
                import unittest


                class DiscoveryTests(unittest.TestCase):
                    def test_discovery_runs_in_the_runner_log_directory(self):
                        Path(os.environ["OCR_TRANSLATOR_LOG_DIR"]).joinpath(
                            "discovery-child-marker.txt"
                        ).write_text("child only", encoding="utf-8")
                        print("DISCOVERY_RAN")
                """,
            )
            caller_log_directory = Path(temporary_directory) / "caller-log-directory"
            caller_log_directory.mkdir()
            caller_marker = caller_log_directory / "caller-marker.txt"
            caller_marker.write_text("unchanged", encoding="utf-8")
            environment = os.environ.copy()
            environment["OCR_TRANSLATOR_LOG_DIR"] = str(caller_log_directory)

            result = subprocess.run(
                [sys.executable, "scripts/run_offline_tests.py"],
                cwd=project_root,
                env=environment,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("unittest discover -s tests -q", result.stdout)
            self.assertIn(
                "unittest test_custom_ai test_custom_ai_startup -q", result.stdout
            )
            self.assertIn("DISCOVERY_RAN", result.stdout)
            self.assertIn("DIRECT_CUSTOM_AI_RAN", result.stdout)
            self.assertIn("DIRECT_CUSTOM_AI_STARTUP_RAN", result.stdout)
            self.assertIn("combined unittest count: 3", result.stdout)
            self.assertEqual(
                sorted(path.name for path in caller_log_directory.iterdir()),
                [caller_marker.name],
            )
            self.assertEqual(caller_marker.read_text(encoding="utf-8"), "unchanged")

    def test_runner_stops_at_the_first_nonzero_child_result(self):
        self.assertTrue(
            RUNNER_SOURCE.is_file(),
            "offline suite runner must exist before it can isolate child tests",
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            project_root = Path(temporary_directory) / "failing-project"
            project_root.mkdir()
            self._write_isolated_project(
                project_root,
                """
                import unittest


                class FailingDiscoveryTests(unittest.TestCase):
                    def test_intentional_failure(self):
                        self.fail("isolated discovery failure")
                """,
            )

            result = subprocess.run(
                [sys.executable, "scripts/run_offline_tests.py"],
                cwd=project_root,
                capture_output=True,
                text=True,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("unittest discover -s tests -q", result.stdout)
            self.assertNotIn("DIRECT_CUSTOM_AI_RAN", result.stdout)
            self.assertNotIn(
                "unittest test_custom_ai test_custom_ai_startup -q", result.stdout
            )


if __name__ == "__main__":
    unittest.main()
