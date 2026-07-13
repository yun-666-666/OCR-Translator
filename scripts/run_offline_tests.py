"""Run every repository unittest entrypoint without contaminating runtime logs."""

from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Optional, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TEST_COMMANDS = (
    (
        "tests discovery",
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-q"],
    ),
    (
        "legacy Custom AI roots",
        [
            sys.executable,
            "-m",
            "unittest",
            "test_custom_ai",
            "test_custom_ai_startup",
            "-q",
        ],
    ),
)
_TEST_COUNT_PATTERN = re.compile(r"Ran (\d+) tests?", re.IGNORECASE)


def _reported_test_count(output: str) -> int:
    match = _TEST_COUNT_PATTERN.search(output)
    return int(match.group(1)) if match else 0


def _write_child_output(completed_process: subprocess.CompletedProcess[str]) -> None:
    if completed_process.stdout:
        sys.stdout.write(completed_process.stdout)
    if completed_process.stderr:
        sys.stderr.write(completed_process.stderr)


def run_offline_tests(repository_root: Path = REPOSITORY_ROOT) -> int:
    """Run both unittest roots with one disposable log directory for child tests."""
    child_environment = os.environ.copy()
    completed_count = 0
    total_test_count = 0

    with tempfile.TemporaryDirectory(prefix="ocr-translator-offline-tests-") as log_directory:
        child_environment["OCR_TRANSLATOR_LOG_DIR"] = log_directory
        print("[offline-tests] child logs use a temporary directory", flush=True)

        for label, command in TEST_COMMANDS:
            print(
                "[offline-tests] command: {}".format(subprocess.list2cmdline(command)),
                flush=True,
            )
            completed_process = subprocess.run(
                command,
                cwd=repository_root,
                env=child_environment,
                check=False,
                capture_output=True,
                text=True,
            )
            _write_child_output(completed_process)
            combined_output = completed_process.stdout + completed_process.stderr
            total_test_count += _reported_test_count(combined_output)
            print(
                "[offline-tests] {} exited with {}".format(
                    label, completed_process.returncode
                ),
                flush=True,
            )

            if completed_process.returncode:
                print("[offline-tests] stopping after failed {}".format(label), flush=True)
                return completed_process.returncode if completed_process.returncode > 0 else 1
            completed_count += 1

    print(
        "[offline-tests] passed {} commands; combined unittest count: {}".format(
            completed_count, total_test_count
        ),
        flush=True,
    )
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Provide a small callable CLI entrypoint for local and CI verification."""
    del argv
    return run_offline_tests()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
