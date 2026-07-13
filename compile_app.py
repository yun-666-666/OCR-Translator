#!/usr/bin/env python3
"""Build the existing PyInstaller specs without changing the active environment.

The default path only verifies the current interpreter and invokes PyInstaller.
Installing PyInstaller is an explicit opt-in operation; this script never
uninstalls, upgrades, or replaces PyTorch.
"""

import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path


_PIP_MUTATION_ACTIONS = frozenset({"install", "uninstall"})


def _is_environment_mutation(command):
    """Return whether an argument-list command changes installed packages."""
    return (
        len(command) >= 4
        and command[1:3] == ["-m", "pip"]
        and command[3] in _PIP_MUTATION_ACTIONS
    )


def run_command(command, description, *, allow_environment_mutation=False):
    """Run an argument-list command and reject package mutations by default."""
    if not isinstance(command, (list, tuple)) or not command or not all(
        isinstance(argument, str) and argument for argument in command
    ):
        raise ValueError("Commands must be non-empty argument lists")

    command = list(command)
    print(f"\n{description}")
    print("-" * len(description))
    if _is_environment_mutation(command) and not allow_environment_mutation:
        print("Refusing package-changing command without --allow-environment-mutation.")
        return False

    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print("Success!")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as error:
        print(f"Error: {error}")
        if error.stdout:
            print("STDOUT:", error.stdout)
        if error.stderr:
            print("STDERR:", error.stderr)
        return False


def ensure_pyinstaller(allow_environment_mutation=False):
    """Use the current PyInstaller or explicitly install it in this interpreter."""
    if importlib.util.find_spec("PyInstaller") is not None:
        return True

    return run_command(
        [sys.executable, "-m", "pip", "install", "PyInstaller"],
        "PyInstaller is missing; installing it in the active environment...",
        allow_environment_mutation=allow_environment_mutation,
    )


def verify_pytorch_installation():
    """Verify the active interpreter's existing PyTorch installation."""
    print("\nVerifying existing PyTorch installation...")
    verification_code = """
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device count: {torch.cuda.device_count()}")
    for index in range(torch.cuda.device_count()):
        print(f"GPU {index}: {torch.cuda.get_device_name(index)}")
else:
    print("CUDA version: N/A")
"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", verification_code],
            capture_output=True,
            text=True,
            check=True,
        )
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as error:
        print(f"Error verifying PyTorch: {error}")
        if error.stderr:
            print(error.stderr)
        return False


def _compile_spec(specification, label, allow_environment_mutation=False):
    print("\n" + "=" * 50)
    print(f"         COMPILING {label}")
    print("=" * 50)

    if not ensure_pyinstaller(allow_environment_mutation):
        print("PyInstaller is unavailable; no build was started.")
        return False
    if not verify_pytorch_installation():
        print("The active environment cannot import PyTorch; no build was started.")
        return False
    if not run_command(
        [sys.executable, "-m", "PyInstaller", specification],
        f"Compiling {specification} with PyInstaller...",
        allow_environment_mutation=allow_environment_mutation,
    ):
        print("Compilation failed!")
        return False

    print("\n" + "=" * 50)
    print(f"    {label} COMPILED SUCCESSFULLY!")
    print("=" * 50)
    return True


def compile_cpu_version(allow_environment_mutation=False):
    """Compile the standard spec using the current interpreter unchanged."""
    return _compile_spec(
        "GameChangingTranslator.spec",
        "STANDARD VERSION",
        allow_environment_mutation,
    )


def compile_gpu_version(allow_environment_mutation=False):
    """Compile the GPU spec using the current interpreter unchanged."""
    return _compile_spec(
        "GameChangingTranslator_GPU.spec",
        "GPU VERSION",
        allow_environment_mutation,
    )


def _parse_arguments(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-environment-mutation",
        action="store_true",
        help="Allow this script to install PyInstaller in the active interpreter if missing.",
    )
    return parser.parse_args(arguments)


def main(arguments=None):
    """Interactively build an existing spec from the active environment."""
    options = _parse_arguments(arguments)
    script_directory = Path(__file__).resolve().parent
    os.chdir(script_directory)
    print(f"Working directory: {script_directory}")
    print("PyTorch is never uninstalled, upgraded, or replaced by this script.")

    while True:
        print("\n" + "=" * 50)
        print("       Game-Changing Translator Compilation Script")
        print("=" * 50)
        print("\nPlease select the version to compile:")
        print("1. Standard version")
        print("2. GPU spec")
        print("3. Exit")
        try:
            choice = input("\nEnter your choice (1-3): ").strip()
            if choice == "1":
                success = compile_cpu_version(options.allow_environment_mutation)
            elif choice == "2":
                success = compile_gpu_version(options.allow_environment_mutation)
            elif choice == "3":
                print("\nThank you for using the Game-Changing Translator Compilation Script!")
                return
            else:
                print("Invalid choice. Please try again.")
                continue

            if success:
                if input("\nCompile another spec? (y/n): ").strip().lower() not in {"y", "yes"}:
                    return
            elif input("\nTry again? (y/n): ").strip().lower() not in {"y", "yes"}:
                return
        except KeyboardInterrupt:
            print("\n\nOperation cancelled by user.")
            return


if __name__ == "__main__":
    main()
