#!/usr/bin/env python3
"""Build the CPU or CUDA-enabled PaddleOCR application with PyInstaller."""

import os
import subprocess
import sys
from pathlib import Path


def run_command(command, description):
    """Run a command without a shell and report captured output."""
    print(f"\n{description}")
    print("-" * len(description))
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print("Success")
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


def verify_paddle_installation(*, require_cuda=False):
    """Verify PaddleOCR imports and optionally require a CUDA-enabled Paddle build."""
    verification_code = f"""
import importlib.util as importlib_util

_real_find_spec = importlib_util.find_spec
def _find_spec_without_optional_torch(name, *args, **kwargs):
    if name == "torch" or name.startswith("torch."):
        return None
    return _real_find_spec(name, *args, **kwargs)

importlib_util.find_spec = _find_spec_without_optional_torch

import paddle
import paddleocr

cuda_enabled = paddle.device.is_compiled_with_cuda()
print(f"Paddle version: {{paddle.__version__}}")
print(f"PaddleOCR version: {{getattr(paddleocr, '__version__', 'unknown')}}")
print(f"CUDA-enabled Paddle build: {{cuda_enabled}}")
if {require_cuda!r} and not cuda_enabled:
    raise SystemExit("GPU build requires a CUDA-enabled paddlepaddle-gpu installation")
if cuda_enabled:
    print(f"Visible CUDA devices: {{paddle.device.cuda.device_count()}}")
"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", verification_code],
            check=True,
            capture_output=True,
            text=True,
        )
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as error:
        print("Paddle environment verification failed.")
        if error.stdout:
            print("STDOUT:", error.stdout)
        if error.stderr:
            print("STDERR:", error.stderr)
        return False


def _build(spec_file):
    return run_command(
        [
            sys.executable,
            "-m",
            "PyInstaller",
            "--clean",
            "--noconfirm",
            spec_file,
        ],
        f"Building with {spec_file}",
    )


def compile_cpu_version():
    """Build against the currently installed CPU-capable Paddle runtime."""
    print("\n" + "=" * 50)
    print("         COMPILING CPU VERSION")
    print("=" * 50)
    if not verify_paddle_installation(require_cuda=False):
        return False
    return _build("GameChangingTranslator.spec")


def compile_gpu_version():
    """Build only when the installed Paddle runtime is CUDA-enabled."""
    print("\n" + "=" * 50)
    print("         COMPILING GPU VERSION")
    print("=" * 50)
    if not verify_paddle_installation(require_cuda=True):
        return False
    return _build("GameChangingTranslator_GPU.spec")


def main():
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    print(f"Working directory: {script_dir}")

    while True:
        print("\n" + "=" * 50)
        print("       Game-Changing Translator Build Script")
        print("=" * 50)
        print("\nPlease select the version to build:")
        print("1. CPU version")
        print("2. GPU version (requires paddlepaddle-gpu)")
        print("3. Exit")

        try:
            choice = input("\nEnter your choice (1-3): ").strip()
            if choice == "1":
                success = compile_cpu_version()
            elif choice == "2":
                success = compile_gpu_version()
            elif choice == "3":
                break
            else:
                print("Invalid choice. Please try again.")
                continue

            prompt = "Build another version? (y/n): " if success else "Try again? (y/n): "
            if input(f"\n{prompt}").strip().lower() not in {"y", "yes"}:
                break
        except KeyboardInterrupt:
            print("\n\nBuild cancelled by user.")
            break


if __name__ == "__main__":
    main()
