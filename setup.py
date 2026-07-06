from pathlib import Path

from setuptools import setup, find_packages


PROJECT_ROOT = Path(__file__).resolve().parent
EXCLUDED_ROOT_MODULES = {"__init__", "setup"}
ROOT_PY_MODULES = sorted(
    module_path.stem
    for module_path in PROJECT_ROOT.glob("*.py")
    if module_path.stem not in EXCLUDED_ROOT_MODULES
    and not module_path.stem.startswith("test_")
)

setup(
    name="ocr_translator",
    version="1.0.0",
    packages=find_packages(),
    py_modules=ROOT_PY_MODULES,
    install_requires=[
        "numpy>=1.19.0",
        "opencv-python>=4.5.0",
        "Pillow>=8.0.0",
        "pyautogui>=0.9.53",
        "requests>=2.25.0",
        "mss>=9.0.0",
        "PySide6==6.7.3",
        "keyboard>=0.13.5",
        "python-bidi>=0.4.2",
        "arabic-reshaper>=3.0.0",
    ],
    extras_require={
        "keyboard": ["keyboard>=0.13.5"],
        "rtl": ["python-bidi>=0.4.2", "arabic-reshaper>=3.0.0"],
        "all": [
            "keyboard>=0.13.5",
            "python-bidi>=0.4.2",
            "arabic-reshaper>=3.0.0",
        ],
    },
    python_requires=">=3.9,<3.13",
    include_package_data=True,
    package_data={
        "ocr_translator": ["*.csv"],
    },
    entry_points={
        "console_scripts": [
            "ocr_translator=main:main_entry_point",
        ],
    },
    license="GPL-3.0-or-later",
    author="Tomasz Kamiński",
    description="Real-time screen OCR and translation tool",
    long_description=(PROJECT_ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    keywords="ocr, translation, screen-capture, real-time",
    url="https://github.com/yourusername/ocr-translator",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: End Users/Desktop",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Utilities",
        "Topic :: Desktop Environment",
        "Topic :: Text Processing :: Linguistic",
    ],
)
