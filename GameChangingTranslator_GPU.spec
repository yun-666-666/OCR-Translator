# -*- mode: python ; coding: utf-8 -*-

"""PyInstaller specification for an installed CUDA-enabled Paddle runtime.

The installed Paddle distribution determines CPU or CUDA execution. Build with:
python -m PyInstaller --clean --noconfirm GameChangingTranslator_GPU.spec
"""

import os
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules


current_dir = os.path.dirname(os.path.abspath("__file__"))
rapidocr_datas = collect_data_files("rapidocr", includes=["*.yaml"])
onnxruntime_binaries = collect_dynamic_libs("onnxruntime")
rapidocr_optional_engine_prefixes = (
    "rapidocr.inference_engine.mnn",
    "rapidocr.inference_engine.openvino",
    "rapidocr.inference_engine.paddle",
    "rapidocr.inference_engine.py" + "to" + "rch",
    "rapidocr.inference_engine.tensorrt",
)
rapidocr_hidden_imports = collect_submodules(
    "rapidocr",
    filter=lambda name: not name.startswith(rapidocr_optional_engine_prefixes),
)

live_hidden_imports = [
    "app_logic",
    "app_capture_ocr",
    "app_configuration",
    "app_lifecycle",
    "config_manager",
    "constants",
    "credential_store",
    "custom_ai",
    "custom_ai_capabilities",
    "custom_ai_policy",
    "custom_ai_profiles",
    "custom_ai_requests",
    "custom_ai_transport",
    "gui_builder",
    "gui_settings_builder",
    "handlers.configuration_handler",
    "handlers.display_manager",
    "handlers.hotkey_handler",
    "handlers.translation_context",
    "handlers.translation_handler",
    "handlers.translation_requests",
    "handlers.translation_results",
    "handlers.ui_interaction_handler",
    "language_manager",
    "language_ui",
    "logger",
    "main",
    "mss",
    "ocr_utils",
    "overlay_manager",
    "paddle",
    "paddle.base.core",
    "paddle.device.cuda",
    "paddleocr",
    "paddlex",
    "paddle_ocr_backend",
    "rapid_ocr_backend",
    "rapidocr",
    "onnxruntime",
    "pyside_overlay",
    "resource_copier",
    "resource_handler",
    "rtl_text_processor",
    "translation_utils",
    "ui_elements",
    "unified_translation_cache",
    "worker_capture",
    "worker_ocr",
    "worker_threads",
    "worker_translation",
    "PySide6.QtCore",
    "PySide6.QtGui",
    "PySide6.QtWidgets",
    "PIL.Image",
    "PIL.ImageTk",
    "arabic_reshaper",
    "bidi.algorithm",
    "cv2",
    "keyboard",
    "numpy",
    "requests",
    "tkinter",
    "tkinter.colorchooser",
    "tkinter.filedialog",
    "tkinter.messagebox",
    "tkinter.ttk",
]
live_hidden_imports += rapidocr_hidden_imports

a = Analysis(
    ["main.py"],
    pathex=[current_dir],
    binaries=onnxruntime_binaries,
    datas=[
        ("resources", "resources"),
        ("docs/user-manual.html", "docs"),
        ("docs/CHANGELOG.md", "docs"),
        ("README.md", "."),
        ("LICENSE", "."),
    ] + rapidocr_datas,
    hiddenimports=live_hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "django",
        "fastapi",
        "flask",
        "jupyter",
        "matplotlib",
        "notebook",
        "PyQt5",
        "PyQt6",
        "PySide2",
        "pytest",
        "sklearn",
        "sentencepiece",
        "tensorflow",
        "torch",
        "torchaudio",
        "torchvision",
        "transformers",
    ],
    optimize=1,
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="GameChangingTranslator",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="GameChangingTranslator",
)
