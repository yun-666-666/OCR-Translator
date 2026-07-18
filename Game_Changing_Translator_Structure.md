# Game-Changing Translator - File Structure Reference

> **ARCHIVED / LEGACY DOCUMENT** — The inventory below originally described Gemini/OpenAI/DeepL/MarianMT packaging. Prefer the current live-path contract in `tests/test_live_path_inventory.py` and the developer live-path note in [`docs/developer-guide.md`](docs/developer-guide.md).

> **For detailed development information, see [`docs/developer-guide.md`](docs/developer-guide.md)**

This document provides a quick reference to the application's file organization and architecture.

## Architecture Overview (current live path)

```
Entry Point (main.py)
    ↓
Central Coordinator (app_logic.py)
    ↓
┌─────────────┬─────────────┬─────────────┬─────────────┐
│   Handlers  │   Workers   │   Utils     │    UI       │
│   (modular) │ (threads)   │ (helpers)   │ (interface) │
└─────────────┴─────────────┴─────────────┴─────────────┘
    ↓
PaddleOCR / Custom AI OCR → Unified Cache → Custom AI Translation → Overlay
```

## Key Features (live path)

### Custom AI translation and OCR
- Profile-based OpenAI-compatible endpoints for translation and API OCR
- Credential store for API keys (no plaintext fallback on store failure)
- Unified short logs and usage reporting (missing usage stays `n/a`, not `$0`)

### Local PaddleOCR
- Packaged CPU/GPU builds validate matching Paddle runtime
- OCR preview and capture pipeline remain on the live path

### Unified Caching System
- In-memory LRU cache + optional persistence for Custom AI translation
- Thread-safe operations with proper locking mechanisms

### Overlay display
- PySide6 RTL-capable translation overlays with tkinter fallback

## Core Application Files (live path)

### Entry Point & Coordination
- **`main.py`** - Application entry point
- **`app_logic.py`** - Central coordinator and main application class

### Modular Handlers (`handlers/`)
- **`configuration_handler.py`** - Settings and configuration management
- **`display_manager.py`** - UI updates for overlays and debug info
- **`hotkey_handler.py`** - Keyboard shortcut management
- **`translation_handler.py`** - Custom AI translation/OCR coordination
- **`translation_context.py`** / **`translation_requests.py`** / **`translation_results.py`** - Live request path; results module keeps **inert archived** legacy adapters only
- **`ui_interaction_handler.py`** - User interface interaction management

### Core Processing
- **`worker_threads.py`** / **`worker_capture.py`** / **`worker_ocr.py`** / **`worker_translation.py`** - Background pipeline
- **`unified_translation_cache.py`** - LRU cache for Custom AI translation
- **`paddle_ocr_backend.py`** - Local OCR backend
- **`custom_ai*.py`** - Custom AI profiles, transport, requests

### Utilities
- **`ocr_utils.py`** - OCR processing and text extraction
- **`translation_utils.py`** - Translation helper functions
- **`language_manager.py`** - Language code mappings (Custom AI lists)
- **`config_manager.py`** - App-data configuration path with atomic saves
- **`resource_handler.py`** / **`resource_copier.py`** - Resource path resolution for builds
- **`logger.py`** - Application logging

### User Interface
- **`gui_builder.py`** / **`gui_settings_builder.py`** - UI construction
- **`language_ui.py`** - Multi-language interface support
- **`overlay_manager.py`** - Source/target overlay window management
- **`pyside_overlay.py`** - PySide6 RTL translation overlays
- **`ui_elements.py`** - Custom UI components
- **`constants.py`** - Application constants

## Archived inventory (removed from live tree)

The following were part of older product surfaces and must not be re-added to packaging without an explicit decision:

- `marian_mt_translator.py`, `convert_marian.py`
- `handlers/gemini_*`, `handlers/openai_*`, `handlers/ocr_provider_base.py`, `handlers/llm_provider_base.py`
- `handlers/cache_manager.py`, `handlers/statistics_handler.py`
- Provider-only CSVs under `resources/` for DeepL / Gemini / OpenAI / MarianMT
- PyInstaller hidden imports for `deepl`, `google.genai`, `torch`/`transformers` Marian stacks, etc.

## Build & Configuration

- **`GameChangingTranslator.spec`** / **`GameChangingTranslator_GPU.spec`** - Live-path PyInstaller specs (no legacy provider deps)
- **`compile_app.py`** - Validates Paddle runtime without mutating dependencies
- **`ocr_translator_config.example.ini`** - Example settings; runtime config resolves to app-data with optional CWD migration
