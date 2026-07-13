# Game-Changing Translator

Copyright (C) 2025-2026 Tomasz Kaminski

This fork was modified with assistance from GPT-5.5.

Current source release: **3.10.2**.

![Game-Changing Translator Logo](docs/screenshots/readme_screen.jpg)

## Overview

Game-Changing Translator is a Windows desktop OCR translation tool for text that cannot be copied directly. Select a source area on screen, choose where the translated overlay should appear, and the app continuously captures, recognizes, translates, caches, and displays the result.

The current project is centered on two active paths:

- **PaddleOCR local OCR** for fast on-device text recognition.
- **Custom AI profiles** for OpenAI-compatible translation endpoints, with optional Custom AI OCR for image-based providers.

Older built-in provider modules and historical resources may still exist in the repository for compatibility and migration context, but new runtime work should treat PaddleOCR plus Custom AI profiles as the supported path. Legacy local OCR support has been removed from the active UI, configuration, dependencies, and runtime routing.

## Core Features

- Region-based screen capture for OCR input and translated overlay output
- Real-time OCR and translation loop with stale-result protection
- Local PaddleOCR backend with configurable language, model size, device, score threshold, and detection limits
- Custom AI OCR image payloads using WebP, PNG, or JPEG
- Custom AI translation through OpenAI-compatible provider profiles
- Unified translation cache to reduce repeated API calls
- Custom prompt support for translation style and terminology
- Floating overlay with configurable font, color, transparency, and geometry
- Hotkeys for controlling translation while another app is focused
- Startup/shutdown paths that preserve settings and overlay placement

## Installation

### Prerequisites

- Windows 10 or Windows 11
- Python 3.9-3.12 when running from source

The bundled application does not require a separate OCR engine installation. `requirements.txt` does not install PaddleOCR. Source users who want local OCR must install the optional PaddleOCR backend separately by following the [official PaddleOCR installation instructions](https://www.paddleocr.ai/latest/en/version3.x/installation.html) for their Python version and hardware, then configure its source directory in Settings. Custom AI OCR does not require PaddleOCR.

### Setup From Source

1. Clone this repository:

   ```bash
   git clone https://github.com/yun-666-666/OCR-Translator.git
   cd OCR-Translator
   ```

2. Install required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Optional local OCR: install the optional PaddleOCR backend separately using the official supported procedure linked above. Skip this step when using only Custom AI OCR.

4. Run the application:

   ```bash
   python main.py
   ```

## Quick Start

1. Launch the application.
2. Select the OCR source area.
3. Select the translation output area.
4. In Settings, choose PaddleOCR for local OCR or configure a Custom AI OCR profile.
5. Configure a Custom AI translation profile for your OpenAI-compatible endpoint.
6. Click **Start** to begin translating.
7. Use the configured hotkey to pause or resume translation while another app is focused.

## Configuration And Secrets

Do not commit real API keys, personal provider profiles, debug logs, runtime caches, or generated local configuration. The runtime config file `ocr_translator_config.ini`, API logs, debug logs, cache files, and local backup folders are intentionally ignored by Git.

Use `ocr_translator_config.example.ini` as a clean publishable template. Custom provider settings belong in Custom AI profiles, and the active translation cache implementation lives in `unified_translation_cache.py`.

## Documentation

- [User Manual](docs/user-manual.html)
- [Installation Guide](docs/installation.html)
- [Troubleshooting](docs/troubleshooting.md)
- [Developer Guide](docs/developer-guide.md)
- [Changelog](CHANGELOG.md)

## Licence

This project is free software, licensed under the GNU General Public Licence version 3 (GPLv3).

You can:

- Use the software for any purpose
- Change the software to suit your needs
- Share the software and your changes with others

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY. See the [LICENSE](LICENSE) file for complete details.

## Acknowledgments

- Modified with assistance from GPT-5.5

## Contributing

Please keep attribution to the original author. Read [ATTRIBUTION.md](ATTRIBUTION.md) before forking or using this code.

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Attribution Required](https://img.shields.io/badge/Attribution-Required-red.svg)](ATTRIBUTION.md)
[![Original Author](https://img.shields.io/badge/Original%20Author-Tomasz%20Kaminski-green.svg)](https://github.com/tomkam1702)
