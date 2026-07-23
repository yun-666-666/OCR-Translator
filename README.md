# Game-Changing Translator

[中文文档](README_ZH.md) | **English**

> **Original project and author:** [Game-Changing Translator](https://github.com/tomkam1702/OCR-Translator) by [Tomasz Kamiński](https://github.com/tomkam1702). This repository is a modified fork; the original author’s copyright and attribution are retained.

![Game-Changing Translator Logo](docs/screenshots/readme_screen.jpg)

Game-Changing Translator is a Windows desktop OCR translation tool for text that cannot be copied directly. Select a source area on screen, choose where the translated overlay should appear, and the app continuously captures, recognizes, translates, caches, and displays the result.

## Overview

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

The bundled application does not require a separate OCR engine installation. Source users should install the Python dependencies and configure PaddleOCR or Custom AI from the app settings.

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

3. Run the application:

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

## Configuration and Privacy

Do not commit real API keys, personal provider profiles, debug logs, runtime caches, or generated local configuration. The runtime config file `ocr_translator_config.ini`, API logs, debug logs, cache files, and local backup folders are intentionally ignored by Git.

Use `ocr_translator_config.example.ini` as a clean publishable template. Custom provider settings belong in Custom AI profiles, and the active translation cache implementation lives in `unified_translation_cache.py`.

## Documentation

- [Chinese documentation](README_ZH.md)
- [User Manual](docs/user-manual.html)
- [Installation Guide](docs/installation.html)
- [Troubleshooting](docs/troubleshooting.md)
- [Developer Guide](docs/developer-guide.md)
- [Changelog](CHANGELOG.md)

## Licence

This modified fork is licensed under the GNU General Public License, version 3 or later (GPL-3.0-or-later). It retains the original project’s copyright notice and GPL licence. See [LICENSE](LICENSE) for the complete licence text and third-party notices.

If you distribute a modified version or binaries, comply with the GPL and retain the original project attribution described in [ATTRIBUTION.md](ATTRIBUTION.md).

## Acknowledgments

- **GPT** — primary AI collaborator for this fork’s development work.
- **Claude** and **Grok** — collaborating AI assistants for implementation, review, and iteration.
- **Tomasz Kamiński** — original author and maintainer of Game-Changing Translator; this fork builds on his original work.

See [CONTRIBUTORS.md](CONTRIBUTORS.md) for the maintained credit record.

## Contributing

Contributions are welcome. Before opening a pull request, please read [CONTRIBUTING.md](CONTRIBUTING.md) and preserve all copyright, licence, and original-author notices.

This fork is based on [Game-Changing Translator by Tomasz Kamiński](https://github.com/tomkam1702/OCR-Translator). Please keep this source acknowledgement and the original author attribution in derivative work.

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Attribution Required](https://img.shields.io/badge/Attribution-Required-red.svg)](ATTRIBUTION.md)
[![Original Author](https://img.shields.io/badge/Original%20Author-Tomasz%20Kamiński-green.svg)](https://github.com/tomkam1702)
