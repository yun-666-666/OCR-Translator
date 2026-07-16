# Hidden Legacy Controls Removal Design

## Goal

Remove the hidden Google, DeepL, Gemini-special, OpenAI-special, and MarianMT settings controls together with the UI/config compatibility state that exists only to construct, hide, and save those controls.

## Chosen scope

This is a bounded live-path cleanup. It removes legacy widget construction, visibility callbacks, model-selection branches, settings persistence, Marian model browsing, and obsolete example/default configuration from the active application seam. PaddleOCR, Custom AI OCR, Custom AI translation profiles, overlays, runtime metrics, and user profile/credential files remain unchanged.

Dedicated legacy provider implementation files and their historical resources remain for a later physical-removal increment. Keeping that deletion separate avoids repeating the prior broad cleanup that broke startup initialization.

## Architecture

- `app_logic.py` initializes only live-path UI variables.
- `gui_settings_builder.py` builds only PaddleOCR and Custom AI settings.
- `handlers/ui_interaction_handler.py` handles only live-path selection, visibility, language state, and persistence.
- `app_configuration.py` and `handlers/configuration_handler.py` expose only callbacks still reachable from the live UI.
- `config_manager.py` migrates legacy selections while dropping obsolete provider settings.
- `ocr_translator_config.example.ini` documents the live path only.

## Safety and testing

A source-level regression test first proves that legacy controls and persisted keys still exist. After implementation it must prove their absence while retaining Custom AI and PaddleOCR controls. Existing startup, PaddleOCR, Custom AI, latency, and UI suites then guard initialization order and live behavior.

