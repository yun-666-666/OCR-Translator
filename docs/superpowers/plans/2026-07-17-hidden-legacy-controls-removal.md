# Hidden Legacy Controls Removal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the hidden legacy Provider settings/UI compatibility layer without changing the PaddleOCR and Custom AI live path.

**Architecture:** Prune the active UI/config seam as one bounded change while leaving physical provider-module deletion for a later increment. Protect startup ordering and live controls with a source-level regression plus existing runtime suites.

**Tech Stack:** Python 3, Tkinter, unittest, Git, PowerShell

---

### Task 1: Add the cleanup contract

**Files:**
- Modify: `tests/test_custom_ai_startup.py`

- [ ] Add a test that rejects legacy widget construction, visibility callbacks, and legacy provider settings writes while requiring Custom AI/PaddleOCR controls.
- [ ] Run the targeted test and confirm it fails on the existing legacy symbols.

### Task 2: Remove the active legacy UI/config seam

**Files:**
- Modify: `app_logic.py`
- Modify: `app_lifecycle.py`
- Modify: `gui_settings_builder.py`
- Modify: `handlers/ui_interaction_handler.py`
- Delete: `handlers/cache_manager.py`
- Modify: `handlers/__init__.py`
- Modify: `handlers/translation_results.py`
- Modify: `language_manager.py`
- Modify: `app_configuration.py`
- Modify: `handlers/configuration_handler.py`
- Modify: `config_manager.py`
- Modify: `ocr_translator_config.example.ini`

- [ ] Remove legacy widget creation and callbacks.
- [ ] Remove legacy provider UI state, Marian model initialization, and legacy settings persistence.
- [ ] Keep translation and OCR migration on the live path.
- [ ] Drop obsolete legacy provider defaults and example keys without touching user config or credentials.
- [ ] Run the targeted test and confirm it passes.

### Task 3: Verify and hand off

**Files:**
- Create: `.codex/handoffs/<timestamp>.md`

- [ ] Run focused startup/UI/config tests.
- [ ] Run broader Custom AI, PaddleOCR, latency, and UI regression suites.
- [ ] Run tracked-source compile checks and `git diff --check`.
- [ ] Perform a startup smoke that constructs and destroys `GameChangingTranslator`.
- [ ] Record backup, commands, results, decisions, and remaining physical-provider cleanup.
- [ ] Stage only intended files and create a local commit on `codex/remove-hidden-legacy-controls-20260716`.
