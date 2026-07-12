# Large Runtime Module Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split the five largest runtime modules into focused implementation modules while preserving all public behavior and import paths.

**Architecture:** Keep each original module as a compatibility facade and extract cohesive implementation groups behind it. Preserve module-level patch points through explicit dependency seams and verify every target independently before starting the next.

**Tech Stack:** Python 3, Tkinter/ttk, unittest, Pillow, existing OCR/translation providers, Windows Computer Use.

---

### Task 1: Establish the rollback and structural baseline

**Files:**
- Backup: `gui_builder.py`
- Backup: `app_logic.py`
- Backup: `handlers/translation_handler.py`
- Backup: `worker_threads.py`
- Backup: `custom_ai.py`
- Backup: `tests/test_custom_ai.py`
- Backup: `tests/test_latency_optimization.py`
- Backup: any additional existing test file modified during the split

- [ ] **Step 1: Create one timestamped backup tree**

Copy each existing file to `.codex/backups/YYYY-MM-DD_HH-mm-ss/<relative-path>`.

- [ ] **Step 2: Verify every backup**

Run `Get-FileHash -Algorithm SHA256` for every source/backup pair and require
all hashes to match before editing.

- [ ] **Step 3: Record the clean behavioral baseline**

Run: `py -m unittest discover -s tests`

Expected: `Ran 491 tests` and `OK` at the current baseline.

### Task 2: Split `gui_builder.py`

**Files:**
- Create: `gui_profile_controls.py`
- Create: `gui_settings_builder.py`
- Create: `gui_diagnostics_builder.py`
- Modify: `gui_builder.py`
- Modify: `tests/test_custom_ai.py`
- Modify: `tests/test_latency_optimization.py`

- [ ] **Step 1: Write a structural compatibility test**

Add assertions that the extracted modules import and that the functions
re-exported from `gui_builder` retain their public names and call signatures.

- [ ] **Step 2: Run the test in RED state**

Run the new exact unittest cases and require failure because the extracted
modules do not exist yet.

- [ ] **Step 3: Extract profile, settings, and diagnostic builders**

Move code by responsibility. Keep `create_main_tab`, `create_settings_tab`,
`create_debug_tab`, and `create_custom_prompt_tab` import-compatible. Pass
`threading`, `messagebox`, and mutable Tk dependencies explicitly where current
tests patch them through `gui_builder`.

- [ ] **Step 4: Verify GREEN and regression coverage**

Run:

```powershell
py -B -m py_compile gui_builder.py gui_profile_controls.py gui_settings_builder.py gui_diagnostics_builder.py
py -m unittest tests.test_custom_ai tests.test_latency_optimization -q
py -m unittest discover -s tests
```

Expected: all commands exit 0.

- [ ] **Step 5: Real application test**

Launch `py main.py`, use Windows UI automation to switch through the main,
settings, debug, and custom-prompt surfaces, inspect for visible errors, then
close normally.

### Task 3: Split `app_logic.py`

**Files:**
- Create: `app_lifecycle.py`
- Create: `app_capture_ocr.py`
- Create: `app_configuration.py`
- Modify: `app_logic.py`
- Modify: `tests/test_latency_optimization.py`
- Modify: `tests/test_paddle_ocr_backend.py`

- [ ] **Step 1: Write a structural compatibility test**

Assert that `GameChangingTranslator` remains available from `app_logic`, uses
the extracted responsibilities, and retains the tested lifecycle entry points.

- [ ] **Step 2: Run the test in RED state**

Run the new exact unittest case and require failure because the extracted
modules do not exist yet.

- [ ] **Step 3: Extract lifecycle, capture/OCR, and configuration responsibilities**

Compose `GameChangingTranslator` from focused mixins or helpers. Keep test
patches of `app_logic.threading`, `app_logic.time`, provider factories, and
`app_logic.log_debug` effective through explicit dependencies.

- [ ] **Step 4: Verify GREEN and regression coverage**

```powershell
py -B -m py_compile app_logic.py app_lifecycle.py app_capture_ocr.py app_configuration.py
py -m unittest tests.test_latency_optimization tests.test_paddle_ocr_backend -q
py -m unittest discover -s tests
```

Expected: all commands exit 0.

- [ ] **Step 5: Real application test**

Launch the real app, switch tabs, exercise safe non-network lifecycle controls,
confirm responsiveness, and close normally.

### Task 4: Split `handlers/translation_handler.py`

**Files:**
- Create: `handlers/translation_context.py`
- Create: `handlers/translation_requests.py`
- Create: `handlers/translation_results.py`
- Modify: `handlers/translation_handler.py`
- Modify: `tests/test_custom_ai.py`
- Modify: `tests/test_runtime_logging.py`

- [ ] **Step 1: Write a structural compatibility test**

Assert that `TranslationHandler` remains available at the old path and composes
the extracted context, request, and result responsibilities.

- [ ] **Step 2: Run the test in RED state**

Run the new exact case and require module-not-found or missing-composition
failure.

- [ ] **Step 3: Extract cohesive handler responsibilities**

Retain constructor and method signatures. Preserve patchability of
`append_rotating_text`, logging, cache collaborators, and provider classes.

- [ ] **Step 4: Verify GREEN and regression coverage**

```powershell
py -B -m py_compile handlers/translation_handler.py handlers/translation_context.py handlers/translation_requests.py handlers/translation_results.py
py -m unittest tests.test_custom_ai tests.test_runtime_logging -q
py -m unittest discover -s tests
```

Expected: all commands exit 0.

- [ ] **Step 5: Real application test**

Launch the app, inspect translation/profile settings and application
responsiveness without starting a paid request, then close normally.

### Task 5: Split `worker_threads.py`

**Files:**
- Create: `worker_capture.py`
- Create: `worker_ocr.py`
- Create: `worker_translation.py`
- Modify: `worker_threads.py`
- Modify: `tests/test_latency_optimization.py`
- Modify: `tests/test_paddle_ocr_backend.py`

- [ ] **Step 1: Write a structural compatibility test**

Assert that existing worker entry points remain on `worker_threads` and delegate
to extracted capture, OCR, and translation responsibilities.

- [ ] **Step 2: Run the test in RED state**

Run the new exact case and require failure because the worker modules are absent.

- [ ] **Step 3: Extract worker responsibilities**

Keep queue/state ownership and current entry-point signatures unchanged. Pass
patchable dependencies such as `time`, `capture_screen_region`,
`start_async_translation`, logging, and Tk factories through the facade.

- [ ] **Step 4: Verify GREEN and regression coverage**

```powershell
py -B -m py_compile worker_threads.py worker_capture.py worker_ocr.py worker_translation.py
py -m unittest tests.test_latency_optimization tests.test_paddle_ocr_backend -q
py -m unittest discover -s tests
```

Expected: all commands exit 0.

- [ ] **Step 5: Real application test**

Launch the app, exercise available capture/start-stop controls, verify the UI
remains responsive, stop cleanly, and close.

### Task 6: Split `custom_ai.py`

**Files:**
- Create: `custom_ai_policy.py`
- Create: `custom_ai_profiles.py`
- Create: `custom_ai_capabilities.py`
- Create: `custom_ai_requests.py`
- Modify: `custom_ai.py`
- Modify: `tests/test_custom_ai.py`

- [ ] **Step 1: Write a structural compatibility test**

Assert that normalization functions, decision types, `CustomAILatencyModeAdvisor`,
`CustomAIProfileManager`, and `CustomAIProvider` retain their old import path and
are backed by extracted focused modules.

- [ ] **Step 2: Run the test in RED state**

Run the exact new cases and require failure because extracted modules are absent.

- [ ] **Step 3: Extract policy, profile, capability, and request responsibilities**

Preserve existing class APIs and keep patches of `custom_ai.time`,
`custom_ai.log_debug`, credential factories, and network dependencies effective.

- [ ] **Step 4: Verify GREEN and regression coverage**

```powershell
py -B -m py_compile custom_ai.py custom_ai_policy.py custom_ai_profiles.py custom_ai_capabilities.py custom_ai_requests.py
py -m unittest tests.test_custom_ai -q
py -m unittest discover -s tests
```

Expected: all commands exit 0.

- [ ] **Step 5: Real application test**

Launch the app, open Custom AI profile/model controls, exercise safe temporary
form interactions without saving secrets or making a provider call, and close.

### Task 7: Final verification and handoff

**Files:**
- Create: `.codex/handoffs/YYYY-MM-DD_HH-mm-ss.md`

- [ ] **Step 1: Run full verification**

```powershell
py -m unittest discover -s tests
py -m unittest discover
py -m compileall -q app_logic.py gui_builder.py custom_ai.py worker_threads.py handlers tests
git diff --check
```

Expected: every command exits 0 and both unittest discovery modes report `OK`.

- [ ] **Step 2: Inspect scope and file sizes**

Review `git diff --stat`, `git diff`, `git status --short`, and final source line
counts. Confirm no unrelated files were modified.

- [ ] **Step 3: Write the handoff**

Record the task summary, files changed/created, backup location and hash result,
every automated and real-application verification result, decisions, and any
remaining issues.
