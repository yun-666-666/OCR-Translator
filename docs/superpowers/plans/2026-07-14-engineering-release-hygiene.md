# Engineering and Release Hygiene Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make offline verification, user-file persistence, packaging, build guidance, and hot-path diagnostics reproducible and safe before the final project push.

**Architecture:** Keep runtime behavior intact while adding small explicit boundaries: one test runner invokes both existing unittest roots in isolated log directories; user-owned prompt writes share the existing same-directory atomic-write discipline; `resources` becomes a data package inspected from a wheel; and build tooling becomes opt-in/non-destructive. Logging changes only suppress proven no-op repetitions.

**Tech Stack:** Python 3.9-3.12, `unittest`, `subprocess`, `tempfile`, `setuptools`, GitHub Actions Windows runners, `os.replace` and `fsync`.

---

## Task 0: Back up each existing target before edits

**Back up:** `app_capture_ocr.py`, `app_configuration.py`, `compile_app.py`, `docs/developer-guide.md`, `README.md`, `setup.py`, `handlers/translation_requests.py`, and all existing test files modified by this increment. Create a timestamped workspace-relative backup tree and SHA-256 verify every pair. New scripts, workflow, tests, and `resources/__init__.py` require no backup.

## Task 1: Provide one isolated offline test entrypoint and Windows CI

**Files:** Create `scripts/run_offline_tests.py`, `.github/workflows/offline-tests.yml`; modify/add focused test coverage as needed.

- [ ] Write a failing subprocess test that runs the entrypoint from a clean temporary log directory and asserts it invokes both `py -m unittest discover -s tests -q` and `py -m unittest test_custom_ai test_custom_ai_startup -q`, fails on either nonzero result, and does not append to the caller's normal log directory.
- [ ] Implement the runner using `sys.executable`, argument lists, `subprocess.run`, and a temporary `OCR_TRANSLATOR_LOG_DIR`; include its own command/result summary but never contacts a provider.
- [ ] Add a Windows GitHub Actions matrix for Python 3.9-3.12, checkout, dependency installation from `requirements.txt`, and `py scripts/run_offline_tests.py`. It must have no secrets and no real-provider calls.
- [ ] Run the runner locally and verify its reported combined count is at least 620 before later tests are added.

## Task 2: Make prompt/config recovery writes atomic and preserve corrupt config

**Files:** Modify `app_configuration.py`, `config_manager.py`; create a small shared atomic-text helper only if it removes duplication; modify/add `tests/test_custom_ai.py` and/or a focused new test module.

- [ ] Write failing tests that inject `os.replace`/write failure while saving a custom prompt and assert old bytes plus in-memory prompt are preserved; successful writes replace content without temporary-file residue.
- [ ] Refactor prompt initialization/save to write a same-directory unique temporary file, flush, `fsync`, and `os.replace`; publish `custom_prompt_text` only after success. Keep missing/empty prompt default behavior and UTF-8-SIG encoding.
- [ ] Write a failing corrupt-INI test: parsing failure preserves original bytes as a timestamped sibling `.corrupt-*` copy before defaults are published, then a valid default config is atomically saved. Normal valid config behavior is unchanged.
- [ ] Implement only the recovery/backup branch in `load_app_config`; never delete/corrupt the source and log only paths/errors without config content or secrets.
- [ ] Run focused prompt/config tests and existing credential/config tests.

## Task 3: Verify resources in a wheel and make build guidance non-destructive

**Files:** Create `resources/__init__.py`; modify `setup.py`, `compile_app.py`, `docs/developer-guide.md`, `README.md`, `tests/test_setup_entrypoint.py`; create a wheel-inspection test if required.

- [ ] Write a failing wheel test using `python -m build --wheel --outdir <temporary>` (or setuptools fallback already available) that opens the wheel zip and requires runtime root modules plus all `resources/*.csv` files.
- [ ] Add `resources/__init__.py`, declare the resources package/data correctly in `setup.py`, and keep flat runtime imports unchanged.
- [ ] Rewrite `compile_app.py` so it invokes all commands as arrays via `sys.executable -m pip` / `sys.executable -m PyInstaller`, never calls `shell=True`, and refuses package-changing operations unless an explicit `--allow-environment-mutation` flag is supplied. The default build path must not uninstall, upgrade, or replace PyTorch.
- [ ] Add unit tests that mock subprocess and prove the exact active interpreter/argument arrays and mutation guard.
- [ ] Update README/developer guide: `requirements.txt` does not install PaddleOCR; document Paddle as an explicit optional backend installation step, state version `3.10.2`, describe the existing spec files and isolated environment workflow only.

## Task 4: Bound diagnosed log noise and hand off

**Files:** Modify `app_capture_ocr.py`, `handlers/translation_requests.py`; modify/add focused tests.

- [ ] Write failing tests proving `reset_clear_timeout` emits no log when the timer is already inactive and emits one log only on an active-to-inactive transition.
- [ ] Change `reset_clear_timeout` to retain state behavior but log only that transition; do not remove meaningful expiration or subtitle-change logs.
- [ ] Replace repeated failover cooling-profile debug output with `log_debug_coalesced`, using a non-secret profile identity/event key and a bounded interval. Preserve each first skip and cooldown decision behavior.
- [ ] Run focused logging/latency tests, the unified offline runner, both direct suites, compile, wheel inspection, `git diff --check`, and write a timestamped handoff.

## Required review gates

For each implementation task: a fresh implementer supplies RED/GREEN evidence, then a specification review and an independent quality review must have no Critical or Important finding before the next task. Do not touch the excluded `Dango-Translator-Ver6.3.1/` directory or generated outputs.
