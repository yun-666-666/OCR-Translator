# API OCR Request Snapshot and Session Isolation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Every behavior change follows RED/GREEN TDD and receives specification plus code-quality review before the next task.

**Goal:** Make each API OCR submission use one immutable request identity so later profile/setting changes cannot alter the provider call or cache key, and make stale callbacks from stopped/restarted sessions unable to mutate the new session.

**Architecture:** Introduce a dependency-light frozen `ApiOcrRequestSnapshot` value object. `run_api_ocr` captures the active profile and every request/cache contract field once, before cache lookup or image encoding. The same snapshot travels through encoding, provider execution, active-call bookkeeping, caching, UI callback, and translation submission. A monotonically increasing application generation qualifies request sequence numbers and is advanced at start, stop, reset, and close boundaries.

**Tech stack:** Python 3.9-3.12, `dataclasses`, `MappingProxyType`, Tk-compatible callback fakes, Pillow test images, `unittest`/`unittest.mock`.

---

## Task 0: Back Up Every Existing Target

**Files:**

- Back up: `app_logic.py`
- Back up: `app_lifecycle.py`
- Back up: `app_capture_ocr.py`
- Back up: `worker_threads.py`
- Back up: `handlers/translation_requests.py`
- Back up: `tests/test_latency_optimization.py`
- Back up: `tests/test_custom_ai.py`

- [ ] Create one timestamped `.codex/backups/YYYY-MM-DD_HH-mm-ss/` tree containing the seven files at their workspace-relative paths.
- [ ] Compare SHA-256 for every source/backup pair and stop unless every value matches.
- [ ] Confirm `git status --short` is clean before implementation.

No backup is required for the new `api_ocr_request.py` module.

## Task 1: Add the Immutable Request Identity

**Files:**

- Create: `api_ocr_request.py`
- Modify: `tests/test_latency_optimization.py`

- [ ] Write failing tests for a frozen `ApiOcrRequestSnapshot` that contains:
  - `generation` and `sequence` with a `(generation, sequence)` request token;
  - provider name and a defensive, read-only copy of the selected OCR profile including the in-memory credential;
  - normalized provider/model/base URL/wire API identity and a non-secret credential scope digest;
  - source language;
  - line-break, latency, reasoning, image-detail, image-format, image-mode, image-quality, and MIME contract;
  - a cache-model identity and cache-mode identity derived only from the frozen fields.
- [ ] Assert later mutations to the original profile do not alter the snapshot, top-level snapshot/profile mutation is rejected, and the API key never appears in `repr(snapshot)`, request token, cache identity, or logs.
- [ ] Implement the frozen dataclass plus small normalization/copy helpers without importing Tk or project application modules.
- [ ] Keep a method such as `profile_copy()` that returns a fresh mutable provider input so provider code cannot mutate the stored snapshot.
- [ ] Run the focused snapshot tests and `tests.test_latency_optimization`.
- [ ] Commit only the new module and its tests as `feat: add immutable API OCR request identity`.

## Task 2: Freeze Profile, Settings, Encoding, and Cache at Submission

**Files:**

- Modify: `app_capture_ocr.py`
- Modify: `worker_threads.py`
- Modify: `handlers/translation_requests.py`
- Modify: `tests/test_latency_optimization.py`
- Modify: `tests/test_custom_ai.py`

- [ ] Add failing integration tests for these races:
  1. Submit under profile A, block execution, switch the manager/UI to profile B, then release. Only A is sent to the provider and only A's snapshot-derived cache key is populated.
  2. Change keep-linebreaks, reasoning, image format/mode/quality/detail, source language, latency mode, base URL/model/wire API, and credential after submission. Encoding, provider arguments, and the cache identity retain the submitted values.
  3. Mutating a provider's received profile does not mutate the snapshot or later cache bookkeeping.
- [ ] Create the snapshot at the start of `run_api_ocr`, before cache lookup and image conversion. Allocate the sequence at this point so cache hits and network submissions share the same identity.
- [ ] Derive the non-secret cache model/mode identities from the snapshot. Do not re-read the active profile or Tk variables later in the pipeline.
- [ ] Extend `convert_to_api_ocr_image` with explicit frozen encoding arguments. Preserve legacy callers, but snapshot-backed calls must not read UI variables. If an encoder fallback would change the frozen format/MIME contract, fail that submission safely instead of silently sending a different contract.
- [ ] Extend `TranslationRequestsMixin.perform_ocr(..., request_snapshot=None)`:
  - snapshot path uses `profile_copy()`, source language, line breaks, latency mode, detail, and MIME from the snapshot;
  - legacy direct callers retain current behavior;
  - no profile object or credential is logged.
- [ ] Pass only the snapshot plus encoded bytes through the thread-pool submission. Remove duplicate provider/source/settings parameters from the snapshot-backed internal path.
- [ ] Run focused API OCR/cache/provider tests, `tests.test_latency_optimization`, and `tests.test_custom_ai`.
- [ ] Commit the five modified files as `fix: freeze API OCR submission contracts`.

## Task 3: Qualify Async Work by Session Generation

**Files:**

- Modify: `app_logic.py`
- Modify: `app_lifecycle.py`
- Modify: `worker_threads.py`
- Modify: `tests/test_latency_optimization.py`

- [ ] Write failing tests proving:
  1. generation advances on start/reset, stop request, cache reset, and close;
  2. an old generation is rejected before the provider call;
  3. a generation changed while the provider is blocked prevents UI callback scheduling;
  4. a queued stale UI callback cannot write cache, display an error, clear/display `<EMPTY>`, alter subtitle/timeout state, or submit translation;
  5. stale cleanup for old `(generation, 1)` cannot remove new `(generation + 1, 1)` from `active_ocr_calls`;
  6. a current-generation cache hit is scheduled on the UI queue and displays normally;
  7. current-generation success/error/empty results preserve existing behavior and sequence ordering.
- [ ] Initialize `ocr_session_generation` in `GameChangingTranslator` and add one helper that advances it, resets sequence/display ordering, and replaces the active-call set at explicit session boundaries.
- [ ] Advance immediately when a stop or close is requested, not only after graceful shutdown, so callbacks landing during shutdown are already stale. Starting/resetting advances again; monotonic extra advances are safe.
- [ ] Store active tokens as `(generation, sequence)`. Network-submit rollback and worker `finally` discard only their own token.
- [ ] Add generation checks at all three required boundaries:
  - before provider execution;
  - after provider return/exception and before `root.after`;
  - at the UI callback before any cache/display/context/timeout/translation mutation.
- [ ] Route cache hits through `root.after(0, ...)` with the same snapshot and generation check instead of directly mutating UI state from the OCR worker.
- [ ] Preserve the existing latest-sequence suppression inside a generation and local PaddleOCR behavior.
- [ ] Run focused session-race tests, `tests.test_latency_optimization`, root lifecycle tests, and the main/root suites.
- [ ] Commit the four modified files as `fix: isolate API OCR callbacks by session generation`.

## Task 4: Increment Verification and Handoff

**Files:**

- Create: `.codex/handoffs/YYYY-MM-DD_HH-mm-ss.md`

- [ ] Run focused suites:

```powershell
py -m unittest tests.test_latency_optimization -q
py -m unittest tests.test_custom_ai -q
```

- [ ] Run both project suites:

```powershell
py -m unittest discover -s tests -q
py -m unittest test_custom_ai test_custom_ai_startup -q
```

- [ ] Compile all tracked root, `handlers/`, and `tests/` Python files from source text.
- [ ] Run `git diff --check`, inspect the increment commit range, and confirm no runtime log/config/cache/backup/generated/Dango path is staged.
- [ ] Create a timestamped handoff listing task summary, exact files, backup path, RED/GREEN evidence, commands/counts, review conclusions, decisions, and remaining external-only validation.
- [ ] Commit the handoff as `docs: hand off API OCR session isolation`.

## Required Review Gates

For Tasks 1-3:

1. A fresh implementer follows the task and reports RED/GREEN evidence.
2. A specification reviewer compares the actual commit with this plan and the approved design.
3. An independent code-quality reviewer checks thread safety, Tk access, mutability, cache identity, secret handling, and stale-callback cleanup.
4. Any Critical or Important finding is fixed by the same implementer, then both reviews are rerun.

## Non-goals

- Do not change local PaddleOCR recognition behavior.
- Do not redesign OCR scheduling, concurrency limits, translation request snapshots, adaptive timeouts, or the UI.
- Do not log profile dictionaries, credentials, request bytes, OCR text, or cache keys containing secrets.
- Do not access, stage, or modify `Dango-Translator-Ver6.3.1/`.
