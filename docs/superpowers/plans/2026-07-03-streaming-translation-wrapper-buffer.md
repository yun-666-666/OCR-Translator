# Streaming Translation Wrapper Buffer Implementation Plan

**Goal:** Filter only known model wrappers from accumulated streaming partials
without adding timer-based latency.

### Task 1: TDD contract

- [x] Add failing preamble-fragment suppression tests.
- [x] Add failing standalone-label and Markdown-fence stream tests.
- [x] Add low-latency ordinary partial and duplicate suppression tests.
- [x] Add preservation tests for inline labels and source-owned wrappers.
- [x] Cover both Chat and Responses streaming paths.
- [x] Run focused tests and confirm RED.

### Task 2: Implementation

- [x] Add a partial-output normalization helper.
- [x] Add a callback wrapper with last-emitted deduplication.
- [x] Hold only unresolved known prefixes.
- [x] Suppress possible closing-fence suffixes.
- [x] Wire the wrapper into stream-mode `translate()`.
- [x] Run focused tests and confirm GREEN.

### Task 3: Full verification

- [x] Run `python -m unittest tests.test_custom_ai -q`.
- [x] Run `python -m unittest tests.test_latency_optimization -q`.
- [x] Run `python -m unittest discover -s tests -q`.
- [x] Run `python -m unittest discover -q`.
- [x] Run `python -m compileall -q custom_ai.py tests`.
- [x] Run `git diff --check`.

### Task 4: Delivery

- [x] Create the required handoff.
- [ ] Stage only the stream-prefix optimization.
- [ ] Commit and push the current branch to GitHub.
