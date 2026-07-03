# Structured Translation I/O Contract Implementation Plan

**Goal:** Make translation input boundaries machine-readable and normalize only
clear model-output wrappers before cache/display.

### Task 1: TDD contract

- [x] Add failing JSON prompt structure and exact round-trip tests.
- [x] Add failing fenced-output and standalone-label cleanup tests.
- [x] Add failing Responses/stream final-output cleanup tests.
- [x] Add preservation tests for inline labels, quotes, and fenced source text.
- [x] Run focused tests and confirm RED.

### Task 2: Implementation

- [x] Serialize current source and paired history as compact JSON.
- [x] Update the system contract for JSON input semantics.
- [x] Add conservative translation-output normalization.
- [x] Apply normalization only in `translate()`.
- [x] Reject empty normalized output.
- [x] Run focused tests and confirm GREEN.

### Task 3: Regression verification

- [x] Review the root compatibility prompt assertion (no format-specific
  assertion required a change).
- [x] Run `python -m unittest tests.test_custom_ai -q`.
- [x] Run `python -m unittest tests.test_latency_optimization -q`.
- [x] Run `python -m unittest discover -s tests -q`.
- [x] Run `python -m unittest discover -q`.
- [x] Run `python -m compileall -q custom_ai.py test_custom_ai.py tests`.
- [x] Run `git diff --check`.

### Task 4: Delivery

- [x] Create the required handoff.
- [ ] Stage only this quality-contract optimization.
- [ ] Commit and push the current branch to GitHub.
