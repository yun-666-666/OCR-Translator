# Transient Translation Request Recovery Implementation Plan

**Goal:** Add one same-endpoint recovery attempt for narrowly defined
non-streaming transient failures.

### Task 1: TDD contract

- [x] Add failing Chat Completions 502-then-success coverage.
- [x] Add failing Responses API 504-then-success coverage.
- [x] Add failing owned-session TLS reset recovery coverage.
- [x] Add no-retry coverage for `none` mode and `Retry-After`.
- [x] Update cached endpoint fallback expectations to include one exhausted
  same-URL retry.
- [x] Run focused tests and confirm RED.

### Task 2: Recovery implementation

- [x] Classify retryable transient responses and exceptions narrowly.
- [x] Send one same-URL retry in optimized non-streaming modes.
- [x] Recreate an owned session after a retryable transport reset.
- [x] Preserve duration, capability fallback, URL memory, cooldown, and error
  sanitation behavior.
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
- [ ] Stage only the transient-recovery optimization.
- [ ] Commit and push the current branch to GitHub.
