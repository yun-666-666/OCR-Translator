# Race Loser Request Suppression Implementation Plan

**Goal:** Track running race futures by profile and suppress overlapping loser
requests across consecutive subtitles.

### Task 1: TDD contract

- [x] Add a failing slow-loser / fast-winner lifecycle test.
- [x] Prove a second race excludes the still-running loser.
- [x] Prove the loser automatically becomes eligible after completion.
- [x] Add submit-failure rollback coverage.
- [x] Run focused tests and confirm RED.

### Task 2: Implementation

- [x] Initialize a race in-flight lock and profile identity set.
- [x] Filter busy candidates only when an idle candidate exists.
- [x] Register each submitted profile and attach cleanup callbacks.
- [x] Clear the winning identity before returning.
- [x] Roll back registration on executor submission failure.
- [x] Run focused tests and confirm GREEN.

### Task 3: Full verification

- [x] Run `python -m unittest tests.test_custom_ai -q`.
- [x] Run `python -m unittest tests.test_latency_optimization -q`.
- [x] Run `python -m unittest discover -s tests -q`.
- [x] Run `python -m unittest discover -q`.
- [x] Run `python -m compileall -q handlers tests`.
- [x] Run `git diff --check`.

### Task 4: Delivery

- [x] Create the required handoff.
- [ ] Stage only this race suppression optimization.
- [ ] Commit and push the current branch to GitHub.
