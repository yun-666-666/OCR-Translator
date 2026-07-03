# Race Profile Health and Semantics Implementation Plan

**Goal:** Restrict race mode to healthy semantically equivalent profiles and
allow a healthy alternative to bypass active-profile cooldown.

### Task 1: TDD contract

- [x] Add failing candidate-signature isolation tests.
- [x] Add failing cooling-active / healthy-alternative scheduler test.
- [x] Add failing single healthy alternative execution test.
- [x] Add all-cooling minimum cooldown and non-race compatibility tests.
- [x] Run focused tests and confirm RED.

### Task 2: Implementation

- [x] Add a normalized race semantic signature.
- [x] Filter model/wire API/reasoning mismatches.
- [x] Filter cooling candidates when healthy alternatives exist.
- [x] Use the remaining candidate in the single-candidate path.
- [x] Return minimum compatible cooldown in race mode.
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
- [ ] Stage only this race optimization.
- [ ] Commit and push the current branch to GitHub.
