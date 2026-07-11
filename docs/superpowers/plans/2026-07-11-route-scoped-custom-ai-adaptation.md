# Route-scoped Custom AI Adaptation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Isolate Custom AI latency and prompt-cache adaptation by canonical provider route.

**Architecture:** `TranslationHandler` will own bounded LRU route-state maps keyed by canonical endpoint, credential scope, wire API, and model. Request resolution, completion observations, error observations, and context-budget decisions will all use the frozen request profile.

**Tech Stack:** Python 3.9-3.12, `unittest`, existing Custom AI provider and translation handler.

---

### Task 1: Add route-isolation regressions

**Files:**
- Modify: `tests/test_custom_ai.py`

- [ ] Add a test proving slow latency samples on route A do not change route B.
- [ ] Add a test proving prompt-cache EMA on route A does not shrink route B's context budget.
- [ ] Add a test proving a frozen snapshot completion records route A after the active profile changes to B.
- [ ] Add a test proving the route state maps retain at most 32 LRU entries.
- [ ] Run the focused tests and confirm they fail for cross-route sharing or missing route-state APIs.

### Task 2: Implement bounded route state

**Files:**
- Modify: `handlers/translation_handler.py`

- [ ] Add canonical route-key and bounded LRU helpers.
- [ ] Resolve and commit adaptive latency mode through the route advisor.
- [ ] Record success and failure against the request profile.
- [ ] Record and read prompt-cache EMA through the route metrics record.
- [ ] Pass the profile into context-budget construction.
- [ ] Run the focused tests and confirm they pass.

### Task 3: Verify compatibility

**Files:**
- Modify only if a regression requires a scoped correction.

- [ ] Run `py -m unittest tests.test_custom_ai tests.test_latency_optimization -q`.
- [ ] Run `py -m unittest discover -s tests`.
- [ ] Run `py -m unittest discover`.
- [ ] Run Python compile checks and `git diff --check`.
- [ ] Request an independent read-only review and resolve all Critical or Important findings.
