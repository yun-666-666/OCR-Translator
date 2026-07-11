# Transactional Custom AI Profiles Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Custom AI profile JSON and credential changes atomic and recoverable.

**Architecture:** The manager will atomically replace its JSON file and wrap each CRUD mutation in a small compensation transaction. Credential deletion occurs only after profile removal is durably committed.

**Tech Stack:** Python standard library, Windows-compatible `os.replace`, existing credential abstraction, `unittest`.

---

### Task 1: Add failure regressions

**Files:**
- Modify: `tests/test_custom_ai.py`

- [ ] Reproduce target replace failure without corrupting the prior file.
- [ ] Reproduce add save failure and assert profile/credential rollback.
- [ ] Reproduce update save failure and assert profile/credential rollback.
- [ ] Reproduce delete save failure and assert credential is retained.
- [ ] Run focused tests and confirm RED.

### Task 2: Implement atomic save and CRUD compensation

**Files:**
- Modify: `custom_ai.py`

- [ ] Write, flush, fsync, and atomically replace from a unique sibling temp.
- [ ] Clean temporary files on all failure paths.
- [ ] Make all CRUD methods raise when durable persistence fails.
- [ ] Restore prior memory and credential values after add/update failure.
- [ ] Rotate API keys through a versioned credential reference and retire the
      old reference only after JSON commit.
- [ ] Persist deletion before best-effort credential cleanup.
- [ ] Ignore and narrowly clean stale profile atomic temporary files.
- [ ] Run focused tests and confirm GREEN.

### Task 3: Verify the profile contract

- [ ] Run all profile-manager and Custom AI tests.
- [ ] Run the full repository test suites and compile checks.
- [ ] Confirm JSON and diagnostics do not expose real credentials.
- [ ] Obtain independent read-only review before release.
