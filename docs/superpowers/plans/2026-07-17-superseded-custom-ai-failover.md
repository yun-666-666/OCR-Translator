# Superseded Custom AI Failover Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop an obsolete live-subtitle request from starting another Custom AI provider after a newer translation request has begun.

**Architecture:** Extend the existing failover obsolescence predicate to combine the scheduler's latest-started sequence with the existing last-displayed sequence. Reuse the two existing failover abort checkpoints; do not add cancellation, locks, or transport changes.

**Tech Stack:** Python 3.12, `unittest`, existing `TranslationHandler` and Custom AI failover tests.

---

### Task 1: Reproduce the superseded failover

**Files:**
- Modify: `tests/test_custom_ai.py:10109-10180`
- Test: `tests/test_custom_ai.py`

- [ ] **Step 1: Add scheduler state to the fixture**

Add `latest_translation_sequence_started=0` to the `SimpleNamespace` returned by `CostProtectedProfileFailoverHandlerTests._make_handler`.

- [ ] **Step 2: Write the failing test**

```python
def test_custom_ai_translation_stops_failover_after_newer_request_starts(self):
    handler, primary, _fallback = self._make_handler()
    try:
        attempted_profiles = []

        def translate(profile, *_args, **_kwargs):
            attempted_profiles.append(profile["id"])
            if profile["id"] == primary["id"]:
                handler.app.latest_translation_sequence_started = 8
                raise ValueError("primary failed after a newer request started")
            return "superseded fallback translation", {}, 0.01

        handler.custom_ai_provider.translate = Mock(side_effect=translate)

        result = handler._custom_ai_translate(
            "source",
            time.monotonic(),
            translation_sequence=7,
        )

        self.assertIsNone(result)
        self.assertEqual(attempted_profiles, [primary["id"]])
    finally:
        handler.close()
```

- [ ] **Step 3: Run the test and verify RED**

Run:

```powershell
python -m unittest tests.test_custom_ai.CostProtectedProfileFailoverHandlerTests.test_custom_ai_translation_stops_failover_after_newer_request_starts -v
```

Expected: FAIL because the fallback profile is attempted and a translation string is returned.

### Task 2: Extend the failover obsolescence predicate

**Files:**
- Modify: `handlers/translation_requests.py:1153-1181`
- Test: `tests/test_custom_ai.py`

- [ ] **Step 1: Implement the minimal predicate**

Read `latest_translation_sequence_started` alongside `last_displayed_translation_sequence` and return true when a positive request sequence is lower than the positive latest-started sequence, or less than or equal to the positive displayed sequence.

```python
return request_sequence > 0 and (
    (latest_started_sequence > 0 and request_sequence < latest_started_sequence)
    or (displayed_sequence > 0 and request_sequence <= displayed_sequence)
)
```

Update the diagnostic line to include both `latest_started` and `last_displayed`; do not include source or translated text.

- [ ] **Step 2: Run the new test and verify GREEN**

Run:

```powershell
python -m unittest tests.test_custom_ai.CostProtectedProfileFailoverHandlerTests.test_custom_ai_translation_stops_failover_after_newer_request_starts -v
```

Expected: PASS with only the primary profile attempted.

- [ ] **Step 3: Run the focused failover class**

Run:

```powershell
python -m unittest tests.test_custom_ai.CostProtectedProfileFailoverHandlerTests -v
```

Expected: all tests pass, including normal fallback, displayed-sequence abort, stopped-app abort, and cache behavior.

### Task 3: Verify the complete change

**Files:**
- Verify: `handlers/translation_requests.py`
- Verify: `tests/test_custom_ai.py`
- Verify: `docs/cleanup/grok-4.5-log-review-2026-07-17.md`

- [ ] **Step 1: Run targeted modules**

```powershell
python -m unittest tests.test_custom_ai -v
python -m unittest tests.test_latency_optimization -v
```

Expected: both modules pass without new warnings or tracebacks.

- [ ] **Step 2: Run repository verification**

```powershell
python -m compileall -q handlers tests
python -m unittest discover -s tests -p "test_*.py"
git diff --check
```

Expected: compilation succeeds, the full suite passes, and `git diff --check` prints nothing.

- [ ] **Step 3: Inspect scope and secret hygiene**

```powershell
git status --short
git diff --stat
git diff -- handlers/translation_requests.py tests/test_custom_ai.py docs/cleanup/grok-4.5-log-review-2026-07-17.md
git check-ignore ocr_translator_config.ini custom_ai_profiles.json translator_debug.log CustomAI_Translation_Short_Log.txt CustomAI_OCR_Short_Log.txt
```

Expected: only the intended source, test, review, design, plan, and handoff files are candidates for staging; runtime configuration, credentials, and logs remain ignored.

- [ ] **Step 4: Commit the verified increment**

```powershell
git add handlers/translation_requests.py tests/test_custom_ai.py docs/cleanup/grok-4.5-log-review-2026-07-17.md docs/superpowers/specs/2026-07-17-superseded-custom-ai-failover-design.md docs/superpowers/plans/2026-07-17-superseded-custom-ai-failover.md
git commit -m "perf: stop superseded Custom AI failover"
```

Expected: one focused local commit with no runtime logs or secrets.

### Self-review

- Spec coverage: the plan covers the new submission-boundary predicate, preserves display obsolescence, avoids in-flight cancellation, and retains normal fallback/cache behavior.
- Placeholder scan: no TODO, TBD, or unspecified implementation steps remain.
- Type consistency: all sequence fields are integer-compatible existing `app` attributes; the test uses the existing `TranslationHandler._custom_ai_translate` API.
