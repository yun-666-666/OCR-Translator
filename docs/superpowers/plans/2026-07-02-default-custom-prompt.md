# Default Custom Prompt Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the stock custom translation prompt with the approved shorter natural-subtitle instruction and keep the runtime fallback, shipped file, and both test entry points synchronized.

**Architecture:** `app_logic.DEFAULT_CUSTOM_PROMPT` remains the single runtime fallback, while `custom_prompt.txt` remains the editable shipped/current value. Exact-value tests in both startup test modules prevent the two independent discovery paths from accepting a stale prompt.

**Tech Stack:** Python 3, `unittest`, `pathlib.Path`

---

### Task 1: Specify the optimized default in both startup suites

**Files:**
- Modify: `test_custom_ai_startup.py:104-115`
- Modify: `tests/test_custom_ai_startup.py:110-121`
- Test: `test_custom_ai_startup.py`
- Test: `tests/test_custom_ai_startup.py`

- [ ] **Step 1: Strengthen the root startup test**

Replace the loose `"subtitle translator"` assertion with the exact approved prompt and a concise-length guard:

```python
def test_missing_custom_prompt_loads_optimized_default_prompt(self):
    import app_logic

    expected_prompt = (
        "Use context to resolve ambiguity. Translate naturally and concisely while preserving meaning, "
        "tone, and character voice. Keep names and game terms consistent."
    )
    with tempfile.TemporaryDirectory() as tmp_dir:
        prompt_path = Path(tmp_dir) / "custom_prompt.txt"
        dummy_app = types.SimpleNamespace(custom_prompt_file=str(prompt_path), custom_prompt_text="")

        app_logic.GameChangingTranslator.load_custom_prompt(dummy_app)

        self.assertTrue(prompt_path.exists())
        self.assertEqual(dummy_app.custom_prompt_text, expected_prompt)
        self.assertLess(len(dummy_app.custom_prompt_text), 180)
        self.assertEqual(prompt_path.read_text(encoding="utf-8-sig"), expected_prompt)
```

- [ ] **Step 2: Apply the same contract to the `tests` startup test**

Use the identical method body and exact `expected_prompt` value in `tests/test_custom_ai_startup.py`.

- [ ] **Step 3: Add a shipped-file synchronization test**

Add this test to both startup modules:

```python
def test_shipped_custom_prompt_matches_runtime_default(self):
    import app_logic

    expected_prompt = (
        "Use context to resolve ambiguity. Translate naturally and concisely while preserving meaning, "
        "tone, and character voice. Keep names and game terms consistent."
    )
    shipped_prompt = Path("custom_prompt.txt").read_text(encoding="utf-8-sig")

    self.assertEqual(app_logic.DEFAULT_CUSTOM_PROMPT, expected_prompt)
    self.assertEqual(shipped_prompt, expected_prompt)
```

- [ ] **Step 4: Run the focused tests and verify RED**

Run:

```powershell
python -m unittest `
  test_custom_ai_startup.StartupOptimizationTests.test_missing_custom_prompt_loads_optimized_default_prompt `
  test_custom_ai_startup.StartupOptimizationTests.test_shipped_custom_prompt_matches_runtime_default `
  tests.test_custom_ai_startup.StartupOptimizationTests.test_missing_custom_prompt_loads_optimized_default_prompt `
  tests.test_custom_ai_startup.StartupOptimizationTests.test_shipped_custom_prompt_matches_runtime_default -v
```

Expected: failures showing the current long stock prompt differs from the approved exact value.

### Task 2: Update the runtime and shipped defaults

**Files:**
- Modify: `app_logic.py:53-57`
- Modify: `custom_prompt.txt`
- Test: `test_custom_ai_startup.py`
- Test: `tests/test_custom_ai_startup.py`

- [ ] **Step 1: Replace the runtime constant**

Set:

```python
DEFAULT_CUSTOM_PROMPT = (
    "Use context to resolve ambiguity. Translate naturally and concisely while preserving meaning, "
    "tone, and character voice. Keep names and game terms consistent."
)
```

- [ ] **Step 2: Replace the shipped/current prompt**

Make `custom_prompt.txt` contain exactly:

```text
Use context to resolve ambiguity. Translate naturally and concisely while preserving meaning, tone, and character voice. Keep names and game terms consistent.
```

Do not add labels, examples, Markdown, or output-format rules already supplied by `custom_ai.py`.

- [ ] **Step 3: Run the focused tests and verify GREEN**

Run the four-test command from Task 1.

Expected: `OK`, 4 tests.

### Task 3: Verify payload composition and regressions

**Files:**
- Verify: `custom_ai.py`
- Verify: `app_logic.py`
- Verify: `custom_prompt.txt`
- Verify: `test_custom_ai_startup.py`
- Verify: `tests/test_custom_ai_startup.py`

- [ ] **Step 1: Inspect one generated payload**

Run:

```powershell
python -c "import app_logic; from custom_ai import CustomAIProvider; p=CustomAIProvider().build_translation_payload({'model':'demo'},'Hello','en','zh-CN',custom_prompt=app_logic.DEFAULT_CUSTOM_PROMPT,context=['Previous line'],keep_linebreaks=False); s=p['messages'][0]['content']; print({'custom_prompt_count':s.count(app_logic.DEFAULT_CUSTOM_PROMPT),'has_output_constraint':'Return only the translation' in s,'prompt_chars':len(app_logic.DEFAULT_CUSTOM_PROMPT)})"
```

Expected:

```text
{'custom_prompt_count': 1, 'has_output_constraint': True, 'prompt_chars': 158}
```

- [ ] **Step 2: Run both startup modules**

Run:

```powershell
python -m unittest test_custom_ai_startup tests.test_custom_ai_startup -v
```

Expected: all startup tests pass.

- [ ] **Step 3: Run complete project verification**

Run:

```powershell
python -m unittest discover -s tests
python -m unittest discover
python -m compileall -q app_logic.py test_custom_ai_startup.py tests\test_custom_ai_startup.py
```

Expected: both discovery commands report `OK`; compilation emits no errors.

- [ ] **Step 4: Record the completed increment**

Write `.codex/handoffs/YYYY-MM-DD_HH-mm-ss.md` with:

- task summary;
- exact files changed;
- backup path `.codex/backups/2026-07-02_13-42-56/`;
- approved prompt text and non-migration decision;
- red/green and full-suite results;
- payload inspection result;
- current status and next-read order.

This workspace has no `.git` directory, so no commit step is available.
