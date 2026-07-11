# Settings Wheel and Model Cooldown Isolation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep Settings inputs unchanged during page-wheel scrolling and isolate relay gateway/capacity cooldown by model without weakening credential-wide rate-limit protection.

**Architecture:** The shared Tk scrollable-tab builder receives an opt-in input guard used only by Settings. Custom AI keeps its transport cooldown key for true rate limits and adds a wire/model request key for transient gateway and capacity failures; lookup observes both scopes.

**Tech Stack:** Python 3, Tkinter/ttk, `unittest`, existing Custom AI provider and translation worker infrastructure.

---

## File structure

- Create `tests/test_ui_elements.py`: wheel normalization and real-Tk Settings scroll regression coverage.
- Modify `ui_elements.py`: centralized, opt-in descendant input wheel guard.
- Modify `gui_builder.py`: enable the guard for the Settings tab only.
- Modify `tests/test_custom_ai.py`: model-isolated capacity cooldown and shared rate-limit regressions.
- Modify `custom_ai.py`: dual transport/request cooldown identities and scoped diagnostics.
- Create `.codex/handoffs/2026-07-10_13-27-53.md`: final evidence and remaining work.

### Task 1: Establish baseline and backups

**Files:**
- Back up: `ui_elements.py`
- Back up: `gui_builder.py`
- Back up: `custom_ai.py`
- Back up: `tests/test_custom_ai.py`

- [ ] **Step 1: Run focused baseline suites**

Run:

```powershell
py -m unittest tests.test_modern_ui -v
py -m unittest tests.test_custom_ai -v
```

Expected: both commands exit 0. Record existing skips separately from failures.

- [ ] **Step 2: Create one timestamped backup tree**

Use PowerShell `Copy-Item -LiteralPath` so each source keeps its workspace-relative path under `.codex/backups/YYYY-MM-DD_HH-mm-ss/`.

- [ ] **Step 3: Verify backup hashes**

Run `Get-FileHash` on every source and backup pair. Expected: each pair has the same SHA-256 hash.

### Task 2: Add the Settings wheel regression tests

**Files:**
- Create: `tests/test_ui_elements.py`

- [ ] **Step 1: Write the failing tests**

```python
import types
import unittest
import tkinter as tk
from tkinter import ttk

from ui_elements import _mousewheel_scroll_units, create_scrollable_tab


class MouseWheelUnitTests(unittest.TestCase):
    def test_normalizes_windows_and_x11_wheel_events(self):
        self.assertEqual(_mousewheel_scroll_units(types.SimpleNamespace(delta=-120, num=None)), 1)
        self.assertEqual(_mousewheel_scroll_units(types.SimpleNamespace(delta=120, num=None)), -1)
        self.assertEqual(_mousewheel_scroll_units(types.SimpleNamespace(delta=0, num=4)), -1)
        self.assertEqual(_mousewheel_scroll_units(types.SimpleNamespace(delta=0, num=5)), 1)


class ScrollableTabInputGuardTests(unittest.TestCase):
    def setUp(self):
        try:
            self.root = tk.Tk()
        except tk.TclError as error:
            self.skipTest(f"Tk display unavailable: {error}")
        self.root.geometry("320x180")

    def tearDown(self):
        if hasattr(self, "root"):
            self.root.destroy()

    def test_hovered_combobox_and_spinbox_keep_values_while_page_scrolls(self):
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True)
        content = create_scrollable_tab(
            notebook,
            "Settings",
            protect_wheel_inputs=True,
        )
        combo = ttk.Combobox(content, values=["one", "two"], state="readonly")
        combo.set("one")
        combo.pack()
        spin = ttk.Spinbox(content, from_=1, to=10)
        spin.set("5")
        spin.pack()
        for index in range(40):
            ttk.Label(content, text=f"row {index}").pack()
        self.root.update()
        canvas = content.master

        before = canvas.yview()[0]
        combo.event_generate("<MouseWheel>", delta=-120)
        self.root.update()
        self.assertEqual(combo.get(), "one")
        self.assertGreater(canvas.yview()[0], before)

        before = canvas.yview()[0]
        spin.event_generate("<MouseWheel>", delta=-120)
        self.root.update()
        self.assertEqual(spin.get(), "5")
        self.assertGreater(canvas.yview()[0], before)
```

- [ ] **Step 2: Run the new test and verify RED**

Run: `py -m unittest tests.test_ui_elements -v`

Expected: import/signature failure because the wheel helper and opt-in guard do not exist yet.

### Task 3: Implement the opt-in scroll guard

**Files:**
- Modify: `ui_elements.py:1-93`
- Modify: `gui_builder.py:489-492`
- Test: `tests/test_ui_elements.py`

- [ ] **Step 1: Add wheel normalization and descendant binding**

Add to `ui_elements.py`:

```python
_WHEEL_INPUT_WIDGET_CLASSES = frozenset({
    "TCombobox", "TSpinbox", "Spinbox", "TEntry", "Entry", "TScale", "Scale",
})


def _mousewheel_scroll_units(event):
    button_number = getattr(event, "num", None)
    if button_number == 4:
        return -1
    if button_number == 5:
        return 1
    delta = int(getattr(event, "delta", 0) or 0)
    if delta == 0:
        return 0
    steps = max(1, abs(delta) // 120)
    return -steps if delta > 0 else steps


def _bind_wheel_input_guards(root_widget, handler):
    stack = list(root_widget.winfo_children())
    while stack:
        widget = stack.pop()
        stack.extend(widget.winfo_children())
        if widget.winfo_class() not in _WHEEL_INPUT_WIDGET_CLASSES:
            continue
        if getattr(widget, "_scroll_wheel_guard_installed", False):
            continue
        widget.bind("<MouseWheel>", handler, add="+")
        widget.bind("<Button-4>", handler, add="+")
        widget.bind("<Button-5>", handler, add="+")
        widget._scroll_wheel_guard_installed = True
```

- [ ] **Step 2: Extend the scrollable-tab API and route guarded events**

Change the signature to:

```python
def create_scrollable_tab(notebook, tab_name, protect_wheel_inputs=False):
```

Use one scroll function for global and guarded events:

```python
def _scroll_for_event(event):
    units = _mousewheel_scroll_units(event)
    if units and scrollbar.winfo_ismapped():
        canvas.yview_scroll(units, "units")


def _on_guarded_input_wheel(event):
    _scroll_for_event(event)
    return "break"
```

Schedule an idempotent descendant scan after idle and after content
configuration changes when `protect_wheel_inputs` is true.

- [ ] **Step 3: Enable the guard only for Settings**

In `gui_builder.py`:

```python
scrollable_content = create_scrollable_tab(
    app.tab_control,
    app.ui_lang.get_label("settings_tab_title"),
    protect_wheel_inputs=True,
)
```

- [ ] **Step 4: Run the UI test and verify GREEN**

Run: `py -m unittest tests.test_ui_elements -v`

Expected: all tests pass; a Tk-unavailable environment may skip only the real-widget test.

### Task 4: Add cooldown-scope regression tests

**Files:**
- Modify: `tests/test_custom_ai.py:2480-2660`

- [ ] **Step 1: Add a model-isolation failing test**

```python
def test_capacity_cooldown_is_scoped_to_wire_api_and_model(self):
    class Response:
        status_code = 502
        headers = {}

    provider = CustomAIProvider(http_client=object())
    failed = {
        "base_url": "https://host.example/v1",
        "api_key": "same-secret",
        "wire_api": "responses",
        "model": "gpt-5.6",
    }
    alternative = dict(failed, model="gpt-5.5")
    with patch("custom_ai.time.monotonic", return_value=100.0):
        provider._activate_rate_limit_cooldown(failed, Response(), "502 Bad gateway")
    with patch("custom_ai.time.monotonic", return_value=101.0):
        self.assertGreater(provider.get_cooldown_remaining(failed), 0.0)
        self.assertEqual(provider.get_cooldown_remaining(alternative), 0.0)
```

- [ ] **Step 2: Add a transport-rate-limit preservation test**

```python
def test_explicit_rate_limit_cooldown_remains_shared_across_models(self):
    class Response:
        status_code = 429
        headers = {"Retry-After": "60"}

    provider = CustomAIProvider(http_client=object())
    failed = {
        "base_url": "https://host.example/v1",
        "api_key": "same-secret",
        "wire_api": "responses",
        "model": "gpt-5.6",
    }
    alternative = dict(failed, model="gpt-5.5")
    with patch("custom_ai.time.monotonic", return_value=100.0):
        provider._activate_rate_limit_cooldown(failed, Response(), "Rate limit exceeded")
    with patch("custom_ai.time.monotonic", return_value=101.0):
        self.assertGreater(provider.get_cooldown_remaining(alternative), 0.0)
```

- [ ] **Step 3: Run the focused tests and verify RED**

Run:

```powershell
py -m unittest tests.test_custom_ai.CustomAIProviderTests.test_capacity_cooldown_is_scoped_to_wire_api_and_model -v
py -m unittest tests.test_custom_ai.CustomAIProviderTests.test_explicit_rate_limit_cooldown_remains_shared_across_models -v
```

Expected: the capacity test fails because the current endpoint-plus-credential key leaks to the alternative model; the explicit 429 preservation test already passes.

### Task 5: Implement dual-scope cooldown identity

**Files:**
- Modify: `custom_ai.py:799-805`
- Modify: `custom_ai.py:1668-1733`
- Test: `tests/test_custom_ai.py`

- [ ] **Step 1: Add request-scope identity and failure classification**

```python
def _request_cooldown_cache_key(self, profile):
    transport_key = self._rate_limit_cache_key(profile)
    profile = profile if isinstance(profile, dict) else {}
    return transport_key + (
        normalize_custom_ai_wire_api(profile.get("wire_api")),
        str(profile.get("model") or "").strip(),
    )


def _cooldown_scope_for_failure(self, response, detail):
    status_code = int(getattr(response, "status_code", 0) or 0)
    if status_code == 429 or self._looks_like_rate_limit_error(detail):
        return "transport"
    return "request"


def _cooldown_cache_key(self, profile, scope):
    if scope == "transport":
        return self._rate_limit_cache_key(profile)
    return self._request_cooldown_cache_key(profile)
```

- [ ] **Step 2: Store and report the selected scope**

In `_activate_rate_limit_cooldown`, calculate the scope before choosing the
cache key. Extend the existing safe log message with:

```python
f"scope={scope} model={str(profile.get('model') or '').strip() or 'unknown'} "
```

- [ ] **Step 3: Observe both scopes and reset both backoff counters**

`get_cooldown_remaining` checks the transport and request keys and returns the
maximum active duration. `_note_rate_limit_success` removes backoff counters for
both keys while keeping existing active-deadline semantics unchanged.

- [ ] **Step 4: Run both cooldown tests and the provider test group**

Run:

```powershell
py -m unittest tests.test_custom_ai.CustomAIProviderTests.test_capacity_cooldown_is_scoped_to_wire_api_and_model -v
py -m unittest tests.test_custom_ai.CustomAIProviderTests.test_explicit_rate_limit_cooldown_remains_shared_across_models -v
py -m unittest tests.test_custom_ai.CustomAIProviderTests -v
```

Expected: all commands exit 0.

### Task 6: Full verification and handoff

**Files:**
- Create: `.codex/handoffs/2026-07-10_13-27-53.md`

- [ ] **Step 1: Run focused suites**

```powershell
py -m unittest tests.test_ui_elements -v
py -m unittest tests.test_custom_ai -v
```

- [ ] **Step 2: Run both discovery modes**

```powershell
py -m unittest discover -s tests
py -m unittest discover
```

- [ ] **Step 3: Run syntax and whitespace checks**

```powershell
py -B -m py_compile ui_elements.py gui_builder.py custom_ai.py tests\test_ui_elements.py tests\test_custom_ai.py
git diff --check
```

Expected: all commands exit 0 with zero failures.

- [ ] **Step 4: Review the requirements against the diff**

Confirm that only Settings opts into the guard; the capacity test distinguishes
models; the 429 test shares cooldown; no secret, raw credential fingerprint, or
new full endpoint is logged.

- [ ] **Step 5: Write the handoff**

Record task summary, changed files, backup location, RED/GREEN evidence, full
verification output, runtime log findings, design decisions, and the next large
optimization candidates.
