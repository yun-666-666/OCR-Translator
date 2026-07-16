# AI Auto Settings and Compact Settings Layout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace five Custom AI controls with one adaptive policy, standardize capture on MSS, tune OCR defaults, shorten the built-in prompt, and compact the requested settings rows.

**Architecture:** Add a focused `ai_optimization.py` policy module that resolves response mode and OCR image payloads without depending on Tk. Configuration migrates legacy settings into `ai_optimization_mode`; runtime callers resolve the policy from active profile, image size, recent OCR latency, and route-scoped capability memory. Existing request, OCR cache, and overlay paths remain the integration seams.

**Tech Stack:** Python 3, Tk/ttk, Pillow, MSS, unittest, existing Custom AI route advisor and runtime metrics.

---

## File structure

- Create `ai_optimization.py`: policy constants, config migration helpers,
  response-mode mapping, provider compatibility detection, route-scoped image
  capability memory, and pure OCR payload resolver.
- Create `tests/test_ai_optimization.py`: focused policy and migration tests.
- Modify `config_manager.py`: new canonical config key and legacy migration.
- Modify `app_logic.py`, `app_configuration.py`: unified Tk variable and runtime
  getters.
- Modify `custom_ai_policy.py`, `handlers/translation_requests.py`,
  `worker_translation.py`: safe/race-only automatic response behavior.
- Modify `app_capture_ocr.py`, `worker_ocr.py`: resolved image contract and
  cache identity.
- Modify `ocr_utils.py`, `worker_capture.py`, `worker_threads.py`: MSS-only
  capture.
- Modify `gui_settings_builder.py`, `handlers/ui_interaction_handler.py`,
  resource CSV files: one AI selector and compact rows.
- Modify `custom_prompt.txt`, `ocr_translator_config.example.ini`,
  `requirements.txt`, and `setup.py`: shipped defaults and dependency cleanup.
- Modify focused existing tests to replace legacy expectations.

### Task 1: Unified policy module and config migration

**Files:**
- Create: `ai_optimization.py`
- Create: `tests/test_ai_optimization.py`
- Modify: `config_manager.py`
- Modify: `ocr_translator_config.example.ini`

- [ ] **Step 1: Write failing policy and migration tests**

Add tests for:

```python
def test_legacy_quality_settings_migrate_to_quality(self):
    settings = {
        "custom_ai_latency_mode": "safe",
        "custom_ai_ocr_image_format": "png",
        "custom_ai_ocr_image_mode": "lossless_webp",
        "custom_ai_ocr_image_quality": "95",
        "custom_ai_ocr_image_detail": "high",
    }
    mode = migrate_legacy_ai_optimization_settings(settings)
    self.assertEqual(mode, "quality")
    self.assertEqual(settings["ai_optimization_mode"], "quality")
    self.assertNotIn("custom_ai_ocr_image_format", settings)


def test_old_ocr_defaults_migrate_without_overwriting_custom_values(self):
    legacy = {"stability_threshold": "2", "paddleocr_min_score": "0.35"}
    migrate_legacy_ocr_defaults(legacy)
    self.assertEqual(legacy["stability_threshold"], "0")
    self.assertEqual(legacy["paddleocr_min_score"], "0.45")

    custom = {"stability_threshold": "4", "paddleocr_min_score": "0.60"}
    migrate_legacy_ocr_defaults(custom)
    self.assertEqual(custom["stability_threshold"], "4")
    self.assertEqual(custom["paddleocr_min_score"], "0.60")
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```powershell
py -m unittest tests.test_ai_optimization -v
```

Expected: import failure because `ai_optimization.py` does not exist.

- [ ] **Step 3: Implement policy constants and migration helpers**

Implement:

```python
AI_OPTIMIZATION_AUTO = "auto"
AI_OPTIMIZATION_SPEED = "speed"
AI_OPTIMIZATION_QUALITY = "quality"
AI_OPTIMIZATION_MODES = {
    AI_OPTIMIZATION_AUTO,
    AI_OPTIMIZATION_SPEED,
    AI_OPTIMIZATION_QUALITY,
}


def normalize_ai_optimization_mode(value):
    normalized = str(value or AI_OPTIMIZATION_AUTO).strip().lower()
    return normalized if normalized in AI_OPTIMIZATION_MODES else AI_OPTIMIZATION_AUTO


def migrate_legacy_ai_optimization_settings(settings):
    if "ai_optimization_mode" in settings:
        mode = normalize_ai_optimization_mode(settings["ai_optimization_mode"])
    else:
        quality = normalize_api_ocr_image_quality(
            settings.get("custom_ai_ocr_image_quality", 85)
        )
        detail = normalize_api_ocr_image_detail(
            settings.get("custom_ai_ocr_image_detail", "auto")
        )
        image_format = normalize_api_ocr_image_format(
            settings.get("custom_ai_ocr_image_format", "webp")
        )
        image_mode = normalize_api_ocr_image_mode(
            settings.get("custom_ai_ocr_image_mode", "balanced_webp")
        )
        if (
            detail == "high"
            or image_format == "png"
            or image_mode == "lossless_webp"
            or quality >= 90
        ):
            mode = AI_OPTIMIZATION_QUALITY
        elif detail == "low" or image_mode == "small_grayscale_webp" or quality <= 75:
            mode = AI_OPTIMIZATION_SPEED
        else:
            mode = AI_OPTIMIZATION_AUTO
    settings["ai_optimization_mode"] = mode
    for legacy_key in LEGACY_AI_SETTING_KEYS:
        settings.pop(legacy_key, None)
    settings.pop("capture_backend", None)
    return mode
```

Set new defaults to `ai_optimization_mode=auto`,
`stability_threshold=0`, and `paddleocr_min_score=0.45`.

- [ ] **Step 4: Run focused tests and config startup tests**

Run:

```powershell
py -m unittest tests.test_ai_optimization tests.test_custom_ai_startup -v
```

Expected: policy tests pass; startup tests may still fail only on old UI/runtime
expectations scheduled for later tasks.

- [ ] **Step 5: Commit the policy/config increment**

```powershell
git add ai_optimization.py tests/test_ai_optimization.py config_manager.py ocr_translator_config.example.ini
git commit -m "feat: unify AI optimization settings"
```

### Task 2: Adaptive response mapping without automatic streaming

**Files:**
- Modify: `ai_optimization.py`
- Modify: `app_logic.py`
- Modify: `app_configuration.py`
- Modify: `custom_ai_policy.py`
- Modify: `handlers/translation_requests.py`
- Modify: `worker_translation.py`
- Modify: `tests/test_ai_optimization.py`
- Modify: `tests/test_custom_ai.py`
- Modify: `tests/test_latency_optimization.py`

- [ ] **Step 1: Write failing response-policy tests**

Cover:

```python
def test_response_mode_mapping(self):
    self.assertEqual(resolve_ai_response_mode("auto"), "adaptive")
    self.assertEqual(resolve_ai_response_mode("speed"), "safe")
    self.assertEqual(resolve_ai_response_mode("quality"), "safe")


def test_adaptive_high_p90_without_second_profile_stays_safe(self):
    advisor = CustomAILatencyModeAdvisor(min_samples=3)
    for duration in (3.0, 3.5, 4.0):
        advisor.observe_request(duration, success=True)
    decision = advisor.resolve(
        "adaptive",
        stream_supported=True,
        healthy_race_profile_count=1,
    )
    self.assertEqual(decision.mode, "safe")


def test_adaptive_very_high_p90_with_two_profiles_uses_bounded_race(self):
    advisor = CustomAILatencyModeAdvisor(
        min_samples=3,
        race_latency_threshold_seconds=3.0,
    )
    for duration in (4.0, 4.5, 5.0):
        advisor.observe_request(duration, success=True)
    decision = advisor.resolve(
        "adaptive",
        stream_supported=True,
        healthy_race_profile_count=2,
    )
    self.assertEqual(decision.mode, "race")
```

- [ ] **Step 2: Run tests and verify RED**

```powershell
py -m unittest tests.test_ai_optimization tests.test_custom_ai.CustomAILatencyModeAdvisorTests -v
```

Expected: missing resolver and old adaptive stream selection failures.

- [ ] **Step 3: Wire the unified variable and response mapping**

Initialize only:

```python
self.ai_optimization_mode_var = tk.StringVar(
    value=normalize_ai_optimization_mode(
        self.config["Settings"].get("ai_optimization_mode", "auto")
    )
)
```

Make `get_custom_ai_latency_mode()` return
`resolve_ai_response_mode(self.ai_optimization_mode_var.get())`. Direct worker
reads must call the getter and fall back to legacy test doubles only when the
new variable/getter is unavailable.

- [ ] **Step 4: Remove automatic stream selection**

In `_choose_adaptive_locked`, keep safe results for high P90 unless the bounded
race conditions are met. Preserve explicit legacy `stream` handling in
`normalize_custom_ai_latency_mode()` and transport code.

- [ ] **Step 5: Run focused response tests**

```powershell
py -m unittest tests.test_ai_optimization tests.test_custom_ai tests.test_latency_optimization -q
```

Expected: all response-policy and existing request tests pass.

- [ ] **Step 6: Commit**

```powershell
git add ai_optimization.py app_logic.py app_configuration.py custom_ai_policy.py handlers/translation_requests.py worker_translation.py tests/test_ai_optimization.py tests/test_custom_ai.py tests/test_latency_optimization.py
git commit -m "perf: adapt AI responses without stream churn"
```

### Task 3: Dynamic OCR image payload resolution

**Files:**
- Modify: `ai_optimization.py`
- Modify: `app_capture_ocr.py`
- Modify: `worker_ocr.py`
- Modify: `handlers/translation_requests.py`
- Modify: `tests/test_ai_optimization.py`
- Modify: `tests/test_latency_optimization.py`
- Modify: `tests/test_custom_ai.py`

- [ ] **Step 1: Write failing image-policy tests**

Cover these exact contracts:

```python
def test_speed_policy_uses_small_low_detail_webp(self):
    decision = resolve_ai_ocr_image_policy(
        "speed", profile={"base_url": "https://api.openai.com/v1"},
        image_size=(1280, 180),
    )
    self.assertEqual(
        decision.contract_key,
        "webp|small_grayscale_webp|72|low",
    )


def test_direct_xai_auto_uses_jpeg(self):
    decision = resolve_ai_ocr_image_policy(
        "auto", profile={"base_url": "https://api.x.ai/v1"},
        image_size=(1280, 180),
    )
    self.assertEqual(decision.image_format, "jpeg")


def test_auto_slow_route_reduces_payload(self):
    decision = resolve_ai_ocr_image_policy(
        "auto",
        profile={"base_url": "https://relay.example/v1"},
        image_size=(1280, 180),
        route_p90_seconds=4.5,
        route_sample_count=8,
    )
    self.assertEqual(decision.image_detail, "low")


def test_route_capability_memory_remembers_rejected_webp(self):
    memory = AiOcrImageCapabilityMemory()
    profile = {"base_url": "https://relay.example/v1", "model": "vision"}
    memory.mark_format_unsupported(profile, "webp")
    decision = resolve_ai_ocr_image_policy(
        "auto", profile=profile, image_size=(1280, 180), capability_memory=memory
    )
    self.assertEqual(decision.image_format, "jpeg")
```

- [ ] **Step 2: Run focused tests and verify RED**

```powershell
py -m unittest tests.test_ai_optimization -v
```

Expected: missing payload decision and capability memory.

- [ ] **Step 3: Implement immutable resolver and capability memory**

Add:

```python
@dataclass(frozen=True)
class AiOcrImageDecision:
    image_format: str
    image_mode: str
    image_quality: int
    image_detail: str
    reason: str

    @property
    def contract_key(self):
        return (
            f"{self.image_format}|{self.image_mode}|"
            f"{self.image_quality}|{self.image_detail}"
        )
```

The resolver must apply policy preset, xAI compatibility, route P90, image
aspect/size, and capability memory in deterministic order.

- [ ] **Step 4: Use the decision for encoding and cache identity**

`convert_to_api_ocr_image()` resolves once from the active OCR profile, PIL
size, and recent `api_ocr_duration` P90. `_get_api_ocr_cache_mode_key()` calls
the same pure resolver with `screenshot_pil.size` before cache lookup.

On explicit unsupported image format errors in Custom AI OCR, call:

```python
self.app.ai_ocr_image_capability_memory.mark_format_unsupported(
    profile,
    attempted_format,
)
```

Only match narrow format/capability error text; do not learn from timeouts,
HTTP 5xx, empty output, or authentication failures.

- [ ] **Step 5: Run OCR image and request tests**

```powershell
py -m unittest tests.test_ai_optimization tests.test_latency_optimization tests.test_custom_ai -q
```

Expected: all pass.

- [ ] **Step 6: Commit**

```powershell
git add ai_optimization.py app_capture_ocr.py worker_ocr.py handlers/translation_requests.py tests/test_ai_optimization.py tests/test_latency_optimization.py tests/test_custom_ai.py
git commit -m "perf: resolve AI OCR payloads automatically"
```

### Task 4: Standardize screenshot capture on MSS

**Files:**
- Modify: `ocr_utils.py`
- Modify: `worker_capture.py`
- Modify: `worker_threads.py`
- Modify: `app_capture_ocr.py`
- Modify: `app_logic.py`
- Modify: `requirements.txt`
- Modify: `setup.py`
- Modify: `tests/test_latency_optimization.py`
- Modify: `tests/test_custom_ai_startup.py`

- [ ] **Step 1: Write failing MSS-only tests**

Tests must assert:

```python
def test_capture_screen_region_uses_mss_only(self):
    image = capture_screen_region(
        (1, 2, 8, 6),
        mss_factory=FakeMssFactory(),
    )
    self.assertEqual(image._gct_capture_backend, "mss")


def test_packaging_no_longer_requires_pyautogui(self):
    self.assertNotIn("pyautogui", Path("requirements.txt").read_text().lower())
    self.assertNotIn("pyautogui", Path("setup.py").read_text().lower())
```

Also assert the settings builder has no capture-backend widget.

- [ ] **Step 2: Run tests and verify RED**

```powershell
py -m unittest tests.test_latency_optimization tests.test_custom_ai_startup -q
```

Expected: PyAutoGUI and selector expectations fail.

- [ ] **Step 3: Remove selector and PyAutoGUI paths**

Make `capture_screen_region(region, mss_factory=None)` call MSS once and tag
metadata as `mss`. Remove backend benchmarking, selector state, backend config,
preview direct PyAutoGUI import, and dependency declarations.

The preview must call the same MSS capture helper:

```python
screenshot_pil = capture_screen_region((x1, y1, width, height))
```

- [ ] **Step 4: Run capture tests**

```powershell
py -m unittest tests.test_latency_optimization tests.test_custom_ai_startup -q
```

Expected: all updated capture/startup tests pass.

- [ ] **Step 5: Commit**

```powershell
git add ocr_utils.py worker_capture.py worker_threads.py app_capture_ocr.py app_logic.py requirements.txt setup.py tests/test_latency_optimization.py tests/test_custom_ai_startup.py
git commit -m "perf: standardize screenshot capture on MSS"
```

### Task 5: Replace legacy AI widgets and compact the settings layout

**Files:**
- Modify: `gui_settings_builder.py`
- Modify: `handlers/ui_interaction_handler.py`
- Modify: `resources/gui_eng.csv`
- Modify: `resources/gui_zh.csv`
- Modify: `resources/gui_pol.csv`
- Modify: `tests/test_custom_ai_startup.py`
- Modify: `tests/test_ui_elements.py`

- [ ] **Step 1: Write failing UI source/behavior tests**

Assert:

```python
self.assertIn("ai_optimization_mode_combobox", settings_source)
self.assertNotIn("custom_ai_ocr_image_format_combobox", settings_source)
self.assertNotIn("capture_backend_combobox", settings_source)
self.assertIn('app.color_displays[color_type].bind("<Button-1>"', settings_source)
self.assertNotIn('get_label("choose_color_btn"', settings_source)
```

Construct a Tk settings frame and verify:

- bold and centered checkboxes have the same grid row;
- the three swatches have the same grid row;
- each swatch has a `<Button-1>` binding.

- [ ] **Step 2: Run focused UI tests and verify RED**

```powershell
py -m unittest tests.test_custom_ai_startup tests.test_ui_elements -v
```

Expected: old widgets/layout still present.

- [ ] **Step 3: Build the unified AI combobox**

Use localized values for `auto`, `speed`, and `quality`; bind selection to
`ai_optimization_mode_var` and existing save behavior. OCR-model visibility
logic shows this single control for Custom AI OCR and hides it otherwise.

- [ ] **Step 4: Compact the checkbox and color rows**

Place bold/center inside one `ttk.Frame`. Build one color row with three small
entry frames. Bind each swatch:

```python
swatch.bind(
    "<Button-1>",
    lambda _event, color_kind=color_type: app.choose_color_for_settings(color_kind),
)
swatch.configure(cursor="hand2")
```

- [ ] **Step 5: Update localization and run UI tests**

```powershell
py -m unittest tests.test_custom_ai_startup tests.test_ui_elements -v
```

Expected: all pass.

- [ ] **Step 6: Commit**

```powershell
git add gui_settings_builder.py handlers/ui_interaction_handler.py resources/gui_eng.csv resources/gui_zh.csv resources/gui_pol.csv tests/test_custom_ai_startup.py tests/test_ui_elements.py
git commit -m "feat: compact automatic AI settings UI"
```

### Task 6: Ship prompt and OCR defaults

**Files:**
- Modify: `app_configuration.py`
- Modify: `custom_prompt.txt`
- Modify: `tests/test_custom_ai_startup.py`
- Modify: `tests/test_paddle_ocr_backend.py`

- [ ] **Step 1: Update tests first**

Expected prompt:

```python
expected_prompt = (
    "Translate naturally and concisely. Preserve meaning, tone, names, and "
    "terminology; use context only when needed."
)
```

Assert `DEFAULT_CONFIG_SETTINGS["stability_threshold"] == "0"` and
`DEFAULT_CONFIG_SETTINGS["paddleocr_min_score"] == "0.45"`.

- [ ] **Step 2: Run tests and verify RED**

```powershell
py -m unittest tests.test_custom_ai_startup tests.test_paddle_ocr_backend -v
```

Expected: old prompt/Paddle default assertions fail.

- [ ] **Step 3: Update shipped values**

Make `DEFAULT_CUSTOM_PROMPT` and `custom_prompt.txt` exactly equal to the new
English prompt. Align all hardcoded PaddleOCR fallback defaults with `0.45`.

- [ ] **Step 4: Run focused tests**

```powershell
py -m unittest tests.test_custom_ai_startup tests.test_paddle_ocr_backend -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```powershell
git add app_configuration.py custom_prompt.txt tests/test_custom_ai_startup.py tests/test_paddle_ocr_backend.py
git commit -m "chore: tune prompt and OCR defaults"
```

### Task 7: Verification and live Tk probe

**Files:**
- Modify only if verification reveals a regression.

- [ ] **Step 1: Run all focused suites**

```powershell
py -m unittest tests.test_ai_optimization tests.test_custom_ai tests.test_latency_optimization tests.test_custom_ai_startup tests.test_paddle_ocr_backend tests.test_translation_display_layout tests.test_ui_elements -q
```

Expected: zero failures.

- [ ] **Step 2: Run legacy root tests**

```powershell
py -m unittest test_custom_ai test_custom_ai_startup -q
```

Expected: zero failures.

- [ ] **Step 3: Run the full test suite**

```powershell
py -m unittest discover -s tests
```

Expected: zero failures.

- [ ] **Step 4: Compile touched Python modules**

```powershell
py -B -m py_compile ai_optimization.py config_manager.py app_logic.py app_configuration.py custom_ai_policy.py app_capture_ocr.py ocr_utils.py worker_capture.py worker_threads.py worker_ocr.py worker_translation.py gui_settings_builder.py handlers/ui_interaction_handler.py handlers/translation_requests.py tests/test_ai_optimization.py tests/test_custom_ai.py tests/test_latency_optimization.py tests/test_custom_ai_startup.py tests/test_paddle_ocr_backend.py tests/test_ui_elements.py
```

Expected: exit code 0.

- [ ] **Step 5: Run Tk construction probe**

Create the settings UI under the active Windows Tk theme, verify the unified
combobox exists, inspect checkbox/swatch grid rows, invoke each swatch binding
with a patched color chooser, and destroy the root.

- [ ] **Step 6: Run diff hygiene**

```powershell
git diff --check codex/backup-before-ai-auto-settings-20260716..HEAD
git status --short --branch
```

Expected: no whitespace errors and only the known pre-existing untracked
release-note file outside task scope.

### Task 8: Handoff and final local Git checkpoint

**Files:**
- Create: `.codex/handoffs/YYYY-MM-DD_HH-mm-ss.md`

- [ ] **Step 1: Write the handoff**

Include task summary, exact files, backup location, RED/GREEN evidence, full
verification output, design decisions, rollback branch, commits, and remaining
manual verification.

- [ ] **Step 2: Cross-check handoff against Git**

```powershell
git diff --name-only codex/backup-before-ai-auto-settings-20260716..HEAD
git log --oneline codex/backup-before-ai-auto-settings-20260716..HEAD
```

Expected: every changed file and commit is represented in the handoff.

- [ ] **Step 3: Commit the handoff**

```powershell
git add .codex/handoffs/<timestamp>.md
git commit -m "docs: hand off automatic AI settings optimization"
```

- [ ] **Step 4: Final verification after the handoff commit**

```powershell
py -m unittest discover -s tests
git diff --check codex/backup-before-ai-auto-settings-20260716..HEAD
git status --short --branch
```

Expected: full suite passes; diff check is clean; only documented pre-existing
untracked noise remains.
