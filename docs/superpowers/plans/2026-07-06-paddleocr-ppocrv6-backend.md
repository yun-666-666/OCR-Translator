# PaddleOCR PP-OCRv6 Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a selectable offline PaddleOCR 3.7 + PP-OCRv6 OCR backend while keeping Tesseract as a fallback.

**Architecture:** Create a focused `paddle_ocr_backend.py` wrapper that lazy-loads PaddleOCR from an installed package or the local `PaddleOCR-3.7.0` source directory, caches engines by settings, and returns plain text plus line confidence data. Wire `ocr_model == "paddleocr"` into config migration, the OCR model combobox, model-specific UI visibility, realtime OCR routing, cache keys, and OCR preview.

**Tech Stack:** Python, Tkinter, PIL, OpenCV/NumPy, PaddleOCR 3.7, PP-OCRv6 small detection and recognition models, `unittest`.

---

### Task 1: PaddleOCR Wrapper

**Files:**
- Create: `paddle_ocr_backend.py`
- Test: `tests/test_paddle_ocr_backend.py`

- [ ] **Step 1: Write failing wrapper tests**

Create tests that assert:

```python
from paddle_ocr_backend import (
    PaddleOCRSettings,
    flatten_paddleocr_result,
    resolve_ppocrv6_model_names,
)

def test_resolve_small_ppocrv6_model_names():
    assert resolve_ppocrv6_model_names("small") == (
        "PP-OCRv6_small_det",
        "PP-OCRv6_small_rec",
    )

def test_flatten_filters_low_scores_and_preserves_linebreaks():
    result = [{
        "rec_texts": [" HELLO ", "noise", "WORLD"],
        "rec_scores": [0.91, 0.12, 0.72],
        "rec_boxes": [],
    }]
    text, lines = flatten_paddleocr_result(result, min_score=0.35, keep_linebreaks=True)
    assert text == "HELLO\nWORLD"
    assert [line.text for line in lines] == ["HELLO", "WORLD"]
```

Run: `python -m unittest tests.test_paddle_ocr_backend`
Expected: FAIL because `paddle_ocr_backend` does not exist.

- [ ] **Step 2: Implement wrapper API**

Add:

```python
@dataclass(frozen=True)
class PaddleOCRSettings:
    source_dir: str = "PaddleOCR-3.7.0"
    lang: str = "en"
    ocr_version: str = "PP-OCRv6"
    model_size: str = "small"
    device: str = "cpu"
    min_score: float = 0.35
    upscale: float = 2.0
    text_det_limit_side_len: int = 960
    text_det_limit_type: str = "max"
    use_textline_orientation: bool = False
```

Add `resolve_ppocrv6_model_names`, `flatten_paddleocr_result`, `recognize_with_paddleocr`, `clear_paddleocr_engines`, and a reusable engine cache.

- [ ] **Step 3: Run wrapper tests**

Run: `python -m unittest tests.test_paddle_ocr_backend`
Expected: PASS.

### Task 2: Config and App State

**Files:**
- Modify: `config_manager.py`
- Modify: `app_logic.py`
- Test: `tests/test_paddle_ocr_backend.py`

- [ ] **Step 1: Write failing config tests**

Add tests that assert:

```python
from config_manager import DEFAULT_CONFIG_SETTINGS

def test_paddleocr_defaults_exist():
    assert DEFAULT_CONFIG_SETTINGS["ocr_model"] == "tesseract"
    assert DEFAULT_CONFIG_SETTINGS["paddleocr_ocr_version"] == "PP-OCRv6"
    assert DEFAULT_CONFIG_SETTINGS["paddleocr_model_size"] == "small"
    assert DEFAULT_CONFIG_SETTINGS["paddleocr_min_score"] == "0.35"
```

Expected: FAIL because PaddleOCR defaults do not exist.

- [ ] **Step 2: Add defaults and model normalization**

Allow `ocr_model` values `tesseract`, `custom_ai`, and `paddleocr`. Add PaddleOCR settings defaults to `DEFAULT_CONFIG_SETTINGS`, app `tk.Variable` initialization, trace callbacks, and display-name initialization.

- [ ] **Step 3: Run tests**

Run: `python -m unittest tests.test_paddle_ocr_backend tests.test_custom_ai_startup`
Expected: PASS.

### Task 3: UI Selection and Visibility

**Files:**
- Modify: `gui_builder.py`
- Modify: `handlers/ui_interaction_handler.py`
- Test: `tests/test_paddle_ocr_backend.py`

- [ ] **Step 1: Write failing UI helper tests**

Add tests for a small helper function or method that builds OCR display names and maps the PaddleOCR display back to `paddleocr`.

- [ ] **Step 2: Add PaddleOCR combobox option**

Add `PaddleOCR PP-OCRv6 (offline)` to the OCR combobox before Custom AI profile names. Selecting it sets `app.ocr_model_var` to `paddleocr`.

- [ ] **Step 3: Update visibility**

When PaddleOCR is selected, hide Tesseract path/preprocessing/remove-trailing-garbage controls, show stability and OCR debug preview, and hide Custom AI image request controls.

- [ ] **Step 4: Run tests**

Run: `python -m unittest tests.test_paddle_ocr_backend tests.test_custom_ai_startup`
Expected: PASS.

### Task 4: Realtime OCR and Preview Routing

**Files:**
- Modify: `worker_threads.py`
- Modify: `app_logic.py`
- Test: `tests/test_paddle_ocr_backend.py`

- [ ] **Step 1: Write failing routing tests**

Add tests that patch `worker_threads.recognize_with_paddleocr` and verify local `ocr_model == "paddleocr"` routes there, not to Tesseract.

- [ ] **Step 2: Implement worker routing**

Add `_get_paddleocr_settings(app)` and `_get_paddleocr_ocr_cache_mode_key(app)`. In `run_ocr_thread`, route PaddleOCR before Tesseract, pass PIL screenshots to `recognize_with_paddleocr`, apply linebreak handling, cache results under PaddleOCR-specific settings, and keep the shared stability/translation path.

- [ ] **Step 3: Implement preview routing**

In `refresh_ocr_preview`, if current model is PaddleOCR, call `recognize_with_paddleocr` on the captured PIL image and display a processed preview image that matches the selected local backend as closely as possible.

- [ ] **Step 4: Run tests**

Run: `python -m unittest tests.test_paddle_ocr_backend tests.test_latency_optimization`
Expected: PASS.

### Task 5: Config Example, Verification, and Handoff

**Files:**
- Modify: `ocr_translator_config.example.ini`
- Create: `.codex/handoffs/YYYY-MM-DD_HH-mm-ss.md`

- [ ] **Step 1: Update example config**

Add PaddleOCR keys and comments showing `ocr_model = paddleocr`.

- [ ] **Step 2: Run focused verification**

Run:

```powershell
python -m unittest tests.test_paddle_ocr_backend
python -m unittest tests.test_custom_ai_startup
python -m unittest tests.test_latency_optimization
python -B -m py_compile paddle_ocr_backend.py app_logic.py gui_builder.py handlers\ui_interaction_handler.py worker_threads.py config_manager.py
git diff --check -- paddle_ocr_backend.py app_logic.py gui_builder.py handlers\ui_interaction_handler.py worker_threads.py config_manager.py ocr_translator_config.example.ini tests\test_paddle_ocr_backend.py
```

Expected: all commands pass.

- [ ] **Step 3: Write handoff**

Create `.codex/handoffs/YYYY-MM-DD_HH-mm-ss.md` with summary, changed files, backup location, commands run, decisions, and remaining dependency/setup notes.
