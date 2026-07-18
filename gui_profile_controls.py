"""Focused Custom AI profile and OCR selection helpers for the Tk UI."""

from logger import log_debug
from custom_ai import (
    CUSTOM_AI_REASONING_EFFORT_HIGH,
    CUSTOM_AI_REASONING_EFFORT_LOW,
    CUSTOM_AI_REASONING_EFFORT_MEDIUM,
    CUSTOM_AI_REASONING_EFFORT_NONE,
    CUSTOM_AI_REASONING_EFFORT_ULTRA,
    normalize_custom_ai_reasoning_effort,
)
from paddle_ocr_backend import PADDLEOCR_DISPLAY_NAME, PADDLEOCR_MODEL_CODE


CUSTOM_AI_REASONING_EFFORT_LABEL_KEYS = (
    (CUSTOM_AI_REASONING_EFFORT_NONE, "custom_ai_reasoning_effort_none", "None"),
    (CUSTOM_AI_REASONING_EFFORT_LOW, "custom_ai_reasoning_effort_low", "Low"),
    (CUSTOM_AI_REASONING_EFFORT_MEDIUM, "custom_ai_reasoning_effort_medium", "Medium"),
    (CUSTOM_AI_REASONING_EFFORT_HIGH, "custom_ai_reasoning_effort_high", "High"),
    (CUSTOM_AI_REASONING_EFFORT_ULTRA, "custom_ai_reasoning_effort_ultra", "Ultra"),
)


def get_paddleocr_ocr_display_name(app):
    return app.ui_lang.get_label("ocr_model_paddleocr", PADDLEOCR_DISPLAY_NAME)


def build_ocr_model_display_options(app):
    options = [
        get_paddleocr_ocr_display_name(app),
    ]
    options.extend([p["name"] for p in app.custom_ai_profiles.list_profiles(enabled_only=True)])
    return options


def resolve_ocr_model_display_selection(app, selected_display):
    if selected_display == get_paddleocr_ocr_display_name(app):
        return PADDLEOCR_MODEL_CODE, None
    for profile in app.custom_ai_profiles.list_profiles(enabled_only=True):
        if profile["name"] == selected_display:
            return "custom_ai", profile["id"]
    return None, None

def filter_model_values(models, query):
    """Return model names containing the query, preserving the original order."""
    query = (query or "").strip().lower()
    if not query:
        return list(models)
    return [model for model in models if query in str(model).lower()]


def _read_var(app, attr, strip=False):
    value = getattr(app, attr).get()
    if strip:
        return str(value or "").strip()
    return value


def _find_selected_custom_ai_profile(app):
    profiles = getattr(app, "custom_ai_profiles", None)
    if profiles is None:
        return None

    selected_id = getattr(app, "ai_profile_selected_id", None)
    if selected_id:
        try:
            profile = profiles.get_profile(selected_id)
        except Exception as e:
            log_debug(f"Custom AI selected profile lookup failed: {e}")
        else:
            if profile:
                return profile

    try:
        current_name = _read_var(app, "ai_profile_name_var", strip=True)
    except Exception:
        current_name = ""
    if not current_name:
        return None

    try:
        profile_list = profiles.list_profiles()
    except Exception as e:
        log_debug(f"Custom AI profile name lookup failed: {e}")
        return None

    return next((profile for profile in profile_list if profile.get("name") == current_name), None)


def _reasoning_effort_display_value(app, effort):
    normalized = normalize_custom_ai_reasoning_effort(effort)
    ui_lang = getattr(app, "ui_lang", None)
    for value, label_key, fallback in CUSTOM_AI_REASONING_EFFORT_LABEL_KEYS:
        if value == normalized:
            if ui_lang is not None:
                return ui_lang.get_label(label_key, fallback)
            return fallback
    return "Low"


def _reasoning_effort_display_values(app):
    return [
        _reasoning_effort_display_value(app, effort)
        for effort, _label_key, _fallback in CUSTOM_AI_REASONING_EFFORT_LABEL_KEYS
    ]


def _reasoning_effort_value_from_display(app, display_value):
    raw_value = str(display_value or "").strip()
    normalized_raw = raw_value.lower().replace("-", "_")
    if normalized_raw in {
        CUSTOM_AI_REASONING_EFFORT_LOW,
        CUSTOM_AI_REASONING_EFFORT_MEDIUM,
        CUSTOM_AI_REASONING_EFFORT_HIGH,
        CUSTOM_AI_REASONING_EFFORT_ULTRA,
    }:
        return normalize_custom_ai_reasoning_effort(raw_value)

    for effort, _label_key, _fallback in CUSTOM_AI_REASONING_EFFORT_LABEL_KEYS:
        if raw_value == _reasoning_effort_display_value(app, effort):
            return effort
    return normalize_custom_ai_reasoning_effort(raw_value)


def build_custom_ai_profile_values_from_form(app):
    values = {
        "name": _read_var(app, "ai_profile_name_var", strip=True),
        "base_url": _read_var(app, "ai_profile_url_var", strip=True),
        "api_key": _read_var(app, "ai_profile_key_var"),
        "model": _read_var(app, "ai_profile_model_var", strip=True),
        "enabled": True,
        "translation_failover_enabled": False,
    }
    selected_profile = _find_selected_custom_ai_profile(app)
    if selected_profile:
        wire_api = str(selected_profile.get("wire_api") or "").strip()
        if wire_api:
            values["wire_api"] = wire_api
        values["reasoning_effort"] = normalize_custom_ai_reasoning_effort(
            selected_profile.get("reasoning_effort")
            or selected_profile.get("model_reasoning_effort")
        )
        structured_output_mode = str(selected_profile.get("structured_output_mode") or "").strip()
        if structured_output_mode:
            values["structured_output_mode"] = structured_output_mode
    reasoning_var = getattr(app, "ai_profile_reasoning_effort_var", None)
    if reasoning_var is not None:
        values["reasoning_effort"] = _reasoning_effort_value_from_display(
            app,
            reasoning_var.get(),
        )
    structured_var = getattr(app, "ai_profile_structured_output_mode_var", None)
    if structured_var is not None:
        values["structured_output_mode"] = _read_var(
            app,
            "ai_profile_structured_output_mode_var",
            strip=True,
        )
    failover_var = getattr(app, "ai_profile_translation_failover_var", None)
    if failover_var is not None:
        values["translation_failover_enabled"] = bool(failover_var.get())
    return values


def _apply_custom_ai_profile_model_selection(app):
    profile_id = getattr(app, "ai_profile_selected_id", None)
    profiles = getattr(app, "custom_ai_profiles", None)
    model_var = getattr(app, "ai_profile_model_var", None)
    if not profile_id or profiles is None or model_var is None:
        return False

    profile = profiles.get_profile(profile_id)
    if not profile:
        return False
    model = str(model_var.get() or "").strip()
    if not model or model == str(profile.get("model") or "").strip():
        return False

    updated_profile = profiles.update_profile(profile_id, model=model)
    active_profile = profiles.get_active_profile("translation")
    if (
        not active_profile
        or active_profile.get("id") != updated_profile.get("id")
    ):
        return True

    translation_handler = getattr(app, "translation_handler", None)
    clear_context = getattr(translation_handler, "_clear_active_context", None)
    if callable(clear_context):
        clear_context()

    if getattr(app, "is_running", False):
        from worker_threads import refresh_translation_after_profile_change

        refresh_translation_after_profile_change(
            app,
            reason="active profile model changed",
        )
    log_debug(
        "Custom AI active translation profile model changed "
        f"profile={updated_profile.get('name', 'Custom AI')} "
        f"model={model}"
    )
    return True

