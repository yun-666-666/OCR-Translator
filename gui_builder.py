# gui_builder.py
import tkinter as tk
from tkinter import ttk, messagebox, colorchooser
import threading
from logger import log_debug
from ui_elements import create_scrollable_tab
from modern_ui import style_tk_text_widget
import tkinter.font as tkFont
from custom_ai import (
    CUSTOM_AI_REASONING_EFFORT_HIGH,
    CUSTOM_AI_REASONING_EFFORT_LOW,
    CUSTOM_AI_REASONING_EFFORT_MEDIUM,
    CUSTOM_AI_REASONING_EFFORT_ULTRA,
    normalize_custom_ai_reasoning_effort,
)
from paddle_ocr_backend import PADDLEOCR_DISPLAY_NAME, PADDLEOCR_MODEL_CODE


CUSTOM_AI_REASONING_EFFORT_LABEL_KEYS = (
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
    return values


def run_profile_network_task_async(app, button, task, on_success, failure_title):
    """Run a Custom AI profile network action without blocking the Tk UI thread."""
    def set_button_state(state):
        if button is None:
            return
        try:
            button.config(state=state)
        except Exception as e:
            log_debug(f"Custom AI profile button state update failed: {e}")

    def finish_success(result):
        set_button_state(tk.NORMAL)
        on_success(result)

    def finish_error(error):
        set_button_state(tk.NORMAL)
        messagebox.showerror(failure_title, str(error), parent=app.root)

    def schedule(callback, *args):
        try:
            app.root.after(0, callback, *args)
        except Exception as e:
            log_debug(f"Custom AI profile async callback scheduling failed: {e}")

    def worker():
        try:
            result = task()
        except Exception as e:
            schedule(finish_error, e)
        else:
            schedule(finish_success, result)

    set_button_state(tk.DISABLED)
    threading.Thread(target=worker, name="CustomAIProfileNetworkTask", daemon=True).start()


_TIMING_LABELS = [
    ("capture_duration", "Capture"),
    ("ocr_duration", "OCR"),
    ("translation_queue_time", "Translation queue"),
    ("translation_worker_time", "Translation worker/API"),
    ("translation_total_latency", "Total translation"),
]

_COUNTER_LABELS = [
    ("instant_cache_hit", "Instant cache hits"),
    ("ocr_frame_cache_hit", "OCR frame cache hits"),
    ("duplicate_inflight_skip", "Duplicate/in-flight skips"),
    ("pending_translation_queued", "Pending translations queued"),
    ("stale_response_discarded", "Stale responses discarded"),
    ("stream_partial_display", "Stream partial displays"),
]


def _format_metric_seconds(value):
    try:
        return f"{float(value):.3f}s"
    except (TypeError, ValueError):
        return "-"


def _format_runtime_metrics_snapshot(snapshot):
    timings = snapshot.get("timings", {}) if isinstance(snapshot, dict) else {}
    counters = snapshot.get("counters", {}) if isinstance(snapshot, dict) else {}
    gauges = snapshot.get("gauges", {}) if isinstance(snapshot, dict) else {}
    labels = snapshot.get("labels", {}) if isinstance(snapshot, dict) else {}

    lines = ["Timings (latest / p50 / p90)"]
    for key, label in _TIMING_LABELS:
        timing = timings.get(key)
        if timing:
            lines.append(
                f"{label}: "
                f"{_format_metric_seconds(timing.get('latest'))} / "
                f"{_format_metric_seconds(timing.get('p50'))} / "
                f"{_format_metric_seconds(timing.get('p90'))}"
            )
        else:
            lines.append(f"{label}: - / - / -")

    lines.append("")
    lines.append("Counters")
    for key, label in _COUNTER_LABELS:
        lines.append(f"{label}: {int(counters.get(key, 0) or 0)}")

    lines.append("")
    lines.append("Current state")
    active_calls = gauges.get("active_translation_calls", 0)
    concurrency_limit = gauges.get("translation_concurrency_limit", 0)
    lines.append(f"Active translation calls: {active_calls} / {concurrency_limit}")
    lines.append(f"OCR queue size: {gauges.get('ocr_queue_size', 0)}")
    lines.append(
        "Provider cooldown remaining: "
        f"{_format_metric_seconds(gauges.get('provider_cooldown_seconds', 0.0))}"
    )
    race_winner = labels.get("race_winner") or "-"
    lines.append(f"Race winner: {race_winner}")
    return "\n".join(lines)


def _widget_exists(widget):
    try:
        return widget is not None and bool(widget.winfo_exists())
    except (AttributeError, tk.TclError):
        return False


def _update_runtime_metric_gauges(app):
    metrics = getattr(app, "runtime_metrics", None)
    set_gauge = getattr(metrics, "set_gauge", None)
    if not callable(set_gauge):
        return

    def safe_set(name, value):
        try:
            set_gauge(name, value)
        except Exception:
            pass

    safe_set("active_translation_calls", len(getattr(app, "active_translation_calls", ()) or ()))
    safe_set("active_ocr_calls", len(getattr(app, "active_ocr_calls", ()) or ()))
    safe_set("ocr_queue_size", 0)
    ocr_queue = getattr(app, "ocr_queue", None)
    if ocr_queue is not None:
        try:
            safe_set("ocr_queue_size", ocr_queue.qsize())
        except Exception:
            pass

    handler = getattr(app, "translation_handler", None)
    concurrency_getter = getattr(handler, "get_translation_concurrency_limit", None)
    if callable(concurrency_getter):
        try:
            safe_set("translation_concurrency_limit", concurrency_getter())
        except Exception:
            safe_set("translation_concurrency_limit", getattr(app, "max_concurrent_translation_calls", 0))
    else:
        safe_set("translation_concurrency_limit", getattr(app, "max_concurrent_translation_calls", 0))

    cooldown_getter = getattr(handler, "get_translation_provider_cooldown_seconds", None)
    if callable(cooldown_getter):
        try:
            safe_set("provider_cooldown_seconds", cooldown_getter())
        except Exception:
            safe_set("provider_cooldown_seconds", 0.0)
    else:
        safe_set("provider_cooldown_seconds", 0.0)


def _cancel_runtime_metrics_refresh(app):
    after_id = getattr(app, "runtime_metrics_refresh_after_id", None)
    if not after_id:
        return
    try:
        app.root.after_cancel(after_id)
    except Exception:
        pass
    app.runtime_metrics_refresh_after_id = None


def _refresh_runtime_metrics_panel(app, schedule_next=True):
    metrics = getattr(app, "runtime_metrics", None)
    text_widget = getattr(app, "runtime_metrics_text", None)
    if metrics is None or not _widget_exists(text_widget):
        return

    try:
        _update_runtime_metric_gauges(app)
        content = _format_runtime_metrics_snapshot(metrics.snapshot())
        text_widget.config(state=tk.NORMAL)
        text_widget.delete("1.0", tk.END)
        text_widget.insert(tk.END, content)
        text_widget.config(state=tk.DISABLED)
    except tk.TclError:
        return
    except Exception as e:
        log_debug(f"Runtime metrics panel refresh failed: {e}")

    if schedule_next:
        try:
            app.runtime_metrics_refresh_after_id = app.root.after(
                1000,
                _refresh_runtime_metrics_panel,
                app,
            )
        except tk.TclError:
            app.runtime_metrics_refresh_after_id = None


def _reset_runtime_metrics_panel(app):
    metrics = getattr(app, "runtime_metrics", None)
    reset = getattr(metrics, "reset", None)
    if callable(reset):
        try:
            reset()
        except Exception as e:
            log_debug(f"Runtime metrics reset failed: {e}")
    _refresh_runtime_metrics_panel(app, schedule_next=False)


def _copy_runtime_metrics_summary(app):
    metrics = getattr(app, "runtime_metrics", None)
    summary_getter = getattr(metrics, "summary_text", None)
    if not callable(summary_getter):
        return
    try:
        summary = summary_getter()
        app.root.clipboard_clear()
        app.root.clipboard_append(summary)
    except tk.TclError:
        return
    except Exception as e:
        log_debug(f"Runtime metrics copy failed: {e}")


def get_system_fonts():
    """Get available system fonts with preferred fonts at the top"""
    try:
        # Get all system fonts
        all_fonts = list(tkFont.families())
        all_fonts.sort()

        # Preferred fonts to show at the top (if available)
        preferred_fonts = ['Arial', 'Times New Roman', 'Calibri', 'Cambria']

        # Build final list with preferred fonts first
        final_fonts = []
        for font in preferred_fonts:
            if font in all_fonts:
                final_fonts.append(font)
                all_fonts.remove(font)  # Remove to avoid duplicates

        # Add remaining fonts alphabetically
        final_fonts.extend(all_fonts)
        return final_fonts
    except Exception:
        # Fallback if system font detection fails
        return ['Arial', 'Times New Roman', 'Calibri', 'Cambria', 'Helvetica', 'Courier New', 'Verdana', 'Tahoma']

def create_main_tab(app):
    # Create a scrollable tab content frame
    scrollable_content = create_scrollable_tab(app.tab_control, app.ui_lang.get_label("main_tab_title"))
    app.tab_main = scrollable_content

    # Create the main frame inside the scrollable area
    frame = ttk.LabelFrame(scrollable_content, text=app.ui_lang.get_label("main_tab_title"))
    frame.pack(fill="both", expand=True, padx=10, pady=10)

    # Add GUI language selection at the top
    language_frame = ttk.Frame(frame)
    language_frame.grid(row=0, column=0, columnspan=2, padx=5, pady=5, sticky="w")

    ttk.Label(language_frame, text=app.ui_lang.get_label("gui_language_label")).pack(side=tk.LEFT, padx=(0,5))
    app.gui_language_combobox = ttk.Combobox(language_frame, textvariable=app.gui_language_var,
                                           values=app.ui_lang.get_language_list(), width=15, state='readonly')
    app.gui_language_combobox.pack(side=tk.LEFT)

    def on_gui_language_changed(event):
        selected_display_name = app.gui_language_var.get()
        lang_code = app.ui_lang.get_language_code_from_name(selected_display_name)

        log_debug(f"GUI Language Change: selected_display='{selected_display_name}', lang_code='{lang_code}', current_lang='{app.ui_lang.current_lang}'")

        # Guard: Only process if the language actually changed
        if lang_code and lang_code != app.ui_lang.current_lang:
            log_debug(f"Changing GUI language from {app.ui_lang.current_lang} to {lang_code}")
            app.ui_lang.load_language(lang_code)

            # Log the new state after language change
            log_debug(f"GUI language changed - new current_lang: '{app.ui_lang.current_lang}'")

            # Update all dropdowns with new language before rebuilding UI
            app.ui_interaction_handler.update_all_dropdowns_for_language_change()

            # This complete UI rebuild is necessary to update all elements
            app.update_ui_language()

            # Save settings only after all UI updates are complete
            # The UI update methods will suppress saves during the update process
            if app._fully_initialized:
                app.save_settings()
        else:
            log_debug(f"GUI language unchanged: {lang_code}")

    app.gui_language_combobox.bind('<<ComboboxSelected>>', on_gui_language_changed)

    ttk.Button(frame, text=app.ui_lang.get_label("select_source_btn"), command=app.select_source_area, width=30).grid(row=1, column=0, padx=5, pady=5, sticky="w")
    ttk.Button(frame, text=app.ui_lang.get_label("select_target_btn"), command=app.select_target_area, width=30).grid(row=2, column=0, padx=5, pady=5, sticky="w")
    app.start_stop_btn = ttk.Button(frame, text=app.ui_lang.get_label("start_btn"), command=app.toggle_translation, width=30)
    app.start_stop_btn.grid(row=3, column=0, padx=5, pady=5, sticky="w")

    # Remove individual tab bindings and add a general binding in app_logic.py after tabs are created
    app.main_tab_start_button = app.start_stop_btn  # Store reference for the tab changed handler in app_logic.py
    ttk.Button(frame, text=app.ui_lang.get_label("hide_source_btn"), command=app.toggle_source_visibility, width=30).grid(row=4, column=0, padx=5, pady=5, sticky="w")
    ttk.Button(frame, text=app.ui_lang.get_label("hide_target_btn"), command=app.toggle_target_visibility, width=30).grid(row=5, column=0, padx=5, pady=5, sticky="w")
    ttk.Button(frame, text=app.ui_lang.get_label("clear_cache_btn"), command=app.clear_cache, width=30).grid(row=6, column=0, padx=5, pady=5, sticky="w")
    ttk.Button(frame, text=app.ui_lang.get_label("clear_debug_log_btn"), command=app.clear_debug_log, width=30).grid(row=7, column=0, padx=5, pady=5, sticky="w")

    # Debug log toggle button
    if app.debug_logging_enabled_var.get():
        initial_debug_toggle_text = app.ui_lang.get_label("toggle_debug_log_disable_btn")
    else:
        initial_debug_toggle_text = app.ui_lang.get_label("toggle_debug_log_enable_btn")
    app.debug_log_toggle_btn = ttk.Button(frame, text=initial_debug_toggle_text, command=app.toggle_debug_logging, width=30)
    app.debug_log_toggle_btn.grid(row=8, column=0, padx=5, pady=5, sticky="w")

    if app.KEYBOARD_AVAILABLE:
        shortcuts_frame = ttk.LabelFrame(frame, text=app.ui_lang.get_label("keyboard_shortcuts_title"))
        shortcuts_frame.grid(row=9, column=0, columnspan=2, padx=5, pady=10, sticky="ew")
        ttk.Label(shortcuts_frame, text="~ : " + app.ui_lang.get_label("shortcut_start_stop", "Start/Stop Translation")).grid(row=0, column=0, padx=10, pady=2, sticky="w")
        ttk.Label(shortcuts_frame, text="Alt+1 : " + app.ui_lang.get_label("shortcut_toggle_source", "Toggle Source Window Visibility")).grid(row=1, column=0, padx=10, pady=2, sticky="w")
        ttk.Label(shortcuts_frame, text="Alt+2 : " + app.ui_lang.get_label("shortcut_toggle_target", "Toggle Translation Window Visibility")).grid(row=2, column=0, padx=10, pady=2, sticky="w")
        ttk.Label(shortcuts_frame, text="Alt+S : " + app.ui_lang.get_label("shortcut_save_settings", "Save Settings")).grid(row=3, column=0, padx=10, pady=2, sticky="w")
        ttk.Label(shortcuts_frame, text="Alt+C : " + app.ui_lang.get_label("shortcut_clear_cache", "Clear Cache")).grid(row=4, column=0, padx=10, pady=2, sticky="w")
        ttk.Label(shortcuts_frame, text="Alt+L : " + app.ui_lang.get_label("shortcut_clear_log", "Clear Debug Log")).grid(row=5, column=0, padx=10, pady=(2,8), sticky="w")
        status_row = 10
        app.status_label = ttk.Label(frame, text=app.ui_lang.get_label("status_ready_hotkey"))
    else:
        status_row = 9
        app.status_label = ttk.Label(frame, text=app.ui_lang.get_label("status_ready"))

    app.status_label.grid(row=status_row, column=0, columnspan=2, padx=5, pady=5, sticky="w")
    frame.columnconfigure(0, weight=1)


def create_settings_tab(app):
    # Create a scrollable tab content frame
    scrollable_content = create_scrollable_tab(app.tab_control, app.ui_lang.get_label("settings_tab_title"))
    app.tab_settings = scrollable_content

    # Create the settings frame inside the scrollable area
    frame = ttk.LabelFrame(scrollable_content, text=app.ui_lang.get_label("settings_tab_title"))
    frame.pack(fill="both", expand=True, padx=10, pady=10)

    def validate_int_range(P, min_val, max_val):
        if P == "": return True
        try:
            value = int(P)
            return min_val <= value <= max_val
        except ValueError: return False

    def validate_float_range(P, min_val, max_val):
        if P == "": return True
        try:
            value = float(P)
            return min_val <= value <= max_val
        except ValueError: return False

    # Validation functions for parameters
    validate_beam_size = frame.register(lambda P: validate_int_range(P, 1, 50))
    validate_scan_interval = frame.register(lambda P: validate_int_range(P, 50, 2000))
    validate_custom_context_window = frame.register(lambda P: validate_int_range(P, 0, 10))
    validate_custom_ai_submit_interval = frame.register(lambda P: validate_int_range(P, 0, 5000))
    validate_custom_ai_ocr_image_quality = frame.register(lambda P: validate_int_range(P, 1, 100))
    validate_timeout = frame.register(lambda P: validate_int_range(P, 0, 60))
    validate_stability = frame.register(lambda P: validate_int_range(P, 0, 5))
    validate_paddleocr_min_score = frame.register(lambda P: validate_float_range(P, 0.0, 1.0))
    validate_font_size = frame.register(lambda P: validate_int_range(P, 8, 72))

    style = ttk.Style()

    # Keep combobox text readable on the white theme, including focused readonly fields.
    style.map('TCombobox',
        foreground=[
            ('readonly', 'focus', '#1f2937'),
            ('readonly', '!focus', '#1f2937'),
            ('!readonly', '#1f2937')
        ]
    )

    # Ensure no insertion cursor is visible
    style.configure('TCombobox',
                    insertwidth=0,
                    insertontime=0
                   )

    # Wrapper function to shift focus after combobox selection
    def create_combobox_handler_wrapper(original_handler_func):
        def wrapper(event):
            # Call the original handler
            original_handler_func(event)

            # Schedule focus shift to the parent tab frame (app.tab_settings)
            widget = event.widget
            if widget.winfo_exists() and app.tab_settings.winfo_exists():
                widget.after_idle(app.tab_settings.focus_set)
        return wrapper

    # Row 0: Translation Model Selection
    ttk.Label(frame, text=app.ui_lang.get_label("translation_model_label")).grid(row=0, column=0, padx=5, pady=5, sticky="w")

    translation_models_available_for_ui = [p["name"] for p in app.custom_ai_profiles.list_profiles(enabled_only=True)]
    log_debug(f"GUI Builder: Available translation models for UI: {translation_models_available_for_ui}")

    if not translation_models_available_for_ui:
        translation_models_available_for_ui.append(app.ui_lang.get_label("custom_ai_no_profiles", "Add an AI model profile"))

    app.translation_model_combobox = ttk.Combobox(frame, textvariable=app.translation_model_display_var,
                                           values=translation_models_available_for_ui, width=25, state='readonly')
    app.translation_model_combobox.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

    active_translation_profile = app.custom_ai_profiles.get_active_profile("translation")
    app.translation_model_display_var.set(active_translation_profile["name"] if active_translation_profile else translation_models_available_for_ui[0])

    def handle_translation_model_selection(event):
        selected_name = app.translation_model_display_var.get()
        for profile in app.custom_ai_profiles.list_profiles(enabled_only=True):
            if profile["name"] == selected_name:
                app.custom_ai_profiles.set_active_profile("translation", profile["id"])
                app.translation_model_var.set("custom_ai")
                break
        app.on_translation_model_selection_changed(event=event, initial_setup=False)
    app.translation_model_combobox.bind('<<ComboboxSelected>>',
        create_combobox_handler_wrapper(handle_translation_model_selection))

    # Row 0.5: OCR Model Selection
    ttk.Label(frame, text=app.ui_lang.get_label("ocr_model_label", "OCR Model")).grid(row=1, column=0, padx=5, pady=5, sticky="w")

    ocr_models_available_for_ui = build_ocr_model_display_options(app)


    app.ocr_model_combobox = ttk.Combobox(frame, textvariable=app.ocr_model_display_var,
                                        values=ocr_models_available_for_ui,
                                        width=25, state='readonly')
    app.ocr_model_combobox.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

    def on_ocr_model_changed(event):
        selected_display = app.ocr_model_display_var.get()
        log_debug(f"OCR model display changed to: {selected_display}")

        # Suppress traces during OCR model update to prevent premature saves
        app.suppress_traces()
        try:
            model_code, profile_id = resolve_ocr_model_display_selection(app, selected_display)
            if model_code == PADDLEOCR_MODEL_CODE:
                app.ocr_model_var.set(PADDLEOCR_MODEL_CODE)
                log_debug("OCR model set to paddleocr")
            elif model_code == 'custom_ai' and profile_id:
                app.custom_ai_profiles.set_active_profile("ocr", profile_id)
                app.ocr_model_var.set('custom_ai')
                log_debug(f"OCR model set to custom_ai profile: {selected_display}")
        finally:
            # Always restore traces
            app.restore_traces()

        # Update UI immediately for responsive feedback
        if hasattr(app, 'ui_interaction_handler'):
            app.ui_interaction_handler.update_ocr_model_ui()

        # Save settings after all variables are properly updated
        if app._fully_initialized:
            app.save_settings()

    app.ocr_model_combobox.bind('<<ComboboxSelected>>',
        create_combobox_handler_wrapper(on_ocr_model_changed))

    # Store the options for later use when updating language
    app.ocr_model_options = ocr_models_available_for_ui

    app.source_lang_label = ttk.Label(frame, text=app.ui_lang.get_label("source_lang_label"))
    app.source_lang_label.grid(row=2, column=0, padx=5, pady=5, sticky="w")
    app.source_lang_combobox = ttk.Combobox(frame, textvariable=app.source_display_var, width=25, state='readonly')
    app.source_lang_combobox.grid(row=2, column=1, padx=5, pady=5, sticky="ew")

    def on_source_lang_gui_changed(event):
        selected_display_name = app.source_display_var.get()
        active_model = app.translation_model_var.get()

        # Get current UI language - more robust detection
        current_ui_language = app.ui_lang.current_lang
        ui_language_for_lookup = 'polish' if current_ui_language == 'pol' else 'english'

        log_debug(f"Source lang GUI changed: selected='{selected_display_name}', model='{active_model}', ui_lang='{current_ui_language}', lookup='{ui_language_for_lookup}'")

        # Convert localized display name back to API code
        # Use the correct provider format for lookup
        provider_for_lookup = active_model  # Keep original format: 'google_api', 'deepl_api'
        api_code = app.language_manager.get_code_from_localized_name(
            selected_display_name, provider_for_lookup, ui_language_for_lookup)

        log_debug(f"Lookup result: '{selected_display_name}' -> '{api_code}'")

        if api_code:
            # Guard: Check if we're actually changing to a different value
            current_stored_value = None
            if active_model == 'google_api':
                current_stored_value = app.google_source_lang
            elif active_model == 'deepl_api':
                current_stored_value = app.deepl_source_lang
            elif active_model == 'gemini_api':
                current_stored_value = app.gemini_source_lang
            elif app.is_openai_model(active_model):
                current_stored_value = app.openai_source_lang
            elif active_model == 'custom_ai':
                current_stored_value = app.custom_source_lang

            # Only update and save if the value actually changed
            if api_code != current_stored_value:
                if active_model == 'google_api':
                    app.google_source_lang = api_code
                    log_debug(f"Google source lang set to: {api_code}")
                elif active_model == 'deepl_api':
                    app.deepl_source_lang = api_code
                    log_debug(f"DeepL source lang set to: {api_code}")
                    # Update DeepL model type options for beta language restriction
                    if hasattr(app, 'update_deepl_model_type_for_language'):
                        app.update_deepl_model_type_for_language()
                elif active_model == 'gemini_api':
                    app.gemini_source_lang = api_code
                    log_debug(f"Gemini source lang set to: {api_code}")
                elif app.is_openai_model(active_model):
                    app.openai_source_lang = api_code
                    log_debug(f"OpenAI source lang set to: {api_code}")
                elif active_model == 'custom_ai':
                    app.custom_source_lang = api_code
                    log_debug(f"Custom AI source lang set to: {api_code}")

                # Clear context for currently active provider when source language is changed
                if (hasattr(app, 'translation_handler') and
                    hasattr(app.translation_handler, '_clear_active_context')):
                    app.translation_handler._clear_active_context()

                app.source_lang_var.set(api_code)
                log_debug(f"Source lang GUI changed for {active_model}: Display='{selected_display_name}', API Code='{api_code}' - SAVING")
                app.save_settings()
            else:
                log_debug(f"Source lang unchanged for {active_model}: '{api_code}' - not saving")
        else:
            log_debug(f"ERROR: Could not find API code for source display '{selected_display_name}' / model '{active_model}' / ui_language '{ui_language_for_lookup}' - not saving invalid value")
            # Don't save invalid values - keep the previous valid selection

    app.source_lang_combobox.bind('<<ComboboxSelected>>',
        create_combobox_handler_wrapper(on_source_lang_gui_changed))


    app.target_lang_label = ttk.Label(frame, text=app.ui_lang.get_label("target_lang_label"))
    app.target_lang_label.grid(row=3, column=0, padx=5, pady=5, sticky="w")
    app.target_lang_combobox = ttk.Combobox(frame, textvariable=app.target_display_var, width=25, state='readonly')
    app.target_lang_combobox.grid(row=3, column=1, padx=5, pady=5, sticky="ew")

    def on_target_lang_gui_changed(event):
        selected_display_name = app.target_display_var.get()
        active_model = app.translation_model_var.get()

        # Get current UI language - more robust detection
        current_ui_language = app.ui_lang.current_lang
        ui_language_for_lookup = 'polish' if current_ui_language == 'pol' else 'english'

        log_debug(f"Target lang GUI changed: selected='{selected_display_name}', model='{active_model}', ui_lang='{current_ui_language}', lookup='{ui_language_for_lookup}'")

        # Convert localized display name back to API code
        # Use the correct provider format for lookup
        provider_for_lookup = active_model  # Keep original format: 'google_api', 'deepl_api'
        api_code = app.language_manager.get_code_from_localized_name(
            selected_display_name, provider_for_lookup, ui_language_for_lookup)

        log_debug(f"Lookup result: '{selected_display_name}' -> '{api_code}'")

        if api_code:
            # Guard: Check if we're actually changing to a different value
            current_stored_value = None
            if active_model == 'google_api':
                current_stored_value = app.google_target_lang
            elif active_model == 'deepl_api':
                current_stored_value = app.deepl_target_lang
            elif active_model == 'gemini_api':
                current_stored_value = app.gemini_target_lang
            elif app.is_openai_model(active_model):
                current_stored_value = app.openai_target_lang
            elif active_model == 'custom_ai':
                current_stored_value = app.custom_target_lang

            # Only update and save if the value actually changed
            if api_code != current_stored_value:
                if active_model == 'google_api':
                    app.google_target_lang = api_code
                    log_debug(f"Google target lang set to: {api_code}")
                elif active_model == 'deepl_api':
                    app.deepl_target_lang = api_code
                    log_debug(f"DeepL target lang set to: {api_code}")
                    # Update DeepL model type options for beta language restriction
                    if hasattr(app, 'update_deepl_model_type_for_language'):
                        app.update_deepl_model_type_for_language()
                elif active_model == 'gemini_api':
                    app.gemini_target_lang = api_code
                    log_debug(f"Gemini target lang set to: {api_code}")
                elif app.is_openai_model(active_model):
                    app.openai_target_lang = api_code
                    log_debug(f"OpenAI target lang set to: {api_code}")
                elif active_model == 'custom_ai':
                    app.custom_target_lang = api_code
                    log_debug(f"Custom AI target lang set to: {api_code}")

                # Clear context for currently active provider when target language is changed
                if (hasattr(app, 'translation_handler') and
                    hasattr(app.translation_handler, '_clear_active_context')):
                    app.translation_handler._clear_active_context()

                app.target_lang_var.set(api_code)
                log_debug(f"Target lang GUI changed for {active_model}: Display='{selected_display_name}', API Code='{api_code}' - SAVING")

                # Check if text direction changed and recreate overlay if needed
                try:
                    if hasattr(app, 'language_manager') and app.language_manager:
                        is_rtl = app.language_manager.is_rtl_language(api_code)
                        current_rtl_status = getattr(app.translation_text, 'is_rtl', False) if app.translation_text else False

                        if is_rtl != current_rtl_status:
                            log_debug(f"Text direction changed (RTL: {is_rtl}), recreating target overlay")
                            # Import and recreate the target overlay with new RTL settings (preserve position)
                            from overlay_manager import create_target_overlay_om
                            create_target_overlay_om(app)  # System recreation, preserve position
                except Exception as e:
                    log_debug(f"Error updating RTL configuration: {e}")

                app.save_settings()
            else:
                log_debug(f"Target lang unchanged for {active_model}: '{api_code}' - not saving")
        else:
            log_debug(f"ERROR: Could not find API code for target display '{selected_display_name}' / model '{active_model}' / ui_language '{ui_language_for_lookup}' - not saving invalid value")
            # Don't save invalid values - keep the previous valid selection
    app.target_lang_combobox.bind('<<ComboboxSelected>>',
        create_combobox_handler_wrapper(on_target_lang_gui_changed))


    app.marian_model_label = ttk.Label(frame, text=app.ui_lang.get_label("marian_model_label"))
    app.marian_model_label.grid(row=4, column=0, padx=5, pady=5, sticky="w")
    app.marian_model_combobox = ttk.Combobox(frame, textvariable=app.marian_model_display_var,
                                             values=app.marian_models_list, width=25, state='readonly')
    app.marian_model_combobox.grid(row=4, column=1, padx=5, pady=5, sticky="ew")

    def handle_marian_model_selection(event):
        # The marian_models_dict now contains localized display names as keys
        # so we can use the existing logic
        app.on_marian_model_selection_changed(event=event, initial_setup=False)
    app.marian_model_combobox.bind('<<ComboboxSelected>>',
        create_combobox_handler_wrapper(handle_marian_model_selection))

    app.google_api_key_label = ttk.Label(frame, text=app.ui_lang.get_label("google_api_key_label"))
    app.google_api_key_label.grid(row=5, column=0, padx=5, pady=5, sticky="w")
    app.google_api_key_entry = ttk.Entry(frame, textvariable=app.google_api_key_var, width=40, show="*")
    app.google_api_key_entry.grid(row=5, column=1, padx=5, pady=5, sticky="ew")
    # Set initial button text based on visibility
    initial_google_text = app.ui_lang.get_label("show_btn", "Show")
    if hasattr(app, 'google_api_key_visible') and app.google_api_key_visible:
        initial_google_text = app.ui_lang.get_label("hide_btn", "Hide")
    app.google_api_key_button = ttk.Button(frame, text=initial_google_text, width=5,
                                          command=lambda: app.toggle_api_key_visibility("google"))
    app.google_api_key_button.grid(row=5, column=2, padx=5, pady=5, sticky="w")

    app.deepl_api_key_label = ttk.Label(frame, text=app.ui_lang.get_label("deepl_api_key_label"))
    app.deepl_api_key_label.grid(row=6, column=0, padx=5, pady=5, sticky="w")
    app.deepl_api_key_entry = ttk.Entry(frame, textvariable=app.deepl_api_key_var, width=40, show="*")
    app.deepl_api_key_entry.grid(row=6, column=1, padx=5, pady=5, sticky="ew")
    # Set initial button text based on visibility
    initial_deepl_text = app.ui_lang.get_label("show_btn", "Show")
    if hasattr(app, 'deepl_api_key_visible') and app.deepl_api_key_visible:
        initial_deepl_text = app.ui_lang.get_label("hide_btn", "Hide")
    app.deepl_api_key_button = ttk.Button(frame, text=initial_deepl_text, width=5,
                                         command=lambda: app.toggle_api_key_visibility("deepl"))
    app.deepl_api_key_button.grid(row=6, column=2, padx=5, pady=5, sticky="w")

    # Gemini API Key input (only visible when Gemini is selected)
    app.gemini_api_key_label = ttk.Label(frame, text=app.ui_lang.get_label("gemini_api_key_label", "Gemini API Key"))
    app.gemini_api_key_label.grid(row=7, column=0, padx=5, pady=5, sticky="w")
    app.gemini_api_key_entry = ttk.Entry(frame, textvariable=app.gemini_api_key_var, width=40, show="*")
    app.gemini_api_key_entry.grid(row=7, column=1, padx=5, pady=5, sticky="ew")
    # Set initial button text based on visibility
    initial_gemini_text = app.ui_lang.get_label("show_btn", "Show")
    if hasattr(app, 'gemini_api_key_visible') and app.gemini_api_key_visible:
        initial_gemini_text = app.ui_lang.get_label("hide_btn", "Hide")
    app.gemini_api_key_button = ttk.Button(frame, text=initial_gemini_text, width=5,
                                          command=lambda: app.toggle_api_key_visibility("gemini"))
    app.gemini_api_key_button.grid(row=7, column=2, padx=5, pady=5, sticky="w")

    # Gemini Context Window Setting (only visible when Gemini is selected)
    app.gemini_context_window_label = ttk.Label(frame, text=app.ui_lang.get_label("gemini_context_window_label", "Context Window"))
    app.gemini_context_window_label.grid(row=8, column=0, padx=5, pady=5, sticky="w")

    context_window_options = [
        (0, app.ui_lang.get_label("gemini_context_window_0", "0 (Disabled)")),
        (1, app.ui_lang.get_label("gemini_context_window_1", "1 (Last subtitle)")),
        (2, app.ui_lang.get_label("gemini_context_window_2", "2 (Two subtitles)")),
        (3, app.ui_lang.get_label("gemini_context_window_3", "3 (Three subtitles)")),
        (4, app.ui_lang.get_label("gemini_context_window_4", "4 (Four subtitles)")),
        (5, app.ui_lang.get_label("gemini_context_window_5", "5 (Five subtitles)"))
    ]

    app.gemini_context_window_display_var = tk.StringVar()
    # Set initial display value based on current setting
    current_context_window = app.gemini_context_window_var.get()
    for value, display in context_window_options:
        if value == current_context_window:
            app.gemini_context_window_display_var.set(display)
            break
    else:
        # Fallback if current setting doesn't match any option
        app.gemini_context_window_display_var.set(context_window_options[1][1])  # Default to 1

    app.gemini_context_window_combobox = ttk.Combobox(frame, textvariable=app.gemini_context_window_display_var,
                                                     values=[display for _, display in context_window_options],
                                                     width=25, state='readonly')
    app.gemini_context_window_combobox.grid(row=8, column=1, padx=5, pady=5, sticky="ew")

    def on_gemini_context_window_changed(event):
        selected_display = app.gemini_context_window_display_var.get()
        # Find the corresponding value
        for value, display in context_window_options:
            if display == selected_display:
                app.gemini_context_window_var.set(value)
                log_debug(f"Gemini context window changed to: {value} (display: {display})")
                # Clear active context when Gemini context window changes
                if hasattr(app, 'translation_handler') and hasattr(app.translation_handler, '_clear_active_context'):
                    app.translation_handler._clear_active_context()
                    log_debug(f"Active provider context cleared due to Gemini context window setting change")
                if app._fully_initialized:
                    app.save_settings()
                break

    app.gemini_context_window_combobox.bind('<<ComboboxSelected>>',
        create_combobox_handler_wrapper(on_gemini_context_window_changed))

    # OpenAI API Key input (only visible when OpenAI is selected)
    app.openai_api_key_label = ttk.Label(frame, text=app.ui_lang.get_label("openai_api_key_label", "OpenAI API Key"))
    app.openai_api_key_label.grid(row=9, column=0, padx=5, pady=5, sticky="w")
    app.openai_api_key_entry = ttk.Entry(frame, textvariable=app.openai_api_key_var, width=40, show="*")
    app.openai_api_key_entry.grid(row=9, column=1, padx=5, pady=5, sticky="ew")
    # Set initial button text based on visibility
    initial_openai_text = app.ui_lang.get_label("show_btn", "Show")
    if hasattr(app, 'openai_api_key_visible') and app.openai_api_key_visible:
        initial_openai_text = app.ui_lang.get_label("hide_btn", "Hide")
    app.openai_api_key_button = ttk.Button(frame, text=initial_openai_text, width=5,
                                          command=lambda: app.toggle_api_key_visibility("openai"))
    app.openai_api_key_button.grid(row=9, column=2, padx=5, pady=5, sticky="w")

    # OpenAI Context Window Setting (only visible when OpenAI is selected)
    app.openai_context_window_label = ttk.Label(frame, text=app.ui_lang.get_label("openai_context_window_label", "Context Window"))
    app.openai_context_window_label.grid(row=10, column=0, padx=5, pady=5, sticky="w")

    openai_context_window_options = [
        (0, app.ui_lang.get_label("openai_context_window_0", "0 (Disabled)")),
        (1, app.ui_lang.get_label("openai_context_window_1", "1 (Last subtitle)")),
        (2, app.ui_lang.get_label("openai_context_window_2", "2 (Two subtitles)")),
        (3, app.ui_lang.get_label("openai_context_window_3", "3 (Three subtitles)")),
        (4, app.ui_lang.get_label("openai_context_window_4", "4 (Four subtitles)")),
        (5, app.ui_lang.get_label("openai_context_window_5", "5 (Five subtitles)"))
    ]

    app.openai_context_window_display_var = tk.StringVar()
    # Set initial display value based on current setting
    current_openai_context_window = app.openai_context_window_var.get()
    for value, display in openai_context_window_options:
        if value == current_openai_context_window:
            app.openai_context_window_display_var.set(display)
            break
    else:
        # Fallback if current setting doesn't match any option
        app.openai_context_window_display_var.set(openai_context_window_options[2][1])  # Default to 2

    app.openai_context_window_combobox = ttk.Combobox(frame, textvariable=app.openai_context_window_display_var,
                                                     values=[display for _, display in openai_context_window_options],
                                                     width=25, state='readonly')
    app.openai_context_window_combobox.grid(row=10, column=1, padx=5, pady=5, sticky="ew")

    def on_openai_context_window_changed(event):
        selected_display = app.openai_context_window_display_var.get()
        # Find the corresponding value
        for value, display in openai_context_window_options:
            if display == selected_display:
                app.openai_context_window_var.set(value)
                log_debug(f"OpenAI context window changed to: {value} (display: {display})")
                # Clear active context when OpenAI context window changes
                if hasattr(app, 'translation_handler') and hasattr(app.translation_handler, '_clear_active_context'):
                    app.translation_handler._clear_active_context()
                    log_debug(f"Active provider context cleared due to OpenAI context window setting change")
                if app._fully_initialized:
                    app.save_settings()
                break

    app.openai_context_window_combobox.bind('<<ComboboxSelected>>',
        create_combobox_handler_wrapper(on_openai_context_window_changed))

    # DeepL Model Type Selection (only visible when DeepL is selected)
    app.deepl_model_type_label = ttk.Label(frame, text=app.ui_lang.get_label("deepl_model_type_label", "Quality"))
    app.deepl_model_type_label.grid(row=11, column=0, padx=5, pady=5, sticky="w")

    # Create model type options with user-friendly names
    deepl_model_options = [
        ("latency_optimized", app.ui_lang.get_label("deepl_classic_model", "Classic")),
        ("quality_optimized", app.ui_lang.get_label("deepl_nextgen_model", "Next-gen"))
    ]

    app.deepl_model_display_var = tk.StringVar()
    # Set initial display value based on current setting
    current_model_type = app.deepl_model_type_var.get()
    for value, display in deepl_model_options:
        if value == current_model_type:
            app.deepl_model_display_var.set(display)
            break
    else:
        # Fallback if current setting doesn't match any option
        app.deepl_model_display_var.set(deepl_model_options[0][1])  # Default to Classic

    app.deepl_model_type_combobox = ttk.Combobox(frame, textvariable=app.deepl_model_display_var,
                                               values=[display for _, display in deepl_model_options],
                                               width=25, state='readonly')
    app.deepl_model_type_combobox.grid(row=11, column=1, padx=5, pady=5, sticky="ew")

    def on_deepl_model_type_changed(event):
        selected_display = app.deepl_model_display_var.get()
        # Find the corresponding value
        for value, display in deepl_model_options:
            if display == selected_display:
                app.deepl_model_type_var.set(value)
                log_debug(f"DeepL model type changed to: {value} (display: {display})")
                if app._fully_initialized:
                    app.save_settings()
                break

    app.deepl_model_type_combobox.bind('<<ComboboxSelected>>',
        create_combobox_handler_wrapper(on_deepl_model_type_changed))

    # Store the options for later use when updating language
    app.deepl_model_options = deepl_model_options

    # DeepL Context Window Setting (only visible when DeepL is selected)
    app.deepl_context_window_label = ttk.Label(frame, text=app.ui_lang.get_label("deepl_context_window_label", "Context Window"))
    app.deepl_context_window_label.grid(row=12, column=0, padx=5, pady=5, sticky="w")

    deepl_context_window_options = [
        (0, app.ui_lang.get_label("deepl_context_window_0", "0 (Disabled)")),
        (1, app.ui_lang.get_label("deepl_context_window_1", "1 (Last subtitle)")),
        (2, app.ui_lang.get_label("deepl_context_window_2", "2 (Two subtitles)")),
        (3, app.ui_lang.get_label("deepl_context_window_3", "3 (Three subtitles)"))
    ]

    app.deepl_context_window_display_var = tk.StringVar()
    # Set initial display value based on current setting
    current_deepl_context_window = app.deepl_context_window_var.get()
    for value, display in deepl_context_window_options:
        if value == current_deepl_context_window:
            app.deepl_context_window_display_var.set(display)
            break
    else:
        # Fallback if current setting doesn't match any option
        app.deepl_context_window_display_var.set(deepl_context_window_options[2][1])  # Default to 2

    app.deepl_context_window_combobox = ttk.Combobox(frame, textvariable=app.deepl_context_window_display_var,
                                                     values=[display for _, display in deepl_context_window_options],
                                                     width=25, state='readonly')
    app.deepl_context_window_combobox.grid(row=12, column=1, padx=5, pady=5, sticky="ew")

    def on_deepl_context_window_changed(event):
        selected_display = app.deepl_context_window_display_var.get()
        # Find the corresponding value
        for value, display in deepl_context_window_options:
            if display == selected_display:
                app.deepl_context_window_var.set(value)
                log_debug(f"DeepL context window changed to: {value} (display: {display})")
                # Clear DeepL context when context window changes
                if hasattr(app, 'translation_handler') and hasattr(app.translation_handler, '_clear_deepl_context'):
                    app.translation_handler._clear_deepl_context()
                if app._fully_initialized:
                    app.save_settings()
                break

    app.deepl_context_window_combobox.bind('<<ComboboxSelected>>',
        create_combobox_handler_wrapper(on_deepl_context_window_changed))

    # Store the options for later use when updating language
    app.deepl_context_window_options = deepl_context_window_options

    # Function to update DeepL model type options when language changes
    def update_deepl_model_type_for_language():
        if hasattr(app, 'deepl_model_type_combobox') and app.deepl_model_type_combobox.winfo_exists():
            from constants import DEEPL_BETA_LANGUAGES

            # Get current model type setting
            current_model_type = app.deepl_model_type_var.get()

            # Check if either source or target language is a beta language
            is_beta_translation = False
            deepl_source = getattr(app, 'deepl_source_lang', '')
            deepl_target = getattr(app, 'deepl_target_lang', '')

            if deepl_source and deepl_source.upper() in DEEPL_BETA_LANGUAGES:
                is_beta_translation = True
                log_debug(f"DeepL source language {deepl_source} is a beta language")
            if deepl_target and deepl_target.upper() in DEEPL_BETA_LANGUAGES:
                is_beta_translation = True
                log_debug(f"DeepL target language {deepl_target} is a beta language")

            # Build options based on whether beta languages are involved
            if is_beta_translation:
                # Beta languages only support quality_optimized
                new_deepl_model_options = [
                    ("quality_optimized", app.ui_lang.get_label("deepl_nextgen_model", "Next-gen"))
                ]
                # Force quality_optimized for beta languages
                if current_model_type != "quality_optimized":
                    app.deepl_model_type_var.set("quality_optimized")
                    current_model_type = "quality_optimized"
                    log_debug("Forced model type to quality_optimized for beta language")
            else:
                # Non-beta languages support both models
                new_deepl_model_options = [
                    ("latency_optimized", app.ui_lang.get_label("deepl_classic_model", "Classic")),
                    ("quality_optimized", app.ui_lang.get_label("deepl_nextgen_model", "Next-gen"))
                ]

            # Update combobox values
            app.deepl_model_type_combobox['values'] = [display for _, display in new_deepl_model_options]

            # Restore selection based on current setting
            for value, display in new_deepl_model_options:
                if value == current_model_type:
                    app.deepl_model_display_var.set(display)
                    break
            else:
                # Fallback to first option if current setting not found
                app.deepl_model_display_var.set(new_deepl_model_options[0][1])

            # Update stored options
            app.deepl_model_options = new_deepl_model_options

            if is_beta_translation:
                log_debug(f"DeepL model type restricted to Next-gen only (beta language)")
            else:
                log_debug(f"DeepL model type options updated (both Classic and Next-gen available)")

    # Store function reference for calling during language updates
    app.update_deepl_model_type_for_language = update_deepl_model_type_for_language

    # Function to update DeepL context window options when language changes
    def update_deepl_context_window_for_language():
        if hasattr(app, 'deepl_context_window_combobox') and app.deepl_context_window_combobox.winfo_exists():
            # Get current context window setting
            current_deepl_context_window = app.deepl_context_window_var.get()

            # Update options with new language
            new_deepl_context_window_options = [
                (0, app.ui_lang.get_label("deepl_context_window_0", "0 (Disabled)")),
                (1, app.ui_lang.get_label("deepl_context_window_1", "1 (Last subtitle)")),
                (2, app.ui_lang.get_label("deepl_context_window_2", "2 (Two subtitles)")),
                (3, app.ui_lang.get_label("deepl_context_window_3", "3 (Three subtitles)"))
            ]

            # Update combobox values
            app.deepl_context_window_combobox['values'] = [display for _, display in new_deepl_context_window_options]

            # Restore selection based on current setting
            for value, display in new_deepl_context_window_options:
                if value == current_deepl_context_window:
                    app.deepl_context_window_display_var.set(display)
                    break
            else:
                # Fallback to default option if current setting not found
                app.deepl_context_window_display_var.set(new_deepl_context_window_options[2][1])  # Default to 2

            # Update stored options
            app.deepl_context_window_options = new_deepl_context_window_options
            log_debug(f"Updated DeepL context window options for language change")

    # Store function reference for calling during language updates
    app.update_deepl_context_window_for_language = update_deepl_context_window_for_language

    # Function to update Gemini context window options when language changes
    def update_gemini_context_window_for_language():
        if hasattr(app, 'gemini_context_window_combobox') and app.gemini_context_window_combobox.winfo_exists():
            # Get current context window setting
            current_context_window = app.gemini_context_window_var.get()

            # Update options with new language
            new_context_window_options = [
                (0, app.ui_lang.get_label("gemini_context_window_0", "0 (Disabled)")),
                (1, app.ui_lang.get_label("gemini_context_window_1", "1 (Last subtitle)")),
                (2, app.ui_lang.get_label("gemini_context_window_2", "2 (Two subtitles)")),
                (3, app.ui_lang.get_label("gemini_context_window_3", "3 (Three subtitles)")),
                (4, app.ui_lang.get_label("gemini_context_window_4", "4 (Four subtitles)")),
                (5, app.ui_lang.get_label("gemini_context_window_5", "5 (Five subtitles)"))
            ]

            # Update combobox values
            app.gemini_context_window_combobox['values'] = [display for _, display in new_context_window_options]

            # Restore selection based on current setting
            for value, display in new_context_window_options:
                if value == current_context_window:
                    app.gemini_context_window_display_var.set(display)
                    break
            else:
                # Fallback to default option if current setting not found
                app.gemini_context_window_display_var.set(new_context_window_options[1][1])  # Default to 1

            log_debug(f"Updated Gemini context window options for language change")

    # Store function reference for calling during language updates
    app.update_gemini_context_window_for_language = update_gemini_context_window_for_language

    # Function to update Gemini labels when language changes
    def update_gemini_labels_for_language():
        if hasattr(app, 'gemini_enable_api_log_checkbox') and app.gemini_enable_api_log_checkbox.winfo_exists():
            app.gemini_enable_api_log_checkbox.config(text=app.ui_lang.get_label("gemini_enable_api_log_checkbox", "Enable API Log"))
        if hasattr(app, 'gemini_reset_log_button') and app.gemini_reset_log_button.winfo_exists():
            app.gemini_reset_log_button.config(text=app.ui_lang.get_label("gemini_reset_log_button", "Reset"))
        if hasattr(app, 'gemini_refresh_stats_button') and app.gemini_refresh_stats_button.winfo_exists():
            app.gemini_refresh_stats_button.config(text=app.ui_lang.get_label("gemini_refresh_stats_button", "Refresh"))
        if hasattr(app, 'gemini_total_words_label') and app.gemini_total_words_label.winfo_exists():
            app.gemini_total_words_label.config(text=app.ui_lang.get_label("gemini_total_words_label", "Total Words"))
        if hasattr(app, 'gemini_total_cost_label') and app.gemini_total_cost_label.winfo_exists():
            app.gemini_total_cost_label.config(text=app.ui_lang.get_label("gemini_total_cost_label", "Total Cost"))

        # Update cost format when language changes
        if hasattr(app, 'gemini_total_cost_var') and app.gemini_total_cost_var is not None:
            try:
                # Get current cost value and reformat it
                current_value = app.gemini_total_cost_var.get()
                # Parse the current cost regardless of format
                import re
                cost_match = re.search(r'[\d,\.]+', current_value)
                if cost_match:
                    cost_str = cost_match.group().replace(',', '.')  # Normalize to decimal point
                    cost_value = float(cost_str)
                    app.gemini_total_cost_var.set(app.format_cost_for_display(cost_value))
            except Exception as e:
                log_debug(f"Error updating cost format for language change: {e}")
                # Fallback to default formatted value
                app.gemini_total_cost_var.set(app.format_cost_for_display(0.0))

        log_debug("Updated Gemini labels for language change")

    # Store function reference for calling during language updates
    app.update_gemini_labels_for_language = update_gemini_labels_for_language

    # Function to update DeepL usage labels when language changes
    def update_deepl_usage_for_language():
        if hasattr(app, 'deepl_usage_label') and app.deepl_usage_label.winfo_exists():
            app.deepl_usage_label.config(text=app.ui_lang.get_label("deepl_usage_label", "DeepL Usage"))

        log_debug("Updated DeepL usage labels for language change")

    # Store function reference for calling during language updates
    app.update_deepl_usage_for_language = update_deepl_usage_for_language

    app.models_file_label = ttk.Label(frame, text=app.ui_lang.get_label("models_file_label"))
    app.models_file_label.grid(row=14, column=0, padx=5, pady=5, sticky="w")
    app.models_file_frame = ttk.Frame(frame)
    app.models_file_frame.grid(row=14, column=1, columnspan=2, padx=5, pady=5, sticky="ew")
    app.models_file_entry = ttk.Entry(app.models_file_frame, textvariable=app.models_file_var)
    app.models_file_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
    app.models_file_button = ttk.Button(app.models_file_frame, text=app.ui_lang.get_label("browse_btn"), command=app.browse_marian_models_file)
    app.models_file_button.pack(side=tk.RIGHT, padx=(5,0))

    app.beam_size_label = ttk.Label(frame, text=app.ui_lang.get_label("beam_size_label"))
    app.beam_size_label.grid(row=15, column=0, padx=5, pady=5, sticky="w")
    app.beam_spinbox = ttk.Spinbox(frame, from_=1, to=50, textvariable=app.num_beams_var, width=10,
                                  validate="key", validatecommand=(validate_beam_size, '%P'))
    app.beam_spinbox.grid(row=15, column=1, padx=5, pady=5, sticky="w")
    def on_beam_spinbox_focus_out(event):
        try:
            value = int(app.num_beams_var.get())
            clamped = max(1, min(50, value))
            if clamped != value: app.num_beams_var.set(clamped)
            app.update_marian_beam_value()
        except (ValueError, tk.TclError): app.num_beams_var.set(2)
        app.save_settings()
    app.beam_spinbox.bind("<FocusOut>", on_beam_spinbox_focus_out)

    app.ai_profile_selected_id = None
    app.ai_profile_name_var = tk.StringVar()
    app.ai_profile_url_var = tk.StringVar()
    app.ai_profile_key_var = tk.StringVar()
    app.ai_profile_model_var = tk.StringVar()
    app.ai_profile_structured_output_mode_var = tk.StringVar(value="auto")
    app.ai_profile_reasoning_effort_var = tk.StringVar(
        value=_reasoning_effort_display_value(app, CUSTOM_AI_REASONING_EFFORT_LOW)
    )
    app.ai_profile_model_values = []

    def get_profile_by_name(name):
        for profile_item in app.custom_ai_profiles.list_profiles():
            if profile_item["name"] == name:
                return profile_item
        return None

    def refresh_custom_profile_controls(select_profile_id=None):
        profiles = app.custom_ai_profiles.list_profiles()
        enabled_profiles = [p for p in profiles if p.get("enabled", True)]
        profile_names = [p["name"] for p in profiles]
        enabled_names = [p["name"] for p in enabled_profiles]

        app.ai_profile_name_combobox.config(values=profile_names)
        app.ai_profile_model_values = []
        app.ai_profile_model_combobox.config(values=[])

        translation_names = enabled_names or [app.ui_lang.get_label("custom_ai_no_profiles", "Add an AI model profile")]
        app.translation_model_combobox.config(values=translation_names)
        active_translation = app.custom_ai_profiles.get_active_profile("translation")
        app.translation_model_display_var.set(active_translation["name"] if active_translation else translation_names[0])

        ocr_names = [get_paddleocr_ocr_display_name(app)]
        ocr_names.extend(enabled_names)
        app.ocr_model_combobox.config(values=ocr_names)
        active_ocr = app.custom_ai_profiles.get_active_profile("ocr")
        if app.ocr_model_var.get() == "custom_ai" and active_ocr:
            app.ocr_model_display_var.set(active_ocr["name"])
        elif app.ocr_model_var.get() == PADDLEOCR_MODEL_CODE:
            app.ocr_model_display_var.set(get_paddleocr_ocr_display_name(app))
        elif app.ocr_model_var.get() != "custom_ai":
            app.ocr_model_display_var.set(get_paddleocr_ocr_display_name(app))

        profile_to_load = None
        if select_profile_id:
            profile_to_load = app.custom_ai_profiles.get_profile(select_profile_id)
        if not profile_to_load and app.ai_profile_selected_id:
            profile_to_load = app.custom_ai_profiles.get_profile(app.ai_profile_selected_id)
        if not profile_to_load and profiles:
            profile_to_load = profiles[0]
        load_profile(profile_to_load)

    def load_profile(profile):
        if not profile:
            app.ai_profile_selected_id = None
            app.ai_profile_name_var.set("")
            app.ai_profile_url_var.set("")
            app.ai_profile_key_var.set("")
            app.ai_profile_model_var.set("")
            app.ai_profile_structured_output_mode_var.set("auto")
            app.ai_profile_reasoning_effort_var.set(
                _reasoning_effort_display_value(
                    app,
                    CUSTOM_AI_REASONING_EFFORT_LOW,
                )
            )
            return
        app.ai_profile_selected_id = profile["id"]
        app.ai_profile_name_var.set(profile.get("name", ""))
        app.ai_profile_url_var.set(profile.get("base_url", ""))
        app.ai_profile_key_var.set(profile.get("api_key", ""))
        app.ai_profile_model_var.set(profile.get("model", ""))
        structured_output_mode = str(
            profile.get("structured_output_mode") or "auto"
        ).strip().lower()
        if structured_output_mode not in {"off", "auto", "strict"}:
            structured_output_mode = "auto"
        app.ai_profile_structured_output_mode_var.set(
            structured_output_mode
        )
        app.ai_profile_reasoning_effort_var.set(
            _reasoning_effort_display_value(
                app,
                profile.get("reasoning_effort")
                or profile.get("model_reasoning_effort"),
            )
        )

    def on_profile_name_selected(event=None):
        load_profile(get_profile_by_name(app.ai_profile_name_var.get()))

    def build_profile_from_form():
        return build_custom_ai_profile_values_from_form(app)

    def add_profile_form():
        load_profile(None)
        app.ai_profile_name_combobox.focus_set()

    def save_profile_form():
        try:
            values = build_profile_from_form()
            if app.ai_profile_selected_id and app.custom_ai_profiles.get_profile(app.ai_profile_selected_id):
                profile = app.custom_ai_profiles.update_profile(app.ai_profile_selected_id, **values)
            else:
                profile = app.custom_ai_profiles.add_profile(**values)
            refresh_custom_profile_controls(profile["id"])
            app.save_settings()
        except Exception as e:
            messagebox.showerror(app.ui_lang.get_label("profile_error_title", "Profile Error"), str(e), parent=app.root)

    def delete_profile_form():
        profile = app.custom_ai_profiles.get_profile(app.ai_profile_selected_id) if app.ai_profile_selected_id else get_profile_by_name(app.ai_profile_name_var.get())
        if not profile:
            return
        title = app.ui_lang.get_label("delete_profile_title", "Delete Profile")
        message = app.ui_lang.get_label("delete_profile_confirm", "Delete profile '{0}'?").format(profile["name"])
        if messagebox.askyesno(title, message, parent=app.root):
            app.custom_ai_profiles.delete_profile(profile["id"])
            app.ai_profile_selected_id = None
            refresh_custom_profile_controls()
            app.save_settings()

    def test_profile_form():
        try:
            values = build_profile_from_form()
            app.custom_ai_profiles._validate_profile(values)
        except Exception as e:
            messagebox.showerror(app.ui_lang.get_label("connection_test_failed_title", "Connection Test Failed"), str(e), parent=app.root)
            return

        def task():
            latency_mode = app.get_custom_ai_latency_mode() if hasattr(app, 'get_custom_ai_latency_mode') else "safe"
            return app.translation_handler.custom_ai_provider.test_profile(values, latency_mode=latency_mode)

        def on_success(result):
            response_text, duration = result
            title = app.ui_lang.get_label("connection_test_title", "Connection Test")
            message = app.ui_lang.get_label("connection_test_success", "Response: {0}\nDuration: {1:.2f}s").format(response_text, duration)
            messagebox.showinfo(title, message, parent=app.root)

        run_profile_network_task_async(
            app,
            getattr(app, "ai_profile_test_button", None),
            task,
            on_success,
            app.ui_lang.get_label("connection_test_failed_title", "Connection Test Failed"),
        )

    def fetch_model_list_form():
        try:
            values = build_profile_from_form()
            if not values["base_url"]:
                raise ValueError("API URL is required")
            if not values["api_key"]:
                raise ValueError("API key is required")
        except Exception as e:
            messagebox.showerror(app.ui_lang.get_label("model_list_failed_title", "Model List Failed"), str(e), parent=app.root)
            return

        def task():
            latency_mode = app.get_custom_ai_latency_mode() if hasattr(app, 'get_custom_ai_latency_mode') else "safe"
            return app.translation_handler.custom_ai_provider.fetch_models(values, latency_mode=latency_mode)

        def on_success(models):
            app.ai_profile_model_values = models
            current_model_filter = app.ai_profile_model_var.get().strip()
            app.ai_profile_model_combobox.config(values=filter_model_values(models, current_model_filter))
            if models and not current_model_filter:
                app.ai_profile_model_var.set(models[0])
            title = app.ui_lang.get_label("model_list_title", "Model List")
            message = app.ui_lang.get_label("model_list_success", "Fetched {0} models.").format(len(models))
            messagebox.showinfo(title, message, parent=app.root)

        run_profile_network_task_async(
            app,
            getattr(app, "ai_profile_fetch_models_button", None),
            task,
            on_success,
            app.ui_lang.get_label("model_list_failed_title", "Model List Failed"),
        )

    def filter_model_list_form(event=None):
        models = getattr(app, "ai_profile_model_values", [])
        if not models:
            return
        query = app.ai_profile_model_var.get()
        app.ai_profile_model_combobox.config(values=filter_model_values(models, query))

    app.ai_profiles_frame = ttk.LabelFrame(frame, text=app.ui_lang.get_label("ai_model_profiles_title", "AI Model Profiles"))
    app.ai_profiles_frame.grid(row=16, column=0, columnspan=3, padx=5, pady=5, sticky="ew")
    app.ai_profiles_frame.columnconfigure(1, weight=1)

    ttk.Label(app.ai_profiles_frame, text=app.ui_lang.get_label("ai_profile_name_label", "name")).grid(row=0, column=0, padx=5, pady=3, sticky="w")
    app.ai_profile_name_combobox = ttk.Combobox(app.ai_profiles_frame, textvariable=app.ai_profile_name_var, width=48)
    app.ai_profile_name_combobox.grid(row=0, column=1, padx=5, pady=3, sticky="ew")
    app.ai_profile_name_combobox.bind("<<ComboboxSelected>>", on_profile_name_selected)

    ttk.Label(app.ai_profiles_frame, text=app.ui_lang.get_label("ai_profile_url_label", "api url")).grid(row=1, column=0, padx=5, pady=3, sticky="w")
    ttk.Entry(app.ai_profiles_frame, textvariable=app.ai_profile_url_var).grid(row=1, column=1, padx=5, pady=3, sticky="ew")

    ttk.Label(app.ai_profiles_frame, text=app.ui_lang.get_label("ai_profile_key_label", "api key")).grid(row=2, column=0, padx=5, pady=3, sticky="w")
    ttk.Entry(app.ai_profiles_frame, textvariable=app.ai_profile_key_var, show="*").grid(row=2, column=1, padx=5, pady=3, sticky="ew")

    ttk.Label(app.ai_profiles_frame, text=app.ui_lang.get_label("ai_profile_model_label", "model")).grid(row=3, column=0, padx=5, pady=3, sticky="w")
    model_frame = ttk.Frame(app.ai_profiles_frame)
    model_frame.grid(row=3, column=1, padx=5, pady=3, sticky="ew")
    model_frame.columnconfigure(0, weight=1)
    app.ai_profile_model_combobox = ttk.Combobox(model_frame, textvariable=app.ai_profile_model_var)
    app.ai_profile_model_combobox.grid(row=0, column=0, sticky="ew")
    app.ai_profile_model_combobox.bind("<KeyRelease>", filter_model_list_form)
    app.ai_profile_fetch_models_button = ttk.Button(model_frame, text=app.ui_lang.get_label("fetch_model_list_btn", "Fetch Models"), command=fetch_model_list_form)
    app.ai_profile_fetch_models_button.grid(row=0, column=1, padx=(5, 0))

    ttk.Label(
        app.ai_profiles_frame,
        text=app.ui_lang.get_label(
            "ai_profile_structured_output_label",
            "Structured output",
        ),
    ).grid(row=4, column=0, padx=5, pady=3, sticky="w")
    app.ai_profile_structured_output_combobox = ttk.Combobox(
        app.ai_profiles_frame,
        textvariable=app.ai_profile_structured_output_mode_var,
        values=["auto", "strict", "off"],
        state="readonly",
        width=12,
    )
    app.ai_profile_structured_output_combobox.grid(
        row=4,
        column=1,
        padx=5,
        pady=3,
        sticky="w",
    )

    ttk.Label(
        app.ai_profiles_frame,
        text=app.ui_lang.get_label(
            "ai_profile_reasoning_effort_label",
            "Reasoning effort",
        ),
    ).grid(row=5, column=0, padx=5, pady=3, sticky="w")
    app.ai_profile_reasoning_effort_combobox = ttk.Combobox(
        app.ai_profiles_frame,
        textvariable=app.ai_profile_reasoning_effort_var,
        values=_reasoning_effort_display_values(app),
        state="readonly",
        width=12,
    )
    app.ai_profile_reasoning_effort_combobox.grid(
        row=5,
        column=1,
        padx=5,
        pady=3,
        sticky="w",
    )

    profile_buttons = ttk.Frame(app.ai_profiles_frame, padding=(10, 8))
    profile_buttons.grid(row=0, column=2, rowspan=6, padx=(10, 8), pady=6, sticky="nsew")
    ttk.Button(
        profile_buttons,
        text=app.ui_lang.get_label("add_btn", "Add"),
        command=add_profile_form,
        style="ProfileAdd.TButton",
        width=12,
    ).pack(fill=tk.X, pady=(0, 7), ipady=1)
    ttk.Button(
        profile_buttons,
        text=app.ui_lang.get_label("save_btn", "Save"),
        command=save_profile_form,
        style="ProfileSave.TButton",
        width=12,
    ).pack(fill=tk.X, pady=(0, 7), ipady=1)
    ttk.Button(
        profile_buttons,
        text=app.ui_lang.get_label("delete_btn", "Delete"),
        command=delete_profile_form,
        style="ProfileDelete.TButton",
        width=12,
    ).pack(fill=tk.X, pady=(0, 7), ipady=1)
    app.ai_profile_test_button = ttk.Button(
        profile_buttons,
        text=app.ui_lang.get_label("test_btn", "Test"),
        command=test_profile_form,
        style="ProfileTest.TButton",
        width=12,
    )
    app.ai_profile_test_button.pack(fill=tk.X, pady=0, ipady=1)

    refresh_custom_profile_controls()

    app.marian_explanation_labels = []
    app.custom_context_window_label = ttk.Label(frame, text=app.ui_lang.get_label("custom_context_window_label", "Custom AI Context Window"))
    app.custom_context_window_label.grid(row=17, column=0, padx=5, pady=5, sticky="w")
    app.custom_context_window_spinbox = ttk.Spinbox(
        frame,
        from_=0,
        to=10,
        textvariable=app.custom_context_window_var,
        width=10,
        validate="key",
        validatecommand=(validate_custom_context_window, '%P'),
    )
    app.custom_context_window_spinbox.grid(row=17, column=1, padx=5, pady=5, sticky="w")

    def on_custom_context_window_focus_out(event):
        try:
            value = int(app.custom_context_window_var.get())
            clamped = max(0, min(10, value))
            if clamped != value:
                app.custom_context_window_var.set(clamped)
        except (ValueError, tk.TclError):
            app.custom_context_window_var.set(5)
        if hasattr(app, 'translation_handler') and hasattr(app.translation_handler, '_clear_active_context'):
            app.translation_handler._clear_active_context()
        app.save_settings()

    app.custom_context_window_spinbox.bind("<FocusOut>", on_custom_context_window_focus_out)

    app.custom_ai_latency_mode_label = ttk.Label(
        frame,
        text=app.ui_lang.get_label("custom_ai_latency_mode_label", "Custom AI response mode:"),
    )
    app.custom_ai_latency_mode_label.grid(row=18, column=0, padx=5, pady=5, sticky="w")
    custom_ai_latency_mode_options = [
        ("none", app.ui_lang.get_label("custom_ai_latency_mode_none", "None")),
        ("safe", app.ui_lang.get_label("custom_ai_latency_mode_safe", "Stable low latency")),
        ("stream", app.ui_lang.get_label("custom_ai_latency_mode_stream", "Streaming subtitles")),
        ("race", app.ui_lang.get_label("custom_ai_latency_mode_race", "Fastest endpoint")),
        ("adaptive", app.ui_lang.get_label("custom_ai_latency_mode_adaptive", "Adaptive")),
    ]
    app.custom_ai_latency_mode_display_var = tk.StringVar()
    current_latency_mode = app.get_custom_ai_latency_mode() if hasattr(app, 'get_custom_ai_latency_mode') else app.custom_ai_latency_mode_var.get()
    for value, display in custom_ai_latency_mode_options:
        if value == current_latency_mode:
            app.custom_ai_latency_mode_display_var.set(display)
            break
    else:
        app.custom_ai_latency_mode_display_var.set(custom_ai_latency_mode_options[1][1])

    app.custom_ai_latency_mode_combobox = ttk.Combobox(
        frame,
        textvariable=app.custom_ai_latency_mode_display_var,
        values=[display for _, display in custom_ai_latency_mode_options],
        width=25,
        state='readonly',
    )
    app.custom_ai_latency_mode_combobox.grid(row=18, column=1, padx=5, pady=5, sticky="ew")

    def on_custom_ai_latency_mode_changed(event):
        selected_display = app.custom_ai_latency_mode_display_var.get()
        for value, display in custom_ai_latency_mode_options:
            if display == selected_display:
                app.custom_ai_latency_mode_var.set(value)
                log_debug(f"Custom AI latency mode changed to: {value}")
                if app._fully_initialized:
                    app.save_settings()
                break

    app.custom_ai_latency_mode_combobox.bind(
        "<<ComboboxSelected>>",
        create_combobox_handler_wrapper(on_custom_ai_latency_mode_changed),
    )
    app.custom_ai_latency_mode_options = custom_ai_latency_mode_options

    app.custom_ai_submit_interval_label = ttk.Label(
        frame,
        text=app.ui_lang.get_label(
            "custom_ai_submit_interval_label",
            "AI translation minimum interval (ms):",
        ),
    )
    app.custom_ai_submit_interval_label.grid(row=19, column=0, padx=5, pady=5, sticky="w")
    app.custom_ai_submit_interval_spinbox = ttk.Spinbox(
        frame,
        from_=0,
        to=5000,
        increment=50,
        textvariable=app.custom_ai_submit_interval_ms_var,
        width=10,
        validate="key",
        validatecommand=(validate_custom_ai_submit_interval, '%P'),
    )
    app.custom_ai_submit_interval_spinbox.grid(row=19, column=1, padx=5, pady=5, sticky="w")

    def on_custom_ai_submit_interval_focus_out(event):
        try:
            value = int(app.custom_ai_submit_interval_ms_var.get())
            app.custom_ai_submit_interval_ms_var.set(max(0, min(5000, value)))
        except (ValueError, tk.TclError):
            app.custom_ai_submit_interval_ms_var.set(300)
        app.save_settings()

    app.custom_ai_submit_interval_spinbox.bind(
        "<FocusOut>",
        on_custom_ai_submit_interval_focus_out,
    )

    app.custom_ai_ocr_image_format_label = ttk.Label(
        frame,
        text=app.ui_lang.get_label("custom_ai_ocr_image_format_label", "OCR image format:"),
    )
    app.custom_ai_ocr_image_format_label.grid(row=20, column=0, padx=5, pady=5, sticky="w")
    custom_ai_ocr_image_format_options = [
        ("webp", app.ui_lang.get_label("custom_ai_ocr_image_format_webp", "WebP")),
        ("png", app.ui_lang.get_label("custom_ai_ocr_image_format_png", "PNG")),
        ("jpeg", app.ui_lang.get_label("custom_ai_ocr_image_format_jpeg", "JPEG")),
    ]
    format_display_by_value = {value: display for value, display in custom_ai_ocr_image_format_options}
    format_value_by_display = {display: value for value, display in custom_ai_ocr_image_format_options}
    app.custom_ai_ocr_image_format_display_var = tk.StringVar(
        value=format_display_by_value.get(
            app.custom_ai_ocr_image_format_var.get(),
            format_display_by_value["webp"],
        )
    )
    app.custom_ai_ocr_image_format_combobox = ttk.Combobox(
        frame,
        textvariable=app.custom_ai_ocr_image_format_display_var,
        values=[display for _, display in custom_ai_ocr_image_format_options],
        width=25,
        state='readonly',
    )
    app.custom_ai_ocr_image_format_combobox.grid(row=20, column=1, padx=5, pady=5, sticky="ew")

    def on_custom_ai_ocr_image_format_changed(event):
        selected_display = app.custom_ai_ocr_image_format_display_var.get()
        app.custom_ai_ocr_image_format_var.set(format_value_by_display.get(selected_display, "webp"))
        if app._fully_initialized:
            app.save_settings()

    app.custom_ai_ocr_image_format_combobox.bind(
        "<<ComboboxSelected>>",
        create_combobox_handler_wrapper(on_custom_ai_ocr_image_format_changed),
    )

    app.custom_ai_ocr_image_mode_label = ttk.Label(
        frame,
        text=app.ui_lang.get_label("custom_ai_ocr_image_mode_label", "OCR image mode:"),
    )
    app.custom_ai_ocr_image_mode_label.grid(row=21, column=0, padx=5, pady=5, sticky="w")
    custom_ai_ocr_image_mode_options = [
        ("lossless_webp", app.ui_lang.get_label("custom_ai_ocr_image_mode_lossless", "Lossless WebP")),
        ("balanced_webp", app.ui_lang.get_label("custom_ai_ocr_image_mode_balanced", "Balanced WebP")),
        ("small_grayscale_webp", app.ui_lang.get_label("custom_ai_ocr_image_mode_grayscale", "Small grayscale WebP")),
    ]
    mode_display_by_value = {value: display for value, display in custom_ai_ocr_image_mode_options}
    mode_value_by_display = {display: value for value, display in custom_ai_ocr_image_mode_options}
    app.custom_ai_ocr_image_mode_display_var = tk.StringVar(
        value=mode_display_by_value.get(
            app.custom_ai_ocr_image_mode_var.get(),
            mode_display_by_value["balanced_webp"],
        )
    )
    app.custom_ai_ocr_image_mode_combobox = ttk.Combobox(
        frame,
        textvariable=app.custom_ai_ocr_image_mode_display_var,
        values=[display for _, display in custom_ai_ocr_image_mode_options],
        width=25,
        state='readonly',
    )
    app.custom_ai_ocr_image_mode_combobox.grid(row=21, column=1, padx=5, pady=5, sticky="ew")

    def on_custom_ai_ocr_image_mode_changed(event):
        selected_display = app.custom_ai_ocr_image_mode_display_var.get()
        app.custom_ai_ocr_image_mode_var.set(mode_value_by_display.get(selected_display, "balanced_webp"))
        if app._fully_initialized:
            app.save_settings()

    app.custom_ai_ocr_image_mode_combobox.bind(
        "<<ComboboxSelected>>",
        create_combobox_handler_wrapper(on_custom_ai_ocr_image_mode_changed),
    )

    app.custom_ai_ocr_image_quality_label = ttk.Label(
        frame,
        text=app.ui_lang.get_label("custom_ai_ocr_image_quality_label", "OCR image quality:"),
    )
    app.custom_ai_ocr_image_quality_label.grid(row=22, column=0, padx=5, pady=5, sticky="w")
    app.custom_ai_ocr_image_quality_spinbox = ttk.Spinbox(
        frame,
        from_=1,
        to=100,
        increment=5,
        textvariable=app.custom_ai_ocr_image_quality_var,
        width=10,
        validate="key",
        validatecommand=(validate_custom_ai_ocr_image_quality, '%P'),
    )
    app.custom_ai_ocr_image_quality_spinbox.grid(row=22, column=1, padx=5, pady=5, sticky="w")

    def on_custom_ai_ocr_image_quality_focus_out(event):
        try:
            value = int(app.custom_ai_ocr_image_quality_var.get())
            app.custom_ai_ocr_image_quality_var.set(max(1, min(100, value)))
        except (ValueError, tk.TclError):
            app.custom_ai_ocr_image_quality_var.set(85)
        app.save_settings()

    app.custom_ai_ocr_image_quality_spinbox.bind(
        "<FocusOut>",
        on_custom_ai_ocr_image_quality_focus_out,
    )

    app.custom_ai_ocr_image_detail_label = ttk.Label(
        frame,
        text=app.ui_lang.get_label("custom_ai_ocr_image_detail_label", "OCR image detail:"),
    )
    app.custom_ai_ocr_image_detail_label.grid(row=23, column=0, padx=5, pady=5, sticky="w")
    custom_ai_ocr_image_detail_options = [
        ("auto", app.ui_lang.get_label("custom_ai_ocr_image_detail_auto", "Auto")),
        ("low", app.ui_lang.get_label("custom_ai_ocr_image_detail_low", "Low")),
        ("high", app.ui_lang.get_label("custom_ai_ocr_image_detail_high", "High")),
    ]
    detail_display_by_value = {value: display for value, display in custom_ai_ocr_image_detail_options}
    detail_value_by_display = {display: value for value, display in custom_ai_ocr_image_detail_options}
    app.custom_ai_ocr_image_detail_display_var = tk.StringVar(
        value=detail_display_by_value.get(
            app.custom_ai_ocr_image_detail_var.get(),
            detail_display_by_value["auto"],
        )
    )
    app.custom_ai_ocr_image_detail_combobox = ttk.Combobox(
        frame,
        textvariable=app.custom_ai_ocr_image_detail_display_var,
        values=[display for _, display in custom_ai_ocr_image_detail_options],
        width=25,
        state='readonly',
    )
    app.custom_ai_ocr_image_detail_combobox.grid(row=23, column=1, padx=5, pady=5, sticky="ew")

    def on_custom_ai_ocr_image_detail_changed(event):
        selected_display = app.custom_ai_ocr_image_detail_display_var.get()
        app.custom_ai_ocr_image_detail_var.set(detail_value_by_display.get(selected_display, "auto"))
        if app._fully_initialized:
            app.save_settings()

    app.custom_ai_ocr_image_detail_combobox.bind(
        "<<ComboboxSelected>>",
        create_combobox_handler_wrapper(on_custom_ai_ocr_image_detail_changed),
    )

    row_offset = 24
    if app.MARIANMT_AVAILABLE:
        texts = [
            app.ui_lang.get_label("marian_beam_explanation", "Higher beam values = better but slower translations"),
            app.ui_lang.get_label("marian_quality_note", "Note: MarianMT provides higher quality translations"),
            app.ui_lang.get_label("marian_quality_vary", "for many language pairs. Quality may vary by language."),
            app.ui_lang.get_label("marian_download_note", "Models are downloaded on first use (requires internet).")
        ]
    else:
        texts = [
            app.ui_lang.get_label("marian_unavailable_line1", "MarianMT is not available. To enable, install:"),
            app.ui_lang.get_label("marian_unavailable_line2", "pip install transformers torch sentencepiece")
        ]
    for i, text_content in enumerate(texts):
        lbl = ttk.Label(frame, text=text_content)
        lbl.grid(row=row_offset + i, column=0, columnspan=3, padx=5, pady=0, sticky="w")
        app.marian_explanation_labels.append(lbl)
    current_row = row_offset + len(texts)

    app.keep_linebreaks_label = ttk.Label(frame, text=app.ui_lang.get_label("keep_linebreaks_label", "Keep Linebreaks"))
    app.keep_linebreaks_label.grid(row=current_row, column=0, padx=5, pady=5, sticky="w")

    def on_keep_linebreaks_clicked():
        # Schedule focus shift to the parent tab frame (app.tab_settings) to remove dotted focus frame
        if app.keep_linebreaks_checkbox.winfo_exists() and app.tab_settings.winfo_exists():
            app.keep_linebreaks_checkbox.after_idle(app.tab_settings.focus_set)

    app.keep_linebreaks_checkbox = ttk.Checkbutton(frame, variable=app.keep_linebreaks_var, command=on_keep_linebreaks_clicked)
    app.keep_linebreaks_checkbox.grid(row=current_row, column=1, padx=5, pady=5, sticky="w")
    current_row += 1

    ttk.Label(frame, text=app.ui_lang.get_label("capture_backend_label", "Capture Backend")).grid(row=current_row, column=0, padx=5, pady=5, sticky="w")
    capture_backend_display_values = [
        app.ui_lang.get_label("capture_backend_auto", "Auto (fastest available)"),
        app.ui_lang.get_label("capture_backend_mss", "MSS (fast)"),
        app.ui_lang.get_label("capture_backend_pyautogui", "PyAutoGUI (compatible)"),
    ]
    capture_backend_code_by_display = {
        capture_backend_display_values[0]: "auto",
        capture_backend_display_values[1]: "mss",
        capture_backend_display_values[2]: "pyautogui",
    }
    capture_backend_display_by_code = {v: k for k, v in capture_backend_code_by_display.items()}
    app.capture_backend_display_var = tk.StringVar(value=capture_backend_display_by_code.get(app.capture_backend_var.get(), capture_backend_display_values[0]))
    app.capture_backend_combobox = ttk.Combobox(
        frame,
        textvariable=app.capture_backend_display_var,
        values=capture_backend_display_values,
        width=25,
        state="readonly",
    )
    app.capture_backend_combobox.grid(row=current_row, column=1, padx=5, pady=5, sticky="ew")

    def on_capture_backend_changed(event):
        selected_backend_display = app.capture_backend_display_var.get()
        app.capture_backend_var.set(capture_backend_code_by_display.get(selected_backend_display, "auto"))
        app.save_settings()

    app.capture_backend_combobox.bind("<<ComboboxSelected>>", create_combobox_handler_wrapper(on_capture_backend_changed))
    current_row += 1

    ttk.Label(frame, text=app.ui_lang.get_label("scan_interval_label")).grid(row=current_row, column=0, padx=5, pady=5, sticky="w")
    scan_spinbox = ttk.Spinbox(frame, from_=50, to=2000, increment=50, textvariable=app.scan_interval_var,
                             width=10, validate="key", validatecommand=(validate_scan_interval, '%P'))
    scan_spinbox.grid(row=current_row, column=1, padx=5, pady=5, sticky="w")
    def on_scan_interval_focus_out(event):
        try:
            value = int(app.scan_interval_var.get())
            if not (50 <= value <= 2000): app.scan_interval_var.set(max(50, min(2000, value)))
        except (ValueError, tk.TclError): app.scan_interval_var.set(200)
        app.save_settings()
    scan_spinbox.bind("<FocusOut>", on_scan_interval_focus_out)
    current_row += 1

    ttk.Label(frame, text=app.ui_lang.get_label("clear_timeout_label")).grid(row=current_row, column=0, padx=5, pady=5, sticky="w")
    timeout_spinbox = ttk.Spinbox(frame, from_=0, to=60, textvariable=app.clear_translation_timeout_var,
                                width=10, validate="key", validatecommand=(validate_timeout, '%P'))
    timeout_spinbox.grid(row=current_row, column=1, padx=5, pady=5, sticky="w")
    def on_timeout_focus_out(event):
        try:
            value = int(app.clear_translation_timeout_var.get())
            if not (0 <= value <= 60): app.clear_translation_timeout_var.set(max(0, min(60, value)))
            app.clear_translation_timeout = app.clear_translation_timeout_var.get()
        except (ValueError, tk.TclError): app.clear_translation_timeout_var.set(0)
        app.save_settings()
    timeout_spinbox.bind("<FocusOut>", on_timeout_focus_out)
    current_row += 1

    app.stability_label = ttk.Label(frame, text=app.ui_lang.get_label("stability_threshold_label"))
    app.stability_label.grid(row=current_row, column=0, padx=5, pady=5, sticky="w")
    app.stability_spinbox = ttk.Spinbox(frame, from_=0, to=5, textvariable=app.stability_var,
                                  width=10, validate="key", validatecommand=(validate_stability, '%P'))
    app.stability_spinbox.grid(row=current_row, column=1, padx=5, pady=5, sticky="w")
    def on_stability_focus_out(event):
        try:
            value = int(app.stability_var.get())
            if not (0 <= value <= 5) : app.stability_var.set(max(0, min(5,value)))
            app.update_stability_from_spinbox()
        except (ValueError, tk.TclError): app.stability_var.set(0)
        app.save_settings()
    app.stability_spinbox.bind("<FocusOut>", on_stability_focus_out)
    current_row += 1

    app.paddleocr_min_score_label = ttk.Label(
        frame,
        text=app.ui_lang.get_label("paddleocr_min_score_label", "PaddleOCR minimum score:"),
    )
    app.paddleocr_min_score_label.grid(row=current_row, column=0, padx=5, pady=5, sticky="w")
    app.paddleocr_min_score_spinbox = ttk.Spinbox(
        frame,
        from_=0.0,
        to=1.0,
        increment=0.05,
        textvariable=app.paddleocr_min_score_var,
        width=10,
        validate="key",
        validatecommand=(validate_paddleocr_min_score, '%P'),
    )
    app.paddleocr_min_score_spinbox.grid(row=current_row, column=1, padx=5, pady=5, sticky="w")

    def on_paddleocr_min_score_focus_out(event):
        try:
            value = float(app.paddleocr_min_score_var.get())
            clamped = max(0.0, min(1.0, value))
            app.paddleocr_min_score_var.set(f"{clamped:.2f}")
        except (ValueError, tk.TclError):
            app.paddleocr_min_score_var.set("0.35")
        app.save_settings()

    app.paddleocr_min_score_spinbox.bind("<FocusOut>", on_paddleocr_min_score_focus_out)
    current_row += 1

    # Store references to OCR debugging widgets for OCR model UI management
    app.ocr_debugging_label = ttk.Label(frame, text=app.ui_lang.get_label("ocr_debugging_label"))
    app.ocr_debugging_label.grid(row=current_row, column=0, padx=5, pady=5, sticky="w")

    # Create a frame for the OCR debugging checkbox and preview button
    app.ocr_debug_frame = ttk.Frame(frame)
    app.ocr_debug_frame.grid(row=current_row, column=1, columnspan=2, padx=5, pady=5, sticky="ew")

    app.ocr_debugging_checkbox = ttk.Checkbutton(app.ocr_debug_frame, text=app.ui_lang.get_label("show_debug_checkbox"), variable=app.ocr_debugging_var)
    app.ocr_debugging_checkbox.pack(side=tk.LEFT)

    # Add Preview button
    app.ocr_preview_button = ttk.Button(app.ocr_debug_frame, text=app.ui_lang.get_label("preview_btn", "Preview"),
               command=app.show_ocr_preview)
    app.ocr_preview_button.pack(side=tk.LEFT, padx=(10,0))
    current_row += 1

    color_options = [
        (app.ui_lang.get_label("source_color_label"), app.source_colour_var, 'source'),
        (app.ui_lang.get_label("target_color_label"), app.target_colour_var, 'target'),
        (app.ui_lang.get_label("target_text_color_label"), app.target_text_colour_var, 'target_text')
    ]
    app.color_displays = {}
    for i, (label_text, var, color_type) in enumerate(color_options):
        ttk.Label(frame, text=label_text).grid(row=current_row + i, column=0, padx=5, pady=5, sticky="w")
        color_frame_inner = ttk.Frame(frame)
        color_frame_inner.grid(row=current_row + i, column=1, columnspan=2, padx=5, pady=5, sticky="w")

        app.color_displays[color_type] = tk.Label(color_frame_inner, width=3, relief="solid", bg=var.get())
        app.color_displays[color_type].pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(color_frame_inner, text=app.ui_lang.get_label("choose_color_btn"), command=lambda ct=color_type: app.choose_color_for_settings(ct)).pack(side=tk.LEFT)
    current_row += len(color_options)

    ttk.Label(frame, text=app.ui_lang.get_label("font_size_label")).grid(row=current_row, column=0, padx=5, pady=5, sticky="w")
    font_spinbox = ttk.Spinbox(frame, from_=8, to=72, textvariable=app.target_font_size_var,
                             width=10, validate="key", validatecommand=(validate_font_size, '%P'))
    font_spinbox.grid(row=current_row, column=1, padx=5, pady=5, sticky="w")
    def on_font_size_focus_out(event):
        try:
            value = int(app.target_font_size_var.get())
            if not (8 <= value <= 72): app.target_font_size_var.set(max(8, min(72, value)))
            app.update_target_font_size()
        except (ValueError, tk.TclError): app.target_font_size_var.set(12)
        app.save_settings()
    font_spinbox.bind("<FocusOut>", on_font_size_focus_out)
    current_row += 1

    # Font type dropdown
    ttk.Label(frame, text=app.ui_lang.get_label("font_type_label")).grid(row=current_row, column=0, padx=5, pady=5, sticky="w")
    font_type_combobox = ttk.Combobox(frame, textvariable=app.target_font_type_var,
                                    values=get_system_fonts(), width=20, state='readonly')
    font_type_combobox.grid(row=current_row, column=1, padx=5, pady=5, sticky="w")

    def on_font_type_change(event):
        app.update_target_font_type()
        if app._fully_initialized:
            app.save_settings()
    font_type_combobox.bind('<<ComboboxSelected>>', create_combobox_handler_wrapper(on_font_type_change))
    current_row += 1

    # Opacity controls
    opacity_frame = ttk.Frame(frame)
    opacity_frame.grid(row=current_row, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

    ttk.Label(opacity_frame, text=app.ui_lang.get_label("opacity_label", "Opacity:")).grid(row=0, column=0, padx=(0, 10), pady=0, sticky="w")

    # Background opacity
    ttk.Label(opacity_frame, text=app.ui_lang.get_label("opacity_background_label", "Background:")).grid(row=0, column=1, padx=(0, 5), pady=0, sticky="w")
    bg_opacity_spinbox = ttk.Spinbox(opacity_frame, from_=0.00, to=1.00, increment=0.05, width=8,
                                   textvariable=app.target_opacity_var, format="%.2f")
    bg_opacity_spinbox.grid(row=0, column=2, padx=(0, 10), pady=0, sticky="w")

    # Text opacity
    ttk.Label(opacity_frame, text=app.ui_lang.get_label("opacity_text_label", "Text:")).grid(row=0, column=3, padx=(0, 5), pady=0, sticky="w")
    text_opacity_spinbox = ttk.Spinbox(opacity_frame, from_=0.00, to=1.00, increment=0.05, width=8,
                                     textvariable=app.target_text_opacity_var, format="%.2f")
    text_opacity_spinbox.grid(row=0, column=4, padx=0, pady=0, sticky="w")

    # Bind focus out events to validate and update overlays
    def on_bg_opacity_focus_out(event):
        try:
            value = float(app.target_opacity_var.get())
            if not (0.00 <= value <= 1.00):
                app.target_opacity_var.set(max(0.00, min(1.00, value)))
        except (ValueError, tk.TclError):
            app.target_opacity_var.set(0.15)
        app.update_target_opacity()
        # Note: Settings are saved automatically via trace callback

    def on_text_opacity_focus_out(event):
        try:
            value = float(app.target_text_opacity_var.get())
            if not (0.00 <= value <= 1.00):
                app.target_text_opacity_var.set(max(0.00, min(1.00, value)))
        except (ValueError, tk.TclError):
            app.target_text_opacity_var.set(1.0)
        app.update_target_text_opacity()
        # Note: Settings are saved automatically via trace callback

    bg_opacity_spinbox.bind("<FocusOut>", on_bg_opacity_focus_out)
    text_opacity_spinbox.bind("<FocusOut>", on_text_opacity_focus_out)
    current_row += 1

    file_cache_frame_outer = ttk.LabelFrame(frame, text=app.ui_lang.get_label("file_cache_frame_title"))
    file_cache_frame_outer.grid(row=current_row, column=0, columnspan=3, padx=5, pady=5, sticky="ew")
    ttk.Label(
        file_cache_frame_outer,
        text=app.ui_lang.get_label("file_cache_description", "Clear cached translations for the active custom AI configuration."),
        wraplength=400
    ).grid(row=0, column=0, columnspan=2, padx=5, pady=2, sticky="w")
    ttk.Button(file_cache_frame_outer, text=app.ui_lang.get_label("clear_caches_btn"), command=app.clear_file_caches).grid(row=1, column=0, padx=5, pady=5, sticky="w")
    current_row += 1

    button_frame_outer = ttk.Frame(frame)
    button_frame_outer.grid(row=current_row, column=0, columnspan=3, pady=10)
    save_settings_button = ttk.Button(button_frame_outer, text=app.ui_lang.get_label("save_settings_btn"), command=app.save_settings)
    save_settings_button.pack(side=tk.LEFT, padx=5)

    # Store reference for the tab changed handler in app_logic.py
    app.settings_tab_save_button = save_settings_button

    frame.columnconfigure(1, weight=1)

def create_debug_tab(app):
    # Create a scrollable tab content frame
    scrollable_content = create_scrollable_tab(app.tab_control, app.ui_lang.get_label("debug_tab_title"))
    app.tab_debug = scrollable_content

    # Create the debug frame inside the scrollable area
    frame = ttk.LabelFrame(scrollable_content, text=app.ui_lang.get_label("debug_tab_title"))
    frame.pack(fill="both", expand=True, padx=10, pady=10)

    image_frame = ttk.Frame(frame)
    image_frame.pack(fill="x", padx=5, pady=5)

    original_frame = ttk.LabelFrame(image_frame, text=app.ui_lang.get_label("original_image_label"))
    original_frame.pack(side=tk.LEFT, padx=5, fill="both", expand=True)
    app.original_image_label = ttk.Label(original_frame, text=app.ui_lang.get_label("no_image_captured", "No image captured yet"))
    app.original_image_label.pack(padx=5, pady=5)

    processed_frame = ttk.LabelFrame(image_frame, text=app.ui_lang.get_label("processed_image_label"))
    processed_frame.pack(side=tk.RIGHT, padx=5, fill="both", expand=True)
    app.processed_image_label = ttk.Label(processed_frame, text=app.ui_lang.get_label("no_image_processed", "No image processed yet"))
    app.processed_image_label.pack(padx=5, pady=5)

    ocr_frame = ttk.LabelFrame(frame, text=app.ui_lang.get_label("ocr_results_label"))
    ocr_frame.pack(fill="x", padx=5, pady=5)
    app.ocr_results_text = tk.Text(ocr_frame, height=16, width=50, wrap=tk.WORD)
    style_tk_text_widget(app.ocr_results_text, getattr(app, "md3_palette", None))
    app.ocr_results_text.pack(fill="both", expand=True, padx=5, pady=5)
    app.ocr_results_text.insert(tk.END, app.ui_lang.get_label("ocr_results_placeholder"))
    app.ocr_results_text.config(state=tk.DISABLED)

    button_frame = ttk.Frame(frame)
    button_frame.pack(fill="x", padx=5, pady=5)
    ttk.Button(button_frame, text=app.ui_lang.get_label("save_debug_images_btn"), command=app.save_debug_images).pack(side=tk.LEFT, padx=5)
    ttk.Button(button_frame, text=app.ui_lang.get_label("refresh_log_btn"), command=app.refresh_debug_log).pack(side=tk.LEFT, padx=5)

    diagnostics_frame = ttk.LabelFrame(
        frame,
        text=app.ui_lang.get_label("performance_diagnostics_title", "Performance Diagnostics"),
    )
    diagnostics_frame.pack(fill="both", expand=False, padx=5, pady=5)

    diagnostics_button_frame = ttk.Frame(diagnostics_frame)
    diagnostics_button_frame.pack(fill="x", padx=5, pady=(5, 0))
    ttk.Button(
        diagnostics_button_frame,
        text=app.ui_lang.get_label("reset_metrics_btn", "Reset metrics"),
        command=lambda: _reset_runtime_metrics_panel(app),
    ).pack(side=tk.LEFT, padx=(0, 5))
    ttk.Button(
        diagnostics_button_frame,
        text=app.ui_lang.get_label("copy_metrics_summary_btn", "Copy summary"),
        command=lambda: _copy_runtime_metrics_summary(app),
    ).pack(side=tk.LEFT, padx=5)

    app.runtime_metrics_text = tk.Text(
        diagnostics_frame,
        height=14,
        width=72,
        wrap=tk.WORD,
    )
    style_tk_text_widget(app.runtime_metrics_text, getattr(app, "md3_palette", None))
    app.runtime_metrics_text.pack(fill="both", expand=True, padx=5, pady=5)
    app.runtime_metrics_text.config(state=tk.DISABLED)
    _cancel_runtime_metrics_refresh(app)
    _refresh_runtime_metrics_panel(app)

    log_frame = ttk.LabelFrame(frame, text=app.ui_lang.get_label("app_log_label"))
    log_frame.pack(fill="both", expand=True, padx=5, pady=5)

    app.log_text = tk.Text(log_frame, wrap=tk.WORD)
    style_tk_text_widget(app.log_text, getattr(app, "md3_palette", None))
    scrollbar = ttk.Scrollbar(log_frame, command=app.log_text.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    app.log_text.config(yscrollcommand=scrollbar.set, state=tk.DISABLED)
    app.log_text.pack(fill="both", expand=True, padx=5, pady=5)

    app.refresh_debug_log()

def create_custom_prompt_tab(app):
    # Create a scrollable tab content frame
    scrollable_content = create_scrollable_tab(app.tab_control, app.ui_lang.get_label("custom_prompt_tab_title", "Custom Prompt"))
    app.tab_custom_prompt = scrollable_content

    # Create the main frame inside the scrollable area
    frame = ttk.LabelFrame(scrollable_content, text=app.ui_lang.get_label("custom_prompt_tab_title", "Custom Prompt"))
    frame.pack(fill="both", expand=True, padx=10, pady=10)

    # Info label
    info_text = app.ui_lang.get_label("custom_prompt_info", "This text will be added at the beginning of the custom AI translation instruction.")
    info_label = ttk.Label(frame, text=info_text, wraplength=550)
    info_label.pack(fill="x", padx=5, pady=5)

    # Text area
    text_frame = ttk.Frame(frame)
    text_frame.pack(fill="both", expand=True, padx=5, pady=5)

    app.custom_prompt_text_widget = tk.Text(text_frame, wrap=tk.WORD, height=15)
    style_tk_text_widget(app.custom_prompt_text_widget, getattr(app, "md3_palette", None))
    scrollbar = ttk.Scrollbar(text_frame, command=app.custom_prompt_text_widget.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    app.custom_prompt_text_widget.pack(side=tk.LEFT, fill="both", expand=True)
    app.custom_prompt_text_widget.config(yscrollcommand=scrollbar.set)

    # Load initial text
    if hasattr(app, "custom_prompt_text"):
        app.custom_prompt_text_widget.insert("1.0", app.custom_prompt_text)

    # Buttons
    button_frame = ttk.Frame(frame)
    button_frame.pack(fill="x", padx=5, pady=10)

    def save_prompt():
        text = app.custom_prompt_text_widget.get("1.0", "end-1c")
        if not app.save_custom_prompt(text):
            messagebox.showerror(app.ui_lang.get_label("error_title", "Error"),
                                 app.ui_lang.get_label("custom_prompt_save_error", "Failed to save custom prompt."))

    def reload_prompt():
        app.load_custom_prompt()
        app.custom_prompt_text_widget.delete("1.0", tk.END)
        app.custom_prompt_text_widget.insert("1.0", app.custom_prompt_text)

    app.save_custom_prompt_btn = ttk.Button(button_frame, text=app.ui_lang.get_label("save_btn", "Save"), command=save_prompt)
    app.save_custom_prompt_btn.pack(side=tk.LEFT, padx=5)

    app.reload_custom_prompt_btn = ttk.Button(button_frame, text=app.ui_lang.get_label("reload_btn", "Reload"), command=reload_prompt)
    app.reload_custom_prompt_btn.pack(side=tk.LEFT, padx=5)

    # Function to update labels on language change
    def update_custom_prompt_labels_for_language():
        if hasattr(app, 'tab_custom_prompt') and app.tab_custom_prompt.winfo_exists():
            try:
                frame.config(text=app.ui_lang.get_label("custom_prompt_tab_title", "Custom Prompt"))
                info_label.config(text=app.ui_lang.get_label("custom_prompt_info", "This text will be added at the beginning of the custom AI translation instruction."))
                app.save_custom_prompt_btn.config(text=app.ui_lang.get_label("save_btn", "Save"))
                app.reload_custom_prompt_btn.config(text=app.ui_lang.get_label("reload_btn", "Reload"))
            except Exception as e:
                log_debug(f"Error updating custom prompt UI language: {e}")

    app.update_custom_prompt_labels_for_language = update_custom_prompt_labels_for_language
