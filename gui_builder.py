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


from gui_profile_controls import (
    _apply_custom_ai_profile_model_selection,
    _find_selected_custom_ai_profile,
    _read_var,
    _reasoning_effort_display_value,
    _reasoning_effort_display_values,
    _reasoning_effort_value_from_display,
    build_custom_ai_profile_values_from_form,
    build_ocr_model_display_options,
    filter_model_values,
    get_paddleocr_ocr_display_name,
    get_rapidocr_ocr_display_name,
    resolve_ocr_model_display_selection,
)

def apply_custom_ai_profile_model_selection(app):
    """Persist an explicit model choice and wake an active translation profile."""
    try:
        return _apply_custom_ai_profile_model_selection(app)
    except Exception as error:
        log_debug(
            "Custom AI profile model apply failed: "
            f"{type(error).__name__} - {error}"
        )
        ui_lang = getattr(app, "ui_lang", None)
        title = (
            ui_lang.get_label("profile_error_title", "Profile Error")
            if ui_lang is not None
            else "Profile Error"
        )
        try:
            messagebox.showerror(
                title,
                str(error),
                parent=getattr(app, "root", None),
            )
        except Exception as dialog_error:
            log_debug(
                "Custom AI profile error dialog failed: "
                f"{type(dialog_error).__name__} - {dialog_error}"
            )
        return False


def _apply_custom_ai_translation_profile_selection(app, selected_name):
    profiles = getattr(app, "custom_ai_profiles", None)
    selected_name = str(selected_name or "").strip()
    if profiles is None or not selected_name:
        return False

    selected_profile = next(
        (
            profile
            for profile in profiles.list_profiles(enabled_only=True)
            if str(profile.get("name") or "") == selected_name
        ),
        None,
    )
    if not selected_profile:
        return False

    active_profile = profiles.get_active_profile("translation")
    if (
        active_profile
        and active_profile.get("id") == selected_profile.get("id")
    ):
        translation_model_var = getattr(app, "translation_model_var", None)
        if translation_model_var is not None:
            translation_model_var.set("custom_ai")
        return False

    updated_profile = profiles.set_active_profile(
        "translation",
        selected_profile["id"],
    )
    translation_model_var = getattr(app, "translation_model_var", None)
    if translation_model_var is not None:
        translation_model_var.set("custom_ai")

    translation_handler = getattr(app, "translation_handler", None)
    clear_context = getattr(translation_handler, "_clear_active_context", None)
    if callable(clear_context):
        clear_context()

    if getattr(app, "is_running", False):
        from worker_threads import refresh_translation_after_profile_change

        refresh_translation_after_profile_change(
            app,
            reason="active translation profile changed",
        )

    log_debug(
        "Custom AI active translation profile changed "
        f"profile={updated_profile.get('name', 'Custom AI')} "
        f"model={updated_profile.get('model', '')}"
    )
    return True


def apply_custom_ai_translation_profile_selection(app, selected_name):
    """Persist and apply a primary translation profile selection immediately."""
    try:
        return _apply_custom_ai_translation_profile_selection(
            app,
            selected_name,
        )
    except Exception as error:
        log_debug(
            "Custom AI translation profile apply failed: "
            f"{type(error).__name__} - {error}"
        )
        ui_lang = getattr(app, "ui_lang", None)
        title = (
            ui_lang.get_label("profile_error_title", "Profile Error")
            if ui_lang is not None
            else "Profile Error"
        )
        try:
            messagebox.showerror(
                title,
                str(error),
                parent=getattr(app, "root", None),
            )
        except Exception as dialog_error:
            log_debug(
                "Custom AI translation profile error dialog failed: "
                f"{type(dialog_error).__name__} - {dialog_error}"
            )
        return False


def handle_translation_profile_selection(
    app,
    selected_name,
    event=None,
):
    """Apply the selected profile before synchronizing the remaining UI state."""
    changed = apply_custom_ai_translation_profile_selection(
        app,
        selected_name,
    )
    # The selection has already been persisted above. Synchronize the rest of
    # the UI through the non-mutating path so the legacy handler cannot save
    # the same active profile a second time outside this error boundary.
    app.on_translation_model_selection_changed(
        event=None,
        initial_setup=False,
        synchronize_ui_only=True,
    )
    return changed


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
    ("paddleocr_prewarm_total_duration", "PaddleOCR prewarm total"),
    ("paddleocr_prewarm_text_recognition_duration", "PaddleOCR prewarm text-rec"),
    ("paddleocr_prewarm_full_engine_duration", "PaddleOCR prewarm full"),
    ("paddleocr_prewarm_wait_duration", "PaddleOCR worker wait"),
    ("paddleocr_first_ocr_wait_duration", "PaddleOCR first OCR wait"),
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
    prewarm_status = labels.get("paddleocr_prewarm_status") or "-"
    prewarm_outcome = labels.get("paddleocr_prewarm_outcome") or "-"
    prewarm_reason = labels.get("paddleocr_prewarm_reason") or "-"
    prewarm_settings = labels.get("paddleocr_prewarm_settings") or "-"
    prewarm_host = labels.get("paddleocr_prewarm_host") or "-"
    lines.append(
        "PaddleOCR prewarm: "
        f"status={prewarm_status} outcome={prewarm_outcome} "
        f"gen={gauges.get('paddleocr_prewarm_generation', 0)} "
        f"active={gauges.get('paddleocr_prewarm_active', 0)} "
        f"ready={gauges.get('paddleocr_prewarm_ready', 0)}"
    )
    lines.append(f"PaddleOCR prewarm reason: {prewarm_reason}")
    lines.append(f"PaddleOCR prewarm settings: {prewarm_settings}")
    lines.append(f"PaddleOCR prewarm host: {prewarm_host}")
    if gauges.get("paddleocr_first_ocr_wait_s") is not None:
        lines.append(
            "PaddleOCR first OCR wait: "
            f"{_format_metric_seconds(gauges.get('paddleocr_first_ocr_wait_s'))}"
        )
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
        # The panel refreshes at 1 Hz for the whole session and is usually
        # idle; skip the Tk delete+insert (a full re-layout) when the rendered
        # text has not changed.
        if content != getattr(app, "_runtime_metrics_last_rendered", None):
            text_widget.config(state=tk.NORMAL)
            text_widget.delete("1.0", tk.END)
            text_widget.insert(tk.END, content)
            text_widget.config(state=tk.DISABLED)
            app._runtime_metrics_last_rendered = content
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
    # Force a re-render even if the formatted text coincidentally matches.
    app._runtime_metrics_last_rendered = None
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
    from gui_settings_builder import create_settings_tab as build_settings_tab

    return build_settings_tab(app)


def create_debug_tab(app):
    from gui_diagnostics_builder import create_debug_tab as build_debug_tab

    return build_debug_tab(app)


def create_custom_prompt_tab(app):
    from gui_diagnostics_builder import create_custom_prompt_tab as build_custom_prompt_tab

    return build_custom_prompt_tab(app)
