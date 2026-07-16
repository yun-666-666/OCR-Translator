"""Settings-tab construction extracted from gui_builder."""

import tkinter as tk
from tkinter import messagebox, ttk

from custom_ai import CUSTOM_AI_REASONING_EFFORT_LOW
from gui_builder import (
    _reasoning_effort_display_value,
    _reasoning_effort_display_values,
    apply_custom_ai_profile_model_selection,
    build_custom_ai_profile_values_from_form,
    build_ocr_model_display_options,
    filter_model_values,
    get_paddleocr_ocr_display_name,
    get_system_fonts,
    handle_translation_profile_selection,
    resolve_ocr_model_display_selection,
    run_profile_network_task_async,
)
from logger import log_debug
from paddle_ocr_backend import PADDLEOCR_MODEL_CODE
from ui_elements import create_scrollable_tab


def _settings_color_grid_position(index):
    """Return a fixed two-column position for a settings color control."""
    return divmod(int(index), 2)


def create_settings_tab(app):
    # Create a scrollable tab content frame
    scrollable_content = create_scrollable_tab(
        app.tab_control,
        app.ui_lang.get_label("settings_tab_title"),
        protect_wheel_inputs=True,
    )
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
    validate_timeout = frame.register(lambda P: validate_int_range(P, 0, 60))
    validate_stability = frame.register(lambda P: validate_int_range(P, 0, 5))
    validate_paddleocr_min_score = frame.register(lambda P: validate_float_range(P, 0.0, 1.0))
    validate_font_size = frame.register(lambda P: validate_int_range(P, 8, 72))
    validate_outline_width = frame.register(lambda P: validate_int_range(P, 0, 6))

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
        handle_translation_profile_selection(
            app,
            selected_name,
            event=event,
        )
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
    app.ai_profile_model_combobox.bind(
        "<<ComboboxSelected>>",
        lambda _event: apply_custom_ai_profile_model_selection(app),
    )
    app.ai_profile_model_combobox.bind(
        "<Return>",
        lambda _event: apply_custom_ai_profile_model_selection(app),
    )
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

    app.ai_optimization_mode_label = ttk.Label(
        frame,
        text=app.ui_lang.get_label(
            "ai_optimization_mode_label",
            "AI optimization:",
        ),
    )
    app.ai_optimization_mode_label.grid(
        row=18,
        column=0,
        padx=5,
        pady=5,
        sticky="w",
    )
    ai_optimization_mode_options = [
        (
            "auto",
            app.ui_lang.get_label(
                "ai_optimization_mode_auto",
                "Automatic (recommended)",
            ),
        ),
        (
            "stream",
            app.ui_lang.get_label(
                "ai_optimization_mode_stream",
                "Streaming",
            ),
        ),
        (
            "speed",
            app.ui_lang.get_label(
                "ai_optimization_mode_speed",
                "Speed",
            ),
        ),
        (
            "quality",
            app.ui_lang.get_label(
                "ai_optimization_mode_quality",
                "Quality",
            ),
        ),
    ]
    app.ai_optimization_mode_display_var = tk.StringVar()
    current_optimization_mode = app.ai_optimization_mode_var.get()
    for value, display in ai_optimization_mode_options:
        if value == current_optimization_mode:
            app.ai_optimization_mode_display_var.set(display)
            break
    else:
        app.ai_optimization_mode_display_var.set(
            ai_optimization_mode_options[0][1]
        )

    app.ai_optimization_mode_combobox = ttk.Combobox(
        frame,
        textvariable=app.ai_optimization_mode_display_var,
        values=[display for _, display in ai_optimization_mode_options],
        width=25,
        state='readonly',
    )
    app.ai_optimization_mode_combobox.grid(
        row=18,
        column=1,
        padx=5,
        pady=5,
        sticky="ew",
    )

    def on_ai_optimization_mode_changed(event):
        selected_display = app.ai_optimization_mode_display_var.get()
        for value, display in ai_optimization_mode_options:
            if display == selected_display:
                app.ai_optimization_mode_var.set(value)
                log_debug(f"AI optimization mode changed to: {value}")
                if app._fully_initialized:
                    app.save_settings()
                break

    app.ai_optimization_mode_combobox.bind(
        "<<ComboboxSelected>>",
        create_combobox_handler_wrapper(on_ai_optimization_mode_changed),
    )
    app.ai_optimization_mode_options = ai_optimization_mode_options

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

    row_offset = 20
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

    ttk.Label(
        frame,
        text=app.ui_lang.get_label(
            "translation_line_layout_label", "Translation subtitle layout:"
        ),
    ).grid(row=current_row, column=0, padx=5, pady=5, sticky="w")
    translation_layout_options = [
        (
            "compact",
            app.ui_lang.get_label(
                "translation_line_layout_compact", "Prefer one line"
            ),
        ),
        (
            "preserve_source_lines",
            app.ui_lang.get_label(
                "translation_line_layout_preserve", "Match source subtitle lines"
            ),
        ),
    ]
    translation_layout_display_by_value = dict(translation_layout_options)
    translation_layout_value_by_display = {
        display: value for value, display in translation_layout_options
    }
    app.translation_line_layout_display_var = tk.StringVar(
        value=translation_layout_display_by_value.get(
            app.translation_line_layout_var.get(), translation_layout_options[0][1]
        )
    )
    app.translation_line_layout_combobox = ttk.Combobox(
        frame,
        textvariable=app.translation_line_layout_display_var,
        values=[display for _, display in translation_layout_options],
        width=25,
        state="readonly",
    )
    app.translation_line_layout_combobox.grid(
        row=current_row, column=1, padx=5, pady=5, sticky="w"
    )

    def on_translation_line_layout_changed(_event):
        selected_display = app.translation_line_layout_display_var.get()
        selected_value = translation_layout_value_by_display.get(
            selected_display, "compact"
        )
        app.translation_line_layout_var.set(selected_value)
        app.keep_linebreaks_var.set(selected_value == "preserve_source_lines")
        if app.tab_settings.winfo_exists():
            app.translation_line_layout_combobox.after_idle(app.tab_settings.focus_set)

    app.translation_line_layout_combobox.bind(
        "<<ComboboxSelected>>",
        create_combobox_handler_wrapper(on_translation_line_layout_changed),
    )
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
            app.paddleocr_min_score_var.set("0.45")
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

    app.colors_row_frame = ttk.Frame(frame)
    app.colors_row_frame.grid(
        row=current_row,
        column=0,
        columnspan=3,
        padx=5,
        pady=5,
        sticky="w",
    )
    color_options = [
        (app.ui_lang.get_label("source_color_label"), app.source_colour_var, 'source'),
        (app.ui_lang.get_label("target_color_label"), app.target_colour_var, 'target'),
        (app.ui_lang.get_label("target_text_color_label"), app.target_text_colour_var, 'target_text'),
        (
            app.ui_lang.get_label(
                "target_text_outline_color_label",
                "Outline Colour:",
            ),
            app.target_text_outline_colour_var,
            'target_outline',
        ),
    ]
    app.color_displays = {}
    for i, (label_text, var, color_type) in enumerate(color_options):
        color_item_frame = ttk.Frame(app.colors_row_frame)
        grid_row, grid_column = _settings_color_grid_position(i)
        color_item_frame.grid(
            row=grid_row,
            column=grid_column,
            padx=(0, 18),
            pady=(0, 6),
            sticky="w",
        )
        ttk.Label(color_item_frame, text=label_text).pack(
            side=tk.LEFT,
            padx=(0, 5),
        )
        color_display = tk.Label(
            color_item_frame,
            width=3,
            relief="solid",
            bg=var.get(),
            cursor="hand2",
        )
        color_display.pack(side=tk.LEFT)
        color_display.bind(
            "<Button-1>",
            lambda _event, ct=color_type: app.choose_color_for_settings(ct),
        )
        app.color_displays[color_type] = color_display
    current_row += 1

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

    ttk.Label(
        frame,
        text=app.ui_lang.get_label(
            "target_text_outline_width_label",
            "Outline Width:",
        ),
    ).grid(row=current_row, column=0, padx=5, pady=5, sticky="w")
    app.target_text_outline_width_spinbox = ttk.Spinbox(
        frame,
        from_=0, to=6,
        textvariable=app.target_text_outline_width_var,
        width=10,
        validate="key",
        validatecommand=(validate_outline_width, '%P'),
        command=app.update_target_text_outline,
    )
    app.target_text_outline_width_spinbox.grid(
        row=current_row,
        column=1,
        padx=5,
        pady=5,
        sticky="w",
    )

    def on_outline_width_focus_out(_event):
        try:
            value = int(app.target_text_outline_width_var.get())
            app.target_text_outline_width_var.set(max(0, min(6, value)))
        except (ValueError, tk.TclError):
            app.target_text_outline_width_var.set(2)
        app.update_target_text_outline()
        app.save_settings()

    app.target_text_outline_width_spinbox.bind(
        "<FocusOut>",
        on_outline_width_focus_out,
    )
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

    app.translation_text_style_frame = ttk.Frame(frame)
    app.translation_text_style_frame.grid(
        row=current_row,
        column=0,
        columnspan=3,
        padx=5,
        pady=5,
        sticky="w",
    )
    app.target_font_bold_checkbox = ttk.Checkbutton(
        app.translation_text_style_frame,
        text=app.ui_lang.get_label(
            "font_bold_label",
            "Bold translated subtitle text",
        ),
        variable=app.target_font_bold_var,
        command=app.update_target_font_weight,
    )
    app.target_font_bold_checkbox.pack(
        side=tk.LEFT,
        padx=(0, 18),
    )

    app.translation_horizontal_centered_checkbox = ttk.Checkbutton(
        app.translation_text_style_frame,
        text=app.ui_lang.get_label(
            "translation_horizontal_centered",
            "Center translated subtitle horizontally",
        ),
        variable=app.translation_horizontal_centered_var,
    )
    app.translation_horizontal_centered_checkbox.pack(side=tk.LEFT)
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
