# gui_builder.py
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, colorchooser
import os
import threading
from logger import log_debug
from ui_elements import create_scrollable_tab
from modern_ui import style_tk_text_widget
import tkinter.font as tkFont

def filter_model_values(models, query):
    """Return model names containing the query, preserving the original order."""
    query = (query or "").strip().lower()
    if not query:
        return list(models)
    return [model for model in models if query in str(model).lower()]


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

    def validate_odd_int_range(P, min_val, max_val):
        if P == "": return True
        try:
            value = int(P)
            return min_val <= value <= max_val and value % 2 == 1  # Must be odd
        except ValueError: return False
            
    # Validation functions for parameters
    validate_block_size = frame.register(lambda P: validate_odd_int_range(P, 3, 101))
    validate_c_value = frame.register(lambda P: validate_int_range(P, -75, 75))
    validate_beam_size = frame.register(lambda P: validate_int_range(P, 1, 50))
    validate_scan_interval = frame.register(lambda P: validate_int_range(P, 50, 2000))
    validate_custom_context_window = frame.register(lambda P: validate_int_range(P, 0, 10))
    validate_timeout = frame.register(lambda P: validate_int_range(P, 0, 60))
    validate_stability = frame.register(lambda P: validate_int_range(P, 0, 5))
    validate_confidence = frame.register(lambda P: validate_int_range(P, 0, 100))
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
    
    ocr_models_available_for_ui = [app.ui_lang.get_label("ocr_model_tesseract", "Tesseract (offline)")]
    if getattr(app, 'WINDOWS_OCR_AVAILABLE', False):
        ocr_models_available_for_ui.append(app.ui_lang.get_label("ocr_model_windows", "Windows OCR (fast, offline)"))
    ocr_models_available_for_ui.extend([p["name"] for p in app.custom_ai_profiles.list_profiles(enabled_only=True)])
    
    
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
            if selected_display == app.ui_lang.get_label("ocr_model_tesseract", "Tesseract (offline)"):
                app.ocr_model_var.set('tesseract')
                log_debug("OCR model set to tesseract")
            elif selected_display == app.ui_lang.get_label("ocr_model_windows", "Windows OCR (fast, offline)") and getattr(app, 'WINDOWS_OCR_AVAILABLE', False):
                app.ocr_model_var.set('windows_ocr')
                log_debug("OCR model set to windows_ocr")
            else:
                for profile in app.custom_ai_profiles.list_profiles(enabled_only=True):
                    if profile["name"] == selected_display:
                        app.custom_ai_profiles.set_active_profile("ocr", profile["id"])
                        app.ocr_model_var.set('custom_ai')
                        log_debug(f"OCR model set to custom_ai profile: {selected_display}")
                        break
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

        ocr_names = [app.ui_lang.get_label("ocr_model_tesseract", "Tesseract (offline)")]
        if getattr(app, 'WINDOWS_OCR_AVAILABLE', False):
            ocr_names.append(app.ui_lang.get_label("ocr_model_windows", "Windows OCR (fast, offline)"))
        ocr_names.extend(enabled_names)
        app.ocr_model_combobox.config(values=ocr_names)
        active_ocr = app.custom_ai_profiles.get_active_profile("ocr")
        if app.ocr_model_var.get() == "custom_ai" and active_ocr:
            app.ocr_model_display_var.set(active_ocr["name"])
        elif app.ocr_model_var.get() == "windows_ocr" and getattr(app, 'WINDOWS_OCR_AVAILABLE', False):
            app.ocr_model_display_var.set(app.ui_lang.get_label("ocr_model_windows", "Windows OCR (fast, offline)"))
        elif app.ocr_model_var.get() != "custom_ai":
            app.ocr_model_display_var.set(app.ui_lang.get_label("ocr_model_tesseract", "Tesseract (offline)"))

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
            return
        app.ai_profile_selected_id = profile["id"]
        app.ai_profile_name_var.set(profile.get("name", ""))
        app.ai_profile_url_var.set(profile.get("base_url", ""))
        app.ai_profile_key_var.set(profile.get("api_key", ""))
        app.ai_profile_model_var.set(profile.get("model", ""))

    def on_profile_name_selected(event=None):
        load_profile(get_profile_by_name(app.ai_profile_name_var.get()))

    def build_profile_from_form():
        return {
            "name": app.ai_profile_name_var.get().strip(),
            "base_url": app.ai_profile_url_var.get().strip(),
            "api_key": app.ai_profile_key_var.get(),
            "model": app.ai_profile_model_var.get().strip(),
            "enabled": True,
        }

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

    profile_buttons = ttk.Frame(app.ai_profiles_frame, padding=(10, 8))
    profile_buttons.grid(row=0, column=2, rowspan=4, padx=(10, 8), pady=6, sticky="nsew")
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

    row_offset = 19
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

    # Store references to Tesseract-specific widgets for OCR model UI management
    app.tesseract_path_label = ttk.Label(frame, text=app.ui_lang.get_label("tesseract_path_label"))
    app.tesseract_path_label.grid(row=current_row, column=0, padx=5, pady=5, sticky="w")
    app.tesseract_path_frame = ttk.Frame(frame) # Store reference to the frame
    app.tesseract_path_frame.grid(row=current_row, column=1, columnspan=2, padx=5, pady=5, sticky="ew")
    app.tesseract_path_entry = ttk.Entry(app.tesseract_path_frame, textvariable=app.tesseract_path_var) 
    app.tesseract_path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
    ttk.Button(app.tesseract_path_frame, text=app.ui_lang.get_label("browse_btn"), command=app.browse_tesseract).pack(side=tk.RIGHT, padx=(5,0))
    current_row += 1

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

    app.confidence_label = ttk.Label(frame, text=app.ui_lang.get_label("confidence_threshold_label"))
    app.confidence_label.grid(row=current_row, column=0, padx=5, pady=5, sticky="w")
    app.confidence_spinbox = ttk.Spinbox(frame, from_=0, to=100, textvariable=app.confidence_var, 
                                   width=10, validate="key", validatecommand=(validate_confidence, '%P'))
    app.confidence_spinbox.grid(row=current_row, column=1, padx=5, pady=5, sticky="w")
    def on_confidence_focus_out(event):
        try:
            value = int(app.confidence_var.get())
            if not (0 <= value <= 100) : app.confidence_var.set(max(0, min(100, value)))
            app.confidence_threshold = app.confidence_var.get() 
        except (ValueError, tk.TclError): app.confidence_var.set(50)
        app.save_settings()
    app.confidence_spinbox.bind("<FocusOut>", on_confidence_focus_out)
    current_row += 1
    
    app.preprocessing_mode_label = ttk.Label(frame, text=app.ui_lang.get_label("preprocessing_mode_label"))
    app.preprocessing_mode_label.grid(row=current_row, column=0, padx=5, pady=5, sticky="w") 
    
    # Create a display mapping for the preprocessing mode
    preprocessing_values = {
        'none': app.ui_lang.get_label("preprocessing_none", "None"),
        'binary': app.ui_lang.get_label("preprocessing_binary", "Binary"),
        'binary_inv': app.ui_lang.get_label("preprocessing_binary_inv", "Binary Inverted"),
        'adaptive': app.ui_lang.get_label("preprocessing_adaptive", "Adaptive")
    }
    
    # Convert values to list for dropdown
    preprocessing_list = ['none', 'binary', 'binary_inv', 'adaptive']
    
    # Create a custom StringVar for display
    display_var = tk.StringVar()
    
    # Set current display value
    current_value = app.preprocessing_mode_var.get()
    display_var.set(preprocessing_values.get(current_value, preprocessing_values['none']))
    
    # Create combobox with display values
    preprocessing_display_values = [preprocessing_values[val] for val in preprocessing_list]
    app.preprocessing_mode_combobox = ttk.Combobox(frame, textvariable=display_var, 
                                       values=preprocessing_display_values, width=15, state='readonly')
    app.preprocessing_mode_combobox.grid(row=current_row, column=1, padx=5, pady=5, sticky="w")
    
    # Create mapping from display name back to value
    display_to_value = {display: value for value, display in preprocessing_values.items()}
    
    def on_preprocessing_combo_selected(event): 
        # Get selected display value
        selected_display = display_var.get()
        # Map back to actual value
        actual_value = display_to_value.get(selected_display, 'none')
        # Update the actual value
        app.preprocessing_mode_var.set(actual_value)
        app.save_settings()
    
    app.preprocessing_mode_combobox.bind('<<ComboboxSelected>>', 
        create_combobox_handler_wrapper(on_preprocessing_combo_selected))
    current_row += 1

    # Adaptive thresholding parameters (shown only when Adaptive is selected)
    app.adaptive_block_size_label = ttk.Label(frame, text=app.ui_lang.get_label("adaptive_block_size_label", "Block Size"))
    app.adaptive_block_size_label.grid(row=current_row, column=0, padx=5, pady=5, sticky="w")
    app.adaptive_block_size_spinbox = ttk.Spinbox(frame, from_=3, to=101, increment=2, textvariable=app.adaptive_block_size_var, 
                                                 width=10, validate="key", validatecommand=(validate_block_size, '%P'))
    app.adaptive_block_size_spinbox.grid(row=current_row, column=1, padx=5, pady=5, sticky="w")
    def on_block_size_focus_out(event):
        try:
            value = int(app.adaptive_block_size_var.get())
            # Ensure odd number in range
            if value % 2 == 0:
                value += 1  # Make odd
            clamped = max(3, min(101, value))
            if clamped != app.adaptive_block_size_var.get():
                app.adaptive_block_size_var.set(clamped)
        except (ValueError, tk.TclError): 
            app.adaptive_block_size_var.set(41)
        app.save_settings()
    app.adaptive_block_size_spinbox.bind("<FocusOut>", on_block_size_focus_out)
    current_row += 1

    app.adaptive_c_label = ttk.Label(frame, text=app.ui_lang.get_label("adaptive_c_label", "C Value"))
    app.adaptive_c_label.grid(row=current_row, column=0, padx=5, pady=5, sticky="w")
    app.adaptive_c_spinbox = ttk.Spinbox(frame, from_=-75, to=75, textvariable=app.adaptive_c_var, 
                                        width=10, validate="key", validatecommand=(validate_c_value, '%P'))
    app.adaptive_c_spinbox.grid(row=current_row, column=1, padx=5, pady=5, sticky="w")
    def on_c_value_focus_out(event):
        try:
            value = int(app.adaptive_c_var.get())
            clamped = max(-75, min(75, value))
            if clamped != app.adaptive_c_var.get():
                app.adaptive_c_var.set(clamped)
        except (ValueError, tk.TclError): 
            app.adaptive_c_var.set(-60)
        app.save_settings()
    app.adaptive_c_spinbox.bind("<FocusOut>", on_c_value_focus_out)
    current_row += 1

    # Function to show/hide adaptive parameters based on preprocessing mode AND OCR model
    def update_adaptive_fields_visibility():
        mode = app.preprocessing_mode_var.get()
        ocr_model = app.ocr_model_var.get()
        # Adaptive parameters should only be visible when using Tesseract OCR AND adaptive preprocessing
        should_show = (ocr_model == 'tesseract') and (mode == 'adaptive')
        
        if should_show:
            app.adaptive_block_size_label.grid()
            app.adaptive_block_size_spinbox.grid()
            app.adaptive_c_label.grid()
            app.adaptive_c_spinbox.grid()
        else:
            app.adaptive_block_size_label.grid_remove()
            app.adaptive_block_size_spinbox.grid_remove()
            app.adaptive_c_label.grid_remove()
            app.adaptive_c_spinbox.grid_remove()
    
    # Store reference to update function for later use
    app.update_adaptive_fields_visibility = update_adaptive_fields_visibility
    
    # Update the preprocessing combo selection handler to show/hide fields
    original_preprocessing_handler = on_preprocessing_combo_selected
    def on_preprocessing_combo_selected_with_visibility(event):
        original_preprocessing_handler(event)
        update_adaptive_fields_visibility()
    
    # Rebind with new handler
    app.preprocessing_mode_combobox.bind('<<ComboboxSelected>>', 
        create_combobox_handler_wrapper(on_preprocessing_combo_selected_with_visibility))
    
    # Initial visibility update
    update_adaptive_fields_visibility()
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

    # Store references to Tesseract-specific widgets for OCR model UI management
    app.remove_trailing_label = ttk.Label(frame, text=app.ui_lang.get_label("remove_trailing_label"))
    app.remove_trailing_label.grid(row=current_row, column=0, padx=5, pady=5, sticky="w")
    app.remove_trailing_checkbox = ttk.Checkbutton(frame, text=app.ui_lang.get_label("remove_trailing_checkbox"), variable=app.remove_trailing_garbage_var)
    app.remove_trailing_checkbox.grid(row=current_row, column=1, padx=5, pady=5, sticky="w")
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
