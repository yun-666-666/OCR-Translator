"""UI configuration, model selection, diagnostics, and prompt persistence."""

import os
import re
import sys
import time
from tkinter import messagebox

from custom_ai import CUSTOM_AI_LATENCY_MODE_SAFE, normalize_custom_ai_latency_mode
from gui_builder import (
    create_custom_prompt_tab,
    create_debug_tab,
    create_main_tab,
    create_settings_tab,
)
from logger import set_debug_logging_enabled
from ocr_utils import (
    API_OCR_IMAGE_DETAIL_DEFAULT,
    API_OCR_IMAGE_FORMAT_DEFAULT,
    API_OCR_IMAGE_MODE_DEFAULT,
    API_OCR_IMAGE_QUALITY_DEFAULT,
    normalize_api_ocr_image_detail,
    normalize_api_ocr_image_format,
    normalize_api_ocr_image_mode,
    normalize_api_ocr_image_quality,
)
from paddle_ocr_backend import PADDLEOCR_MODEL_CODE

DEFAULT_CUSTOM_PROMPT = (
    "Use context to resolve ambiguity. Translate naturally and concisely while preserving meaning, "
    "tone, and character voice. Keep names and game terms consistent."
)


def _log_debug(message):
    facade = sys.modules.get("app_logic")
    if facade is not None:
        return facade.log_debug(message)


class AppConfigurationMixin:
    def save_settings(self):
        if self._fully_initialized:
            saved = self.ui_interaction_handler.save_settings()
            if saved and not getattr(self, "_app_is_closing", False):
                try:
                    if self.get_ocr_model_setting() == PADDLEOCR_MODEL_CODE:
                        self.ensure_paddleocr_ready_if_selected("settings saved")
                except Exception as e:
                    _log_debug(f"PaddleOCR prewarm after settings save failed: {e}")
            return saved
        _log_debug("Attempted to save settings before full initialization.")
        return False

    def suppress_traces(self):
        """Suppress StringVar traces during UI updates to prevent cascading saves"""
        self._suppress_traces = True
        _log_debug("StringVar traces suppressed")

    def restore_traces(self):
        """Restore StringVar traces after UI updates complete"""
        self._suppress_traces = False
        _log_debug("StringVar traces restored")

    def start_ui_update(self):
        """Mark the start of a UI update operation to suppress all saves"""
        self._ui_update_in_progress = True
        self.suppress_traces()
        _log_debug("UI update operation started - all saves suppressed")

    def end_ui_update(self):
        """Mark the end of a UI update operation and restore normal save behavior"""
        self._ui_update_in_progress = False
        self.restore_traces()
        _log_debug("UI update operation ended - saves restored")

    def on_window_configure(self, event):
        self.configuration_handler.on_window_configure(event)

    def save_current_window_geometry(self):
        self.configuration_handler.save_current_window_geometry()

    def browse_marian_models_file(self):
        self.configuration_handler.browse_marian_models_file()

    def update_translation_text(self, text_to_display):
        self.display_manager.update_translation_text(text_to_display)

    def update_debug_display(self, original_img_pil, processed_img_cv, ocr_text_content):
        self.display_manager.update_debug_display(original_img_pil, processed_img_cv, ocr_text_content)

    def update_marian_active_model(self, model_name, source_lang=None, target_lang=None):
        return self.translation_handler.update_marian_active_model(model_name, source_lang, target_lang)

    def update_marian_beam_value(self):
        self.translation_handler.update_marian_beam_value()

    def choose_color_for_settings(self, color_type):
        self.ui_interaction_handler.choose_color_for_settings(color_type)

    def update_stability_from_spinbox(self):
        self.ui_interaction_handler.update_stability_from_spinbox()

    def update_target_font_size(self):
        self.ui_interaction_handler.update_target_font_size()

    def update_target_font_type(self):
        self.ui_interaction_handler.update_target_font_type()

    def update_target_font_weight(self):
        self.ui_interaction_handler.update_target_font_weight()

    def update_target_opacity(self):
        self.ui_interaction_handler.update_target_opacity()

    def update_target_text_opacity(self):
        self.ui_interaction_handler.update_target_text_opacity()

    def refresh_debug_log(self):
        self.ui_interaction_handler.refresh_debug_log()

    def save_debug_images(self):
        self.ui_interaction_handler.save_debug_images()

    def toggle_api_key_visibility(self, api_type):
        self.ui_interaction_handler.toggle_api_key_visibility(api_type)

    def update_translation_model_ui(self):
        self.ui_interaction_handler.update_translation_model_ui()

    def on_marian_model_selection_changed(self, event=None, preload=False, initial_setup=False):
        self.ui_interaction_handler.on_marian_model_selection_changed(event, preload, initial_setup)
        if not initial_setup and self._fully_initialized :
             self.save_settings()


    def on_translation_model_selection_changed(
        self,
        event=None,
        initial_setup=False,
        synchronize_ui_only=False,
    ):
        # Handle session management for translation method changes
        if (
            not synchronize_ui_only
            and hasattr(self, 'translation_handler')
            and self.is_running
            and not initial_setup
        ):
            current_model = self.translation_model_var.get()

            # End translation session if switching away from Gemini
            if current_model != 'gemini_api':
                self.translation_handler.request_end_translation_session()

            # Start translation session if switching to Gemini
            if current_model == 'gemini_api':
                self.translation_handler.start_translation_session()

            # Handle OpenAI session management if needed
            if current_model == 'openai_api':
                # OpenAI doesn't require special session management like Gemini
                # But we could add any OpenAI-specific initialization here if needed
                pass
                self.translation_handler.start_translation_session()

        self.ui_interaction_handler.on_translation_model_selection_changed(event, initial_setup)
        if (
            not synchronize_ui_only
            and not initial_setup
            and self._fully_initialized
        ):
            self.save_settings()

    def clear_debug_log(self):
        self.ui_interaction_handler.clear_debug_log()

    def reset_gemini_api_log(self):
        """Reset/clear the Gemini API call log file."""
        try:
            if hasattr(self.translation_handler, 'gemini_log_file'):
                log_file_path = self.translation_handler.gemini_log_file

                # Clear the file by truncating it
                if os.path.exists(log_file_path):
                    with open(log_file_path, 'w', encoding='utf-8') as f:
                        f.write('')  # Clear the file
                    _log_debug(f"Gemini API log file cleared: {log_file_path}")

                    # Reinitialize the log with header
                    if hasattr(self.translation_handler, '_initialize_gemini_log'):
                        self.translation_handler._initialize_gemini_log()

                    messagebox.showinfo(
                        self.ui_lang.get_label("gemini_reset_success_title", "Success"),
                        self.ui_lang.get_label("gemini_reset_success_msg", "Gemini API log has been reset.")
                    )
                else:
                    _log_debug(f"Gemini API log file does not exist: {log_file_path}")
                    messagebox.showwarning(
                        self.ui_lang.get_label("gemini_reset_warning_title", "Warning"),
                        self.ui_lang.get_label("gemini_reset_warning_msg", "Gemini API log file does not exist.")
                    )
            else:
                _log_debug("Gemini log file path not available")
                messagebox.showerror(
                    self.ui_lang.get_label("gemini_reset_error_title", "Error"),
                    self.ui_lang.get_label("gemini_reset_error_msg", "Could not access Gemini log file.")
                )
        except Exception as e:
            _log_debug(f"Error resetting Gemini API log: {e}")
            messagebox.showerror(
                self.ui_lang.get_label("gemini_reset_error_title", "Error"),
                f"{self.ui_lang.get_label('gemini_reset_error_failed', 'Failed to reset Gemini API log:')} {str(e)}"
            )

    def update_openai_stats(self):
        """Update the OpenAI statistics fields by reading the log file."""
        try:
            # Check if all required components are available
            if not hasattr(self, 'openai_total_words_var') or self.openai_total_words_var is None:
                _log_debug("OpenAI stats variables not initialized yet")
                return

            if not hasattr(self, 'openai_total_cost_var') or self.openai_total_cost_var is None:
                _log_debug("OpenAI total cost variable not initialized yet")
                return

            # Get cumulative totals from OpenAI log file
            total_words, total_cost = self._get_cumulative_openai_totals()

            # Update GUI fields
            self.openai_total_words_var.set(self.format_number_with_separators(total_words))
            self.openai_total_cost_var.set(self.format_cost_for_display(total_cost))

            _log_debug(f"Updated OpenAI stats: {total_words} words, ${total_cost:.8f}")
        except Exception as e:
            _log_debug(f"Error updating OpenAI stats: {e}")
            # Set default values if there's an error
            if hasattr(self, 'openai_total_words_var') and self.openai_total_words_var is not None:
                self.openai_total_words_var.set(self.format_number_with_separators(0))
            if hasattr(self, 'openai_total_cost_var') and self.openai_total_cost_var is not None:
                self.openai_total_cost_var.set(self.format_cost_for_display(0.0))

    def _get_cumulative_openai_totals(self):
        """Read the cumulative totals from the OpenAI API log file."""
        try:
            # Get the log file path
            if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
                base_dir = os.path.dirname(sys.executable)
            else:
                base_dir = os.path.dirname(os.path.abspath(__file__))

            openai_log_file = os.path.join(base_dir, "OpenAI_API_call_logs.txt")

            if not os.path.exists(openai_log_file):
                _log_debug(f"OpenAI log file does not exist: {openai_log_file}")
                return 0, 0.0

            # Read the most recent cumulative cost and words from the log
            cumulative_cost = 0.0
            cumulative_words = 0

            with open(openai_log_file, 'r', encoding='utf-8') as f:
                content = f.read()

                # Find all instances of cumulative totals
                cost_matches = re.findall(r'Cumulative Log Cost: \$([0-9.]+)', content)
                word_matches = re.findall(r'Total Translated Words \(so far\): ([0-9,]+)', content)

                if cost_matches:
                    cumulative_cost = float(cost_matches[-1])  # Get the last (most recent) value

                if word_matches:
                    # Remove commas from word count and convert to int
                    word_str = word_matches[-1].replace(',', '')
                    cumulative_words = int(word_str)

            _log_debug(f"OpenAI cumulative totals: {cumulative_words} words, ${cumulative_cost:.8f}")
            return cumulative_words, cumulative_cost

        except Exception as e:
            _log_debug(f"Error reading OpenAI cumulative totals: {e}")
            return 0, 0.0

    def reset_openai_api_log(self):
        """Reset/clear the OpenAI API call log file."""
        try:
            if hasattr(self.translation_handler, 'openai_log_file'):
                log_file_path = self.translation_handler.openai_log_file

                # Clear the file by truncating it
                if os.path.exists(log_file_path):
                    with open(log_file_path, 'w', encoding='utf-8') as f:
                        f.write('')  # Clear the file
                    _log_debug(f"OpenAI API log file cleared: {log_file_path}")

                    # Reinitialize the log with header
                    if hasattr(self.translation_handler, '_initialize_openai_log'):
                        self.translation_handler._initialize_openai_log()

                    # Update the GUI fields
                    self.update_openai_stats()

                    messagebox.showinfo(
                        self.ui_lang.get_label("openai_reset_success_title", "Success"),
                        self.ui_lang.get_label("openai_reset_success_msg", "OpenAI API log has been reset.")
                    )
                else:
                    _log_debug(f"OpenAI API log file does not exist: {log_file_path}")
                    messagebox.showwarning(
                        self.ui_lang.get_label("openai_reset_warning_title", "Warning"),
                        self.ui_lang.get_label("openai_reset_warning_msg", "OpenAI API log file does not exist.")
                    )
            else:
                _log_debug("OpenAI log file path not available")
                messagebox.showerror(
                    self.ui_lang.get_label("openai_reset_error_title", "Error"),
                    self.ui_lang.get_label("openai_reset_error_msg", "Could not access OpenAI log file.")
                )
        except Exception as e:
            _log_debug(f"Error resetting OpenAI API log: {e}")
            messagebox.showerror(
                self.ui_lang.get_label("openai_reset_error_title", "Error"),
                f"{self.ui_lang.get_label('openai_reset_error_failed', 'Failed to reset OpenAI API log:')} {str(e)}"
            )

    def format_currency_for_display(self, amount, unit_suffix=""):
        """Format currency amount according to current UI language."""
        try:
            if self.ui_lang.current_lang == 'pol':
                # Polish format: "0,04941340 USD/min"
                amount_str = f"{amount:.8f}"
                amount_str = amount_str.replace('.', ',')  # Replace decimal point with comma

                # Add thousand separators (space) for large numbers
                parts = amount_str.split(',')
                integer_part = parts[0]
                decimal_part = parts[1] if len(parts) > 1 else ""

                # Add space thousand separators to integer part
                if len(integer_part) > 3:
                    formatted_integer = ""
                    for i, digit in enumerate(reversed(integer_part)):
                        if i > 0 and i % 3 == 0:
                            formatted_integer = " " + formatted_integer
                        formatted_integer = digit + formatted_integer
                    integer_part = formatted_integer

                if decimal_part:
                    amount_str = f"{integer_part},{decimal_part}"
                else:
                    amount_str = integer_part

                # Translate unit suffixes for Polish
                if unit_suffix == "/min":
                    unit_suffix = " USD/min"
                elif unit_suffix == "/hr":
                    unit_suffix = " USD/godz."
                elif unit_suffix == "":
                    unit_suffix = " USD"

                return f"{amount_str}{unit_suffix}"
            else:
                # English format: "$0.04941340/min"
                prefix = "$" if not unit_suffix else "$"
                return f"{prefix}{amount:.8f}{unit_suffix}"
        except Exception as e:
            _log_debug(f"Error formatting currency: {e}")
            return f"${amount:.8f}{unit_suffix}"  # Fallback to English format

    def format_cost_for_display(self, cost_value):
        """Format cost value according to current UI language (legacy method)."""
        return self.format_currency_for_display(cost_value, " USD" if self.ui_lang.current_lang == 'pol' else "")

    def format_number_with_separators(self, number):
        """Format integer numbers with thousand separators according to current UI language."""
        try:
            # Convert to integer to avoid decimal formatting issues
            num = int(number)

            if self.ui_lang.current_lang == 'pol':
                # Polish format: use space as thousand separator
                num_str = str(num)
                if len(num_str) > 3:
                    formatted = ""
                    for i, digit in enumerate(reversed(num_str)):
                        if i > 0 and i % 3 == 0:
                            formatted = " " + formatted
                        formatted = digit + formatted
                    return formatted
                else:
                    return num_str
            else:
                # English format: use comma as thousand separator
                return f"{num:,}"
        except Exception as e:
            _log_debug(f"Error formatting number with separators: {e}")
            return str(number)  # Fallback to string representation

    def toggle_debug_logging(self):
        """Toggle debug logging on/off and update button text."""
        current_state = self.debug_logging_enabled_var.get()
        new_state = not current_state

        # Log state change before changing the state
        if current_state:
            _log_debug("Debug logging disabled by user")

        # Update the state
        self.debug_logging_enabled_var.set(new_state)
        set_debug_logging_enabled(new_state)

        # Log state change after enabling (if we're enabling)
        if new_state:
            _log_debug("Debug logging enabled by user")

        # Update button text
        if hasattr(self, 'debug_log_toggle_btn') and self.debug_log_toggle_btn.winfo_exists():
            if new_state:
                button_text = self.ui_lang.get_label("toggle_debug_log_disable_btn")
            else:
                button_text = self.ui_lang.get_label("toggle_debug_log_enable_btn")
            self.debug_log_toggle_btn.config(text=button_text)

        # Save settings
        if self._fully_initialized:
            self.save_settings()

    def update_translation_model_names(self):
        """Update translation model names with localized strings from CSV files."""
        self.translation_model_names = {
            'custom_ai': self.ui_lang.get_label("translation_model_custom_ai", "Custom AI Translation")
        }
        self.translation_model_values = {v: k for k, v in self.translation_model_names.items()}
        _log_debug(f"Updated translation model names: {self.translation_model_names}")

    def get_custom_ai_latency_mode(self):
        """Return the selected Custom AI response mode, normalized to a supported value."""
        var = getattr(self, 'custom_ai_latency_mode_var', None)
        try:
            return normalize_custom_ai_latency_mode(var.get() if var is not None else CUSTOM_AI_LATENCY_MODE_SAFE)
        except Exception:
            return CUSTOM_AI_LATENCY_MODE_SAFE

    def get_custom_ai_ocr_image_format(self):
        """Return the selected Custom AI OCR image file format."""
        var = getattr(self, 'custom_ai_ocr_image_format_var', None)
        try:
            return normalize_api_ocr_image_format(var.get() if var is not None else API_OCR_IMAGE_FORMAT_DEFAULT)
        except Exception:
            return API_OCR_IMAGE_FORMAT_DEFAULT

    def get_custom_ai_ocr_image_mode(self):
        """Return the selected Custom AI OCR image encoding mode."""
        var = getattr(self, 'custom_ai_ocr_image_mode_var', None)
        try:
            return normalize_api_ocr_image_mode(var.get() if var is not None else API_OCR_IMAGE_MODE_DEFAULT)
        except Exception:
            return API_OCR_IMAGE_MODE_DEFAULT

    def get_custom_ai_ocr_image_quality(self):
        """Return the selected Custom AI OCR image quality."""
        var = getattr(self, 'custom_ai_ocr_image_quality_var', None)
        try:
            return normalize_api_ocr_image_quality(var.get() if var is not None else API_OCR_IMAGE_QUALITY_DEFAULT)
        except Exception:
            return API_OCR_IMAGE_QUALITY_DEFAULT

    def get_custom_ai_ocr_image_detail(self):
        """Return the selected Custom AI OCR vision detail mode."""
        var = getattr(self, 'custom_ai_ocr_image_detail_var', None)
        try:
            return normalize_api_ocr_image_detail(var.get() if var is not None else API_OCR_IMAGE_DETAIL_DEFAULT)
        except Exception:
            return API_OCR_IMAGE_DETAIL_DEFAULT

    def get_current_gemini_model_for_translation(self):
        """Get the API name of currently selected Gemini translation model."""
        display_name = self.gemini_translation_model_var.get()
        return self.gemini_models_manager.get_api_name_by_display_name(display_name)

    def get_current_gemini_model_for_ocr(self):
        """Get the API name of currently selected Gemini OCR model."""
        display_name = self.gemini_ocr_model_var.get()
        return self.gemini_models_manager.get_api_name_by_display_name(display_name)

    def get_current_openai_model_for_translation(self):
        """Get the API name of currently selected OpenAI translation model."""
        display_name = self.openai_translation_model_var.get()
        return self.openai_models_manager.get_api_name_by_display_name(display_name)

    def is_openai_model(self, model_name):
        """Check if the given model name is an OpenAI model."""
        return False

    def is_gemini_model(self, model_name):
        """Check if the given model name is a Gemini model."""
        return False

    def get_current_openai_model_for_ocr(self):
        """Get the API name of currently selected OpenAI OCR model."""
        # Read from the new, specific variable
        display_name = self.openai_ocr_model_var.get()
        api_name = self.openai_models_manager.get_api_name_by_display_name(display_name)
        if api_name:
            return api_name
        return 'gpt-4o'  # Default fallback if lookup fails

    def is_api_based_ocr_model(self, model_name=None):
        """Check if the given (or current) OCR model is API-based and needs session management."""
        if model_name is None:
            model_name = self.get_ocr_model_setting()

        return model_name == 'custom_ai'

    def update_ui_language(self):
        """Rebuild visible UI tabs after the UI language changes."""
        try:
            self.start_ui_update()
            self.update_translation_model_names()
            selected_index = 0
            try:
                selected_index = self.tab_control.index(self.tab_control.select())
            except Exception:
                selected_index = 0

            for i in range(self.tab_control.index('end') - 1, -1, -1):
                self.tab_control.forget(i)

            self.tab_main = None
            self.tab_settings = None
            self.tab_custom_prompt = None
            self.tab_debug = None

            create_main_tab(self)
            create_settings_tab(self)
            create_custom_prompt_tab(self)
            create_debug_tab(self)

            self.ui_interaction_handler.update_translation_model_ui()
            self.ui_interaction_handler.update_ocr_model_ui()
            self.root.after_idle(lambda: self.ui_interaction_handler.update_ocr_model_ui())

            if self.translation_model_var.get() == 'custom_ai':
                active_profile = self.custom_ai_profiles.get_active_profile("translation")
                display_name = active_profile["name"] if active_profile else self.ui_lang.get_label("custom_ai_no_profiles", "Add an AI model profile")
                self.translation_model_display_var.set(display_name)

            self.ui_interaction_handler.update_all_dropdowns_for_language_change()

            if hasattr(self, 'update_deepl_model_type_for_language'):
                self.update_deepl_model_type_for_language()
            if hasattr(self, 'update_deepl_context_window_for_language'):
                self.update_deepl_context_window_for_language()
            if hasattr(self, 'update_gemini_context_window_for_language'):
                self.update_gemini_context_window_for_language()
            if hasattr(self, 'update_gemini_labels_for_language'):
                self.update_gemini_labels_for_language()

            def on_tab_changed(event):
                current_index = self.tab_control.index(self.tab_control.select())
                if current_index == 0 and hasattr(self, 'main_tab_start_button') and self.main_tab_start_button.winfo_exists():
                    self.main_tab_start_button.focus_set()
                elif current_index == 1 and hasattr(self, 'settings_tab_save_button') and self.settings_tab_save_button.winfo_exists():
                    self.settings_tab_save_button.focus_set()

            self.tab_control.bind("<<NotebookTabChanged>>", on_tab_changed)

            tab_count = self.tab_control.index('end')
            if tab_count > 0:
                self.tab_control.select(min(selected_index, tab_count - 1))

            if self.is_running:
                status_text = "Status: " + self.ui_lang.get_label("status_running", "Running (Press ~ to Stop)")
                self.start_stop_btn.config(text=self.ui_lang.get_label("stop_btn"))
            else:
                status_text = "Status: " + self.ui_lang.get_label("status_stopped", "Stopped (Press ~ to Start)")
                if not self.KEYBOARD_AVAILABLE:
                    status_text = self.ui_lang.get_label("status_ready", "Status: Ready")
            self.status_label.config(text=status_text)

            if hasattr(self, 'debug_log_toggle_btn') and self.debug_log_toggle_btn.winfo_exists():
                button_key = "toggle_debug_log_disable_btn" if self.debug_logging_enabled_var.get() else "toggle_debug_log_enable_btn"
                self.debug_log_toggle_btn.config(text=self.ui_lang.get_label(button_key))

            _log_debug(f"UI language completely rebuilt for: {self.ui_lang.current_lang}")
        except Exception as e:
            _log_debug(f"Error updating UI language: {e}")
        finally:
            self.end_ui_update()

    def setup_network_cleanup(self):
        """Setup periodic network connection cleanup to prevent stack corruption."""
        def cleanup_network_connections():
            try:
                # Force client recreation to clear connection pools
                if hasattr(self, 'translation_handler') and hasattr(self.translation_handler, 'gemini_client'):
                    if self.translation_handler.gemini_client is not None:
                        old_client = self.translation_handler.gemini_client

                        # Force client refresh
                        self.translation_handler._force_client_refresh()

                        # Try to close old client connections if possible
                        try:
                            if hasattr(old_client, 'close'):
                                old_client.close()
                            elif hasattr(old_client, '_transport') and hasattr(old_client._transport, 'close'):
                                old_client._transport.close()
                        except Exception as close_error:
                            _log_debug(f"Error closing old client: {close_error}")

                        _log_debug("Performed periodic network connection cleanup")

                # Also flush DNS cache
                self.flush_dns_cache_if_needed()

            except Exception as e:
                _log_debug(f"Error during periodic network cleanup: {e}")

            # Schedule next cleanup in 20 minutes
            if self.is_running:  # Only schedule if application is still running
                self.root.after(1200000, cleanup_network_connections)  # 20 minutes = 1200000ms

        # Start cleanup cycle after 20 minutes of operation
        self.root.after(1200000, cleanup_network_connections)
        _log_debug("Scheduled periodic network cleanup every 20 minutes")

    def flush_dns_cache_if_needed(self):
        """Flush system DNS cache if network performance degrades."""
        if not hasattr(self, 'last_dns_flush'):
            self.last_dns_flush = time.time()
            return

        current_time = time.time()
        # Flush DNS every hour during active use
        if current_time - self.last_dns_flush > 3600:  # 1 hour
            try:
                import subprocess
                result = subprocess.run(['ipconfig', '/flushdns'],
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    self.last_dns_flush = current_time
                    _log_debug("Successfully flushed DNS cache for network maintenance")
                else:
                    _log_debug(f"DNS flush command failed: {result.stderr}")
            except subprocess.TimeoutExpired:
                _log_debug("DNS flush command timed out")
            except Exception as e:
                _log_debug(f"Could not flush DNS cache: {e}")

    def load_custom_prompt(self):
        """Loads the custom prompt text from file."""
        try:
            if os.path.exists(self.custom_prompt_file):
                with open(self.custom_prompt_file, 'r', encoding='utf-8-sig') as f:
                    self.custom_prompt_text = f.read()
                if not self.custom_prompt_text.strip():
                    self.custom_prompt_text = DEFAULT_CUSTOM_PROMPT
                    with open(self.custom_prompt_file, 'w', encoding='utf-8-sig') as f:
                        f.write(self.custom_prompt_text)
                    _log_debug("Initialized empty custom prompt with default text")
                else:
                    _log_debug(f"Loaded custom prompt ({len(self.custom_prompt_text)} characters)")
            else:
                self.custom_prompt_text = DEFAULT_CUSTOM_PROMPT
                with open(self.custom_prompt_file, 'w', encoding='utf-8-sig') as f:
                    f.write(self.custom_prompt_text)
                _log_debug("Created custom prompt file with default text")
        except Exception as e:
            _log_debug(f"Error loading custom prompt: {e}")
            self.custom_prompt_text = DEFAULT_CUSTOM_PROMPT

    def save_custom_prompt(self, text):
        """Saves the custom prompt text to file."""
        try:
            self.custom_prompt_text = text
            with open(self.custom_prompt_file, 'w', encoding='utf-8-sig') as f:
                f.write(text)
            _log_debug(f"Saved custom prompt ({len(text)} characters)")
            return True
        except Exception as e:
            _log_debug(f"Error saving custom prompt: {e}")
            return False
