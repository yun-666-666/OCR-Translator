"""UI configuration, model selection, diagnostics, and prompt persistence."""

import os
import sys
import time
from tkinter import messagebox

from ai_optimization import (
    AI_OPTIMIZATION_AUTO,
    ai_ocr_route_metric_name,
    normalize_ai_optimization_mode,
    resolve_ai_ocr_image_policy,
    resolve_ai_response_mode,
)
from gui_builder import (
    create_custom_prompt_tab,
    create_debug_tab,
    create_main_tab,
    create_settings_tab,
)
from logger import set_debug_logging_enabled
from paddle_ocr_backend import PADDLEOCR_MODEL_CODE
from rapid_ocr_backend import RAPIDOCR_MODEL_CODE

DEFAULT_CUSTOM_PROMPT = (
    "Translate naturally and concisely. Preserve meaning, tone, names, and terminology; "
    "use context only when needed."
)


def _log_debug(message):
    facade = sys.modules.get("app_logic")
    if facade is not None:
        return facade.log_debug(message)


class AppConfigurationMixin:
    def _cancel_scheduled_settings_save(self):
        timer_id = getattr(self, "_save_settings_timer", None)
        if timer_id is None:
            return False
        self._save_settings_timer = None
        try:
            self.root.after_cancel(timer_id)
        except Exception:
            pass
        return True

    def _flush_scheduled_settings_save(self):
        self._save_settings_timer = None
        if getattr(self, "_app_is_closing", False):
            return False
        return self.save_settings()

    def schedule_settings_save(self, delay_ms=400):
        if (
            not getattr(self, "_fully_initialized", False)
            or getattr(self, "_app_is_closing", False)
        ):
            return False
        self._cancel_scheduled_settings_save()
        try:
            self._save_settings_timer = self.root.after(
                max(0, int(delay_ms)),
                self._flush_scheduled_settings_save,
            )
            return True
        except Exception as error:
            self._save_settings_timer = None
            _log_debug(
                "Settings save scheduling failed: "
                f"{type(error).__name__} - {error}"
            )
            return self.save_settings()

    def save_settings(self, force=False):
        self._cancel_scheduled_settings_save()
        if self._fully_initialized:
            saved = self.ui_interaction_handler.save_settings(force=True)
            if saved and not getattr(self, "_app_is_closing", False):
                try:
                    selected_ocr = self.get_ocr_model_setting()
                    if selected_ocr == RAPIDOCR_MODEL_CODE:
                        self.ensure_rapidocr_ready_if_selected("settings saved")
                    elif selected_ocr == PADDLEOCR_MODEL_CODE:
                        self.ensure_paddleocr_ready_if_selected("settings saved")
                except Exception as e:
                    _log_debug(f"Local OCR prewarm after settings save failed: {e}")
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

    def update_translation_text(self, text_to_display):
        self.display_manager.update_translation_text(text_to_display)

    def update_debug_display(self, original_img_pil, processed_img_cv, ocr_text_content):
        self.display_manager.update_debug_display(original_img_pil, processed_img_cv, ocr_text_content)

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

    def update_target_text_outline(self):
        self.ui_interaction_handler.update_target_text_outline()

    def update_target_opacity(self):
        self.ui_interaction_handler.update_target_opacity()

    def update_target_text_opacity(self):
        self.ui_interaction_handler.update_target_text_opacity()

    def refresh_debug_log(self):
        self.ui_interaction_handler.refresh_debug_log()

    def save_debug_images(self):
        self.ui_interaction_handler.save_debug_images()

    def update_translation_model_ui(self):
        self.ui_interaction_handler.update_translation_model_ui()

    def on_translation_model_selection_changed(
        self,
        event=None,
        initial_setup=False,
        synchronize_ui_only=False,
    ):
        self.translation_model_var.set('custom_ai')
        self.ui_interaction_handler.on_translation_model_selection_changed(
            event, initial_setup
        )
        if (
            not synchronize_ui_only
            and not initial_setup
            and self._fully_initialized
        ):
            self.save_settings()

    def clear_debug_log(self):

        self.ui_interaction_handler.clear_debug_log()


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
        """Resolve the internal response mode from the unified AI policy."""
        var = getattr(self, 'ai_optimization_mode_var', None)
        try:
            if var is not None:
                return resolve_ai_response_mode(var.get())
        except Exception:
            pass

        legacy_var = getattr(self, 'custom_ai_latency_mode_var', None)
        try:
            legacy_mode = str(
                legacy_var.get() if legacy_var is not None else ""
            ).strip().lower()
            if legacy_mode in {"none", "safe", "stream", "race", "adaptive"}:
                return legacy_mode
        except Exception:
            pass
        return resolve_ai_response_mode(AI_OPTIMIZATION_AUTO)

    def get_ai_optimization_mode(self):
        """Return the normalized user-facing AI optimization policy."""
        var = getattr(self, 'ai_optimization_mode_var', None)
        try:
            return normalize_ai_optimization_mode(
                var.get() if var is not None else AI_OPTIMIZATION_AUTO
            )
        except Exception:
            return AI_OPTIMIZATION_AUTO

    def get_ai_ocr_image_decision(self, image_size=None):
        """Resolve the current API OCR image contract from policy and route health."""
        profile = None
        profiles = getattr(self, "custom_ai_profiles", None)
        if profiles is not None:
            try:
                profile = profiles.get_active_profile("ocr")
            except Exception as e:
                _log_debug(
                    "Could not resolve active OCR profile for image policy: "
                    f"{type(e).__name__} - {e}"
                )

        route_p90_seconds = 0.0
        route_sample_count = 0
        metrics = getattr(self, "runtime_metrics", None)
        if metrics is not None:
            try:
                timings = metrics.snapshot().get("timings", {})
                metric_name = (
                    ai_ocr_route_metric_name(profile)
                    if profile
                    else "api_ocr_duration"
                )
                timing = timings.get(metric_name, {})
                route_p90_seconds = timing.get("p90", 0.0)
                route_sample_count = timing.get("count", 0)
            except Exception:
                pass

        optimization_getter = getattr(self, "get_ai_optimization_mode", None)
        if callable(optimization_getter):
            optimization_mode = optimization_getter()
        else:
            optimization_var = getattr(self, "ai_optimization_mode_var", None)
            optimization_mode = normalize_ai_optimization_mode(
                optimization_var.get()
                if optimization_var is not None
                else AI_OPTIMIZATION_AUTO
            )

        return resolve_ai_ocr_image_policy(
            optimization_mode,
            profile=profile,
            image_size=image_size,
            route_p90_seconds=route_p90_seconds,
            route_sample_count=route_sample_count,
            capability_memory=getattr(
                self,
                "ai_ocr_image_capability_memory",
                None,
            ),
        )

    def get_custom_ai_ocr_image_format(self):
        """Return the automatically resolved Custom AI OCR image format."""
        return self.get_ai_ocr_image_decision().image_format

    def get_custom_ai_ocr_image_mode(self):
        """Return the automatically resolved Custom AI OCR image mode."""
        return self.get_ai_ocr_image_decision().image_mode

    def get_custom_ai_ocr_image_quality(self):
        """Return the automatically resolved Custom AI OCR image quality."""
        return self.get_ai_ocr_image_decision().image_quality

    def get_custom_ai_ocr_image_detail(self):
        """Return the automatically resolved Custom AI OCR vision detail."""
        return self.get_ai_ocr_image_decision().image_detail

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
