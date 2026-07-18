# handlers/ui_interaction_handler.py
import os
import time
import re
import tkinter as tk
from tkinter import messagebox, colorchooser
from config_manager import (
    TRANSLATION_LINE_LAYOUT_PRESERVE_SOURCE_LINES,
    normalize_translation_line_layout,
    save_app_config,
)
from logger import (
    clear_debug_log as clear_runtime_debug_log,
    log_debug,
    read_debug_log_tail,
)
from paddle_ocr_backend import PADDLEOCR_MODEL_CODE
import traceback


class UIInteractionHandler:
    def __init__(self, app):
        self.app = app
        self._save_in_progress = False
        self._last_save_time = 0
        self._save_debounce_interval = 0.1  # 100ms debounce

    def choose_color_for_settings(self, color_type):
        initial_color = ""
        if color_type == 'source': initial_color = self.app.source_colour_var.get()
        elif color_type == 'target': initial_color = self.app.target_colour_var.get()
        elif color_type == 'target_text': initial_color = self.app.target_text_colour_var.get()
        elif color_type == 'target_outline': initial_color = self.app.target_text_outline_colour_var.get()

        # Map color_type to appropriate translation key
        title_key_map = {
            'source': 'choose_source_color_title',
            'target': 'choose_target_color_title',
            'target_text': 'choose_target_text_color_title',
            'target_outline': 'choose_target_text_outline_color_title',
        }

        # Fallback titles for backward compatibility
        fallback_titles = {
            'source': 'Choose Source Color',
            'target': 'Choose Target Color',
            'target_text': 'Choose Target Text Color',
            'target_outline': 'Choose Target Text Outline Color',
        }

        title_key = title_key_map.get(color_type, 'choose_source_color_title')
        fallback_title = fallback_titles.get(color_type, 'Choose Color')
        dialog_title = self.app.ui_lang.get_label(title_key, fallback_title)

        chosen_color_tuple = colorchooser.askcolor(color=initial_color,
                                                 title=dialog_title,
                                                 parent=self.app.root)
        if chosen_color_tuple and chosen_color_tuple[1]:
            hex_color = chosen_color_tuple[1]

            if color_type == 'source':
                self.app.source_colour_var.set(hex_color)
                if hasattr(self.app, 'color_displays') and 'source' in self.app.color_displays:
                    self.app.color_displays['source'].configure(bg=hex_color)
                if self.app.source_overlay and self.app.source_overlay.winfo_exists():
                    self.app.source_overlay.update_color(hex_color)
            elif color_type == 'target':
                self.app.target_colour_var.set(hex_color)
                if hasattr(self.app, 'color_displays') and 'target' in self.app.color_displays:
                    self.app.color_displays['target'].configure(bg=hex_color)
                if self.app.target_overlay and self.app.target_overlay.winfo_exists():
                    self.app.target_overlay.update_color(hex_color)
                    if self.app.translation_text and self.app.translation_text.winfo_exists():
                        self.app.translation_text.configure(bg=hex_color)
            elif color_type == 'target_text':
                self.app.target_text_colour_var.set(hex_color)
                if hasattr(self.app, 'color_displays') and 'target_text' in self.app.color_displays:
                    self.app.color_displays['target_text'].configure(bg=hex_color)

                # Update text color for both tkinter and PySide overlays
                if self.app.translation_text and self.app.translation_text.winfo_exists():
                    self.app.translation_text.configure(fg=hex_color)

                # Also update the overlay if it has a specific text color update method
                if (self.app.target_overlay and
                    hasattr(self.app.target_overlay, 'update_text_color')):
                    self.app.target_overlay.update_text_color(hex_color)
            elif color_type == 'target_outline':
                self.app.target_text_outline_colour_var.set(hex_color)
                if hasattr(self.app, 'color_displays') and 'target_outline' in self.app.color_displays:
                    self.app.color_displays['target_outline'].configure(bg=hex_color)
                self.update_target_text_outline()
            log_debug(f"Color {color_type} changed to: {hex_color}")

    def update_translation_model_ui(self):
        selected_model_ui_code = self.app.translation_model_var.get()
        if selected_model_ui_code != 'custom_ai':
            selected_model_ui_code = 'custom_ai'
            self.app.translation_model_var.set('custom_ai')
        log_debug(f"Updating UI visibility for model code: {selected_model_ui_code}")

        def manage_grid(widget, show=True):
            if not widget or not hasattr(widget, 'grid_remove') or not hasattr(widget, 'grid'):
                return
            is_gridded = False
            try:
                if widget.grid_info():
                    is_gridded = True
            except tk.TclError:
                is_gridded = False
            if show and not is_gridded:
                widget.grid()
            elif not show and is_gridded:
                widget.grid_remove()

        is_custom = True
        is_custom_ai_ocr = (
            getattr(self.app, "ocr_model_var", None) is not None
            and self.app.ocr_model_var.get() == "custom_ai"
        )
        show_ai_optimization = is_custom or is_custom_ai_ocr

        for attr, show in [
            ('ai_optimization_mode_label', show_ai_optimization),
            ('ai_optimization_mode_combobox', show_ai_optimization),
            ('custom_ai_submit_interval_label', is_custom),
            ('custom_ai_submit_interval_spinbox', is_custom),
            ('source_lang_label', True),
            ('source_lang_combobox', True),
            ('target_lang_label', True),
            ('target_lang_combobox', True),
            ('keep_linebreaks_label', True),
            ('keep_linebreaks_checkbox', True),
        ]:
            manage_grid(getattr(self.app, attr, None), show=show)

        if hasattr(self.app, 'keep_linebreaks_checkbox') and self.app.keep_linebreaks_checkbox:
            self.app.keep_linebreaks_checkbox.config(state=tk.NORMAL)

        self._update_language_dropdowns_for_model('custom_ai')

    def update_ocr_model_ui(self):
        """Update UI visibility for OCR model-specific settings."""
        selected_ocr_model = self.app.ocr_model_var.get()
        log_debug(f"Updating OCR model UI for: {selected_ocr_model}")

        def manage_grid(widget, show=True):
            if widget and hasattr(widget, 'grid_remove') and hasattr(widget, 'grid'):
                is_gridded = False
                try:
                    if widget.grid_info():
                        is_gridded = True
                except tk.TclError:
                    is_gridded = False

                if show and not is_gridded:
                    widget.grid()
                elif not show and is_gridded:
                    widget.grid_remove()

        is_paddleocr = (selected_ocr_model == PADDLEOCR_MODEL_CODE)
        is_local_ocr = is_paddleocr
        is_custom_ai_ocr = selected_ocr_model == "custom_ai"
        is_custom_translation = (
            getattr(self.app, "translation_model_var", None) is not None
            and self.app.translation_model_var.get() == "custom_ai"
        )
        show_ai_optimization = is_custom_translation or is_custom_ai_ocr

        if hasattr(self.app, 'ai_optimization_mode_label'):
            manage_grid(
                self.app.ai_optimization_mode_label,
                show=show_ai_optimization,
            )
        if hasattr(self.app, 'ai_optimization_mode_combobox'):
            manage_grid(
                self.app.ai_optimization_mode_combobox,
                show=show_ai_optimization,
            )

        if hasattr(self.app, 'paddleocr_min_score_label'):
            manage_grid(self.app.paddleocr_min_score_label, show=is_paddleocr)
        if hasattr(self.app, 'paddleocr_min_score_spinbox'):
            manage_grid(self.app.paddleocr_min_score_spinbox, show=is_paddleocr)
        if hasattr(self.app, 'stability_label'):
            manage_grid(self.app.stability_label, show=is_local_ocr)
        if hasattr(self.app, 'stability_spinbox'):
            manage_grid(self.app.stability_spinbox, show=is_local_ocr)

        # Handle OCR debugging - keep it for local OCR engines.
        if hasattr(self.app, 'ocr_debugging_label'):
            manage_grid(self.app.ocr_debugging_label, show=is_local_ocr)
        if hasattr(self.app, 'ocr_debug_frame'):
            manage_grid(self.app.ocr_debug_frame, show=is_local_ocr)

        log_debug(
            f"OCR model UI updated for {selected_ocr_model}: "
            f"PaddleOCR local={'visible' if is_paddleocr else 'hidden'}"
        )

    def update_all_dropdowns_for_language_change(self):
        try:
            self._update_language_dropdowns_for_model('custom_ai')
        except Exception as e:
            log_debug(f"Error updating dropdowns for language change: {e}")

    def get_current_ui_language_for_lookup(self):
        """Get the current UI language in the format expected by localization methods."""
        current_ui_language = self.app.ui_lang.current_lang
        ui_language_for_lookup = 'polish' if current_ui_language == 'pol' else 'english'

        # Add debugging to track UI language state
        log_debug(f"UI Language Detection: app.ui_lang.current_lang='{current_ui_language}' -> lookup='{ui_language_for_lookup}'")

        return ui_language_for_lookup

    def _update_language_dropdowns_for_model(self, active_model_code):
        lm = self.app.language_manager
        ui_language_for_lookup = self.get_current_ui_language_for_lookup()

        def localized_names(lang_pairs):
            names = []
            for english_name, _code in lang_pairs:
                names.append(
                    lm.get_localized_language_name(
                        english_name, ui_language_for_lookup, 'custom_ai'
                    )
                )
            return names

        def set_display(var, api_code, names, fallback):
            display = (
                lm.get_name_from_code(api_code, 'custom_ai', 'source')
                or lm.get_name_from_code(api_code, 'custom_ai', 'target')
            )
            if display:
                display = lm.get_localized_language_name(
                    display, ui_language_for_lookup, 'custom_ai'
                )
            if not display or display not in names:
                display = names[0] if names else fallback
            var.set(display)

        source_pairs = lm.get_language_lists('custom_ai', 'source')
        target_pairs = lm.get_language_lists('custom_ai', 'target')
        source_names_list = localized_names(source_pairs)
        target_names_list = localized_names(target_pairs)

        if hasattr(self.app, 'source_lang_combobox') and self.app.source_lang_combobox:
            self.app.source_lang_combobox['values'] = source_names_list
        if hasattr(self.app, 'target_lang_combobox') and self.app.target_lang_combobox:
            self.app.target_lang_combobox['values'] = target_names_list

        set_display(
            self.app.source_display_var,
            getattr(self.app, 'custom_source_lang', 'auto'),
            source_names_list,
            'Auto',
        )
        set_display(
            self.app.target_display_var,
            getattr(self.app, 'custom_target_lang', 'en'),
            target_names_list,
            'English',
        )
        self.app.source_lang_var.set(getattr(self.app, 'custom_source_lang', 'auto'))
        self.app.target_lang_var.set(getattr(self.app, 'custom_target_lang', 'en'))
        log_debug(
            "Language dropdowns updated for custom_ai: "
            f"{getattr(self.app, 'custom_source_lang', 'auto')} -> "
            f"{getattr(self.app, 'custom_target_lang', 'en')} "
            f"[UI: {ui_language_for_lookup}]"
        )

    def on_translation_model_selection_changed(self, event=None, initial_setup=False):
        self.app.translation_model_var.set('custom_ai')
        self.update_translation_model_ui()
        if not initial_setup:
            self.app.save_settings()

    def update_stability_from_spinbox(self):
        try:
            new_threshold = self.app.stability_var.get()
            if new_threshold != self.app.stable_threshold:
                self.app.stable_threshold = new_threshold
        except tk.TclError: pass

    def _target_font_spec(self):
        font_size = self.app.target_font_size_var.get()
        font_type = self.app.target_font_type_var.get()
        font_bold_var = getattr(self.app, "target_font_bold_var", None)
        font_bold = bool(
            font_bold_var.get()
            if font_bold_var is not None
            else False
        )
        if font_bold:
            return (font_type, font_size, "bold")
        return (font_type, font_size)

    def _update_target_font(self):
        if self.app.translation_text and self.app.translation_text.winfo_exists():
            try:
                self.app.translation_text.configure(
                    font=self._target_font_spec()
                )
            except tk.TclError: pass

    def update_target_font_size(self):
        self._update_target_font()

    def update_target_font_type(self):
        self._update_target_font()

    def update_target_font_weight(self):
        self._update_target_font()

    def update_target_text_outline(self):
        target_overlay = getattr(self.app, "target_overlay", None)
        if (
            target_overlay
            and target_overlay.winfo_exists()
            and hasattr(target_overlay, "update_text_outline")
        ):
            try:
                outline_width = int(
                    self.app.target_text_outline_width_var.get()
                )
            except (AttributeError, TypeError, ValueError, tk.TclError):
                outline_width = 2
            outline_width = max(0, min(6, outline_width))
            try:
                target_overlay.update_text_outline(
                    self.app.target_text_outline_colour_var.get(),
                    outline_width,
                )
            except Exception as e:
                log_debug(f"Error updating target text outline: {e}")

    def update_target_opacity(self):
        """Update the background opacity of the translation overlay"""
        if self.app.target_overlay and self.app.target_overlay.winfo_exists():
            try:
                # For PySide overlays, update opacity through the update_color method
                if hasattr(self.app.target_overlay, 'update_color'):
                    current_bg_color = self.app.target_colour_var.get()
                    current_opacity = self.app.target_opacity_var.get()
                    self.app.target_overlay.update_color(current_bg_color, current_opacity)
                # For tkinter overlays, update alpha attribute
                else:
                    opacity = self.app.target_opacity_var.get()
                    self.app.target_overlay.attributes("-alpha", opacity)
            except Exception as e:
                from logger import log_debug
                log_debug(f"Error updating target opacity: {e}")

    def update_target_text_opacity(self):
        """Update the text opacity of the translation overlay"""
        if self.app.target_overlay and self.app.target_overlay.winfo_exists():
            try:
                # For PySide overlays, update text color with new opacity
                if hasattr(self.app.target_overlay, 'update_text_color'):
                    from overlay_manager import _hex_to_rgba_om
                    text_hex_color = self.app.target_text_colour_var.get()
                    text_opacity = self.app.target_text_opacity_var.get()
                    text_rgba_color = _hex_to_rgba_om(text_hex_color, text_opacity)
                    self.app.target_overlay.update_text_color(text_rgba_color)
                # For tkinter overlays, text opacity is not supported
                # (background and text share same opacity level)
            except Exception as e:
                from logger import log_debug
                log_debug(f"Error updating target text opacity: {e}")

    def refresh_debug_log(self):
        if not hasattr(self.app, 'log_text') or not self.app.log_text or not self.app.log_text.winfo_exists(): return
        try:
            self.app.log_text.config(state=tk.NORMAL)
            self.app.log_text.delete(1.0, tk.END)
            try:
                log_lines = read_debug_log_tail(max_lines=200)
                for line in log_lines:
                    self.app.log_text.insert(tk.END, line)
                self.app.log_text.see(tk.END)
            except FileNotFoundError:
                self.app.log_text.insert(tk.END, "Log file not found: translator_debug.log")
            except Exception as read_err:
                self.app.log_text.insert(tk.END, f"Error reading log: {read_err}")
            self.app.log_text.config(state=tk.DISABLED)
        except Exception as e: log_debug(f"Error refreshing log text: {e}")

    def save_debug_images(self):
        try:
            import cv2

            if not self.app.ocr_debugging_var.get():
                messagebox.showinfo("Debug", "OCR Debugging disabled.", parent=self.app.root); return
            if self.app.last_screenshot is None:
                messagebox.showinfo("Debug", "No screenshot captured.", parent=self.app.root); return
            debug_dir = "debug_images"; os.makedirs(debug_dir, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            original_fn = os.path.join(debug_dir, f"original_{ts}.png")
            self.app.last_screenshot.save(original_fn)
            log_debug(f"Saved original debug image: {original_fn}")
            if isinstance(self.app.last_processed_image, cv2.typing.MatLike) or isinstance(self.app.last_processed_image, type(cv2.imread('dummy.png'))): # Check type
                processed_fn = os.path.join(debug_dir, f"processed_ocr_{ts}.png")
                if cv2.imwrite(processed_fn, self.app.last_processed_image):
                    log_debug(f"Saved processed debug image: {processed_fn}")
                    messagebox.showinfo(
                        self.app.ui_lang.get_label("dialog_debug_images_saved_title", "Debug Images Saved"),
                        self.app.ui_lang.get_label("dialog_debug_images_saved_message", "Images saved to '{0}'.").format(debug_dir),
                        parent=self.app.root
                    )
                else: messagebox.showerror("Error", f"Failed to save processed image to {processed_fn}", parent=self.app.root)
            else:
                messagebox.showinfo(
                    self.app.ui_lang.get_label("dialog_debug_image_saved_title", "Debug Image Saved"),
                    self.app.ui_lang.get_label("dialog_debug_image_saved_message", "Original image saved to '{0}'. No processed image.").format(debug_dir),
                    parent=self.app.root
                )
        except Exception as e:
            log_debug(f"Error saving debug images: {e}")
            messagebox.showerror("Error", f"Failed to save debug images: {e}", parent=self.app.root)

    def save_settings(self, force=False):
        import time
        current_time = time.time()

        if self._save_in_progress:
            log_debug("Save settings already in progress, skipping duplicate save operation")
            return True

        # Debounce rapid successive save attempts
        if (
            not force
            and current_time - self._last_save_time < self._save_debounce_interval
        ):
            log_debug(f"Save settings debounced (last save {current_time - self._last_save_time:.3f}s ago)")
            return True

        # Check if UI update is in progress
        if hasattr(self.app, '_ui_update_in_progress') and self.app._ui_update_in_progress:
            log_debug("Save settings skipped - UI update in progress")
            return True

        self._last_save_time = current_time
        self._save_in_progress = True
        try:
            cfg = self.app.config['Settings']

            active_model_code = self.app.translation_model_var.get()
            cfg['translation_model'] = active_model_code

            # Validate and save language codes (never save display names!)
            custom_source = self.app.custom_source_lang
            custom_target = self.app.custom_target_lang

            def is_valid_code(code):
                if not code:
                    return False
                code = str(code)
                if (
                    '(' in code
                    or ')' in code
                    or 'na' in code.lower()
                    or any(
                        word in code.lower()
                        for word in [
                            'english',
                            'chinese',
                            'german',
                            'french',
                            'spanish',
                            'polish',
                        ]
                    )
                ):
                    return False
                return True

            if is_valid_code(custom_source):
                cfg['custom_source_lang'] = custom_source
                log_debug(f"Saving Custom AI source lang: {custom_source}")
            else:
                log_debug(
                    f"ERROR: Invalid Custom AI source lang code '{custom_source}' - not saving"
                )

            if is_valid_code(custom_target):
                cfg['custom_target_lang'] = custom_target
                log_debug(f"Saving Custom AI target lang: {custom_target}")
            else:
                log_debug(
                    f"ERROR: Invalid Custom AI target lang code '{custom_target}' - not saving"
                )
            cfg['custom_ai_profiles_file'] = self.app.config['Settings'].get(
                'custom_ai_profiles_file', 'custom_ai_profiles.json'
            )

            cfg['scan_interval'] = str(self.app.scan_interval_var.get())
            cfg['ocr_frame_cache_size'] = str(self.app.ocr_frame_cache_size_var.get())
            cfg['enable_instant_cache_display'] = str(self.app.enable_instant_cache_display_var.get())
            cfg['stability_threshold'] = str(self.app.stability_var.get())
            cfg['clear_translation_timeout'] = str(self.app.clear_translation_timeout_var.get())
            cfg['ocr_debugging'] = str(self.app.ocr_debugging_var.get())
            cfg['debug_logging_enabled'] = str(self.app.debug_logging_enabled_var.get())
            cfg['source_area_colour'] = self.app.source_colour_var.get()
            cfg['target_area_colour'] = self.app.target_colour_var.get()
            cfg['target_text_colour'] = self.app.target_text_colour_var.get()
            cfg['target_text_outline_colour'] = self.app.target_text_outline_colour_var.get()
            cfg['target_text_outline_width'] = str(self.app.target_text_outline_width_var.get())
            cfg['target_font_size'] = str(self.app.target_font_size_var.get())
            cfg['target_font_type'] = self.app.target_font_type_var.get()
            cfg['target_font_bold'] = str(self.app.target_font_bold_var.get())
            cfg['target_opacity'] = str(self.app.target_opacity_var.get())
            cfg['target_text_opacity'] = str(self.app.target_text_opacity_var.get())
            cfg['gui_language'] = self.app.ui_lang.normalize_display_name(self.app.gui_language_var.get())
            cfg['ocr_model'] = self.app.ocr_model_var.get()  # OCR Model Selection (Phase 2)
            translation_line_layout = normalize_translation_line_layout(
                self.app.translation_line_layout_var.get()
            )
            cfg['translation_line_layout'] = translation_line_layout
            cfg['translation_horizontal_centered'] = str(
                self.app.translation_horizontal_centered_var.get()
            )
            cfg['keep_linebreaks'] = str(
                translation_line_layout
                == TRANSLATION_LINE_LAYOUT_PRESERVE_SOURCE_LINES
            )


            # OpenAI-specific settings
            cfg['custom_context_window'] = str(self.app.custom_context_window_var.get())
            cfg['custom_ai_log_content_enabled'] = str(
                self.app.custom_ai_log_content_enabled_var.get()
            )
            cfg['ai_optimization_mode'] = self.app.get_ai_optimization_mode()
            cfg['custom_ai_submit_interval_ms'] = str(
                max(0, min(5000, int(self.app.custom_ai_submit_interval_ms_var.get())))
            )
            cfg['paddleocr_source_dir'] = self.app.paddleocr_source_dir_var.get()
            cfg['paddleocr_lang'] = self.app.paddleocr_lang_var.get()
            cfg['paddleocr_ocr_version'] = self.app.paddleocr_ocr_version_var.get()
            cfg['paddleocr_model_size'] = self.app.paddleocr_model_size_var.get()
            cfg['paddleocr_device'] = self.app.paddleocr_device_var.get()
            cfg['paddleocr_min_score'] = self.app.paddleocr_min_score_var.get()
            cfg['paddleocr_upscale'] = self.app.paddleocr_upscale_var.get()
            cfg['paddleocr_text_det_limit_side_len'] = self.app.paddleocr_text_det_limit_side_len_var.get()
            cfg['paddleocr_text_det_limit_type'] = self.app.paddleocr_text_det_limit_type_var.get()
            cfg['paddleocr_use_textline_orientation'] = str(
                self.app.paddleocr_use_textline_orientation_var.get()
            )
            if self.app.source_overlay and self.app.source_overlay.winfo_exists():
                area = self.app.source_overlay.get_geometry()
                if area: cfg['source_area_x1'], cfg['source_area_y1'], \
                         cfg['source_area_x2'], cfg['source_area_y2'] = map(str, area)
                cfg['source_area_visible'] = str(self.app.source_overlay.winfo_viewable())
            if self.app.target_overlay and self.app.target_overlay.winfo_exists():
                area = self.app.target_overlay.get_geometry()
                if area: cfg['target_area_x1'], cfg['target_area_y1'], \
                         cfg['target_area_x2'], cfg['target_area_y2'] = map(str, area)
                cfg['target_area_visible'] = str(self.app.target_overlay.winfo_viewable())

            self.app.configuration_handler.save_current_window_geometry()

            if not save_app_config(self.app.config):
                messagebox.showerror("Error", "Failed to write settings to config file.", parent=self.app.root)
                return False

            self.app.stable_threshold = int(cfg['stability_threshold'])
            self.app.clear_translation_timeout = int(cfg['clear_translation_timeout'])
            log_debug("Settings saved successfully by UIInteractionHandler.save_settings.")

            if hasattr(self.app, 'status_label') and self.app.status_label.winfo_exists():
                original_status_text = self.app.status_label.cget("text")
                self.app.status_label.config(text=self.app.ui_lang.get_label("settings_saved", "Status: Settings Saved"))
                if self.app.root.winfo_exists():
                    self.app.root.after(2000, lambda: self.app.status_label.config(text=original_status_text) if self.app.status_label.winfo_exists() else None)
            return True
        except Exception as e:
             log_debug(f"Error saving settings: {e}\n{traceback.format_exc()}")
             if self.app.root.winfo_exists():
                messagebox.showerror("Error", f"Failed to save settings:\n{e}", parent=self.app.root)
             return False
        finally:
            self._save_in_progress = False

    def clear_debug_log(self):
        try:
            clear_runtime_debug_log()
            self.refresh_debug_log()
            if hasattr(self.app, 'status_label') and self.app.status_label.winfo_exists():
                original_status_text = self.app.status_label.cget("text")
                self.app.status_label.config(text="Status: Debug log cleared")
                if self.app.root.winfo_exists():
                    self.app.root.after(2000, lambda: self.app.status_label.config(text=original_status_text) if self.app.status_label.winfo_exists() else None)
        except Exception as e:
            log_debug(f"Error clearing debug log: {e}")
            if self.app.root.winfo_exists():
                messagebox.showerror("Error", f"Failed to clear debug log: {e}", parent=self.app.root)
