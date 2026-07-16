"""Capture, OCR adaptation, preview, and overlay responsibilities."""

import math
import sys
import time
import tkinter as tk
from dataclasses import replace
from tkinter import ttk

from config_manager import (
    load_ocr_preview_geometry,
    save_app_config,
    save_ocr_preview_geometry,
)
from modern_ui import style_tk_canvas, style_tk_text_widget
from ocr_utils import (
    encode_image_for_api_ocr_payload,
)
from overlay_manager import (
    create_source_overlay_om,
    create_target_overlay_om,
    load_areas_from_config_om,
    select_source_area_om,
    select_target_area_om,
    toggle_source_visibility_om,
    toggle_target_visibility_om,
)
from paddle_ocr_backend import (
    PADDLEOCR_MODEL_CODE,
    prepare_paddleocr_image,
    recognize_with_paddleocr,
)


CUSTOM_AI_OCR_CONCURRENCY_LIMIT = 2
LOCAL_OCR_ADAPTIVE_MIN_SAMPLES = 8
LOCAL_OCR_SCHEDULING_MARGIN_MS = 25
LOCAL_OCR_INTERVAL_QUANTUM_MS = 25
LOCAL_OCR_MAX_BASE_MULTIPLIER = 2


def _log_debug(message):
    facade = sys.modules.get("app_logic")
    if facade is not None:
        return facade.log_debug(message)


class AppCaptureOcrMixin:
    def convert_to_api_ocr_image(self, pil_image):
        """Convert a PIL image using the current automatic API OCR contract."""
        decision_getter = getattr(self, "get_ai_ocr_image_decision", None)
        if callable(decision_getter):
            decision = decision_getter(image_size=getattr(pil_image, "size", None))
        else:
            from ai_optimization import resolve_ai_ocr_image_policy

            decision = resolve_ai_ocr_image_policy("auto", image_size=pil_image.size)
        image_format = decision.image_format
        mode = decision.image_mode
        quality = decision.image_quality
        detail = decision.image_detail
        start = time.monotonic()

        try:
            encoded_image = encode_image_for_api_ocr_payload(
                pil_image,
                mode=mode,
                quality=quality,
                image_format=image_format,
            )
        except Exception as e:
            fallback_format = "png" if image_format == "jpeg" else "jpeg"
            _log_debug(
                "API OCR image encoding failed "
                f"format={image_format} mode={mode} quality={quality} detail={detail}: "
                f"{type(e).__name__} - {e}; retrying format={fallback_format}"
            )
            try:
                image_format = fallback_format
                encoded_image = encode_image_for_api_ocr_payload(
                    pil_image,
                    mode=mode,
                    quality=quality,
                    image_format=image_format,
                )
            except Exception as fallback_error:
                _log_debug(
                    "API OCR image encoding failed "
                    f"format={fallback_format} mode={mode} quality={quality} detail={detail}: "
                    f"{type(fallback_error).__name__} - {fallback_error}"
                )
                return None

        duration = time.monotonic() - start
        encoded_image = replace(
            encoded_image,
            image_detail=detail,
            policy_reason=decision.reason,
        )
        _log_debug(
            "API OCR image encoded "
            f"format={encoded_image.image_format} mime={encoded_image.mime_type} "
            f"mode={mode} bytes={len(encoded_image.data)} detail={detail} "
            f"policy_reason={decision.reason} duration={duration:.3f}s"
        )
        return encoded_image

    def convert_to_webp_for_api(self, pil_image):
        """Convert a PIL image for API OCR calls and return bytes for legacy callers."""
        encoded_image = AppCaptureOcrMixin.convert_to_api_ocr_image(self, pil_image)
        return encoded_image.data if encoded_image is not None else None

    def _pre_initialize_gemini_model(self):
        """Pre-configure Gemini API at startup to avoid thread initialization delays."""
        _log_debug("Gemini pre-initialization skipped; built-in Gemini provider is disabled.")

    # Gemini OCR Batch Processing Methods (Phase 1)
    def get_ocr_model_setting(self):
        """Get the current OCR model setting."""
        return self.ocr_model_var.get()

    def get_effective_ocr_concurrency_limit(self, provider_name=None):
        """Return the provider-aware OCR request capacity."""
        if provider_name is None:
            provider_name = self.get_ocr_model_setting()
        try:
            configured_limit = max(0, int(self.max_concurrent_ocr_calls))
        except (AttributeError, OverflowError, TypeError, ValueError):
            configured_limit = 1
        if provider_name == "custom_ai":
            return min(configured_limit, CUSTOM_AI_OCR_CONCURRENCY_LIMIT)
        return configured_limit

    def _get_local_ocr_adaptive_interval(self, base_interval):
        """Return a conservative local OCR interval from recent real work."""
        try:
            timing = (
                self.runtime_metrics.snapshot()
                .get("timings", {})
                .get("local_ocr_duration", {})
            )
            sample_count = int(timing.get("count", 0) or 0)
            p50_seconds = float(timing.get("p50", 0.0) or 0.0)
        except (AttributeError, OverflowError, TypeError, ValueError):
            return int(base_interval), 0, 0.0
        if (
            sample_count < LOCAL_OCR_ADAPTIVE_MIN_SAMPLES
            or not math.isfinite(p50_seconds)
            or p50_seconds <= 0.0
        ):
            return int(base_interval), sample_count, 0.0
        observed_ms = (
            p50_seconds * 1000.0
            + LOCAL_OCR_SCHEDULING_MARGIN_MS
        )
        rounded_ms = int(
            math.ceil(observed_ms / LOCAL_OCR_INTERVAL_QUANTUM_MS)
            * LOCAL_OCR_INTERVAL_QUANTUM_MS
        )
        adaptive_interval = max(
            int(base_interval),
            min(
                int(base_interval) * LOCAL_OCR_MAX_BASE_MULTIPLIER,
                rounded_ms,
            ),
        )
        return adaptive_interval, sample_count, p50_seconds

    def update_adaptive_scan_interval(self):
        """Adjust scan interval based on current OCR API load to prevent bottlenecks."""
        now = time.monotonic()

        # Check load every 2 seconds
        if now - self.load_check_timer < 2.0:
            return

        self.load_check_timer = now

        # Get user's preferred base interval
        base_interval = self.scan_interval_var.get()  # User's setting in milliseconds

        # Update base_scan_interval to track user changes
        self.base_scan_interval = base_interval

        selected_ocr_model = self.get_ocr_model_setting()
        if selected_ocr_model == PADDLEOCR_MODEL_CODE:
            (
                adaptive_interval,
                local_sample_count,
                local_p50_seconds,
            ) = self._get_local_ocr_adaptive_interval(base_interval)
            adaptive_state = (
                "local-runtime"
                if adaptive_interval > base_interval
                else "normal"
            )
            previous_log_state = getattr(
                self,
                "_last_adaptive_log_state",
                None,
            )
            previous_log_time = getattr(
                self,
                "_last_adaptive_log_time",
                0.0,
            )
            should_log_state = (
                adaptive_state != previous_log_state
                or now - previous_log_time >= 30.0
            )
            self.current_scan_interval = adaptive_interval
            self.overload_detected = adaptive_interval > base_interval
            if should_log_state:
                _log_debug(
                    "ADAPTIVE: local OCR runtime pacing "
                    f"p50={local_p50_seconds:.3f}s "
                    f"samples={local_sample_count} "
                    f"scan interval={adaptive_interval}ms"
                )
                self._last_adaptive_log_state = adaptive_state
                self._last_adaptive_log_time = now
            return

        # Measure current OCR load
        active_ocr_count = len(self.active_ocr_calls)
        max_ocr_calls = self.get_effective_ocr_concurrency_limit()
        overload_threshold = max(1, math.ceil(max_ocr_calls * 0.75))
        moderate_threshold = max(1, overload_threshold - 1)

        if active_ocr_count >= overload_threshold:
            adaptive_state = "overloaded"
        elif active_ocr_count >= moderate_threshold:
            adaptive_state = "moderate"
        else:
            adaptive_state = "normal"

        previous_log_state = getattr(self, "_last_adaptive_log_state", None)
        previous_log_time = getattr(self, "_last_adaptive_log_time", 0.0)
        should_log_state = (
            adaptive_state != previous_log_state
            or now - previous_log_time >= 30.0
        )
        if previous_log_state == "local-runtime":
            self.current_scan_interval = base_interval
            self.overload_detected = False
        adaptive_log_message = None

        if adaptive_state == "overloaded":
            if not self.overload_detected:
                # First detection of overload
                self.current_scan_interval = int(base_interval * 1.5)  # 150%
                self.overload_detected = True
                adaptive_log_message = (
                    f"ADAPTIVE: OCR overload detected ({active_ocr_count} active calls), "
                    f"increasing scan interval to {self.current_scan_interval}ms"
                )
            elif should_log_state:
                # Already in overload state, maintain increased interval
                adaptive_log_message = (
                    f"ADAPTIVE: OCR still overloaded ({active_ocr_count} active calls), "
                    f"maintaining scan interval at {self.current_scan_interval}ms"
                )
            # Stay at increased interval while overloaded

        elif adaptive_state == "normal":
            if self.overload_detected:
                # Load has decreased, return to normal
                self.current_scan_interval = base_interval
                self.overload_detected = False
                adaptive_log_message = (
                    f"ADAPTIVE: OCR load normalized ({active_ocr_count} active calls), "
                    f"returning scan interval to {self.current_scan_interval}ms"
                )
            elif should_log_state:
                # Normal state, no change needed
                adaptive_log_message = (
                    f"ADAPTIVE: OCR load normal ({active_ocr_count} active calls), "
                    f"scan interval remains at {self.current_scan_interval}ms"
                )
        else:
            # Near the provider limit, maintain the current interval.
            if should_log_state:
                adaptive_log_message = (
                    f"ADAPTIVE: OCR load moderate ({active_ocr_count} active calls), "
                    f"scan interval unchanged at {self.current_scan_interval}ms"
                )

        if adaptive_log_message:
            _log_debug(adaptive_log_message)
            self._last_adaptive_log_state = adaptive_state
            self._last_adaptive_log_time = now

    def handle_empty_ocr_result(self):
        """Handle <EMPTY> OCR result and manage clear translation timeout."""
        current_time = time.monotonic()

        # Only start timeout if we have a timeout value configured
        if self.clear_translation_timeout_var.get() <= 0:
            return  # Timeout disabled, do nothing

        if self.clear_timeout_timer_start is None:
            # First EMPTY result - start timer
            self.clear_timeout_timer_start = current_time
            _log_debug("Clear timeout timer started for <EMPTY> OCR result")
        else:
            # Check if timeout period exceeded
            elapsed = current_time - self.clear_timeout_timer_start
            timeout_seconds = self.clear_translation_timeout_var.get()

            if elapsed >= timeout_seconds:
                # Clear the translation display
                self.update_translation_text("")
                self.last_local_ocr_submitted_text = None
                self.last_local_ocr_submitted_norm = None
                self.last_local_ocr_submitted_scope = None
                self.reset_clear_timeout()
                _log_debug(f"Translation cleared after {elapsed:.1f}s timeout")

    def handle_successive_identical_subtitle(self, reason):
        """Handle identical subtitles that are the SAME as the immediately previous one."""
        # 1. Do NOT update caches (LRU, file cache) - no new content
        # 2. Do NOT update context window - successive identical subtitle
        # 3. Keep displaying last translation (no API call needed)
        # 4. Reset clear timeout (text is still present)

        self.reset_clear_timeout()  # Text still present
        # Display remains unchanged (last translation stays)
        # self.last_processed_subtitle stays the same (no change)
        _log_debug(f"Successive identical subtitle detected ({reason}), maintaining current translation")
        # No context window update - subtitle hasn't changed

    def reset_clear_timeout(self):
        """Reset clear translation timeout timer."""
        self.clear_timeout_timer_start = None
        _log_debug("Clear timeout timer reset - text detected")

    def show_ocr_preview(self):
        """Show/create the OCR Preview window."""
        # Check if window already exists and is valid
        if self.ocr_preview_window is not None:
            try:
                if self.ocr_preview_window.winfo_exists():
                    # Window already exists, just bring to front
                    self.ocr_preview_window.lift()
                    self.ocr_preview_window.attributes('-topmost', True)
                    self.ocr_preview_window.after(100, lambda: self.ocr_preview_window.attributes('-topmost', False))
                    return
            except tk.TclError:
                # Window was destroyed but variable wasn't cleared
                self.ocr_preview_window = None

        # Create new preview window
        self.ocr_preview_window = tk.Toplevel(self.root)
        self.ocr_preview_window.title(self.ui_lang.get_label("ocr_preview_title", "OCR Preview"))
        self.ocr_preview_window.minsize(400, 500)

        # Load window geometry from config
        load_ocr_preview_geometry(self.config, self.ocr_preview_window)

        # Create main frame
        main_frame = ttk.Frame(self.ocr_preview_window)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Image section - with horizontal scrollbar (no extra space) - NEW APPROACH
        image_frame = ttk.LabelFrame(main_frame, text=self.ui_lang.get_label("processed_image_preview", "Processed Image (1:1 scale)"))
        image_frame.pack(fill="both", expand=True, padx=5, pady=5)

        # Create frame for image content that won't expand
        content_frame = ttk.Frame(image_frame)
        content_frame.pack(fill="x", padx=5, pady=5)

        # Create canvas with scrollbars - but don't let it expand vertically
        image_canvas = tk.Canvas(content_frame, bd=0, highlightthickness=0, relief='flat', height=200)
        style_tk_canvas(image_canvas, self.md3_palette)
        h_scrollbar = ttk.Scrollbar(content_frame, orient="horizontal", command=image_canvas.xview)
        v_scrollbar = ttk.Scrollbar(content_frame, orient="vertical", command=image_canvas.yview)

        image_canvas.configure(xscrollcommand=h_scrollbar.set, yscrollcommand=v_scrollbar.set)

        # Pack with no expand for vertical
        v_scrollbar.pack(side="right", fill="y")
        h_scrollbar.pack(side="bottom", fill="x")
        image_canvas.pack(side="left", fill="both", expand=True)

        # Create label inside canvas for image display
        self.preview_image_label = ttk.Label(image_canvas, text=self.ui_lang.get_label("no_image_processed", "No image processed yet"),
                                            anchor="center", justify="center")

        # Add label to canvas
        self.preview_image_canvas_item = image_canvas.create_window(0, 0, anchor="nw", window=self.preview_image_label)

        # Store canvas reference for updating scroll region
        self.preview_image_canvas = image_canvas

        # Bind canvas resize to update scroll region
        def on_canvas_configure(event):
            # Update the scroll region to encompass the image
            image_canvas.configure(scrollregion=image_canvas.bbox("all"))

        image_canvas.bind('<Configure>', on_canvas_configure)

        # Text section
        text_frame = ttk.LabelFrame(main_frame, text=self.ui_lang.get_label("recognized_text_preview", "Recognized Text"))
        text_frame.pack(fill="x", padx=5, pady=5)

        self.preview_text_widget = tk.Text(text_frame, height=8, wrap=tk.WORD)
        style_tk_text_widget(self.preview_text_widget, self.md3_palette)
        text_scrollbar = ttk.Scrollbar(text_frame, orient="vertical", command=self.preview_text_widget.yview)
        self.preview_text_widget.configure(yscrollcommand=text_scrollbar.set)

        text_scrollbar.pack(side="right", fill="y")
        self.preview_text_widget.pack(side="left", fill="both", expand=True, padx=5, pady=5)

        # Control buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill="x", pady=5)

        ttk.Button(button_frame, text=self.ui_lang.get_label("refresh_preview", "Refresh Preview"),
                  command=self.refresh_ocr_preview).pack(side="left", padx=5)
        ttk.Button(button_frame, text=self.ui_lang.get_label("close_btn", "Close"),
                  command=self.close_ocr_preview).pack(side="right", padx=5)

        # Set up proper window close protocol
        self.ocr_preview_window.protocol("WM_DELETE_WINDOW", self.close_ocr_preview)

        # Set up window geometry saving on window events
        def on_preview_configure(event):
            if event.widget == self.ocr_preview_window:
                # Save geometry when window is moved or resized
                if hasattr(self, '_preview_geometry_timer'):
                    self.root.after_cancel(self._preview_geometry_timer)
                self._preview_geometry_timer = self.root.after(500, self.save_preview_geometry)

        self.ocr_preview_window.bind('<Configure>', on_preview_configure)

        # Start continuous real-time updates (regardless of translation state)
        self.start_preview_realtime_updates()

        # Initial preview update
        self.refresh_ocr_preview()

    def close_ocr_preview(self):
        """Properly close the OCR Preview window."""
        if self.ocr_preview_window is not None:
            try:
                # Save window geometry before closing
                self.save_preview_geometry()

                # Cancel any pending refresh timer
                if hasattr(self, '_preview_refresh_timer'):
                    self.root.after_cancel(self._preview_refresh_timer)

                # Cancel geometry save timer
                if hasattr(self, '_preview_geometry_timer'):
                    self.root.after_cancel(self._preview_geometry_timer)

                # Stop real-time updates
                self.stop_preview_realtime_updates()

                # Destroy the window
                self.ocr_preview_window.destroy()
            except tk.TclError:
                # Window might already be destroyed
                pass
            finally:
                # Always clear the reference
                self.ocr_preview_window = None

    def save_preview_geometry(self):
        """Save OCR Preview window geometry to config."""
        if self.ocr_preview_window is not None:
            try:
                save_ocr_preview_geometry(self.config, self.ocr_preview_window)
                save_app_config(self.config)
            except Exception as e:
                _log_debug(f"Error saving OCR Preview geometry: {e}")

    def start_preview_realtime_updates(self):
        """Start continuous real-time updates for OCR Preview window regardless of translation state."""
        if self.ocr_preview_window is not None:
            try:
                if self.ocr_preview_window.winfo_exists():
                    # Update every 500ms continuously (both when translation is running and stopped)
                    self._preview_realtime_timer = self.root.after(500, self.preview_realtime_update)
                else:
                    self.ocr_preview_window = None
            except tk.TclError:
                self.ocr_preview_window = None

    def stop_preview_realtime_updates(self):
        """Stop real-time updates for OCR Preview window."""
        if hasattr(self, '_preview_realtime_timer'):
            self.root.after_cancel(self._preview_realtime_timer)
            delattr(self, '_preview_realtime_timer')

    def preview_realtime_update(self):
        """Real-time update function for OCR Preview window - works regardless of translation state."""
        if self.ocr_preview_window is not None:
            try:
                if self.ocr_preview_window.winfo_exists():
                    # Refresh preview with current data (works both when translation is on/off)
                    self.refresh_ocr_preview()
                    # Schedule next update
                    self._preview_realtime_timer = self.root.after(500, self.preview_realtime_update)
                else:
                    self.ocr_preview_window = None
            except tk.TclError:
                self.ocr_preview_window = None

    def refresh_ocr_preview(self):
        """Refresh the OCR preview with current settings and captured image."""
        # Check if window still exists
        if self.ocr_preview_window is None:
            return

        try:
            if not self.ocr_preview_window.winfo_exists():
                self.ocr_preview_window = None
                return
        except tk.TclError:
            # Window was destroyed
            self.ocr_preview_window = None
            return

        try:
            # Always try to capture from source area for real-time preview (independent of translation state)
            screenshot_pil = None
            if self.source_overlay and self.source_overlay.winfo_exists():
                try:
                    area = self.source_overlay.get_geometry()
                    if area:
                        x1, y1, x2, y2 = map(int, area)
                        width, height = x2-x1, y2-y1
                        if width > 0 and height > 0:
                            import pyautogui
                            screenshot_pil = pyautogui.screenshot(region=(x1, y1, width, height))
                        else:
                            screenshot_pil = None
                    else:
                        screenshot_pil = None
                except Exception as e:
                    _log_debug(f"Error capturing for preview: {e}")
                    screenshot_pil = None

            # Fallback to using last_screenshot only if direct capture failed
            if screenshot_pil is None and hasattr(self, 'last_screenshot') and self.last_screenshot:
                screenshot_pil = self.last_screenshot

            if screenshot_pil:
                from PIL import Image, ImageTk
                current_ocr_model = self.get_ocr_model_setting()

                if current_ocr_model == PADDLEOCR_MODEL_CODE:
                    from worker_threads import get_paddleocr_settings_from_app

                    paddleocr_settings = get_paddleocr_settings_from_app(self)
                    processed_pil = prepare_paddleocr_image(screenshot_pil, paddleocr_settings)
                    ocr_cleaned_text, _lines = recognize_with_paddleocr(
                        screenshot_pil,
                        paddleocr_settings,
                        keep_linebreaks=bool(self.keep_linebreaks_var.get()),
                    )
                else:
                    processed_pil = screenshot_pil.convert("RGB")
                    ocr_cleaned_text = self.ui_lang.get_label(
                        "ocr_preview_local_only",
                        "OCR preview is available for PaddleOCR.",
                    )

                # Convert processed image to PIL for display
                processed_tk = ImageTk.PhotoImage(processed_pil)

                # Update image display in canvas
                self.preview_image_label.configure(image=processed_tk, text="")
                self.preview_image_label.image = processed_tk  # Keep reference

                # Update canvas scroll region to fit the image
                self.preview_image_label.update_idletasks()  # Ensure label has correct size
                image_width = processed_tk.width()
                image_height = processed_tk.height()

                # Adjust canvas height to fit image (with reasonable limits)
                canvas_height = min(image_height, 400)  # Max height of 400 pixels
                self.preview_image_canvas.configure(height=canvas_height)

                # Update the canvas window size and scroll region
                self.preview_image_canvas.itemconfig(self.preview_image_canvas_item, width=image_width, height=image_height)
                self.preview_image_canvas.configure(scrollregion=(0, 0, image_width, image_height))

                # Update text display
                self.preview_text_widget.config(state=tk.NORMAL)
                self.preview_text_widget.delete(1.0, tk.END)
                self.preview_text_widget.insert(tk.END, ocr_cleaned_text if ocr_cleaned_text else self.ui_lang.get_label("no_text_recognized", "No text recognized"))
                self.preview_text_widget.config(state=tk.DISABLED)

            else:
                # No image available
                self.preview_image_label.configure(image="", text=self.ui_lang.get_label("no_image_captured", "No image captured yet"))
                self.preview_image_label.image = None

                # Reset canvas to default size for text display
                self.preview_image_canvas.configure(height=100)  # Small height for text

                # Reset canvas scroll region for text display
                self.preview_image_label.update_idletasks()
                label_width = self.preview_image_label.winfo_reqwidth()
                label_height = self.preview_image_label.winfo_reqheight()

                self.preview_image_canvas.itemconfig(self.preview_image_canvas_item, width=label_width, height=label_height)
                self.preview_image_canvas.configure(scrollregion=(0, 0, label_width, label_height))

                self.preview_text_widget.config(state=tk.NORMAL)
                self.preview_text_widget.delete(1.0, tk.END)
                self.preview_text_widget.insert(tk.END, self.ui_lang.get_label("no_image_for_ocr", "No image available for OCR"))
                self.preview_text_widget.config(state=tk.DISABLED)

        except Exception as e:
            _log_debug(f"Error refreshing OCR preview: {e}")
            # Show error in preview
            if hasattr(self, 'preview_text_widget') and self.preview_text_widget.winfo_exists():
                self.preview_text_widget.config(state=tk.NORMAL)
                self.preview_text_widget.delete(1.0, tk.END)
                self.preview_text_widget.insert(tk.END, f"Error: {str(e)}")
                self.preview_text_widget.config(state=tk.DISABLED)

    def load_initial_overlay_areas(self):
        load_areas_from_config_om(self)

    def select_source_area(self):
        select_source_area_om(self)
        self.save_settings()

    def select_target_area(self):
        select_target_area_om(self)
        self.save_settings()

    def create_source_overlay(self):
        create_source_overlay_om(self)

    def create_target_overlay(self):
        create_target_overlay_om(self)  # System recreation, preserve position

    def toggle_source_visibility(self):
        toggle_source_visibility_om(self)
        self.save_settings()

    def toggle_target_visibility(self):
        toggle_target_visibility_om(self)
        self.save_settings()
