"""Capture, OCR adaptation, preview, and overlay responsibilities."""

import concurrent.futures
import math
import sys
import threading
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
    def convert_to_api_ocr_image(self, pil_image, decision=None):
        """Convert a PIL image using the current automatic API OCR contract."""
        if decision is None:
            decision_getter = getattr(self, "get_ai_ocr_image_decision", None)
            if callable(decision_getter):
                decision = decision_getter(
                    image_size=getattr(pil_image, "size", None)
                )
            else:
                from ai_optimization import resolve_ai_ocr_image_policy

                decision = resolve_ai_ocr_image_policy(
                    "auto",
                    image_size=pil_image.size,
                )
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
            self.publish_capture_ui_snapshot(reason="local adaptive scan interval")
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
        # Keep the capture worker on plain Python values.
        self.publish_capture_ui_snapshot(reason="adaptive scan interval")

    def publish_capture_ui_snapshot(self, *, bump_generation=False, reason=""):
        """Publish an immutable capture snapshot for the worker hot path."""
        from worker_capture import publish_capture_ui_snapshot

        return publish_capture_ui_snapshot(
            self,
            bump_generation=bump_generation,
            reason=reason,
        )

    def get_capture_ui_snapshot(self):
        from worker_capture import get_capture_ui_snapshot

        return get_capture_ui_snapshot(self)

    def start_capture_ui_snapshot_refresh(self):
        """Refresh adaptive capture inputs from the Tk thread while running."""
        self.stop_capture_ui_snapshot_refresh()
        generation = int(
            getattr(self, "_capture_ui_snapshot_refresh_generation", 0) or 0
        ) + 1
        self._capture_ui_snapshot_refresh_generation = generation
        self._refresh_capture_ui_snapshot_on_ui_thread(generation)

    def stop_capture_ui_snapshot_refresh(self):
        """Cancel the UI-owned adaptive snapshot refresh, if one is pending."""
        after_id = getattr(self, "_capture_ui_snapshot_refresh_after_id", None)
        self._capture_ui_snapshot_refresh_after_id = None
        self._capture_ui_snapshot_refresh_generation = int(
            getattr(self, "_capture_ui_snapshot_refresh_generation", 0) or 0
        ) + 1
        if after_id is not None:
            try:
                self.root.after_cancel(after_id)
            except Exception:
                pass

    def _refresh_capture_ui_snapshot_on_ui_thread(self, generation):
        """Run adaptive interval reads and snapshot publication on Tk's thread."""
        if generation != getattr(self, "_capture_ui_snapshot_refresh_generation", 0):
            return
        if not getattr(self, "is_running", False):
            self._capture_ui_snapshot_refresh_after_id = None
            return
        self.update_adaptive_scan_interval()
        if generation != getattr(self, "_capture_ui_snapshot_refresh_generation", 0):
            return
        if not getattr(self, "is_running", False):
            self._capture_ui_snapshot_refresh_after_id = None
            return
        try:
            self._capture_ui_snapshot_refresh_after_id = self.root.after(
                500,
                self._refresh_capture_ui_snapshot_on_ui_thread,
                generation,
            )
        except Exception:
            self._capture_ui_snapshot_refresh_after_id = None

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

    def _ensure_preview_ocr_runtime(self):
        """Initialize Preview OCR offload state on first use."""
        if not hasattr(self, "_preview_ocr_lock"):
            self._preview_ocr_lock = threading.Lock()
        if not hasattr(self, "_preview_ocr_generation"):
            self._preview_ocr_generation = 0
        if not hasattr(self, "_preview_ocr_in_flight"):
            self._preview_ocr_in_flight = False
        if not hasattr(self, "_preview_ocr_pending_frame"):
            self._preview_ocr_pending_frame = None
        if not hasattr(self, "_preview_ocr_executor") or self._preview_ocr_executor is None:
            self._preview_ocr_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="PreviewOCR",
            )
        if not hasattr(self, "_preview_last_ocr_text"):
            self._preview_last_ocr_text = None

    def _bump_preview_ocr_generation(self, reason=""):
        """Invalidate in-flight Preview OCR completions."""
        self._ensure_preview_ocr_runtime()
        with self._preview_ocr_lock:
            self._preview_ocr_generation = int(self._preview_ocr_generation or 0) + 1
            self._preview_ocr_pending_frame = None
            generation = self._preview_ocr_generation
        if reason:
            _log_debug(f"Preview OCR generation advanced to {generation} ({reason})")
        return generation

    def shutdown_preview_ocr_executor(self):
        """Stop accepting Preview OCR work during application shutdown."""
        executor = getattr(self, "_preview_ocr_executor", None)
        if executor is None:
            return False

        self._bump_preview_ocr_generation("application closing")
        lock = getattr(self, "_preview_ocr_lock", None)
        if lock is not None:
            with lock:
                self._preview_ocr_pending_frame = None

        try:
            executor.shutdown(wait=False, cancel_futures=True)
        except Exception as error:
            _log_debug(
                "Preview OCR executor shutdown failed: "
                f"{type(error).__name__} - {error}"
            )
            return False
        return True

    def _discard_preview_ocr_completion_from_worker(self, reason):
        """Release worker state when Tk can no longer accept a completion."""
        lock = getattr(self, "_preview_ocr_lock", None)
        if lock is not None:
            with lock:
                self._preview_ocr_in_flight = False
                self._preview_ocr_pending_frame = None
        _log_debug(f"Preview OCR completion discarded: {reason}")

    def _preview_window_is_open(self):
        window = getattr(self, "ocr_preview_window", None)
        if window is None:
            return False
        try:
            return bool(window.winfo_exists())
        except tk.TclError:
            self.ocr_preview_window = None
            return False

    def _capture_preview_screenshot(self):
        """Capture the current source area for Preview without running OCR."""
        screenshot_pil = None
        source_overlay = getattr(self, "source_overlay", None)
        if source_overlay is not None:
            try:
                if source_overlay.winfo_exists():
                    area = source_overlay.get_geometry()
                    if area:
                        x1, y1, x2, y2 = map(int, area)
                        width, height = x2 - x1, y2 - y1
                        if width > 0 and height > 0:
                            from ocr_utils import capture_screen_region

                            screenshot_pil = capture_screen_region(
                                (x1, y1, width, height)
                            )
            except Exception as error:
                _log_debug(f"Error capturing for preview: {error}")
                screenshot_pil = None

        if screenshot_pil is None and getattr(self, "last_screenshot", None):
            screenshot_pil = self.last_screenshot
        return screenshot_pil

    def _preview_placeholder_text(self):
        latest_main = getattr(self, "last_processed_subtitle", None)
        if isinstance(latest_main, str) and latest_main.strip():
            return latest_main
        cached = getattr(self, "_preview_last_ocr_text", None)
        if isinstance(cached, str) and cached.strip():
            return cached
        return self.ui_lang.get_label(
            "no_text_recognized",
            "No text recognized",
        )

    def _apply_preview_image(self, processed_pil):
        from PIL import ImageTk

        processed_tk = ImageTk.PhotoImage(processed_pil)
        self.preview_image_label.configure(image=processed_tk, text="")
        self.preview_image_label.image = processed_tk
        self.preview_image_label.update_idletasks()
        image_width = processed_tk.width()
        image_height = processed_tk.height()
        canvas_height = min(image_height, 400)
        self.preview_image_canvas.configure(height=canvas_height)
        self.preview_image_canvas.itemconfig(
            self.preview_image_canvas_item,
            width=image_width,
            height=image_height,
        )
        self.preview_image_canvas.configure(
            scrollregion=(0, 0, image_width, image_height)
        )

    def _apply_preview_text(self, text):
        display_text = text if text else self.ui_lang.get_label(
            "no_text_recognized",
            "No text recognized",
        )
        self.preview_text_widget.config(state=tk.NORMAL)
        self.preview_text_widget.delete(1.0, tk.END)
        self.preview_text_widget.insert(tk.END, display_text)
        self.preview_text_widget.config(state=tk.DISABLED)

    def _apply_preview_empty_state(self):
        self.preview_image_label.configure(
            image="",
            text=self.ui_lang.get_label(
                "no_image_captured",
                "No image captured yet",
            ),
        )
        self.preview_image_label.image = None
        self.preview_image_canvas.configure(height=100)
        self.preview_image_label.update_idletasks()
        label_width = self.preview_image_label.winfo_reqwidth()
        label_height = self.preview_image_label.winfo_reqheight()
        self.preview_image_canvas.itemconfig(
            self.preview_image_canvas_item,
            width=label_width,
            height=label_height,
        )
        self.preview_image_canvas.configure(
            scrollregion=(0, 0, label_width, label_height)
        )
        self._apply_preview_text(
            self.ui_lang.get_label(
                "no_image_for_ocr",
                "No image available for OCR",
            )
        )

    def _run_preview_ocr_job(self, screenshot_pil, paddleocr_settings, keep_linebreaks):
        """Background-only Preview OCR. Must not touch Tk widgets."""
        processed_pil = prepare_paddleocr_image(screenshot_pil, paddleocr_settings)
        ocr_cleaned_text, _lines = recognize_with_paddleocr(
            screenshot_pil,
            paddleocr_settings,
            keep_linebreaks=keep_linebreaks,
        )
        return processed_pil, ocr_cleaned_text

    def _schedule_preview_ocr(self, screenshot_pil, paddleocr_settings, keep_linebreaks):
        """Keep at most one in-flight Preview OCR and one latest pending frame."""
        self._ensure_preview_ocr_runtime()
        job = {
            "screenshot": screenshot_pil,
            "settings": paddleocr_settings,
            "keep_linebreaks": bool(keep_linebreaks),
            "generation": int(self._preview_ocr_generation or 0),
        }
        with self._preview_ocr_lock:
            if self._preview_ocr_in_flight:
                # Replace any older pending frame; do not queue a backlog.
                self._preview_ocr_pending_frame = job
                return False
            self._preview_ocr_in_flight = True
            active_job = job

        try:
            future = self._preview_ocr_executor.submit(
                self._run_preview_ocr_job,
                active_job["screenshot"],
                active_job["settings"],
                active_job["keep_linebreaks"],
            )
        except Exception as error:
            with self._preview_ocr_lock:
                self._preview_ocr_in_flight = False
            _log_debug(
                "Preview OCR submit failed: "
                f"{type(error).__name__} - {error}"
            )
            return False

        def _on_done(done_future, generation=active_job["generation"]):
            try:
                result = done_future.result()
                error_text = None
            except Exception as error:
                result = None
                error_text = f"{type(error).__name__}: {error}"
            root = getattr(self, "root", None)
            if root is None:
                self._discard_preview_ocr_completion_from_worker(
                    "root is unavailable"
                )
                return
            try:
                root.after(
                    0,
                    lambda: self._finish_preview_ocr_job(
                        generation,
                        result,
                        error_text,
                    ),
                )
            except Exception:
                self._discard_preview_ocr_completion_from_worker(
                    "Tk callback scheduling failed"
                )

        future.add_done_callback(_on_done)
        return True

    def _finish_preview_ocr_job(self, generation, result, error_text=None):
        """UI-thread completion handler for Preview OCR futures."""
        self._ensure_preview_ocr_runtime()
        with self._preview_ocr_lock:
            current_generation = int(self._preview_ocr_generation or 0)
            pending = self._preview_ocr_pending_frame
            self._preview_ocr_pending_frame = None
            self._preview_ocr_in_flight = False

        if int(generation or 0) != current_generation:
            # Stale completion after close/settings change; ignore UI writes.
            if pending and int(pending.get("generation") or 0) == current_generation:
                self._schedule_preview_ocr(
                    pending["screenshot"],
                    pending["settings"],
                    pending["keep_linebreaks"],
                )
            return

        if not self._preview_window_is_open():
            return

        try:
            if error_text:
                self._apply_preview_text(f"Error: {error_text}")
            elif result is not None:
                processed_pil, ocr_cleaned_text = result
                if processed_pil is not None:
                    self._apply_preview_image(processed_pil)
                if ocr_cleaned_text:
                    self._preview_last_ocr_text = ocr_cleaned_text
                self._apply_preview_text(
                    ocr_cleaned_text
                    if ocr_cleaned_text
                    else self.ui_lang.get_label(
                        "no_text_recognized",
                        "No text recognized",
                    )
                )
        except tk.TclError:
            self.ocr_preview_window = None
            return
        except Exception as error:
            _log_debug(
                "Preview OCR UI apply failed: "
                f"{type(error).__name__} - {error}"
            )

        if pending and int(pending.get("generation") or 0) == current_generation:
            self._schedule_preview_ocr(
                pending["screenshot"],
                pending["settings"],
                pending["keep_linebreaks"],
            )

    def show_ocr_preview(self):
        """Show/create the OCR Preview window."""
        self._ensure_preview_ocr_runtime()
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
        self._bump_preview_ocr_generation("preview opened")

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
        # Invalidate any in-flight Preview OCR before tearing down widgets.
        self._bump_preview_ocr_generation("preview closed")
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
                    # Lightweight UI/schedule only; never run full OCR here.
                    self.refresh_ocr_preview()
                    # Schedule next update
                    self._preview_realtime_timer = self.root.after(500, self.preview_realtime_update)
                else:
                    self.ocr_preview_window = None
            except tk.TclError:
                self.ocr_preview_window = None

    def refresh_ocr_preview(self):
        """Refresh Preview image immediately and schedule at most one OCR job."""
        # Check if window still exists
        if not self._preview_window_is_open():
            return

        try:
            screenshot_pil = self._capture_preview_screenshot()
            if screenshot_pil is None:
                self._apply_preview_empty_state()
                return

            current_ocr_model = self.get_ocr_model_setting()
            if current_ocr_model == PADDLEOCR_MODEL_CODE:
                from worker_threads import get_paddleocr_settings_from_app

                paddleocr_settings = get_paddleocr_settings_from_app(self)
                try:
                    keep_linebreaks = bool(self.keep_linebreaks_var.get())
                except Exception:
                    keep_linebreaks = False

                # Show the latest captured frame immediately on the UI thread.
                # Full PaddleOCR recognition always runs off-thread.
                try:
                    self._apply_preview_image(screenshot_pil.convert("RGB"))
                except Exception:
                    pass
                self._apply_preview_text(self._preview_placeholder_text())
                self._schedule_preview_ocr(
                    screenshot_pil,
                    paddleocr_settings,
                    keep_linebreaks,
                )
            else:
                processed_pil = screenshot_pil.convert("RGB")
                self._apply_preview_image(processed_pil)
                self._apply_preview_text(
                    self.ui_lang.get_label(
                        "ocr_preview_local_only",
                        "OCR preview is available for PaddleOCR.",
                    )
                )
        except tk.TclError:
            self.ocr_preview_window = None
        except Exception as e:
            _log_debug(f"Error refreshing OCR preview: {e}")
            # Show error in preview
            if hasattr(self, 'preview_text_widget'):
                try:
                    if self.preview_text_widget.winfo_exists():
                        self._apply_preview_text(f"Error: {str(e)}")
                except tk.TclError:
                    pass

    def load_initial_overlay_areas(self):
        load_areas_from_config_om(self)

    def select_source_area(self):
        select_source_area_om(self)
        self.publish_capture_ui_snapshot(
            bump_generation=True,
            reason="source area selected",
        )
        self.save_settings()

    def select_target_area(self):
        select_target_area_om(self)
        self.save_settings()

    def create_source_overlay(self):
        create_source_overlay_om(self)
        self.publish_capture_ui_snapshot(
            bump_generation=True,
            reason="source overlay created",
        )

    def create_target_overlay(self):
        create_target_overlay_om(self)  # System recreation, preserve position

    def toggle_source_visibility(self):
        toggle_source_visibility_om(self)
        self.save_settings()

    def toggle_target_visibility(self):
        toggle_target_visibility_om(self)
        self.save_settings()
