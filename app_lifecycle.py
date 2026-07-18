"""Translation lifecycle, queue reset, and shutdown responsibilities."""

import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import messagebox

from config_manager import save_app_config
from overlay_manager import ensure_overlays_ready_om


def _log_debug(message):
    facade = sys.modules.get("app_logic")
    if facade is not None:
        return facade.log_debug(message)


class AppLifecycleMixin:
    def initialize_async_translation_infrastructure(self):
        """Initialize async translation infrastructure if not already present."""
        if not hasattr(self, 'translation_sequence_counter'):
            self.translation_sequence_counter = 0
            _log_debug("Initialized translation_sequence_counter")

        if not hasattr(self, 'last_displayed_translation_sequence'):
            self.last_displayed_translation_sequence = 0
            _log_debug("Initialized last_displayed_translation_sequence")

        if not hasattr(self, 'active_translation_calls'):
            self.active_translation_calls = set()
            _log_debug("Initialized active_translation_calls")

        if not hasattr(self, 'active_translation_inflight_keys'):
            self.active_translation_inflight_keys = set()
            _log_debug("Initialized active_translation_inflight_keys")

        if not hasattr(self, 'active_translation_started_monotonic'):
            self.active_translation_started_monotonic = {}
            _log_debug("Initialized active_translation_started_monotonic")

        if not hasattr(self, 'max_concurrent_translation_calls'):
            self.max_concurrent_translation_calls = 6
            _log_debug("Initialized max_concurrent_translation_calls")

        if not hasattr(self, 'translation_supersede_after_seconds'):
            self.translation_supersede_after_seconds = 1.5
            _log_debug("Initialized translation_supersede_after_seconds")

        if not hasattr(self, 'last_translation_submit_monotonic'):
            self.last_translation_submit_monotonic = 0.0
            _log_debug("Initialized last_translation_submit_monotonic")

        if not hasattr(self, 'pending_translation_request'):
            self.pending_translation_request = None
            _log_debug("Initialized pending_translation_request")

        if not hasattr(self, 'pending_translation_flush_scheduled'):
            self.pending_translation_flush_scheduled = False
            _log_debug("Initialized pending_translation_flush_scheduled")

        if not hasattr(self, 'pending_translation_flush_deadline_monotonic'):
            self.pending_translation_flush_deadline_monotonic = 0.0
            _log_debug("Initialized pending_translation_flush_deadline_monotonic")

        if not hasattr(self, 'pending_translation_flush_generation'):
            self.pending_translation_flush_generation = 0
            _log_debug("Initialized pending_translation_flush_generation")

    def check_clear_timeout(self):
        """Check if clear timeout should be triggered and return True if timeout exceeded."""
        if self.clear_timeout_timer_start is None:
            return False

        if self.clear_translation_timeout_var.get() <= 0:
            return False  # Timeout disabled

        current_time = time.monotonic()
        elapsed = current_time - self.clear_timeout_timer_start
        timeout_seconds = self.clear_translation_timeout_var.get()

        return elapsed >= timeout_seconds

    def translate_text(self, text_content):
        return self.translation_handler.translate_text(text_content)

    def is_placeholder_text(self, text_content):
        return self.translation_handler.is_placeholder_text(text_content)

    def calculate_text_similarity(self, text1, text2):
        return self.translation_handler.calculate_text_similarity(text1, text2)

    def clear_file_caches(self):
        return self.clear_cache()

    def clear_cache(self):
        """Clear unified translation cache - FIXED VERSION (No pause/resume needed)."""
        try:
            _log_debug("Clearing unified translation cache...")

            # Clear unified cache (thread-safe, no need to pause translation)
            self.translation_handler.clear_cache()

            # Clear in-memory file cache representations (Level 2 persistence remains)
            _log_debug("Cleared in-memory representations of file caches.")

            if hasattr(self, 'ocr_frame_cache'):
                self.ocr_frame_cache.clear()
                _log_debug("Cleared OCR frame cache.")

            # Clear queues
            self._clear_queue(self.ocr_queue)
            self._clear_queue(self.translation_queue)
            _log_debug("Cleared OCR and translation queues.")

            # Reset text processing state
            self.text_stability_counter = 0
            self.previous_text = ""
            self.last_processed_subtitle = None
            self.last_local_ocr_submitted_text = None
            self.last_local_ocr_submitted_norm = None
            self.last_local_ocr_submitted_scope = None
            self.clear_ocr_stability_gate("cache cleared")
            if hasattr(self, 'active_translation_inflight_keys'):
                self.active_translation_inflight_keys.clear()
            _log_debug("Unified translation cache and related states cleared successfully.")

            # Update status briefly
            original_status_text = self.status_label.cget("text")
            self.status_label.config(text="Status: Cache cleared")
            if self.root.winfo_exists():
                self.root.after(2000, lambda: self.status_label.config(text=original_status_text) if self.status_label.winfo_exists() else None)

        except Exception as e_cc:
            _log_debug(f"Error clearing unified cache: {e_cc}")
            if self.root.winfo_exists():
                messagebox.showerror("Error", f"Failed to clear cache: {e_cc}", parent=self.root)
            original_status_text = self.status_label.cget("text")
            self.status_label.config(text="Status: Cache clearing failed")
            self.root.after(2000, lambda: self.status_label.config(text=original_status_text) if self.status_label.winfo_exists() else None)

    def _clear_queue(self, q_to_clear):
        items_cleared_count = 0
        while not q_to_clear.empty():
            try:
                q_to_clear.get_nowait()
                items_cleared_count += 1
            except queue.Empty:
                break
            except Exception as e_cq:
                _log_debug(f"Error clearing queue {type(q_to_clear).__name__}: {e_cq}")
                break
        if items_cleared_count > 0:
            _log_debug(f"Cleared {items_cleared_count} items from {type(q_to_clear).__name__}.")

    def _reset_translation_scheduler_session_state(self, reason):
        """Invalidate async translation scheduler state at a session boundary."""
        from worker_threads import reset_translation_scheduler_session_state

        reset_translation_scheduler_session_state(self, reason)

    def _reset_gemini_batch_state(self):
        """Reset Gemini OCR batch management state for clean start."""
        self.batch_sequence_counter = 0
        self.last_displayed_batch_sequence = 0
        self.active_ocr_calls = set()
        self.last_processed_subtitle = None
        self.last_local_ocr_submitted_text = None
        self.last_local_ocr_submitted_norm = None
        self.last_local_ocr_submitted_scope = None
        self.clear_ocr_stability_gate("OCR batch state reset")
        self.clear_timeout_timer_start = None
        _log_debug("Gemini OCR batch state reset")

    def _root_window_alive(self):
        try:
            if not self.root:
                return False
            if not hasattr(self.root, 'winfo_exists'):
                return True
            return bool(self.root.winfo_exists())
        except Exception:
            return False

    def _stop_translation_for_app_exit(self):
        """Stop worker activity for application exit without scheduling UI callbacks."""
        if not self.is_running:
            _log_debug("Process was not running at close time.")
            return

        _log_debug("Stopping running OCR/translation process for app exit...")
        self.is_running = False
        self.toggle_in_progress = False

        active_threads_copy = list(getattr(self, 'threads', []) or [])
        try:
            self.threads.clear()
        except Exception:
            self.threads = []

        for thread_obj in active_threads_copy:
            try:
                if thread_obj.is_alive():
                    thread_obj.join(timeout=0.5)
            except Exception as join_error:
                _log_debug(f"Error joining thread during app exit: {join_error}")

        if hasattr(self, 'active_ocr_calls'):
            self.active_ocr_calls.clear()
        if hasattr(self, 'active_translation_calls'):
            self.active_translation_calls.clear()

        handler = getattr(self, 'translation_handler', None)
        if handler is not None:
            for method_name in ('request_end_ocr_session', 'request_end_translation_session'):
                method = getattr(handler, method_name, None)
                if callable(method):
                    try:
                        method()
                    except Exception as session_error:
                        _log_debug(f"Error ending session during app exit: {session_error}")

    def _graceful_shutdown_poll(self):
        """
        Non-blocking poll to check if all async API calls have finished.
        This allows the tkinter event loop to process callbacks that decrement pending call counters.
        """
        if getattr(self, '_app_is_closing', False) or getattr(self, '_shutdown_finalized', False):
            _log_debug("Graceful shutdown poll ignored because application shutdown is already finalizing.")
            return
        if not self._root_window_alive():
            _log_debug("Graceful shutdown poll ignored because root window no longer exists.")
            return

        # Calculate pending calls from all providers
        pending_ocr = 0
        translation_handler = getattr(self, 'translation_handler', None)
        if hasattr(translation_handler, 'ocr_providers'):
            for provider in translation_handler.ocr_providers.values():
                pending_ocr += provider._pending_ocr_calls
        if hasattr(self, 'active_ocr_calls'):
            pending_ocr += len(self.active_ocr_calls)

        pending_translation = 0
        if hasattr(translation_handler, 'providers'):
            for provider in translation_handler.providers.values():
                pending_translation += provider._pending_translation_calls
        if hasattr(self, 'active_translation_calls'):
            pending_translation += len(self.active_translation_calls)

        # Check if timeout is reached or all calls are done
        now = time.monotonic()
        elapsed = now - self._shutdown_start_time
        if (pending_ocr == 0 and pending_translation == 0) or elapsed > 20.0:
            if elapsed > 20.0:
                _log_debug(f"Warning: Shutdown timeout of 20.0s reached. Some API calls may not have completed.")
            else:
                _log_debug("All pending API calls have completed.")

            self._shutdown_wait_log_next_at = 0.0
            _log_debug(f"Graceful shutdown for thread pools completed in {elapsed:.2f}s.")
            self._finalize_shutdown() # Proceed to the final steps
            return

        # If not done, poll again shortly
        next_wait_log_at = float(
            getattr(self, "_shutdown_wait_log_next_at", 0.0) or 0.0
        )
        if now >= next_wait_log_at:
            _log_debug(
                "Waiting for pending API calls to complete... "
                f"OCR: {pending_ocr}, Translation: {pending_translation}"
            )
            self._shutdown_wait_log_next_at = now + 1.0
        if self._root_window_alive() and not getattr(self, '_app_is_closing', False):
            self.root.after(100, self._graceful_shutdown_poll)

    def _finalize_shutdown(self):
        """Contains the final steps of the shutdown process after graceful polling."""
        if getattr(self, '_shutdown_finalized', False):
            _log_debug("Finalize shutdown ignored because it already ran.")
            return
        self._shutdown_finalized = True

        # End the sessions HERE, after all pending calls are confirmed to be finished.
        if hasattr(self, 'translation_handler'):
            self.translation_handler.request_end_ocr_session()
            self.translation_handler.request_end_translation_session()

        self._clear_queue(self.ocr_queue)
        self._clear_queue(self.translation_queue)
        self._reset_translation_scheduler_session_state("translation stopped")
        self.clear_ocr_stability_gate("translation stopped")

        if self.translation_text and self.translation_text.winfo_exists():
            try:
                self.translation_text.config(state=tk.NORMAL)
                self.translation_text.delete(1.0, tk.END)
                self.translation_text.config(state=tk.DISABLED)
            except tk.TclError as e_ctt:
                _log_debug(f"Error clearing translation text on stop: {e_ctt}")

        if self.source_overlay and self.source_overlay.winfo_exists() and self.source_overlay.winfo_viewable():
            try: self.source_overlay.hide()
            except tk.TclError: _log_debug("Error hiding source overlay on stop (likely closed).")

        if self.target_overlay and self.target_overlay.winfo_exists() and self.target_overlay.winfo_viewable():
            try: self.target_overlay.hide()
            except tk.TclError: _log_debug("Error hiding target overlay on stop (likely closed).")

        self.start_stop_btn.config(state=tk.NORMAL)
        status_text_stopped = "Status: " + self.ui_lang.get_label("status_stopped", "Stopped (Press ~ to Start)")
        self.status_label.config(text=status_text_stopped)
        _log_debug("Translation process stopped.")

        self.toggle_in_progress = False # Release the lock here

    def _ensure_overlays_for_start(self):
        """Repair missing overlays from saved geometry before Start validation."""
        if ensure_overlays_ready_om(self, force_hidden=True):
            return True
        messagebox.showerror(
            "Start Error",
            "Could not initialize the saved source and target areas. Select the areas again.",
            parent=self.root,
        )
        return False

    def toggle_translation(self):
        # Add re-entrancy lock
        if self.toggle_in_progress:
            _log_debug("Toggle translation already in progress, ignoring call.")
            return

        self.toggle_in_progress = True

        if self.is_running:
            _log_debug("Stopping translation process requested by user.")
            self.is_running = False
            self._shutdown_finalized = False

            # DO NOT request session ends here. This will be done in _finalize_shutdown.
            # Context clearing is now handled automatically after session end logging in the translation handler

            self.start_stop_btn.config(text="Start", state=tk.DISABLED)
            self.status_label.config(text="Status: Stopping...")
            self.root.update_idletasks()

            active_threads_copy = self.threads[:]
            self.threads.clear()

            thread_stop_start_time = time.monotonic()
            _log_debug(f"Waiting for main worker threads to join: {[t.name for t in active_threads_copy if t.is_alive()]}")

            for thread_obj in active_threads_copy:
                if thread_obj.is_alive():
                    try:
                        thread_obj.join(timeout=1.0) # Short timeout for main threads
                    except Exception as join_err_tt:
                        _log_debug(f"Error joining thread {thread_obj.name}: {join_err_tt}")

            _log_debug(f"Main worker threads joined in {time.monotonic() - thread_stop_start_time:.2f}s.")

            # Use non-blocking poll for graceful shutdown
            _log_debug("Starting graceful shutdown poll for API call thread pools...")
            self._shutdown_start_time = time.monotonic()
            self.root.after(0, self._graceful_shutdown_poll)
            # The rest of the shutdown logic is now in _finalize_shutdown()
            # The lock will be released in _finalize_shutdown()

        else:
            try:
                _log_debug("Starting translation process requested by user...")
                self.start_stop_btn.config(state=tk.DISABLED)
                self.status_label.config(text="Status: Initializing...")
                self.root.update_idletasks()

                valid_start_flag = self._ensure_overlays_for_start()

                if valid_start_flag and (not self.source_overlay or not self._widget_exists_safely(self.source_overlay)):
                    messagebox.showerror("Start Error", "Source area overlay missing. Select source area.", parent=self.root)
                    valid_start_flag = False
                if valid_start_flag and (not self.target_overlay or not self._widget_exists_safely(self.target_overlay)):
                    messagebox.showerror("Start Error", "Target area overlay missing. Select target area.", parent=self.root)
                    valid_start_flag = False
                if valid_start_flag and (not self.translation_text or not self._widget_exists_safely(self.translation_text)):
                    messagebox.showerror("Start Error", "Target text display widget missing. Reselect target area.", parent=self.root)
                    valid_start_flag = False

                if valid_start_flag and not self.custom_ai_profiles.get_active_profile("translation"):
                    messagebox.showerror(
                        self.ui_lang.get_label("start_error_title", "Start Error"),
                        self.ui_lang.get_label("start_error_no_translation_profile", "Add and select an AI model profile before starting."),
                        parent=self.root
                    )
                    valid_start_flag = False

                if valid_start_flag and self.get_ocr_model_setting() == 'custom_ai' and not self.custom_ai_profiles.get_active_profile("ocr"):
                    messagebox.showerror(
                        self.ui_lang.get_label("start_error_title", "Start Error"),
                        self.ui_lang.get_label("start_error_no_ocr_profile", "Add and select an AI model profile for OCR before starting, or choose PaddleOCR."),
                        parent=self.root
                    )
                    valid_start_flag = False

                if valid_start_flag:
                     try:
                         self.source_area = self.source_overlay.get_geometry()
                         self.target_area = self.target_overlay.get_geometry()
                         if not self._validate_area_coords(self.source_area, "source"): valid_start_flag = False
                         if valid_start_flag and not self._validate_area_coords(self.target_area, "target"): valid_start_flag = False
                     except (tk.TclError, AttributeError) as e_gog:
                         messagebox.showerror("Start Error", f"Could not get overlay geometry: {e_gog}", parent=self.root)
                         valid_start_flag = False

                if not valid_start_flag:
                    self.start_stop_btn.config(state=tk.NORMAL)
                    status_text_failed = "Status: Start Failed"
                    if self.KEYBOARD_AVAILABLE: status_text_failed += " (Press ~ to Retry)"
                    self.status_label.config(text=status_text_failed)
                    _log_debug("Start aborted due to failed pre-start validation checks.")
                    return

                _log_debug("Pre-start checks passed. Preparing to start threads...")
                self.text_stability_counter = 0
                self.previous_text = ""
                self.last_image_hash = None
                self.last_screenshot = None
                self.last_processed_image = None

                self._reset_gemini_batch_state()

                try:
                    if self.target_overlay and self.target_overlay.winfo_exists() and not self.target_overlay.winfo_viewable():
                        self.target_overlay.show()
                except tk.TclError:
                    _log_debug("Warning: Error ensuring target overlay visibility at start (likely closed).")

                self._clear_queue(self.ocr_queue)
                self._clear_queue(self.translation_queue)
                self._reset_translation_scheduler_session_state(
                    "translation starting"
                )
                self.last_local_ocr_submitted_text = None
                self.last_local_ocr_submitted_norm = None
                self.last_local_ocr_submitted_scope = None
                self.clear_ocr_stability_gate("translation starting")

                self._app_is_closing = False
                self._shutdown_finalized = False
                self.is_running = True

                if hasattr(self, 'translation_handler'):
                    if self.is_api_based_ocr_model():
                        self.translation_handler.start_ocr_session()
                    self.translation_handler.start_translation_session()

                self.start_stop_btn.config(text="Stop", state=tk.NORMAL)
                status_text_running = "Status: " + self.ui_lang.get_label("status_running", "Running (Press ~ to Stop)")
                self.status_label.config(text=status_text_running)
                self.root.update_idletasks()

                from worker_threads import run_capture_thread, run_ocr_thread, run_translation_thread

                capture_thread_instance = threading.Thread(target=run_capture_thread, args=(self,), name="CaptureThread", daemon=True)
                ocr_thread_instance = threading.Thread(target=run_ocr_thread, args=(self,), name="OCRThread", daemon=True)
                translation_thread_instance = threading.Thread(target=run_translation_thread, args=(self,), name="TranslationThread", daemon=True)

                self.threads = [capture_thread_instance, ocr_thread_instance, translation_thread_instance]
                for t_obj in self.threads:
                    t_obj.start()
                _log_debug(f"Threads started: {[t.name for t in self.threads]}")

                # Release lock after successful start
                self.toggle_in_progress = False

            finally:
                # Release lock if start failed before threads were launched
                if not self.is_running:
                    self.toggle_in_progress = False

    def _validate_area_coords(self, area_coordinates, area_type_str):
        min_dimension = 10
        if not area_coordinates or len(area_coordinates) != 4:
            messagebox.showerror("Area Validation Error", f"Invalid {area_type_str} area data: {area_coordinates}.", parent=self.root)
            return False
        try:
            x1_val, y1_val, x2_val, y2_val = map(int, area_coordinates)
            width_val = x2_val - x1_val
            height_val = y2_val - y1_val
            if width_val < min_dimension or height_val < min_dimension:
                messagebox.showerror("Area Validation Error",
                                     f"{area_type_str.capitalize()} area too small ({width_val}x{height_val}). Min {min_dimension}x{min_dimension}.",
                                     parent=self.root)
                return False
            return True
        except (ValueError, TypeError) as e_vac:
            messagebox.showerror("Area Validation Error", f"Invalid coordinates in {area_type_str} area: {area_coordinates}. Error: {e_vac}", parent=self.root)
            return False

    def stop_translation_from_thread(self):
        if self.is_running:
            _log_debug("Requesting stop translation from worker thread.")
            if self.root.winfo_exists():
                self.root.after(0, self.toggle_translation)

    def on_closing(self):
        _log_debug("Main window close requested. Initiating shutdown...")
        self._app_is_closing = True
        if getattr(self, "runtime_metrics_refresh_after_id", None):
            try:
                self.root.after_cancel(self.runtime_metrics_refresh_after_id)
            except Exception:
                pass
            self.runtime_metrics_refresh_after_id = None

        # Close OCR Preview window if open
        if self.ocr_preview_window is not None:
            try:
                _log_debug("Closing OCR Preview window...")
                self.close_ocr_preview()
            except Exception as e:
                _log_debug(f"Error closing OCR Preview window: {e}")

        shutdown_preview_ocr = getattr(
            self,
            "shutdown_preview_ocr_executor",
            None,
        )
        if callable(shutdown_preview_ocr):
            try:
                shutdown_preview_ocr()
            except Exception as error:
                _log_debug(
                    "Error shutting down Preview OCR executor: "
                    f"{type(error).__name__} - {error}"
                )

        self._stop_translation_for_app_exit()

        # # Force end any remaining sessions when application closes
        # if hasattr(self, 'translation_handler'):
        #     try:
        #         self.translation_handler.force_end_sessions_on_app_close()
        #     except Exception as e:
        #         _log_debug(f"Error ending sessions on app close: {e}")

        # Shutdown OCR and translation thread pools
        if hasattr(self, 'ocr_thread_pool'):
            try:
                _log_debug("Shutting down OCR thread pool...")
                self.ocr_thread_pool.shutdown(wait=False, cancel_futures=True)
                _log_debug("OCR thread pool shutdown complete.")
            except Exception as e_otp:
                _log_debug(f"Error shutting down OCR thread pool: {e_otp}")

        if hasattr(self, 'translation_thread_pool'):
            try:
                _log_debug("Shutting down translation thread pool...")
                self.translation_thread_pool.shutdown(wait=False, cancel_futures=True)
                _log_debug("Translation thread pool shutdown complete.")
            except Exception as e_ttp:
                _log_debug(f"Error shutting down translation thread pool: {e_ttp}")
        if hasattr(self, 'translation_handler'):
            try:
                self.translation_handler.close()
            except Exception as e_close:
                _log_debug(f"Error closing translation handler: {e_close}")
        try:
            _log_debug("Saving final settings before closing...")
            if self._fully_initialized:
                # Save OCR Preview geometry if window is open
                if self.ocr_preview_window is not None:
                    self.save_preview_geometry()
                self.save_settings(force=True)
            else:
                # Save OCR Preview geometry even if not fully initialized
                if self.ocr_preview_window is not None:
                    self.save_preview_geometry()
                self.configuration_handler.save_current_window_geometry()
                save_app_config(self.config)
        except Exception as e_ssc:
            _log_debug(f"Error saving settings during closing: {e_ssc}")

        if self.KEYBOARD_AVAILABLE:
            try:
                import keyboard
                keyboard.unhook_all()
                _log_debug("Unhooked all keyboard shortcuts.")
            except Exception as e_uhk:
                _log_debug(f"Error unhooking keyboard shortcuts: {e_uhk}")

        _log_debug("Destroying overlay windows if they exist...")
        for overlay_attr_name in ['source_overlay', 'target_overlay']:
            overlay_widget = getattr(self, overlay_attr_name, None)
            if overlay_widget:
                try:
                    # Preserve target overlay position before destroying during shutdown
                    if overlay_attr_name == 'target_overlay':
                        from overlay_manager import _preserve_overlay_position
                        _preserve_overlay_position(self)
                        _log_debug("Preserved target overlay position during app shutdown")

                    # Handle tkinter overlays
                    if hasattr(overlay_widget, 'winfo_exists') and overlay_widget.winfo_exists():
                        overlay_widget.destroy()
                    # Handle PySide overlays
                    elif hasattr(overlay_widget, 'close'):
                        overlay_widget.close()

                except Exception as e_dow:
                    _log_debug(f"Error destroying {overlay_attr_name}: {e_dow}")
            setattr(self, overlay_attr_name, None)
        self.translation_text = None

        _log_debug("Destroying root window...")
        try:
            if self.root and hasattr(self.root, 'winfo_exists') and self.root.winfo_exists():
                self.root.destroy()
            _log_debug("Root window destroyed.")
        except Exception as e_drw:
             _log_debug(f"Error destroying root window: {e_drw}")
        _log_debug("Application shutdown sequence complete.")
