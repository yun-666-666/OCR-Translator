# worker_threads.py (Complete, Corrected File)

import tkinter as tk # For tk.Toplevel type check in capture_thread
from tkinter import messagebox # For Tesseract error in OCR thread
import time
import queue
import numpy as np
import cv2
import pytesseract
import pyautogui
import hashlib
import random
import re
import traceback 
from datetime import datetime

from logger import log_debug
from ocr_utils import (
    preprocess_for_ocr, get_tesseract_model_params, 
    ocr_region_with_confidence, post_process_ocr_text_general, 
    remove_text_after_last_punctuation_mark, capture_screen_region,
    build_capture_signature, build_ocr_frame_cache_key, recognize_windows_ocr
)
from translation_utils import post_process_translation_text
from PIL import Image # For hashing in capture_thread

def run_capture_thread(app):
    log_debug("WT: Capture thread started.")
    last_cap_time = 0.0
    last_cap_signature = None
    last_capture_geometry_signature = None
    min_interval = 0.05  # 50ms minimum safety floor - user can control via Settings tab
    similar_frames = 0
    current_scan_interval_sec = min_interval # Initialize

    while app.is_running:
        now = time.monotonic()
        try:
            # Update adaptive scan interval based on OCR load
            app.update_adaptive_scan_interval()
            
            # Use dynamic interval instead of static setting
            scan_interval_ms = app.current_scan_interval  # ← Use adaptive value
            base_scan_interval = max(min_interval, scan_interval_ms / 1000.0)
            
            # DEBUG: Log when using adaptive interval (every 20 seconds to avoid spam)
            if not hasattr(app, '_last_adaptive_debug') or now - app._last_adaptive_debug > 20.0:
                app._last_adaptive_debug = now
                log_debug(f"ADAPTIVE: Capture thread using scan interval: {scan_interval_ms}ms (base: {app.scan_interval_var.get()}ms)")
            
            ocr_model = app.get_ocr_model_setting()
            # Use a simpler, more adaptive logic for all API-based OCR models
            if app.is_api_based_ocr_model(ocr_model):
                # For API-based OCR: Simple, strict interval - no complex adaptive logic
                if now - last_cap_time < base_scan_interval:
                    sleep_duration = base_scan_interval - (now - last_cap_time)
                    slept_time = 0
                    while slept_time < sleep_duration and app.is_running:
                        chunk = min(0.05, sleep_duration - slept_time)
                        time.sleep(chunk)
                        slept_time += chunk
                    if not app.is_running: break
                    continue
                current_scan_interval_sec = base_scan_interval
            else:
                # Adaptive logic for Tesseract OCR (existing behavior)
                q_fullness = app.ocr_queue.qsize() / (app.ocr_queue.maxsize or 1)
                if q_fullness > 0.7: current_scan_interval_sec = base_scan_interval * (1 + q_fullness)
                elif q_fullness > 0.4: current_scan_interval_sec = base_scan_interval * 1.25
                else: current_scan_interval_sec = max(min_interval, current_scan_interval_sec * 0.95)
                
                if now - last_cap_time < current_scan_interval_sec:
                    sleep_duration = current_scan_interval_sec - (now - last_cap_time)
                    slept_time = 0
                    while slept_time < sleep_duration and app.is_running:
                        chunk = min(0.05, sleep_duration - slept_time)
                        time.sleep(chunk)
                        slept_time += chunk
                    if not app.is_running: break
                    continue
            
            overlay = app.source_overlay
            if not overlay or not isinstance(overlay, tk.Toplevel) or not overlay.winfo_exists():
                if app.is_running: time.sleep(max(current_scan_interval_sec, 0.5))
                continue
            
            try:
                area = overlay.get_geometry()
            except tk.TclError:
                if app.is_running: time.sleep(max(current_scan_interval_sec, 0.5))
                continue
            if not area:
                if app.is_running: time.sleep(max(current_scan_interval_sec, 0.2))
                continue

            x1, y1, x2, y2 = map(int, area); width, height = x2-x1, y2-y1
            if width <=0 or height <=0: continue

            capture_moment = time.monotonic()
            capture_backend_var = getattr(app, 'capture_backend_var', None)
            capture_backend = capture_backend_var.get() if capture_backend_var is not None else 'auto'
            geometry_signature = (x1, y1, width, height, capture_backend, ocr_model)
            if geometry_signature != last_capture_geometry_signature:
                log_debug(f"CAPTURE: source context changed to {geometry_signature}; clearing stale OCR state")
                last_capture_geometry_signature = geometry_signature
                last_cap_signature = None
                similar_frames = 0
                app.last_processed_subtitle = None
                app.previous_text = ""
                app.text_stability_counter = 0
                if hasattr(app, 'ocr_frame_cache'):
                    app.ocr_frame_cache.clear()
                try:
                    while True:
                        app.ocr_queue.get_nowait()
                except queue.Empty:
                    pass

            screenshot = capture_screen_region((x1, y1, width, height), backend=capture_backend)
            capture_duration = time.monotonic() - capture_moment
            last_cap_time = capture_moment
            log_debug(f"LATENCY: capture backend={capture_backend} region={width}x{height} took {capture_duration:.3f}s")

            img_small = screenshot.resize((max(1, width//4), max(1, height//4)), Image.Resampling.NEAREST if hasattr(Image, "Resampling") else Image.NEAREST)
            img_hash = hashlib.md5(img_small.tobytes()).hexdigest()
            capture_signature = build_capture_signature(img_hash, (x1, y1, width, height), capture_backend)
            try:
                screenshot._gct_frame_hash = img_hash
                screenshot._gct_region_origin = (x1, y1)
                screenshot._gct_capture_signature = capture_signature
                screenshot._gct_capture_monotonic = capture_moment
                screenshot._gct_capture_duration = capture_duration
            except Exception:
                pass

            if app.is_api_based_ocr_model(ocr_model):
                if capture_signature == last_cap_signature:
                    continue
                last_cap_signature = capture_signature
            else: # Tesseract-specific deduplication
                if capture_signature == last_cap_signature:
                    similar_frames +=1
                    skip_probability = min(0.95, 0.5 + (similar_frames*0.05))
                    if random.random() < skip_probability:
                        time.sleep(min(0.1, current_scan_interval_sec*0.5))
                        continue
                else:
                    similar_frames = 0
                last_cap_signature = capture_signature

            try:
                if not app.ocr_queue.full():
                    app.ocr_queue.put_nowait(screenshot)
            except queue.Full:
                pass # Skip frame if queue is full
            except Exception as q_err_wt_put:
                log_debug(f"WT: Capture: Error putting to OCR queue - {type(q_err_wt_put).__name__}: {q_err_wt_put}")

        except tk.TclError:
            log_debug("WT: Capture thread TclError (UI likely gone).")
            if not app.is_running: break
            time.sleep(0.1) 
        except Exception as loop_err_wt_capture:
            log_debug(f"WT: Capture thread error: {type(loop_err_wt_capture).__name__} - {loop_err_wt_capture}\n{traceback.format_exc()}")
            sleep_after_error = current_scan_interval_sec if 'current_scan_interval_sec' in locals() else 0.5
            time.sleep(max(sleep_after_error, 0.5))
    log_debug("WT: Capture thread finished.")


def run_ocr_thread(app):
    log_debug("WT: OCR thread started.")
    
    if app.get_ocr_model_setting() == 'tesseract':
        tess_langs = app.get_tesseract_lang_code()
        log_debug(f"WT: OCR using Tesseract with language: {tess_langs}")
    else:
        tess_langs = None
        log_debug(f"WT: OCR using {app.get_ocr_model_setting()}, skipping Tesseract language initialization")
    
    last_lang_check = time.monotonic()
    last_ocr_proc_time = 0
    min_ocr_interval = 0.1
    similar_texts_count = 0
    prev_ocr_text = ""
    current_conf_thresh = app.confidence_threshold
    
    cached_prep_mode = None
    cached_tess_params = None

    while app.is_running:
        now = time.monotonic()
        try:
            if now - last_lang_check > 5.0:
                if app.get_ocr_model_setting() == 'tesseract':
                    new_langs = app.get_tesseract_lang_code()
                    if new_langs != tess_langs:
                        tess_langs = new_langs
                        log_debug(f"WT: OCR lang changed to {tess_langs}")
                
                new_conf = app.confidence_var.get() 
                if new_conf != current_conf_thresh: 
                    current_conf_thresh = new_conf
                    app.confidence_threshold = new_conf
                    log_debug(f"WT: Confidence threshold updated to {new_conf}")
                last_lang_check = now
            
            ocr_model = app.get_ocr_model_setting()
            
            # No artificial delay for API-based OCR
            if not app.is_api_based_ocr_model(ocr_model):
                q_sz = app.ocr_queue.qsize()
                ocr_q_max = app.ocr_queue.maxsize or 1
                adaptive_ocr_interval = min_ocr_interval * (0.8 if q_sz <=1 else (1.0 + (q_sz / ocr_q_max)))

                if now - last_ocr_proc_time < adaptive_ocr_interval:
                    sleep_duration = adaptive_ocr_interval - (now - last_ocr_proc_time)
                    slept_time = 0
                    while slept_time < sleep_duration and app.is_running:
                        chunk = min(0.05, sleep_duration - slept_time)
                        time.sleep(chunk)
                        slept_time += chunk
                    if not app.is_running: break
                    continue
            
            try:
                screenshot_pil = app.ocr_queue.get(timeout=0.5)
            except queue.Empty:
                time.sleep(0.05)
                continue

            ocr_proc_start_time = time.monotonic()
            last_ocr_proc_time = ocr_proc_start_time
            app.last_screenshot = screenshot_pil
            processed_cv_img = None

            frame_hash = getattr(screenshot_pil, '_gct_frame_hash', None)
            if not frame_hash:
                cache_img = screenshot_pil.resize(
                    (max(1, screenshot_pil.width//4), max(1, screenshot_pil.height//4)),
                    Image.Resampling.NEAREST if hasattr(Image, "Resampling") else Image.NEAREST
                )
                frame_hash = hashlib.md5(cache_img.tobytes()).hexdigest()

            prep_mode = app.preprocessing_mode_var.get()
            block_size = app.adaptive_block_size_var.get()
            c_value = app.adaptive_c_var.get()
            region_origin = getattr(screenshot_pil, '_gct_region_origin', (0, 0))
            ocr_cache_key = None
            if hasattr(app, 'ocr_frame_cache') and not app.is_api_based_ocr_model(ocr_model):
                cache_lang = tess_langs or (app.get_tesseract_lang_code() if ocr_model == 'windows_ocr' else getattr(app, 'custom_source_lang', 'auto'))
                ocr_cache_key = build_ocr_frame_cache_key(
                    frame_hash,
                    ocr_model,
                    cache_lang,
                    prep_mode,
                    screenshot_pil.size,
                    region_origin=region_origin,
                )
                cached_ocr_text = app.ocr_frame_cache.get(ocr_cache_key)
                if cached_ocr_text is not None:
                    ocr_cleaned_text = cached_ocr_text
                    log_debug(f"LATENCY: OCR cache hit for {ocr_model} took {time.monotonic() - ocr_proc_start_time:.3f}s")
                    if app.ocr_debugging_var.get() and app.last_processed_image is not None:
                        app.root.after(0, app.update_debug_display, screenshot_pil, app.last_processed_image, ocr_cleaned_text)
                    # Jump to shared post-OCR routing below.
                    goto_post_ocr = True
                else:
                    goto_post_ocr = False
            else:
                goto_post_ocr = False

            if goto_post_ocr:
                pass
            # ==================== OCR MODEL ROUTING ====================
            elif app.is_api_based_ocr_model(ocr_model):
                run_api_ocr(app, screenshot_pil)
                continue # Skip to the next loop iteration
            
            elif ocr_model == 'windows_ocr':
                log_debug("WT: OCR routing to Windows OCR")
                try:
                    windows_lang = tess_langs or app.get_tesseract_lang_code()
                    ocr_raw_text = recognize_windows_ocr(screenshot_pil, windows_lang)
                    ocr_cleaned_text = post_process_ocr_text_general(ocr_raw_text, windows_lang)
                    if app.keep_linebreaks_var.get():
                        ocr_cleaned_text = ocr_cleaned_text.replace('\n', '<br>')
                    else:
                        ocr_cleaned_text = ocr_cleaned_text.replace('\n', ' ')
                    if ocr_cache_key is not None:
                        app.ocr_frame_cache.put(ocr_cache_key, ocr_cleaned_text)
                    log_debug(f"LATENCY: Windows OCR took {time.monotonic() - ocr_proc_start_time:.3f}s")
                except Exception as e:
                    log_debug(f"WT: Windows OCR failed, falling back to Tesseract: {type(e).__name__} - {e}")
                    ocr_model = 'tesseract'
                    goto_post_ocr = False
                else:
                    goto_post_ocr = True

            elif ocr_model == 'tesseract':
                log_debug("WT: OCR routing to Tesseract OCR")
                pass
            
            else:
                log_debug(f"WT: OCR: Unknown OCR model '{ocr_model}', falling back to Tesseract")
                pass

            if not goto_post_ocr:
                # ==================== TESSERACT OCR PROCESSING ====================
                img_np = np.array(screenshot_pil)
                img_shape = img_np.shape
                
                if len(img_shape) == 3:
                    if img_shape[2] == 3:
                        img_cv_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                    elif img_shape[2] == 4:
                        img_cv_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
                    else:
                        raise ValueError(f"WT: OCR: Unexpected 3D image channels: {img_shape[2]}")
                elif len(img_shape) == 2:
                    img_cv_bgr = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
                else:
                    raise ValueError(f"WT: OCR: Unexpected image dimensions: {len(img_shape)}D")

                processed_cv_img = preprocess_for_ocr(img_cv_bgr, prep_mode, block_size, c_value)
                app.last_processed_image = processed_cv_img

                if app.ocr_debugging_var.get():
                    app.root.after(0, app.update_debug_display, screenshot_pil, processed_cv_img, "Processing...")

                if cached_prep_mode != prep_mode:
                    cached_prep_mode = prep_mode
                    cached_tess_params = get_tesseract_model_params(prep_mode if prep_mode in ['gaming','document','subtitle'] else 'general')
                    log_debug(f"WT: OCR parameters cached for mode: {prep_mode}")
                
                full_img_region = (0,0, processed_cv_img.shape[1], processed_cv_img.shape[0])
                ocr_raw_text = ocr_region_with_confidence(processed_cv_img, full_img_region, tess_langs, cached_tess_params, current_conf_thresh)
                
                ocr_cleaned_text = post_process_ocr_text_general(ocr_raw_text, tess_langs)
                
                # Apply conditional linebreak conversion for Tesseract
                if app.keep_linebreaks_var.get():
                    ocr_cleaned_text = ocr_cleaned_text.replace('\n', '<br>')
                else:
                    ocr_cleaned_text = ocr_cleaned_text.replace('\n', ' ')

                if ocr_cache_key is not None:
                    app.ocr_frame_cache.put(ocr_cache_key, ocr_cleaned_text)
                log_debug(f"LATENCY: Tesseract OCR took {time.monotonic() - ocr_proc_start_time:.3f}s")
            
            if app.remove_trailing_garbage_var.get() and ocr_cleaned_text:
                pattern = r'[.!?]|\.{3}|…' 
                if not list(re.finditer(pattern, ocr_cleaned_text)):
                    app.text_stability_counter = 0
                    app.previous_text = ""
                    continue 
                ocr_cleaned_text = remove_text_after_last_punctuation_mark(ocr_cleaned_text)
            
            if app.ocr_debugging_var.get() and processed_cv_img is not None: 
                app.root.after(0, app.update_debug_display, screenshot_pil, processed_cv_img, ocr_cleaned_text)

            if not ocr_cleaned_text or app.is_placeholder_text(ocr_cleaned_text):
                app.text_stability_counter = 0
                app.previous_text = ""
                continue

            similarity = app.calculate_text_similarity(ocr_cleaned_text, prev_ocr_text)
            if similarity > 0.9:
                similar_texts_count+=1
            else:
                similar_texts_count = 0
                prev_ocr_text = ocr_cleaned_text
            
            if similar_texts_count > 2 and (now - app.last_successful_translation_time) < 1.0:
                continue

            if ocr_cleaned_text == app.previous_text:
                app.text_stability_counter +=1
            else:
                app.text_stability_counter = 0
                app.previous_text = ocr_cleaned_text
            
            if app.text_stability_counter >= app.stable_threshold:
                s_count = len(re.findall(r'[.!?]+', ocr_cleaned_text)) + 1
                txt_len = len(ocr_cleaned_text)
                adaptive_trans_interval = max(0.2, min(app.min_translation_interval, app.min_translation_interval * (0.5 + (0.1*s_count) + (txt_len/1000))))
                
                if (now - app.last_successful_translation_time >= adaptive_trans_interval):
                    start_async_translation(app, ocr_cleaned_text, 0)
                    app.last_successful_translation_time = now
                    app.text_stability_counter = 0
                    if ocr_cleaned_text != app.previous_text:
                        app.previous_text = ocr_cleaned_text
                    similar_texts_count = 0
        
        except pytesseract.TesseractNotFoundError:
            log_debug(f"WT: OCR Error: Tesseract not found. Path: {pytesseract.pytesseract.tesseract_cmd}")
            app.root.after(0, lambda: messagebox.showerror("Tesseract Error", f"Tesseract executable not found during OCR:\n{pytesseract.pytesseract.tesseract_cmd}\nPlease check path and restart.", parent=app.root))
            app.root.after(0, app.stop_translation_from_thread)
            break 
        except tk.TclError:
            log_debug("WT: OCR thread TclError.")
            if not app.is_running: break
            time.sleep(0.1)
        except Exception as e_ocr_loop_wt:
            log_debug(f"WT: OCR thread error: {type(e_ocr_loop_wt).__name__} - {e_ocr_loop_wt}\n{traceback.format_exc()}")
            app.text_stability_counter=0
            app.previous_text=""
            time.sleep(0.2)
    log_debug("WT: OCR thread finished.")

def run_translation_thread(app):
    """Simplified translation thread - mainly handles Tesseract timeout logic."""
    log_debug("WT: Translation thread started (simplified for async processing).")
    thread_local_last_translation_display_time = time.monotonic() 

    while app.is_running:
        now = time.monotonic()
        try:
            ocr_model = app.get_ocr_model_setting()
            
            if not app.is_api_based_ocr_model(ocr_model):
                inactive_duration = now - thread_local_last_translation_display_time
                if app.clear_translation_timeout > 0 and inactive_duration > app.clear_translation_timeout:
                    if not app.previous_text or app.previous_text == "":
                        app.update_translation_text("")
                        log_debug(f"WT: Cleared translation after {inactive_duration:.1f}s of inactivity with no source text (timeout: {app.clear_translation_timeout}s)")
                    else:
                        log_debug(f"WT: Not clearing translation despite {inactive_duration:.1f}s inactivity because source area still has text")
                    thread_local_last_translation_display_time = now

            if app.last_successful_translation_time > thread_local_last_translation_display_time:
                thread_local_last_translation_display_time = app.last_successful_translation_time

            try:
                text_to_translate = app.translation_queue.get(timeout=0.1)
                if text_to_translate and not app.is_placeholder_text(text_to_translate):
                    log_debug(f"WT: Processing legacy queue item: '{text_to_translate}'")
                    start_async_translation(app, text_to_translate, 0)
            except queue.Empty:
                pass
            
            time.sleep(0.1)

        except tk.TclError:
            log_debug("WT: Translation thread TclError.")
            if not app.is_running: break
            time.sleep(0.1)
        except Exception as e_trans_loop_wt:
            log_debug(f"WT: Translation thread error: {type(e_trans_loop_wt).__name__} - {e_trans_loop_wt}\n{traceback.format_exc()}")
            time.sleep(0.2)
    log_debug("WT: Translation thread finished.")


# ==================== GENERIC ASYNC API OCR WORKFLOW ====================

def run_api_ocr(app, screenshot_pil):
    """Start API-based OCR processing for a screenshot using the currently selected provider."""
    try:
        provider_name = app.get_ocr_model_setting()
        
        if not hasattr(app, 'batch_sequence_counter'):
            app.batch_sequence_counter = 0
        
        webp_image_data = app.convert_to_webp_for_api(screenshot_pil)
        if not webp_image_data:
            log_debug(f"Failed to convert image to WebP for {provider_name} OCR")
            return
        
        active_translation_model = app.translation_model_var.get()
        if app.is_gemini_model(active_translation_model):
            source_lang = getattr(app, 'gemini_source_lang', 'en')
        elif app.is_openai_model(active_translation_model):
            source_lang = getattr(app, 'openai_source_lang', 'en')
        else:
            source_lang = app.source_lang_var.get()

        app.batch_sequence_counter += 1
        sequence_number = app.batch_sequence_counter
        
        if len(app.active_ocr_calls) >= app.max_concurrent_ocr_calls:
            log_debug(f"Max concurrent OCR calls ({app.max_concurrent_ocr_calls}) reached, skipping {provider_name} batch {sequence_number}")
            return
        
        app.active_ocr_calls.add(sequence_number)
        app.ocr_thread_pool.submit(
            process_api_ocr_async,
            app, webp_image_data, source_lang, sequence_number, provider_name
        )
        log_debug(f"Started {provider_name} OCR batch {sequence_number} (active calls: {len(app.active_ocr_calls)})")
        
    except Exception as e:
        log_debug(f"Error starting API OCR batch: {type(e).__name__} - {e}")

def process_api_ocr_async(app, webp_image_data, source_lang, sequence_number, provider_name):
    """Process an API OCR call asynchronously. This is the generic worker function."""
    try:
        log_debug(f"Processing {provider_name} OCR batch {sequence_number}")
        
        ocr_result = app.translation_handler.perform_ocr(webp_image_data, source_lang)
        
        log_debug(f"{provider_name} OCR batch {sequence_number} completed: '{ocr_result}', scheduling response")
        app.root.after(0, process_api_ocr_response, app, ocr_result, sequence_number, source_lang, provider_name)
        
    except Exception as e:
        log_debug(f"Error in async {provider_name} OCR batch {sequence_number}: {type(e).__name__} - {e}")
        error_msg = f"<e>: OCR batch {sequence_number} error: {str(e)}"
        app.root.after(0, process_api_ocr_response, app, error_msg, sequence_number, source_lang, provider_name)
    
    finally:
        app.active_ocr_calls.discard(sequence_number)
        log_debug(f"{provider_name} OCR batch {sequence_number} finished (active calls: {len(app.active_ocr_calls)})")

def process_api_ocr_response(app, ocr_result, sequence_number, source_lang, provider_name):
    """Process any API OCR response with chronological order enforcement. This is the generic callback."""
    try:
        log_debug(f"Processing {provider_name} OCR response for batch {sequence_number}: '{ocr_result}'")
        
        if not hasattr(app, 'last_displayed_batch_sequence'):
            app.last_displayed_batch_sequence = 0
        
        if sequence_number <= app.last_displayed_batch_sequence:
            log_debug(f"{provider_name} OCR batch {sequence_number}: Sequence too old, discarding")
            return
            
        log_debug(f"{provider_name} OCR batch {sequence_number}: Processing newer sequence")
        
        if isinstance(ocr_result, str) and ocr_result.startswith("<e>:"):
            log_debug(f"OCR error in {provider_name} batch {sequence_number}: {ocr_result}")
            visible_error = ocr_result[len("<e>:"):].strip() or ocr_result
            app.update_translation_text(f"OCR Error:\n{visible_error}")
            app.last_displayed_batch_sequence = sequence_number
            return
        
        if ocr_result == "<EMPTY>":
            app.handle_empty_ocr_result()
            app.last_displayed_batch_sequence = sequence_number
            return
        
        if hasattr(app, 'last_processed_subtitle') and ocr_result == app.last_processed_subtitle:
            app.reset_clear_timeout()
            log_debug(f"Keeping existing translation for successive identical {provider_name} OCR: '{ocr_result}'")
            app.last_displayed_batch_sequence = sequence_number
            return
        
        app.last_processed_subtitle = ocr_result
        app.reset_clear_timeout()
        start_async_translation(app, ocr_result, sequence_number)
        app.last_displayed_batch_sequence = sequence_number
        
    except Exception as e:
        log_debug(f"Error processing {provider_name} OCR response for batch {sequence_number}: {type(e).__name__} - {e}")


# ==================== ASYNC TRANSLATION PROCESSING (Phase 2) ====================

def start_async_translation(app, text_to_translate, ocr_sequence_number):
    """Start async translation processing to eliminate queue bottlenecks."""
    try:
        app.initialize_async_translation_infrastructure()
        
        app.translation_sequence_counter += 1
        translation_sequence = app.translation_sequence_counter
        app.latest_translation_sequence_started = translation_sequence

        latency_mode_var = getattr(app, 'custom_ai_latency_mode_var', None)
        try:
            latency_mode = latency_mode_var.get() if latency_mode_var is not None else "safe"
        except Exception:
            latency_mode = "safe"
        dedupe_inflight = latency_mode != "none"

        enable_instant_var = getattr(app, 'enable_instant_cache_display_var', None)
        instant_cache_enabled = enable_instant_var.get() if enable_instant_var is not None else True
        if instant_cache_enabled and hasattr(app, 'translation_handler'):
            cached_translation = app.translation_handler.get_cached_translation_for_display(text_to_translate)
            if cached_translation:
                final_processed_translation = post_process_translation_text(cached_translation)
                display_schedule_start = time.monotonic()
                app.update_translation_text(final_processed_translation)
                log_debug(f"LATENCY: display scheduling took {time.monotonic() - display_schedule_start:.3f}s")
                app.last_displayed_translation_sequence = translation_sequence
                app.last_successful_translation_time = time.monotonic()
                log_debug(f"LATENCY: instant translation cache display for sequence {translation_sequence}: '{final_processed_translation}'")
                return

        if not hasattr(app, 'active_translation_inflight_keys'):
            app.active_translation_inflight_keys = set()

        inflight_key = None
        if hasattr(app, 'translation_handler') and hasattr(app.translation_handler, 'get_inflight_translation_key'):
            try:
                inflight_key = app.translation_handler.get_inflight_translation_key(text_to_translate)
            except Exception as key_error:
                log_debug(f"LATENCY: failed to build translation inflight key: {type(key_error).__name__} - {key_error}")
        if inflight_key is None:
            inflight_key = ("raw", text_to_translate)

        if dedupe_inflight and inflight_key in app.active_translation_inflight_keys:
            log_debug(f"LATENCY: duplicate in-flight translation skipped for sequence {translation_sequence}: '{text_to_translate}'")
            return
        
        if len(app.active_translation_calls) >= app.max_concurrent_translation_calls:
            log_debug(f"Max concurrent translation calls ({app.max_concurrent_translation_calls}) reached, skipping translation {translation_sequence}")
            return
        
        app.active_translation_calls.add(translation_sequence)
        if dedupe_inflight:
            app.active_translation_inflight_keys.add(inflight_key)
        
        try:
            future = app.translation_thread_pool.submit(
                process_translation_async,
                app, text_to_translate, translation_sequence, ocr_sequence_number, inflight_key
            )
        except Exception:
            app.active_translation_calls.discard(translation_sequence)
            if dedupe_inflight:
                app.active_translation_inflight_keys.discard(inflight_key)
            raise
        
        log_debug(f"Started async translation {translation_sequence} for OCR batch {ocr_sequence_number} (active calls: {len(app.active_translation_calls)}): '{text_to_translate}'")
        
    except Exception as e:
        log_debug(f"Error starting async translation: {type(e).__name__} - {e}")


def process_translation_async(app, text_to_translate, translation_sequence, ocr_sequence_number, inflight_key=None):
    """Process translation API call asynchronously with timeout and staleness handling."""
    start_time = time.monotonic()
    
    try:
        log_debug(f"Processing async translation {translation_sequence}")

        stream_callback = None
        latency_mode_var = getattr(app, 'custom_ai_latency_mode_var', None)
        try:
            latency_mode = latency_mode_var.get() if latency_mode_var is not None else ""
        except Exception:
            latency_mode = ""

        if latency_mode == "stream":
            def stream_callback(partial_text):
                def display_partial(text=partial_text):
                    try:
                        if not getattr(app, 'is_running', True):
                            return
                        latest_started = getattr(app, 'latest_translation_sequence_started', translation_sequence)
                        if translation_sequence < latest_started:
                            return
                        last_displayed = getattr(app, 'last_displayed_translation_sequence', 0)
                        if translation_sequence < last_displayed:
                            return
                        if isinstance(text, str) and text.strip():
                            app.update_translation_text(post_process_translation_text(text))
                            app.last_successful_translation_time = time.monotonic()
                    except Exception as stream_error:
                        log_debug(f"Streaming translation display failed: {type(stream_error).__name__} - {stream_error}")

                app.root.after(0, display_partial)
        
        translation_result = app.translation_handler.translate_text_with_timeout(
            text_to_translate,
            timeout_seconds=10.0,
            ocr_batch_number=ocr_sequence_number,
            stream_callback=stream_callback,
            translation_sequence=translation_sequence,
        )
        
        elapsed_time = time.monotonic() - start_time
        if elapsed_time > 5.0:
            log_debug(f"Translation {translation_sequence} took {elapsed_time:.1f}s, may be stale but will attempt display")
        
        log_debug(f"Translation {translation_sequence} completed in {elapsed_time:.3f}s: '{translation_result}'")
        
        app.root.after(0, process_translation_response, app, translation_result, translation_sequence, text_to_translate, ocr_sequence_number)
        
    except Exception as e:
        elapsed_time = time.monotonic() - start_time
        log_debug(f"Error in async translation {translation_sequence} after {elapsed_time:.2f}s: {type(e).__name__} - {e}")
        
        error_msg = f"Translation error: {str(e)}"
        app.root.after(0, process_translation_response, app, error_msg, translation_sequence, text_to_translate, ocr_sequence_number)
    
    finally:
        try:
            app.active_translation_calls.discard(translation_sequence)
            if inflight_key is not None and hasattr(app, 'active_translation_inflight_keys'):
                app.active_translation_inflight_keys.discard(inflight_key)
            log_debug(f"Translation {translation_sequence} finished (active calls: {len(app.active_translation_calls)})")
        except Exception as cleanup_error:
            log_debug(f"Error cleaning up translation {translation_sequence}: {cleanup_error}")


def process_translation_response(app, translation_result, translation_sequence, original_text, ocr_sequence_number):
    """Process translation response with chronological order enforcement - same logic as OCR."""
    try:
        log_debug(f"Processing translation response for sequence {translation_sequence}: '{translation_result}'")
        
        if translation_result is None:
            log_debug(f"Translation {translation_sequence}: Timeout occurred, no message displayed (suppressed)")
            return
        
        if not hasattr(app, 'last_displayed_translation_sequence'):
            app.last_displayed_translation_sequence = 0
        
        if translation_sequence <= app.last_displayed_translation_sequence:
            log_debug(f"Translation {translation_sequence}: Sequence too old (last displayed: {app.last_displayed_translation_sequence}), discarding but caching result")
            return
        
        log_debug(f"Translation {translation_sequence}: Processing newer sequence (last displayed: {app.last_displayed_translation_sequence})")
        
        error_prefixes = ("Err:", "MarianMT error:", "Google API error:", "DeepL API error:", 
                          "No translation for model:", "MarianMT not initialized.", 
                          "MarianMT language pair not determined:", "Google API key missing:",
                          "DeepL API key missing:", "Google Client init error:", 
                          "DeepL Client init error:", "Translation error:", 
                          "Google Translate API client not initialized", 
                          "DeepL API client not initialized", 
                          "MarianMT translator not initialized")
        
        if isinstance(translation_result, str) and any(translation_result.startswith(p) for p in error_prefixes):
            log_debug(f"Translation error in sequence {translation_sequence}: {translation_result}")
            app.update_translation_text(f"Translation Error:\n{translation_result}")
            app.last_displayed_translation_sequence = translation_sequence
            app.last_successful_translation_time = time.monotonic()
            return
        
        if isinstance(translation_result, str) and translation_result.strip():
            final_processed_translation = post_process_translation_text(translation_result)
            display_schedule_start = time.monotonic()
            app.update_translation_text(final_processed_translation)
            log_debug(f"LATENCY: display scheduling took {time.monotonic() - display_schedule_start:.3f}s")
            log_debug(f"Translation {translation_sequence} displayed: '{final_processed_translation}' (from OCR batch {ocr_sequence_number})")
            app.last_displayed_translation_sequence = translation_sequence
            app.last_successful_translation_time = time.monotonic()
        else:
            log_debug(f"Translation {translation_sequence}: Empty or invalid result, not displaying")
            
    except Exception as e:
        log_debug(f"Error processing translation response for sequence {translation_sequence}: {type(e).__name__} - {e}")
        try:
            app.update_translation_text(f"Translation Processing Error:\n{type(e).__name__}")
        except:
            pass
