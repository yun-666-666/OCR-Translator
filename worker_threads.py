# worker_threads.py (Complete, Corrected File)

import tkinter as tk # For tk.Toplevel type check in capture_thread
from tkinter import messagebox # For Tesseract error in OCR thread
from difflib import SequenceMatcher
import time
import queue
import numpy as np
import cv2
import pyautogui
import hashlib
import random
import re
import threading
import traceback 
from datetime import datetime

from logger import log_debug
from ocr_utils import (
    preprocess_for_ocr,
    ocr_region_with_confidence, post_process_ocr_text_general, 
    remove_text_after_last_punctuation_mark, capture_screen_region,
    build_capture_signature, build_ocr_frame_cache_key,
    get_tesseract_ocr_config, resolve_tessdata_dir_from_tesseract_path,
    TesseractOcrUnavailableError, CaptureBackendSelector,
)
from translation_utils import (
    is_translation_error_result,
    post_process_translation_text,
)
from PIL import Image # For hashing in capture_thread


DEFAULT_TRANSLATION_SUPERSEDE_AFTER_SECONDS = 1.5
MAX_SUPERSEDED_TRANSLATION_CONCURRENCY = 2


def _runtime_metrics(app):
    return getattr(app, 'runtime_metrics', None)


def _record_metric_timing(app, name, seconds):
    metrics = _runtime_metrics(app)
    recorder = getattr(metrics, 'record_timing', None)
    if callable(recorder):
        try:
            recorder(name, seconds)
        except Exception:
            pass


def _increment_metric(app, name, amount=1):
    metrics = _runtime_metrics(app)
    incrementer = getattr(metrics, 'increment', None)
    if callable(incrementer):
        try:
            incrementer(name, amount)
        except Exception:
            pass


def _set_metric_gauge(app, name, value):
    metrics = _runtime_metrics(app)
    setter = getattr(metrics, 'set_gauge', None)
    if callable(setter):
        try:
            setter(name, value)
        except Exception:
            pass


def _safe_queue_size(queue_obj):
    if queue_obj is None:
        return 0
    try:
        return int(queue_obj.qsize())
    except Exception:
        return 0


def _get_capture_backend_selector(app):
    selector = getattr(app, 'capture_backend_selector', None)
    if selector is None:
        selector = CaptureBackendSelector()
        try:
            app.capture_backend_selector = selector
        except Exception:
            pass
    return selector


def _resolve_capture_backend(app, configured_backend, region):
    configured_backend = str(configured_backend or 'auto').strip().lower()
    if configured_backend not in ('auto', 'mss', 'pyautogui'):
        log_debug(f"CAPTURE_SELECTOR: unknown configured backend={configured_backend}; using auto")
        configured_backend = 'auto'
    if configured_backend != 'auto':
        return configured_backend
    selector = _get_capture_backend_selector(app)
    resolver = getattr(selector, 'resolve_backend', None)
    if callable(resolver):
        return resolver(configured_backend, region)
    return configured_backend


def _refresh_translation_metric_gauges(app, concurrency_limit=None, cooldown_remaining=None):
    _set_metric_gauge(
        app,
        "active_translation_calls",
        len(getattr(app, 'active_translation_calls', ()) or ()),
    )
    if concurrency_limit is not None:
        _set_metric_gauge(app, "translation_concurrency_limit", concurrency_limit)
    if cooldown_remaining is not None:
        _set_metric_gauge(app, "provider_cooldown_seconds", cooldown_remaining)


def _refresh_ocr_queue_metric(app):
    _set_metric_gauge(app, "ocr_queue_size", _safe_queue_size(getattr(app, 'ocr_queue', None)))


def _get_screenshot_frame_hash(screenshot_pil):
    frame_hash = getattr(screenshot_pil, '_gct_frame_hash', None)
    if frame_hash:
        return frame_hash

    cache_img = screenshot_pil.resize(
        (max(1, screenshot_pil.width//4), max(1, screenshot_pil.height//4)),
        Image.Resampling.NEAREST if hasattr(Image, "Resampling") else Image.NEAREST
    )
    return hashlib.md5(cache_img.tobytes()).hexdigest()


def _get_api_ocr_cache_model_key(app, provider_name):
    provider_key = str(provider_name or '').strip().lower()
    if provider_key != 'custom_ai' or not hasattr(app, 'custom_ai_profiles'):
        return provider_key

    try:
        profile = app.custom_ai_profiles.get_active_profile("ocr")
    except Exception as e:
        log_debug(f"Could not resolve active OCR profile for API OCR cache key: {type(e).__name__} - {e}")
        return provider_key

    if not profile:
        return f"{provider_key}|ocr_profile=<missing>"

    profile_id = str(profile.get("id") or "").strip()
    base_url = str(profile.get("base_url") or "").strip().rstrip("/")
    model = str(profile.get("model") or "").strip()
    return f"{provider_key}|ocr_profile={profile_id}|{base_url}|{model}"


def _normalize_local_ocr_submit_text(text_to_translate):
    if not isinstance(text_to_translate, str):
        return ""

    normalized = text_to_translate.replace("<br>", " ")
    normalized = normalized.translate(str.maketrans({
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "–": "-",
        "—": "-",
        "―": "-",
    }))
    normalized = normalized.lower()
    normalized = re.sub(r"\s+", " ", normalized).strip()
    normalized = re.sub(r"[.!?…]+$", "", normalized).strip()
    return normalized


def _get_local_ocr_submit_scope(app, text_to_translate):
    handler = getattr(app, "translation_handler", None)
    getter = getattr(handler, "get_inflight_translation_key", None)
    if not callable(getter):
        return None

    try:
        inflight_key = getter(text_to_translate)
    except Exception as scope_error:
        log_debug(
            "LATENCY: failed to build local OCR submit scope: "
            f"{type(scope_error).__name__} - {scope_error}"
        )
        return None

    if isinstance(inflight_key, (tuple, list)):
        if len(inflight_key) >= 3:
            return tuple(inflight_key[2:])
        return tuple(inflight_key)
    return inflight_key


def _looks_like_safe_local_ocr_near_repeat(current_norm, last_norm):
    if len(current_norm) < 8 or len(last_norm) < 8:
        return False

    if abs(len(current_norm) - len(last_norm)) > 2:
        return False

    if SequenceMatcher(None, current_norm, last_norm).ratio() < 0.97:
        return False

    current_tokens = re.findall(r"[a-z0-9']+", current_norm)
    last_tokens = re.findall(r"[a-z0-9']+", last_norm)
    if not current_tokens or not last_tokens:
        return False

    if len(current_tokens) != len(last_tokens):
        return False

    mismatch_count = 0
    for current_token, last_token in zip(current_tokens, last_tokens):
        if current_token == last_token:
            continue
        if current_token.isdigit() or last_token.isdigit():
            return False
        if SequenceMatcher(None, current_token, last_token).ratio() < 0.85:
            return False
        mismatch_count += 1
        if mismatch_count > 1:
            return False

    return mismatch_count > 0


def _should_skip_local_ocr_resubmit(app, text_to_translate):
    current_norm = _normalize_local_ocr_submit_text(text_to_translate)
    if not current_norm:
        return False

    current_scope = _get_local_ocr_submit_scope(app, text_to_translate)
    last_scope = getattr(app, "last_local_ocr_submitted_scope", None)
    if current_scope is not None and last_scope is not None and current_scope != last_scope:
        return False

    last_norm = _normalize_local_ocr_submit_text(
        getattr(app, "last_local_ocr_submitted_text", None)
    )
    if not last_norm:
        last_norm = _normalize_local_ocr_submit_text(
            getattr(app, "last_local_ocr_submitted_norm", None)
        )

    if not last_norm:
        return False

    if current_norm == last_norm:
        return True

    return _looks_like_safe_local_ocr_near_repeat(current_norm, last_norm)


def _remember_local_ocr_submit(app, text_to_translate):
    app.last_local_ocr_submitted_text = text_to_translate
    app.last_local_ocr_submitted_norm = _normalize_local_ocr_submit_text(text_to_translate)
    app.last_local_ocr_submitted_scope = _get_local_ocr_submit_scope(app, text_to_translate)


def _clear_local_ocr_submit_state(app):
    app.last_local_ocr_submitted_text = None
    app.last_local_ocr_submitted_norm = None
    app.last_local_ocr_submitted_scope = None


def _get_api_ocr_cache_mode_key(app):
    keep_linebreaks_var = getattr(app, 'keep_linebreaks_var', None)
    if keep_linebreaks_var is None:
        return 'api'

    try:
        keep_linebreaks = bool(keep_linebreaks_var.get())
    except Exception:
        keep_linebreaks = False
    return f"api|keep_linebreaks={keep_linebreaks}"


def _get_tesseract_ocr_cache_mode_key(app):
    preprocessing_mode = getattr(app, 'preprocessing_mode_var', None)
    adaptive_block_size = getattr(app, 'adaptive_block_size_var', None)
    adaptive_c = getattr(app, 'adaptive_c_var', None)
    confidence_threshold = getattr(app, 'confidence_threshold', None)
    keep_linebreaks_var = getattr(app, 'keep_linebreaks_var', None)
    remove_trailing_garbage_var = getattr(app, 'remove_trailing_garbage_var', None)

    try:
        preprocessing_mode_value = preprocessing_mode.get() if preprocessing_mode is not None else 'none'
    except Exception:
        preprocessing_mode_value = 'none'

    try:
        block_size_value = int(adaptive_block_size.get()) if adaptive_block_size is not None else 41
    except Exception:
        block_size_value = 41

    try:
        c_value = int(adaptive_c.get()) if adaptive_c is not None else -60
    except Exception:
        c_value = -60

    try:
        confidence_value = int(confidence_threshold)
    except Exception:
        confidence_value = 60

    try:
        keep_linebreaks = bool(keep_linebreaks_var.get()) if keep_linebreaks_var is not None else False
    except Exception:
        keep_linebreaks = False

    try:
        remove_trailing_garbage = bool(remove_trailing_garbage_var.get()) if remove_trailing_garbage_var is not None else False
    except Exception:
        remove_trailing_garbage = False

    return (
        f"tesseract|prep={str(preprocessing_mode_value).lower()}"
        f"|block={block_size_value}"
        f"|c={c_value}"
        f"|confidence={confidence_value}"
        f"|keep_linebreaks={keep_linebreaks}"
        f"|remove_trailing_garbage={remove_trailing_garbage}"
    )


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
            resolved_capture_backend = _resolve_capture_backend(app, capture_backend, (x1, y1, width, height))
            geometry_signature = (x1, y1, width, height, capture_backend, resolved_capture_backend, ocr_model)
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

            screenshot = capture_screen_region((x1, y1, width, height), backend=resolved_capture_backend)
            capture_duration = time.monotonic() - capture_moment
            last_cap_time = capture_moment
            actual_capture_backend = getattr(screenshot, '_gct_capture_backend', resolved_capture_backend)
            fallback_reason = getattr(screenshot, '_gct_capture_fallback_reason', None)
            if fallback_reason and actual_capture_backend != resolved_capture_backend:
                selector = getattr(app, 'capture_backend_selector', None)
                recorder = getattr(selector, 'record_backend_fallback', None)
                if callable(recorder):
                    recorder(
                        capture_backend,
                        resolved_capture_backend,
                        actual_capture_backend,
                        (x1, y1, width, height),
                        reason=fallback_reason,
                    )
            log_debug(
                f"LATENCY: capture backend={actual_capture_backend} "
                f"configured={capture_backend} resolved={resolved_capture_backend} "
                f"region={width}x{height} took {capture_duration:.3f}s"
            )
            _record_metric_timing(app, "capture_duration", capture_duration)
            _refresh_ocr_queue_metric(app)

            img_small = screenshot.resize((max(1, width//4), max(1, height//4)), Image.Resampling.NEAREST if hasattr(Image, "Resampling") else Image.NEAREST)
            img_hash = hashlib.md5(img_small.tobytes()).hexdigest()
            capture_signature = build_capture_signature(img_hash, (x1, y1, width, height), actual_capture_backend)
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
                    _refresh_ocr_queue_metric(app)
            except queue.Full:
                _refresh_ocr_queue_metric(app)
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
        tessdata_dir = resolve_tessdata_dir_from_tesseract_path(app.tesseract_path_var.get())
        log_debug(f"WT: OCR using Tesseract with language: {tess_langs}")
        log_debug(f"WT: OCR tessdata directory: {tessdata_dir}")
    else:
        tess_langs = None
        tessdata_dir = None
        log_debug(f"WT: OCR using {app.get_ocr_model_setting()}, skipping Tesseract language initialization")
    
    last_lang_check = time.monotonic()
    last_ocr_proc_time = 0
    min_ocr_interval = 0.1
    similar_texts_count = 0
    prev_ocr_text = ""
    current_conf_thresh = app.confidence_threshold
    
    cached_prep_mode = None

    while app.is_running:
        now = time.monotonic()
        try:
            if now - last_lang_check > 5.0:
                if app.get_ocr_model_setting() == 'tesseract':
                    new_langs = app.get_tesseract_lang_code()
                    new_tessdata_dir = resolve_tessdata_dir_from_tesseract_path(app.tesseract_path_var.get())
                    if new_langs != tess_langs:
                        tess_langs = new_langs
                        log_debug(f"WT: OCR lang changed to {tess_langs}")
                    if new_tessdata_dir != tessdata_dir:
                        tessdata_dir = new_tessdata_dir
                        log_debug(f"WT: OCR tessdata directory changed to {tessdata_dir}")
                last_lang_check = now
            
            new_conf = app.confidence_var.get() 
            if new_conf != current_conf_thresh: 
                current_conf_thresh = new_conf
                app.confidence_threshold = new_conf
                log_debug(f"WT: Confidence threshold updated to {new_conf}")
            
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
                _refresh_ocr_queue_metric(app)
            except queue.Empty:
                time.sleep(0.05)
                continue

            ocr_proc_start_time = time.monotonic()
            last_ocr_proc_time = ocr_proc_start_time
            app.last_screenshot = screenshot_pil
            processed_cv_img = None

            frame_hash = _get_screenshot_frame_hash(screenshot_pil)

            prep_mode = app.preprocessing_mode_var.get()
            block_size = app.adaptive_block_size_var.get()
            c_value = app.adaptive_c_var.get()
            region_origin = getattr(screenshot_pil, '_gct_region_origin', (0, 0))
            ocr_cache_key = None
            if hasattr(app, 'ocr_frame_cache') and not app.is_api_based_ocr_model(ocr_model):
                cache_lang = tess_langs or getattr(app, 'custom_source_lang', 'auto')
                ocr_cache_key = build_ocr_frame_cache_key(
                    frame_hash,
                    ocr_model,
                    cache_lang,
                    _get_tesseract_ocr_cache_mode_key(app),
                    screenshot_pil.size,
                    region_origin=region_origin,
                )
                cached_ocr_text = app.ocr_frame_cache.get(ocr_cache_key)
                if cached_ocr_text is not None:
                    ocr_cleaned_text = cached_ocr_text
                    ocr_duration = time.monotonic() - ocr_proc_start_time
                    log_debug(f"LATENCY: OCR cache hit for {ocr_model} took {ocr_duration:.3f}s")
                    _increment_metric(app, "ocr_frame_cache_hit")
                    _record_metric_timing(app, "ocr_duration", ocr_duration)
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
                    log_debug(f"WT: OCR parameters cached for mode: {prep_mode}")
                
                full_img_region = (0,0, processed_cv_img.shape[1], processed_cv_img.shape[0])
                ocr_raw_text = ocr_region_with_confidence(
                    processed_cv_img,
                    full_img_region,
                    tess_langs,
                    get_tesseract_ocr_config(prep_mode if prep_mode in ['gaming', 'document', 'subtitle'] else 'general'),
                    current_conf_thresh,
                    tessdata_dir=tessdata_dir,
                )
                
                ocr_cleaned_text = post_process_ocr_text_general(ocr_raw_text, tess_langs)
                
                # Apply conditional linebreak conversion for Tesseract
                if app.keep_linebreaks_var.get():
                    ocr_cleaned_text = ocr_cleaned_text.replace('\n', '<br>')
                else:
                    ocr_cleaned_text = ocr_cleaned_text.replace('\n', ' ')

                if ocr_cache_key is not None:
                    app.ocr_frame_cache.put(ocr_cache_key, ocr_cleaned_text)
                ocr_duration = time.monotonic() - ocr_proc_start_time
                log_debug(f"LATENCY: Tesseract OCR took {ocr_duration:.3f}s")
                _record_metric_timing(app, "ocr_duration", ocr_duration)
            
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

            if ocr_cleaned_text == app.previous_text:
                app.text_stability_counter +=1
            else:
                app.text_stability_counter = 0
                app.previous_text = ocr_cleaned_text
            
            if app.text_stability_counter >= app.stable_threshold:
                s_count = len(re.findall(r'[.!?]+', ocr_cleaned_text)) + 1
                txt_len = len(ocr_cleaned_text)
                adaptive_trans_interval = max(0.2, min(app.min_translation_interval, app.min_translation_interval * (0.5 + (0.1*s_count) + (txt_len/1000))))
                provider_gate_interval = _get_local_ocr_translation_gate_seconds(
                    app,
                    ocr_cleaned_text,
                    adaptive_trans_interval,
                )
                elapsed_since_submit = _get_local_ocr_elapsed_since_last_submit(app, now)

                if (
                    similar_texts_count > 2
                    and elapsed_since_submit < provider_gate_interval
                ):
                    continue

                if elapsed_since_submit >= provider_gate_interval:
                    if _should_skip_local_ocr_resubmit(app, ocr_cleaned_text):
                        app.reset_clear_timeout()
                        app.text_stability_counter = 0
                        similar_texts_count = 0
                        log_debug(
                            "LATENCY: skipped local OCR resubmit for near-duplicate stable text: "
                            f"'{ocr_cleaned_text}'"
                        )
                        continue

                    start_async_translation(app, ocr_cleaned_text, 0)
                    app.text_stability_counter = 0
                    if ocr_cleaned_text != app.previous_text:
                        app.previous_text = ocr_cleaned_text
                    similar_texts_count = 0
        
        except TesseractOcrUnavailableError as e_tess:
            log_debug(f"WT: OCR Error: {e_tess}")
            app.root.after(0, lambda: messagebox.showerror("Tesseract Error", f"{e_tess}\nPlease check the Tesseract installation and tessdata path, then restart.", parent=app.root))
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
        ocr_start_time = time.monotonic()
        provider_name = app.get_ocr_model_setting()
        
        if not hasattr(app, 'batch_sequence_counter'):
            app.batch_sequence_counter = 0
        
        if provider_name == 'custom_ai':
            source_lang = getattr(app, 'custom_source_lang', None) or app.source_lang_var.get()
        else:
            active_translation_model = app.translation_model_var.get()
            if app.is_gemini_model(active_translation_model):
                source_lang = getattr(app, 'gemini_source_lang', 'en')
            elif app.is_openai_model(active_translation_model):
                source_lang = getattr(app, 'openai_source_lang', 'en')
            else:
                source_lang = app.source_lang_var.get()

        ocr_cache_key = None
        if hasattr(app, 'ocr_frame_cache'):
            frame_hash = _get_screenshot_frame_hash(screenshot_pil)
            region_origin = getattr(screenshot_pil, '_gct_region_origin', (0, 0))
            cache_model_key = _get_api_ocr_cache_model_key(app, provider_name)
            ocr_cache_key = build_ocr_frame_cache_key(
                frame_hash,
                cache_model_key,
                source_lang,
                _get_api_ocr_cache_mode_key(app),
                screenshot_pil.size,
                region_origin=region_origin,
            )
            cached_ocr_text = app.ocr_frame_cache.get(ocr_cache_key)
            if cached_ocr_text is not None:
                app.batch_sequence_counter += 1
                sequence_number = app.batch_sequence_counter
                log_debug(f"LATENCY: API OCR cache hit for {provider_name} batch {sequence_number}")
                _increment_metric(app, "ocr_frame_cache_hit")
                _record_metric_timing(app, "ocr_duration", time.monotonic() - ocr_start_time)
                process_api_ocr_response(app, cached_ocr_text, sequence_number, source_lang, provider_name, ocr_cache_key=ocr_cache_key)
                return

        if len(app.active_ocr_calls) >= app.max_concurrent_ocr_calls:
            _set_metric_gauge(app, "active_ocr_calls", len(app.active_ocr_calls))
            log_debug(f"Max concurrent OCR calls ({app.max_concurrent_ocr_calls}) reached, skipping {provider_name} OCR before image conversion")
            return
        
        webp_image_data = app.convert_to_webp_for_api(screenshot_pil)
        if not webp_image_data:
            log_debug(f"Failed to convert image to WebP for {provider_name} OCR")
            return

        app.batch_sequence_counter += 1
        sequence_number = app.batch_sequence_counter
        
        app.active_ocr_calls.add(sequence_number)
        _set_metric_gauge(app, "active_ocr_calls", len(app.active_ocr_calls))
        try:
            app.ocr_thread_pool.submit(
                process_api_ocr_async,
                app, webp_image_data, source_lang, sequence_number, provider_name, ocr_cache_key
            )
        except Exception:
            app.active_ocr_calls.discard(sequence_number)
            raise
        log_debug(f"Started {provider_name} OCR batch {sequence_number} (active calls: {len(app.active_ocr_calls)})")
        
    except Exception as e:
        log_debug(f"Error starting API OCR batch: {type(e).__name__} - {e}")

def process_api_ocr_async(app, webp_image_data, source_lang, sequence_number, provider_name, ocr_cache_key=None):
    """Process an API OCR call asynchronously. This is the generic worker function."""
    try:
        latest_started_sequence = getattr(app, 'batch_sequence_counter', sequence_number)
        if sequence_number < latest_started_sequence:
            log_debug(
                f"{provider_name} OCR batch {sequence_number} is stale (latest started: {latest_started_sequence}); "
                "skipping provider call"
            )
            return

        log_debug(f"Processing {provider_name} OCR batch {sequence_number}")
        
        ocr_start_time = time.monotonic()
        ocr_result = app.translation_handler.perform_ocr(webp_image_data, source_lang)
        _record_metric_timing(app, "ocr_duration", time.monotonic() - ocr_start_time)
        
        log_debug(f"{provider_name} OCR batch {sequence_number} completed: '{ocr_result}', scheduling response")
        app.root.after(0, process_api_ocr_response, app, ocr_result, sequence_number, source_lang, provider_name, ocr_cache_key)
        
    except Exception as e:
        log_debug(f"Error in async {provider_name} OCR batch {sequence_number}: {type(e).__name__} - {e}")
        error_msg = f"<e>: OCR batch {sequence_number} error: {str(e)}"
        app.root.after(0, process_api_ocr_response, app, error_msg, sequence_number, source_lang, provider_name, ocr_cache_key)
    
    finally:
        app.active_ocr_calls.discard(sequence_number)
        _set_metric_gauge(app, "active_ocr_calls", len(app.active_ocr_calls))
        log_debug(f"{provider_name} OCR batch {sequence_number} finished (active calls: {len(app.active_ocr_calls)})")

def process_api_ocr_response(app, ocr_result, sequence_number, source_lang, provider_name, ocr_cache_key=None):
    """Process any API OCR response with chronological order enforcement. This is the generic callback."""
    try:
        log_debug(f"Processing {provider_name} OCR response for batch {sequence_number}: '{ocr_result}'")
        
        if not hasattr(app, 'last_displayed_batch_sequence'):
            app.last_displayed_batch_sequence = 0
        
        if sequence_number <= app.last_displayed_batch_sequence:
            log_debug(f"{provider_name} OCR batch {sequence_number}: Sequence too old, discarding")
            return
            
        log_debug(f"{provider_name} OCR batch {sequence_number}: Processing newer sequence")

        if (
            ocr_cache_key is not None
            and hasattr(app, 'ocr_frame_cache')
            and isinstance(ocr_result, str)
            and ocr_result.strip()
            and not ocr_result.startswith("<e>:")
            and ocr_result != "<EMPTY>"
        ):
            app.ocr_frame_cache.put(ocr_cache_key, ocr_result)
        
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

def _get_translation_submit_interval_seconds(app, text_to_translate):
    handler = getattr(app, 'translation_handler', None)
    getter = getattr(handler, 'get_translation_submit_interval_seconds', None)
    if callable(getter):
        try:
            return max(0.0, float(getter(text_to_translate)))
        except Exception as interval_error:
            log_debug(
                "LATENCY: failed to read translation submit interval: "
                f"{type(interval_error).__name__} - {interval_error}"
            )
    return max(0.0, float(getattr(app, 'min_translation_interval', 0.3) or 0.3))


def _get_translation_provider_cooldown_seconds(app):
    handler = getattr(app, 'translation_handler', None)
    getter = getattr(handler, 'get_translation_provider_cooldown_seconds', None)
    if callable(getter):
        try:
            return max(0.0, float(getter()))
        except Exception as cooldown_error:
            log_debug(
                "LATENCY: failed to read translation provider cooldown: "
                f"{type(cooldown_error).__name__} - {cooldown_error}"
            )
    return 0.0


def _get_local_ocr_translation_gate_seconds(app, text_to_translate, adaptive_interval_seconds):
    submit_interval = _get_translation_submit_interval_seconds(app, text_to_translate)
    return max(
        0.0,
        float(adaptive_interval_seconds or 0.0),
        float(submit_interval or 0.0),
    )


def _get_local_ocr_elapsed_since_last_submit(app, now):
    if hasattr(app, 'last_translation_submit_monotonic'):
        last_submit = float(getattr(app, 'last_translation_submit_monotonic', 0.0) or 0.0)
        if last_submit > 0.0:
            return max(0.0, float(now) - last_submit)

    last_success = float(getattr(app, 'last_successful_translation_time', 0.0) or 0.0)
    return max(0.0, float(now) - last_success)


def _get_translation_concurrency_limit(app):
    handler = getattr(app, 'translation_handler', None)
    getter = getattr(handler, 'get_translation_concurrency_limit', None)
    if callable(getter):
        try:
            return max(1, int(getter()))
        except Exception as limit_error:
            log_debug(
                "LATENCY: failed to read translation concurrency limit: "
                f"{type(limit_error).__name__} - {limit_error}"
            )
    return max(1, int(getattr(app, 'max_concurrent_translation_calls', 1) or 1))


def _get_translation_supersede_after_seconds(app):
    try:
        configured = float(
            getattr(
                app,
                'translation_supersede_after_seconds',
                DEFAULT_TRANSLATION_SUPERSEDE_AFTER_SECONDS,
            )
        )
    except (TypeError, ValueError):
        configured = DEFAULT_TRANSLATION_SUPERSEDE_AFTER_SECONDS
    return max(0.25, min(10.0, configured))


def _get_translation_latency_mode(app):
    latency_mode_var = getattr(app, 'custom_ai_latency_mode_var', None)
    try:
        return str(
            latency_mode_var.get() if latency_mode_var is not None else ""
        ).strip().lower()
    except Exception:
        return ""


def _get_oldest_active_translation_age(app, now):
    started_by_sequence = getattr(
        app,
        'active_translation_started_monotonic',
        None,
    )
    if not isinstance(started_by_sequence, dict):
        return None

    active_sequences = tuple(getattr(app, 'active_translation_calls', ()))
    started_values = []
    for sequence in active_sequences:
        if sequence not in started_by_sequence:
            continue
        try:
            started_values.append(float(started_by_sequence[sequence]))
        except (TypeError, ValueError):
            continue
    if not started_values:
        return None
    return max(0.0, float(now) - min(started_values))


def _invalidate_pending_translation_request(app, reason):
    had_pending_state = bool(
        getattr(app, 'pending_translation_request', None)
        or getattr(app, 'pending_translation_flush_scheduled', False)
    )
    generation = int(
        getattr(app, 'pending_translation_flush_generation', 0) or 0
    ) + 1
    app.pending_translation_flush_generation = generation
    app.pending_translation_request = None
    app.pending_translation_flush_scheduled = False
    app.pending_translation_flush_deadline_monotonic = 0.0
    if had_pending_state:
        log_debug(
            "LATENCY: invalidated pending translation request "
            f"generation={generation} reason={reason}"
        )
    return generation


def _flush_pending_translation_request(app, flush_generation=None):
    try:
        current_generation = int(
            getattr(app, 'pending_translation_flush_generation', 0) or 0
        )
        if flush_generation is not None and flush_generation != current_generation:
            log_debug(
                "LATENCY: ignored stale pending translation timer "
                f"generation={flush_generation} current={current_generation}"
            )
            return

        app.pending_translation_flush_scheduled = False
        app.pending_translation_flush_deadline_monotonic = 0.0
        pending_request = getattr(app, 'pending_translation_request', None)
        if not pending_request:
            return
        app.pending_translation_request = None

        if hasattr(app, 'is_running') and not app.is_running:
            log_debug("LATENCY: dropped pending translation because the app is stopped")
            return

        start_async_translation(
            app,
            pending_request["text"],
            pending_request["ocr_sequence_number"],
            requested_at_monotonic=pending_request.get("requested_at_monotonic"),
        )
    except Exception as flush_error:
        log_debug(
            "LATENCY: failed to flush pending translation request: "
            f"{type(flush_error).__name__} - {flush_error}"
        )


def _queue_pending_translation_request(
    app,
    text_to_translate,
    ocr_sequence_number,
    delay_seconds,
    reason,
    requested_at_monotonic=None,
):
    now = time.monotonic()
    if requested_at_monotonic is None:
        requested_at_monotonic = now
    _increment_metric(app, "pending_translation_queued")
    app.pending_translation_request = {
        "text": text_to_translate,
        "ocr_sequence_number": ocr_sequence_number,
        "requested_at_monotonic": float(requested_at_monotonic),
    }
    log_debug(
        "LATENCY: queued latest translation request "
        f"for OCR batch {ocr_sequence_number} delay={delay_seconds:.3f}s "
        f"reason={reason}: '{text_to_translate}'"
    )

    desired_deadline = now + max(0.0, float(delay_seconds or 0.0))
    current_deadline = float(
        getattr(app, 'pending_translation_flush_deadline_monotonic', 0.0) or 0.0
    )
    if (
        getattr(app, 'pending_translation_flush_scheduled', False)
        and current_deadline > 0.0
        and current_deadline <= desired_deadline
    ):
        return

    generation = int(
        getattr(app, 'pending_translation_flush_generation', 0) or 0
    ) + 1
    app.pending_translation_flush_generation = generation
    app.pending_translation_flush_deadline_monotonic = desired_deadline
    app.pending_translation_flush_scheduled = True
    remaining_seconds = max(0.0, desired_deadline - time.monotonic())
    delay_ms = max(1, int(remaining_seconds * 1000))
    try:
        app.root.after(
            delay_ms,
            _flush_pending_translation_request,
            app,
            generation,
        )
    except Exception:
        if int(getattr(app, 'pending_translation_flush_generation', 0) or 0) == generation:
            app.pending_translation_flush_scheduled = False
            app.pending_translation_flush_deadline_monotonic = 0.0
        raise


def _expedite_pending_translation_request(app):
    pending_request = getattr(app, 'pending_translation_request', None)
    if not pending_request:
        return
    if hasattr(app, 'is_running') and not app.is_running:
        return
    _queue_pending_translation_request(
        app,
        pending_request["text"],
        pending_request["ocr_sequence_number"],
        0.0,
        "active translation completed",
        requested_at_monotonic=pending_request.get("requested_at_monotonic"),
    )


def _submit_async_translation_request(
    app,
    text_to_translate,
    ocr_sequence_number,
    inflight_key,
    requested_at_monotonic=None,
):
    app.translation_sequence_counter += 1
    translation_sequence = app.translation_sequence_counter
    app.latest_translation_sequence_started = translation_sequence
    submitted_at = time.monotonic()
    app.last_translation_submit_monotonic = submitted_at
    app.active_translation_calls.add(translation_sequence)
    app.active_translation_inflight_keys.add(inflight_key)
    _refresh_translation_metric_gauges(app)
    started_by_sequence = getattr(
        app,
        'active_translation_started_monotonic',
        None,
    )
    if not isinstance(started_by_sequence, dict):
        started_by_sequence = {}
        app.active_translation_started_monotonic = started_by_sequence
    started_by_sequence[translation_sequence] = submitted_at

    try:
        app.translation_thread_pool.submit(
            process_translation_async,
            app,
            text_to_translate,
            translation_sequence,
            ocr_sequence_number,
            inflight_key,
            requested_at_monotonic,
        )
    except Exception:
        app.active_translation_calls.discard(translation_sequence)
        app.active_translation_inflight_keys.discard(inflight_key)
        started_by_sequence.pop(translation_sequence, None)
        _refresh_translation_metric_gauges(app)
        raise

    log_debug(
        f"Started async translation {translation_sequence} for OCR batch {ocr_sequence_number} "
        f"(active calls: {len(app.active_translation_calls)}): '{text_to_translate}'"
    )

def start_async_translation(
    app,
    text_to_translate,
    ocr_sequence_number,
    requested_at_monotonic=None,
):
    """Start async translation processing to eliminate queue bottlenecks."""
    try:
        app.initialize_async_translation_infrastructure()

        enable_instant_var = getattr(app, 'enable_instant_cache_display_var', None)
        instant_cache_enabled = enable_instant_var.get() if enable_instant_var is not None else True
        if instant_cache_enabled and hasattr(app, 'translation_handler'):
            cached_translation = app.translation_handler.get_cached_translation_for_display(text_to_translate)
            if cached_translation:
                _increment_metric(app, "instant_cache_hit")
                final_processed_translation = post_process_translation_text(cached_translation)
                _invalidate_pending_translation_request(
                    app,
                    "newest subtitle satisfied by instant cache",
                )
                display_schedule_start = time.monotonic()
                app.update_translation_text(final_processed_translation)
                log_debug(f"LATENCY: display scheduling took {time.monotonic() - display_schedule_start:.3f}s")
                app.translation_sequence_counter += 1
                app.latest_translation_sequence_started = (
                    app.translation_sequence_counter
                )
                app.last_displayed_translation_sequence = app.translation_sequence_counter
                app.last_successful_translation_time = time.monotonic()
                log_debug(
                    "LATENCY: instant translation cache display "
                    f"for sequence {app.translation_sequence_counter}: '{final_processed_translation}'"
                )
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

        if inflight_key in app.active_translation_inflight_keys:
            _increment_metric(app, "duplicate_inflight_skip")
            log_debug(
                f"LATENCY: duplicate in-flight translation skipped for OCR batch {ocr_sequence_number}: "
                f"'{text_to_translate}'"
            )
            return

        now = time.monotonic()
        if requested_at_monotonic is None:
            requested_at_monotonic = now
        submit_interval = _get_translation_submit_interval_seconds(app, text_to_translate)
        cooldown_remaining = _get_translation_provider_cooldown_seconds(app)
        concurrency_limit = _get_translation_concurrency_limit(app)
        active_translation_count = len(app.active_translation_calls)
        _refresh_translation_metric_gauges(
            app,
            concurrency_limit=concurrency_limit,
            cooldown_remaining=cooldown_remaining,
        )
        elapsed_since_submit = now - float(getattr(app, 'last_translation_submit_monotonic', 0.0) or 0.0)

        queue_delay = 0.0
        queue_reasons = []

        if cooldown_remaining > 0:
            queue_delay = max(queue_delay, cooldown_remaining)
            queue_reasons.append(f"provider cooldown {cooldown_remaining:.3f}s")

        if elapsed_since_submit < submit_interval:
            remaining_interval = submit_interval - elapsed_since_submit
            queue_delay = max(queue_delay, remaining_interval)
            queue_reasons.append(f"submit interval {remaining_interval:.3f}s")

        may_supersede_stale_call = False
        if active_translation_count >= concurrency_limit:
            oldest_active_age = _get_oldest_active_translation_age(app, now)
            supersede_after = _get_translation_supersede_after_seconds(app)
            may_use_overflow_slot = (
                concurrency_limit == 1
                and active_translation_count < MAX_SUPERSEDED_TRANSLATION_CONCURRENCY
                and oldest_active_age is not None
                and _get_translation_latency_mode(app) != "race"
            )
            if may_use_overflow_slot and oldest_active_age >= supersede_after:
                may_supersede_stale_call = True
            else:
                if may_use_overflow_slot:
                    concurrency_delay = max(
                        0.001,
                        supersede_after - oldest_active_age,
                    )
                else:
                    concurrency_delay = max(submit_interval or 0.25, 0.25)
                queue_delay = max(queue_delay, concurrency_delay)
                queue_reasons.append(
                    f"active calls {active_translation_count}/"
                    f"{MAX_SUPERSEDED_TRANSLATION_CONCURRENCY if concurrency_limit == 1 else concurrency_limit}"
                )

        if queue_reasons:
            _queue_pending_translation_request(
                app,
                text_to_translate,
                ocr_sequence_number,
                queue_delay,
                ", ".join(queue_reasons),
                requested_at_monotonic=requested_at_monotonic,
            )
            return

        if may_supersede_stale_call:
            log_debug(
                "LATENCY: newest translation using one bounded overflow slot "
                f"for OCR batch {ocr_sequence_number} "
                f"age={oldest_active_age:.3f}s threshold={supersede_after:.3f}s"
            )

        _submit_async_translation_request(
            app,
            text_to_translate,
            ocr_sequence_number,
            inflight_key,
            requested_at_monotonic=requested_at_monotonic,
        )
        
    except Exception as e:
        log_debug(f"Error starting async translation: {type(e).__name__} - {e}")


def _build_streaming_display_callback(app, translation_sequence):
    state_lock = threading.Lock()
    state = {
        "latest_text": None,
        "scheduled": False,
    }

    def display_latest_partial():
        with state_lock:
            text = state["latest_text"]
            state["scheduled"] = False

        try:
            if not getattr(app, 'is_running', True):
                return
            latest_started = getattr(
                app,
                'latest_translation_sequence_started',
                translation_sequence,
            )
            if translation_sequence < latest_started:
                return
            last_displayed = getattr(
                app,
                'last_displayed_translation_sequence',
                0,
            )
            if translation_sequence <= last_displayed:
                return
            if isinstance(text, str) and text.strip():
                processed_text = post_process_translation_text(text)
                app.update_translation_text(processed_text)
                _increment_metric(app, "stream_partial_display")
                app.last_streamed_translation_display = (
                    translation_sequence,
                    processed_text,
                )
                app.last_successful_translation_time = time.monotonic()
        except Exception as stream_error:
            log_debug(
                "Streaming translation display failed: "
                f"{type(stream_error).__name__} - {stream_error}"
            )

    def stream_callback(partial_text):
        with state_lock:
            state["latest_text"] = partial_text
            if state["scheduled"]:
                return
            state["scheduled"] = True

        try:
            app.root.after(0, display_latest_partial)
        except Exception:
            with state_lock:
                state["scheduled"] = False
            raise

    return stream_callback


def process_translation_async(
    app,
    text_to_translate,
    translation_sequence,
    ocr_sequence_number,
    inflight_key=None,
    requested_at_monotonic=None,
):
    """Process translation API call asynchronously with timeout and staleness handling."""
    start_time = time.monotonic()
    if requested_at_monotonic is None:
        requested_at_monotonic = start_time
    
    try:
        log_debug(f"Processing async translation {translation_sequence}")

        stream_callback = None
        latency_mode_var = getattr(app, 'custom_ai_latency_mode_var', None)
        try:
            latency_mode = latency_mode_var.get() if latency_mode_var is not None else ""
        except Exception:
            latency_mode = ""

        if latency_mode == "stream":
            stream_callback = _build_streaming_display_callback(
                app,
                translation_sequence,
            )
        
        translation_result = app.translation_handler.translate_text_with_timeout(
            text_to_translate,
            timeout_seconds=10.0,
            ocr_batch_number=ocr_sequence_number,
            stream_callback=stream_callback,
            translation_sequence=translation_sequence,
            latency_mode=latency_mode,
        )
        
        completed_at = time.monotonic()
        elapsed_time = completed_at - start_time
        queue_time = max(0.0, start_time - float(requested_at_monotonic))
        total_time = max(0.0, completed_at - float(requested_at_monotonic))
        log_debug(
            "LATENCY: translation timing "
            f"sequence={translation_sequence} queue={queue_time:.3f}s "
            f"worker={elapsed_time:.3f}s total={total_time:.3f}s"
        )
        _record_metric_timing(app, "translation_queue_time", queue_time)
        _record_metric_timing(app, "translation_worker_time", elapsed_time)
        _record_metric_timing(app, "translation_total_latency", total_time)
        if elapsed_time > 5.0:
            log_debug(f"Translation {translation_sequence} took {elapsed_time:.1f}s, may be stale but will attempt display")
        
        log_debug(f"Translation {translation_sequence} completed in {elapsed_time:.3f}s: '{translation_result}'")
        
        app.root.after(0, process_translation_response, app, translation_result, translation_sequence, text_to_translate, ocr_sequence_number)
        
    except Exception as e:
        completed_at = time.monotonic()
        elapsed_time = completed_at - start_time
        queue_time = max(0.0, start_time - float(requested_at_monotonic))
        total_time = max(0.0, completed_at - float(requested_at_monotonic))
        log_debug(
            "LATENCY: translation timing "
            f"sequence={translation_sequence} queue={queue_time:.3f}s "
            f"worker={elapsed_time:.3f}s total={total_time:.3f}s"
        )
        _record_metric_timing(app, "translation_queue_time", queue_time)
        _record_metric_timing(app, "translation_worker_time", elapsed_time)
        _record_metric_timing(app, "translation_total_latency", total_time)
        log_debug(f"Error in async translation {translation_sequence} after {elapsed_time:.2f}s: {type(e).__name__} - {e}")
        
        error_msg = f"Translation error: {str(e)}"
        app.root.after(0, process_translation_response, app, error_msg, translation_sequence, text_to_translate, ocr_sequence_number)
    
    finally:
        try:
            app.active_translation_calls.discard(translation_sequence)
            if inflight_key is not None and hasattr(app, 'active_translation_inflight_keys'):
                app.active_translation_inflight_keys.discard(inflight_key)
            started_by_sequence = getattr(
                app,
                'active_translation_started_monotonic',
                None,
            )
            if isinstance(started_by_sequence, dict):
                started_by_sequence.pop(translation_sequence, None)
            _refresh_translation_metric_gauges(app)
            log_debug(f"Translation {translation_sequence} finished (active calls: {len(app.active_translation_calls)})")
            _expedite_pending_translation_request(app)
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
            _increment_metric(app, "stale_response_discarded")
            log_debug(f"Translation {translation_sequence}: Sequence too old (last displayed: {app.last_displayed_translation_sequence}), discarding but caching result")
            return
        
        log_debug(f"Translation {translation_sequence}: Processing newer sequence (last displayed: {app.last_displayed_translation_sequence})")
        
        if is_translation_error_result(translation_result):
            log_debug(f"Translation error in sequence {translation_sequence}: {translation_result}")
            app.update_translation_text(f"Translation Error:\n{translation_result}")
            app.last_displayed_translation_sequence = translation_sequence
            _clear_local_ocr_submit_state(app)
            return
        
        if isinstance(translation_result, str) and translation_result.strip():
            final_processed_translation = post_process_translation_text(translation_result)
            streamed_display = getattr(
                app,
                'last_streamed_translation_display',
                None,
            )
            if streamed_display != (
                translation_sequence,
                final_processed_translation,
            ):
                display_schedule_start = time.monotonic()
                app.update_translation_text(final_processed_translation)
                log_debug(
                    "LATENCY: display scheduling took "
                    f"{time.monotonic() - display_schedule_start:.3f}s"
                )
            else:
                log_debug(
                    "LATENCY: skipped duplicate final display after stream "
                    f"for sequence={translation_sequence}"
                )
            log_debug(f"Translation {translation_sequence} displayed: '{final_processed_translation}' (from OCR batch {ocr_sequence_number})")
            app.last_displayed_translation_sequence = translation_sequence
            app.last_successful_translation_time = time.monotonic()
            if ocr_sequence_number == 0:
                _remember_local_ocr_submit(app, original_text)
        else:
            log_debug(f"Translation {translation_sequence}: Empty or invalid result, not displaying")
            _clear_local_ocr_submit_state(app)
            
    except Exception as e:
        log_debug(f"Error processing translation response for sequence {translation_sequence}: {type(e).__name__} - {e}")
        try:
            app.update_translation_text(f"Translation Processing Error:\n{type(e).__name__}")
        except:
            pass
