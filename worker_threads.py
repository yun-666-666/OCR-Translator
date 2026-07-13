# worker_threads.py (Complete, Corrected File)

import tkinter as tk # For tk.Toplevel type check in capture_thread
from difflib import SequenceMatcher
import time
import queue
import numpy as np
import cv2
import pyautogui
import hashlib
import math
import re
import threading
import traceback
from datetime import datetime

from logger import log_debug, log_debug_coalesced, summarize_text_for_log
from api_ocr_request import ApiOcrRequestSnapshot
from custom_ai import (
    CUSTOM_AI_LATENCY_MODE_ADAPTIVE,
    CUSTOM_AI_LATENCY_MODE_SAFE,
    normalize_custom_ai_latency_mode,
)
from ocr_utils import (
    capture_screen_region,
    build_capture_signature, build_ocr_frame_cache_key,
    CaptureBackendSelector,
    API_OCR_IMAGE_DETAIL_DEFAULT, API_OCR_IMAGE_FORMAT_DEFAULT, API_OCR_IMAGE_MODE_DEFAULT,
    API_OCR_IMAGE_QUALITY_DEFAULT, normalize_api_ocr_image_detail,
    normalize_api_ocr_image_format, normalize_api_ocr_image_mode,
    normalize_api_ocr_image_quality,
)
from paddle_ocr_backend import (
    PADDLEOCR_MODEL_CODE,
    PaddleOCRSettings,
    prepare_paddleocr_image,
    recognize_subtitle_with_paddleocr,
)
from translation_utils import (
    is_translation_error_result,
    post_process_translation_text,
)
from PIL import Image # For hashing in capture_thread


DEFAULT_TRANSLATION_SUPERSEDE_AFTER_SECONDS = 1.5
DEFAULT_TRANSLATION_REQUEST_TIMEOUT_SECONDS = 10.0
MAX_SUPERSEDED_TRANSLATION_CONCURRENCY = 2
OCR_STABILITY_GATE_MIN_WAIT_SECONDS = 0.12
OCR_STABILITY_GATE_MAX_WAIT_SECONDS = 0.25
OCR_STABILITY_GATE_SUSPICIOUS_SHORT_LENGTH = 12
OCR_STABILITY_GATE_NOISE_RATIO = 0.35
CAPTURE_SLOW_SECONDS_MSS = 0.050
CAPTURE_SLOW_SECONDS_PYAUTOGUI = 0.250
OCR_CACHE_HIT_SLOW_SECONDS = 0.050
PADDLE_OCR_SLOW_SECONDS = 0.500


from worker_capture import (
    run_capture_thread,
    _log_hot_path_timing,
    _request_snapshot_timeout_seconds,
    _log_paddle_ocr_route,
    _runtime_metrics,
    _record_metric_timing,
    _increment_metric,
    _set_metric_gauge,
    _safe_queue_size,
    _get_capture_backend_selector,
    _resolve_capture_backend,
    _refresh_translation_metric_gauges,
    _refresh_ocr_queue_metric,
    _get_screenshot_frame_hash,
    _advance_local_capture_signature,
    _get_api_ocr_cache_model_key,
    _read_app_var,
    _coerce_float,
    _coerce_int,
    _coerce_bool,
    get_paddleocr_settings_from_app,
    get_paddleocr_ocr_cache_mode_key,
    _pil_to_debug_bgr,
    process_local_ocr_frame,
)

from worker_ocr import (
    _normalize_custom_ai_ocr_reasoning_contract,
    _get_custom_ai_ocr_reasoning_contract,
    _normalize_local_ocr_submit_text,
    _ocr_candidate_has_terminal_punctuation,
    _ocr_candidate_tokens,
    _ocr_candidate_noise_ratio,
    _looks_like_low_quality_ocr_candidate,
    _is_transient_custom_ai_provider_error,
    _ocr_candidate_quality_score,
    _looks_like_more_complete_ocr_candidate,
    OcrStabilityDecision,
    OcrStabilityGate,
    _get_local_ocr_submit_scope,
    _looks_like_safe_local_ocr_near_repeat,
    _should_skip_local_ocr_resubmit,
    _remember_local_ocr_submit,
    _clear_local_ocr_submit_state,
    _get_ocr_stability_gate,
    _has_pending_ocr_stability_candidate,
    _translation_display_epoch,
    _advance_translation_display_activity,
    _translation_clear_blocker,
    _apply_inactive_translation_clear,
    _schedule_inactive_translation_clear,
    _clear_ocr_stability_gate,
    enqueue_ocr_frame_for_model,
    _submit_final_local_ocr_text,
    _schedule_ocr_stability_flush,
    _flush_ocr_stability_candidate,
    _route_local_ocr_candidate_for_translation,
    _get_api_ocr_cache_mode_key,
)

from worker_translation import (
    _get_translation_submit_interval_seconds,
    _get_translation_provider_cooldown_seconds,
    _get_local_ocr_translation_gate_seconds,
    _get_local_ocr_elapsed_since_last_submit,
    _get_translation_concurrency_limit,
    _get_translation_supersede_after_seconds,
    _get_translation_latency_mode,
    _get_oldest_active_translation_age,
    _invalidate_pending_translation_request,
    reset_translation_scheduler_session_state,
    _flush_pending_translation_request,
    _queue_pending_translation_request,
    _expedite_pending_translation_request,
    _apply_translation_profile_refresh,
    refresh_translation_after_profile_change,
    _submit_async_translation_request,
    _coalesce_matching_pending_translation_request,
    _build_streaming_display_callback,
)




def run_ocr_thread(app):
    log_debug("WT: OCR thread started.")
    log_debug(f"WT: OCR using {app.get_ocr_model_setting()}")

    last_lang_check = time.monotonic()
    last_ocr_proc_time = 0
    min_ocr_interval = 0.1
    similar_texts_count = 0
    prev_ocr_text = ""

    while app.is_running:
        now = time.monotonic()
        try:
            if now - last_lang_check > 5.0:
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
                _refresh_ocr_queue_metric(app)
            except queue.Empty:
                time.sleep(0.05)
                continue

            ocr_proc_start_time = time.monotonic()
            last_ocr_proc_time = ocr_proc_start_time
            app.last_screenshot = screenshot_pil
            processed_cv_img = None

            frame_hash = _get_screenshot_frame_hash(screenshot_pil)

            region_origin = getattr(screenshot_pil, '_gct_region_origin', (0, 0))
            ocr_cache_key = None
            if hasattr(app, 'ocr_frame_cache') and not app.is_api_based_ocr_model(ocr_model):
                if ocr_model == PADDLEOCR_MODEL_CODE:
                    cache_lang = _read_app_var(app, "paddleocr_lang_var", "en")
                    cache_mode_key = get_paddleocr_ocr_cache_mode_key(app)
                else:
                    ocr_model = PADDLEOCR_MODEL_CODE
                    cache_lang = _read_app_var(app, "paddleocr_lang_var", "en")
                    cache_mode_key = get_paddleocr_ocr_cache_mode_key(app)
                ocr_cache_key = build_ocr_frame_cache_key(
                    frame_hash,
                    ocr_model,
                    cache_lang,
                    cache_mode_key,
                    screenshot_pil.size,
                    region_origin=region_origin,
                )
                cached_ocr_text = app.ocr_frame_cache.get(ocr_cache_key)
                if cached_ocr_text is not None:
                    ocr_cleaned_text = cached_ocr_text
                    ocr_duration = time.monotonic() - ocr_proc_start_time
                    _log_hot_path_timing(
                        ("ocr-cache-hit-timing", ocr_model),
                        f"LATENCY: OCR cache hit for {ocr_model} "
                        f"took {ocr_duration:.3f}s",
                        ocr_duration,
                        OCR_CACHE_HIT_SLOW_SECONDS,
                    )
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

            elif ocr_model == PADDLEOCR_MODEL_CODE:
                _log_paddle_ocr_route()

            else:
                log_debug(f"WT: OCR: Unknown OCR model '{ocr_model}', falling back to PaddleOCR")
                ocr_model = PADDLEOCR_MODEL_CODE

            if not goto_post_ocr:
                # ==================== LOCAL OCR PROCESSING ====================
                if app.ocr_debugging_var.get():
                    app.root.after(0, app.update_debug_display, screenshot_pil, _pil_to_debug_bgr(screenshot_pil), "Processing...")

                ocr_cleaned_text, processed_cv_img, engine_label = process_local_ocr_frame(
                    app,
                    screenshot_pil,
                    ocr_model,
                )
                app.last_processed_image = processed_cv_img

                if ocr_cache_key is not None:
                    app.ocr_frame_cache.put(ocr_cache_key, ocr_cleaned_text)
                ocr_duration = time.monotonic() - ocr_proc_start_time
                _log_hot_path_timing(
                    ("ocr-processing-timing", engine_label),
                    f"LATENCY: {engine_label} OCR took {ocr_duration:.3f}s",
                    ocr_duration,
                    PADDLE_OCR_SLOW_SECONDS,
                )
                _record_metric_timing(app, "ocr_duration", ocr_duration)


            if app.ocr_debugging_var.get() and processed_cv_img is not None:
                app.root.after(0, app.update_debug_display, screenshot_pil, processed_cv_img, ocr_cleaned_text)

            if not ocr_cleaned_text or app.is_placeholder_text(ocr_cleaned_text):
                app.text_stability_counter = 0
                app.previous_text = ""
                _clear_ocr_stability_gate(app, "empty or placeholder OCR")
                continue

            if ocr_model == PADDLEOCR_MODEL_CODE:
                route_result = _route_local_ocr_candidate_for_translation(
                    app,
                    ocr_cleaned_text,
                    now=time.monotonic(),
                    ocr_sequence_number=0,
                )
                if route_result in ("submitted", "skipped", "dropped", "pending"):
                    app.text_stability_counter = 0
                    similar_texts_count = 0
                    continue

            if _has_pending_ocr_stability_candidate(app):
                route_result = _route_local_ocr_candidate_for_translation(
                    app,
                    ocr_cleaned_text,
                    now=now,
                    ocr_sequence_number=0,
                )
                if route_result in ("submitted", "skipped", "dropped"):
                    app.text_stability_counter = 0
                    similar_texts_count = 0
                    continue
                if route_result == "pending":
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
                    _route_local_ocr_candidate_for_translation(
                        app,
                        ocr_cleaned_text,
                        now=now,
                        ocr_sequence_number=0,
                    )
                    app.text_stability_counter = 0
                    if ocr_cleaned_text != app.previous_text:
                        app.previous_text = ocr_cleaned_text
                    similar_texts_count = 0

        except tk.TclError:
            log_debug("WT: OCR thread TclError.")
            if not app.is_running: break
            time.sleep(0.1)
        except Exception as e_ocr_loop_wt:
            log_debug(f"WT: OCR thread error: {type(e_ocr_loop_wt).__name__} - {e_ocr_loop_wt}\n{traceback.format_exc()}")
            app.text_stability_counter=0
            app.previous_text=""
            time.sleep(0.2)
    _clear_ocr_stability_gate(app, "OCR thread stopped")
    log_debug("WT: OCR thread finished.")

def run_translation_thread(app):
    """Simplified translation thread for async processing."""
    log_debug("WT: Translation thread started (simplified for async processing).")
    thread_local_last_translation_display_time = time.monotonic()
    thread_local_last_displayed_sequence = int(
        getattr(app, 'last_displayed_translation_sequence', 0) or 0
    )

    while app.is_running:
        now = time.monotonic()
        try:
            (
                thread_local_last_translation_display_time,
                thread_local_last_displayed_sequence,
            ) = _advance_translation_display_activity(
                app,
                now,
                thread_local_last_translation_display_time,
                thread_local_last_displayed_sequence,
            )
            ocr_model = app.get_ocr_model_setting()

            if not app.is_api_based_ocr_model(ocr_model):
                inactive_duration = now - thread_local_last_translation_display_time
                if app.clear_translation_timeout > 0 and inactive_duration > app.clear_translation_timeout:
                    _schedule_inactive_translation_clear(
                        app,
                        inactive_duration,
                        app.clear_translation_timeout,
                    )

            try:
                text_to_translate = app.translation_queue.get(timeout=0.1)
                if text_to_translate and not app.is_placeholder_text(text_to_translate):
                    log_debug(
                        "WT: Processing legacy queue item "
                        f"{summarize_text_for_log(text_to_translate)}"
                    )
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

def _read_api_ocr_submit_value(app, getter_name, var_name, default):
    """Read one UI setting on the API OCR submission thread only."""
    getter = getattr(app, getter_name, None) if getter_name else None
    if callable(getter):
        try:
            return getter()
        except Exception:
            return default
    variable = getattr(app, var_name, None)
    getter = getattr(variable, 'get', None)
    if callable(getter):
        try:
            return getter()
        except Exception:
            return default
    return default


def _build_api_ocr_request_snapshot(
    app,
    provider_name,
    source_lang,
    sequence,
    frame_hash,
    region_size,
    region_origin,
):
    """Freeze Custom AI OCR inputs before cache lookup or image conversion."""
    if str(provider_name or '').strip().lower() != 'custom_ai':
        return None

    profiles = getattr(app, 'custom_ai_profiles', None)
    profile_getter = getattr(profiles, 'get_active_profile', None)
    if not callable(profile_getter):
        return None
    try:
        profile = profile_getter('ocr')
    except Exception as error:
        log_debug(
            "Could not resolve active Custom AI OCR profile for request snapshot: "
            f"{type(error).__name__}"
        )
        return None
    if not isinstance(profile, dict) or not profile:
        log_debug("No active custom AI model profile configured for OCR request snapshot")
        return None

    reasoning_effort = (
        profile.get('reasoning_effort')
        or profile.get('model_reasoning_effort')
        or 'low'
    )
    provider = getattr(
        getattr(app, 'translation_handler', None),
        'custom_ai_provider',
        None,
    )
    reasoning_getter = getattr(provider, 'reasoning_effort_request_contract', None)
    if callable(reasoning_getter):
        try:
            reasoning_effort = reasoning_getter(profile, 'ocr')
        except Exception as error:
            log_debug(
                "Could not resolve Custom AI OCR reasoning request contract: "
                f"{type(error).__name__}"
            )

    latency_mode = normalize_custom_ai_latency_mode(
        _read_api_ocr_submit_value(
            app,
            'get_custom_ai_latency_mode',
            'custom_ai_latency_mode_var',
            CUSTOM_AI_LATENCY_MODE_SAFE,
        )
    )
    if latency_mode == CUSTOM_AI_LATENCY_MODE_ADAPTIVE:
        latency_mode = CUSTOM_AI_LATENCY_MODE_SAFE
    image_format = normalize_api_ocr_image_format(
        _read_api_ocr_submit_value(
            app,
            'get_custom_ai_ocr_image_format',
            'custom_ai_ocr_image_format_var',
            API_OCR_IMAGE_FORMAT_DEFAULT,
        )
    )
    try:
        return ApiOcrRequestSnapshot.create(
            generation=getattr(app, 'api_ocr_request_generation', 0),
            sequence=sequence,
            provider=provider_name,
            profile=profile,
            source_language=source_lang,
            keep_linebreaks=_read_api_ocr_submit_value(
                app,
                None,
                'keep_linebreaks_var',
                False,
            ),
            latency_mode=latency_mode,
            reasoning_effort=_normalize_custom_ai_ocr_reasoning_contract(
                reasoning_effort
            ),
            image_detail=normalize_api_ocr_image_detail(
                _read_api_ocr_submit_value(
                    app,
                    'get_custom_ai_ocr_image_detail',
                    'custom_ai_ocr_image_detail_var',
                    API_OCR_IMAGE_DETAIL_DEFAULT,
                )
            ),
            image_format=image_format,
            image_mode=normalize_api_ocr_image_mode(
                _read_api_ocr_submit_value(
                    app,
                    'get_custom_ai_ocr_image_mode',
                    'custom_ai_ocr_image_mode_var',
                    API_OCR_IMAGE_MODE_DEFAULT,
                )
            ),
            image_quality=normalize_api_ocr_image_quality(
                _read_api_ocr_submit_value(
                    app,
                    'get_custom_ai_ocr_image_quality',
                    'custom_ai_ocr_image_quality_var',
                    API_OCR_IMAGE_QUALITY_DEFAULT,
                )
            ),
            mime_type={
                'webp': 'image/webp',
                'png': 'image/png',
                'jpeg': 'image/jpeg',
            }[image_format],
            frame_hash=frame_hash,
            region_size=region_size,
            region_origin=region_origin,
        )
    except (TypeError, ValueError) as error:
        log_debug(
            "Could not create Custom AI OCR request snapshot: "
            f"{type(error).__name__}"
        )
        return None

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

        request_snapshot = None
        is_custom_ai_provider = (
            str(provider_name or '').strip().lower() == 'custom_ai'
        )
        if is_custom_ai_provider and hasattr(app, 'custom_ai_profiles'):
            frame_hash = _get_screenshot_frame_hash(screenshot_pil)
            region_origin = getattr(screenshot_pil, '_gct_region_origin', (0, 0))
            app.batch_sequence_counter += 1
            sequence_number = app.batch_sequence_counter
            request_snapshot = _build_api_ocr_request_snapshot(
                app,
                provider_name,
                source_lang,
                sequence_number,
                frame_hash,
                screenshot_pil.size,
                region_origin,
            )
            if request_snapshot is None:
                return

        ocr_cache_key = None
        if hasattr(app, 'ocr_frame_cache'):
            if request_snapshot is not None:
                ocr_cache_key = request_snapshot.frame_cache_key
            else:
                frame_hash = _get_screenshot_frame_hash(screenshot_pil)
                region_origin = getattr(screenshot_pil, '_gct_region_origin', (0, 0))
                ocr_cache_key = build_ocr_frame_cache_key(
                    frame_hash,
                    _get_api_ocr_cache_model_key(app, provider_name),
                    source_lang,
                    _get_api_ocr_cache_mode_key(app, provider_name),
                    screenshot_pil.size,
                    region_origin=region_origin,
                )
            cached_ocr_text = app.ocr_frame_cache.get(ocr_cache_key)
            if cached_ocr_text is not None:
                if request_snapshot is None:
                    app.batch_sequence_counter += 1
                    sequence_number = app.batch_sequence_counter
                log_debug(f"LATENCY: API OCR cache hit for {provider_name} batch {sequence_number}")
                _increment_metric(app, "ocr_frame_cache_hit")
                _record_metric_timing(app, "ocr_duration", time.monotonic() - ocr_start_time)
                if request_snapshot is not None:
                    process_api_ocr_snapshot_response(
                        app,
                        cached_ocr_text,
                        request_snapshot,
                    )
                else:
                    process_api_ocr_response(
                        app,
                        cached_ocr_text,
                        sequence_number,
                        source_lang,
                        provider_name,
                        ocr_cache_key=ocr_cache_key,
                    )
                return

        if len(app.active_ocr_calls) >= app.max_concurrent_ocr_calls:
            _set_metric_gauge(app, "active_ocr_calls", len(app.active_ocr_calls))
            log_debug(f"Max concurrent OCR calls ({app.max_concurrent_ocr_calls}) reached, skipping {provider_name} OCR before image conversion")
            return

        encoded_image = None
        metadata_encoder = getattr(app, 'convert_to_api_ocr_image', None)
        if callable(metadata_encoder):
            if request_snapshot is not None:
                encoded_image = metadata_encoder(
                    screenshot_pil,
                    image_format=request_snapshot.image_format,
                    mode=request_snapshot.image_mode,
                    quality=request_snapshot.image_quality,
                    detail=request_snapshot.image_detail,
                    mime_type=request_snapshot.mime_type,
                )
            else:
                encoded_image = metadata_encoder(screenshot_pil)
        else:
            legacy_bytes = app.convert_to_webp_for_api(screenshot_pil)
            if legacy_bytes:
                encoded_image = type(
                    "EncodedApiOcrImageCompat",
                    (),
                    {
                        "data": legacy_bytes,
                        "mime_type": "image/webp",
                        "image_format": "webp",
                    },
                )()

        if request_snapshot is not None and (
            str(getattr(encoded_image, 'image_format', '')).strip().lower()
            != request_snapshot.image_format
            or str(getattr(encoded_image, 'mime_type', '')).strip().lower()
            != request_snapshot.mime_type
        ):
            log_debug(
                f"Custom AI OCR batch {sequence_number} image contract mismatch; skipping provider call"
            )
            return

        if not encoded_image or not getattr(encoded_image, 'data', None):
            log_debug(f"Failed to convert image for {provider_name} OCR")
            return
        image_data = encoded_image.data
        image_mime_type = getattr(encoded_image, 'mime_type', 'image/webp')

        if request_snapshot is None:
            app.batch_sequence_counter += 1
            sequence_number = app.batch_sequence_counter

        app.active_ocr_calls.add(sequence_number)
        _set_metric_gauge(app, "active_ocr_calls", len(app.active_ocr_calls))
        try:
            if request_snapshot is not None:
                app.ocr_thread_pool.submit(
                    process_api_ocr_snapshot_async,
                    app,
                    image_data,
                    request_snapshot,
                )
            else:
                app.ocr_thread_pool.submit(
                    process_api_ocr_async,
                    app,
                    image_data,
                    source_lang,
                    sequence_number,
                    provider_name,
                    ocr_cache_key,
                    image_mime_type,
                )
        except Exception:
            app.active_ocr_calls.discard(sequence_number)
            raise
        log_debug(f"Started {provider_name} OCR batch {sequence_number} (active calls: {len(app.active_ocr_calls)})")

    except Exception as e:
        log_debug(f"Error starting API OCR batch: {type(e).__name__} - {e}")

def process_api_ocr_snapshot_async(app, image_data, request_snapshot):
    """Run one fully frozen Custom AI OCR request without configuration reads."""
    sequence_number = request_snapshot.sequence
    provider_name = request_snapshot.provider
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
        ocr_result = app.translation_handler.perform_ocr(
            image_data,
            request_snapshot=request_snapshot,
        )
        _record_metric_timing(app, "ocr_duration", time.monotonic() - ocr_start_time)
        log_debug(
            f"{provider_name} OCR batch {sequence_number} completed, "
            f"scheduling response {summarize_text_for_log(ocr_result)}"
        )
        app.root.after(
            0,
            process_api_ocr_snapshot_response,
            app,
            ocr_result,
            request_snapshot,
        )
    except Exception as e:
        log_debug(f"Error in async {provider_name} OCR batch {sequence_number}: {type(e).__name__} - {e}")
        error_msg = f"<e>: OCR batch {sequence_number} error: {str(e)}"
        app.root.after(
            0,
            process_api_ocr_snapshot_response,
            app,
            error_msg,
            request_snapshot,
        )
    finally:
        app.active_ocr_calls.discard(sequence_number)
        _set_metric_gauge(app, "active_ocr_calls", len(app.active_ocr_calls))
        log_debug(f"{provider_name} OCR batch {sequence_number} finished (active calls: {len(app.active_ocr_calls)})")


def process_api_ocr_snapshot_response(app, ocr_result, request_snapshot):
    """Schedule response bookkeeping solely from an API OCR request snapshot."""
    return process_api_ocr_response(
        app,
        ocr_result,
        request_snapshot.sequence,
        request_snapshot.source_language,
        request_snapshot.provider,
        ocr_cache_key=request_snapshot.frame_cache_key,
    )


def process_api_ocr_async(
    app,
    image_data,
    source_lang,
    sequence_number,
    provider_name,
    ocr_cache_key=None,
    image_mime_type="image/webp",
):
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
        ocr_result = app.translation_handler.perform_ocr(
            image_data,
            source_lang,
            image_mime_type=image_mime_type,
        )
        _record_metric_timing(app, "ocr_duration", time.monotonic() - ocr_start_time)

        log_debug(
            f"{provider_name} OCR batch {sequence_number} completed, "
            f"scheduling response {summarize_text_for_log(ocr_result)}"
        )
        app.root.after(
            0,
            process_api_ocr_response,
            app,
            ocr_result,
            sequence_number,
            source_lang,
            provider_name,
            ocr_cache_key,
        )

    except Exception as e:
        log_debug(f"Error in async {provider_name} OCR batch {sequence_number}: {type(e).__name__} - {e}")
        error_msg = f"<e>: OCR batch {sequence_number} error: {str(e)}"
        app.root.after(
            0,
            process_api_ocr_response,
            app,
            error_msg,
            sequence_number,
            source_lang,
            provider_name,
            ocr_cache_key,
        )

    finally:
        app.active_ocr_calls.discard(sequence_number)
        _set_metric_gauge(app, "active_ocr_calls", len(app.active_ocr_calls))
        log_debug(f"{provider_name} OCR batch {sequence_number} finished (active calls: {len(app.active_ocr_calls)})")

def process_api_ocr_response(
    app,
    ocr_result,
    sequence_number,
    source_lang,
    provider_name,
    ocr_cache_key=None,
):
    """Process any API OCR response with chronological order enforcement. This is the generic callback."""
    try:
        log_debug(
            f"Processing {provider_name} OCR response for batch "
            f"{sequence_number} {summarize_text_for_log(ocr_result)}"
        )

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
            log_debug(
                f"OCR error in {provider_name} batch {sequence_number} "
                f"{summarize_text_for_log(ocr_result)}"
            )
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
            log_debug(
                "Keeping existing translation for successive identical "
                f"{provider_name} OCR {summarize_text_for_log(ocr_result)}"
            )
            app.last_displayed_batch_sequence = sequence_number
            return

        app.last_processed_subtitle = ocr_result
        app.reset_clear_timeout()
        start_async_translation(app, ocr_result, sequence_number)
        app.last_displayed_batch_sequence = sequence_number

    except Exception as e:
        log_debug(f"Error processing {provider_name} OCR response for batch {sequence_number}: {type(e).__name__} - {e}")


# ==================== ASYNC TRANSLATION PROCESSING (Phase 2) ====================

def start_async_translation(
    app,
    text_to_translate,
    ocr_sequence_number,
    requested_at_monotonic=None,
    configuration_refresh=False,
):
    """Start async translation processing to eliminate queue bottlenecks."""
    try:
        app.initialize_async_translation_infrastructure()

        if requested_at_monotonic is None:
            requested_at_monotonic = time.monotonic()
        app.latest_translation_candidate = {
            "text": text_to_translate,
            "ocr_sequence_number": ocr_sequence_number,
            "requested_at_monotonic": float(requested_at_monotonic),
        }

        if _coalesce_matching_pending_translation_request(
            app,
            text_to_translate,
            ocr_sequence_number,
        ):
            return

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
                    f"for sequence {app.translation_sequence_counter} "
                    f"{summarize_text_for_log(final_processed_translation)}"
                )
                return

        if not hasattr(app, 'active_translation_inflight_keys'):
            app.active_translation_inflight_keys = set()

        inflight_key = None
        latency_mode = None
        request_snapshot = None
        handler = getattr(app, 'translation_handler', None)
        snapshot_getter = getattr(
            handler,
            'get_custom_ai_translation_request_snapshot',
            None,
        )
        if callable(snapshot_getter):
            try:
                request_snapshot = snapshot_getter(
                    text_to_translate,
                    commit=False,
                )
            except Exception as snapshot_error:
                log_debug(
                    "LATENCY: failed to build translation request snapshot: "
                    f"{type(snapshot_error).__name__} - {snapshot_error}"
                )
                request_snapshot = None
        if isinstance(request_snapshot, dict):
            inflight_key = request_snapshot.get("inflight_key")
            latency_mode = request_snapshot.get("latency_mode")
        if inflight_key is None and hasattr(app, 'translation_handler') and hasattr(app.translation_handler, 'get_inflight_translation_key'):
            try:
                inflight_key = app.translation_handler.get_inflight_translation_key(text_to_translate)
            except Exception as key_error:
                log_debug(f"LATENCY: failed to build translation inflight key: {type(key_error).__name__} - {key_error}")
        if inflight_key is None:
            inflight_key = ("raw", text_to_translate)
        if latency_mode is None:
            latency_mode = _get_translation_latency_mode(app)

        if inflight_key in app.active_translation_inflight_keys:
            _increment_metric(app, "duplicate_inflight_skip")
            log_debug_coalesced(
                "translation-duplicate-inflight",
                "LATENCY: duplicate in-flight translation skipped "
                f"for OCR batch {ocr_sequence_number}",
                interval_seconds=5.0,
            )
            return

        now = time.monotonic()
        submit_interval = _get_translation_submit_interval_seconds(app, text_to_translate)
        cooldown_remaining = _get_translation_provider_cooldown_seconds(
            app,
            latency_mode=latency_mode,
        )
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

        if not configuration_refresh and elapsed_since_submit < submit_interval:
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
                and _get_translation_latency_mode(app, latency_mode) != "race"
                and (
                    configuration_refresh
                    or oldest_active_age is not None
                )
            )
            if may_use_overflow_slot and (
                configuration_refresh
                or oldest_active_age >= supersede_after
            ):
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
                configuration_refresh=configuration_refresh,
            )
            return

        if may_supersede_stale_call:
            oldest_active_age_text = (
                f"{oldest_active_age:.3f}s"
                if oldest_active_age is not None
                else "unknown"
            )
            log_debug(
                "LATENCY: newest translation using one bounded overflow slot "
                f"for OCR batch {ocr_sequence_number} "
                f"age={oldest_active_age_text} "
                f"threshold={supersede_after:.3f}s"
            )

        _submit_async_translation_request(
            app,
            text_to_translate,
            ocr_sequence_number,
            inflight_key,
            requested_at_monotonic=requested_at_monotonic,
            latency_mode=latency_mode,
            request_snapshot=request_snapshot,
        )
        commit_snapshot = getattr(
            getattr(app, 'translation_handler', None),
            'commit_custom_ai_latency_mode_snapshot',
            None,
        )
        if callable(commit_snapshot) and isinstance(request_snapshot, dict):
            try:
                commit_snapshot(request_snapshot)
            except Exception as commit_error:
                log_debug(
                    "LATENCY: failed to commit translation latency snapshot: "
                    f"{type(commit_error).__name__} - {commit_error}"
                )

    except Exception as e:
        log_debug(f"Error starting async translation: {type(e).__name__} - {e}")


def process_translation_async(
    app,
    text_to_translate,
    translation_sequence,
    ocr_sequence_number,
    inflight_key=None,
    requested_at_monotonic=None,
    latency_mode=None,
    request_snapshot=None,
):
    """Process translation API call asynchronously with timeout and staleness handling."""
    start_time = time.monotonic()
    if requested_at_monotonic is None:
        requested_at_monotonic = start_time

    try:
        log_debug(f"Processing async translation {translation_sequence}")

        stream_callback = None
        if latency_mode is None:
            latency_mode_var = getattr(app, 'custom_ai_latency_mode_var', None)
            try:
                latency_mode = latency_mode_var.get() if latency_mode_var is not None else ""
            except Exception:
                latency_mode = ""
        latency_mode = str(latency_mode or "").strip().lower()

        if latency_mode == "stream":
            stream_callback = _build_streaming_display_callback(
                app,
                translation_sequence,
            )

        translation_kwargs = {
            "timeout_seconds": _request_snapshot_timeout_seconds(
                request_snapshot,
                latency_mode,
            ),
            "ocr_batch_number": ocr_sequence_number,
            "stream_callback": stream_callback,
            "translation_sequence": translation_sequence,
            "latency_mode": latency_mode,
        }
        if request_snapshot is not None:
            translation_kwargs["request_snapshot"] = request_snapshot
        translation_result = app.translation_handler.translate_text_with_timeout(
            text_to_translate,
            **translation_kwargs,
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

        log_debug(
            f"Translation {translation_sequence} completed in {elapsed_time:.3f}s "
            f"{summarize_text_for_log(translation_result)}"
        )

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
        log_debug(
            "Processing translation response for sequence "
            f"{translation_sequence} {summarize_text_for_log(translation_result)}"
        )

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
            if _is_transient_custom_ai_provider_error(translation_result):
                log_debug(
                    "Translation transient provider error suppressed in "
                    f"sequence {translation_sequence} "
                    f"{summarize_text_for_log(translation_result)}"
                )
                _clear_local_ocr_submit_state(app)
                return
            log_debug(
                f"Translation error in sequence {translation_sequence} "
                f"{summarize_text_for_log(translation_result)}"
            )
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
            log_debug(
                f"Translation {translation_sequence} displayed "
                f"{summarize_text_for_log(final_processed_translation)} "
                f"(from OCR batch {ocr_sequence_number})"
            )
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
