# worker_threads.py (Complete, Corrected File)

import tkinter as tk # For tk.Toplevel type check in capture_thread
from difflib import SequenceMatcher
import time
import queue
import numpy as np
import cv2
import hashlib
import math
import re
import threading
import traceback
from datetime import datetime

from logger import log_debug, log_debug_coalesced, summarize_text_for_log
from ai_optimization import ai_ocr_route_metric_name
from ocr_utils import (
    capture_screen_region,
    build_capture_signature, build_ocr_frame_cache_key,
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
from app_capture_ocr import CUSTOM_AI_OCR_CONCURRENCY_LIMIT


DEFAULT_TRANSLATION_SUPERSEDE_AFTER_SECONDS = 1.5
DEFAULT_TRANSLATION_REQUEST_TIMEOUT_SECONDS = 10.0
MAX_SUPERSEDED_TRANSLATION_CONCURRENCY = 2
OCR_STABILITY_GATE_MIN_WAIT_SECONDS = 0.12
OCR_STABILITY_GATE_MAX_WAIT_SECONDS = 0.25
OCR_STABILITY_GATE_SUSPICIOUS_SHORT_LENGTH = 12
OCR_STABILITY_GATE_NOISE_RATIO = 0.35
CAPTURE_SLOW_SECONDS_MSS = 0.050
OCR_CACHE_HIT_SLOW_SECONDS = 0.050
PADDLE_OCR_SLOW_SECONDS = 0.500
API_OCR_REPEAT_BACKOFF_SECONDS = 0.75
from worker_capture import (
    CaptureUISnapshot,
    run_capture_thread,
    _api_ocr_capture_is_saturated,
    _log_hot_path_timing,
    _request_snapshot_timeout_seconds,
    _log_paddle_ocr_route,
    _runtime_metrics,
    _record_metric_timing,
    _increment_metric,
    _set_metric_gauge,
    _safe_queue_size,
    _refresh_translation_metric_gauges,
    _refresh_ocr_queue_metric,
    _get_screenshot_frame_hash,
    _advance_local_capture_signature,
    _get_api_ocr_cache_model_key,
    _coerce_float,
    _coerce_int,
    _coerce_bool,
    get_paddleocr_settings_from_app,
    get_paddleocr_ocr_cache_mode_key,
    get_paddleocr_ocr_cache_mode_key_from_settings,
    get_capture_ui_snapshot,
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
    _schedule_ui_callback,
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
    reset_translation_failure_visibility,
    note_transient_translation_failure,
    clear_transient_translation_failure_status,
    _flush_pending_translation_request,
    _queue_pending_translation_request,
    _expedite_pending_translation_request,
    _apply_translation_profile_refresh,
    refresh_translation_after_profile_change,
    _submit_async_translation_request,
    _coalesce_matching_pending_translation_request,
    _build_streaming_display_callback,
)




def _custom_ai_ocr_cooldown_seconds(app):
    """Return the active Custom AI OCR profile cooldown without breaking OCR."""
    try:
        handler = getattr(app, "translation_handler", None)
        getter = getattr(handler, "get_active_custom_ai_ocr_cooldown_seconds", None)
        if not callable(getter):
            return 0.0
        return max(0.0, float(getter() or 0.0))
    except Exception as cooldown_error:
        log_debug_coalesced(
            "custom-ai-ocr-cooldown-read-error",
            "WT: Could not read Custom AI OCR cooldown; keeping selected OCR "
            f"provider ({type(cooldown_error).__name__})",
            interval_seconds=10.0,
        )
        return 0.0


def _effective_ocr_model_for_frame(app, selected_model):
    """Use PaddleOCR for this frame while the selected AI OCR profile cools down."""
    if selected_model != "custom_ai":
        return selected_model

    cooldown_seconds = _custom_ai_ocr_cooldown_seconds(app)
    if cooldown_seconds <= 0.0:
        return selected_model

    log_debug_coalesced(
        "custom-ai-ocr-paddle-fallback",
        "WT: Custom AI OCR is cooling down for about "
        f"{math.ceil(cooldown_seconds)}s; temporarily using PaddleOCR",
        interval_seconds=5.0,
    )
    return PADDLEOCR_MODEL_CODE


def _api_ocr_concurrency_limit(app, provider_name):
    """Return the same provider-aware limit used by adaptive capture."""
    getter = getattr(app, "get_effective_ocr_concurrency_limit", None)
    if callable(getter):
        try:
            return max(0, int(getter(provider_name)))
        except (AttributeError, OverflowError, TypeError, ValueError):
            pass
    try:
        configured_limit = max(0, int(app.max_concurrent_ocr_calls))
    except (AttributeError, OverflowError, TypeError, ValueError):
        configured_limit = 1
    if provider_name == "custom_ai":
        return min(configured_limit, CUSTOM_AI_OCR_CONCURRENCY_LIMIT)
    return configured_limit


def run_ocr_thread(app):
    log_debug("WT: OCR thread started.")

    last_lang_check = time.monotonic()
    last_ocr_proc_time = 0
    min_ocr_interval = 0.1
    similar_texts_count = 0
    prev_ocr_text = ""

    while app.is_running:
        now = time.monotonic()
        try:
            worker_snapshot = get_capture_ui_snapshot(
                app,
                allow_unpublished_build=False,
            )
            if not isinstance(worker_snapshot, CaptureUISnapshot):
                time.sleep(0.05)
                continue
            if now - last_lang_check > 5.0:
                last_lang_check = now

            selected_ocr_model = worker_snapshot.ocr_model
            ocr_model = _effective_ocr_model_for_frame(app, selected_ocr_model)

            # No artificial delay for API-based OCR
            if not (
                worker_snapshot.is_api_based
                and ocr_model == selected_ocr_model
            ):
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

            frame_snapshot = getattr(
                screenshot_pil,
                "_gct_capture_snapshot",
                worker_snapshot,
            )
            if not isinstance(frame_snapshot, CaptureUISnapshot):
                frame_snapshot = worker_snapshot
            selected_ocr_model = frame_snapshot.ocr_model
            ocr_model = _effective_ocr_model_for_frame(app, selected_ocr_model)
            is_api_ocr = (
                frame_snapshot.is_api_based
                and ocr_model == selected_ocr_model
            )

            ocr_proc_start_time = time.monotonic()
            last_ocr_proc_time = ocr_proc_start_time
            app.last_screenshot = screenshot_pil
            processed_cv_img = None

            frame_hash = _get_screenshot_frame_hash(screenshot_pil)

            region_origin = getattr(screenshot_pil, '_gct_region_origin', (0, 0))
            ocr_cache_key = None
            if hasattr(app, 'ocr_frame_cache') and not is_api_ocr:
                if ocr_model == PADDLEOCR_MODEL_CODE:
                    settings = frame_snapshot.paddleocr_settings
                    if settings is None:
                        raise RuntimeError(
                            "Captured frame is missing PaddleOCR settings"
                        )
                    cache_lang = settings.lang
                    cache_mode_key = (
                        get_paddleocr_ocr_cache_mode_key_from_settings(
                            settings,
                            frame_snapshot.keep_linebreaks,
                        )
                    )
                else:
                    ocr_model = PADDLEOCR_MODEL_CODE
                    settings = frame_snapshot.paddleocr_settings
                    if settings is None:
                        raise RuntimeError(
                            "Captured frame is missing PaddleOCR settings"
                        )
                    cache_lang = settings.lang
                    cache_mode_key = (
                        get_paddleocr_ocr_cache_mode_key_from_settings(
                            settings,
                            frame_snapshot.keep_linebreaks,
                        )
                    )
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
                    if frame_snapshot.ocr_debugging and app.last_processed_image is not None:
                        _schedule_ui_callback(app, app.update_debug_display, screenshot_pil, app.last_processed_image, ocr_cleaned_text)
                    # Jump to shared post-OCR routing below.
                    goto_post_ocr = True
                else:
                    goto_post_ocr = False
            else:
                goto_post_ocr = False

            if goto_post_ocr:
                pass
            # ==================== OCR MODEL ROUTING ====================
            elif is_api_ocr:
                run_api_ocr(
                    app,
                    screenshot_pil,
                    capture_snapshot=frame_snapshot,
                )
                continue # Skip to the next loop iteration

            elif ocr_model == PADDLEOCR_MODEL_CODE:
                _log_paddle_ocr_route()

            else:
                log_debug(f"WT: OCR: Unknown OCR model '{ocr_model}', falling back to PaddleOCR")
                ocr_model = PADDLEOCR_MODEL_CODE

            if not goto_post_ocr:
                # ==================== LOCAL OCR PROCESSING ====================
                if frame_snapshot.ocr_debugging:
                    _schedule_ui_callback(app, app.update_debug_display, screenshot_pil, _pil_to_debug_bgr(screenshot_pil), "Processing...")

                ocr_cleaned_text, processed_cv_img, engine_label = process_local_ocr_frame(
                    app,
                    screenshot_pil,
                    ocr_model,
                    capture_snapshot=frame_snapshot,
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
                _record_metric_timing(
                    app,
                    "local_ocr_duration",
                    ocr_duration,
                )


            if frame_snapshot.ocr_debugging and processed_cv_img is not None:
                _schedule_ui_callback(app, app.update_debug_display, screenshot_pil, processed_cv_img, ocr_cleaned_text)

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

def run_api_ocr(app, screenshot_pil, capture_snapshot=None):
    """Start API-based OCR processing for a screenshot using the currently selected provider."""
    try:
        ocr_start_time = time.monotonic()
        keep_linebreaks = None
        if isinstance(capture_snapshot, CaptureUISnapshot):
            provider_name = capture_snapshot.ocr_model
            source_lang = capture_snapshot.source_lang
            keep_linebreaks = bool(capture_snapshot.keep_linebreaks)
        else:
            # Compatibility for direct callers outside the OCR worker.  The
            # production worker always supplies the immutable frame snapshot.
            provider_name = app.get_ocr_model_setting()
            source_lang = None

        if not hasattr(app, 'batch_sequence_counter'):
            app.batch_sequence_counter = 0

        if source_lang:
            pass
        elif provider_name == 'custom_ai':
            source_lang = getattr(app, 'custom_source_lang', None) or app.source_lang_var.get()
        else:
            active_translation_model = app.translation_model_var.get()
            if app.is_gemini_model(active_translation_model):
                source_lang = getattr(app, 'gemini_source_lang', 'en')
            elif app.is_openai_model(active_translation_model):
                source_lang = getattr(app, 'openai_source_lang', 'en')
            else:
                source_lang = app.source_lang_var.get()

        # Freeze keep_linebreaks once for cache identity and provider request.
        # Snapshot frames always freeze. Compatibility callers without a snapshot
        # freeze the current live var when present; if the var is missing, leave
        # None so the mode key keeps the historical "api" shape.
        if keep_linebreaks is None:
            keep_linebreaks_var = getattr(app, "keep_linebreaks_var", None)
            if keep_linebreaks_var is not None:
                try:
                    keep_linebreaks = bool(keep_linebreaks_var.get())
                except Exception:
                    keep_linebreaks = False

        ocr_cache_key = None
        image_decision = None
        image_decision_getter = getattr(
            app,
            "get_ai_ocr_image_decision",
            None,
        )
        if callable(image_decision_getter):
            try:
                image_decision = image_decision_getter(
                    image_size=screenshot_pil.size
                )
            except Exception as e:
                log_debug(
                    "Could not resolve API OCR image contract: "
                    f"{type(e).__name__} - {e}"
                )
        if hasattr(app, 'ocr_frame_cache'):
            frame_hash = _get_screenshot_frame_hash(screenshot_pil)
            region_origin = getattr(screenshot_pil, '_gct_region_origin', (0, 0))
            cache_model_key = _get_api_ocr_cache_model_key(app, provider_name)
            ocr_cache_key = build_ocr_frame_cache_key(
                frame_hash,
                cache_model_key,
                source_lang,
                _get_api_ocr_cache_mode_key(
                    app,
                    provider_name,
                    image_size=screenshot_pil.size,
                    image_decision=image_decision,
                    keep_linebreaks=keep_linebreaks,
                ),
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
                _schedule_ui_callback(
                    app,
                    process_api_ocr_response,
                    app,
                    cached_ocr_text,
                    sequence_number,
                    source_lang,
                    provider_name,
                    ocr_cache_key,
                )
                return

        repeat_scope = tuple(ocr_cache_key[1:]) if ocr_cache_key else None
        try:
            repeat_until = float(
                getattr(app, "api_ocr_repeat_backoff_until_monotonic", 0.0)
                or 0.0
            )
        except (TypeError, ValueError):
            repeat_until = 0.0
        if (
            repeat_scope
            and repeat_scope == getattr(app, "api_ocr_repeat_backoff_scope", None)
            and time.monotonic() < repeat_until
        ):
            _increment_metric(app, "api_ocr_repeat_backoff_skip")
            log_debug_coalesced(
                ("api-ocr-repeat-backoff", provider_name),
                "LATENCY: API OCR repeat backoff skipped a remote request "
                f"provider={provider_name}",
                interval_seconds=5.0,
            )
            return

        concurrency_limit = _api_ocr_concurrency_limit(app, provider_name)
        if len(app.active_ocr_calls) >= concurrency_limit:
            _set_metric_gauge(app, "active_ocr_calls", len(app.active_ocr_calls))
            log_debug(
                f"Max concurrent OCR calls ({concurrency_limit}) reached, "
                f"skipping {provider_name} OCR before image conversion"
            )
            return

        encoded_image = None
        metadata_encoder = getattr(app, 'convert_to_api_ocr_image', None)
        if callable(metadata_encoder):
            if image_decision is not None:
                encoded_image = metadata_encoder(
                    screenshot_pil,
                    decision=image_decision,
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

        if not encoded_image or not getattr(encoded_image, 'data', None):
            log_debug(f"Failed to convert image for {provider_name} OCR")
            return
        image_data = encoded_image.data
        image_mime_type = getattr(encoded_image, 'mime_type', 'image/webp')
        image_format = getattr(encoded_image, 'image_format', 'webp')
        image_detail = getattr(encoded_image, 'image_detail', 'auto')
        route_metric_name = None
        if provider_name == "custom_ai":
            try:
                active_profile = app.custom_ai_profiles.get_active_profile(
                    "ocr"
                )
                if active_profile:
                    route_metric_name = ai_ocr_route_metric_name(
                        active_profile
                    )
            except Exception:
                pass

        app.batch_sequence_counter += 1
        sequence_number = app.batch_sequence_counter

        app.active_ocr_calls.add(sequence_number)
        _set_metric_gauge(app, "active_ocr_calls", len(app.active_ocr_calls))
        try:
            app.ocr_thread_pool.submit(
                process_api_ocr_async,
                app,
                image_data,
                source_lang,
                sequence_number,
                provider_name,
                ocr_cache_key,
                image_mime_type,
                image_detail,
                image_format,
                route_metric_name,
                keep_linebreaks,
            )
        except Exception:
            app.active_ocr_calls.discard(sequence_number)
            raise
        log_debug(f"Started {provider_name} OCR batch {sequence_number} (active calls: {len(app.active_ocr_calls)})")

    except Exception as e:
        log_debug(f"Error starting API OCR batch: {type(e).__name__} - {e}")

def process_api_ocr_async(
    app,
    image_data,
    source_lang,
    sequence_number,
    provider_name,
    ocr_cache_key=None,
    image_mime_type="image/webp",
    image_detail="auto",
    image_format="webp",
    route_metric_name=None,
    keep_linebreaks=None,
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
            image_detail=image_detail,
            image_format=image_format,
            keep_linebreaks=keep_linebreaks,
        )
        ocr_duration = time.monotonic() - ocr_start_time
        _record_metric_timing(app, "ocr_duration", ocr_duration)
        _record_metric_timing(app, "api_ocr_duration", ocr_duration)
        if route_metric_name:
            _record_metric_timing(app, route_metric_name, ocr_duration)

        log_debug(
            f"{provider_name} OCR batch {sequence_number} completed, "
            f"scheduling response {summarize_text_for_log(ocr_result)}"
        )
        _schedule_ui_callback(app, process_api_ocr_response, app, ocr_result, sequence_number, source_lang, provider_name, ocr_cache_key)

    except Exception as e:
        log_debug(f"Error in async {provider_name} OCR batch {sequence_number}: {type(e).__name__} - {e}")
        error_msg = f"<e>: OCR batch {sequence_number} error: {str(e)}"
        _schedule_ui_callback(app, process_api_ocr_response, app, error_msg, sequence_number, source_lang, provider_name, ocr_cache_key)

    finally:
        app.active_ocr_calls.discard(sequence_number)
        _set_metric_gauge(app, "active_ocr_calls", len(app.active_ocr_calls))
        log_debug(f"{provider_name} OCR batch {sequence_number} finished (active calls: {len(app.active_ocr_calls)})")

def process_api_ocr_response(app, ocr_result, sequence_number, source_lang, provider_name, ocr_cache_key=None):
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
            if (
                provider_name == "custom_ai"
                and _custom_ai_ocr_cooldown_seconds(app) > 0.0
            ):
                log_debug_coalesced(
                    "custom-ai-ocr-error-hidden-during-fallback",
                    "WT: Preserving the last useful translation while PaddleOCR "
                    "covers a Custom AI OCR cooldown",
                    interval_seconds=5.0,
                )
                app.last_displayed_batch_sequence = sequence_number
                return
            visible_error = ocr_result[len("<e>:"):].strip() or ocr_result
            app.update_translation_text(f"OCR Error:\n{visible_error}")
            app.last_displayed_batch_sequence = sequence_number
            return

        if ocr_result == "<EMPTY>":
            app.handle_empty_ocr_result()
            app.last_displayed_batch_sequence = sequence_number
            return

        if hasattr(app, 'last_processed_subtitle') and ocr_result == app.last_processed_subtitle:
            if ocr_cache_key is not None:
                app.api_ocr_repeat_backoff_scope = tuple(ocr_cache_key[1:])
                app.api_ocr_repeat_backoff_until_monotonic = (
                    time.monotonic() + API_OCR_REPEAT_BACKOFF_SECONDS
                )
                _increment_metric(app, "api_ocr_repeat_backoff_armed")
            app.reset_clear_timeout()
            log_debug(
                "Keeping existing translation for successive identical "
                f"{provider_name} OCR {summarize_text_for_log(ocr_result)}"
            )
            app.last_displayed_batch_sequence = sequence_number
            return

        app.api_ocr_repeat_backoff_scope = None
        app.api_ocr_repeat_backoff_until_monotonic = 0.0
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
            supersede_after = _get_translation_supersede_after_seconds(
                app,
                request_snapshot,
            )
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
            latency_mode_getter = getattr(app, "get_custom_ai_latency_mode", None)
            if callable(latency_mode_getter):
                try:
                    latency_mode = latency_mode_getter()
                except Exception:
                    latency_mode = ""
            else:
                latency_mode_var = getattr(app, 'custom_ai_latency_mode_var', None)
                try:
                    latency_mode = (
                        latency_mode_var.get()
                        if latency_mode_var is not None
                        else ""
                    )
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

        _schedule_ui_callback(app, process_translation_response, app, translation_result, translation_sequence, text_to_translate, ocr_sequence_number)

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
        _schedule_ui_callback(app, process_translation_response, app, error_msg, translation_sequence, text_to_translate, ocr_sequence_number)

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

        latest_started_sequence = getattr(
            app,
            "latest_translation_sequence_started",
            translation_sequence,
        )
        if translation_sequence < latest_started_sequence:
            _increment_metric(app, "stale_response_discarded")
            log_debug(
                f"Translation {translation_sequence}: newer request "
                f"{latest_started_sequence} has already started; discarding"
            )
            return

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
                note_transient_translation_failure(app, translation_sequence)
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
            clear_transient_translation_failure_status(
                app,
                translation_sequence=translation_sequence,
                reason="success",
            )
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
