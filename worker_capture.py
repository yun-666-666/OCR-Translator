"""Capture backend, runtime metrics, and local PaddleOCR helpers."""

import cv2
import hashlib
import math
import queue
import sys
import time
import tkinter as tk
import traceback

import numpy as np
from PIL import Image

from ocr_utils import build_capture_signature
from paddle_ocr_backend import (
    PADDLEOCR_MODEL_CODE,
    PaddleOCRSettings,
    prepare_paddleocr_image,
    recognize_subtitle_with_paddleocr,
)

DEFAULT_TRANSLATION_REQUEST_TIMEOUT_SECONDS = 10.0
CAPTURE_SLOW_SECONDS_MSS = 0.050


def _log_debug(message):
    facade = sys.modules.get("worker_threads")
    if facade is not None:
        return facade.log_debug(message)


def _log_debug_coalesced(*args, **kwargs):
    facade = sys.modules.get("worker_threads")
    if facade is not None:
        return facade.log_debug_coalesced(*args, **kwargs)


def _capture_screen_region(*args, **kwargs):
    return sys.modules["worker_threads"].capture_screen_region(*args, **kwargs)


def _clear_ocr_stability_gate_facade(*args, **kwargs):
    return sys.modules["worker_threads"]._clear_ocr_stability_gate(*args, **kwargs)


def _enqueue_ocr_frame_for_model(*args, **kwargs):
    return sys.modules["worker_threads"].enqueue_ocr_frame_for_model(*args, **kwargs)


def _get_api_ocr_capture_load(app, ocr_model):
    """Return active calls, effective capacity, and source saturation state."""
    try:
        if not app.is_api_based_ocr_model(ocr_model):
            return 0, 0, False
    except Exception:
        return 0, 0, False

    facade = sys.modules.get("worker_threads")
    effective_model_getter = getattr(
        facade,
        "_effective_ocr_model_for_frame",
        None,
    )
    effective_model = ocr_model
    if callable(effective_model_getter):
        try:
            effective_model = effective_model_getter(app, ocr_model)
        except Exception:
            effective_model = ocr_model
    try:
        if not app.is_api_based_ocr_model(effective_model):
            return 0, 0, False
    except Exception:
        return 0, 0, False

    limit_getter = getattr(facade, "_api_ocr_concurrency_limit", None)
    if not callable(limit_getter):
        return 0, 0, False
    try:
        effective_limit = max(0, int(limit_getter(app, ocr_model)))
        active_count = len(getattr(app, "active_ocr_calls", ()))
    except (AttributeError, TypeError, ValueError):
        return 0, 0, False
    return (
        active_count,
        effective_limit,
        effective_limit == 0 or active_count >= effective_limit,
    )


def _api_ocr_capture_is_saturated(app, ocr_model):
    """Return whether API OCR capture should pause before taking a frame."""
    return _get_api_ocr_capture_load(app, ocr_model)[2]


def _prepare_paddleocr_image(*args, **kwargs):
    return sys.modules["worker_threads"].prepare_paddleocr_image(*args, **kwargs)


def _recognize_subtitle_with_paddleocr(*args, **kwargs):
    return sys.modules["worker_threads"].recognize_subtitle_with_paddleocr(
        *args,
        **kwargs,
    )


def _log_hot_path_timing(
    event_key,
    message,
    duration_seconds,
    slow_threshold_seconds,
    normal_interval_seconds=5.0,
    slow_interval_seconds=1.0,
):
    """Coalesce normal timing samples separately from slow-path samples."""
    try:
        duration_seconds = max(0.0, float(duration_seconds))
    except (TypeError, ValueError):
        duration_seconds = 0.0
    try:
        slow_threshold_seconds = max(0.0, float(slow_threshold_seconds))
    except (TypeError, ValueError):
        slow_threshold_seconds = 0.0

    is_slow = duration_seconds >= slow_threshold_seconds
    channel = "slow" if is_slow else "normal"
    interval_seconds = (
        slow_interval_seconds if is_slow else normal_interval_seconds
    )
    prepared_message = f"SLOW: {message}" if is_slow else message
    return _log_debug_coalesced(
        (event_key, channel),
        prepared_message,
        interval_seconds=interval_seconds,
    )


def _request_snapshot_timeout_seconds(request_snapshot, latency_mode):
    normalized_mode = str(latency_mode or "").strip().lower()
    if normalized_mode in {"stream", "race"}:
        return DEFAULT_TRANSLATION_REQUEST_TIMEOUT_SECONDS
    if not isinstance(request_snapshot, dict):
        return DEFAULT_TRANSLATION_REQUEST_TIMEOUT_SECONDS
    try:
        timeout_seconds = float(request_snapshot.get("timeout_seconds"))
    except (TypeError, ValueError):
        return DEFAULT_TRANSLATION_REQUEST_TIMEOUT_SECONDS
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0.0:
        return DEFAULT_TRANSLATION_REQUEST_TIMEOUT_SECONDS
    return min(
        timeout_seconds,
        DEFAULT_TRANSLATION_REQUEST_TIMEOUT_SECONDS,
    )


def _log_paddle_ocr_route():
    return _log_debug_coalesced(
        "ocr-routing-paddle",
        "WT: OCR routing to PaddleOCR PP-OCRv6",
        interval_seconds=5.0,
    )


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


def _advance_local_capture_signature(
    last_signature,
    repeat_count,
    current_signature,
):
    """Return updated local-frame state and whether to enqueue this frame."""
    if current_signature != last_signature:
        return current_signature, 0, True

    repeat_count = max(0, int(repeat_count or 0)) + 1
    return current_signature, repeat_count, repeat_count <= 1


def _get_api_ocr_cache_model_key(app, provider_name):
    provider_key = str(provider_name or '').strip().lower()
    if provider_key != 'custom_ai' or not hasattr(app, 'custom_ai_profiles'):
        return provider_key

    try:
        profile = app.custom_ai_profiles.get_active_profile("ocr")
    except Exception as e:
        _log_debug(f"Could not resolve active OCR profile for API OCR cache key: {type(e).__name__} - {e}")
        return provider_key

    if not profile:
        return f"{provider_key}|ocr_profile=<missing>"

    profile_id = str(profile.get("id") or "").strip()
    base_url = str(profile.get("base_url") or "").strip().rstrip("/")
    model = str(profile.get("model") or "").strip()
    return f"{provider_key}|ocr_profile={profile_id}|{base_url}|{model}"


def _read_app_var(app, attr, default=None):
    var = getattr(app, attr, None)
    getter = getattr(var, "get", None)
    if callable(getter):
        try:
            return getter()
        except Exception:
            return default
    return default


def _coerce_float(value, default, min_value=None, max_value=None):
    try:
        coerced = float(value)
    except (TypeError, ValueError):
        coerced = float(default)
    if min_value is not None:
        coerced = max(float(min_value), coerced)
    if max_value is not None:
        coerced = min(float(max_value), coerced)
    return coerced


def _coerce_int(value, default, min_value=None, max_value=None):
    try:
        coerced = int(value)
    except (TypeError, ValueError):
        coerced = int(default)
    if min_value is not None:
        coerced = max(int(min_value), coerced)
    if max_value is not None:
        coerced = min(int(max_value), coerced)
    return coerced


def _coerce_bool(value, default=False):
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def get_paddleocr_settings_from_app(app):
    return PaddleOCRSettings(
        source_dir=str(_read_app_var(app, "paddleocr_source_dir_var", "PaddleOCR-3.7.0") or "PaddleOCR-3.7.0"),
        lang=str(_read_app_var(app, "paddleocr_lang_var", "en") or "en"),
        ocr_version=str(_read_app_var(app, "paddleocr_ocr_version_var", "PP-OCRv6") or "PP-OCRv6"),
        model_size=str(_read_app_var(app, "paddleocr_model_size_var", "tiny") or "tiny"),
        device=str(_read_app_var(app, "paddleocr_device_var", "cpu") or "cpu"),
        min_score=_coerce_float(_read_app_var(app, "paddleocr_min_score_var", "0.45"), 0.45, 0.0, 1.0),
        upscale=_coerce_float(_read_app_var(app, "paddleocr_upscale_var", "1.0"), 1.0, 1.0, 4.0),
        text_det_limit_side_len=_coerce_int(
            _read_app_var(app, "paddleocr_text_det_limit_side_len_var", "960"),
            960,
            128,
            4096,
        ),
        text_det_limit_type=str(_read_app_var(app, "paddleocr_text_det_limit_type_var", "max") or "max"),
        use_textline_orientation=_coerce_bool(
            _read_app_var(app, "paddleocr_use_textline_orientation_var", False),
            False,
        ),
    )


def get_paddleocr_ocr_cache_mode_key(app):
    settings = get_paddleocr_settings_from_app(app)
    try:
        keep_linebreaks = bool(app.keep_linebreaks_var.get())
    except Exception:
        keep_linebreaks = False
    return (
        f"paddleocr|version={settings.ocr_version}"
        f"|size={settings.model_size}"
        f"|device={settings.device}"
        f"|min_score={settings.min_score}"
        f"|upscale={settings.upscale}"
        f"|det_limit={settings.text_det_limit_side_len}"
        f"|det_limit_type={settings.text_det_limit_type}"
        f"|orientation={settings.use_textline_orientation}"
        f"|keep_linebreaks={keep_linebreaks}"
    )


def _pil_to_debug_bgr(pil_image):
    rgb = np.array(pil_image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def process_local_ocr_frame(
    app,
    screenshot_pil,
    ocr_model,
):
    if ocr_model == PADDLEOCR_MODEL_CODE:
        settings = get_paddleocr_settings_from_app(app)
        wait_for_prewarm = getattr(
            app,
            "wait_for_paddleocr_prewarm",
            None,
        )
        if callable(wait_for_prewarm):
            wait_for_prewarm(settings, timeout=20.0)
        try:
            keep_linebreaks = bool(app.keep_linebreaks_var.get())
        except Exception:
            keep_linebreaks = False
        ocr_cleaned_text, _lines = _recognize_subtitle_with_paddleocr(
            screenshot_pil,
            settings,
            keep_linebreaks=keep_linebreaks,
        )
        preview_pil = _prepare_paddleocr_image(screenshot_pil, settings)
        return ocr_cleaned_text, _pil_to_debug_bgr(preview_pil), "PaddleOCR"

    raise ValueError(f"Unsupported local OCR model: {ocr_model}")


def run_capture_thread(app):
    _log_debug("WT: Capture thread started.")
    last_cap_time = 0.0
    last_cap_signature = None
    last_capture_geometry_signature = None
    min_interval = 0.05  # 50ms minimum safety floor - user can control via Settings tab
    duplicate_capture_count = 0
    current_scan_interval_sec = min_interval # Initialize

    while app.is_running:
        now = time.monotonic()
        try:
            # Update adaptive scan interval based on OCR load
            app.update_adaptive_scan_interval()

            # Use dynamic interval instead of static setting
            scan_interval_ms = app.current_scan_interval  # 鈫?Use adaptive value
            base_scan_interval = max(min_interval, scan_interval_ms / 1000.0)

            # DEBUG: Log when using adaptive interval (every 20 seconds to avoid spam)
            if not hasattr(app, '_last_adaptive_debug') or now - app._last_adaptive_debug > 20.0:
                app._last_adaptive_debug = now
                _log_debug(f"ADAPTIVE: Capture thread using scan interval: {scan_interval_ms}ms (base: {app.scan_interval_var.get()}ms)")

            ocr_model = app.get_ocr_model_setting()
            # Use a simpler, more adaptive logic for all API-based OCR models
            if app.is_api_based_ocr_model(ocr_model):
                (
                    active_ocr_count,
                    effective_ocr_limit,
                    api_ocr_saturated,
                ) = _get_api_ocr_capture_load(app, ocr_model)
                if api_ocr_saturated:
                    _increment_metric(app, "api_ocr_capture_backpressure_skip")
                    _log_debug_coalesced(
                        ("api-ocr-capture-backpressure", ocr_model),
                        "CAPTURE: API OCR saturated "
                        f"provider={ocr_model} active={active_ocr_count} "
                        f"limit={effective_ocr_limit}; waiting "
                        f"{scan_interval_ms}ms before capturing a fresh frame",
                        interval_seconds=5.0,
                    )
                    current_scan_interval_sec = base_scan_interval
                    slept_time = 0.0
                    while slept_time < base_scan_interval and app.is_running:
                        chunk = min(0.05, base_scan_interval - slept_time)
                        time.sleep(chunk)
                        slept_time += chunk
                    continue

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
                # Adaptive logic for local OCR queue pressure.
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
            geometry_signature = (x1, y1, width, height, "mss", ocr_model)
            if geometry_signature != last_capture_geometry_signature:
                _log_debug(f"CAPTURE: source context changed to {geometry_signature}; clearing stale OCR state")
                last_capture_geometry_signature = geometry_signature
                last_cap_signature = None
                duplicate_capture_count = 0
                app.last_processed_subtitle = None
                app.previous_text = ""
                app.text_stability_counter = 0
                _clear_ocr_stability_gate_facade(app, "source context changed")
                if hasattr(app, 'ocr_frame_cache'):
                    app.ocr_frame_cache.clear()
                try:
                    while True:
                        app.ocr_queue.get_nowait()
                except queue.Empty:
                    pass

            screenshot = _capture_screen_region((x1, y1, width, height))
            capture_duration = time.monotonic() - capture_moment
            last_cap_time = capture_moment
            actual_capture_backend = "mss"
            _log_hot_path_timing(
                ("capture-timing", actual_capture_backend),
                f"LATENCY: capture backend={actual_capture_backend} "
                f"region={width}x{height} took {capture_duration:.3f}s",
                capture_duration,
                CAPTURE_SLOW_SECONDS_MSS,
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
            else:
                (
                    last_cap_signature,
                    duplicate_capture_count,
                    should_enqueue,
                ) = _advance_local_capture_signature(
                    last_cap_signature,
                    duplicate_capture_count,
                    capture_signature,
                )
                if not should_enqueue:
                    _increment_metric(app, "capture_exact_duplicate_skip")
                    time.sleep(min(0.1, current_scan_interval_sec * 0.5))
                    continue

            try:
                _enqueue_ocr_frame_for_model(app, screenshot, ocr_model)
            except queue.Full:
                _refresh_ocr_queue_metric(app)
                pass # Skip frame if queue is full
            except Exception as q_err_wt_put:
                _log_debug(f"WT: Capture: Error putting to OCR queue - {type(q_err_wt_put).__name__}: {q_err_wt_put}")

        except tk.TclError:
            _log_debug("WT: Capture thread TclError (UI likely gone).")
            if not app.is_running: break
            time.sleep(0.1)
        except Exception as loop_err_wt_capture:
            _log_debug(f"WT: Capture thread error: {type(loop_err_wt_capture).__name__} - {loop_err_wt_capture}\n{traceback.format_exc()}")
            sleep_after_error = current_scan_interval_sec if 'current_scan_interval_sec' in locals() else 0.5
            time.sleep(max(sleep_after_error, 0.5))
    _log_debug("WT: Capture thread finished.")
