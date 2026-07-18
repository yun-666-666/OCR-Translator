"""Capture backend, runtime metrics, and local PaddleOCR helpers."""

import cv2
import hashlib
import math
import queue
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from typing import Optional, Tuple

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


@dataclass(frozen=True)
class CaptureUISnapshot:
    """Immutable, widget-free capture inputs published by the UI thread."""

    generation: int
    source_geometry: Optional[Tuple[int, int, int, int]]
    ocr_model: str
    scan_interval_ms: int
    base_scan_interval_ms: int
    keep_linebreaks: bool
    is_api_based: bool
    paddleocr_settings: Optional[PaddleOCRSettings] = None


def _next_local_capture_interval(
    base_scan_interval,
    current_scan_interval,
    queue_fullness,
):
    if queue_fullness > 0.7:
        return base_scan_interval * (1 + queue_fullness)
    if queue_fullness > 0.4:
        return base_scan_interval * 1.25
    return max(base_scan_interval, current_scan_interval * 0.95)


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


def _normalize_source_geometry(area):
    if not area:
        return None
    try:
        x1, y1, x2, y2 = [int(value) for value in area[:4]]
    except (TypeError, ValueError, IndexError):
        return None
    width = x2 - x1
    height = y2 - y1
    if width <= 0 or height <= 0:
        return None
    return (x1, y1, x2, y2)


def _read_source_geometry_from_app(app):
    """UI-thread helper: prefer live overlay geometry, fall back to source_area."""
    overlay = getattr(app, "source_overlay", None)
    if overlay is not None:
        exists = getattr(overlay, "winfo_exists", None)
        try:
            if callable(exists) and exists():
                getter = getattr(overlay, "get_geometry", None)
                if callable(getter):
                    geometry = _normalize_source_geometry(getter())
                    if geometry is not None:
                        return geometry
        except Exception:
            pass

    return _normalize_source_geometry(getattr(app, "source_area", None))


def build_capture_ui_snapshot(
    app,
    *,
    source_geometry=None,
    ocr_model=None,
    scan_interval_ms=None,
    base_scan_interval_ms=None,
    keep_linebreaks=None,
    generation=None,
):
    """Build an immutable capture snapshot from plain Python values."""
    if generation is None:
        try:
            generation = int(getattr(app, "capture_ui_generation", 0) or 0)
        except (TypeError, ValueError):
            generation = 0

    if source_geometry is None:
        source_geometry = _normalize_source_geometry(
            getattr(app, "source_area", None)
        )

    if ocr_model is None:
        getter = getattr(app, "get_ocr_model_setting", None)
        if callable(getter):
            try:
                ocr_model = getter()
            except Exception:
                ocr_model = PADDLEOCR_MODEL_CODE
        else:
            ocr_model = getattr(app, "ocr_model", PADDLEOCR_MODEL_CODE)
    ocr_model = str(ocr_model or PADDLEOCR_MODEL_CODE)

    if scan_interval_ms is None:
        scan_interval_ms = getattr(app, "current_scan_interval", None)
    if scan_interval_ms is None:
        scan_interval_ms = getattr(app, "base_scan_interval", 100)
    scan_interval_ms = _coerce_int(scan_interval_ms, 100, 1, 600000)

    if base_scan_interval_ms is None:
        base_scan_interval_ms = getattr(app, "base_scan_interval", scan_interval_ms)
    base_scan_interval_ms = _coerce_int(
        base_scan_interval_ms,
        scan_interval_ms,
        1,
        600000,
    )

    if keep_linebreaks is None:
        keep_linebreaks = _coerce_bool(
            _read_app_var(app, "keep_linebreaks_var", False),
            False,
        )
    else:
        keep_linebreaks = bool(keep_linebreaks)

    api_checker = getattr(app, "is_api_based_ocr_model", None)
    if callable(api_checker):
        try:
            is_api_based = bool(api_checker(ocr_model))
        except TypeError:
            try:
                is_api_based = bool(api_checker())
            except Exception:
                is_api_based = ocr_model != PADDLEOCR_MODEL_CODE
        except Exception:
            is_api_based = ocr_model != PADDLEOCR_MODEL_CODE
    else:
        is_api_based = ocr_model != PADDLEOCR_MODEL_CODE

    paddleocr_settings = None
    if not is_api_based and ocr_model == PADDLEOCR_MODEL_CODE:
        paddleocr_settings = get_paddleocr_settings_from_app(app)

    return CaptureUISnapshot(
        generation=int(generation),
        source_geometry=source_geometry,
        ocr_model=ocr_model,
        scan_interval_ms=scan_interval_ms,
        base_scan_interval_ms=base_scan_interval_ms,
        keep_linebreaks=keep_linebreaks,
        is_api_based=is_api_based,
        paddleocr_settings=paddleocr_settings,
    )


def publish_capture_ui_snapshot(app, *, bump_generation=False, reason=""):
    """UI-thread publisher for the immutable capture snapshot."""
    if not hasattr(app, "_capture_ui_snapshot_lock"):
        app._capture_ui_snapshot_lock = threading.Lock()
    if not hasattr(app, "capture_ui_generation"):
        app.capture_ui_generation = 0

    source_geometry = _read_source_geometry_from_app(app)
    if source_geometry is not None:
        app.source_area = list(source_geometry)

    with app._capture_ui_snapshot_lock:
        if bump_generation:
            app.capture_ui_generation = int(app.capture_ui_generation or 0) + 1
        snapshot = build_capture_ui_snapshot(
            app,
            source_geometry=source_geometry,
            generation=app.capture_ui_generation,
        )
        app.capture_ui_snapshot = snapshot

    if reason:
        _log_debug(
            "CAPTURE: published UI snapshot "
            f"generation={snapshot.generation} "
            f"ocr_model={snapshot.ocr_model} "
            f"geometry={snapshot.source_geometry} "
            f"reason={reason}"
        )
    return snapshot


def get_capture_ui_snapshot(app, *, allow_unpublished_build=True):
    """Worker-safe reader for the latest immutable capture snapshot."""
    lock = getattr(app, "_capture_ui_snapshot_lock", None)
    if lock is not None:
        with lock:
            snapshot = getattr(app, "capture_ui_snapshot", None)
            if isinstance(snapshot, CaptureUISnapshot):
                return snapshot

    snapshot = getattr(app, "capture_ui_snapshot", None)
    if isinstance(snapshot, CaptureUISnapshot):
        return snapshot
    if not allow_unpublished_build:
        return None
    return build_capture_ui_snapshot(app)


def get_paddleocr_ocr_cache_mode_key(app):
    settings = get_paddleocr_settings_from_app(app)
    keep_linebreaks = _coerce_bool(
        _read_app_var(app, "keep_linebreaks_var", False),
        False,
    )
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
    capture_snapshot=None,
):
    if ocr_model == PADDLEOCR_MODEL_CODE:
        if (
            isinstance(capture_snapshot, CaptureUISnapshot)
            and capture_snapshot.paddleocr_settings is not None
        ):
            settings = capture_snapshot.paddleocr_settings
            keep_linebreaks = bool(capture_snapshot.keep_linebreaks)
        else:
            settings = get_paddleocr_settings_from_app(app)
            keep_linebreaks = _coerce_bool(
                _read_app_var(app, "keep_linebreaks_var", False),
                False,
            )
        wait_for_prewarm = getattr(
            app,
            "wait_for_paddleocr_prewarm",
            None,
        )
        if callable(wait_for_prewarm):
            wait_for_prewarm(settings, timeout=20.0)
        ocr_cleaned_text, _lines = _recognize_subtitle_with_paddleocr(
            screenshot_pil,
            settings,
            keep_linebreaks=keep_linebreaks,
        )
        preview_pil = _prepare_paddleocr_image(screenshot_pil, settings)
        return ocr_cleaned_text, _pil_to_debug_bgr(preview_pil), "PaddleOCR"

    raise ValueError(f"Unsupported local OCR model: {ocr_model}")


def _clear_capture_generation_state(app, reason):
    """Clear OCR state that is bound to an old capture generation."""
    app.last_processed_subtitle = None
    app.previous_text = ""
    app.text_stability_counter = 0
    _clear_ocr_stability_gate_facade(app, reason)
    if hasattr(app, "ocr_frame_cache"):
        try:
            app.ocr_frame_cache.clear()
        except Exception:
            pass
    try:
        while True:
            app.ocr_queue.get_nowait()
    except queue.Empty:
        pass
    except Exception:
        pass
    _refresh_ocr_queue_metric(app)


def run_capture_thread(app):
    _log_debug("WT: Capture thread started.")
    last_cap_time = 0.0
    last_cap_signature = None
    last_capture_context_signature = None
    min_interval = 0.05  # 50ms minimum safety floor - user can control via Settings tab
    duplicate_capture_count = 0
    current_scan_interval_sec = min_interval # Initialize
    # Seed with the currently published generation so startup does not treat the
    # first snapshot as a geometry/settings change that clears OCR state.
    try:
        initial_snapshot = get_capture_ui_snapshot(
            app,
            allow_unpublished_build=False,
        )
        if initial_snapshot is not None:
            last_capture_context_signature = (
                initial_snapshot.generation,
                None,
                None,
                None,
                None,
                None,
                initial_snapshot.ocr_model,
            )
    except Exception:
        last_capture_context_signature = None

    while app.is_running:
        now = time.monotonic()
        try:
            snapshot = get_capture_ui_snapshot(
                app,
                allow_unpublished_build=False,
            )
            if snapshot is None:
                time.sleep(min_interval)
                continue
            scan_interval_ms = max(1, int(snapshot.scan_interval_ms or 100))
            base_scan_interval = max(min_interval, scan_interval_ms / 1000.0)
            ocr_model = snapshot.ocr_model
            is_api_based = bool(snapshot.is_api_based)

            # DEBUG: Log when using adaptive interval (every 20 seconds to avoid spam)
            if not hasattr(app, '_last_adaptive_debug') or now - app._last_adaptive_debug > 20.0:
                app._last_adaptive_debug = now
                _log_debug(
                    "ADAPTIVE: Capture thread using scan interval: "
                    f"{scan_interval_ms}ms "
                    f"(base: {snapshot.base_scan_interval_ms}ms, "
                    f"generation={snapshot.generation})"
                )

            # Use a simpler, more adaptive logic for all API-based OCR models
            if is_api_based:
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
                current_scan_interval_sec = _next_local_capture_interval(
                    base_scan_interval,
                    current_scan_interval_sec,
                    q_fullness,
                )

                if now - last_cap_time < current_scan_interval_sec:
                    sleep_duration = current_scan_interval_sec - (now - last_cap_time)
                    slept_time = 0
                    while slept_time < sleep_duration and app.is_running:
                        chunk = min(0.05, sleep_duration - slept_time)
                        time.sleep(chunk)
                        slept_time += chunk
                    if not app.is_running: break
                    continue

            geometry = snapshot.source_geometry
            if not geometry:
                if app.is_running:
                    time.sleep(max(current_scan_interval_sec, 0.2))
                continue

            x1, y1, x2, y2 = geometry
            width, height = x2 - x1, y2 - y1
            if width <= 0 or height <= 0:
                continue

            capture_moment = time.monotonic()
            context_signature = (
                snapshot.generation,
                x1,
                y1,
                width,
                height,
                "mss",
                ocr_model,
            )
            previous_signature = last_capture_context_signature
            if previous_signature is None or context_signature != previous_signature:
                # Only clear OCR state when generation or geometry/settings change
                # after the worker has already observed a prior context.
                if previous_signature is not None and (
                    previous_signature[0] != context_signature[0]
                    or previous_signature[1:] != context_signature[1:]
                ):
                    _log_debug(
                        "CAPTURE: source context changed to "
                        f"{context_signature}; clearing stale OCR state"
                    )
                    last_cap_signature = None
                    duplicate_capture_count = 0
                    _clear_capture_generation_state(app, "source context changed")
                last_capture_context_signature = context_signature

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
                screenshot._gct_capture_generation = snapshot.generation
            except Exception:
                pass

            if is_api_based:
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

        except Exception as loop_err_wt_capture:
            _log_debug(f"WT: Capture thread error: {type(loop_err_wt_capture).__name__} - {loop_err_wt_capture}\n{traceback.format_exc()}")
            sleep_after_error = current_scan_interval_sec if 'current_scan_interval_sec' in locals() else 0.5
            time.sleep(max(sleep_after_error, 0.5))
    _log_debug("WT: Capture thread finished.")
