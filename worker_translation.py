"""Pending translation scheduling and streaming display helpers."""

import math
import sys
import threading
import time
import tkinter as tk

from logger import summarize_text_for_log
from translation_utils import post_process_translation_text
from worker_capture import _increment_metric, _refresh_translation_metric_gauges

DEFAULT_TRANSLATION_SUPERSEDE_AFTER_SECONDS = 1.5
ROUTE_SUPERSEDE_MIN_SAMPLES = 8
ROUTE_SUPERSEDE_LEARNING_SECONDS = 3.0
ROUTE_SUPERSEDE_P90_FRACTION = 0.5
ROUTE_SUPERSEDE_MAX_SECONDS = 4.0


def _facade():
    return sys.modules.get("worker_threads")


def _log_debug(message):
    facade = _facade()
    if facade is not None:
        return facade.log_debug(message)


def _log_debug_coalesced(*args, **kwargs):
    facade = _facade()
    if facade is not None:
        return facade.log_debug_coalesced(*args, **kwargs)


def _start_async_translation(*args, **kwargs):
    return _facade().start_async_translation(*args, **kwargs)


def _process_translation_async(*args, **kwargs):
    return _facade().process_translation_async(*args, **kwargs)


def _schedule_ui_callback(app, callback, *args):
    """Schedule a worker result only while the Tk root is still usable."""
    if getattr(app, "_app_is_closing", False):
        return False
    if hasattr(app, "is_running") and not bool(app.is_running):
        return False
    root = getattr(app, "root", None)
    if root is None:
        return False
    exists = getattr(root, "winfo_exists", None)
    try:
        if callable(exists) and not bool(exists()):
            return False
        root.after(0, callback, *args)
        return True
    except (RuntimeError, tk.TclError) as schedule_error:
        _log_debug(
            "LATENCY: worker UI callback dropped: "
            f"{type(schedule_error).__name__} - {schedule_error}"
        )
        return False


def _get_translation_submit_interval_seconds(app, text_to_translate):
    handler = getattr(app, 'translation_handler', None)
    getter = getattr(handler, 'get_translation_submit_interval_seconds', None)
    if callable(getter):
        try:
            return max(0.0, float(getter(text_to_translate)))
        except Exception as interval_error:
            _log_debug(
                "LATENCY: failed to read translation submit interval: "
                f"{type(interval_error).__name__} - {interval_error}"
            )
    return max(0.0, float(getattr(app, 'min_translation_interval', 0.3) or 0.3))


def _get_translation_provider_cooldown_seconds(app, latency_mode=None):
    handler = getattr(app, 'translation_handler', None)
    getter = getattr(handler, 'get_translation_provider_cooldown_seconds', None)
    if callable(getter):
        try:
            return max(0.0, float(getter(latency_mode=latency_mode)))
        except TypeError:
            try:
                return max(0.0, float(getter()))
            except Exception as cooldown_error:
                _log_debug(
                    "LATENCY: failed to read translation provider cooldown: "
                    f"{type(cooldown_error).__name__} - {cooldown_error}"
                )
        except Exception as cooldown_error:
            _log_debug(
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
            _log_debug(
                "LATENCY: failed to read translation concurrency limit: "
                f"{type(limit_error).__name__} - {limit_error}"
            )
    return max(1, int(getattr(app, 'max_concurrent_translation_calls', 1) or 1))


def _get_translation_supersede_after_seconds(app, request_snapshot=None):
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
    configured = max(0.25, min(10.0, configured))
    if not isinstance(request_snapshot, dict):
        return configured
    try:
        sample_count = int(
            request_snapshot.get(
                "route_sample_count",
                request_snapshot.get("sample_count", 0),
            )
            or 0
        )
        p90_seconds = float(
            request_snapshot.get(
                "route_p90_seconds",
                request_snapshot.get("p90_seconds", 0.0),
            )
            or 0.0
        )
    except (OverflowError, TypeError, ValueError):
        return configured
    if (
        sample_count < ROUTE_SUPERSEDE_MIN_SAMPLES
        or not math.isfinite(p90_seconds)
        or p90_seconds <= 0.0
    ):
        return max(configured, ROUTE_SUPERSEDE_LEARNING_SECONDS)
    route_threshold = min(
        ROUTE_SUPERSEDE_MAX_SECONDS,
        p90_seconds * ROUTE_SUPERSEDE_P90_FRACTION,
    )
    return max(configured, route_threshold)


def _get_translation_latency_mode(app, latency_mode=None):
    if latency_mode:
        return str(latency_mode).strip().lower()
    getter = getattr(app, "get_custom_ai_latency_mode", None)
    if callable(getter):
        try:
            return str(getter() or "").strip().lower()
        except Exception:
            pass
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
        _log_debug(
            "LATENCY: invalidated pending translation request "
            f"generation={generation} reason={reason}"
        )
    return generation


def reset_translation_scheduler_session_state(app, reason):
    """Invalidate scheduler state that must not cross stop/start boundaries."""
    pending_generation = _invalidate_pending_translation_request(app, reason)
    app.latest_translation_candidate = None
    app.translation_profile_refresh_generation = int(
        getattr(app, "translation_profile_refresh_generation", 0) or 0
    ) + 1
    app.last_translation_submit_monotonic = 0.0

    sequence_floor = 0
    for sequence_name in (
        "last_displayed_translation_sequence",
        "latest_translation_sequence_started",
        "translation_sequence_counter",
    ):
        try:
            sequence_floor = max(
                sequence_floor,
                int(getattr(app, sequence_name, 0) or 0),
            )
        except (TypeError, ValueError):
            continue
    app.last_displayed_translation_sequence = sequence_floor

    active_sequences = set(getattr(app, "active_translation_calls", ()) or ())
    inflight_keys = getattr(app, "active_translation_inflight_keys", None)
    started_by_sequence = getattr(
        app,
        "active_translation_started_monotonic",
        None,
    )
    if active_sequences:
        if isinstance(started_by_sequence, dict):
            for sequence in tuple(started_by_sequence):
                if sequence not in active_sequences:
                    started_by_sequence.pop(sequence, None)
    else:
        if hasattr(inflight_keys, "clear"):
            inflight_keys.clear()
        if hasattr(started_by_sequence, "clear"):
            started_by_sequence.clear()

    _log_debug(
        "LATENCY: reset translation scheduler session state "
        f"pending_generation={pending_generation} "
        f"profile_generation={app.translation_profile_refresh_generation} "
        f"sequence_floor={sequence_floor} "
        f"active_calls={len(active_sequences)} "
        f"reason={reason}"
    )


def _flush_pending_translation_request(app, flush_generation=None):
    try:
        current_generation = int(
            getattr(app, 'pending_translation_flush_generation', 0) or 0
        )
        if flush_generation is not None and flush_generation != current_generation:
            _log_debug_coalesced(
                "translation-stale-pending-timer",
                "LATENCY: ignored stale pending translation timer "
                f"generation={flush_generation} current={current_generation}",
                interval_seconds=5.0,
            )
            return

        app.pending_translation_flush_scheduled = False
        app.pending_translation_flush_deadline_monotonic = 0.0
        pending_request = getattr(app, 'pending_translation_request', None)
        if not pending_request:
            return
        app.pending_translation_request = None

        if hasattr(app, 'is_running') and not app.is_running:
            _log_debug("LATENCY: dropped pending translation because the app is stopped")
            return

        submit_kwargs = {
            "requested_at_monotonic": pending_request.get(
                "requested_at_monotonic"
            )
        }
        if pending_request.get("configuration_refresh", False):
            submit_kwargs["configuration_refresh"] = True
        _start_async_translation(
            app,
            pending_request["text"],
            pending_request["ocr_sequence_number"],
            **submit_kwargs,
        )
    except Exception as flush_error:
        _log_debug(
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
    configuration_refresh=False,
):
    now = time.monotonic()
    if requested_at_monotonic is None:
        requested_at_monotonic = now
    _increment_metric(app, "pending_translation_queued")
    app.pending_translation_request = {
        "text": text_to_translate,
        "ocr_sequence_number": ocr_sequence_number,
        "requested_at_monotonic": float(requested_at_monotonic),
        "configuration_refresh": bool(configuration_refresh),
    }
    _log_debug_coalesced(
        "translation-pending-queue",
        "LATENCY: queued latest translation request "
        f"for OCR batch {ocr_sequence_number} delay={delay_seconds:.3f}s "
        f"reason={reason}",
        interval_seconds=5.0,
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
        configuration_refresh=bool(
            pending_request.get("configuration_refresh", False)
        ),
    )


def _apply_translation_profile_refresh(
    app,
    refresh_generation,
    captured_candidate,
    reason,
):
    current_generation = int(
        getattr(app, "translation_profile_refresh_generation", 0) or 0
    )
    if refresh_generation != current_generation:
        _log_debug(
            "LATENCY: ignored stale translation profile refresh "
            f"generation={refresh_generation} current={current_generation}"
        )
        return False
    if not getattr(app, "is_running", False):
        return False

    pending_request = getattr(app, "pending_translation_request", None)
    latest_candidate = getattr(app, "latest_translation_candidate", None)
    if isinstance(pending_request, dict):
        candidate = pending_request
    elif isinstance(latest_candidate, dict):
        candidate = latest_candidate
    else:
        candidate = captured_candidate
    if not isinstance(candidate, dict) or not candidate.get("text"):
        return False

    _invalidate_pending_translation_request(
        app,
        "applying translation profile refresh",
    )
    _start_async_translation(
        app,
        candidate["text"],
        candidate.get("ocr_sequence_number", 0),
        requested_at_monotonic=candidate.get("requested_at_monotonic"),
        configuration_refresh=True,
    )
    return True


def refresh_translation_after_profile_change(app, reason="profile changed"):
    """Re-evaluate the latest subtitle without inheriting an old model timer."""
    pending_request = getattr(app, "pending_translation_request", None)
    latest_candidate = getattr(app, "latest_translation_candidate", None)
    candidate = pending_request if isinstance(pending_request, dict) else None
    if candidate is None and isinstance(latest_candidate, dict):
        candidate = latest_candidate
    if not candidate or not getattr(app, "is_running", False):
        return False
    if not candidate.get("text"):
        return False

    previous_generation = int(
        getattr(app, "translation_profile_refresh_generation", 0) or 0
    )
    refresh_generation = previous_generation + 1
    app.translation_profile_refresh_generation = refresh_generation
    try:
        app.root.after(
            0,
            _apply_translation_profile_refresh,
            app,
            refresh_generation,
            dict(candidate),
            reason,
        )
    except Exception:
        app.translation_profile_refresh_generation = previous_generation
        raise

    _invalidate_pending_translation_request(app, reason)
    _log_debug(
        "LATENCY: scheduled latest translation refresh after profile change "
        f"reason={reason}"
    )
    return True


def _submit_async_translation_request(
    app,
    text_to_translate,
    ocr_sequence_number,
    inflight_key,
    requested_at_monotonic=None,
    latency_mode=None,
    request_snapshot=None,
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
            _process_translation_async,
            app,
            text_to_translate,
            translation_sequence,
            ocr_sequence_number,
            inflight_key,
            requested_at_monotonic,
            latency_mode,
            request_snapshot,
        )
    except Exception:
        app.active_translation_calls.discard(translation_sequence)
        app.active_translation_inflight_keys.discard(inflight_key)
        started_by_sequence.pop(translation_sequence, None)
        _refresh_translation_metric_gauges(app)
        raise

    _log_debug(
        f"Started async translation {translation_sequence} for OCR batch {ocr_sequence_number} "
        f"(active calls: {len(app.active_translation_calls)}) "
        f"{summarize_text_for_log(text_to_translate)}"
    )


def _coalesce_matching_pending_translation_request(
    app,
    text_to_translate,
    ocr_sequence_number,
):
    pending_request = getattr(app, 'pending_translation_request', None)
    if (
        not isinstance(pending_request, dict)
        or pending_request.get('text') != text_to_translate
        or not getattr(app, 'pending_translation_flush_scheduled', False)
    ):
        return False

    pending_request['ocr_sequence_number'] = ocr_sequence_number
    _increment_metric(app, 'pending_translation_coalesced')
    return True


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
            _log_debug(
                "Streaming translation display failed: "
                f"{type(stream_error).__name__} - {stream_error}"
            )

    def stream_callback(partial_text):
        with state_lock:
            state["latest_text"] = partial_text
            if state["scheduled"]:
                return
            state["scheduled"] = True

        if not _schedule_ui_callback(app, display_latest_partial):
            with state_lock:
                state["scheduled"] = False

    return stream_callback
