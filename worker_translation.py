"""Pending translation scheduling and streaming display helpers."""

import math
import sys
import threading
import time
import tkinter as tk
from urllib.parse import urlsplit, urlunsplit

from logger import summarize_text_for_log
from translation_utils import post_process_translation_text
from worker_capture import _increment_metric, _refresh_translation_metric_gauges

DEFAULT_TRANSLATION_SUPERSEDE_AFTER_SECONDS = 1.5
ROUTE_SUPERSEDE_MIN_SAMPLES = 8
ROUTE_SUPERSEDE_LEARNING_SECONDS = 3.0
ROUTE_SUPERSEDE_P90_FRACTION = 0.5
ROUTE_SUPERSEDE_MAX_SECONDS = 4.0
# Adaptive overflow protection for proven-slow routes.
# Run A (healthy): worker avg ~1.3s. Run B (slow): worker avg ~2.6s, p90 ~3.6s+.
OVERFLOW_PROTECTION_MIN_SAMPLES = ROUTE_SUPERSEDE_MIN_SAMPLES
OVERFLOW_PROTECTION_ENTER_P90_SECONDS = 3.0
OVERFLOW_PROTECTION_RECOVER_P90_SECONDS = 2.5
OVERFLOW_PROTECTION_SLOW_WORKER_SECONDS = 5.0
OVERFLOW_PROTECTION_SLOW_STREAK_ENTER = 2
OVERFLOW_PROTECTION_FAST_WORKER_SECONDS = 2.0
OVERFLOW_PROTECTION_FAST_STREAK_RECOVER = 3
OVERFLOW_PROTECTION_MAX_ROUTES = 16
_OVERFLOW_PROTECTION_LOCK_INIT = threading.Lock()
TRANSIENT_FAILURE_STATUS_STREAK = 3
TRANSIENT_FAILURE_STATUS_DURATION_SECONDS = 8.0
TRANSIENT_FAILURE_STATUS_THROTTLE_SECONDS = 30.0


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


def _schedule_ui_callback(app, callback, *args, delay_ms=0):
    """Schedule a worker UI callback only while the Tk root is still usable."""
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
        root.after(max(0, int(delay_ms or 0)), callback, *args)
        return True
    except (RuntimeError, tk.TclError) as schedule_error:
        _log_debug(
            "LATENCY: worker UI callback dropped: "
            f"{type(schedule_error).__name__} - {schedule_error}"
        )
        return False


def _failure_visibility_session_generation(app):
    try:
        return int(getattr(app, "translation_failure_visibility_generation", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _bump_failure_visibility_session_generation(app):
    generation = _failure_visibility_session_generation(app) + 1
    app.translation_failure_visibility_generation = generation
    return generation


def _get_failure_visibility_provider_key(app):
    profiles = getattr(app, "custom_ai_profiles", None)
    getter = getattr(profiles, "get_active_profile", None)
    profile = None
    if callable(getter):
        try:
            profile = getter("translation")
        except Exception:
            profile = None
    if isinstance(profile, dict):
        profile_id = str(profile.get("id") or "").strip()
        if profile_id:
            return f"custom_ai:{profile_id}"
        profile_name = str(profile.get("name") or "").strip()
        if profile_name:
            return f"custom_ai:name:{profile_name}"
    return "custom_ai:default"


def _empty_failure_visibility_state(provider_key, session_generation, now=None):
    return {
        "provider_key": provider_key,
        "session_generation": int(session_generation or 0),
        "streak": 0,
        "first_failure_monotonic": None if now is None else float(now),
        "last_shown_monotonic": 0.0,
        "last_sequence": 0,
        "shown": False,
        "ui_generation": 0,
    }


def _get_failure_visibility_state(app):
    state = getattr(app, "translation_failure_visibility", None)
    if isinstance(state, dict):
        return state
    return None


def _is_current_failure_visibility_sequence(app, sequence):
    """Reject old in-flight replies after a newer translation was submitted."""
    try:
        latest_started = int(
            getattr(app, "latest_translation_sequence_started", 0) or 0
        )
    except (TypeError, ValueError):
        latest_started = 0
    return latest_started <= 0 or sequence >= latest_started


def _status_label_is_usable(app):
    if getattr(app, "_app_is_closing", False):
        return False
    if hasattr(app, "is_running") and not bool(app.is_running):
        return False
    status_label = getattr(app, "status_label", None)
    if status_label is None:
        return False
    exists = getattr(status_label, "winfo_exists", None)
    try:
        if callable(exists) and not bool(exists()):
            return False
    except Exception:
        return False
    return True


def _running_status_text(app):
    ui_lang = getattr(app, "ui_lang", None)
    getter = getattr(ui_lang, "get_label", None)
    if callable(getter):
        try:
            running = getter("status_running", "Running (Press ~ to Stop)")
        except Exception:
            running = "Running (Press ~ to Stop)"
    else:
        running = "Running (Press ~ to Stop)"
    running = str(running or "Running (Press ~ to Stop)")
    if running.startswith("Status:"):
        return running
    return f"Status: {running}"


def _transient_failure_status_text(app):
    ui_lang = getattr(app, "ui_lang", None)
    getter = getattr(ui_lang, "get_label", None)
    if callable(getter):
        try:
            message = getter(
                "status_provider_transient",
                "Translation service temporarily unavailable",
            )
        except Exception:
            message = "Translation service temporarily unavailable"
    else:
        message = "Translation service temporarily unavailable"
    message = str(message or "Translation service temporarily unavailable")
    if message.startswith("Status:"):
        return message
    return f"Status: {message}"


def _set_status_label_text(app, text):
    if not _status_label_is_usable(app):
        return False
    status_label = getattr(app, "status_label", None)
    config = getattr(status_label, "config", None)
    if not callable(config):
        return False
    try:
        current = None
        cget = getattr(status_label, "cget", None)
        if callable(cget):
            try:
                current = cget("text")
            except Exception:
                current = None
        if current == text:
            return False
        config(text=text)
        return True
    except Exception as status_error:
        _log_debug(
            "LATENCY: failed to update failure visibility status: "
            f"{type(status_error).__name__} - {status_error}"
        )
        return False


def reset_translation_failure_visibility(app, clear_status=False, reason=""):
    """Reset provider/session-scoped transient failure visibility state."""
    state = _get_failure_visibility_state(app)
    was_shown = bool(isinstance(state, dict) and state.get("shown"))
    session_generation = _bump_failure_visibility_session_generation(app)
    provider_key = _get_failure_visibility_provider_key(app)
    app.translation_failure_visibility = _empty_failure_visibility_state(
        provider_key,
        session_generation,
        now=None,
    )
    if clear_status and was_shown:
        _set_status_label_text(app, _running_status_text(app))
    if reason:
        _log_debug_coalesced(
            "translation-failure-visibility-reset",
            "LATENCY: reset translation failure visibility "
            f"reason={reason} provider={provider_key} "
            f"session_generation={session_generation}",
            interval_seconds=5.0,
        )
    return app.translation_failure_visibility


def _ensure_failure_visibility_state(app, provider_key=None):
    session_generation = _failure_visibility_session_generation(app)
    provider_key = provider_key or _get_failure_visibility_provider_key(app)
    state = _get_failure_visibility_state(app)
    if not isinstance(state, dict):
        state = _empty_failure_visibility_state(
            provider_key,
            session_generation,
            now=None,
        )
        app.translation_failure_visibility = state
        return state

    state_provider = str(state.get("provider_key") or "")
    state_generation = int(state.get("session_generation") or 0)
    if state_provider != provider_key or state_generation != session_generation:
        was_shown = bool(state.get("shown"))
        state = _empty_failure_visibility_state(
            provider_key,
            session_generation,
            now=None,
        )
        app.translation_failure_visibility = state
        if was_shown:
            _set_status_label_text(app, _running_status_text(app))
    return state


def _should_show_transient_failure_status(state, now):
    streak = int(state.get("streak") or 0)
    first_failure = state.get("first_failure_monotonic")
    duration_hit = False
    if first_failure is not None:
        try:
            duration_hit = (
                float(now) - float(first_failure)
            ) >= TRANSIENT_FAILURE_STATUS_DURATION_SECONDS
        except (TypeError, ValueError):
            duration_hit = False
    threshold_hit = (
        streak >= TRANSIENT_FAILURE_STATUS_STREAK or duration_hit
    )
    if not threshold_hit:
        return False
    if not state.get("shown"):
        return True
    last_shown = float(state.get("last_shown_monotonic") or 0.0)
    return (float(now) - last_shown) >= TRANSIENT_FAILURE_STATUS_THROTTLE_SECONDS


def note_transient_translation_failure(app, translation_sequence):
    """Track consecutive transient failures and show a throttled status hint."""
    if getattr(app, "_app_is_closing", False):
        return False
    if hasattr(app, "is_running") and not bool(app.is_running):
        return False

    try:
        sequence = int(translation_sequence or 0)
    except (TypeError, ValueError):
        sequence = 0
    if sequence <= 0:
        return False
    if not _is_current_failure_visibility_sequence(app, sequence):
        return False

    provider_key = _get_failure_visibility_provider_key(app)
    state = _ensure_failure_visibility_state(app, provider_key=provider_key)
    last_sequence = int(state.get("last_sequence") or 0)
    if sequence < last_sequence:
        return False
    if sequence == last_sequence and int(state.get("streak") or 0) > 0:
        return False

    now = time.monotonic()
    streak = int(state.get("streak") or 0)
    if streak <= 0 or state.get("first_failure_monotonic") is None:
        state["first_failure_monotonic"] = now
        streak = 0
    streak += 1
    state["streak"] = streak
    state["last_sequence"] = sequence
    state["provider_key"] = provider_key
    state["session_generation"] = _failure_visibility_session_generation(app)

    if not _should_show_transient_failure_status(state, now):
        return False
    if not _status_label_is_usable(app):
        return False

    status_text = _transient_failure_status_text(app)
    if not _set_status_label_text(app, status_text):
        # Either unusable or already showing the same text.
        state["shown"] = True
        if not state.get("last_shown_monotonic"):
            state["last_shown_monotonic"] = now
        return False

    state["shown"] = True
    state["last_shown_monotonic"] = now
    state["ui_generation"] = int(state.get("ui_generation") or 0) + 1
    _log_debug_coalesced(
        "translation-failure-visibility-shown",
        "LATENCY: showing throttled transient provider status "
        f"provider={provider_key} streak={streak} sequence={sequence}",
        interval_seconds=5.0,
    )
    return True


def clear_transient_translation_failure_status(
    app,
    translation_sequence=None,
    reason="success",
):
    """Clear throttled failure status after the first successful translation."""
    state = _get_failure_visibility_state(app)
    if not isinstance(state, dict):
        return False

    if translation_sequence is not None:
        try:
            sequence = int(translation_sequence or 0)
        except (TypeError, ValueError):
            sequence = 0
        if not _is_current_failure_visibility_sequence(app, sequence):
            return False
        last_sequence = int(state.get("last_sequence") or 0)
        if sequence and last_sequence and sequence < last_sequence:
            return False

    was_shown = bool(state.get("shown"))
    provider_key = _get_failure_visibility_provider_key(app)
    session_generation = _failure_visibility_session_generation(app)
    app.translation_failure_visibility = _empty_failure_visibility_state(
        provider_key,
        session_generation,
        now=None,
    )

    if was_shown and _status_label_is_usable(app):
        _set_status_label_text(app, _running_status_text(app))
        _log_debug_coalesced(
            "translation-failure-visibility-cleared",
            "LATENCY: cleared transient provider status "
            f"reason={reason} provider={provider_key}",
            interval_seconds=5.0,
        )
        return True
    return was_shown


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


def _translation_overflow_protection_route_key(request_snapshot):
    """Return a stable route key without logging secrets or full URLs."""
    if not isinstance(request_snapshot, dict):
        return ("default",)
    profile = request_snapshot.get("profile")
    if isinstance(profile, dict):
        profile_id = str(profile.get("id") or "").strip() or "profile"
        model = str(profile.get("model") or "").strip() or "model"
        wire_api = str(profile.get("wire_api") or "").strip().lower() or "default"
        base = str(profile.get("base_url") or "").strip()
        try:
            parts = urlsplit(base)
            scheme = str(parts.scheme or "").lower()
            host = str(parts.hostname or "").lower()
            if scheme and host:
                if ":" in host:
                    host = f"[{host}]"
                port = parts.port
                default_port = (
                    (scheme == "https" and port == 443)
                    or (scheme == "http" and port == 80)
                )
                netloc = host if port is None or default_port else f"{host}:{port}"
                endpoint = urlunsplit(
                    (scheme, netloc, str(parts.path or "").rstrip("/"), "", "")
                )
            else:
                endpoint = "endpoint"
        except (TypeError, ValueError):
            endpoint = "endpoint"
        return (profile_id, model, endpoint, wire_api)
    inflight = request_snapshot.get("inflight_key")
    if isinstance(inflight, tuple) and inflight:
        return tuple(str(part) for part in inflight[:3])
    latency_mode = str(request_snapshot.get("latency_mode") or "").strip() or "mode"
    return ("snapshot", latency_mode)


def _get_translation_overflow_protection_map(app):
    state_map = getattr(app, "_translation_overflow_protection_by_route", None)
    if not isinstance(state_map, dict):
        state_map = {}
        app._translation_overflow_protection_by_route = state_map
    return state_map


def _get_translation_overflow_protection_lock(app):
    """Return the app-scoped lock shared by UI scheduling and worker completion."""
    lock = getattr(app, "_translation_overflow_protection_lock", None)
    if lock is None or not hasattr(lock, "__enter__"):
        # A worker can complete while the UI scheduler evaluates overflow.
        # Serialize first-time setup so both threads share the same lock.
        with _OVERFLOW_PROTECTION_LOCK_INIT:
            lock = getattr(app, "_translation_overflow_protection_lock", None)
            if lock is None or not hasattr(lock, "__enter__"):
                lock = threading.RLock()
                app._translation_overflow_protection_lock = lock
    return lock


def _get_translation_overflow_protection_state(app, request_snapshot):
    route_key = _translation_overflow_protection_route_key(request_snapshot)
    state_map = _get_translation_overflow_protection_map(app)
    state = state_map.get(route_key)
    if not isinstance(state, dict):
        state = {
            "active": False,
            "slow_streak": 0,
            "fast_streak": 0,
            "last_p90": None,
            "last_samples": 0,
            "entered_reason": "",
        }
        state_map[route_key] = state
        while len(state_map) > OVERFLOW_PROTECTION_MAX_ROUTES:
            try:
                state_map.pop(next(iter(state_map)))
            except Exception:
                break
    return route_key, state


def _read_route_latency_stats(request_snapshot):
    if not isinstance(request_snapshot, dict):
        return 0, 0.0
    try:
        sample_count = int(
            request_snapshot.get(
                "route_sample_count",
                request_snapshot.get("timeout_sample_count", 0),
            )
            or 0
        )
    except (TypeError, ValueError):
        sample_count = 0
    try:
        p90_seconds = float(
            request_snapshot.get(
                "route_p90_seconds",
                request_snapshot.get("timeout_p90_seconds", 0.0),
            )
            or 0.0
        )
    except (OverflowError, TypeError, ValueError):
        p90_seconds = 0.0
    if not math.isfinite(p90_seconds) or p90_seconds < 0.0:
        p90_seconds = 0.0
    return max(0, sample_count), float(p90_seconds)


def _enter_translation_overflow_protection(app, route_key, state, reason, p90_seconds, sample_count):
    if state.get("active"):
        return False
    state["active"] = True
    state["entered_reason"] = str(reason or "slow_route")
    state["last_p90"] = p90_seconds
    state["last_samples"] = sample_count
    state["fast_streak"] = 0
    _increment_metric(app, "translation_overflow_protection_entered")
    _log_debug_coalesced(
        f"translation-overflow-protection-entered:{route_key[0]}",
        "LATENCY: overflow protection entered "
        f"reason={state['entered_reason']} "
        f"p90={float(p90_seconds or 0.0):.3f}s "
        f"samples={int(sample_count or 0)}",
        interval_seconds=5.0,
    )
    return True


def _recover_translation_overflow_protection(app, route_key, state, reason, p90_seconds, sample_count):
    if not state.get("active"):
        return False
    state["active"] = False
    state["entered_reason"] = ""
    state["slow_streak"] = 0
    state["fast_streak"] = 0
    state["last_p90"] = p90_seconds
    state["last_samples"] = sample_count
    _increment_metric(app, "translation_overflow_protection_recovered")
    _log_debug_coalesced(
        f"translation-overflow-protection-recovered:{route_key[0]}",
        "LATENCY: overflow protection recovered "
        f"reason={reason} "
        f"p90={float(p90_seconds or 0.0):.3f}s "
        f"samples={int(sample_count or 0)}",
        interval_seconds=5.0,
    )
    return True


def note_translation_overflow_route_sample(
    app,
    request_snapshot,
    worker_seconds,
    *,
    success=True,
):
    """Update route-scoped overflow protection from a completed worker sample."""
    try:
        worker_seconds = float(worker_seconds or 0.0)
    except (TypeError, ValueError):
        worker_seconds = 0.0
    if not math.isfinite(worker_seconds) or worker_seconds < 0.0:
        worker_seconds = 0.0

    with _get_translation_overflow_protection_lock(app):
        route_key, state = _get_translation_overflow_protection_state(app, request_snapshot)
        sample_count, p90_seconds = _read_route_latency_stats(request_snapshot)
        state["last_p90"] = p90_seconds
        state["last_samples"] = sample_count

        if success and worker_seconds >= OVERFLOW_PROTECTION_SLOW_WORKER_SECONDS:
            state["slow_streak"] = int(state.get("slow_streak") or 0) + 1
            state["fast_streak"] = 0
            if state["slow_streak"] >= OVERFLOW_PROTECTION_SLOW_STREAK_ENTER:
                _enter_translation_overflow_protection(
                    app,
                    route_key,
                    state,
                    reason="slow_worker_streak",
                    p90_seconds=p90_seconds,
                    sample_count=sample_count,
                )
            return state

        if success and worker_seconds > 0.0 and worker_seconds < OVERFLOW_PROTECTION_FAST_WORKER_SECONDS:
            state["fast_streak"] = int(state.get("fast_streak") or 0) + 1
            state["slow_streak"] = 0
        elif success:
            # Medium samples neither prove recovery nor escalate.
            state["slow_streak"] = 0
            state["fast_streak"] = 0

        if state.get("active"):
            recovered_by_p90 = (
                sample_count >= OVERFLOW_PROTECTION_MIN_SAMPLES
                and 0.0 < p90_seconds < OVERFLOW_PROTECTION_RECOVER_P90_SECONDS
            )
            recovered_by_fast_streak = (
                int(state.get("fast_streak") or 0) >= OVERFLOW_PROTECTION_FAST_STREAK_RECOVER
            )
            if recovered_by_p90 or recovered_by_fast_streak:
                reason = "route_p90_recovered" if recovered_by_p90 else "fast_worker_streak"
                _recover_translation_overflow_protection(
                    app,
                    route_key,
                    state,
                    reason=reason,
                    p90_seconds=p90_seconds,
                    sample_count=sample_count,
                )
        return state


def should_block_translation_overflow_for_slow_route(app, request_snapshot):
    """Return True when bounded overflow should stay closed for a slow route."""
    with _get_translation_overflow_protection_lock(app):
        route_key, state = _get_translation_overflow_protection_state(app, request_snapshot)
        sample_count, p90_seconds = _read_route_latency_stats(request_snapshot)
        state["last_p90"] = p90_seconds
        state["last_samples"] = sample_count

        if not state.get("active"):
            if (
                sample_count >= OVERFLOW_PROTECTION_MIN_SAMPLES
                and p90_seconds >= OVERFLOW_PROTECTION_ENTER_P90_SECONDS
            ):
                _enter_translation_overflow_protection(
                    app,
                    route_key,
                    state,
                    reason="route_p90_high",
                    p90_seconds=p90_seconds,
                    sample_count=sample_count,
                )
            elif int(state.get("slow_streak") or 0) >= OVERFLOW_PROTECTION_SLOW_STREAK_ENTER:
                _enter_translation_overflow_protection(
                    app,
                    route_key,
                    state,
                    reason="slow_worker_streak",
                    p90_seconds=p90_seconds,
                    sample_count=sample_count,
                )

        if state.get("active"):
            # Allow auto-exit if snapshot already shows recovery before next sample.
            if (
                sample_count >= OVERFLOW_PROTECTION_MIN_SAMPLES
                and 0.0 < p90_seconds < OVERFLOW_PROTECTION_RECOVER_P90_SECONDS
            ):
                _recover_translation_overflow_protection(
                    app,
                    route_key,
                    state,
                    reason="route_p90_recovered",
                    p90_seconds=p90_seconds,
                    sample_count=sample_count,
                )
                return False
            _increment_metric(app, "translation_overflow_protection_blocked")
            _log_debug_coalesced(
                f"translation-overflow-protection-blocked:{route_key[0]}",
                "LATENCY: overflow blocked by slow-route protection "
                f"reason={state.get('entered_reason') or 'active'} "
                f"p90={float(p90_seconds or 0.0):.3f}s "
                f"samples={int(sample_count or 0)}",
                interval_seconds=5.0,
            )
            return True
        return False


def clear_translation_overflow_protection_state(app, reason="session reset"):
    with _get_translation_overflow_protection_lock(app):
        state_map = getattr(app, "_translation_overflow_protection_by_route", None)
        if isinstance(state_map, dict):
            state_map.clear()
        elif state_map is not None:
            app._translation_overflow_protection_by_route = {}
    _log_debug(
        "LATENCY: cleared translation overflow protection state "
        f"reason={reason}"
    )


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
    clear_translation_overflow_protection_state(app, reason=reason)
    reset_translation_failure_visibility(
        app,
        clear_status=False,
        reason=reason,
    )

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
    scheduled = _schedule_ui_callback(
        app,
        _flush_pending_translation_request,
        app,
        generation,
        delay_ms=delay_ms,
    )
    if not scheduled:
        if int(getattr(app, 'pending_translation_flush_generation', 0) or 0) == generation:
            app.pending_translation_flush_scheduled = False
            app.pending_translation_flush_deadline_monotonic = 0.0
        _log_debug(
            "LATENCY: failed to schedule pending translation flush: "
            f"generation={generation} reason={reason}"
        )
        return


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
        reset_translation_failure_visibility(
            app,
            clear_status=True,
            reason=reason,
        )
        return False
    if not candidate.get("text"):
        reset_translation_failure_visibility(
            app,
            clear_status=True,
            reason=reason,
        )
        return False

    previous_generation = int(
        getattr(app, "translation_profile_refresh_generation", 0) or 0
    )
    refresh_generation = previous_generation + 1
    app.translation_profile_refresh_generation = refresh_generation
    reset_translation_failure_visibility(
        app,
        clear_status=True,
        reason=reason,
    )
    scheduled = _schedule_ui_callback(
        app,
        _apply_translation_profile_refresh,
        app,
        refresh_generation,
        dict(candidate),
        reason,
    )
    if not scheduled:
        app.translation_profile_refresh_generation = previous_generation
        _log_debug(
            "LATENCY: failed to schedule translation profile refresh: "
            f"reason={reason}"
        )
        return False

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
    if getattr(app, "_async_submissions_frozen", False):
        return False
    # A real submit supersedes any older pending request. Instant-cache hits
    # already invalidate pending; without this, completion of an older active
    # call can expedite stale pending text and re-bill remote translation.
    pending_request = getattr(app, "pending_translation_request", None)
    if isinstance(pending_request, dict):
        pending_text = pending_request.get("text")
        if pending_text != text_to_translate:
            _invalidate_pending_translation_request(
                app,
                "newer translation submitted",
            )
            _increment_metric(app, "pending_translation_superseded")
        else:
            # Same text already pending: clear timer/state so completion
            # expedite cannot double-submit the just-started request.
            _invalidate_pending_translation_request(
                app,
                "matching pending translation already submitted",
            )

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
        future = app.translation_thread_pool.submit(
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
        tracker = getattr(app, "track_async_future", None)
        if callable(tracker):
            tracker(future, "translation")
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
    return True


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
                clear_transient_translation_failure_status(
                    app,
                    translation_sequence=translation_sequence,
                    reason="stream_success",
                )
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
