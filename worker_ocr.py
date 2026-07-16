"""OCR candidate quality, stability gating, and local OCR routing."""

from difflib import SequenceMatcher
import queue
import re
import sys
import time

from logger import summarize_text_for_log
from paddle_ocr_backend import PADDLEOCR_MODEL_CODE
from worker_capture import _increment_metric, _refresh_ocr_queue_metric

OCR_STABILITY_GATE_MIN_WAIT_SECONDS = 0.12
OCR_STABILITY_GATE_MAX_WAIT_SECONDS = 0.25
OCR_STABILITY_GATE_SUSPICIOUS_SHORT_LENGTH = 12
OCR_STABILITY_GATE_NOISE_RATIO = 0.35


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


def _normalize_custom_ai_ocr_reasoning_contract(value):
    normalized = str(value or 'low').strip().lower().replace('-', '_')
    if normalized in {'low', 'medium', 'high', 'ultra', 'none'}:
        return normalized
    return 'low'


def _get_custom_ai_ocr_reasoning_contract(app):
    profiles = getattr(app, 'custom_ai_profiles', None)
    if profiles is None:
        return None
    try:
        profile = profiles.get_active_profile("ocr")
    except Exception as e:
        _log_debug(f"Could not resolve active OCR profile for API OCR reasoning cache key: {type(e).__name__} - {e}")
        return 'none'
    if not profile:
        return 'none'

    provider = getattr(
        getattr(app, 'translation_handler', None),
        'custom_ai_provider',
        None,
    )
    if provider is not None and hasattr(provider, 'reasoning_effort_request_contract'):
        try:
            return _normalize_custom_ai_ocr_reasoning_contract(
                provider.reasoning_effort_request_contract(profile, "ocr")
            )
        except Exception as e:
            _log_debug(f"Could not resolve Custom AI OCR reasoning contract: {type(e).__name__} - {e}")

    return _normalize_custom_ai_ocr_reasoning_contract(
        profile.get("reasoning_effort") or profile.get("model_reasoning_effort")
    )


def _normalize_local_ocr_submit_text(text_to_translate):
    if not isinstance(text_to_translate, str):
        return ""

    normalized = text_to_translate.replace("<br>", " ")
    normalized = normalized.translate(str.maketrans({
        "\u201c": '"',
        "\u201d": '"',
        "\u2018": "'",
        "\u2019": "'",
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
    }))
    normalized = normalized.lower()
    normalized = re.sub(r"\s+", " ", normalized).strip()
    normalized = re.sub(r"[.!?\u2026]+$", "", normalized).strip()
    return normalized


def _ocr_candidate_has_terminal_punctuation(text):
    return bool(re.search(r"[.!?]+[\"')\]]*$", str(text or "").strip()))


def _ocr_candidate_tokens(normalized_text):
    return re.findall(r"[^\W_]+", normalized_text, flags=re.UNICODE)


def _ocr_candidate_noise_ratio(text):
    chars = [char for char in str(text or "") if not char.isspace()]
    if not chars:
        return 1.0

    allowed_punctuation = set(".,!?;:'\"-()[]{}<>/\\|")
    noisy_count = 0
    for char in chars:
        if char.isalnum() or char in allowed_punctuation:
            continue
        noisy_count += 1
    return noisy_count / len(chars)


def _looks_like_low_quality_ocr_candidate(text):
    normalized = _normalize_local_ocr_submit_text(text)
    if not normalized:
        return True
    if not any(char.isalnum() for char in normalized):
        return True
    if _ocr_candidate_noise_ratio(text) > OCR_STABILITY_GATE_NOISE_RATIO:
        return True

    has_terminal_punctuation = _ocr_candidate_has_terminal_punctuation(text)
    tokens = _ocr_candidate_tokens(normalized)
    if (
        len(tokens) == 1
        and len(tokens[0]) <= 8
        and not has_terminal_punctuation
        and not any(char in "aeiou" for char in tokens[0])
    ):
        return True
    if len(normalized) <= 2 and not has_terminal_punctuation:
        return True
    if (
        len(normalized) < OCR_STABILITY_GATE_SUSPICIOUS_SHORT_LENGTH
        and len(tokens) >= 2
        and not has_terminal_punctuation
    ):
        return True
    if (
        len(tokens) >= 2
        and len(tokens[-1]) <= 2
        and not has_terminal_punctuation
    ):
        return True
    if (
        len(tokens) >= 4
        and len(tokens[-1]) == 3
        and not has_terminal_punctuation
    ):
        return True
    return False


def _is_transient_custom_ai_provider_error(value):
    if not isinstance(value, str):
        return False
    normalized = value.casefold()
    if not normalized.lstrip().startswith("custom ai translation error:"):
        return False
    transient_markers = (
        "streaming api response did not contain message content",
        "streaming responses api response did not contain output text",
        "structured translation response contained empty translation",
        "tls/ssl connection was closed",
        "ssleoferror",
        "unexpected_eof_while_reading",
        "read timed out",
        "connectionreseterror",
        "connection aborted",
        "remote host forcibly closed",
        "杩滅▼涓绘満寮鸿揩鍏抽棴",
    )
    return any(marker in normalized for marker in transient_markers)


def _ocr_candidate_quality_score(text):
    normalized = _normalize_local_ocr_submit_text(text)
    if not normalized:
        return -100

    score = min(len(normalized), 80)
    if _ocr_candidate_has_terminal_punctuation(text):
        score += 20
    score -= int(_ocr_candidate_noise_ratio(text) * 50)
    if _looks_like_low_quality_ocr_candidate(text):
        score -= 20
    return score


def _looks_like_more_complete_ocr_candidate(new_text, old_text):
    new_norm = _normalize_local_ocr_submit_text(new_text)
    old_norm = _normalize_local_ocr_submit_text(old_text)
    if not new_norm or not old_norm:
        return False
    if len(new_norm) <= len(old_norm) + 2:
        return False
    if new_norm.startswith(old_norm):
        return True

    similarity = SequenceMatcher(None, new_norm, old_norm).ratio()
    if (
        similarity >= 0.45
        and len(new_norm) >= max(len(old_norm) + 4, int(len(old_norm) * 1.25))
    ):
        return True
    return (
        _ocr_candidate_has_terminal_punctuation(new_text)
        and not _ocr_candidate_has_terminal_punctuation(old_text)
        and similarity >= 0.40
    )


class OcrStabilityDecision:
    def __init__(
        self,
        action,
        text=None,
        delay_seconds=0.0,
        reason="",
        requested_at_monotonic=None,
    ):
        self.action = action
        self.text = text
        self.delay_seconds = max(0.0, float(delay_seconds or 0.0))
        self.reason = reason
        self.requested_at_monotonic = requested_at_monotonic


class OcrStabilityGate:
    def __init__(
        self,
        min_wait_seconds=OCR_STABILITY_GATE_MIN_WAIT_SECONDS,
        max_wait_seconds=OCR_STABILITY_GATE_MAX_WAIT_SECONDS,
        monotonic_clock=time.monotonic,
    ):
        self.min_wait_seconds = max(0.0, float(min_wait_seconds))
        self.max_wait_seconds = max(
            self.min_wait_seconds,
            float(max_wait_seconds),
        )
        self.monotonic_clock = monotonic_clock
        self.pending_text = None
        self.pending_norm = ""
        self.pending_first_seen_monotonic = 0.0
        self.pending_last_seen_monotonic = 0.0
        self.flush_scheduled = False
        self.flush_deadline_monotonic = 0.0
        self.generation = 0

    def has_pending(self):
        return self.pending_text is not None

    def clear(self):
        had_pending = bool(self.pending_text or self.flush_scheduled)
        self.pending_text = None
        self.pending_norm = ""
        self.pending_first_seen_monotonic = 0.0
        self.pending_last_seen_monotonic = 0.0
        self.flush_scheduled = False
        self.flush_deadline_monotonic = 0.0
        self.generation += 1
        return had_pending

    def _queue_pending(self, text, normalized, now, reason, keep_first_seen=False):
        if not keep_first_seen or self.pending_text is None:
            self.pending_first_seen_monotonic = float(now)
        self.pending_text = text
        self.pending_norm = normalized
        self.pending_last_seen_monotonic = float(now)
        return OcrStabilityDecision(
            "wait",
            text=text,
            delay_seconds=self.remaining_wait_seconds(now),
            reason=reason,
            requested_at_monotonic=self.pending_first_seen_monotonic,
        )

    def remaining_wait_seconds(self, now=None):
        if now is None:
            now = self.monotonic_clock()
        if self.pending_text is None:
            return self.max_wait_seconds
        age = max(0.0, float(now) - self.pending_first_seen_monotonic)
        return max(0.0, self.max_wait_seconds - age)

    def evaluate(self, text, now=None):
        if now is None:
            now = self.monotonic_clock()
        now = float(now)
        normalized = _normalize_local_ocr_submit_text(text)
        if not normalized or not any(char.isalnum() for char in normalized):
            return OcrStabilityDecision("drop", reason="invalid OCR candidate")

        low_quality = _looks_like_low_quality_ocr_candidate(text)
        if self.pending_text is None:
            if low_quality:
                return self._queue_pending(text, normalized, now, "low quality")
            return OcrStabilityDecision(
                "submit",
                text=text,
                reason="clear candidate",
                requested_at_monotonic=now,
            )

        age = max(0.0, now - self.pending_first_seen_monotonic)
        same_candidate = normalized == self.pending_norm
        near_same_candidate = (
            not same_candidate
            and SequenceMatcher(None, normalized, self.pending_norm).ratio() >= 0.98
        )
        if same_candidate or near_same_candidate:
            if _ocr_candidate_quality_score(text) > _ocr_candidate_quality_score(self.pending_text):
                self.pending_text = text
                self.pending_norm = normalized
            self.pending_last_seen_monotonic = now
            if age >= self.min_wait_seconds:
                return OcrStabilityDecision(
                    "submit",
                    text=self.pending_text,
                    reason="candidate repeated",
                    requested_at_monotonic=self.pending_first_seen_monotonic,
                )
            return OcrStabilityDecision(
                "wait",
                text=self.pending_text,
                delay_seconds=self.remaining_wait_seconds(now),
                reason="candidate waiting for confirmation",
                requested_at_monotonic=self.pending_first_seen_monotonic,
            )

        if _looks_like_more_complete_ocr_candidate(text, self.pending_text):
            if not low_quality:
                return OcrStabilityDecision(
                    "submit",
                    text=text,
                    reason="more complete candidate",
                    requested_at_monotonic=now,
                )
            return self._queue_pending(
                text,
                normalized,
                now,
                "more complete but still low quality",
                keep_first_seen=True,
            )

        if not low_quality:
            return OcrStabilityDecision(
                "submit",
                text=text,
                reason="new clear candidate",
                requested_at_monotonic=now,
            )

        if age >= self.max_wait_seconds:
            best_text = self.pending_text
            if _ocr_candidate_quality_score(text) > _ocr_candidate_quality_score(best_text):
                best_text = text
            return OcrStabilityDecision(
                "submit",
                text=best_text,
                reason="max wait reached",
                requested_at_monotonic=self.pending_first_seen_monotonic,
            )

        if _ocr_candidate_quality_score(text) > _ocr_candidate_quality_score(self.pending_text):
            return self._queue_pending(
                text,
                normalized,
                now,
                "better low-quality candidate",
                keep_first_seen=True,
            )
        return OcrStabilityDecision(
            "wait",
            text=self.pending_text,
            delay_seconds=self.remaining_wait_seconds(now),
            reason="waiting for stable OCR",
            requested_at_monotonic=self.pending_first_seen_monotonic,
        )

    def consume_pending_for_flush(self):
        if self.pending_text is None:
            return None, None

        text = self.pending_text
        requested_at = self.pending_first_seen_monotonic
        normalized = self.pending_norm
        self.clear()
        if not normalized or not any(char.isalnum() for char in normalized):
            return None, None
        return text, requested_at


def _get_local_ocr_submit_scope(app, text_to_translate):
    handler = getattr(app, "translation_handler", None)
    getter = getattr(handler, "get_inflight_translation_key", None)
    if not callable(getter):
        return None

    try:
        inflight_key = getter(text_to_translate)
    except Exception as scope_error:
        _log_debug(
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


def _get_ocr_stability_gate(app):
    gate = getattr(app, "ocr_stability_gate", None)
    if gate is None or not hasattr(gate, "evaluate"):
        gate = OcrStabilityGate()
        try:
            app.ocr_stability_gate = gate
        except Exception:
            pass
    return gate


def _has_pending_ocr_stability_candidate(app):
    gate = getattr(app, "ocr_stability_gate", None)
    checker = getattr(gate, "has_pending", None)
    if callable(checker):
        try:
            return bool(checker())
        except Exception:
            return False
    return bool(getattr(gate, "pending_text", None))


def _translation_display_epoch(app):
    return (
        float(
            getattr(app, "last_successful_translation_time", 0.0)
            or 0.0
        ),
        int(
            getattr(app, "last_displayed_translation_sequence", 0)
            or 0
        ),
    )


def _advance_translation_display_activity(
    app,
    now,
    last_display_time,
    last_display_sequence,
):
    successful_display_time, displayed_sequence = (
        _translation_display_epoch(app)
    )
    if displayed_sequence != last_display_sequence:
        return (
            max(
                float(last_display_time),
                successful_display_time,
                float(now),
            ),
            displayed_sequence,
        )
    return (
        max(float(last_display_time), successful_display_time),
        last_display_sequence,
    )


def _translation_clear_blocker(app):
    if getattr(app, "is_running", True) is False:
        return "app stopped"
    if str(getattr(app, "previous_text", "") or "").strip():
        return "source text present"
    if getattr(app, "active_translation_calls", None):
        return "active translation calls"
    if getattr(app, "active_translation_inflight_keys", None):
        return "active translation identities"
    if getattr(app, "pending_translation_request", None) is not None:
        return "pending translation request"
    if getattr(app, "pending_translation_flush_scheduled", False):
        return "pending translation flush"
    if _has_pending_ocr_stability_candidate(app):
        return "pending OCR stability candidate"

    translation_queue = getattr(app, "translation_queue", None)
    if translation_queue is not None:
        try:
            if not translation_queue.empty():
                return "legacy translation queue"
        except Exception:
            pass
    return None


def _apply_inactive_translation_clear(
    app,
    expected_epoch,
    inactive_duration,
    timeout_seconds,
):
    if (
        getattr(
            app,
            "translation_inactivity_clear_scheduled_epoch",
            None,
        )
        == expected_epoch
    ):
        app.translation_inactivity_clear_scheduled_epoch = None

    current_epoch = _translation_display_epoch(app)
    if current_epoch != expected_epoch:
        _log_debug(
            "LATENCY: skipped stale translation inactivity clear "
            f"expected_epoch={expected_epoch} current_epoch={current_epoch}"
        )
        return False

    blocker = _translation_clear_blocker(app)
    if blocker:
        _log_debug(
            "LATENCY: deferred translation inactivity clear on UI thread "
            f"reason={blocker}"
        )
        return False

    try:
        display_manager = getattr(app, "display_manager", None)
        direct_updater = getattr(
            display_manager,
            "_update_translation_text_on_main_thread",
            None,
        )
        if callable(direct_updater):
            direct_updater("")
        else:
            app.update_translation_text("")
    except Exception as clear_error:
        _log_debug(
            "LATENCY: failed to apply translation inactivity clear: "
            f"{type(clear_error).__name__} - {clear_error}"
        )
        return False

    _clear_local_ocr_submit_state(app)
    app.translation_inactivity_cleared_epoch = expected_epoch
    _increment_metric(app, "translation_inactivity_clear")
    _log_debug(
        "WT: Cleared translation once after sustained inactivity "
        f"duration={float(inactive_duration):.1f}s "
        f"timeout={float(timeout_seconds):.1f}s"
    )
    return True


def _schedule_inactive_translation_clear(
    app,
    inactive_duration,
    timeout_seconds,
):
    expected_epoch = _translation_display_epoch(app)
    if expected_epoch == (0.0, 0):
        return False
    if (
        getattr(app, "translation_inactivity_cleared_epoch", None)
        == expected_epoch
    ):
        return False
    if (
        getattr(
            app,
            "translation_inactivity_clear_scheduled_epoch",
            None,
        )
        == expected_epoch
    ):
        return False
    if _translation_clear_blocker(app):
        return False

    app.translation_inactivity_clear_scheduled_epoch = expected_epoch
    try:
        app.root.after(
            0,
            _apply_inactive_translation_clear,
            app,
            expected_epoch,
            float(inactive_duration),
            float(timeout_seconds),
        )
    except Exception as schedule_error:
        if (
            getattr(
                app,
                "translation_inactivity_clear_scheduled_epoch",
                None,
            )
            == expected_epoch
        ):
            app.translation_inactivity_clear_scheduled_epoch = None
        _log_debug(
            "LATENCY: failed to schedule translation inactivity clear: "
            f"{type(schedule_error).__name__} - {schedule_error}"
        )
        return False
    return True


def _clear_ocr_stability_gate(app, reason):
    gate = getattr(app, "ocr_stability_gate", None)
    clearer = getattr(gate, "clear", None)
    if not callable(clearer):
        return False
    try:
        had_pending = bool(clearer())
    except Exception as clear_error:
        _log_debug(
            "LATENCY: failed to clear OCR stability gate "
            f"reason={reason}: {type(clear_error).__name__} - {clear_error}"
        )
        return False
    if had_pending:
        _log_debug(f"LATENCY: cleared pending OCR stability candidate reason={reason}")
    return had_pending


def enqueue_ocr_frame_for_model(app, screenshot, ocr_model):
    ocr_queue = app.ocr_queue
    if ocr_model == PADDLEOCR_MODEL_CODE:
        try:
            while True:
                ocr_queue.get_nowait()
        except queue.Empty:
            pass

    try:
        if not ocr_queue.full():
            ocr_queue.put_nowait(screenshot)
            _refresh_ocr_queue_metric(app)
            return True
        if ocr_model == PADDLEOCR_MODEL_CODE:
            try:
                ocr_queue.get_nowait()
            except queue.Empty:
                pass
            ocr_queue.put_nowait(screenshot)
            _refresh_ocr_queue_metric(app)
            return True
    except queue.Full:
        _refresh_ocr_queue_metric(app)
        return False
    _refresh_ocr_queue_metric(app)
    return False


def _submit_final_local_ocr_text(
    app,
    text_to_translate,
    ocr_sequence_number,
    requested_at_monotonic=None,
):
    if _should_skip_local_ocr_resubmit(app, text_to_translate):
        app.reset_clear_timeout()
        _log_debug_coalesced(
            "local-ocr-near-duplicate-skip",
            "LATENCY: skipped local OCR resubmit for near-duplicate stable text: "
            f"{summarize_text_for_log(text_to_translate)}",
            interval_seconds=5.0,
        )
        return "skipped"

    _start_async_translation(
        app,
        text_to_translate,
        ocr_sequence_number,
        requested_at_monotonic=requested_at_monotonic,
    )
    return "submitted"


def _schedule_ocr_stability_flush(app, gate, delay_seconds):
    desired_deadline = time.monotonic() + max(0.0, float(delay_seconds or 0.0))
    current_deadline = float(getattr(gate, "flush_deadline_monotonic", 0.0) or 0.0)
    if (
        getattr(gate, "flush_scheduled", False)
        and current_deadline > 0.0
        and current_deadline <= desired_deadline
    ):
        return

    gate.generation += 1
    generation = gate.generation
    gate.flush_scheduled = True
    gate.flush_deadline_monotonic = desired_deadline
    remaining_seconds = max(0.0, desired_deadline - time.monotonic())
    delay_ms = max(1, int(remaining_seconds * 1000))
    try:
        app.root.after(delay_ms, _flush_ocr_stability_candidate, app, generation)
    except Exception:
        if int(getattr(gate, "generation", 0) or 0) == generation:
            gate.flush_scheduled = False
            gate.flush_deadline_monotonic = 0.0
        raise


def _flush_ocr_stability_candidate(app, generation=None):
    gate = getattr(app, "ocr_stability_gate", None)
    if gate is None:
        return
    if generation is not None and generation != getattr(gate, "generation", None):
        _log_debug(
            "LATENCY: ignored stale OCR stability timer "
            f"generation={generation} current={getattr(gate, 'generation', None)}"
        )
        return

    gate.flush_scheduled = False
    gate.flush_deadline_monotonic = 0.0
    if hasattr(app, 'is_running') and not app.is_running:
        _clear_ocr_stability_gate(app, "app stopped")
        return

    consumer = getattr(gate, "consume_pending_for_flush", None)
    if not callable(consumer):
        return
    text_to_translate, requested_at = consumer()
    if not text_to_translate:
        _log_debug("LATENCY: dropped invalid OCR stability candidate on flush")
        return

    _submit_final_local_ocr_text(
        app,
        text_to_translate,
        0,
        requested_at_monotonic=requested_at,
    )


def _route_local_ocr_candidate_for_translation(
    app,
    text_to_translate,
    now=None,
    ocr_sequence_number=0,
):
    if now is None:
        now = time.monotonic()

    if _should_skip_local_ocr_resubmit(app, text_to_translate):
        _clear_ocr_stability_gate(app, "near-duplicate local OCR")
        app.reset_clear_timeout()
        _log_debug_coalesced(
            "local-ocr-near-duplicate-skip",
            "LATENCY: skipped local OCR resubmit for near-duplicate stable text: "
            f"{summarize_text_for_log(text_to_translate)}",
            interval_seconds=5.0,
        )
        return "skipped"

    gate = _get_ocr_stability_gate(app)
    decision = gate.evaluate(text_to_translate, now=now)
    if decision.action == "submit":
        _clear_ocr_stability_gate(app, decision.reason or "OCR candidate submitted")
        return _submit_final_local_ocr_text(
            app,
            decision.text,
            ocr_sequence_number,
            requested_at_monotonic=decision.requested_at_monotonic,
        )
    if decision.action == "drop":
        _clear_ocr_stability_gate(app, decision.reason or "OCR candidate dropped")
        _log_debug_coalesced(
            "local-ocr-stability-drop",
            "LATENCY: dropped OCR stability candidate "
            f"reason={decision.reason} "
            f"{summarize_text_for_log(text_to_translate)}",
            interval_seconds=5.0,
        )
        return "dropped"

    _schedule_ocr_stability_flush(app, gate, decision.delay_seconds)
    _log_debug_coalesced(
        "local-ocr-stability-pending",
        "LATENCY: pending OCR stability candidate "
        f"delay={decision.delay_seconds:.3f}s reason={decision.reason}: "
        f"{summarize_text_for_log(decision.text)}",
        interval_seconds=5.0,
    )
    return "pending"


def _get_api_ocr_cache_mode_key(app, provider_name=None, image_size=None):
    keep_linebreaks_var = getattr(app, 'keep_linebreaks_var', None)
    parts = ['api']
    if keep_linebreaks_var is None:
        keep_linebreaks = None
    else:
        try:
            keep_linebreaks = bool(keep_linebreaks_var.get())
        except Exception:
            keep_linebreaks = False
        parts.append(f"keep_linebreaks={keep_linebreaks}")

    image_decision_getter = getattr(app, "get_ai_ocr_image_decision", None)
    if callable(image_decision_getter):
        try:
            image_decision = image_decision_getter(image_size=image_size)
        except Exception:
            image_decision = None
        if image_decision is not None:
            parts.append(f"image_contract={image_decision.contract_key}")

    provider_key = str(provider_name or '').strip().lower()
    if provider_key == 'custom_ai' or (provider_name is None and hasattr(app, 'custom_ai_profiles')):
        reasoning_contract = _get_custom_ai_ocr_reasoning_contract(app)
        if reasoning_contract is not None:
            parts.append(f"reasoning_effort={reasoning_contract}")

    return "|".join(parts)


