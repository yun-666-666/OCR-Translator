"""Custom AI normalization helpers and adaptive latency policy."""

from collections import deque
import math
import sys
import threading
import time

ACTIVE_PROFILE_KINDS = {"translation", "ocr"}
CUSTOM_AI_CREDENTIAL_SERVICE = "OCR-Translator-CustomAI"
CUSTOM_AI_PROFILE_TEMP_STALE_SECONDS = 300.0
CUSTOM_AI_LATENCY_MODE_NONE = "none"
CUSTOM_AI_LATENCY_MODE_SAFE = "safe"
CUSTOM_AI_LATENCY_MODE_STREAM = "stream"
CUSTOM_AI_LATENCY_MODE_RACE = "race"
CUSTOM_AI_LATENCY_MODE_ADAPTIVE = "adaptive"
CUSTOM_AI_LATENCY_MODES = {
    CUSTOM_AI_LATENCY_MODE_NONE,
    CUSTOM_AI_LATENCY_MODE_SAFE,
    CUSTOM_AI_LATENCY_MODE_STREAM,
    CUSTOM_AI_LATENCY_MODE_RACE,
    CUSTOM_AI_LATENCY_MODE_ADAPTIVE,
}
CUSTOM_AI_SAFE_TRANSPORT_MODES = {
    CUSTOM_AI_LATENCY_MODE_SAFE,
    CUSTOM_AI_LATENCY_MODE_STREAM,
    CUSTOM_AI_LATENCY_MODE_RACE,
}
CUSTOM_AI_WIRE_API_CHAT_COMPLETIONS = "chat_completions"
CUSTOM_AI_WIRE_API_RESPONSES = "responses"
CUSTOM_AI_WIRE_APIS = {
    CUSTOM_AI_WIRE_API_CHAT_COMPLETIONS,
    CUSTOM_AI_WIRE_API_RESPONSES,
}
CUSTOM_AI_STRUCTURED_OUTPUT_OFF = "off"
CUSTOM_AI_STRUCTURED_OUTPUT_AUTO = "auto"
CUSTOM_AI_STRUCTURED_OUTPUT_STRICT = "strict"
CUSTOM_AI_STRUCTURED_OUTPUT_MODES = {
    CUSTOM_AI_STRUCTURED_OUTPUT_OFF,
    CUSTOM_AI_STRUCTURED_OUTPUT_AUTO,
    CUSTOM_AI_STRUCTURED_OUTPUT_STRICT,
}
CUSTOM_AI_STRUCTURED_OUTPUT_CONTRACT_TEXT = "text"
CUSTOM_AI_STRUCTURED_OUTPUT_CONTRACT_JSON_SCHEMA = "json_schema"
CUSTOM_AI_REASONING_EFFORT_LOW = "low"
CUSTOM_AI_REASONING_EFFORT_MEDIUM = "medium"
CUSTOM_AI_REASONING_EFFORT_HIGH = "high"
CUSTOM_AI_REASONING_EFFORT_ULTRA = "ultra"
CUSTOM_AI_REASONING_EFFORT_NONE = "none"
CUSTOM_AI_REASONING_EFFORTS = {
    CUSTOM_AI_REASONING_EFFORT_NONE,
    CUSTOM_AI_REASONING_EFFORT_LOW,
    CUSTOM_AI_REASONING_EFFORT_MEDIUM,
    CUSTOM_AI_REASONING_EFFORT_HIGH,
    CUSTOM_AI_REASONING_EFFORT_ULTRA,
}
CUSTOM_AI_REASONING_EFFORT_CONTRACTS = {
    *CUSTOM_AI_REASONING_EFFORTS,
    CUSTOM_AI_REASONING_EFFORT_NONE,
}
CUSTOM_AI_REASONING_REQUEST_KINDS = {
    "translation",
    "ocr",
}
DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS = 15.0
TRANSLATION_MIN_OUTPUT_TOKENS = 64
TRANSLATION_MAX_OUTPUT_TOKENS = 2048
TRANSLATION_OUTPUT_TOKENS_PER_CHAR = 4
TRANSLATION_OUTPUT_WRAPPER_LABELS = frozenset({
    "translation:",
    "translation：",
    "translated text:",
    "translated text：",
    "translation result:",
    "translation result：",
    "译文:",
    "译文：",
    "翻译:",
    "翻译：",
    "翻译结果:",
    "翻译结果：",
})
TRANSLATION_OUTPUT_PREAMBLES = (
    "sure, here is the translation:",
    "sure, here's the translation:",
    "certainly, here is the translation:",
    "certainly, here's the translation:",
    "here is the translation:",
    "here's the translation:",
)



def _log_debug(message):
    facade = sys.modules.get("custom_ai")
    if facade is not None:
        return facade.log_debug(message)


def normalize_custom_ai_latency_mode(mode):
    mode = str(mode or "").strip().lower()
    if mode in CUSTOM_AI_LATENCY_MODES:
        return mode
    return CUSTOM_AI_LATENCY_MODE_SAFE


def normalize_custom_ai_wire_api(wire_api):
    wire_api = str(wire_api or "").strip().lower().replace("-", "_")
    if wire_api in {"response", "responses", "openai_responses"}:
        return CUSTOM_AI_WIRE_API_RESPONSES
    if wire_api in {"chat", "chat_completion", "chat_completions", "openai_chat_completions"}:
        return CUSTOM_AI_WIRE_API_CHAT_COMPLETIONS
    return CUSTOM_AI_WIRE_API_CHAT_COMPLETIONS


def normalize_custom_ai_structured_output_mode(mode):
    mode = str(mode or "").strip().lower().replace("-", "_")
    if mode in CUSTOM_AI_STRUCTURED_OUTPUT_MODES:
        return mode
    return CUSTOM_AI_STRUCTURED_OUTPUT_AUTO


def normalize_custom_ai_reasoning_effort(value):
    effort = str(value or "").strip().lower().replace("-", "_")
    if effort in CUSTOM_AI_REASONING_EFFORTS:
        return effort
    return CUSTOM_AI_REASONING_EFFORT_LOW


def normalize_custom_ai_reasoning_request_kind(request_kind):
    kind = str(request_kind or "").strip().lower()
    if kind in CUSTOM_AI_REASONING_REQUEST_KINDS:
        return kind
    return ""


def build_translation_json_schema():
    return {
        "type": "object",
        "properties": {
            "translation": {"type": "string"},
        },
        "required": ["translation"],
        "additionalProperties": False,
    }


def build_translation_response_format():
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "translation_result",
            "strict": True,
            "schema": build_translation_json_schema(),
        },
    }


class CustomAILatencyModeDecision:
    def __init__(
        self,
        mode,
        reason,
        p90_seconds=0.0,
        sample_count=0,
    ):
        self.mode = mode
        self.reason = reason
        self.p90_seconds = p90_seconds
        self.sample_count = sample_count


class CustomAIRequestTimeoutDecision:
    def __init__(
        self,
        seconds,
        reason,
        p90_seconds=0.0,
        sample_count=0,
    ):
        self.seconds = seconds
        self.reason = reason
        self.p90_seconds = p90_seconds
        self.sample_count = sample_count


class CustomAILatencyModeAdvisor:
    """Conservatively resolve adaptive Custom AI latency mode."""

    def __init__(
        self,
        max_samples=20,
        min_samples=3,
        stream_latency_threshold_seconds=1.5,
        race_latency_threshold_seconds=3.5,
        min_hold_seconds=10.0,
        race_cooldown_seconds=30.0,
        consecutive_error_threshold=2,
        clock=None,
    ):
        self.max_samples = max(1, int(max_samples))
        self.min_samples = max(1, int(min_samples))
        self.stream_latency_threshold_seconds = max(
            0.0,
            float(stream_latency_threshold_seconds),
        )
        self.race_latency_threshold_seconds = max(
            self.stream_latency_threshold_seconds,
            float(race_latency_threshold_seconds),
        )
        self.min_hold_seconds = max(0.0, float(min_hold_seconds))
        self.race_cooldown_seconds = max(0.0, float(race_cooldown_seconds))
        self.consecutive_error_threshold = max(
            1,
            int(consecutive_error_threshold),
        )
        self._clock = clock or time.monotonic
        self._durations = deque(maxlen=self.max_samples)
        self._consecutive_errors = 0
        self._last_mode = CUSTOM_AI_LATENCY_MODE_SAFE
        self._last_mode_changed_at = 0.0
        self._last_race_at = None
        self._lock = threading.RLock()

    def observe_request(self, duration_seconds=None, success=True):
        with self._lock:
            try:
                duration = float(duration_seconds)
            except (TypeError, ValueError):
                duration = None
            if duration is not None and duration >= 0.0:
                self._durations.append(duration)
            if success:
                self._consecutive_errors = 0
            else:
                self._consecutive_errors += 1

    def resolve(
        self,
        configured_mode,
        stream_supported=False,
        healthy_race_profile_count=1,
        primary_cooldown_seconds=0.0,
        commit=False,
    ):
        configured_mode = normalize_custom_ai_latency_mode(configured_mode)
        if configured_mode != CUSTOM_AI_LATENCY_MODE_ADAPTIVE:
            return CustomAILatencyModeDecision(
                configured_mode,
                "configured",
                0.0,
                0,
            )

        with self._lock:
            p90_seconds = self._p90_locked()
            sample_count = len(self._durations)
            mode, reason = self._choose_adaptive_locked(
                p90_seconds,
                sample_count,
                bool(stream_supported),
                max(0, int(healthy_race_profile_count or 0)),
                max(0.0, float(primary_cooldown_seconds or 0.0)),
            )
            decision = CustomAILatencyModeDecision(
                mode,
                reason,
                p90_seconds,
                sample_count,
            )
            if commit:
                self._commit_mode_locked(mode)
            return decision

    def commit_mode(self, mode):
        mode = normalize_custom_ai_latency_mode(mode)
        with self._lock:
            self._commit_mode_locked(mode)

    def resolve_request_timeout(
        self,
        configured_timeout_seconds,
        latency_mode=CUSTOM_AI_LATENCY_MODE_SAFE,
        min_samples=8,
        floor_seconds=4.0,
        p90_multiplier=4.0,
    ):
        try:
            configured_timeout = float(configured_timeout_seconds)
        except (TypeError, ValueError):
            configured_timeout = 10.0
        if not math.isfinite(configured_timeout) or configured_timeout <= 0.0:
            configured_timeout = 10.0

        latency_mode = normalize_custom_ai_latency_mode(latency_mode)
        with self._lock:
            p90_seconds = self._p90_locked()
            sample_count = len(self._durations)

            if latency_mode in {
                CUSTOM_AI_LATENCY_MODE_STREAM,
                CUSTOM_AI_LATENCY_MODE_RACE,
            }:
                return CustomAIRequestTimeoutDecision(
                    configured_timeout,
                    "mode_uses_full_timeout",
                    p90_seconds,
                    sample_count,
                )

            if sample_count < max(1, int(min_samples)):
                return CustomAIRequestTimeoutDecision(
                    configured_timeout,
                    "insufficient_samples",
                    p90_seconds,
                    sample_count,
                )

            if self._consecutive_errors > 0:
                return CustomAIRequestTimeoutDecision(
                    configured_timeout,
                    "recent_error_probe",
                    p90_seconds,
                    sample_count,
                )

            candidate = max(
                0.0,
                float(floor_seconds),
                p90_seconds * max(1.0, float(p90_multiplier)),
            )
            if candidate >= configured_timeout:
                return CustomAIRequestTimeoutDecision(
                    configured_timeout,
                    "route_requires_full_timeout",
                    p90_seconds,
                    sample_count,
                )

            return CustomAIRequestTimeoutDecision(
                candidate,
                "fast_route_tail_guard",
                p90_seconds,
                sample_count,
            )

    def _choose_adaptive_locked(
        self,
        p90_seconds,
        sample_count,
        stream_supported,
        healthy_race_profile_count,
        primary_cooldown_seconds,
    ):
        now = float(self._clock())
        if (
            primary_cooldown_seconds > 0.0
            and healthy_race_profile_count <= 0
        ):
            return CUSTOM_AI_LATENCY_MODE_SAFE, "cooldown"

        if self._consecutive_errors > 0:
            return CUSTOM_AI_LATENCY_MODE_SAFE, "recent_error"

        if (
            primary_cooldown_seconds > 0.0
            and healthy_race_profile_count > 0
        ):
            return CUSTOM_AI_LATENCY_MODE_SAFE, "cooldown_alternative_safe"

        if sample_count < self.min_samples:
            return CUSTOM_AI_LATENCY_MODE_SAFE, "insufficient_samples"

        if (
            p90_seconds >= self.race_latency_threshold_seconds
            and healthy_race_profile_count >= 2
        ):
            if self._race_available_locked(now):
                return CUSTOM_AI_LATENCY_MODE_RACE, "p90_very_high"
            return CUSTOM_AI_LATENCY_MODE_SAFE, "race_cooldown"

        if p90_seconds >= self.stream_latency_threshold_seconds:
            if stream_supported:
                return CUSTOM_AI_LATENCY_MODE_STREAM, "p90_high_stream"
            return CUSTOM_AI_LATENCY_MODE_SAFE, "high_latency_safe"

        return CUSTOM_AI_LATENCY_MODE_SAFE, "low_latency"

    def _race_available_locked(self, now):
        if self._last_race_at is None:
            return True
        return now - self._last_race_at >= self.race_cooldown_seconds

    def _commit_mode_locked(self, mode):
        now = float(self._clock())
        if mode != self._last_mode:
            self._last_mode = mode
            self._last_mode_changed_at = now
        if mode == CUSTOM_AI_LATENCY_MODE_RACE:
            self._last_race_at = now

    def _p90_locked(self):
        if not self._durations:
            return 0.0
        ordered = sorted(self._durations)
        rank = int(math.ceil(0.90 * len(ordered)))
        return ordered[max(0, min(len(ordered) - 1, rank - 1))]
