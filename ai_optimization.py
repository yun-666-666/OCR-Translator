"""Unified Custom AI response and OCR image optimization policy."""

from dataclasses import dataclass
import threading
from urllib.parse import urlsplit

from ocr_utils import (
    normalize_api_ocr_image_detail,
    normalize_api_ocr_image_format,
    normalize_api_ocr_image_mode,
    normalize_api_ocr_image_quality,
)


AI_OPTIMIZATION_AUTO = "auto"
AI_OPTIMIZATION_SPEED = "speed"
AI_OPTIMIZATION_QUALITY = "quality"
AI_OPTIMIZATION_MODES = {
    AI_OPTIMIZATION_AUTO,
    AI_OPTIMIZATION_SPEED,
    AI_OPTIMIZATION_QUALITY,
}

LEGACY_AI_SETTING_KEYS = (
    "custom_ai_latency_mode",
    "custom_ai_ocr_image_format",
    "custom_ai_ocr_image_mode",
    "custom_ai_ocr_image_quality",
    "custom_ai_ocr_image_detail",
)

LEGACY_STABILITY_DEFAULT = "2"
OPTIMIZED_STABILITY_DEFAULT = "0"
LEGACY_PADDLEOCR_MIN_SCORE_DEFAULT = "0.35"
OPTIMIZED_PADDLEOCR_MIN_SCORE_DEFAULT = "0.45"


def normalize_ai_optimization_mode(value):
    normalized = str(value or AI_OPTIMIZATION_AUTO).strip().lower()
    if normalized in AI_OPTIMIZATION_MODES:
        return normalized
    return AI_OPTIMIZATION_AUTO


def migrate_legacy_ai_optimization_settings(settings):
    """Migrate the five legacy AI settings into one canonical policy."""
    if "ai_optimization_mode" in settings:
        mode = normalize_ai_optimization_mode(settings.get("ai_optimization_mode"))
    else:
        image_format = normalize_api_ocr_image_format(
            settings.get("custom_ai_ocr_image_format", "webp")
        )
        image_mode = normalize_api_ocr_image_mode(
            settings.get("custom_ai_ocr_image_mode", "balanced_webp")
        )
        image_quality = normalize_api_ocr_image_quality(
            settings.get("custom_ai_ocr_image_quality", 85)
        )
        image_detail = normalize_api_ocr_image_detail(
            settings.get("custom_ai_ocr_image_detail", "auto")
        )
        if (
            image_detail == "high"
            or image_format == "png"
            or image_mode == "lossless_webp"
            or image_quality >= 90
        ):
            mode = AI_OPTIMIZATION_QUALITY
        elif (
            image_detail == "low"
            or image_mode == "small_grayscale_webp"
            or image_quality <= 75
        ):
            mode = AI_OPTIMIZATION_SPEED
        else:
            mode = AI_OPTIMIZATION_AUTO

    settings["ai_optimization_mode"] = mode
    for legacy_key in LEGACY_AI_SETTING_KEYS:
        settings.pop(legacy_key, None)
    settings.pop("capture_backend", None)
    return mode


def migrate_legacy_ocr_defaults(settings):
    """Upgrade only values that exactly match the former shipped defaults."""
    changed = False
    if str(settings.get("stability_threshold", "")).strip() == LEGACY_STABILITY_DEFAULT:
        settings["stability_threshold"] = OPTIMIZED_STABILITY_DEFAULT
        changed = True
    if (
        str(settings.get("paddleocr_min_score", "")).strip()
        == LEGACY_PADDLEOCR_MIN_SCORE_DEFAULT
    ):
        settings["paddleocr_min_score"] = OPTIMIZED_PADDLEOCR_MIN_SCORE_DEFAULT
        changed = True
    return changed


def resolve_ai_response_mode(optimization_mode):
    if normalize_ai_optimization_mode(optimization_mode) == AI_OPTIMIZATION_AUTO:
        return "adaptive"
    return "safe"


def _profile_route_key(profile):
    profile = profile if isinstance(profile, dict) else {}
    return (
        str(profile.get("base_url") or "").strip().lower().rstrip("/"),
        str(profile.get("model") or "").strip().lower(),
        str(profile.get("wire_api") or "chat_completions").strip().lower(),
    )


def _is_direct_xai_profile(profile):
    if not isinstance(profile, dict):
        return False
    try:
        hostname = (urlsplit(str(profile.get("base_url") or "")).hostname or "").lower()
    except (TypeError, ValueError):
        return False
    return hostname == "api.x.ai" or hostname.endswith(".x.ai")


class AiOcrImageCapabilityMemory:
    """Remember route-scoped image formats rejected by an upstream API."""

    def __init__(self):
        self._unsupported_formats = set()
        self._lock = threading.RLock()

    def mark_format_unsupported(self, profile, image_format):
        normalized_format = normalize_api_ocr_image_format(image_format)
        with self._lock:
            self._unsupported_formats.add(
                (_profile_route_key(profile), normalized_format)
            )

    def is_format_unsupported(self, profile, image_format):
        normalized_format = normalize_api_ocr_image_format(image_format)
        with self._lock:
            return (
                _profile_route_key(profile),
                normalized_format,
            ) in self._unsupported_formats


@dataclass(frozen=True)
class AiOcrImageDecision:
    image_format: str
    image_mode: str
    image_quality: int
    image_detail: str
    reason: str

    @property
    def contract_key(self):
        return (
            f"{self.image_format}|{self.image_mode}|"
            f"{self.image_quality}|{self.image_detail}"
        )


def resolve_ai_ocr_image_policy(
    optimization_mode,
    profile=None,
    image_size=None,
    route_p90_seconds=0.0,
    route_sample_count=0,
    capability_memory=None,
):
    """Return a deterministic OCR image contract for the current request."""
    policy = normalize_ai_optimization_mode(optimization_mode)
    try:
        width, height = image_size or (0, 0)
        width = max(0, int(width))
        height = max(0, int(height))
    except (TypeError, ValueError):
        width, height = 0, 0
    try:
        route_p90 = max(0.0, float(route_p90_seconds or 0.0))
    except (TypeError, ValueError):
        route_p90 = 0.0
    try:
        route_samples = max(0, int(route_sample_count or 0))
    except (TypeError, ValueError):
        route_samples = 0

    if policy == AI_OPTIMIZATION_SPEED:
        image_format = "webp"
        image_mode = "small_grayscale_webp"
        image_quality = 72
        image_detail = "low"
        reason = "speed_policy"
    elif policy == AI_OPTIMIZATION_QUALITY:
        image_format = "png"
        image_mode = "lossless_webp"
        image_quality = 100
        image_detail = "high"
        reason = "quality_policy"
    else:
        image_format = "webp"
        image_mode = "balanced_webp"
        image_quality = 85
        image_detail = "auto"
        reason = "auto_balanced"
        if route_samples >= 5 and route_p90 >= 3.5:
            image_mode = "small_grayscale_webp"
            image_quality = 75
            image_detail = "low"
            reason = "auto_slow_route"
        elif height and (
            height <= 160
            or (width >= 960 and width / max(1, height) >= 7.0)
        ):
            image_quality = 90
            image_detail = "high"
            reason = "auto_thin_text"

    if _is_direct_xai_profile(profile):
        image_format = "jpeg"
        if policy == AI_OPTIMIZATION_SPEED:
            image_quality = 78
        elif policy == AI_OPTIMIZATION_QUALITY:
            image_quality = 95
        reason += "_xai_jpeg"

    if (
        capability_memory is not None
        and capability_memory.is_format_unsupported(profile, image_format)
    ):
        if image_format != "jpeg" and not capability_memory.is_format_unsupported(
            profile, "jpeg"
        ):
            image_format = "jpeg"
        else:
            image_format = "png"
        reason += "_capability_fallback"

    return AiOcrImageDecision(
        image_format=normalize_api_ocr_image_format(image_format),
        image_mode=normalize_api_ocr_image_mode(image_mode),
        image_quality=normalize_api_ocr_image_quality(image_quality),
        image_detail=normalize_api_ocr_image_detail(image_detail),
        reason=reason,
    )


def looks_like_unsupported_image_format_error(error_text, image_format):
    text = str(error_text or "").strip().lower()
    normalized_format = normalize_api_ocr_image_format(image_format)
    if normalized_format not in text:
        return False
    return any(
        marker in text
        for marker in (
            "unsupported image",
            "unsupported format",
            "invalid image format",
            "image format is not supported",
            "only jpeg",
            "only jpg",
            "only png",
        )
    )
