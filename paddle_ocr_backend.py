import os
import platform
import re
import sys
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from logger import log_debug, log_debug_coalesced, summarize_text_for_log


PADDLEOCR_MODEL_CODE = "paddleocr"
PADDLEOCR_DISPLAY_NAME = "PaddleOCR PP-OCRv6 (offline)"
PADDLEOCR_AUTO_LANG = "auto"
PADDLEOCR_SUPPORTED_VERSIONS = (
    "PP-OCRv3",
    "PP-OCRv4",
    "PP-OCRv5",
    "PP-OCRv6",
)
PADDLEOCR_MODEL_SIZES = ("tiny", "small", "medium")
PADDLEOCR_DEVICE_OPTIONS = ("cpu", "gpu", "gpu:0")
PADDLEOCR_TEXT_DET_LIMIT_TYPES = ("max", "min")

# Kept in sync with PaddleOCR 3.x's public language groups.  The UI exposes
# PaddleOCR language codes directly so a saved value has one unambiguous
# runtime meaning.
_PADDLEOCR_LATIN_LANGS = frozenset({
    "af", "az", "bs", "ca", "cs", "cy", "da", "de", "es", "et",
    "eu", "fi", "fr", "french", "ga", "german", "gl", "hr", "hu",
    "id", "is", "it", "la", "lb", "lt", "lv", "mi", "ms", "mt",
    "nl", "no", "oc", "pl", "pt", "qu", "rm", "ro", "rs_latin",
    "sk", "sl", "sq", "sv", "sw", "tl", "tr", "uz", "vi",
})
_PADDLEOCR_ARABIC_LANGS = frozenset({"ar", "fa", "ug", "ur", "ps", "ku", "sd", "bal"})
_PADDLEOCR_ESLAV_LANGS = frozenset({"ru", "be", "uk"})
_PADDLEOCR_CYRILLIC_LANGS = frozenset({
    "ru", "rs_cyrillic", "be", "bg", "uk", "mn", "kk", "ky", "tg",
    "mk", "tt", "cv", "ba", "mo", "os",
})
_PADDLEOCR_DEVANAGARI_LANGS = frozenset({
    "hi", "mr", "ne", "bh", "mai", "bho", "new", "gom", "sa",
})
_PADDLEOCR_SPECIFIC_LANGS = frozenset({
    "ch", "en", "korean", "japan", "chinese_cht", "te", "ka", "ta",
})
_PADDLEOCR_V6_LANGS = frozenset({"ch", "chinese_cht", "en", "japan"}) | _PADDLEOCR_LATIN_LANGS
_PADDLEOCR_V5_LANGS = (
    _PADDLEOCR_V6_LANGS
    | _PADDLEOCR_ARABIC_LANGS
    | _PADDLEOCR_ESLAV_LANGS
    | _PADDLEOCR_CYRILLIC_LANGS
    | _PADDLEOCR_DEVANAGARI_LANGS
    | frozenset({"korean", "th", "el", "te", "ta"})
)
_PADDLEOCR_V3_LANGS = (
    _PADDLEOCR_LATIN_LANGS
    | _PADDLEOCR_ARABIC_LANGS
    | _PADDLEOCR_CYRILLIC_LANGS
    | _PADDLEOCR_DEVANAGARI_LANGS
    | _PADDLEOCR_SPECIFIC_LANGS
)
PADDLEOCR_LANGUAGE_OPTIONS = (PADDLEOCR_AUTO_LANG,) + tuple(sorted(
    _PADDLEOCR_V3_LANGS | _PADDLEOCR_V5_LANGS
))

_PADDLEOCR_LANG_ALIASES = {
    "zh": "ch",
    "zh-cn": "ch",
    "zh-hans": "ch",
    "zh-tw": "chinese_cht",
    "zh-hant": "chinese_cht",
    "ja": "japan",
    "jp": "japan",
    "ko": "korean",
}

# Content-free, aggregated subtitle fast-path diagnostics.
# These counters never store OCR text, pixels, paths, or absolute coordinates.
_SUBTITLE_DIAG_LOCK = threading.Lock()
_SUBTITLE_DIAG_COUNTERS = {}


def _inc_subtitle_diag(name, amount=1):
    key = str(name or "").strip()
    if not key:
        return
    try:
        delta = int(amount)
    except (TypeError, ValueError):
        return
    if delta == 0:
        return
    with _SUBTITLE_DIAG_LOCK:
        _SUBTITLE_DIAG_COUNTERS[key] = int(_SUBTITLE_DIAG_COUNTERS.get(key, 0) or 0) + delta


def get_subtitle_fastpath_diagnostics_snapshot():
    """Return a copy of aggregated subtitle fast-path diagnostic counters."""
    with _SUBTITLE_DIAG_LOCK:
        return dict(_SUBTITLE_DIAG_COUNTERS)


def reset_subtitle_fastpath_diagnostics():
    """Clear aggregated subtitle fast-path diagnostic counters (tests/tools)."""
    with _SUBTITLE_DIAG_LOCK:
        _SUBTITLE_DIAG_COUNTERS.clear()


def _ratio_bucket(ratio):
    try:
        value = float(ratio)
    except (TypeError, ValueError):
        return "unknown"
    if value < 0.05:
        return "lt05"
    if value < 0.10:
        return "05_10"
    if value < 0.14:
        return "10_14"
    if value < 0.20:
        return "14_20"
    if value < 0.40:
        return "20_40"
    if value < 0.80:
        return "40_80"
    return "ge80"


def _diag_note(diagnostics, key, amount=1):
    if diagnostics is None:
        return
    try:
        diagnostics[key] = int(diagnostics.get(key, 0) or 0) + int(amount)
    except Exception:
        pass


class PaddleOCRUnavailableError(RuntimeError):
    """Raised when PaddleOCR cannot be imported or initialized."""


def _sanitize_exception_reason_for_log(error, max_chars=120):
    """Return a bounded single-line diagnostic without quoted OCR content."""
    try:
        reason = str(error)
    except Exception:
        reason = ""
    reason = re.sub(r"(['\"]).*?\1", "<redacted>", reason)
    reason = re.sub(r"https?://\S+", "<url>", reason, flags=re.IGNORECASE)
    reason = re.sub(r"\b[A-Za-z]:[\\/]\S+", "<path>", reason)
    reason = re.sub(r"\s+", " ", reason).strip()
    if not reason:
        return "no detail"
    return reason[: max(1, int(max_chars))].rstrip()


def _elapsed_seconds(started_at, clock=None):
    clock = clock or time.monotonic
    try:
        return max(0.0, float(clock()) - float(started_at))
    except Exception:
        return 0.0


def summarize_paddleocr_settings(settings):
    """Return a path-safe settings summary for diagnostics (no user paths)."""
    settings = normalize_paddleocr_settings(settings)
    source_dir_configured = bool(str(settings.source_dir or "").strip())
    return {
        "lang": settings.lang,
        "ocr_version": settings.ocr_version,
        "model_size": settings.model_size,
        "device": settings.device,
        "min_score": round(float(settings.min_score), 3),
        "upscale": round(float(settings.upscale), 3),
        "text_det_limit_side_len": int(settings.text_det_limit_side_len),
        "text_det_limit_type": settings.text_det_limit_type,
        "use_textline_orientation": bool(settings.use_textline_orientation),
        "source_dir_configured": source_dir_configured,
        "source_dir_present": bool(_resolve_local_source_dir(settings.source_dir)),
    }


def get_paddleocr_runtime_host_info():
    """Return coarse host facts for prewarm diagnostics (no env values)."""
    try:
        cpu_count = os.cpu_count() or 0
    except Exception:
        cpu_count = 0
    try:
        system_name = platform.system() or "unknown"
    except Exception:
        system_name = "unknown"
    try:
        machine = platform.machine() or "unknown"
    except Exception:
        machine = "unknown"
    return {
        "cpu_count": int(cpu_count),
        "platform": str(system_name),
        "machine": str(machine),
        "python_bits": 64 if sys.maxsize > 2**32 else 32,
    }


def _probe_ppocrv6_model_files(model_name):
    """Classify whether official model files appear present without exposing paths."""
    name = str(model_name or "").strip()
    if not name:
        return "unknown"
    candidates = []
    try:
        home = os.path.expanduser("~")
        if home and home != "~":
            candidates.append(os.path.join(home, ".paddlex", "official_models", name))
    except Exception:
        pass
    for env_key in ("PADDLE_PDX_CACHE_HOME", "PADDLEX_HOME", "PADDLE_HOME"):
        try:
            root = os.environ.get(env_key)
        except Exception:
            root = None
        if root:
            candidates.append(os.path.join(str(root), "official_models", name))
            candidates.append(os.path.join(str(root), name))
    saw_parent = False
    for candidate in candidates:
        try:
            parent = os.path.dirname(candidate)
            if parent and os.path.isdir(parent):
                saw_parent = True
            if os.path.isdir(candidate):
                return "present"
        except Exception:
            continue
    if saw_parent or candidates:
        return "absent"
    return "unknown"


def _classify_engine_build_kind(files_before, files_after, cache_hit):
    if cache_hit:
        return "cache_hit"
    if files_before == "absent" and files_after == "present":
        return "model_download_and_build"
    if files_before == "present":
        return "model_build_cached_files"
    if files_before == "absent":
        return "model_construct_files_absent"
    return "model_construct"


def _record_phase_metrics(phase_metrics, **values):
    if phase_metrics is None:
        return
    try:
        phase_metrics.update(values)
    except Exception:
        pass


@dataclass(frozen=True)
class PaddleOCRSettings:
    source_dir: str = "PaddleOCR-3.7.0"
    lang: str = PADDLEOCR_AUTO_LANG
    ocr_version: str = "PP-OCRv6"
    model_size: str = "tiny"
    device: str = "cpu"
    min_score: float = 0.45
    upscale: float = 1.0
    text_det_limit_side_len: int = 960
    text_det_limit_type: str = "max"
    use_textline_orientation: bool = False


@dataclass(frozen=True)
class PaddleOCRLine:
    text: str
    confidence: float
    bbox: object = None


_PADDLEOCR_ENGINE_CACHE = {}
_PADDLEOCR_ENGINE_CACHE_LOCK = threading.RLock()
_PADDLEOCR_TEXT_REC_ENGINE_CACHE = {}
_PADDLEOCR_TEXT_REC_ENGINE_CACHE_LOCK = threading.RLock()


def _normalize_model_size(model_size):
    normalized = str(model_size or "tiny").strip().lower()
    if normalized in {"tiny", "small", "medium"}:
        return normalized
    return "tiny"


def _normalize_paddleocr_lang(lang):
    normalized = str(lang or PADDLEOCR_AUTO_LANG).strip().lower().replace("_", "-")
    normalized = _PADDLEOCR_LANG_ALIASES.get(normalized, normalized)
    return normalized or PADDLEOCR_AUTO_LANG


def _normalize_paddleocr_version(ocr_version):
    normalized = str(ocr_version or "PP-OCRv6").strip().lower()
    for supported in PADDLEOCR_SUPPORTED_VERSIONS:
        if normalized == supported.lower():
            return supported
    return str(ocr_version or "PP-OCRv6").strip() or "PP-OCRv6"


def resolve_ppocrv6_model_names(model_size):
    normalized = _normalize_model_size(model_size)
    return (
        f"PP-OCRv6_{normalized}_det",
        f"PP-OCRv6_{normalized}_rec",
    )


def _resolve_official_language_model_names(lang, ocr_version):
    if ocr_version == "PP-OCRv6":
        if lang not in _PADDLEOCR_V6_LANGS:
            return None, None
        return "PP-OCRv6_medium_det", "PP-OCRv6_medium_rec"
    if ocr_version == "PP-OCRv5":
        if lang in {"ch", "chinese_cht", "japan"}:
            rec_model_name = "PP-OCRv5_server_rec"
        elif lang == "en":
            rec_model_name = "en_PP-OCRv5_mobile_rec"
        elif lang in _PADDLEOCR_LATIN_LANGS:
            rec_model_name = "latin_PP-OCRv5_mobile_rec"
        elif lang in _PADDLEOCR_ESLAV_LANGS:
            rec_model_name = "eslav_PP-OCRv5_mobile_rec"
        elif lang in _PADDLEOCR_ARABIC_LANGS:
            rec_model_name = "arabic_PP-OCRv5_mobile_rec"
        elif lang in _PADDLEOCR_CYRILLIC_LANGS:
            rec_model_name = "cyrillic_PP-OCRv5_mobile_rec"
        elif lang in _PADDLEOCR_DEVANAGARI_LANGS:
            rec_model_name = "devanagari_PP-OCRv5_mobile_rec"
        elif lang in {"korean", "th", "el", "te", "ta"}:
            rec_model_name = f"{lang}_PP-OCRv5_mobile_rec"
        else:
            return None, None
        return "PP-OCRv5_server_det", rec_model_name
    if ocr_version == "PP-OCRv4":
        if lang == "ch":
            return "PP-OCRv4_mobile_det", "PP-OCRv4_mobile_rec"
        if lang == "en":
            return "PP-OCRv4_mobile_det", "en_PP-OCRv4_mobile_rec"
        return None, None

    if lang in _PADDLEOCR_LATIN_LANGS:
        rec_lang = "latin"
    elif lang in _PADDLEOCR_ARABIC_LANGS:
        rec_lang = "arabic"
    elif lang in _PADDLEOCR_CYRILLIC_LANGS:
        rec_lang = "cyrillic"
    elif lang in _PADDLEOCR_DEVANAGARI_LANGS:
        rec_lang = "devanagari"
    elif lang in _PADDLEOCR_SPECIFIC_LANGS:
        rec_lang = lang
    else:
        return None, None
    rec_model_name = (
        "PP-OCRv3_mobile_rec"
        if rec_lang == "ch"
        else f"{rec_lang}_PP-OCRv3_mobile_rec"
    )
    return "PP-OCRv3_mobile_det", rec_model_name


def resolve_paddleocr_model_selection(settings):
    """Validate settings and return (det, rec, use_language_selector)."""
    settings = normalize_paddleocr_settings(settings)
    if settings.ocr_version not in PADDLEOCR_SUPPORTED_VERSIONS:
        raise ValueError(
            "Unsupported PaddleOCR version "
            f"{settings.ocr_version!r}; choose one of {', '.join(PADDLEOCR_SUPPORTED_VERSIONS)}"
        )
    if settings.text_det_limit_type not in PADDLEOCR_TEXT_DET_LIMIT_TYPES:
        raise ValueError(
            "Unsupported PaddleOCR detection limit type "
            f"{settings.text_det_limit_type!r}; choose max or min"
        )
    if settings.lang == PADDLEOCR_AUTO_LANG:
        if settings.ocr_version != "PP-OCRv6":
            raise ValueError(
                "PaddleOCR language 'auto' is only available with PP-OCRv6; "
                "select a concrete PaddleOCR language code for older versions"
            )
        det_model_name, rec_model_name = resolve_ppocrv6_model_names(settings.model_size)
        return det_model_name, rec_model_name, False
    if settings.lang not in PADDLEOCR_LANGUAGE_OPTIONS:
        raise ValueError(f"Unsupported PaddleOCR language code: {settings.lang!r}")
    if settings.model_size != "medium":
        raise ValueError(
            "Language-specific PaddleOCR selection requires model size 'medium'. "
            "Use language 'auto' with PP-OCRv6 for tiny or small models."
        )
    det_model_name, rec_model_name = _resolve_official_language_model_names(
        settings.lang,
        settings.ocr_version,
    )
    if not det_model_name or not rec_model_name:
        raise ValueError(
            "No PaddleOCR models are available for "
            f"language={settings.lang!r}, version={settings.ocr_version!r}"
        )
    return det_model_name, rec_model_name, True


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


def normalize_paddleocr_settings(settings):
    if settings is None:
        settings = PaddleOCRSettings()
    if not isinstance(settings, PaddleOCRSettings):
        settings = PaddleOCRSettings(**dict(settings))
    return PaddleOCRSettings(
        source_dir=str(settings.source_dir or "PaddleOCR-3.7.0").strip() or "PaddleOCR-3.7.0",
        lang=_normalize_paddleocr_lang(settings.lang),
        ocr_version=_normalize_paddleocr_version(settings.ocr_version),
        model_size=_normalize_model_size(settings.model_size),
        device=str(settings.device or "cpu").strip() or "cpu",
        min_score=_coerce_float(settings.min_score, 0.45, 0.0, 1.0),
        upscale=_coerce_float(settings.upscale, 1.0, 1.0, 4.0),
        text_det_limit_side_len=_coerce_int(settings.text_det_limit_side_len, 960, 128, 4096),
        text_det_limit_type=str(settings.text_det_limit_type or "max").strip().lower() or "max",
        use_textline_orientation=bool(settings.use_textline_orientation),
    )


def _resolve_local_source_dir(source_dir):
    if not source_dir:
        return None
    candidate = os.path.abspath(os.path.expanduser(str(source_dir)))
    if os.path.isdir(candidate):
        return candidate
    repo_relative = os.path.abspath(os.path.join(os.path.dirname(__file__), str(source_dir)))
    if os.path.isdir(repo_relative):
        return repo_relative
    return None


def _import_paddleocr(settings):
    try:
        from paddleocr import PaddleOCR

        return PaddleOCR
    except Exception as first_error:
        local_source_dir = _resolve_local_source_dir(settings.source_dir)
        if local_source_dir and local_source_dir not in sys.path:
            sys.path.insert(0, local_source_dir)
            log_debug(f"PaddleOCR local source directory added to sys.path: {local_source_dir}")
        try:
            from paddleocr import PaddleOCR

            return PaddleOCR
        except Exception as second_error:
            raise PaddleOCRUnavailableError(
                "PaddleOCR is not available. Install paddleocr/paddlex/paddlepaddle "
                f"or keep a usable source tree at {settings.source_dir!r}. "
                f"Import errors: {type(first_error).__name__}: {first_error}; "
                f"{type(second_error).__name__}: {second_error}"
            ) from second_error


def _import_text_recognition(settings):
    try:
        from paddleocr import TextRecognition

        return TextRecognition
    except Exception as first_error:
        local_source_dir = _resolve_local_source_dir(settings.source_dir)
        if local_source_dir and local_source_dir not in sys.path:
            sys.path.insert(0, local_source_dir)
            log_debug(f"PaddleOCR local source directory added to sys.path: {local_source_dir}")
        try:
            from paddleocr import TextRecognition

            return TextRecognition
        except Exception as second_error:
            raise PaddleOCRUnavailableError(
                "PaddleOCR TextRecognition is not available. Install paddleocr/paddlex/paddlepaddle "
                f"or keep a usable source tree at {settings.source_dir!r}. "
                f"Import errors: {type(first_error).__name__}: {first_error}; "
                f"{type(second_error).__name__}: {second_error}"
            ) from second_error


def _build_paddleocr_engine(settings, phase_metrics=None, clock=None):
    settings = normalize_paddleocr_settings(settings)
    clock = clock or time.monotonic
    total_started = clock()
    import_started = clock()
    PaddleOCR = _import_paddleocr(settings)
    import_duration = _elapsed_seconds(import_started, clock)
    det_model_name, rec_model_name, use_language_selector = (
        resolve_paddleocr_model_selection(settings)
    )
    probe_started = clock()
    det_files_before = _probe_ppocrv6_model_files(det_model_name)
    rec_files_before = _probe_ppocrv6_model_files(rec_model_name)
    probe_duration = _elapsed_seconds(probe_started, clock)
    kwargs = {
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": settings.use_textline_orientation,
        "text_det_limit_side_len": settings.text_det_limit_side_len,
        "text_det_limit_type": settings.text_det_limit_type,
        "text_rec_score_thresh": settings.min_score,
    }
    if use_language_selector:
        kwargs["lang"] = settings.lang
        kwargs["ocr_version"] = settings.ocr_version
    else:
        kwargs["text_detection_model_name"] = det_model_name
        kwargs["text_recognition_model_name"] = rec_model_name
    if settings.device:
        kwargs["device"] = settings.device
        if str(settings.device).strip().lower().split(":", 1)[0] == "cpu":
            kwargs["enable_mkldnn"] = False
    log_debug(
        "Initializing PaddleOCR "
        f"version={settings.ocr_version} lang={settings.lang} "
        f"det={det_model_name} rec={rec_model_name} "
        f"device={settings.device} min_score={settings.min_score}"
    )
    construct_started = clock()
    engine = PaddleOCR(**kwargs)
    construct_duration = _elapsed_seconds(construct_started, clock)
    det_files_after = _probe_ppocrv6_model_files(det_model_name)
    rec_files_after = _probe_ppocrv6_model_files(rec_model_name)
    files_before = (
        "present"
        if det_files_before == "present" and rec_files_before == "present"
        else (
            "absent"
            if det_files_before == "absent" or rec_files_before == "absent"
            else "unknown"
        )
    )
    files_after = (
        "present"
        if det_files_after == "present" and rec_files_after == "present"
        else (
            "absent"
            if det_files_after == "absent" or rec_files_after == "absent"
            else "unknown"
        )
    )
    build_kind = _classify_engine_build_kind(files_before, files_after, cache_hit=False)
    _record_phase_metrics(
        phase_metrics,
        engine_kind="full",
        cache_hit=False,
        import_s=round(import_duration, 4),
        model_files_probe_s=round(probe_duration, 4),
        model_files_before=files_before,
        model_files_after=files_after,
        construct_s=round(construct_duration, 4),
        build_kind=build_kind,
        total_s=round(_elapsed_seconds(total_started, clock), 4),
    )
    return engine


def get_paddleocr_engine(settings, phase_metrics=None, clock=None):
    settings = normalize_paddleocr_settings(settings)
    clock = clock or time.monotonic
    total_started = clock()
    with _PADDLEOCR_ENGINE_CACHE_LOCK:
        engine = _PADDLEOCR_ENGINE_CACHE.get(settings)
        if engine is None:
            engine = _build_paddleocr_engine(
                settings,
                phase_metrics=phase_metrics,
                clock=clock,
            )
            _PADDLEOCR_ENGINE_CACHE[settings] = engine
        else:
            _record_phase_metrics(
                phase_metrics,
                engine_kind="full",
                cache_hit=True,
                import_s=0.0,
                model_files_probe_s=0.0,
                model_files_before="present",
                model_files_after="present",
                construct_s=0.0,
                build_kind="cache_hit",
                total_s=round(_elapsed_seconds(total_started, clock), 4),
            )
        return engine


def _build_paddleocr_text_recognition_engine(settings, phase_metrics=None, clock=None):
    settings = normalize_paddleocr_settings(settings)
    clock = clock or time.monotonic
    total_started = clock()
    import_started = clock()
    TextRecognition = _import_text_recognition(settings)
    import_duration = _elapsed_seconds(import_started, clock)
    _det_model_name, rec_model_name, _use_language_selector = (
        resolve_paddleocr_model_selection(settings)
    )
    probe_started = clock()
    files_before = _probe_ppocrv6_model_files(rec_model_name)
    probe_duration = _elapsed_seconds(probe_started, clock)
    kwargs = {
        "model_name": rec_model_name,
    }
    if settings.device:
        kwargs["device"] = settings.device
        if str(settings.device).strip().lower().split(":", 1)[0] == "cpu":
            kwargs["enable_mkldnn"] = False
    log_debug(
        "Initializing PaddleOCR TextRecognition "
        f"rec={rec_model_name} device={settings.device} min_score={settings.min_score}"
    )
    construct_started = clock()
    engine = TextRecognition(**kwargs)
    construct_duration = _elapsed_seconds(construct_started, clock)
    files_after = _probe_ppocrv6_model_files(rec_model_name)
    build_kind = _classify_engine_build_kind(files_before, files_after, cache_hit=False)
    _record_phase_metrics(
        phase_metrics,
        engine_kind="text_recognition",
        cache_hit=False,
        import_s=round(import_duration, 4),
        model_files_probe_s=round(probe_duration, 4),
        model_files_before=files_before,
        model_files_after=files_after,
        construct_s=round(construct_duration, 4),
        build_kind=build_kind,
        total_s=round(_elapsed_seconds(total_started, clock), 4),
    )
    return engine


def get_paddleocr_text_recognition_engine(settings, phase_metrics=None, clock=None):
    settings = normalize_paddleocr_settings(settings)
    clock = clock or time.monotonic
    total_started = clock()
    with _PADDLEOCR_TEXT_REC_ENGINE_CACHE_LOCK:
        engine = _PADDLEOCR_TEXT_REC_ENGINE_CACHE.get(settings)
        if engine is None:
            engine = _build_paddleocr_text_recognition_engine(
                settings,
                phase_metrics=phase_metrics,
                clock=clock,
            )
            _PADDLEOCR_TEXT_REC_ENGINE_CACHE[settings] = engine
        else:
            _record_phase_metrics(
                phase_metrics,
                engine_kind="text_recognition",
                cache_hit=True,
                import_s=0.0,
                model_files_probe_s=0.0,
                model_files_before="present",
                model_files_after="present",
                construct_s=0.0,
                build_kind="cache_hit",
                total_s=round(_elapsed_seconds(total_started, clock), 4),
            )
        return engine


def clear_paddleocr_engines():
    with _PADDLEOCR_ENGINE_CACHE_LOCK:
        _PADDLEOCR_ENGINE_CACHE.clear()
    with _PADDLEOCR_TEXT_REC_ENGINE_CACHE_LOCK:
        _PADDLEOCR_TEXT_REC_ENGINE_CACHE.clear()


def prepare_paddleocr_image(pil_image, settings):
    settings = normalize_paddleocr_settings(settings)
    if pil_image is None:
        raise ValueError("PaddleOCR image is missing")
    image = pil_image.convert("RGB")
    if settings.upscale > 1.0:
        width = max(1, int(round(image.width * settings.upscale)))
        height = max(1, int(round(image.height * settings.upscale)))
        resample = Image.Resampling.BICUBIC if hasattr(Image, "Resampling") else Image.BICUBIC
        image = image.resize((width, height), resample)
    return image


def _extract_subtitle_line_images_from_mask(image, raw_mask, max_lines=3, diagnostics=None):
    if raw_mask is None or not bool(raw_mask.any()):
        _diag_note(diagnostics, "mask_empty", 1)
        return []

    mask = (raw_mask.astype(np.uint8) * 255)
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 2)),
    )
    mask = cv2.dilate(
        mask,
        cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3)),
        iterations=1,
    )

    component_count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, 8)
    components = []
    image_area = max(1, image.width * image.height)
    reject_counts = {
        "tiny": 0,
        "tall_block": 0,
        "huge_block": 0,
        "fullwidth_short": 0,
        "large_area": 0,
    }
    for label_index in range(1, component_count):
        x, y, width, height, area = (int(value) for value in stats[label_index])
        if width < 8 or height < 6 or area < 20:
            reject_counts["tiny"] += 1
            continue
        aspect_ratio = width / max(1, height)
        area_ratio = area / image_area
        if height > image.height * 0.55 and aspect_ratio < 2.5:
            reject_counts["tall_block"] += 1
            continue
        if width > image.width * 0.80 and height > image.height * 0.40:
            reject_counts["huge_block"] += 1
            continue
        if width > image.width * 0.90 and height < image.height * 0.14:
            reject_counts["fullwidth_short"] += 1
            width_bucket = _ratio_bucket(width / max(1, image.width))
            height_bucket = _ratio_bucket(height / max(1, image.height))
            _diag_note(
                diagnostics,
                f"reject_fullwidth_short_w{width_bucket}_h{height_bucket}",
                1,
            )
            continue
        if area_ratio > 0.35 and height > image.height * 0.30:
            reject_counts["large_area"] += 1
            continue
        components.append({
            "x1": x,
            "y1": y,
            "x2": x + width,
            "y2": y + height,
            "height": height,
            "center_y": y + (height / 2.0),
        })

    for reject_name, count in reject_counts.items():
        if count:
            _diag_note(diagnostics, f"reject_{reject_name}", count)
    _diag_note(diagnostics, "components_kept", len(components))

    line_groups = []
    for component in sorted(components, key=lambda item: item["center_y"]):
        matched_group = None
        for group in line_groups:
            overlap = min(component["y2"], group["y2"]) - max(component["y1"], group["y1"])
            min_height = min(component["height"], group["height"])
            center_delta = abs(component["center_y"] - group["center_y"])
            center_limit = max(8.0, max(component["height"], group["height"]) * 0.75)
            if overlap >= min_height * 0.25 or center_delta <= center_limit:
                matched_group = group
                break
        if matched_group is None:
            line_groups.append(dict(component))
            continue
        matched_group["x1"] = min(matched_group["x1"], component["x1"])
        matched_group["y1"] = min(matched_group["y1"], component["y1"])
        matched_group["x2"] = max(matched_group["x2"], component["x2"])
        matched_group["y2"] = max(matched_group["y2"], component["y2"])
        matched_group["height"] = matched_group["y2"] - matched_group["y1"]
        matched_group["center_y"] = matched_group["y1"] + (matched_group["height"] / 2.0)

    candidates = []
    line_reject = {
        "narrow_line": 0,
        "bad_height": 0,
        "low_aspect": 0,
        "small_crop": 0,
        "portrait_crop": 0,
    }
    for group in line_groups:
        x1 = int(group["x1"])
        y1 = int(group["y1"])
        x2 = int(group["x2"])
        y2 = int(group["y2"])
        line_width = x2 - x1
        line_height = y2 - y1
        if line_width < max(32, int(image.width * 0.05)):
            line_reject["narrow_line"] += 1
            continue
        if line_height < 6 or line_height > image.height * 0.80:
            line_reject["bad_height"] += 1
            continue
        if line_width / max(1, line_height) < 1.4:
            line_reject["low_aspect"] += 1
            continue
        x_padding = max(6, int(line_height * 0.35))
        y_padding = max(4, int(line_height * 0.20))
        crop_box = (
            max(0, x1 - x_padding),
            max(0, y1 - y_padding),
            min(image.width, x2 + x_padding),
            min(image.height, y2 + y_padding),
        )
        crop_width = crop_box[2] - crop_box[0]
        crop_height = crop_box[3] - crop_box[1]
        if crop_width < 16 or crop_height < 8:
            line_reject["small_crop"] += 1
            continue
        if crop_width < crop_height:
            line_reject["portrait_crop"] += 1
            continue
        center_x = (crop_box[0] + crop_box[2]) / 2.0
        center_y = (crop_box[1] + crop_box[3]) / 2.0
        center_penalty = abs(center_x - (image.width / 2.0)) / max(1.0, image.width / 2.0)
        bottom_bias = center_y / max(1.0, image.height)
        score = (crop_width * crop_height) * (1.0 + (0.35 * bottom_bias)) * (1.0 - (0.20 * center_penalty))
        candidates.append((crop_box[1], score, image.crop(crop_box)))

    for reject_name, count in line_reject.items():
        if count:
            _diag_note(diagnostics, f"line_{reject_name}", count)

    if len(candidates) > max_lines:
        candidates = sorted(candidates, key=lambda item: item[1], reverse=True)[:max_lines]
    crops = [crop for _y, _score, crop in sorted(candidates, key=lambda item: item[0])]
    _diag_note(diagnostics, "crops", len(crops))
    return crops


def _build_subtitle_edge_mask(gray):
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    gradient = cv2.morphologyEx(
        blurred,
        cv2.MORPH_GRADIENT,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
    )
    gradient_threshold = max(10.0, float(np.percentile(gradient, 90)))
    contrast_mask = gradient >= gradient_threshold
    edges = cv2.Canny(blurred, 50, 150) > 0
    return np.logical_or(contrast_mask, edges)


def prepare_paddleocr_subtitle_line_images(
    pil_image,
    settings,
    max_lines=3,
    diagnostics=None,
    prepared_image=None,
):
    image = (
        prepared_image
        if prepared_image is not None
        else prepare_paddleocr_image(pil_image, settings)
    )
    if image.width <= 0 or image.height <= 0:
        _diag_note(diagnostics, "invalid_image", 1)
        return []

    rgb = np.array(image if image.mode == "RGB" else image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    threshold = max(145.0, float(np.percentile(gray, 88)))
    bright_mask = gray >= threshold
    if bright_mask.mean() > 0.30:
        threshold = max(180.0, float(np.percentile(gray, 95)))
        bright_mask = gray >= threshold
    if not bool(bright_mask.any()):
        _diag_note(diagnostics, "bright_mask_empty", 1)
        edge_mask = _build_subtitle_edge_mask(gray)
        _diag_note(diagnostics, "edge_fallback_attempted", 1)
        edge_diag = {}
        crops = _extract_subtitle_line_images_from_mask(
            image,
            edge_mask,
            max_lines=max_lines,
            diagnostics=edge_diag,
        )
        if diagnostics is not None:
            for key, value in edge_diag.items():
                _diag_note(diagnostics, f"edge_{key}", value)
            if crops:
                _diag_note(diagnostics, "edge_fallback_crop_success", 1)
            else:
                _diag_note(diagnostics, "edge_fallback_no_crop", 1)
                # Promote top edge reject reason for no-crop classification.
                top_reject = _top_reject_reason(edge_diag, prefix="")
                if top_reject:
                    _diag_note(diagnostics, f"no_crop_reason_edge_{top_reject}", 1)
                else:
                    _diag_note(diagnostics, "no_crop_reason_edge_no_candidates", 1)
        return crops

    bright_diag = {}
    line_images = _extract_subtitle_line_images_from_mask(
        image,
        bright_mask,
        max_lines=max_lines,
        diagnostics=bright_diag,
    )
    if diagnostics is not None:
        for key, value in bright_diag.items():
            _diag_note(diagnostics, f"bright_{key}", value)
    if line_images:
        _diag_note(diagnostics, "bright_crop_success", 1)
        return line_images

    _diag_note(diagnostics, "bright_no_crop", 1)
    top_bright_reject = _top_reject_reason(bright_diag, prefix="")
    if top_bright_reject:
        _diag_note(diagnostics, f"no_crop_reason_bright_{top_bright_reject}", 1)
    else:
        _diag_note(diagnostics, "no_crop_reason_bright_no_candidates", 1)

    edge_mask = _build_subtitle_edge_mask(gray)
    _diag_note(diagnostics, "edge_fallback_attempted", 1)
    edge_diag = {}
    crops = _extract_subtitle_line_images_from_mask(
        image,
        edge_mask,
        max_lines=max_lines,
        diagnostics=edge_diag,
    )
    if diagnostics is not None:
        for key, value in edge_diag.items():
            _diag_note(diagnostics, f"edge_{key}", value)
        if crops:
            _diag_note(diagnostics, "edge_fallback_crop_success", 1)
        else:
            _diag_note(diagnostics, "edge_fallback_no_crop", 1)
            top_edge_reject = _top_reject_reason(edge_diag, prefix="")
            if top_edge_reject:
                _diag_note(diagnostics, f"no_crop_reason_edge_{top_edge_reject}", 1)
            else:
                _diag_note(diagnostics, "no_crop_reason_edge_no_candidates", 1)
    return crops


def _top_reject_reason(diag, prefix=""):
    if not isinstance(diag, dict):
        return ""
    best_name = ""
    best_count = 0
    for key, value in diag.items():
        if not str(key).startswith("reject_"):
            continue
        try:
            count = int(value or 0)
        except (TypeError, ValueError):
            continue
        if count > best_count:
            best_count = count
            best_name = str(key)
    if not best_name:
        # Fall back to line-level rejects.
        for key, value in diag.items():
            if not str(key).startswith("line_"):
                continue
            try:
                count = int(value or 0)
            except (TypeError, ValueError):
                continue
            if count > best_count:
                best_count = count
                best_name = str(key)
    if not best_name:
        return ""
    return f"{prefix}{best_name}" if prefix else best_name


def _tolist_if_possible(value):
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return tolist()
        except Exception:
            return value
    return value


def _get_result_value(result_item, key, default=None):
    if isinstance(result_item, dict):
        return result_item.get(key, default)
    try:
        return result_item[key]
    except Exception:
        return getattr(result_item, key, default)


def _iter_paddleocr_result_items(result):
    if result is None:
        return
    if isinstance(result, dict):
        yield result
        return
    for item in result:
        yield item


def flatten_paddleocr_result(result, min_score=0.45, keep_linebreaks=False):
    min_score = _coerce_float(min_score, 0.45, 0.0, 1.0)
    lines = []
    for item in _iter_paddleocr_result_items(result) or ():
        texts = _get_result_value(item, "rec_texts", []) or []
        scores = _get_result_value(item, "rec_scores", []) or []
        boxes = _get_result_value(item, "rec_boxes", None)
        if boxes is None:
            boxes = _get_result_value(item, "rec_polys", None)
        if boxes is None:
            boxes = []
        for index, text in enumerate(texts):
            clean_text = str(text or "").strip()
            if not clean_text:
                continue
            try:
                confidence = float(scores[index]) if index < len(scores) else 0.0
            except Exception:
                confidence = 0.0
            if confidence < min_score:
                log_debug_coalesced(
                    "paddle-full-low-confidence",
                    "PaddleOCR filtered low-confidence line "
                    f"{summarize_text_for_log(clean_text)} "
                    f"confidence={confidence:.3f}",
                    interval_seconds=5.0,
                )
                continue
            bbox = _tolist_if_possible(boxes[index]) if index < len(boxes) else None
            lines.append(PaddleOCRLine(clean_text, round(confidence, 3), bbox))
    separator = "\n" if keep_linebreaks else " "
    return separator.join(line.text for line in lines), lines


def flatten_paddleocr_text_recognition_result(result, min_score=0.45, diagnostics=None):
    min_score = _coerce_float(min_score, 0.45, 0.0, 1.0)
    lines = []
    low_conf = 0
    noise = 0
    empty = 0
    for item in _iter_paddleocr_result_items(result) or ():
        nested = _get_result_value(item, "res", None)
        if nested is not None:
            item = nested
        clean_text = _clean_subtitle_recognition_text(
            _get_result_value(item, "rec_text", "")
        )
        if not clean_text:
            empty += 1
            continue
        try:
            confidence = float(_get_result_value(item, "rec_score", 0.0) or 0.0)
        except Exception:
            confidence = 0.0
        if confidence < min_score:
            low_conf += 1
            log_debug_coalesced(
                "paddle-subtitle-fast-path-low-confidence",
                "PaddleOCR subtitle fast path filtered low-confidence line "
                f"{summarize_text_for_log(clean_text)} "
                f"confidence={confidence:.3f}",
                interval_seconds=5.0,
            )
            continue
        if _looks_like_subtitle_symbol_noise(clean_text):
            noise += 1
            log_debug_coalesced(
                "paddle-subtitle-fast-path-symbol-noise",
                "PaddleOCR subtitle fast path filtered noisy line "
                f"{summarize_text_for_log(clean_text)} "
                f"confidence={confidence:.3f}",
                interval_seconds=5.0,
            )
            continue
        lines.append(PaddleOCRLine(clean_text, round(confidence, 3), None))
    if diagnostics is not None:
        _diag_note(diagnostics, "rec_empty", empty)
        _diag_note(diagnostics, "rec_low_confidence", low_conf)
        _diag_note(diagnostics, "rec_noise", noise)
        _diag_note(diagnostics, "rec_kept", len(lines))
    return " ".join(line.text for line in lines), lines


def _clean_subtitle_recognition_text(text):
    clean_text = str(text or "").strip()
    if not clean_text:
        return ""
    clean_text = re.sub(
        r"(?<=[A-Za-z])([,;:!?])(?=[A-Za-z])",
        r"\1 ",
        clean_text,
    )
    clean_text = re.sub(
        (
            r"\bI(?=(mean|can|got|have|think|know|want|need|was|will|would|"
            r"should|could|just|really|see)\b)"
        ),
        "I ",
        clean_text,
    )
    return re.sub(r"\s+", " ", clean_text).strip()


def _looks_like_subtitle_symbol_noise(text):
    clean_text = str(text or "").strip()
    if not clean_text:
        return True
    chars = [char for char in clean_text if not char.isspace()]
    if not chars:
        return True
    alnum_count = sum(1 for char in chars if char.isalnum())
    alpha_count = sum(1 for char in chars if char.isalpha())
    if alnum_count == 0:
        return True
    if len(chars) <= 2:
        lowered = clean_text.casefold()
        if lowered not in {"i", "a"}:
            return True
    noisy_count = len(chars) - alnum_count
    noisy_ratio = noisy_count / max(1, len(chars))
    return noisy_ratio > 0.45 and alpha_count < 3


def _publish_subtitle_diag(event_key, message, diagnostics=None, *, outcome=None):
    if isinstance(diagnostics, dict):
        published_counts = diagnostics.get("_published_diag_counts")
        if not isinstance(published_counts, dict):
            published_counts = {}
            diagnostics["_published_diag_counts"] = published_counts
        for key, value in diagnostics.items():
            if str(key).startswith("_"):
                continue
            try:
                amount = int(value or 0)
            except (TypeError, ValueError):
                continue
            previous_amount = int(published_counts.get(key, 0) or 0)
            if amount > previous_amount:
                _inc_subtitle_diag(key, amount - previous_amount)
                published_counts[key] = amount
        # Keep log lines short and content-free: only a compact outcome code.
        # Full bucket breakdown lives in get_subtitle_fastpath_diagnostics_snapshot().
        if outcome is None:
            for key in (
                "fast_path_success",
                "fast_path_no_usable_text",
                "fast_path_error",
                "fast_path_no_line_crop",
                "full_fallback_text",
                "full_fallback_empty",
                "full_fallback_error",
            ):
                if int(diagnostics.get(key, 0) or 0):
                    outcome = key
                    break
        reason = None
        reason_items = [
            (k, int(v or 0))
            for k, v in diagnostics.items()
            if str(k).startswith("no_crop_reason_")
        ]
        reason_items.sort(key=lambda item: item[1], reverse=True)
        if reason_items and reason_items[0][1] > 0:
            reason = reason_items[0][0]
        bits = []
        if outcome:
            bits.append(f"outcome={outcome}")
        if reason:
            bits.append(f"reason={reason}")
        rec_low = int(diagnostics.get("rec_low_confidence", 0) or 0)
        rec_noise = int(diagnostics.get("rec_noise", 0) or 0)
        if rec_low:
            bits.append(f"low_conf={rec_low}")
        if rec_noise:
            bits.append(f"noise={rec_noise}")
        if bits:
            message = f"{message} [{' '.join(bits)}]"
    # Bound log size for existing content-free contracts.
    if len(message) > 220:
        message = message[:217].rstrip() + "..."
    log_debug_coalesced(event_key, message, interval_seconds=5.0)


def recognize_with_paddleocr(
    pil_image, settings=None, keep_linebreaks=False, prepared_image=None
):
    settings = normalize_paddleocr_settings(settings)
    engine = get_paddleocr_engine(settings)
    if prepared_image is None:
        prepared_image = prepare_paddleocr_image(pil_image, settings)
    result = engine.predict(np.array(prepared_image))
    return flatten_paddleocr_result(
        result,
        min_score=settings.min_score,
        keep_linebreaks=keep_linebreaks,
    )


def recognize_subtitle_with_paddleocr(pil_image, settings=None, keep_linebreaks=False):
    settings = normalize_paddleocr_settings(settings)
    # Prepare the frame once and reuse it for both the line-crop fast path and
    # the full-frame fallback below (fallback previously re-prepared + re-upscaled).
    prepared_image = prepare_paddleocr_image(pil_image, settings)
    diagnostics = {}
    line_images = prepare_paddleocr_subtitle_line_images(
        pil_image,
        settings,
        diagnostics=diagnostics,
        prepared_image=prepared_image,
    )
    if line_images:
        try:
            engine = get_paddleocr_text_recognition_engine(settings)
            recognized_lines = []
            line_inputs = [
                np.array(
                    line_image
                    if line_image.mode == "RGB"
                    else line_image.convert("RGB")
                )
                for line_image in line_images
            ]
            predict_input = line_inputs if len(line_inputs) > 1 else line_inputs[0]
            result = engine.predict(
                input=predict_input,
                batch_size=max(1, len(line_inputs)),
            )
            rec_diag = {}
            _line_text, recognized_lines = flatten_paddleocr_text_recognition_result(
                result,
                min_score=settings.min_score,
                diagnostics=rec_diag,
            )
            for key, value in rec_diag.items():
                _diag_note(diagnostics, key, value)
            if recognized_lines:
                separator = "\n" if keep_linebreaks else " "
                text = separator.join(line.text for line in recognized_lines)
                _diag_note(diagnostics, "fast_path_success", 1)
                _publish_subtitle_diag(
                    "paddle-subtitle-fast-path-success",
                    "PaddleOCR subtitle fast path recognized "
                    f"lines={len(recognized_lines)} chars={len(text)}",
                    diagnostics,
                    outcome="fast_path_success",
                )
                return text, recognized_lines
            _diag_note(diagnostics, "fast_path_no_usable_text", 1)
            _publish_subtitle_diag(
                "paddle-subtitle-fast-path-no-usable-text",
                "PaddleOCR subtitle fast path found no usable text; "
                "falling back to full OCR",
                diagnostics,
                outcome="fast_path_no_usable_text",
            )
        except Exception as subtitle_error:
            reason = _sanitize_exception_reason_for_log(subtitle_error)
            _diag_note(diagnostics, "fast_path_error", 1)
            _publish_subtitle_diag(
                "paddle-subtitle-fast-path-error",
                "PaddleOCR subtitle fast path failed; falling back to full OCR: "
                f"{type(subtitle_error).__name__}: {reason}",
                diagnostics,
                outcome="fast_path_error",
            )
    else:
        _diag_note(diagnostics, "fast_path_no_line_crop", 1)
        _publish_subtitle_diag(
            "paddle-subtitle-fast-path-no-line-crop",
            "PaddleOCR subtitle fast path found no line crop; "
            "falling back to full OCR",
            diagnostics,
            outcome="fast_path_no_line_crop",
        )

    try:
        text, lines = recognize_with_paddleocr(
            pil_image,
            settings,
            keep_linebreaks=keep_linebreaks,
            prepared_image=prepared_image,
        )
        if text and str(text).strip():
            _diag_note(diagnostics, "full_fallback_text", 1)
            _publish_subtitle_diag(
                "paddle-subtitle-full-fallback-text",
                "PaddleOCR full OCR fallback produced text "
                f"lines={len(lines)} chars={len(str(text))}",
                diagnostics,
                outcome="full_fallback_text",
            )
        else:
            _diag_note(diagnostics, "full_fallback_empty", 1)
            _publish_subtitle_diag(
                "paddle-subtitle-full-fallback-empty",
                "PaddleOCR full OCR fallback produced no text",
                diagnostics,
                outcome="full_fallback_empty",
            )
        return text, lines
    except Exception as full_error:
        reason = _sanitize_exception_reason_for_log(full_error)
        _diag_note(diagnostics, "full_fallback_error", 1)
        _publish_subtitle_diag(
            "paddle-subtitle-full-fallback-error",
            "PaddleOCR full OCR fallback failed: "
            f"{type(full_error).__name__}: {reason}",
            diagnostics,
            outcome="full_fallback_error",
        )
        raise
