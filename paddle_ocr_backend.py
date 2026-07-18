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
    lang: str = "en"
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


def resolve_ppocrv6_model_names(model_size):
    normalized = _normalize_model_size(model_size)
    return (
        f"PP-OCRv6_{normalized}_det",
        f"PP-OCRv6_{normalized}_rec",
    )


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
        lang=str(settings.lang or "en").strip() or "en",
        ocr_version=str(settings.ocr_version or "PP-OCRv6").strip() or "PP-OCRv6",
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
    det_model_name, rec_model_name = resolve_ppocrv6_model_names(settings.model_size)
    probe_started = clock()
    det_files_before = _probe_ppocrv6_model_files(det_model_name)
    rec_files_before = _probe_ppocrv6_model_files(rec_model_name)
    probe_duration = _elapsed_seconds(probe_started, clock)
    kwargs = {
        "ocr_version": settings.ocr_version,
        "text_detection_model_name": det_model_name,
        "text_recognition_model_name": rec_model_name,
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": settings.use_textline_orientation,
        "text_det_limit_side_len": settings.text_det_limit_side_len,
        "text_det_limit_type": settings.text_det_limit_type,
        "text_rec_score_thresh": settings.min_score,
    }
    if settings.device:
        kwargs["device"] = settings.device
        if str(settings.device).strip().lower().split(":", 1)[0] == "cpu":
            kwargs["enable_mkldnn"] = False
    log_debug(
        "Initializing PaddleOCR "
        f"version={settings.ocr_version} det={det_model_name} rec={rec_model_name} "
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
    _det_model_name, rec_model_name = resolve_ppocrv6_model_names(settings.model_size)
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


def _extract_subtitle_line_images_from_mask(image, raw_mask, max_lines=3):
    if raw_mask is None or not bool(raw_mask.any()):
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
    for label_index in range(1, component_count):
        x, y, width, height, area = (int(value) for value in stats[label_index])
        if width < 8 or height < 6 or area < 20:
            continue
        aspect_ratio = width / max(1, height)
        area_ratio = area / image_area
        if height > image.height * 0.55 and aspect_ratio < 2.5:
            continue
        if width > image.width * 0.80 and height > image.height * 0.40:
            continue
        if width > image.width * 0.90 and height < image.height * 0.14:
            continue
        if area_ratio > 0.35 and height > image.height * 0.30:
            continue
        components.append({
            "x1": x,
            "y1": y,
            "x2": x + width,
            "y2": y + height,
            "height": height,
            "center_y": y + (height / 2.0),
        })

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
    for group in line_groups:
        x1 = int(group["x1"])
        y1 = int(group["y1"])
        x2 = int(group["x2"])
        y2 = int(group["y2"])
        line_width = x2 - x1
        line_height = y2 - y1
        if line_width < max(32, int(image.width * 0.05)):
            continue
        if line_height < 6 or line_height > image.height * 0.80:
            continue
        if line_width / max(1, line_height) < 1.4:
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
            continue
        if crop_width < crop_height:
            continue
        center_x = (crop_box[0] + crop_box[2]) / 2.0
        center_y = (crop_box[1] + crop_box[3]) / 2.0
        center_penalty = abs(center_x - (image.width / 2.0)) / max(1.0, image.width / 2.0)
        bottom_bias = center_y / max(1.0, image.height)
        score = (crop_width * crop_height) * (1.0 + (0.35 * bottom_bias)) * (1.0 - (0.20 * center_penalty))
        candidates.append((crop_box[1], score, image.crop(crop_box)))

    if len(candidates) > max_lines:
        candidates = sorted(candidates, key=lambda item: item[1], reverse=True)[:max_lines]
    return [crop for _y, _score, crop in sorted(candidates, key=lambda item: item[0])]


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


def prepare_paddleocr_subtitle_line_images(pil_image, settings, max_lines=3):
    image = prepare_paddleocr_image(pil_image, settings)
    if image.width <= 0 or image.height <= 0:
        return []

    rgb = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    threshold = max(145.0, float(np.percentile(gray, 88)))
    bright_mask = gray >= threshold
    if bright_mask.mean() > 0.30:
        threshold = max(180.0, float(np.percentile(gray, 95)))
        bright_mask = gray >= threshold
    if not bool(bright_mask.any()):
        edge_mask = _build_subtitle_edge_mask(gray)
        return _extract_subtitle_line_images_from_mask(image, edge_mask, max_lines=max_lines)

    line_images = _extract_subtitle_line_images_from_mask(image, bright_mask, max_lines=max_lines)
    if line_images:
        return line_images

    edge_mask = _build_subtitle_edge_mask(gray)
    return _extract_subtitle_line_images_from_mask(image, edge_mask, max_lines=max_lines)


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


def flatten_paddleocr_text_recognition_result(result, min_score=0.45):
    min_score = _coerce_float(min_score, 0.45, 0.0, 1.0)
    lines = []
    for item in _iter_paddleocr_result_items(result) or ():
        nested = _get_result_value(item, "res", None)
        if nested is not None:
            item = nested
        clean_text = _clean_subtitle_recognition_text(
            _get_result_value(item, "rec_text", "")
        )
        if not clean_text:
            continue
        try:
            confidence = float(_get_result_value(item, "rec_score", 0.0) or 0.0)
        except Exception:
            confidence = 0.0
        if confidence < min_score:
            log_debug_coalesced(
                "paddle-subtitle-fast-path-low-confidence",
                "PaddleOCR subtitle fast path filtered low-confidence line "
                f"{summarize_text_for_log(clean_text)} "
                f"confidence={confidence:.3f}",
                interval_seconds=5.0,
            )
            continue
        if _looks_like_subtitle_symbol_noise(clean_text):
            log_debug_coalesced(
                "paddle-subtitle-fast-path-symbol-noise",
                "PaddleOCR subtitle fast path filtered noisy line "
                f"{summarize_text_for_log(clean_text)} "
                f"confidence={confidence:.3f}",
                interval_seconds=5.0,
            )
            continue
        lines.append(PaddleOCRLine(clean_text, round(confidence, 3), None))
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


def recognize_with_paddleocr(pil_image, settings=None, keep_linebreaks=False):
    settings = normalize_paddleocr_settings(settings)
    engine = get_paddleocr_engine(settings)
    prepared_image = prepare_paddleocr_image(pil_image, settings)
    result = engine.predict(np.array(prepared_image))
    return flatten_paddleocr_result(
        result,
        min_score=settings.min_score,
        keep_linebreaks=keep_linebreaks,
    )


def recognize_subtitle_with_paddleocr(pil_image, settings=None, keep_linebreaks=False):
    settings = normalize_paddleocr_settings(settings)
    line_images = prepare_paddleocr_subtitle_line_images(pil_image, settings)
    if line_images:
        try:
            engine = get_paddleocr_text_recognition_engine(settings)
            recognized_lines = []
            line_inputs = [np.array(line_image.convert("RGB")) for line_image in line_images]
            predict_input = line_inputs if len(line_inputs) > 1 else line_inputs[0]
            result = engine.predict(
                input=predict_input,
                batch_size=max(1, len(line_inputs)),
            )
            _line_text, recognized_lines = flatten_paddleocr_text_recognition_result(
                result,
                min_score=settings.min_score,
            )
            if recognized_lines:
                separator = "\n" if keep_linebreaks else " "
                text = separator.join(line.text for line in recognized_lines)
                log_debug_coalesced(
                    "paddle-subtitle-fast-path-success",
                    "PaddleOCR subtitle fast path recognized "
                    f"lines={len(recognized_lines)} chars={len(text)}",
                    interval_seconds=5.0,
                )
                return text, recognized_lines
            log_debug_coalesced(
                "paddle-subtitle-fast-path-no-usable-text",
                "PaddleOCR subtitle fast path found no usable text; "
                "falling back to full OCR",
                interval_seconds=5.0,
            )
        except Exception as subtitle_error:
            reason = _sanitize_exception_reason_for_log(subtitle_error)
            log_debug_coalesced(
                "paddle-subtitle-fast-path-error",
                "PaddleOCR subtitle fast path failed; falling back to full OCR: "
                f"{type(subtitle_error).__name__}: {reason}",
                interval_seconds=5.0,
            )
    else:
        log_debug_coalesced(
            "paddle-subtitle-fast-path-no-line-crop",
            "PaddleOCR subtitle fast path found no line crop; "
            "falling back to full OCR",
            interval_seconds=5.0,
        )

    return recognize_with_paddleocr(
        pil_image,
        settings,
        keep_linebreaks=keep_linebreaks,
    )
